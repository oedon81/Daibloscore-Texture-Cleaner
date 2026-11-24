#!/usr/bin/env python3
"""Daibloscore Stage 1 Trainer (dual-input PC DXT5 + Android ASTC 6x6 -> premultiplied RGBA).
  - 384x16 norm-free Residual CNN with Local Channel Attention (~43.0M params), ASTC base + couple_edge
  - On-the-fly batched /dev/shm patch encoding across worker threads (DXT5 + astcenc 3.5 medium/thorough)
  - bf16/fp16 AMP, EMA 0.999, sparse_prob, straight_prob, active-pixel loss normalization, L1 + RMSE + hue + Seam + SSIM
  - Supports CUDA, ROCm, TorchTPU, and full auto-resume via ckpt_state.pt
"""
import os, sys, copy, math, time, argparse
import numpy as np, torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dclean import codec as C, models as M, metrics as MT, data as D, pipeline as PL, xla_util as XU

EVAL_FAST = ['60350_skin_Ijen03b', '30431_skin_Lepus04c', '30180_skin_Machairodus04']


def _lr_at(step, total, lr, lr_min, warmup, cosine_from):
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    c0 = int(total * cosine_from)
    if step < c0:
        return lr
    t = (step - c0) / max(1, total - c0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


def score_set(net_fn, real, dev):
    rs = []
    for name, t, y, _ in real:
        r = MT.score(net_fn(t), y, dev)
        r['name'] = name
        rs.append(r)
    m = {k: float(np.mean([r[k] for r in rs])) for k in ('psnr', 'edges', 'seam', 'ssim')}
    return m, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--val-json', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--init', default=None)
    ap.add_argument('--channels', type=int, default=384)
    ap.add_argument('--blocks', type=int, default=16)
    ap.add_argument('--no-ca', dest='use_ca', action='store_false', default=True)
    ap.add_argument('--steps', type=int, default=120000)
    ap.add_argument('--eval-every', type=int, default=2500)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--patch', type=int, default=168)
    ap.add_argument('--sparse-prob', type=float, default=0.10)
    ap.add_argument('--straight-prob', type=float, default=0.20, help='probability of keeping straight-alpha sources straight-encoded (rest premultiplied before encode)')
    ap.add_argument('--rim-anneal', type=int, default=0, help='smoothly anneal legacy rim_gate from 1.0->0.0 over N steps when fine-tuning from a rim-gated checkpoint')
    ap.add_argument('--thorough-frac', type=float, default=1 / 3)
    ap.add_argument('--daiblos-weight', type=float, default=0.40)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lr-min', type=float, default=2e-6)
    ap.add_argument('--cosine-from', type=float, default=0.35)
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--l1-weight', type=float, default=0.5)
    ap.add_argument('--l2-weight', type=float, default=1.0)
    ap.add_argument('--hue-weight', type=float, default=2.0)
    ap.add_argument('--seam-weight', type=float, default=1.0)
    ap.add_argument('--seam-l2-weight', type=float, default=2.0)
    ap.add_argument('--ssim-weight', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--pool', type=int, default=6)
    ap.add_argument('--no-compile', dest='compile', action='store_false', default=True)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()

    dev, backend, amp_dtype, ac_dev = XU.get_device()
    if backend == 'cuda':
        torch.backends.cudnn.benchmark = True
    use_amp = XU.autocast_ok(ac_dev, amp_dtype)
    workers = min(a.workers, max(2, (os.cpu_count() or 8) - 2))

    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(a.out_dir, 'log.txt'), 'a')
    def say(*x):
        s = ' '.join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    say(f'=== {time.strftime("%Y-%m-%d %H:%M")}  ' + ' '.join(sys.argv[1:]))

    # Discover clean PNG groups with content-based leakage & dedup filtering (name + SHA-256 + 32x32 thumbnail)
    val_fp = D.build_val_fingerprints(a.val_json)
    groups = []
    daib_dir = os.path.join(a.data_root, 'daiblos_clean')
    other_dir = os.path.join(a.data_root, 'other')
    echo_dir = os.path.join(a.data_root, 'echo_train')
    d_files = D.list_clean_pngs(daib_dir, val_fp=val_fp, log_fn=say) if os.path.isdir(daib_dir) else []
    d_shas = {D._sha256(f) for f in d_files}
    o_files = D.list_clean_pngs(other_dir, val_fp=val_fp, dedup_shas=d_shas, log_fn=say) if os.path.isdir(other_dir) else []
    e_files = D.list_clean_pngs(echo_dir, val_fp=val_fp, dedup_shas=d_shas, log_fn=say) if os.path.isdir(echo_dir) else []
    all_other = o_files + e_files
    if d_files and all_other:
        groups = [(d_files, a.daiblos_weight), (all_other, 1.0 - a.daiblos_weight)]
    elif all_other:
        groups = [(all_other, 1.0)]
    else:
        groups = [(d_files, 1.0)]
    say('data:', ', '.join(f'{len(g)} PNGs (weight {w:.2f})' for g, w in groups))

    real = D.real_set(a.val_json, EVAL_FAST)
    say('val:', ', '.join(n for n, _, _, _ in real))

    use_rim_gate = bool(a.rim_anneal > 0)
    net = M.stage1(a.channels, a.blocks, use_ca=a.use_ca, couple_edge=True, rim_gate=use_rim_gate).to(dev)
    if a.init and os.path.exists(a.init):
        ck0 = torch.load(a.init, map_location='cpu', weights_only=False)
        net.load_state_dict(ck0['net'])
        say(f'initialized weights from {a.init} (rim_anneal={a.rim_anneal})')
    net.train()
    ema = copy.deepcopy(net).eval() if a.ema > 0 else None
    if ema is not None:
        # EMA always evaluates cleanly ungated once annealing completes (or at step > 0 when rim_anneal == 0)
        ema._rim_alpha.fill_(0.0)
        for p in ema.parameters():
            p.requires_grad_(False)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and backend == 'cuda' and amp_dtype == torch.float16))
    start = 0
    best = {'psnr': -1.0, 'seam': 1e9}

    state_path = os.path.join(a.out_dir, 'ckpt_state.pt')
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        net.load_state_dict(st['net'])
        if ema is not None and 'ema' in st:
            ema.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        start = st['step']
        best = st.get('best', best)
        say(f'resumed from {state_path} at step {start} (best PSNR {best["psnr"]:.2f} dB)')

    run_net = net
    compiled = False
    if backend == 'tpu' and a.compile:
        try:
            run_net = torch.compile(net, backend='tpu', dynamic=False)
            compiled = True
            say('enabled torch.compile(backend="tpu", dynamic=False)')
        except Exception as e:
            say(f'torch.compile skipped (using eager TPU): {e}')

    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    say(f'stage 1: {n_params:.2f}M params ({a.channels}x{a.blocks}) on {XU.device_name(dev, backend)} (amp={use_amp}, {amp_dtype}), patch={a.patch}, batch={a.batch}, workers={workers}')

    P = (a.patch // 12) * 12
    stream = D.ThreadedPatchStream(
        groups, P, a.batch, a.thorough_frac, a.pool, a.sparse_prob,
        for_refiner=False, num_workers=workers, straight_prob=a.straight_prob,
    )
    scale = torch.from_numpy(C.SIDE_SCALE).to(dev).view(1, -1, 1, 1)
    lumw = torch.tensor([0.3, 0.59, 0.11], device=dev).view(1, 3, 1, 1)

    ev_compiled = None
    if backend == 'tpu' and a.compile and ema is not None:
        try:
            ev_compiled = torch.compile(ema, backend='tpu', dynamic=False)
        except Exception:
            ev_compiled = None

    def evaluate(step):
        t_ev = time.time()
        ev_net = ema if (ema is not None and step > 0) else net
        ev_net.eval()
        cur_rim_eval = float(ev_net._rim_alpha.item()) if getattr(ev_net, 'rim_gate', False) else 0.0
        ev_run = ev_compiled if (ev_net is ema and ev_compiled is not None) else ev_net
        m, rs = score_set(lambda t: PL.finish(PL.ensemble(ev_run, t.x15(), dev, tta=False, half=True)), real, dev)
        if ev_net is net:
            net.train()
        say(f'--- step {step}: real test ({"ema" if ev_net is ema else "raw"}, rim={cur_rim_eval:.2f}, {time.time() - t_ev:.1f}s)  {MT.fmt(m)}   [' +
            '; '.join(f"{r['name'].split('_')[-1]} {r['psnr']:.2f}/{r['edges']:.2f}" for r in rs) + ']')
        if step == 0 and a.rim_anneal > 0:
            # Don't overwrite best / daiblos_stage1.pt with the pre-anneal rim-gated step-0 checkpoint
            return
        meta = dict(step=step, kind='stage1', channels=a.channels, blocks=a.blocks, ca=a.use_ca,
                    couple_edge=True, rim_gate=(cur_rim_eval > 0.0), score=m)
        M.save(ev_net, os.path.join(a.out_dir, 'daiblos_stage1_last.pt'), **meta)
        torch.save({
            'net': {k: v.detach().cpu() for k, v in net.state_dict().items()},
            'ema': {k: v.detach().cpu() for k, v in ema.state_dict().items()} if ema is not None else None,
            'opt': opt.state_dict(), 'step': step, 'best': best,
        }, state_path + '.part')
        os.replace(state_path + '.part', state_path)
        if m['psnr'] > best['psnr']:
            best['psnr'] = m['psnr']
            M.save(ev_net, os.path.join(a.out_dir, 'daiblos_stage1.pt'), **meta)
        if m['seam'] < best['seam']:
            best['seam'] = m['seam']
            M.save(ev_net, os.path.join(a.out_dir, 'daiblos_stage1_best_seam.pt'), **meta)

    if start == 0:
        for name, t, y, _ in real:
            say(f'    raw {name}: DXT5 {MT.fmt(MT.score(t.dxt, y, dev))} | ASTC {MT.fmt(MT.score(t.astc, y, dev))}')
        if a.init and os.path.exists(a.init):
            evaluate(0)

    say('starting training loop (step 1 includes one-time compile & worker warmup)...')
    t0 = time.time()
    t_steady = None
    for s in range(start, a.steps):
        cur_lr = _lr_at(s, a.steps, a.lr, a.lr_min, a.warmup, a.cosine_from)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        if use_rim_gate:
            cur_rim = max(0.0, 1.0 - float(s) / float(a.rim_anneal))
            net._rim_alpha.fill_(cur_rim)
        x, y = next(stream)
        x = x.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / scale
        y = y.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0

        def _forward_loss(model):
            with torch.autocast(ac_dev, dtype=amp_dtype, enabled=use_amp):
                o = model(x)
            o = o.float()
            ya = y[:, 3:4]
            live = (ya > 0).float()
            w = (1.0 + 3.0 * ((ya > 0) & (ya < 1)).float()) * (0.1 + 0.9 * live)
            wnorm = w.sum().clamp(min=1.0)
            diff = o - y
            loss = torch.zeros((), device=dev)
            if a.l1_weight > 0:
                loss = loss + a.l1_weight * (w * diff.abs()).sum() / (wnorm * 4.0)
            if a.l2_weight > 0:
                loss = loss + a.l2_weight * ((w * diff * diff).sum() / (wnorm * 4.0) + 1e-10).sqrt()
            if a.hue_weight > 0:
                hm = (ya > 2 / 255).float()
                loss = loss + a.hue_weight * (hm * (o[:, :3] - y[:, :3]).abs() / ya.clamp(min=0.03)).sum() / (hm.sum().clamp(min=1.0) * 3.0)
            if a.seam_weight > 0 or a.seam_l2_weight > 0 or a.ssim_weight > 0:
                with torch.no_grad():
                    S, valid = MT.own_colour(y, sigmas=(1.5, 3, 6, 12))
                vwnorm = (valid * w).sum().clamp(min=1.0)
                if a.seam_weight > 0 or a.seam_l2_weight > 0:
                    D_se = (o[:, :3] - y[:, :3]) - S * (o[:, 3:4] - ya)
                    e = D_se * valid * w
                    if a.seam_weight > 0:
                        loss = loss + a.seam_weight * (e.abs().sum() / (vwnorm * 3.0) + 2.0 * MT.gauss((e * lumw).sum(1, keepdim=True), 1.5).abs().sum() / vwnorm)
                    if a.seam_l2_weight > 0:
                        band = (torch.nn.functional.max_pool2d(((ya > 0) & (ya < 1)).float() * valid, 3, 1, 1) > 0).float() * valid
                        coh = MT.gauss((D_se * lumw).sum(1, keepdim=True) * band, 1.5) / MT.gauss(band, 1.5).clamp(min=1e-3)
                        loss = loss + a.seam_l2_weight * ((coh * coh * band).sum() / band.sum().clamp(min=1.0) + 1e-10).sqrt()
                if a.ssim_weight > 0:
                    sm = MT.ssim_map(o[:, :3] + (1 - o[:, 3:4]) * S, y[:, :3] + (1 - ya) * S).mean(1, keepdim=True)
                    loss = loss + a.ssim_weight * ((1 - sm) * valid * w).sum() / vwnorm
            return loss

        opt.zero_grad(set_to_none=True)
        try:
            loss = _forward_loss(run_net)
            if backend not in ('tpu', 'xla') and not torch.isfinite(loss):
                say(f'step {s + 1}: non-finite loss, skipping')
                opt.zero_grad(set_to_none=True)
                continue
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                XU.step_optimizer(opt, backend)
        except Exception as e:
            if compiled and s == start:
                say(f'torch.compile failed on first step ({type(e).__name__}: {e}); falling back to eager TPU!')
                run_net = net
                compiled = False
                opt.zero_grad(set_to_none=True)
                loss = _forward_loss(run_net)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                XU.step_optimizer(opt, backend)
            else:
                raise

        if ema is not None:
            if use_rim_gate and s == a.rim_anneal:
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), net.parameters()):
                        pe.copy_(p)
            else:
                s_ema = max(0, s - a.rim_anneal) if use_rim_gate else s
                d = min(a.ema, (1 + s_ema) / (10 + s_ema))
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), net.parameters()):
                        pe.mul_(d).add_(p, alpha=1.0 - d)
        XU.mark_step(backend)

        if s == start:
            XU.sync(backend)
            t_steady = time.time()
            lv = float(loss.detach().cpu().item())
            say(f'step {s + 1}/{a.steps} done in {t_steady - t0:.1f}s (compile + warmup)  loss {lv:.5f}')
        elif (s + 1 - start) in (10, 50, 100) or (s + 1) % 250 == 0:
            XU.sync(backend)
            el = max(1e-6, time.time() - (t_steady or t0))
            done = max(1, s - start)
            lv = float(loss.detach().cpu().item())
            print(f'step {s + 1}/{a.steps}  loss {lv:.5f}  lr {cur_lr:.2e}  {done / el:.2f} steps/s  ~{el / done * (a.steps - s - 1) / 60:.0f} min left', flush=True)
        if (s + 1) % a.eval_every == 0 or s + 1 == a.steps:
            evaluate(s + 1)

    say(f'DONE! best PSNR {best["psnr"]:.2f} dB, best seam {best["seam"]:.3f} in {(time.time() - t0) / 60:.1f} min')
    log.close()


if __name__ == '__main__':
    main()
