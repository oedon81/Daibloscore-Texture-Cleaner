#!/usr/bin/env python3
"""Daibloscore Stage 2 Refiner Trainer (28-ch split RGB/Alpha mismatch + unrolled Pass-2).
  - 256x10 norm-free Residual CNN with Local Channel Attention (~12.2M params)
  - Re-encodes Stage-1 / Pass-1 outputs with the exact PC DXT5 + Android ASTC 6x6 encoders in /dev/shm
  - Uses bf16/fp16 AMP, EMA 0.999, unrolled Pass-2 rollouts (--pass2-prob 0.5), 2x TTA Stage-1 inputs, and --resume
"""
import os, sys, copy, math, time, random, argparse
import numpy as np, torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dclean import codec as C, models as M, metrics as MT, data as D, pipeline as PL, xla_util as XU
from train_stage1 import _lr_at, score_set, EVAL_FAST


@torch.no_grad()
def _stage1_tta2(net1, xg, amp_dtype, use_amp, ac_dev):
    """2x horizontal-flip TTA on Stage 1 with canonical pos preserved (matches echoclean v2.1 --s1-tta 2)."""
    xg_flip = torch.cat([torch.flip(xg[:, :-4], (3,)), xg[:, -4:]], dim=1)
    with torch.autocast(ac_dev, dtype=amp_dtype, enabled=use_amp):
        o0 = net1(xg).float()
        o1 = torch.flip(net1(xg_flip).float(), (3,))
    return 0.5 * (o0 + o1)


def _build_ref_batch(xn, o_cur_n, flags, preset, pos, extended, astc_threads):
    o_u8 = [PL.finish(o) for o in o_cur_n]
    st = [C.unpremul(u) if f else u for u, f in zip(o_u8, flags)]
    dre, are = C.game_encode_batch(st, preset, astc_threads=astc_threads, parallel=True)
    ins = []
    for i in range(len(st)):
        d_re, a_re = (C.premul(dre[i]), C.premul(are[i])) if flags[i] else (dre[i], are[i])
        ins.append(PL.refiner_input_arrays(xn[i, ..., :4], xn[i, ..., 4:8], o_cur_n[i], d_re, a_re, pos, extended=extended))
    return np.stack(ins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--val-json', required=True)
    ap.add_argument('--stage1', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--init', default=None)
    ap.add_argument('--channels', type=int, default=256)
    ap.add_argument('--blocks', type=int, default=10)
    ap.add_argument('--no-ca', dest='use_ca', action='store_false', default=True)
    ap.add_argument('--steps', type=int, default=35000)
    ap.add_argument('--eval-every', type=int, default=2500)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--patch', type=int, default=144)
    ap.add_argument('--sparse-prob', type=float, default=0.10)
    ap.add_argument('--straight-prob', type=float, default=0.20)
    ap.add_argument('--thorough-frac', type=float, default=0.25)
    ap.add_argument('--daiblos-weight', type=float, default=0.40)
    ap.add_argument('--pass2-prob', type=float, default=0.5)
    ap.add_argument('--s1-tta', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr-min', type=float, default=1e-6)
    ap.add_argument('--cosine-from', type=float, default=0.35)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--l1-weight', type=float, default=0.5)
    ap.add_argument('--l2-weight', type=float, default=1.0)
    ap.add_argument('--hue-weight', type=float, default=2.0)
    ap.add_argument('--seam-weight', type=float, default=1.0)
    ap.add_argument('--seam-l2-weight', type=float, default=2.0)
    ap.add_argument('--ssim-weight', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--astc-threads', type=int, default=8)
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

    val_fp = D.build_val_fingerprints(a.val_json)
    daib_dir = os.path.join(a.data_root, 'daiblos_clean')
    other_dir = os.path.join(a.data_root, 'other')
    echo_dir = os.path.join(a.data_root, 'echo_train')
    d_files = D.list_clean_pngs(daib_dir, val_fp=val_fp, log_fn=say) if os.path.isdir(daib_dir) else []
    d_shas = {D._sha256(f) for f in d_files}
    all_other = (D.list_clean_pngs(other_dir, val_fp=val_fp, dedup_shas=d_shas, log_fn=say) if os.path.isdir(other_dir) else []) + \
                (D.list_clean_pngs(echo_dir, val_fp=val_fp, dedup_shas=d_shas, log_fn=say) if os.path.isdir(echo_dir) else [])
    groups = [(d_files, a.daiblos_weight), (all_other, 1.0 - a.daiblos_weight)] if (d_files and all_other) else [(all_other or d_files, 1.0)]
    real = D.real_set(a.val_json, EVAL_FAST)

    net1, ck1 = M.load(a.stage1, dev)
    for p in net1.parameters():
        p.requires_grad_(False)
    ref = M.Refiner(a.channels, a.blocks, cin=M.CIN2_V21, use_ca=a.use_ca).to(dev)
    if a.init and os.path.exists(a.init):
        ck0 = torch.load(a.init, map_location='cpu', weights_only=False)
        ref.load_state_dict(ck0['net'])
        say(f'initialized refiner from {a.init}')
    ref.train()
    ema = copy.deepcopy(ref).eval() if a.ema > 0 else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)

    opt = torch.optim.Adam(ref.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and backend == 'cuda' and amp_dtype == torch.float16))
    start = 0
    best = {'psnr': -1.0, 'seam': 1e9}

    state_path = os.path.join(a.out_dir, 'ckpt_state.pt')
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        ref.load_state_dict(st['net'])
        if ema is not None and 'ema' in st:
            ema.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        start = st['step']
        best = st.get('best', best)
        say(f'resumed from {state_path} at step {start} (best PSNR {best["psnr"]:.2f} dB)')

    run_ref = ref
    compiled = False
    if backend == 'tpu' and a.compile:
        try:
            run_ref = torch.compile(ref, backend='tpu', dynamic=False)
            compiled = True
            say('enabled torch.compile(backend="tpu", dynamic=False) on refiner')
        except Exception as e:
            say(f'torch.compile skipped (using eager TPU): {e}')

    say(f'refiner: {sum(p.numel() for p in ref.parameters()) / 1e6:.2f}M params ({a.channels}x{a.blocks}, 28-ch) on {XU.device_name(dev, backend)} (amp={use_amp}, {amp_dtype}); workers={workers}, astc_threads={a.astc_threads}; stage1={a.stage1}')

    P = (a.patch // 12) * 12
    stream = D.ThreadedPatchStream(
        groups, P, a.batch, a.thorough_frac, pool_size=4, sparse_prob=a.sparse_prob,
        for_refiner=True, num_workers=workers, straight_prob=a.straight_prob,
    )
    scale = torch.from_numpy(C.SIDE_SCALE).to(dev).view(1, -1, 1, 1)
    lumw = torch.tensor([0.3, 0.59, 0.11], device=dev).view(1, 3, 1, 1)
    pos = C.pos_u8(P, 0, 0, P, P)

    ev_ref_compiled = None
    ev_s1_compiled = net1
    if backend == 'tpu' and a.compile:
        try:
            ev_s1_compiled = torch.compile(net1, backend='tpu', dynamic=False)
            if ema is not None:
                ev_ref_compiled = torch.compile(ema, backend='tpu', dynamic=False)
        except Exception:
            ev_s1_compiled, ev_ref_compiled = net1, None

    def evaluate(step):
        t_ev = time.time()
        ev_ref = ema if (ema is not None and step > 0) else ref
        ev_ref.eval()
        ev_run = ev_ref_compiled if (ev_ref is ema and ev_ref_compiled is not None) else ev_ref
        presets = []
        def fn(t):
            o1 = PL.ensemble(ev_s1_compiled, t.x15(), dev, tta=False, half=True)
            p = PL.pick_preset(t, o1, astc_threads=a.astc_threads)
            presets.append(p)
            o2 = PL.ensemble(ev_run, PL.refiner_input(t, o1, p, extended=True, astc_threads=a.astc_threads), dev, tta=False, half=True)
            o3 = PL.ensemble(ev_run, PL.refiner_input(t, o2, p, extended=True, astc_threads=a.astc_threads), dev, tta=False, half=True)
            return PL.finish((1.0 / 3.0) * o2 + (2.0 / 3.0) * o3)
        if step == 0:
            m0, _ = score_set(lambda t: PL.finish(PL.ensemble(ev_s1_compiled, t.x15(), dev, tta=False, half=True)), real, dev)
            say(f'    stage 1 alone (single pass): {MT.fmt(m0)}')
        m, rs = score_set(fn, real, dev)
        if ev_ref is ref:
            ref.train()
        say(f'--- step {step}: real test (2-pass {"ema" if ev_ref is ema else "raw"}, {time.time() - t_ev:.1f}s)  {MT.fmt(m)}   [' +
            '; '.join(f"{r['name'].split('_')[-1]} {r['psnr']:.2f}/{r['edges']:.2f} ({p})" for r, p in zip(rs, presets)) + ']')
        meta = dict(step=step, kind='refiner', channels=a.channels, blocks=a.blocks, cin=M.CIN2_V21, ca=a.use_ca, score=m, stage1=a.stage1)
        M.save(ev_ref, os.path.join(a.out_dir, 'daiblos_refiner_last.pt'), **meta)
        torch.save({
            'net': {k: v.detach().cpu() for k, v in ref.state_dict().items()},
            'ema': {k: v.detach().cpu() for k, v in ema.state_dict().items()} if ema is not None else None,
            'opt': opt.state_dict(), 'step': step, 'best': best,
        }, state_path + '.part')
        os.replace(state_path + '.part', state_path)
        if m['psnr'] > best['psnr']:
            best['psnr'] = m['psnr']
            M.save(ev_ref, os.path.join(a.out_dir, 'daiblos_refiner.pt'), **meta)
        if m['seam'] < best['seam']:
            best['seam'] = m['seam']
            M.save(ev_ref, os.path.join(a.out_dir, 'daiblos_refiner_best_seam.pt'), **meta)

    if start == 0 and a.init and os.path.exists(a.init):
        evaluate(0)

    say('starting refiner training loop (step 1 includes one-time compile & worker warmup)...')
    t0 = time.time()
    t_steady = None
    for s in range(start, a.steps):
        cur_lr = _lr_at(s, a.steps, a.lr, a.lr_min, a.warmup, a.cosine_from)
        for g in opt.param_groups:
            g['lr'] = cur_lr
        x15, y, flags, pid = next(stream)
        preset = C.PRESETS[int(pid)]
        xg = x15.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / scale

        if a.s1_tta >= 2:
            o1 = _stage1_tta2(net1, xg, amp_dtype, use_amp, ac_dev)
        else:
            with torch.no_grad(), torch.autocast(ac_dev, dtype=amp_dtype, enabled=use_amp):
                o1 = net1(xg).float()
        XU.mark_step(backend)

        o1n = (o1.clamp(0, 1) * 255.0).permute(0, 2, 3, 1).cpu().numpy()
        xn = x15.numpy()
        fl = flags.numpy()

        # Optional unrolled Pass-2 rollout: run Pass 1 with no_grad, re-encode Pass-1 output, then train on Pass 2!
        if a.pass2_prob > 0 and random.random() < a.pass2_prob:
            xin1 = _build_ref_batch(xn, o1n, fl, preset, pos, True, a.astc_threads)
            xin1_t = torch.from_numpy(xin1).to(dev, non_blocking=True).permute(0, 3, 1, 2)
            with torch.no_grad(), torch.autocast(ac_dev, dtype=amp_dtype, enabled=use_amp):
                o_p1 = ref(xin1_t).float()
            XU.mark_step(backend)
            o_cur_n = (o_p1.clamp(0, 1) * 255.0).permute(0, 2, 3, 1).cpu().numpy()
            xin = _build_ref_batch(xn, o_cur_n, fl, preset, pos, True, a.astc_threads)
        else:
            xin = _build_ref_batch(xn, o1n, fl, preset, pos, True, a.astc_threads)

        xin_t = torch.from_numpy(xin).to(dev, non_blocking=True).permute(0, 3, 1, 2)
        y_t = y.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0

        def _forward_loss(model):
            with torch.autocast(ac_dev, dtype=amp_dtype, enabled=use_amp):
                o = model(xin_t)
            o = o.float()
            ya = y_t[:, 3:4]
            live = (ya > 0).float()
            w = (1.0 + 3.0 * ((ya > 0) & (ya < 1)).float()) * (0.1 + 0.9 * live)
            wnorm = w.sum().clamp(min=1.0)
            diff = o - y_t
            loss = torch.zeros((), device=dev)
            if a.l1_weight > 0:
                loss = loss + a.l1_weight * (w * diff.abs()).sum() / (wnorm * 4.0)
            if a.l2_weight > 0:
                loss = loss + a.l2_weight * ((w * diff * diff).sum() / (wnorm * 4.0) + 1e-10).sqrt()
            if a.hue_weight > 0:
                hm = (ya > 2 / 255).float()
                loss = loss + a.hue_weight * (hm * (o[:, :3] - y_t[:, :3]).abs() / ya.clamp(min=0.03)).sum() / (hm.sum().clamp(min=1.0) * 3.0)
            if a.seam_weight > 0 or a.seam_l2_weight > 0 or a.ssim_weight > 0:
                with torch.no_grad():
                    S, valid = MT.own_colour(y_t, sigmas=(1.5, 3, 6, 12))
                vwnorm = (valid * w).sum().clamp(min=1.0)
                if a.seam_weight > 0 or a.seam_l2_weight > 0:
                    D_se = (o[:, :3] - y_t[:, :3]) - S * (o[:, 3:4] - ya)
                    e = D_se * valid * w
                    if a.seam_weight > 0:
                        loss = loss + a.seam_weight * (e.abs().sum() / (vwnorm * 3.0) + 2.0 * MT.gauss((e * lumw).sum(1, keepdim=True), 1.5).abs().sum() / vwnorm)
                    if a.seam_l2_weight > 0:
                        band = (torch.nn.functional.max_pool2d(((ya > 0) & (ya < 1)).float() * valid, 3, 1, 1) > 0).float() * valid
                        coh = MT.gauss((D_se * lumw).sum(1, keepdim=True) * band, 1.5) / MT.gauss(band, 1.5).clamp(min=1e-3)
                        loss = loss + a.seam_l2_weight * ((coh * coh * band).sum() / band.sum().clamp(min=1.0) + 1e-10).sqrt()
                if a.ssim_weight > 0:
                    sm = MT.ssim_map(o[:, :3] + (1 - o[:, 3:4]) * S, y_t[:, :3] + (1 - ya) * S).mean(1, keepdim=True)
                    loss = loss + a.ssim_weight * ((1 - sm) * valid * w).sum() / vwnorm
            return loss

        opt.zero_grad(set_to_none=True)
        try:
            loss = _forward_loss(run_ref)
            if backend not in ('tpu', 'xla') and not torch.isfinite(loss):
                say(f'step {s + 1}: non-finite loss, skipping')
                opt.zero_grad(set_to_none=True)
                continue
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
                XU.step_optimizer(opt, backend)
        except Exception as e:
            if compiled and s == start:
                say(f'torch.compile failed on first step ({type(e).__name__}: {e}); falling back to eager TPU!')
                run_ref = ref
                compiled = False
                opt.zero_grad(set_to_none=True)
                loss = _forward_loss(run_ref)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
                XU.step_optimizer(opt, backend)
            else:
                raise

        if ema is not None:
            d = min(a.ema, (1 + s) / (10 + s))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), ref.parameters()):
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
