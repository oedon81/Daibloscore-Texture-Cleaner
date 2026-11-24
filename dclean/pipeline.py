"""Full Daibloscore cleaner pipeline: Stage 1 + Stage 2 Refiner (2 passes, linear 1/3+2/3 blend).
Memory-safe tiled execution: keeps full canvas on CPU in fp16/fp32 and streams one tile at a time to GPU
(< 2.5 GB VRAM and < 1.5 GB RAM even on 4096x4096 textures)."""
import os, numpy as np, torch
from . import codec as C, models as M, xla_util as XU

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1 = os.path.join(ROOT, 'models', 'daiblos_stage1.pt')
REFINER = os.path.join(ROOT, 'models', 'daiblos_refiner.pt')
PAD = 60  # multiple of LCM(4, 6) = 12 so PixelUnshuffle(2) and 4x4/6x6 block grids stay phase-aligned across tiles

_TTA_COMBOS = {
    1: ((0, False),),
    2: ((0, False), (2, True)),
    4: ((0, False), (1, False), (2, False), (3, False)),
    8: tuple((k, f) for k in range(4) for f in (False, True)),
}


def finish(o):
    o = np.clip(o + 0.5, 0, 255).astype(np.uint8)
    o[..., :3] = np.minimum(o[..., :3], o[..., 3:4])
    return o


class Texture:
    """Holds one texture's inputs. dxt, astc: decoded game textures, top-down, stored alpha domain."""
    def __init__(self, dxt, astc, straight=None):
        self.straight = C.is_straight(astc, 0.05) if straight is None else straight
        self.side = C.side_u8(dxt)
        self.H, self.W = dxt.shape[:2]
        self.pos = C.pos_u8(self.H, 0, 0, self.H, self.W) if self.H == self.W else _pos_rect(self.H, self.W)
        self.dxt_st, self.astc_st = dxt, astc
        self.dxt, self.astc = (C.premul(dxt), C.premul(astc)) if self.straight else (dxt, astc)

    def x15(self, half=False):
        dt = np.float16 if half else np.float32
        H, W = self.H, self.W
        out = np.empty((H, W, 15), dt)
        inv255 = np.float32(1.0 / 255.0)
        out[..., 0:4] = self.dxt.astype(np.float32) * inv255
        out[..., 4:8] = self.astc.astype(np.float32) * inv255
        out[..., 8:10] = self.side[..., :2].astype(np.float32) * inv255
        out[..., 10] = self.side[..., 2]
        out[..., 11:13] = self.pos[..., :2].astype(np.float32) * np.float32(1.0 / 3.0)
        out[..., 13:15] = self.pos[..., 2:4].astype(np.float32) * np.float32(1.0 / 5.0)
        return out


def _pos_rect(H, W):
    ys = (H - 1 - np.arange(H))[:, None]
    xs = np.arange(W)[None, :]
    out = np.empty((H, W, 4), np.uint8)
    out[..., 0], out[..., 1] = xs % 4, ys % 4
    out[..., 2], out[..., 3] = xs % 6, ys % 6
    return out


def _extract_tile(t_cpu, yy, xx, tw, hh, ww):
    """Extract a (1, C, tw, tw) zero-padded tile centered at [yy-PAD : yy-PAD+tw, xx-PAD : xx-PAD+tw] from CPU tensor."""
    y0, x0 = yy - PAD, xx - PAD
    y1, x1 = y0 + tw, x0 + tw
    sy0, sy1 = max(0, y0), min(hh, y1)
    sx0, sx1 = max(0, x0), min(ww, x1)
    sub = t_cpu[:, :, sy0:sy1, sx0:sx1]
    pt, pb = sy0 - y0, y1 - sy1
    pl, pr = sx0 - x0, x1 - sx1
    if pt or pb or pl or pr:
        return torch.nn.functional.pad(sub, (pl, pr, pt, pb))
    return sub


def _tiles(net, t_cpu, dev, tile, half, dtype=torch.float16, tile_batch=1, recompute_pos=True):
    """Run net over (1, C, hh, ww) CPU tensor one tile (or tile_batch) at a time, computing periodic pos on-the-fly."""
    hh, ww = t_cpu.shape[2], t_cpu.shape[3]
    is_xla = XU.is_xla_dev(dev)
    is_tpu = XU.is_tpu_dev(dev)
    offload = is_xla or is_tpu
    ac_dev = 'xla' if is_xla else ('tpu' if is_tpu else torch.device(dev).type)
    p0 = next((p for p in net.parameters()), None)
    net_half = (p0 is not None and p0.dtype in (torch.float16, torch.bfloat16))
    net_cl = (p0 is not None and p0.ndim == 4 and p0.is_contiguous(memory_format=torch.channels_last))
    use_ac = half and (not net_half) and (ac_dev in ('cuda', 'tpu', 'xla')) and XU.autocast_ok(ac_dev, dtype)
    in_dt = p0.dtype if net_half else torch.float32
    tw = tile + 2 * PAD
    # Keep full-image output on CPU so GPU VRAM only ever holds a single tile (~1.8-2.5 GB max on 4K)
    out = torch.zeros((4, hh, ww), dtype=torch.float32, device='cpu')
    coords = [(yy, xx) for yy in range(0, hh, tile) for xx in range(0, ww, tile)]
    tb = max(1, int(tile_batch))
    dev_t = torch.device(dev)
    ar_tw = torch.arange(tw, device=dev_t, dtype=torch.float32) if recompute_pos else None

    for i in range(0, len(coords), tb):
        chunk = coords[i:i + tb]
        patches = []
        for yy, xx in chunk:
            patch = _extract_tile(t_cpu, yy, xx, tw, hh, ww)
            if patch.device != dev_t:
                patch = patch.to(dev_t, dtype=in_dt, non_blocking=False)
            else:
                patch = patch.to(dtype=in_dt)
            if recompute_pos:
                xs = ar_tw + float(xx - PAD)
                ys = float(hh - 1 - (yy - PAD)) - ar_tw
                px4 = torch.remainder(xs, 4.0).div_(3.0).to(in_dt).view(1, 1, 1, tw).expand(1, 1, tw, tw)
                py4 = torch.remainder(ys, 4.0).div_(3.0).to(in_dt).view(1, 1, tw, 1).expand(1, 1, tw, tw)
                px6 = torch.remainder(xs, 6.0).div_(5.0).to(in_dt).view(1, 1, 1, tw).expand(1, 1, tw, tw)
                py6 = torch.remainder(ys, 6.0).div_(5.0).to(in_dt).view(1, 1, tw, 1).expand(1, 1, tw, tw)
                patch = torch.cat([patch, px4, py4, px6, py6], dim=1)
            patches.append(patch)
        n_real = len(patches)
        while len(patches) < tb and offload:
            patches.append(patches[-1])
        batch_t = patches[0] if len(patches) == 1 else torch.cat(patches, dim=0)
        if net_cl:
            batch_t = batch_t.contiguous(memory_format=torch.channels_last)
        with torch.autocast(ac_dev, dtype=dtype, enabled=use_ac):
            r_all = net(batch_t).float()
        if is_xla:
            XU.mark_step('xla')
        r_cpu = r_all[:, :, PAD:PAD + tile, PAD:PAD + tile].cpu()
        del batch_t, patches, r_all
        for bi in range(n_real):
            yy, xx = chunk[bi]
            vh, vw = min(tile, hh - yy), min(tile, ww - xx)
            out[:, yy:yy + vh, xx:xx + vw] = r_cpu[bi, :, :vh, :vw]
    return out


def optimize_for_gpu(net, dev, half=True, compile_model=False):
    """Prepare a loaded inference model for fast GPU execution (fp16 weights + channels_last + optional torch.compile)."""
    dev_t = torch.device(dev)
    if dev_t.type != 'cuda':
        return net
    if half:
        # fp16 preserves 0..255 inputs to ~0.25 gray-level precision (unlike bf16's 1-level ULP) and is fast on CUDA & ROCm
        net = net.to(device=dev_t, dtype=torch.float16)
    net = net.to(memory_format=torch.channels_last)
    if compile_model:
        try:
            import shutil, sys
            if not shutil.which('g++') and not shutil.which('clang++'):
                try:
                    import ziglang  # noqa: F401
                    os.environ.setdefault('CXX', f'{sys.executable} -m ziglang c++')
                except Exception:
                    pass
            net = torch.compile(net, dynamic=False)
        except Exception as e:
            print(f'Warning: torch.compile failed ({e}); using eager GPU.', flush=True)
    return net


def _parse_tta(tta):
    if isinstance(tta, bool):
        return _TTA_COMBOS[8 if tta else 1]
    n = int(tta)
    if n not in _TTA_COMBOS:
        raise ValueError(f'tta must be one of {sorted(_TTA_COMBOS)} or bool, got {tta!r}')
    return _TTA_COMBOS[n]


@torch.no_grad()
def ensemble(net, x, dev, tta=4, half=True, tile=None, tile_batch=None, recompute_pos=None):
    """Run net over (H, W, C) float input (with 4 trailing pos channels) in 1/2/4/8x D4 TTA -> (H, W, 4) float 0..255.
    Pads top/right to a multiple of LCM(4, 6)=12 so bottom-left-anchored DXT5 (4x4) and ASTC (6x6) block grids stay
    aligned at exact 0 mod 12 across every tile and D4 orientation, with periodic pos computed per tile."""
    offload = XU.is_tpu_or_xla(dev)
    if tile is None:
        tile = 504 if offload else 768  # multiples of 12; 768 keeps peak GPU VRAM ~2.5 GB on 4096x4096
    if tile_batch is None:
        tile_batch = 4 if offload else 1
    if recompute_pos is None:
        recompute_pos = (x.shape[2] >= 15)
    if offload:
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    combos = _parse_tta(tta)
    H, W = x.shape[:2]
    src_dt = torch.float16 if (half and not offload and torch.device(dev).type == 'cuda') else torch.float32
    if recompute_pos:
        pt, pr = (-H) % 12, (-W) % 12
        # Build (1, C-4, H+pt, W+pr) directly on CPU without duplicating full arrays
        c_no_pos = x.shape[2] - 4
        base = torch.zeros((1, c_no_pos, H + pt, W + pr), dtype=src_dt)
        # Copy channel-by-channel to avoid huge HWC->CHW temporary transpose peak
        for ci in range(c_no_pos):
            base[0, ci, pt:pt + H, :W] = torch.from_numpy(x[..., ci]).to(src_dt)
    else:
        pt, pr = 0, 0
        c_all = x.shape[2]
        base = torch.empty((1, c_all, H, W), dtype=src_dt)
        for ci in range(c_all):
            base[0, ci] = torch.from_numpy(x[..., ci]).to(src_dt)

    acc = None
    for k, f in combos:
        t = torch.rot90(base, k, (2, 3)) if k else base
        t = torch.flip(t, (3,)) if f else t
        o = _tiles(net, t, dev, tile, half, dtype, tile_batch=tile_batch, recompute_pos=recompute_pos).clamp_(0, 1).mul_(255)
        o = torch.flip(o, (2,)) if f else o
        if k:
            o = torch.rot90(o, -k, (1, 2))
        if acc is None:
            acc = o
        else:
            acc.add_(o)
        del t, o
    del base
    if len(combos) > 1:
        acc.div_(len(combos))
    out = acc.permute(1, 2, 0).numpy()
    if pt or pr:
        out = out[pt:pt + H, :W].copy()
    else:
        out = out.copy()
    return out


def _mismatch(x, y, b):
    h, w = x.shape[:2]
    ph, pw = (-h) % b, (-w) % b
    m = np.zeros((h + ph, w + pw), bool)
    m[ph:, :w] = (x != y).any(-1) if x.ndim == 3 else (x != y)
    m = m.reshape((h + ph) // b, b, (w + pw) // b, b).any((1, 3))
    return np.repeat(np.repeat(m, b, 0), b, 1)[ph:, :w]


def refiner_input_arrays(dxt, astc, o, dxt_re, astc_re, pos, extended=True, half=False):
    """Build 28-ch (or 26-ch legacy) refiner input in a single pre-allocated array (minimal RAM peak)."""
    H, W = dxt.shape[:2]
    cin = 28 if extended else 26
    dt = np.float16 if half else np.float32
    out = np.empty((H, W, cin), dt)
    inv255 = np.float32(1.0 / 255.0)
    out[..., 0:4] = dxt.astype(np.float32) * inv255
    out[..., 4:8] = astc.astype(np.float32) * inv255
    out[..., 8:12] = o.astype(np.float32) * inv255
    out[..., 12:16] = (dxt_re.astype(np.int16) - dxt.astype(np.int16)).astype(np.float32) * inv255
    out[..., 16:20] = (astc_re.astype(np.int16) - astc.astype(np.int16)).astype(np.float32) * inv255
    if extended:
        out[..., 20] = _mismatch(dxt_re[..., :3], dxt[..., :3], 4)
        out[..., 21] = _mismatch(dxt_re[..., 3], dxt[..., 3], 4)
        out[..., 22] = _mismatch(astc_re[..., :3], astc[..., :3], 6)
        out[..., 23] = _mismatch(astc_re[..., 3], astc[..., 3], 6)
        p0 = 24
    else:
        out[..., 20] = _mismatch(dxt_re, dxt, 4)
        out[..., 21] = _mismatch(astc_re, astc, 6)
        p0 = 22
    out[..., p0:p0 + 2] = pos[..., :2].astype(np.float32) * np.float32(1.0 / 3.0)
    out[..., p0 + 2:p0 + 4] = pos[..., 2:4].astype(np.float32) * np.float32(1.0 / 5.0)
    return out


def refiner_input(t, o, preset, enc=None, extended=True, astc_threads=None, half=False):
    u = finish(o)
    st = C.unpremul(u) if t.straight else u
    e = enc if enc is not None else C.game_encode(st, preset, astc_threads=astc_threads)
    d, a = (C.premul(e['dxt5']), C.premul(e['astc'])) if t.straight else (e['dxt5'], e['astc'])
    return refiner_input_arrays(t.dxt, t.astc, o, d, a, t.pos, extended=extended, half=half)


def pick_preset(t, o, astc_threads=None, crop=768):
    """Detect whether the game encoded ASTC 6x6 with -medium or -thorough.
    Tests a single 6x6-phase-aligned high-detail crop (~0.5s) instead of re-encoding a full 4K texture (~40s)."""
    u = finish(o)
    st = C.unpremul(u) if t.straight else u
    ref_astc = t.astc_st
    H, W = st.shape[:2]
    if crop and H > crop and W > crop:
        cs = (crop // 6) * 6
        best_pos = (H - cs, 0)
        best_score = -1.0
        # Bottom-anchored y0: (H - (y0 + cs)) % 6 == 0  =>  y0 = H - cs - 6*k
        ystep = max(6, ((H - cs) // 3 // 6) * 6)
        xstep = max(6, ((W - cs) // 3 // 6) * 6)
        ys = list(range(H - cs, -1, -ystep))
        xs = list(range(0, W - cs + 1, xstep))
        a = t.astc[..., 3]
        for y0 in ys:
            for x0 in xs:
                sub_a = a[y0:y0 + cs, x0:x0 + cs]
                sc = float(((sub_a > 0) & (sub_a < 255)).mean() * 4.0 + (sub_a > 0).mean())
                if sc > best_score:
                    best_score = sc
                    best_pos = (y0, x0)
        y0, x0 = best_pos
        st = np.ascontiguousarray(st[y0:y0 + cs, x0:x0 + cs])
        ref_astc = np.ascontiguousarray(ref_astc[y0:y0 + cs, x0:x0 + cs])
    best = None
    for p in C.PRESETS:
        a = C.game_encode(st, p, ('astc',), astc_threads=astc_threads)['astc']
        ok = 1.0 - _mismatch(a, ref_astc, 6).mean()
        if best is None or ok > best[1]:
            best = (p, ok)
    return best[0]


def clear_background(o, t, thr=2):
    if not thr:
        return o
    o = o.copy()
    o[(o[..., 3] <= thr) & (t.dxt[..., 3] == 0) & (t.astc[..., 3] == 0)] = 0
    return o


@torch.no_grad()
def run(t, net1, ref=None, dev='cuda', tta=4, ref_tta=1, passes=2, half=True, preset=None, clear_bg=2, blend='linear', astc_threads=None, tile=None, tile_batch=None):
    """Run full cleaner pipeline.
    Default (tta=4, ref_tta=1, passes=2): 6 total passes (4 Stage-1 rotations + 2 single-pass refiner iterations)
    for near-identical quality to 24 passes (~0.04 dB / 0.001 seam difference) at ~3-4x higher speed."""
    use_hp_arr = bool(half and torch.device(dev).type == 'cuda')
    x15 = t.x15(half=use_hp_arr)
    o = ensemble(net1, x15, dev, tta, half, tile=tile, tile_batch=tile_batch)
    del x15
    info = {}
    if ref is not None:
        ext = getattr(ref, 'cin', 26) == 28
        p = preset or pick_preset(t, o, astc_threads=astc_threads)
        info['preset'] = p
        outs = []
        for _ in range(passes):
            ri = refiner_input(t, o, p, extended=ext, astc_threads=astc_threads, half=use_hp_arr)
            o = ensemble(ref, ri, dev, ref_tta, half, tile=tile, tile_batch=tile_batch)
            del ri
            outs.append(o)
        if blend == 'linear' and len(outs) == 2:
            o = (1.0 / 3.0) * outs[0] + (2.0 / 3.0) * outs[1]
        else:
            o = sum(outs) / len(outs)
    return clear_background(finish(o), t, clear_bg), info
