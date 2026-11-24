"""Daibloscore exact game encoders (PC DXT5 + Android ASTC 6x6 3.5) and side information.
Optimized for Colab: uses /dev/shm RAM-disk, explicit astcenc -j thread control, and parallel DXT5+ASTC batch calls."""
import os, sys, subprocess, tempfile, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from . import dxt5 as dxt5_game

SHM = '/dev/shm' if os.path.isdir('/dev/shm') else None
PRESETS = ('medium', 'thorough')
_tls = threading.local()
_enc_pool = None
_enc_pool_lock = threading.Lock()


def _astcenc_path():
    p = os.environ.get('DAIBLOS_ASTCENC')
    if p and os.path.exists(p):
        return p
    for cand in [
        os.path.join(HERE, 'bin', 'astcenc-3.5'),
        os.path.join(HERE, 'enc_check', 'tools', 'bin', 'astcenc-3.5'),
    ]:
        if os.path.exists(cand):
            return cand
    return 'astcenc-3.5'


def _astc_decode(data, w, h):
    import astc_encoder as A
    ctx = getattr(_tls, 'actx', None)
    if ctx is None:
        ctx = A.ASTCContext(A.ASTCConfig(A.ASTCProfile.LDR, 6, 6, 1, 60.0))
        _tls.actx = ctx
    im = A.ASTCImage(A.ASTCType.U8, w, h, 1)
    ctx.decompress(data, im, A.ASTCSwizzle())
    return np.frombuffer(im.data, np.uint8).reshape(h, w, 4).copy()


def astc_bu(bu, preset, threads=None):
    """bottom-up image -> decoded bottom-up image after the game's ASTC 6x6 encode."""
    h, w = bu.shape[:2]
    cmd = [_astcenc_path(), '-cl']
    with tempfile.TemporaryDirectory(dir=SHM) as d:
        inp = d + '/i.png'
        out = d + '/o.astc'
        Image.fromarray(np.ascontiguousarray(bu)).save(inp, compress_level=0)
        args = cmd + [inp, out, '6x6', '-' + preset, '-silent']
        if threads is not None:
            args += ['-j', str(int(threads))]
        subprocess.run(args, check=True, capture_output=True)
        return _astc_decode(open(out, 'rb').read()[16:], w, h)


def dxt5_bu(bu):
    """bottom-up image (any size) -> decoded bottom-up image after the game's DXT5 encode."""
    h, w = bu.shape[:2]
    H4, W4 = -(-h // 4) * 4, -(-w // 4) * 4
    p = np.zeros((H4, W4, 4), np.uint8)
    p[:h, :w] = bu
    return dxt5_game.decode(dxt5_game.encode(p))[:h, :w]


def game_encode(td, preset, which=('dxt5', 'astc'), astc_threads=None):
    """top-down stored-domain image -> {'dxt5': decoded, 'astc': decoded}, top-down."""
    bu = np.ascontiguousarray(td[::-1])
    out = {}
    if 'dxt5' in which and 'astc' in which:
        global _enc_pool
        if _enc_pool is None:
            _enc_pool = ThreadPoolExecutor(max_workers=2)
        fd = _enc_pool.submit(dxt5_bu, bu)
        fa = _enc_pool.submit(astc_bu, bu, preset, astc_threads)
        out['dxt5'] = np.ascontiguousarray(fd.result()[::-1])
        out['astc'] = np.ascontiguousarray(fa.result()[::-1])
    else:
        if 'dxt5' in which:
            out['dxt5'] = np.ascontiguousarray(dxt5_bu(bu)[::-1])
        if 'astc' in which:
            out['astc'] = np.ascontiguousarray(astc_bu(bu, preset, astc_threads)[::-1])
    return out


def game_encode_batch(patches, preset, astc_threads=None, parallel=False):
    """Many P x P top-down patches (P a multiple of 12), each treated as its own texture: one encoder call per format."""
    P = patches[0].shape[0]
    assert P % 12 == 0
    bu = np.ascontiguousarray(np.concatenate([p[::-1] for p in patches], 0))
    if parallel:
        global _enc_pool
        if _enc_pool is None:
            _enc_pool = ThreadPoolExecutor(max_workers=2)
        fd = _enc_pool.submit(dxt5_bu, bu)
        fa = _enc_pool.submit(astc_bu, bu, preset, astc_threads)
        d, a = fd.result(), fa.result()
    else:
        d, a = dxt5_bu(bu), astc_bu(bu, preset, astc_threads)
    n = len(patches)
    return (
        [np.ascontiguousarray(d[i * P:(i + 1) * P][::-1]) for i in range(n)],
        [np.ascontiguousarray(a[i * P:(i + 1) * P][::-1]) for i in range(n)],
    )


def is_straight(a, thr=0.02):
    part = (a[..., 3] > 0) & (a[..., 3] < 255)
    return bool(part.any() and (a[..., :3].max(2) > a[..., 3])[part].mean() >= thr)


def premul(a):
    f = a.astype(np.float32)
    f[..., :3] *= f[..., 3:] / 255.0
    return (f + 0.5).astype(np.uint8)


def unpremul(a):
    f = a.astype(np.float32)
    al = f[..., 3:]
    f[..., :3] = np.where(al > 0, f[..., :3] * 255.0 / np.maximum(al, 1.0), 0.0)
    return (f.clip(0, 255) + 0.5).astype(np.uint8)


NSIDE = 7
SIDE_SCALE = np.array([255] * 8 + [255, 255, 1, 3, 3, 5, 5], np.float32)


def side_u8(dxt_td):
    """From decoded DXT5 texture (stored domain), top-down -> (H, W, 3) uint8: (alpha_lo, alpha_hi, solid_flag)."""
    h, w = dxt_td.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    pad = np.zeros((h + ph, w + pw, 4), np.uint8)
    pad[ph:, :w] = dxt_td
    H4, W4 = (h + ph) // 4, (w + pw) // 4
    bl = pad.reshape(H4, 4, W4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(H4, W4, 16, 4)
    a = bl[..., 3]
    a0, a1 = a.max(2, keepdims=True).astype(np.int32), a.min(2, keepdims=True).astype(np.int32)
    step = (a0 - a1) / 14.0
    lo = np.maximum(a1, np.floor(a - step)).astype(np.uint8)
    hi = np.minimum(a0, np.ceil(a + step)).astype(np.uint8)
    solid = (bl[..., :3].max(2) == bl[..., :3].min(2)).all(-1).astype(np.uint8)
    s = np.empty((H4, W4, 16, 3), np.uint8)
    s[..., 0], s[..., 1], s[..., 2] = lo, hi, solid[..., None]
    return s.reshape(H4, W4, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(H4 * 4, W4 * 4, 3)[ph:, :w].copy()


def pos_u8(P, y0, x0, H, W):
    """Bottom-up block-phase coordinates for a (P, P) crop starting at (y0, x0) in an (H, W) texture."""
    ys = (H - 1 - (y0 + np.arange(P)))[:, None]
    xs = (x0 + np.arange(P))[None, :]
    out = np.empty((P, P, 4), np.uint8)
    out[..., 0], out[..., 1] = xs % 4, ys % 4
    out[..., 2], out[..., 3] = xs % 6, ys % 6
    return out


def stage1_u8(dxt, astc, side, pos):
    return np.concatenate([dxt, astc, side, pos], -1)
