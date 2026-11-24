"""Daibloscore PC DXT5 encoder (ISPC precise SSE2 BC1 + stb_dxt single-colour fix + ISPC BC3 alpha).
Uses /dev/shm RAM-disk when available to eliminate temp file I/O overhead."""
import os, subprocess, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHM = '/dev/shm' if os.path.isdir('/dev/shm') else None


def _bin(name):
    env_dir = os.environ.get('DAIBLOS_BIN_DIR')
    cands = []
    if env_dir:
        cands.append(os.path.join(env_dir, name))
    cands.extend([
        os.path.join(ROOT, 'bin', name),
        os.path.join(ROOT, 'enc_check', 'tools', name),
    ])
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Missing encoder binary '{name}'. Run './setup_encoders.sh' first (or use --no-refiner)."
    )


def _run(binary, img, *extra):
    h, w = img.shape[:2]
    exe = _bin(binary)
    with tempfile.TemporaryDirectory(dir=SHM) as d:
        np.ascontiguousarray(img).tofile(d + '/in.rgba')
        subprocess.run(
            [exe, d + '/in.rgba', str(w), str(h), d + '/out.bc3', *extra],
            check=True, capture_output=True,
        )
        return np.frombuffer(open(d + '/out.bc3', 'rb').read()[-w * h:], np.uint8).reshape(h // 4, w // 4, 16).copy()


def encode(img):
    h, w = img.shape[:2]
    assert h % 4 == 0 and w % 4 == 0
    out = _run('bin_ispcv_head_precise_sse2', img)
    deg = (out[..., 8:10] == out[..., 10:12]).all(-1)
    if deg.any():
        bl = img.reshape(h // 4, 4, w // 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(h // 4, w // 4, 16, 4)
        mean = (bl[..., :3].astype(np.int32).sum(2) // 16).astype(np.uint8)
        flat = np.concatenate([np.repeat(np.repeat(mean, 4, 0), 4, 1), np.full((h, w, 1), 255, np.uint8)], -1)
        solid = _run('bin_enc_stb', flat, '0')
        out[deg, 8:] = solid[deg, 8:]
    return out


def decode(blocks):
    """BC3 decode matching UnityPy (texture2ddecoder), returns RGBA in the same row order as the blocks."""
    import texture2ddecoder
    H4, W4 = blocks.shape[:2]
    return np.frombuffer(
        texture2ddecoder.decode_bc3(np.ascontiguousarray(blocks).tobytes(), W4 * 4, H4 * 4), np.uint8
    ).reshape(H4 * 4, W4 * 4, 4)[..., [2, 1, 0, 3]].copy()
