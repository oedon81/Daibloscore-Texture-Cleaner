#!/usr/bin/env python3
"""Score the cleaner on a real triple: PC DXT5 + Android ASTC 6x6 + Clean original.

  python evaluate.py --pc path/to/pc_dxt5 --android path/to/android_astc --clean path/to/clean.png

Inputs may be decoded RGBA PNGs or Unity AssetBundles.
"""
import os, argparse
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pc', required=True, help='PC DXT5 texture PNG or Unity bundle')
    ap.add_argument('--android', required=True, help='Android ASTC 6x6 texture PNG or Unity bundle')
    ap.add_argument('--clean', required=True, help='Clean original PNG or lossless Unity bundle')
    ap.add_argument('--name', default=None, help='Texture name inside bundle (auto if single Texture2D)')
    ap.add_argument('--stage1', default=None, help='stage-1 model (default: models/daiblos_stage1.pt)')
    ap.add_argument('--refiner', default=None, help='refiner model (default: models/daiblos_refiner.pt)')
    ap.add_argument('--astc-threads', type=int, default=None)
    ap.add_argument('--tile', type=int, default=None, help='tile size multiple of 12 (default: 1032 GPU)')
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()

    from dclean import bundle as B, codec as C, metrics as MT, models as M, pipeline as PL, xla_util as XU

    pc_map = B.extract_textures(a.pc)
    an_map = B.extract_textures(a.android)
    cl_map = B.extract_textures(a.clean)

    def _pick(m, fallback_key):
        if a.name and a.name in m:
            return m[a.name][1]
        if fallback_key in m:
            return m[fallback_key][1]
        if len(m) == 1:
            return next(iter(m.values()))[1]
        raise SystemExit(f'Multiple textures found {list(m.keys())}; specify --name.')

    key = a.name or next(iter(pc_map))
    dxt = _pick(pc_map, key)
    astc = _pick(an_map, key)
    y_raw = _pick(cl_map, key)
    if dxt.shape != y_raw.shape or astc.shape != y_raw.shape:
        raise SystemExit(f'Size mismatch: PC {dxt.shape[:2]}, Android {astc.shape[:2]}, Clean {y_raw.shape[:2]}')

    tex = PL.Texture(dxt, astc)
    y_pm = C.premul(y_raw) if C.is_straight(y_raw, 0.05) else y_raw

    dev = 'cpu' if a.cpu else XU.get_device()[0]
    half = not a.fp32
    net1, _ = M.load(a.stage1 or PL.STAGE1, dev)
    ref_path = a.refiner or PL.REFINER
    ref = M.load(ref_path, dev)[0] if os.path.exists(ref_path) else None
    net1 = PL.optimize_for_gpu(net1, dev, half=half)
    if ref is not None:
        ref = PL.optimize_for_gpu(ref, dev, half=half)

    print(f"{'raw PC DXT5':36s} {MT.fmt(MT.score(tex.dxt, y_pm, dev))}", flush=True)
    print(f"{'raw Android ASTC 6x6':36s} {MT.fmt(MT.score(tex.astc, y_pm, dev))}", flush=True)

    x15 = tex.x15(half=(half and dev != 'cpu'))
    o1_1x = PL.ensemble(net1, x15, dev, tta=1, half=half, tile=a.tile)
    print(f"{'stage 1, single pass (1x)':36s} {MT.fmt(MT.score(PL.clear_background(PL.finish(o1_1x), tex), y_pm, dev))}", flush=True)
    del o1_1x

    o1_4x = PL.ensemble(net1, x15, dev, tta=4, half=half, tile=a.tile)
    print(f"{'stage 1, 4x self-ensemble':36s} {MT.fmt(MT.score(PL.clear_background(PL.finish(o1_4x), tex), y_pm, dev))}", flush=True)
    del x15

    if ref is not None:
        ext = getattr(ref, 'cin', 26) == 28
        p = PL.pick_preset(tex, o1_4x, astc_threads=a.astc_threads)
        ri1 = PL.refiner_input(tex, o1_4x, p, extended=ext, astc_threads=a.astc_threads, half=(half and dev != 'cpu'))
        o_p1 = PL.ensemble(ref, ri1, dev, tta=1, half=half, tile=a.tile)
        del ri1
        ri2 = PL.refiner_input(tex, o_p1, p, extended=ext, astc_threads=a.astc_threads, half=(half and dev != 'cpu'))
        o_p2 = PL.ensemble(ref, ri2, dev, tta=1, half=half, tile=a.tile)
        del ri2
        full = PL.clear_background(PL.finish((1.0 / 3.0) * o_p1 + (2.0 / 3.0) * o_p2), tex)
        print(f"{'full pipeline (default, ' + p + ')':36s} {MT.fmt(MT.score(full, y_pm, dev))}", flush=True)


if __name__ == '__main__':
    main()
