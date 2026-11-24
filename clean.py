#!/usr/bin/env python3
"""Clean Daibloscore Spine textures from paired PC (DXT5) + Android (ASTC 6x6) assets.

  python clean.py --pc PC_INPUT --android ANDROID_INPUT -o OUTPUT_DIR

PC_INPUT and ANDROID_INPUT can each be:
  - A Unity AssetBundle straight from the PC / Android game client or CDN
  - A decoded RGBA PNG pair (PC DXT5 + Android ASTC 6x6 of the same texture)
  - A directory of bundles or PNGs (matched by relative path / filename)

Outputs clean premultiplied-alpha PNGs into OUTPUT_DIR. With --copy-others,
extracts Spine .atlas and .json sidecars from the bundle (or copies them from
the input folder) so OUTPUT_DIR is a complete Spine 4.2 model.
"""
import os, time, shutil, argparse
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


def _save_png(out, dst):
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    tmp = dst + '.tmp.png'
    Image.fromarray(out, 'RGBA').save(tmp, compress_level=3)
    os.replace(tmp, dst)


def _pair_inputs(pc_path, android_path):
    if os.path.isdir(pc_path) and os.path.isdir(android_path):
        pc_root, an_root = os.path.abspath(pc_path), os.path.abspath(android_path)
        pairs = []
        for d, _, fs in os.walk(pc_root):
            for f in sorted(fs):
                p_src = os.path.join(d, f)
                rel = os.path.relpath(p_src, pc_root)
                a_src = os.path.join(an_root, rel)
                if os.path.isfile(a_src):
                    pairs.append((p_src, a_src, rel))
        return pairs
    if os.path.isfile(pc_path) and os.path.isfile(android_path):
        return [(pc_path, android_path, os.path.basename(pc_path))]
    raise SystemExit('Error: --pc and --android must both be files or both be directories.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pc', required=True, help='PC DXT5 texture, Unity bundle, or folder')
    ap.add_argument('--android', required=True, help='Android ASTC 6x6 texture, Unity bundle, or folder')
    ap.add_argument('-o', '--out', required=True, help='output directory')
    ap.add_argument('--stage1', default=None, help='stage-1 model (default: models/daiblos_stage1.pt)')
    ap.add_argument('--refiner', default=None, help='refiner model (default: models/daiblos_refiner.pt)')
    ap.add_argument('--no-refiner', action='store_true', help='skip stage 2 (no encoder binaries needed; ~0.2 dB lower)')
    ap.add_argument('--tta', type=int, default=4, choices=[1, 2, 4, 8], help='stage-1 self-ensemble passes (default: 4)')
    ap.add_argument('--refiner-tta', type=int, default=1, choices=[1, 2, 4, 8], help='refiner self-ensemble passes per iteration (default: 1)')
    ap.add_argument('--no-ensemble', action='store_true', help='shortcut for --tta 1 --refiner-tta 1 (~2.5x faster, ~0.3 dB lower)')
    ap.add_argument('--max-quality', action='store_true', help='shortcut for --tta 8 --refiner-tta 8 (24 total passes; +0.04 dB)')
    ap.add_argument('--refiner-passes', type=int, default=2, help='stage-2 re-encode passes (default 2, linear 1/3+2/3 blend)')
    ap.add_argument('--preset', default=None, choices=['medium', 'thorough'], help='override ASTC 6x6 preset (default: auto-detect)')
    ap.add_argument('--astc-threads', type=int, default=None, help='threads for astcenc-3.5 (default: auto)')
    ap.add_argument('--tile', type=int, default=None, help='tile size multiple of 12 (default: 768 GPU, 504 TPU)')
    ap.add_argument('--compile', action='store_true', help='enable torch.compile on GPU (faster for large batches after warmup)')
    ap.add_argument('--fp32', action='store_true', help='full precision on GPU (default: fp16)')
    ap.add_argument('--cpu', action='store_true', help='run on CPU')
    ap.add_argument('--copy-others', action='store_true', help='extract/copy .atlas and .json sidecars to output')
    ap.add_argument('--overwrite', action='store_true', help='overwrite existing outputs')
    a = ap.parse_args()

    from dclean import bundle as B, models as M, pipeline as PL, xla_util as XU

    if a.cpu:
        dev = 'cpu'
    else:
        dev, _, _, _ = XU.get_device()

    s1_path = a.stage1 or PL.STAGE1
    ref_path = None if a.no_refiner else (a.refiner or PL.REFINER)
    net1, _ = M.load(s1_path, dev)
    ref = M.load(ref_path, dev)[0] if (ref_path and os.path.exists(ref_path)) else None
    if not a.no_refiner and ref is None:
        print(f'Note: refiner not found at {ref_path}; running Stage 1 only.', flush=True)
    net1 = PL.optimize_for_gpu(net1, dev, half=not a.fp32, compile_model=a.compile)
    if ref is not None:
        ref = PL.optimize_for_gpu(ref, dev, half=not a.fp32, compile_model=a.compile)

    pairs = _pair_inputs(a.pc, a.android)
    print(f'{len(pairs)} paired input(s); running on {dev} (stage1 + {"2-pass refiner" if ref else "no refiner"})', flush=True)

    t0 = time.time()
    n = 0
    pending = []
    with ThreadPoolExecutor(max_workers=2) as io_pool:
        for pc_src, an_src, rel in pairs:
            is_ab = B.is_bundle(pc_src) or B.is_bundle(an_src)
            if is_ab:
                pc_texs = B.extract_textures(pc_src)
                an_texs = B.extract_textures(an_src)
                sub_dir = os.path.join(a.out, os.path.splitext(rel)[0]) if len(pairs) > 1 else a.out
                if a.copy_others and B.is_bundle(pc_src):
                    B.extract_spine_sidecars(pc_src, sub_dir)
                names = [k for k in pc_texs if k in an_texs]
            else:
                if not pc_src.lower().endswith('.png'):
                    if a.copy_others:
                        dst = os.path.join(a.out, rel)
                        os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
                        shutil.copy2(pc_src, dst)
                    continue
                pc_texs = B.extract_textures(pc_src)
                an_texs = B.extract_textures(an_src)
                k_pc = next(iter(pc_texs))
                k_an = next(iter(an_texs))
                an_texs = {k_pc: an_texs[k_an]}
                sub_dir = os.path.join(a.out, os.path.dirname(rel))
                names = [k_pc]

            for name in names:
                dst = os.path.join(sub_dir, name + '.png')
                if os.path.exists(dst) and not a.overwrite:
                    continue
                _, dxt_arr = pc_texs[name]
                _, astc_arr = an_texs[name]
                if dxt_arr.shape != astc_arr.shape:
                    print(f'  skip {name}: size mismatch PC {dxt_arr.shape[:2]} vs Android {astc_arr.shape[:2]}', flush=True)
                    continue
                tex = PL.Texture(dxt_arr, astc_arr)
                s1_tta = 1 if a.no_ensemble else (8 if a.max_quality else a.tta)
                ref_tta = 1 if a.no_ensemble else (8 if a.max_quality else a.refiner_tta)
                out, info = PL.run(
                    tex, net1, ref, dev=dev,
                    tta=s1_tta,
                    ref_tta=ref_tta,
                    passes=max(1, a.refiner_passes),
                    half=not a.fp32,
                    preset=a.preset,
                    astc_threads=a.astc_threads,
                    tile=a.tile,
                )
                pending.append(io_pool.submit(_save_png, out, dst))
                if len(pending) >= 4:
                    pending.pop(0).result()
                n += 1
                ps = f" [{info['preset']}]" if 'preset' in info else ''
                h, w = dxt_arr.shape[:2]
                print(f'[{n}] {name}.png  {w}x{h}{ps}  ({(time.time() - t0) / n:.1f}s/tex)', flush=True)
        for fut in pending:
            fut.result()
    print(f'Done: cleaned {n} texture(s) into {a.out}')


if __name__ == '__main__':
    main()
