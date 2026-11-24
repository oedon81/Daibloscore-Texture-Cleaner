"""On-the-fly patch-only encoder stream for Colab A100 (zero 45-min pre-encoding wait, zero disk I/O via /dev/shm).
Bulletproof leakage protection: excludes the 7 held-out real Daibloscore eval textures from BOTH daiblos_clean/
and other/ by filename, exact SHA-256 hash, AND 32x32 perceptual thumbnail match (catches renamed or re-saved PNGs)."""
import os, glob, json, random, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from PIL import Image
from torch.utils.data import IterableDataset
from . import codec as C, pipeline as PL

Image.MAX_IMAGE_PIXELS = None

HELD_OUT_STEMS = (
    '30431_skin_lepus04a', '30431_skin_lepus04b', '30431_skin_lepus04c',
    '60350_skin_ijen03b', '10340_skin_uluru03b', '60350_skin_ijensp04a',
    '30180_skin_machairodus04',
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _thumb32(path):
    with Image.open(path) as im:
        sz = im.size
        t = np.asarray(im.convert('RGBA').resize((32, 32), Image.Resampling.BOX), dtype=np.int16)
    return sz, t


def build_val_fingerprints(val_json_path):
    """Return (sha_set, size_to_thumbs, stem_set) for all clean PNGs in real.json."""
    base = os.path.dirname(os.path.abspath(val_json_path))
    items = json.load(open(val_json_path, encoding='utf-8'))
    shas = set()
    by_size = {}
    stems = set(HELD_OUT_STEMS)
    for e in items:
        stems.add(e['name'].lower())
        if e.get('atlas'):
            stems.add(e['atlas'].lower())
        fp = e['clean'] if os.path.isabs(e['clean']) else os.path.join(base, e['clean'])
        if os.path.exists(fp):
            shas.add(_sha256(fp))
            sz, th = _thumb32(fp)
            by_size.setdefault(sz, []).append((e['name'], th))
    return shas, by_size, stems


def list_clean_pngs(root, exclude_held=True, val_fp=None, dedup_shas=None, log_fn=None):
    """List clean PNGs under root, filtering out any held-out eval texture by name, SHA-256, OR 32x32 thumbnail."""
    files = sorted(glob.glob(os.path.join(root, '**', '*.png'), recursive=True))
    if not exclude_held and not dedup_shas:
        return files

    val_shas, val_by_size, stems = val_fp if val_fp is not None else (set(), {}, set(HELD_OUT_STEMS))
    dup_shas = dedup_shas or set()
    val_sizes = set(val_by_size.keys())

    def _check(fp):
        low_name = os.path.basename(fp).lower()
        low_dir = os.path.basename(os.path.dirname(fp)).lower()
        if exclude_held and any(h in low_name or h in low_dir for h in stems):
            return (fp, 'held_name')
        if val_shas or dup_shas:
            sh = _sha256(fp)
            if exclude_held and sh in val_shas:
                return (fp, 'held_sha')
            if sh in dup_shas:
                return (fp, 'dup_daiblos')
        if exclude_held and val_sizes:
            try:
                with Image.open(fp) as im:
                    sz = im.size
                    if sz in val_sizes:
                        th = np.asarray(im.convert('RGBA').resize((32, 32), Image.Resampling.BOX), dtype=np.int16)
                        for vname, vth in val_by_size[sz]:
                            if np.abs(th - vth).mean() < 3.0:
                                return (fp, f'held_thumb:{vname}')
            except Exception:
                return (fp, 'corrupt')
        return (fp, 'keep')

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_check, files))

    kept = []
    reasons = {}
    for fp, status in results:
        if status == 'keep':
            kept.append(fp)
        else:
            reasons[status] = reasons.get(status, 0) + 1
    if log_fn and reasons:
        log_fn(f'  filtered {root}: kept {len(kept)}/{len(files)} | excluded {reasons}')
    return kept


def real_set(real_json_path, names=None):
    base = os.path.dirname(os.path.abspath(real_json_path))
    items = json.load(open(real_json_path, encoding='utf-8'))
    out = []
    for e in items:
        if names and e['name'] not in names:
            continue
        def _rd(p):
            fp = p if os.path.isabs(p) else os.path.join(base, p)
            return np.asarray(Image.open(fp).convert('RGBA'))
        clean, dxt, astc = _rd(e['clean']), _rd(e['dxt5']), _rd(e['astc'])
        t = PL.Texture(dxt, astc)
        y = C.premul(clean) if t.straight else clean
        out.append((e['name'], t, y, e))
    return out


def _crop_patch(im, P, rng, sparse_prob=0.10):
    H, W = im.shape[:2]
    if H < P or W < P:
        pad = np.zeros((max(H, P), max(W, P), 4), np.uint8)
        pad[:H, :W] = im
        im = pad
        H, W = im.shape[:2]
    want_sparse = rng.random() < sparse_prob
    best = None
    for _ in range(12):
        y0 = rng.randint(0, H - P)
        x0 = rng.randint(0, W - P)
        p = im[y0:y0 + P, x0:x0 + P]
        a = p[..., 3]
        live = (a > 0).mean()
        if live < 0.02:
            continue
        soft = ((a > 0) & (a < 255)).mean()
        if want_sparse:
            if 0.02 <= live <= 0.35 and soft >= 0.01:
                best = p
                break
        else:
            if soft >= 0.015 or live >= 0.25:
                best = p
                break
        if best is None:
            best = p
    if best is None:
        y0 = rng.randint(0, H - P)
        x0 = rng.randint(0, W - P)
        best = im[y0:y0 + P, x0:x0 + P]
    k = rng.randint(0, 3)
    if k:
        best = np.rot90(best, k)
    if rng.random() < 0.5:
        best = best[:, ::-1]
    return np.ascontiguousarray(best)


class OnTheFlyPatchStream(IterableDataset):
    """Yields (x15 uint8, y_pm uint8) for Stage 1, or (x15 uint8, y_pm uint8, flags uint8, pid int) for Stage 2.
    straight_prob: when a source PNG is straight-alpha (e.g. RiichiCity), keep it straight-encoded with this
    probability and premultiply it BEFORE encoding otherwise (matching Daibloscore's ~85-98% premultiplied ratio)."""
    def __init__(self, groups, P, batch, thorough_frac=1 / 3, pool_size=6, sparse_prob=0.10, for_refiner=False, straight_prob=0.20):
        self.groups = [(list(fs), float(w)) for fs, w in groups if fs]
        tot = sum(w for _, w in self.groups)
        self.weights = [w / tot for _, w in self.groups]
        self.P = P
        self.batch = batch
        self.thorough_frac = thorough_frac
        self.pool_size = pool_size
        self.sparse_prob = sparse_prob
        self.for_refiner = for_refiner
        self.straight_prob = float(straight_prob)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        seed = (os.getpid() * 1315423911 + (wi.id if wi else 0) * 2654435761) & 0xFFFFFFFF
        rng = random.Random(seed)
        P = self.P
        pos = C.pos_u8(P, 0, 0, P, P)

        pools = [[] for _ in self.groups]
        def _load_img(gi):
            fs = self.groups[gi][0]
            for _ in range(10):
                fp = rng.choice(fs)
                try:
                    im = np.asarray(Image.open(fp).convert('RGBA'))
                    st = C.is_straight(im, 0.02)
                    return (im, st, 0)
                except Exception:
                    continue
            raise RuntimeError('Failed to load any clean PNG')

        for gi in range(len(self.groups)):
            while len(pools[gi]) < min(self.pool_size, len(self.groups[gi][0])):
                pools[gi].append(_load_img(gi))

        while True:
            pid = 1 if rng.random() < self.thorough_frac else 0
            preset = C.PRESETS[pid]
            stored_patches, ys, flags = [], [], []
            for _ in range(self.batch):
                r = rng.random()
                acc = 0.0
                gi = len(self.groups) - 1
                for idx, w in enumerate(self.weights):
                    acc += w
                    if r <= acc:
                        gi = idx
                        break
                pi = rng.randrange(len(pools[gi]))
                im, st, uses = pools[gi][pi]
                patch = _crop_patch(im, P, rng, self.sparse_prob)
                if st and rng.random() >= self.straight_prob:
                    # Premultiply straight-alpha sources before encoding so training matches premultiplied game damage
                    patch = C.premul(patch)
                    keep_st = False
                else:
                    keep_st = st
                stored_patches.append(patch)
                ys.append(C.premul(patch) if keep_st else patch)
                flags.append(1 if keep_st else 0)
                uses += 1
                if uses >= 48:
                    pools[gi][pi] = _load_img(gi)
                else:
                    pools[gi][pi] = (im, st, uses)

            d_list, a_list = C.game_encode_batch(stored_patches, preset, astc_threads=1, parallel=False)
            xs, ys_out = [], []
            for i in range(self.batch):
                d_st, a_st = d_list[i], a_list[i]
                sd = C.side_u8(d_st)
                d_pm, a_pm = (C.premul(d_st), C.premul(a_st)) if flags[i] else (d_st, a_st)
                xs.append(C.stage1_u8(d_pm, a_pm, sd, pos))
                ys_out.append(ys[i])

            x_t = torch.from_numpy(np.stack(xs))
            y_t = torch.from_numpy(np.stack(ys_out))
            if self.for_refiner:
                yield x_t, y_t, torch.tensor(flags, dtype=torch.uint8), pid
            else:
                yield x_t, y_t


def forever(loader):
    while True:
        for batch in loader:
            yield batch


class ThreadedPatchStream:
    """Thread-pool prefetcher for OnTheFlyPatchStream.
    Avoids PyTorch DataLoader subprocess workers (which call torch.manual_seed -> torch_tpu -> libtpu lock crash!),
    starts instantaneously, has zero IPC copy overhead, and runs astcenc/ispc subprocesses across all CPU cores."""
    def __init__(self, groups, P, batch, thorough_frac=1 / 3, pool_size=4, sparse_prob=0.10, for_refiner=False, num_workers=12, straight_prob=0.20):
        import queue, threading
        self.ds = OnTheFlyPatchStream(groups, P, batch, thorough_frac, pool_size, sparse_prob, for_refiner, straight_prob=straight_prob)
        self.q = queue.Queue(maxsize=max(16, int(num_workers) * 2))
        self.err = None
        self.stop = threading.Event()
        self.threads = []
        for wid in range(max(1, int(num_workers))):
            t = threading.Thread(target=self._run, args=(wid,), daemon=True)
            t.start()
            self.threads.append(t)

    def _run(self, wid):
        import queue
        ds = self.ds
        rng = random.Random(((os.getpid() * 1315423911) ^ (wid * 2654435761) ^ id(self)) & 0xFFFFFFFF)
        P = ds.P
        pos = C.pos_u8(P, 0, 0, P, P)
        pools = [[] for _ in ds.groups]

        def _load_img(gi):
            fs = ds.groups[gi][0]
            for _ in range(10):
                fp = rng.choice(fs)
                try:
                    im = np.asarray(Image.open(fp).convert('RGBA'))
                    st = C.is_straight(im, 0.02)
                    return (im, st, 0)
                except Exception:
                    continue
            raise RuntimeError('Failed to load any clean PNG')

        try:
            for gi in range(len(ds.groups)):
                while len(pools[gi]) < min(ds.pool_size, len(ds.groups[gi][0])):
                    pools[gi].append(_load_img(gi))

            while not self.stop.is_set():
                pid = 1 if rng.random() < ds.thorough_frac else 0
                preset = C.PRESETS[pid]
                stored_patches, ys, flags = [], [], []
                for _ in range(ds.batch):
                    r = rng.random()
                    acc = 0.0
                    gi = len(ds.groups) - 1
                    for idx, w in enumerate(ds.weights):
                        acc += w
                        if r <= acc:
                            gi = idx
                            break
                    pi = rng.randrange(len(pools[gi]))
                    im, st, uses = pools[gi][pi]
                    patch = _crop_patch(im, P, rng, ds.sparse_prob)
                    if st and rng.random() >= ds.straight_prob:
                        patch = C.premul(patch)
                        keep_st = False
                    else:
                        keep_st = st
                    stored_patches.append(patch)
                    ys.append(C.premul(patch) if keep_st else patch)
                    flags.append(1 if keep_st else 0)
                    uses += 1
                    if uses >= 48:
                        pools[gi][pi] = _load_img(gi)
                    else:
                        pools[gi][pi] = (im, st, uses)

                d_list, a_list = C.game_encode_batch(stored_patches, preset, astc_threads=1, parallel=False)
                xs, ys_out = [], []
                for i in range(ds.batch):
                    d_st, a_st = d_list[i], a_list[i]
                    sd = C.side_u8(d_st)
                    d_pm, a_pm = (C.premul(d_st), C.premul(a_st)) if flags[i] else (d_st, a_st)
                    xs.append(C.stage1_u8(d_pm, a_pm, sd, pos))
                    ys_out.append(ys[i])

                x_t = torch.from_numpy(np.stack(xs))
                y_t = torch.from_numpy(np.stack(ys_out))
                item = (x_t, y_t, torch.tensor(flags, dtype=torch.uint8), pid) if ds.for_refiner else (x_t, y_t)
                while not self.stop.is_set():
                    try:
                        self.q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self.err = e

    def __iter__(self):
        return self

    def __next__(self):
        import queue
        while True:
            if self.err is not None:
                raise self.err
            try:
                return self.q.get(timeout=0.5)
            except queue.Empty:
                continue
