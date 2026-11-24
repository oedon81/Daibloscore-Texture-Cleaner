"""UnityFS AssetBundle loader for Daibloscore (unwraps any leading custom prefix before UnityFS)."""
import os, re, struct
import numpy as np
from PIL import Image


def is_bundle(path):
    """Return True if file looks like a UnityFS bundle (possibly prefixed)."""
    try:
        with open(path, 'rb') as f:
            head = f.read(256)
    except OSError:
        return False
    return (not head.startswith(b'\x89PNG\r\n\x1a\n')) and (b'UnityFS\x00' in head)


def unwrap_bundle(data):
    for m in re.finditer(rb'UnityFS\x00', data):
        o = m.start()
        p = o + 12
        try:
            p = data.index(b'\x00', p) + 1
            p = data.index(b'\x00', p) + 1
        except ValueError:
            continue
        if p + 8 > len(data):
            continue
        size = struct.unpack('>q', data[p:p + 8])[0]
        if o + size == len(data):
            return data[o:]
    return data


def load_bundle(path):
    import UnityPy
    UnityPy.config.FALLBACK_UNITY_VERSION = '2022.3.62f1'
    with open(path, 'rb') as f:
        return UnityPy.load(unwrap_bundle(f.read()))


def extract_textures(path):
    """Return dict {name: (format_name, rgba_uint8_topdown)} from a Unity bundle or single PNG."""
    if not is_bundle(path):
        im = np.array(Image.open(path).convert('RGBA'))
        stem = os.path.splitext(os.path.basename(path))[0]
        return {stem: ('PNG', im)}
    from UnityPy.enums import TextureFormat
    env = load_bundle(path)
    out = {}
    for o in env.objects:
        if o.type.name != 'Texture2D':
            continue
        t = o.read()
        fmt = TextureFormat(t.m_TextureFormat).name
        out[t.m_Name] = (fmt, np.asarray(t.image.convert('RGBA')))
    return out


def extract_raw_textures(path):
    """Return dict {name: (format_name, w, h, raw_bytes)} from a Unity bundle."""
    from UnityPy.enums import TextureFormat
    env = load_bundle(path)
    out = {}
    for o in env.objects:
        if o.type.name != 'Texture2D':
            continue
        t = o.read()
        try:
            d = t.get_image_data()
        except Exception:
            d = bytes(t.image_data)
        out[t.m_Name] = (TextureFormat(t.m_TextureFormat).name, t.m_Width, t.m_Height, bytes(d))
    return out


def extract_spine_sidecars(bundle_path, out_dir):
    """Extract .atlas and .json Spine TextAssets from a Unity bundle into out_dir."""
    env = load_bundle(bundle_path)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for o in env.objects:
        if o.type.name != 'TextAsset':
            continue
        t = o.read()
        s = t.m_Script
        raw = s.encode('utf-8', 'surrogateescape') if isinstance(s, str) else bytes(s)
        name = t.m_Name
        if name.endswith('.atlas'):
            fn = name
        elif raw.lstrip().startswith(b'{'):
            fn = name if name.endswith('.json') else name + '.json'
        else:
            continue
        dst = os.path.join(out_dir, fn)
        with open(dst, 'wb') as f:
            f.write(raw)
        written.append(dst)
    return written
