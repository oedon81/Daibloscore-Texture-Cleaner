"""Daibloscore Texture Cleaner models:
  - Stage 1 (Stage1Net, 15-ch input -> 4-ch premul RGBA): norm-free PixelUnshuffle(2) Residual CNN with
    Local Channel Attention (LocalCALayer, 25x25 pool = 50x50 px -> spatially adaptive DXT5 vs ASTC gating
    with zero tile-size or sparse-background shift) and ASTC base + differentiable alpha-coupled residual.
    Default size: c=384, n=16 (~43.0M params, 138x138 receptive field).
  - Stage 2 Refiner (28-ch with split RGB/Alpha block mismatch for DXT5 & ASTC, or 26-ch legacy):
    Default size: c=256, n=10 (~12.2M params) with LocalCALayer.
Backward-compatible load(): auto-detects channels, depth, cin (15/26/28), LocalCA, couple_edge, and legacy rim_gate."""
import os, torch, torch.nn as nn, torch.nn.functional as F
from .codec import NSIDE

CIN1 = 8 + NSIDE
CIN2_LEGACY = 26
CIN2_V21 = 28


def _valid(y):
    a = y[:, 3:4].clamp(0, 1)
    return torch.cat([torch.minimum(y[:, :3].clamp(0, 1), a), a], 1)


class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        self.r = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.b(self.r(self.a(x)))


class _SeparableAvgPool2d(nn.Module):
    """Mathematically identical to AvgPool2d((k,k), padding=k//2, count_include_pad=False),
    implemented via separable count_include_pad=True pools plus exact analytic border scaling.
    Avoids a TorchTPU/libtpu v5e backward-compile hang on count_include_pad=False while adding
    zero parameters/buffers and preserving exact numerical parity (~4e-8 fp32 diff)."""
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.pad = k // 2
        self.px = nn.AvgPool2d((1, k), stride=1, padding=(0, self.pad), count_include_pad=True)
        self.py = nn.AvgPool2d((k, 1), stride=1, padding=(self.pad, 0), count_include_pad=True)
        self._scale_cache = {}

    def _scales(self, h, w, device, dtype):
        compiling = hasattr(torch, 'compiler') and torch.compiler.is_compiling()
        key = (h, w, device, dtype)
        if not compiling:
            hit = self._scale_cache.get(key)
            if hit is not None:
                return hit
        k, p = self.k, self.pad
        iw = torch.arange(w, device=device, dtype=torch.float32)
        dx = (float(k) / (torch.clamp(iw + (p + 1), max=w) - torch.clamp(iw - p, min=0))).to(dtype).view(1, 1, 1, w)
        ih = torch.arange(h, device=device, dtype=torch.float32)
        dy = (float(k) / (torch.clamp(ih + (p + 1), max=h) - torch.clamp(ih - p, min=0))).to(dtype).view(1, 1, h, 1)
        if not compiling:
            if len(self._scale_cache) > 16:
                self._scale_cache.clear()
            self._scale_cache[key] = (dx, dy)
        return dx, dy

    def forward(self, x):
        dx, dy = self._scales(x.shape[-2], x.shape[-1], x.device, x.dtype)
        return self.py(self.px(x) * dx) * dy


class LocalCALayer(nn.Module):
    """Norm-free local channel attention (25x25 pool at H/2 = 50x50 px).
    Unlike global AdaptiveAvgPool2d(1), this has zero train/test sparsity shift on sparse Spine atlases
    and lets hair strands (DXT5-friendly) and skin/rims (ASTC-friendly) receive different channel gates."""
    def __init__(self, c, r=16, k=25):
        super().__init__()
        mid = max(16, c // r)
        self.pool = _SeparableAvgPool2d(k)
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)

    def forward(self, x):
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(self.pool(x)), inplace=True)))
        return x * g


class RCAB(nn.Module):
    """Norm-free Residual Channel Attention Block with 0.1 residual scaling for deep 16-block stability."""
    def __init__(self, c, res_scale=0.1):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        self.r = nn.ReLU(inplace=True)
        self.ca = LocalCALayer(c)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.ca(self.b(self.r(self.a(x))))


class Stage1Net(nn.Module):
    def __init__(self, cin=CIN1, c=384, n=16, base=1, couple_edge=True, use_ca=True, rim_gate=False):
        super().__init__()
        self.cin, self.base, self.couple_edge, self.use_ca, self.rim_gate = cin, base, couple_edge, use_ca, rim_gate
        self.down = nn.PixelUnshuffle(2)
        self.head = nn.Conv2d(cin * 4, c, 3, padding=1)
        blk = RCAB if use_ca else RB
        self.body = nn.Sequential(*[blk(c) for _ in range(n)])
        self.tail = nn.Conv2d(c, 16, 3, padding=1)
        self.up = nn.PixelShuffle(2)
        # Non-persistent buffer so smooth rim_gate annealing (1.0 -> 0.0) never triggers TPU recompiles or changes state_dict keys.
        self.register_buffer('_rim_alpha', torch.tensor(1.0 if rim_gate else 0.0, dtype=torch.float32), persistent=False)

    def forward(self, x):
        base = x[:, 4 * self.base:4 * self.base + 4]
        d = self.up(self.tail(self.body(self.head(self.down(x)))))
        # Keep couple_edge + residual skip in fp32 so bf16's 1-gray-level ULP at [0.5, 1.0]
        # never rounds away sub-level residuals or creates a seam floor.
        with torch.amp.autocast(x.device.type, enabled=False):
            base = base.float()
            d = d.float()
            if self.couple_edge and self.base == 1:
                base_a = base[:, 3:4]
                base_st = (base[:, :3] / base_a.clamp(min=1 / 255)).clamp(0, 1)
                if self.rim_gate:
                    soft_zone = F.max_pool2d(((base_a > 0) & (base_a < 252 / 255)).float(), kernel_size=7, stride=1, padding=3)
                    st_min = base_st.min(dim=1, keepdim=True).values
                    st_min_valid = torch.where(base_a > 8 / 255, st_min, torch.ones_like(st_min))
                    local_min_st = -F.max_pool2d(-st_min_valid, kernel_size=5, stride=1, padding=2)
                    is_bright_rim = ((local_min_st - 0.52) / 0.12).clamp(0, 1) * soft_zone
                    rim_w = F.avg_pool2d(is_bright_rim, kernel_size=5, stride=1, padding=2)
                    d = d * (1.0 - self._rim_alpha * rim_w)
                a = (base_a + d[:, 3:4]).clamp(0, 1)
                y_rgb = base[:, :3] + base_st * (a - base_a) + d[:, :3]
            else:
                a = (base[:, 3:4] + d[:, 3:4]).clamp(0, 1)
                y_rgb = base[:, :3] + d[:, :3]
            rgb = torch.minimum(y_rgb.clamp(0, 1), a)
            return torch.cat([rgb, a], 1)


def stage1(c=384, n=16, zero_tail=True, use_ca=True, couple_edge=True, rim_gate=False):
    net = Stage1Net(CIN1, c, n, base=1, couple_edge=couple_edge, use_ca=use_ca, rim_gate=rim_gate)
    if zero_tail:
        nn.init.zeros_(net.tail.weight)
        nn.init.zeros_(net.tail.bias)
    return net


class Refiner(nn.Module):
    def __init__(self, c=256, n=10, cin=CIN2_V21, use_ca=True):
        super().__init__()
        self.cin, self.use_ca = cin, use_ca
        self.down = nn.PixelUnshuffle(2)
        self.head = nn.Conv2d(cin * 4, c, 3, padding=1)
        blk = RCAB if use_ca else RB
        self.body = nn.Sequential(*[blk(c) for _ in range(n)])
        self.tail = nn.Conv2d(c, 16, 3, padding=1)
        self.up = nn.PixelShuffle(2)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        d = self.up(self.tail(self.body(self.head(self.down(x)))))
        with torch.amp.autocast(x.device.type, enabled=False):
            return _valid(x[:, 8:12].float() + d.float())


def save(net, path, **meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'net': {k: v.detach().cpu() for k, v in net.state_dict().items()}, **meta}, path + '.part')
    os.replace(path + '.part', path)


def load(path, dev='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['net']
    c = sd['head.weight'].shape[0]
    cin = sd['head.weight'].shape[1] // 4
    n = len({k.split('.')[1] for k in sd if k.startswith('body.')})
    use_ca = any('.ca.' in k for k in sd)
    if ck.get('kind') == 'stage1' or cin == CIN1:
        couple_edge = ck.get('couple_edge', True)
        # Legacy checkpoints written before rim_gate removal did not store 'rim_gate'; default True only for them.
        rim_gate = ck.get('rim_gate', True)
        net = stage1(c, n, zero_tail=False, use_ca=use_ca, couple_edge=couple_edge, rim_gate=rim_gate)
    else:
        net = Refiner(c, n, cin=cin, use_ca=use_ca)
    net.load_state_dict(sd)
    return net.to(dev).eval(), ck
