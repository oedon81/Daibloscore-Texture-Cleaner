"""Unified CUDA / native TorchTPU (torch.device('tpu')) / legacy torch_xla / CPU helper."""
import torch

_AC_CACHE = {}


def autocast_ok(ac_dev, dtype):
    key = (ac_dev, dtype)
    if key in _AC_CACHE:
        return _AC_CACHE[key]
    ok = False
    if ac_dev in ('cuda', 'tpu', 'xla'):
        try:
            dev = torch.device(ac_dev) if ac_dev != 'xla' else get_device()[0]
            with torch.autocast(ac_dev, dtype=dtype):
                x = torch.ones((1, 4, 4, 4), device=dev)
                _ = (x + 1.0).sum()
            ok = True
        except Exception:
            ok = False
    _AC_CACHE[key] = ok
    return ok


def _has_native_tpu():
    try:
        import torch_tpu  # noqa: F401
    except Exception:
        pass
    try:
        if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
            acc = torch.accelerator.current_accelerator()
            if acc is not None and acc.type == 'tpu':
                return True
    except Exception:
        pass
    try:
        t = torch.empty(1, device='tpu')
        return t.device.type == 'tpu'
    except Exception:
        return False


def get_device():
    """Return (dev, backend, amp_dtype, ac_dev) where backend is 'cuda' | 'tpu' | 'xla' | 'cpu'."""
    if torch.cuda.is_available():
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device('cuda'), 'cuda', dt, 'cuda'
    if _has_native_tpu():
        return torch.device('tpu'), 'tpu', torch.bfloat16, 'tpu'
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        dev = torch_xla.device() if hasattr(torch_xla, 'device') else xm.xla_device()
        return dev, 'xla', torch.bfloat16, 'xla'
    except Exception:
        return torch.device('cpu'), 'cpu', torch.float32, 'cpu'


def is_xla_dev(dev):
    return str(dev).startswith('xla')


def is_tpu_dev(dev):
    return str(dev).startswith('tpu')


def is_tpu_or_xla(dev):
    s = str(dev)
    return s.startswith('tpu') or s.startswith('xla')


def mark_step(backend):
    if backend == 'xla' or backend is True:
        try:
            import torch_xla
            if hasattr(torch_xla, 'sync'):
                torch_xla.sync()
                return
        except Exception:
            pass
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def sync(backend):
    if backend == 'cuda':
        torch.cuda.synchronize()
    elif backend == 'tpu':
        if hasattr(torch, 'accelerator') and hasattr(torch.accelerator, 'synchronize'):
            torch.accelerator.synchronize()
    elif backend == 'xla':
        mark_step('xla')


def step_optimizer(opt, backend):
    if backend == 'xla' or backend is True:
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(opt, barrier=False)
    else:
        opt.step()


def device_name(dev, backend):
    if backend == 'cuda':
        return torch.cuda.get_device_name(0)
    if backend == 'tpu':
        return f'TorchTPU ({dev})'
    if backend == 'xla':
        return f'TPU/XLA ({dev})'
    return 'CPU'
