import torch
from torch import nn


_BACKENDS = {}
_BACKEND_PROBES = {}


class BackendUnavailableError(RuntimeError):
    pass


def register_backend(name, factory, probe=None):
    _BACKENDS[name] = factory
    _BACKEND_PROBES[name] = probe


def registered_backends():
    return tuple(sorted(_BACKENDS))


def _device_status(device):
    device = torch.device(device)
    if device.type == "cpu":
        return True, ""
    if device.type == "cuda":
        return (True, "") if torch.cuda.is_available() else (False, "PyTorch CUDA/ROCm is not available")
    if device.type == "xpu":
        available = hasattr(torch, "xpu") and torch.xpu.is_available()
        return (True, "") if available else (False, "PyTorch XPU is not available")
    return False, f"unsupported torch device type {device.type!r}"


def backend_status(name, device="cpu"):
    if name not in _BACKENDS:
        return False, f"backend {name!r} is not registered"
    probe = _BACKEND_PROBES.get(name)
    if probe is None:
        return True, ""
    try:
        result = probe(torch.device(device))
    except Exception as exc:
        return False, str(exc)
    if isinstance(result, tuple):
        return bool(result[0]), str(result[1])
    return bool(result), "" if result else "backend probe returned unavailable"


def available_backends(device="cpu"):
    return tuple(name for name in registered_backends() if backend_status(name, device)[0])


def _load_torch(**kwargs):
    from pytorch.loits import TorchLOITS

    return TorchLOITS(**kwargs)


def _probe_torch(device):
    return _device_status(device)


register_backend("torch", _load_torch, _probe_torch)


def _load_cpp(**kwargs):
    from cpp.backend import CppLOITS

    return CppLOITS(**kwargs)


def _probe_cpp(device):
    if device.type != "cpu":
        return False, "C++ backend is CPU-only"
    return True, ""


register_backend("cpp", _load_cpp, _probe_cpp)


def _load_openmp(**kwargs):
    from openmp.backend import OpenMPLOITS

    return OpenMPLOITS(**kwargs)


def _probe_openmp(device):
    if device.type != "cpu":
        return False, "OpenMP backend is CPU-only"
    return True, ""


register_backend("openmp", _load_openmp, _probe_openmp)


def _load_sycl(**kwargs):
    from sycl.backend import SYCLLOITS

    return SYCLLOITS(**kwargs)


def _probe_sycl(device):
    from sycl.backend import availability

    return availability(device)


register_backend("sycl", _load_sycl, _probe_sycl)


def _load_triton(**kwargs):
    from triton_backend.backend import TritonLOITS

    return TritonLOITS(**kwargs)


def _probe_triton(device):
    try:
        from triton_backend.backend import availability
    except ImportError as exc:
        return False, f"Triton is not available: {exc}"

    return availability(device)


register_backend("triton", _load_triton, _probe_triton)


class LOITS(nn.Module):
    def __init__(self, backend="torch", **kwargs):
        super().__init__()
        if backend not in _BACKENDS:
            available = ", ".join(registered_backends())
            raise ValueError(f"backend={backend!r} is not registered; registered: {available}")

        device = kwargs.get("device", "cpu")
        ok, reason = backend_status(backend, device)
        if not ok:
            raise BackendUnavailableError(f"backend={backend!r} is unavailable for device={device!s}: {reason}")

        self.backend = backend
        self.impl = _BACKENDS[backend](**kwargs)

    def forward(self, theory_outputs, n_events):
        return self.impl(theory_outputs, n_events)
