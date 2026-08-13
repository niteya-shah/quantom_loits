from torch import nn


_BACKENDS = {}


def register_backend(name, factory):
    _BACKENDS[name] = factory


def _load_torch(**kwargs):
    from pytorch.loits import TorchLOITS

    return TorchLOITS(**kwargs)


register_backend("torch", _load_torch)


def _load_cpp(**kwargs):
    from cpp.backend import CppLOITS

    return CppLOITS(**kwargs)


register_backend("cpp", _load_cpp)


def _load_openmp(**kwargs):
    from openmp.backend import OpenMPLOITS

    return OpenMPLOITS(**kwargs)


register_backend("openmp", _load_openmp)


class LOITS(nn.Module):
    def __init__(self, backend="torch", **kwargs):
        super().__init__()
        if backend not in _BACKENDS:
            available = ", ".join(sorted(_BACKENDS))
            raise ValueError(f"backend={backend!r} is not registered; available: {available}")
        self.backend = backend
        self.impl = _BACKENDS[backend](**kwargs)

    def forward(self, theory_outputs, n_events):
        return self.impl(theory_outputs, n_events)
