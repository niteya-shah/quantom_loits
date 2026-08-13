import ctypes
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.profiler import record_function
from torch.utils.cpp_extension import load


_EXTENSION = None
_CORE_HANDLE = None


def _root():
    return Path(__file__).resolve().parent


def core_library_path():
    return _root() / "build" / "libquantom_loits_sycl.so"


def is_built():
    return core_library_path().is_file()


def configured_torch_device():
    marker = _root() / "build" / "torch_device.txt"
    if marker.is_file():
        value = marker.read_text().strip()
        if value and value != "auto":
            return value
    return "cpu"


def load_extension(verbose=False):
    global _EXTENSION, _CORE_HANDLE
    if _EXTENSION is not None:
        return _EXTENSION

    root = _root()
    build = root / "build"
    library = core_library_path()
    if not library.is_file():
        raise RuntimeError(
            "SYCL core is not built. Run sycl/build-acpp.sh <target> or "
            "sycl/build-dpcpp.sh <target> first."
        )

    # Load the selected SYCL runtime/core first so the regular Torch binding can
    # remain a host C++ extension with no SYCL headers or device-specific code.
    _CORE_HANDLE = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    _EXTENSION = load(
        name="quantom_loits_sycl_binding",
        sources=[str(root / "bindings.cpp")],
        build_directory=str(build),
        extra_include_paths=[str(root)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_ldflags=[
            f"-L{build}",
            "-lquantom_loits_sycl",
            f"-Wl,-rpath,{build}",
        ],
        verbose=verbose,
    )
    return _EXTENSION


def _region(enabled, name):
    return record_function(name) if enabled else nullcontext()


def _sync_torch(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


class _SYCLLOITSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance, n_events, seed, sequence, profile_regions):
        profile_regions = bool(profile_regions)
        _sync_torch(xsec_x.device)
        with _region(profile_regions, "loits::binding::forward"):
            state = load_extension().forward(
                x_bins,
                xsec_x,
                q2_bins,
                xsec_q2,
                weights,
                acceptance,
                int(n_events),
                int(seed),
                int(sequence),
                profile_regions,
            )
        events, *saved = state
        ctx.profile_regions = profile_regions
        ctx.save_for_backward(x_bins, xsec_x, q2_bins, xsec_q2, acceptance, *saved)
        return events

    @staticmethod
    def backward(ctx, grad_events):
        with _region(ctx.profile_regions, "loits::autograd::backward"):
            x_bins, xsec_x, q2_bins, xsec_q2, acceptance, *saved = ctx.saved_tensors
            grad_events = grad_events.contiguous()
            _sync_torch(grad_events.device)
            with _region(ctx.profile_regions, "loits::binding::backward"):
                grad_xsec_x, grad_xsec_q2 = load_extension().backward(
                    grad_events,
                    x_bins,
                    xsec_x,
                    q2_bins,
                    xsec_q2,
                    acceptance,
                    *saved,
                    ctx.profile_regions,
                )
            return None, grad_xsec_x, None, grad_xsec_q2, None, None, None, None, None, None


class SYCLLOITS(nn.Module):
    def __init__(self, device="cpu", compile=False, profile_regions=False, epsilon=1e-5):
        super().__init__()
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda", "xpu"}:
            raise ValueError("The SYCL backend expects a torch cpu, cuda/ROCm, or xpu device")
        if epsilon != 1e-5:
            raise ValueError("The SYCL backend currently uses the LOITS epsilon 1e-5")
        self.profile_regions = profile_regions
        self.seed = torch.initial_seed()
        self.sequence = 0
        extension = load_extension()
        if not extension.supports_fp64():
            raise RuntimeError(f"selected SYCL device does not support float64: {extension.device_name()}")

    def forward(self, theory_outputs, n_events):
        x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = theory_outputs[:6]
        if xsec_x.device != self.device:
            raise ValueError(f"SYCL backend configured for {self.device}, got tensors on {xsec_x.device}")
        sequence = self.sequence
        self.sequence += 1
        with _region(self.profile_regions, "loits::autograd::forward"):
            return _SYCLLOITSFunction.apply(
                x_bins,
                xsec_x,
                q2_bins,
                xsec_q2,
                weights,
                acceptance,
                n_events,
                self.seed,
                sequence,
                self.profile_regions,
            )
