import ctypes
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.profiler import record_function
from torch.utils.cpp_extension import load


_EXTENSION = None
_CORE_HANDLE = None
_LOADED_VARIANT = None
_DPCPP_HIP_BRIDGE_STREAMS = {}


def _root():
    return Path(__file__).resolve().parent


def _build_root():
    return _root() / "build"


def selected_variant():
    variant = os.environ.get("QUANTOM_SYCL_VARIANT")
    if not variant:
        raise RuntimeError("QUANTOM_SYCL_VARIANT must be set explicitly for the SYCL backend")
    return variant


def variant_build_dir(variant=None):
    return _build_root() / (variant or selected_variant())


def core_library_path(variant=None):
    return variant_build_dir(variant) / "libquantom_loits_sycl.so"


def _metadata_path(variant=None):
    return variant_build_dir(variant) / "variant.py"


def _complete_build(variant):
    return core_library_path(variant).is_file() and _metadata_path(variant).is_file()


def built_variants():
    root = _build_root()
    if not root.is_dir():
        return ()
    variants = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if _complete_build(path.name):
            variants.append(path.name)
    return tuple(sorted(variants))


def is_built(variant=None):
    return _complete_build(variant or selected_variant())


def _read_metadata(variant=None):
    namespace = {}
    exec(_metadata_path(variant).read_text(), namespace)
    return namespace["METADATA"]


def configured_toolchain():
    return _read_metadata()["toolchain"]


def configured_target():
    return _read_metadata()["target"]


def configured_torch_device_mode():
    return _read_metadata()["torch_device"]


def configured_torch_device():
    value = configured_torch_device_mode()
    if value == "auto":
        raise RuntimeError(
            "the selected generic SYCL build has no fixed Torch device; "
            "set QUANTOM_SYCL_TEST_DEVICE explicitly for tests"
        )
    return value


def _selection_error():
    variants = built_variants()
    suffix = f" Built variants: {', '.join(variants)}." if variants else " No built variants were found."
    return "QUANTOM_SYCL_VARIANT must be set explicitly for the SYCL backend." + suffix


def availability(device="cpu"):
    device = torch.device(device)
    try:
        variant = selected_variant()
    except RuntimeError as exc:
        if "QUANTOM_SYCL_VARIANT" not in os.environ:
            return False, _selection_error()
        return False, str(exc)

    if not is_built(variant):
        variants = built_variants()
        suffix = f" Built variants: {', '.join(variants)}." if variants else ""
        return False, f"selected SYCL variant {variant!r} is not built.{suffix}"

    configured = configured_torch_device_mode()
    if configured != "auto" and configured != device.type:
        return False, (
            f"selected SYCL variant {variant!r} expects torch device "
            f"{configured!r}, requested {device.type!r}"
        )

    if device.type == "cpu":
        return True, ""
    if device.type == "cuda":
        return (True, "") if torch.cuda.is_available() else (False, "PyTorch CUDA/ROCm is not available")
    if device.type == "xpu":
        available = hasattr(torch, "xpu") and torch.xpu.is_available()
        return (True, "") if available else (False, "PyTorch XPU is not available")
    return False, f"unsupported torch device type {device.type!r}"


def load_extension(verbose=False, allow_incomplete=False):
    global _EXTENSION, _CORE_HANDLE, _LOADED_VARIANT

    variant = selected_variant()
    if _EXTENSION is not None:
        if variant != _LOADED_VARIANT:
            raise RuntimeError(
                f"SYCL variant {_LOADED_VARIANT!r} is already loaded in this Python process; "
                f"cannot switch to {variant!r}. Start a new process to use another variant."
            )
        return _EXTENSION

    root = _root()
    build = variant_build_dir(variant)
    library = core_library_path(variant)
    if allow_incomplete:
        if not library.is_file():
            raise RuntimeError(
                f"selected SYCL variant {variant!r} has no compiled core library"
            )
    elif not is_built(variant):
        raise RuntimeError(
            f"selected SYCL variant {variant!r} is not built. "
            "Build it explicitly with sycl/build-acpp.sh or sycl/build-dpcpp.sh."
        )

    # Load this variant's SYCL runtime/core first so the regular Torch binding
    # can remain host C++ with no SYCL headers or device-specific code.
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
    _LOADED_VARIANT = variant
    return _EXTENSION


def _dpcpp_hip_bridge_stream(device_index):
    """Return a persistent PyTorch-owned non-default HIP stream."""
    stream = _DPCPP_HIP_BRIDGE_STREAMS.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device_index)
        _DPCPP_HIP_BRIDGE_STREAMS[device_index] = stream
    return stream


def _bind_dpcpp_hip_stream(extension, device):
    """Bind DPC++/HIP execution to a PyTorch-owned native HIP stream."""
    if configured_toolchain() != "dpcpp" or configured_target() != "hip":
        return

    device = torch.device(device)
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    stream = torch.cuda.current_stream(device_index)
    native_stream = int(stream.cuda_stream)

    # DPC++'s UR HIP adapter calls hipStreamGetFlags() while importing a
    # native queue. HIP rejects the legacy/default null stream handle (0), so
    # use a persistent non-default stream allocated and owned by PyTorch.
    #
    # LOITS is intentionally synchronous at this boundary: callers synchronize
    # PyTorch before entering the native backend, and the native forward/backward
    # paths wait for the SYCL queue before returning. Therefore a bridge stream
    # preserves ordering without changing the backend's execution semantics.
    if native_stream == 0:
        stream = _dpcpp_hip_bridge_stream(device_index)
        native_stream = int(stream.cuda_stream)

    extension.bind_torch_hip_stream(native_stream, int(device_index))


def _sync_torch(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def prepare_extension(device, *, synchronize=True, verbose=False):
    device = torch.device(device)
    if synchronize:
        _sync_torch(device)
    extension = load_extension(verbose=verbose)
    _bind_dpcpp_hip_stream(extension, device)
    return extension


def _region(enabled, name):
    return record_function(name) if enabled else nullcontext()


class _SYCLLOITSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance, n_events, seed, sequence, profile_regions):
        profile_regions = bool(profile_regions)
        extension = prepare_extension(xsec_x.device)
        with _region(profile_regions, "loits::binding::forward"):
            state = extension.forward(
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
            extension = prepare_extension(grad_events.device)
            with _region(ctx.profile_regions, "loits::binding::backward"):
                grad_xsec_x, grad_xsec_q2 = extension.backward(
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
        if epsilon != 1e-5:
            raise ValueError("The SYCL backend currently uses the LOITS epsilon 1e-5")
        self.profile_regions = profile_regions
        self.seed = torch.initial_seed()
        self.sequence = 0

    def forward(self, theory_outputs, n_events):
        x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = theory_outputs[:6]
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
