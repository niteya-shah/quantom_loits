from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.profiler import record_function
from torch.utils.cpp_extension import load


_EXTENSION = None


def load_extension(verbose=False):
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    root = Path(__file__).resolve().parent
    build = root / "build"
    build.mkdir(exist_ok=True)
    _EXTENSION = load(
        name="quantom_loits_cpp",
        sources=[str(root / "bindings.cpp"), str(root / "loits_core.cpp")],
        build_directory=str(build),
        extra_cflags=["-O3", "-march=native", "-DNDEBUG", "-fno-math-errno"],
        verbose=verbose,
    )
    return _EXTENSION


def _region(enabled, name):
    return record_function(name) if enabled else nullcontext()


class _CppLOITSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance, n_events, seed, sequence, profile_regions):
        profile_regions = bool(profile_regions)
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
            with _region(ctx.profile_regions, "loits::binding::backward"):
                grad_xsec_x, grad_xsec_q2 = load_extension().backward(
                    grad_events.contiguous(),
                    x_bins,
                    xsec_x,
                    q2_bins,
                    xsec_q2,
                    acceptance,
                    *saved,
                    ctx.profile_regions,
                )
            return None, grad_xsec_x, None, grad_xsec_q2, None, None, None, None, None, None


class CppLOITS(nn.Module):
    def __init__(self, device="cpu", compile=False, profile_regions=False, epsilon=1e-5):
        super().__init__()
        self.device = torch.device(device)
        if epsilon != 1e-5:
            raise ValueError("The C++ backend currently uses the LOITS epsilon 1e-5")
        self.profile_regions = profile_regions
        self.seed = torch.initial_seed()
        self.sequence = 0
        load_extension()

    def forward(self, theory_outputs, n_events):
        x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = theory_outputs[:6]
        sequence = self.sequence
        self.sequence += 1
        with _region(self.profile_regions, "loits::autograd::forward"):
            return _CppLOITSFunction.apply(
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
