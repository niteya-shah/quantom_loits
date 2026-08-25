from contextlib import nullcontext

import torch
from torch import nn

from .kernels import _EPSILON, _backward_impl, _forward_impl, _philox_uniform


def availability(device: torch.device | str = "cpu") -> tuple[bool, str]:
    device = torch.device(device)
    if device.type == "cuda":
        return (True, "") if torch.cuda.is_available() else (
            False,
            "PyTorch CUDA/ROCm is not available",
        )
    if device.type == "xpu":
        available = hasattr(torch, "xpu") and torch.xpu.is_available()
        return (True, "") if available else (False, "PyTorch XPU is not available")
    return False, "Triton backend is GPU-only (cuda/ROCm or xpu)"


def _device_guard(device: torch.device | str):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.device(device)
    if device.type == "xpu":
        return torch.xpu.device(device)
    return nullcontext()


class _TritonLOITSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_bins,
        xsec_x,
        q_bins,
        xsec_q,
        weights,
        acceptance,
        n_events,
        seed,
        sequence,
        profile_regions,
    ):
        profile_regions = bool(profile_regions)
        with _device_guard(xsec_x.device):
            events, state = _forward_impl(
                x_bins,
                xsec_x,
                q_bins,
                xsec_q,
                weights,
                acceptance,
                int(n_events),
                int(seed),
                int(sequence),
                profile_regions,
            )
        ctx.profile_regions = profile_regions
        ctx.save_for_backward(
            x_bins,
            xsec_x,
            q_bins,
            xsec_q,
            acceptance,
            *state,
        )
        return events

    @staticmethod
    def backward(ctx, grad_events):
        (
            x_bins,
            xsec_x,
            q_bins,
            xsec_q,
            acceptance,
            *state,
        ) = ctx.saved_tensors
        with _device_guard(grad_events.device):
            grad_xsec_x, grad_xsec_q = _backward_impl(
                grad_events.contiguous(),
                x_bins,
                xsec_x,
                q_bins,
                xsec_q,
                acceptance,
                state,
                ctx.profile_regions,
            )
        return (
            None,
            grad_xsec_x,
            None,
            grad_xsec_q,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class TritonLOITS(nn.Module):
    def __init__(
        self,
        device: torch.device | str = "cuda",
        compile: bool = False,
        profile_regions: bool = False,
        epsilon: float = _EPSILON,
    ):
        super().__init__()
        self.device: torch.device = torch.device(device)
        ok, reason = availability(self.device)
        if not ok:
            raise RuntimeError(reason)
        if epsilon != _EPSILON:
            raise ValueError("The Triton backend currently uses the LOITS epsilon 1e-5")
        self.profile_regions: bool = bool(profile_regions)
        self.seed: int = torch.initial_seed()
        self.sequence: int = 0

    @staticmethod
    def _validate(theory_outputs: tuple[torch.Tensor, ...]) -> None:
        x_bins, xsec_x, q_bins, xsec_q, weights, acceptance = theory_outputs[:6]
        floating = (x_bins, xsec_x, q_bins, xsec_q, weights)
        if any(t.dtype != torch.float64 for t in floating):
            raise TypeError("Triton LOITS requires float64 theory tensors")
        if acceptance.dtype != torch.bool:
            raise TypeError("Triton LOITS requires boolean acceptance")
        if any(not t.is_contiguous() for t in theory_outputs[:6]):
            raise ValueError("Triton LOITS requires contiguous theory tensors")

    def forward(self, theory_outputs: tuple[torch.Tensor, ...], n_events: int) -> torch.Tensor:
        self._validate(theory_outputs)
        x_bins, xsec_x, q_bins, xsec_q, weights, acceptance = theory_outputs[:6]
        if xsec_x.device.type != self.device.type or (
            self.device.index is not None and xsec_x.device.index != self.device.index
        ):
            raise ValueError(
                f"Triton LOITS configured for {self.device}, got tensors on {xsec_x.device}"
            )
        sequence = self.sequence
        self.sequence += 1
        return _TritonLOITSFunction.apply(
            x_bins,
            xsec_x,
            q_bins,
            xsec_q,
            weights,
            acceptance,
            n_events,
            self.seed,
            sequence,
            self.profile_regions,
        )
