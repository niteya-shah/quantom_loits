import os

import pytest
import torch

pytest.importorskip("triton")

from pytorch.loits import TorchLOITSCore
from pytorch.theory import TorchProxyTheoryLite
from triton_backend.backend import TritonLOITS, _philox_uniform


def triton_device():
    requested = os.environ.get("QUANTOM_TRITON_TEST_DEVICE")
    if requested is not None:
        if requested == "cuda":
            if not torch.cuda.is_available():
                pytest.skip("PyTorch CUDA/ROCm device is not available")
            return torch.device("cuda")
        if requested == "xpu":
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                pytest.skip("PyTorch XPU device is not available")
            return torch.device("xpu")
        raise RuntimeError(
            f"unsupported QUANTOM_TRITON_TEST_DEVICE={requested!r}"
        )

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    pytest.skip("no Triton-capable GPU is available")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def make_theory(device, grid=3, k=5):
    return TorchProxyTheoryLite(
        {
            "n_points_x": grid,
            "n_points_y": grid,
            "n_cdf_points_x": k,
            "n_cdf_points_y": k,
            "average": False,
        },
        str(device),
    )


def make_params(device):
    return torch.tensor(
        [[0.67, 0.20, 0.23, 0.67, 0.50]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )


def torch_with_samples(outputs, n_events, u_x, u_q):
    x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = outputs[:6]
    core = TorchLOITSCore()
    grid0, grid1 = weights.shape[1:]
    mask = core.allocation(weights, n_events)
    rho_x = core.rho_x(x_bins, xsec_x)
    rho_q2 = core.rho_q2(q2_bins, xsec_q2)
    cdf_x = core.cdf_x(x_bins, rho_x, acceptance, True)
    cdf_q2 = core.cdf_q2(q2_bins, rho_q2, acceptance, False)
    cdf_x, bins_x = core.flatten_x(x_bins, cdf_x, grid0, grid1, True)
    cdf_q2, bins_q2 = core.flatten_q2(q2_bins, cdf_q2, grid0, grid1, False)
    x = core.interpolation_x(u_x, cdf_x, bins_x, mask)
    q2 = core.interpolation_q2(u_q, cdf_q2, bins_q2, mask)
    return core.stream_compaction(x, q2, mask)


def test_triton_philox4x32_10_known_answer():
    device = triton_device()
    samples = _philox_uniform(4, seed=0, stream=0, device=device)
    sync(device)

    words = [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    expected = torch.tensor(
        [(word >> 8) * (2.0**-24) for word in words],
        dtype=torch.float64,
        device=device,
    )
    assert torch.equal(samples, expected)


@pytest.mark.parametrize("k", [4, 5, 7, 10, 16, 32])
def test_triton_forward_and_reverse_vjp_match_torch_for_same_samples(k):
    device = triton_device()
    seed = 23
    n_events = 257
    torch.manual_seed(seed)

    theory = make_theory(device, k=k)
    params = make_params(device)
    outputs = theory(params)
    x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = outputs[:6]

    sampler = TritonLOITS(device=str(device))
    events = sampler(outputs, n_events)

    core = TorchLOITSCore()
    mask = core.allocation(weights, n_events)
    u_x = _philox_uniform(mask.numel(), seed, 0, device).reshape(mask.shape)
    u_q = _philox_uniform(mask.numel(), seed, 1, device).reshape(mask.shape)
    reference = torch_with_samples(outputs, n_events, u_x, u_q)

    sync(device)
    torch.testing.assert_close(events, reference, rtol=2e-10, atol=2e-11)

    torch.manual_seed(seed + 1)
    upstream = torch.randn_like(events)

    grad_triton_x, grad_triton_q = torch.autograd.grad(
        (events * upstream).sum(),
        (xsec_x, xsec_q2),
        retain_graph=True,
    )
    grad_ref_x, grad_ref_q = torch.autograd.grad(
        (reference * upstream).sum(),
        (xsec_x, xsec_q2),
        retain_graph=True,
    )
    sync(device)

    torch.testing.assert_close(
        grad_triton_x,
        grad_ref_x,
        rtol=2e-9,
        atol=2e-10,
    )
    torch.testing.assert_close(
        grad_triton_q,
        grad_ref_q,
        rtol=2e-9,
        atol=2e-10,
    )


def test_triton_public_autograd_path_is_float64():
    device = triton_device()
    torch.manual_seed(17)

    theory = make_theory(device)
    params = make_params(device)
    sampler = TritonLOITS(device=str(device))
    events = sampler(theory(params), 50)
    events.square().mean().backward()
    sync(device)

    assert events.dtype == torch.float64
    assert events.ndim == 2 and events.shape[1] == 2 and events.shape[0] > 0
    assert params.grad is not None
    assert params.grad.dtype == torch.float64
    assert torch.isfinite(params.grad).all()
