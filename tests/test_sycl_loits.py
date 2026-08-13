import os

import pytest
import torch

from cpp.backend import load_extension as load_cpp_extension
from loits import LOITS
from pytorch.gan import GANTrainer
from pytorch.loits import TorchLOITSCore
from pytorch.profiler import TrainingProfiler
from pytorch.theory import TorchProxyTheoryLite
from sycl.backend import configured_torch_device, is_built, load_extension


pytestmark = pytest.mark.skipif(not is_built(), reason="SYCL backend has not been built")


def sycl_device():
    requested = os.environ.get("QUANTOM_SYCL_TEST_DEVICE", configured_torch_device())
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("PyTorch CUDA/ROCm device is not available")
        return torch.device("cuda")
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            pytest.skip("PyTorch XPU device is not available")
        return torch.device("xpu")
    raise RuntimeError(f"unsupported QUANTOM_SYCL_TEST_DEVICE={requested!r}")




def manual_outputs(device):
    x_bins = torch.tensor(
        [[[0.10, 0.20, 0.30, 0.40, 0.50], [0.20, 0.30, 0.40, 0.50, 0.60]]],
        dtype=torch.float64,
        device=device,
    )
    q_bins = torch.tensor(
        [[[1.0, 1.5, 2.0, 2.5, 3.0], [2.0, 2.5, 3.0, 3.5, 4.0]]],
        dtype=torch.float64,
        device=device,
    )
    xsec_x = torch.tensor(
        [[
            [[1.0, 1.1, 1.2, 1.3, 1.4], [1.1, 1.2, 1.3, 1.4, 1.5]],
            [[1.2, 1.3, 1.4, 1.5, 1.6], [1.3, 1.4, 1.5, 1.6, 1.7]],
        ]],
        dtype=torch.float64,
        device=device,
    )
    xsec_q = torch.tensor(
        [[
            [[1.0, 1.2, 1.4, 1.6, 1.8], [1.1, 1.3, 1.5, 1.7, 1.9]],
            [[1.2, 1.4, 1.6, 1.8, 2.0], [1.3, 1.5, 1.7, 1.9, 2.1]],
        ]],
        dtype=torch.float64,
        device=device,
    )
    weights = torch.tensor(
        [[[0.20, 0.30], [0.40, 0.10]]], dtype=torch.float64, device=device
    )
    acceptance = torch.ones((1, 2, 2), dtype=torch.bool, device=device)
    return x_bins, xsec_x, q_bins, xsec_q, weights, acceptance


def make_theory(device, grid=3):
    return TorchProxyTheoryLite(
        {
            "n_points_x": grid,
            "n_points_y": grid,
            "n_cdf_points_x": 5,
            "n_cdf_points_y": 5,
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


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def native_state(outputs, *, seed=23, sequence=0, n_events=80):
    device = outputs[1].device
    sync(device)
    x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = outputs[:6]
    state = load_extension().forward(
        x_bins,
        xsec_x,
        q2_bins,
        xsec_q2,
        weights,
        acceptance,
        n_events,
        seed,
        sequence,
        False,
    )
    return state


def test_sycl_philox4x32_10_known_answer():
    device = sycl_device()
    outputs = make_theory(device)(make_params(device))
    state = native_state(outputs, seed=0, sequence=0)
    u_x = state[5].flatten()[:4].cpu()

    words = [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    expected = torch.tensor(
        [(word >> 8) * (2.0**-24) for word in words], dtype=torch.float32
    )
    assert u_x.numel() == 4
    assert torch.equal(u_x, expected)


def test_sycl_philox_streams_match_cpp_exactly():
    device = sycl_device()
    sycl_outputs = manual_outputs(device)
    cpu_outputs = tuple(t.cpu() for t in sycl_outputs)

    sycl_state = native_state(sycl_outputs, seed=23, sequence=7, n_events=80)
    cpp_state = load_cpp_extension().forward(
        *cpu_outputs,
        80,
        23,
        7,
        False,
    )

    assert torch.equal(sycl_state[5].cpu(), cpp_state[5])
    assert torch.equal(sycl_state[6].cpu(), cpp_state[6])


def test_sycl_forward_and_reverse_vjp_match_torch_for_same_samples():
    device = sycl_device()
    theory = make_theory(device)
    params = make_params(device)
    outputs = theory(params)
    x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = outputs[:6]

    state = native_state(outputs)
    events, norm_x, norm_q, cdf_x, cdf_q, u_x, u_q, interval_x, interval_q, packed, row_offsets = state
    reference = torch_with_samples(outputs, 80, u_x, u_q)
    torch.testing.assert_close(events, reference, rtol=2e-10, atol=2e-11)

    upstream = torch.randn_like(events)
    grad_ref_x, grad_ref_q = torch.autograd.grad(
        (reference * upstream).sum(),
        (xsec_x, xsec_q2),
        retain_graph=True,
    )
    sync(device)
    grad_sycl_x, grad_sycl_q = load_extension().backward(
        upstream.contiguous(),
        x_bins,
        xsec_x,
        q2_bins,
        xsec_q2,
        acceptance,
        norm_x,
        norm_q,
        cdf_x,
        cdf_q,
        u_x,
        u_q,
        interval_x,
        interval_q,
        packed,
        row_offsets,
        False,
    )
    torch.testing.assert_close(grad_sycl_x, grad_ref_x, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(grad_sycl_q, grad_ref_q, rtol=2e-9, atol=2e-10)


def test_sycl_public_autograd_path():
    device = sycl_device()
    torch.manual_seed(17)
    theory = make_theory(device)
    params = make_params(device)
    sampler = LOITS(backend="sycl", device=str(device))
    events = sampler(theory(params), 50)
    events.square().mean().backward()
    assert events.ndim == 2 and events.shape[1] == 2 and events.shape[0] > 0
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def test_sycl_region_profiler():
    device = sycl_device()
    trainer = GANTrainer(
        backend="sycl",
        device=str(device),
        n_events=30,
        grid_size=3,
        compile=False,
        profile_regions=True,
    )
    profiler = TrainingProfiler(str(device))
    prof = profiler.run(trainer, warmup=0, iterations=1)
    names = {event.name for event in prof.events()}

    assert "gan::training_iteration" in names
    assert "loits::autograd::forward" in names
    assert "loits::binding::forward" in names
    assert "loits::forward" in names
    assert "loits::forward::allocation" in names
    assert "loits::forward::random_x" in names
    assert "loits::forward::interpolation_x" in names
    assert "loits::forward::stream_compaction" in names
    assert "loits::autograd::backward" in names
    assert "loits::binding::backward" in names
    assert "loits::backward" in names
    assert "loits::backward::interpolation_x" in names
    assert "loits::backward::cdf_x" in names
    assert "loits::backward::rho_x" in names

    rows = profiler.rows(prof, "sycl", str(device), 30)
    forward = next(row for row in rows if row["region"] == "loits::forward")
    random_x = next(row for row in rows if row["region"] == "loits::forward::random_x")
    assert random_x["parent_event_id"] == forward["event_id"]
    assert random_x["parent_region"] == "loits::forward"


def test_sycl_gan_training_is_reproducible():
    device = sycl_device()
    first = GANTrainer(
        backend="sycl",
        device=str(device),
        n_events=30,
        grid_size=3,
        compile=False,
        seed=31,
    )
    second = GANTrainer(
        backend="sycl",
        device=str(device),
        n_events=30,
        grid_size=3,
        compile=False,
        seed=31,
    )

    start = first.params.detach().clone()
    for _ in range(2):
        d_first, g_first = first.step()
        d_second, g_second = second.step()
        torch.testing.assert_close(d_first, d_second, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(g_first, g_second, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(first.params, second.params, rtol=1e-11, atol=1e-11)
    assert not torch.equal(first.params, start)
