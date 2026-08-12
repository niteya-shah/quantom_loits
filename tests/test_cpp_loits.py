import torch

from cpp.backend import load_extension
from loits import LOITS
from pytorch.gan import GANTrainer
from pytorch.loits import TorchLOITSCore
from pytorch.profiler import TrainingProfiler
from pytorch.theory import TorchProxyTheoryLite


def make_theory(grid=3):
    return TorchProxyTheoryLite(
        {
            "n_points_x": grid,
            "n_points_y": grid,
            "n_cdf_points_x": 5,
            "n_cdf_points_y": 5,
            "average": False,
        },
        "cpu",
    )


def make_params():
    return torch.tensor(
        [[0.67, 0.20, 0.23, 0.67, 0.50]],
        dtype=torch.float64,
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


def test_cpp_forward_and_reverse_vjp_match_torch_for_same_samples():
    theory = make_theory()
    params = make_params()
    outputs = theory(params)
    x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = outputs[:6]

    state = load_extension().forward(
        x_bins,
        xsec_x,
        q2_bins,
        xsec_q2,
        weights,
        acceptance,
        80,
        23,
        0,
        False,
    )
    events, norm_x, norm_q, cdf_x, cdf_q, u_x, u_q, interval_x, interval_q, packed = state
    reference = torch_with_samples(outputs, 80, u_x, u_q)
    torch.testing.assert_close(events, reference, rtol=1e-12, atol=1e-12)

    upstream = torch.randn_like(events)
    grad_ref_x, grad_ref_q = torch.autograd.grad(
        (reference * upstream).sum(),
        (xsec_x, xsec_q2),
        retain_graph=True,
    )
    grad_cpp_x, grad_cpp_q = load_extension().backward(
        upstream,
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
        False,
    )
    torch.testing.assert_close(grad_cpp_x, grad_ref_x, rtol=1e-10, atol=2e-12)
    torch.testing.assert_close(grad_cpp_q, grad_ref_q, rtol=1e-10, atol=2e-12)


def test_cpp_public_autograd_path():
    torch.manual_seed(17)
    theory = make_theory()
    params = make_params()
    sampler = LOITS(backend="cpp", device="cpu")
    events = sampler(theory(params), 50)
    events.square().mean().backward()
    assert events.ndim == 2 and events.shape[1] == 2 and events.shape[0] > 0
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def test_cpp_gan_training_and_region_profiler():
    trainer = GANTrainer(
        backend="cpp",
        device="cpu",
        n_events=30,
        grid_size=3,
        compile=False,
        profile_regions=True,
    )
    profiler = TrainingProfiler("cpu")
    prof = profiler.run(trainer, warmup=0, iterations=1)
    names = {event.name for event in prof.events()}

    assert "gan::training_iteration" in names
    assert "loits::autograd::forward" in names
    assert "loits::binding::forward" in names
    assert "loits::forward" in names
    assert "loits::forward::validation" in names
    assert "loits::forward::allocation" in names
    assert "loits::forward::stream_compaction" in names
    assert "loits::forward::state_pack" in names
    assert "loits::autograd::backward" in names
    assert "loits::binding::backward" in names
    assert "loits::backward" in names
    assert "loits::backward::validation" in names
    assert "loits::backward::interpolation_x" in names
    assert "loits::backward::cdf_x" in names
    assert "loits::backward::rho_x" in names
    assert "loits::backward::state_pack" in names

    rows = profiler.rows(prof, "cpp", "cpu", 30)
    forward = next(row for row in rows if row["region"] == "loits::forward")
    rho_x = next(row for row in rows if row["region"] == "loits::forward::rho_x")
    assert rho_x["parent_event_id"] == forward["event_id"]
    assert rho_x["parent_region"] == "loits::forward"
    assert rho_x["start_us"] >= forward["start_us"]
    assert rho_x["end_us"] <= forward["end_us"]


def test_cpp_gan_training_is_reproducible():
    first = GANTrainer(
        backend="cpp",
        device="cpu",
        n_events=30,
        grid_size=3,
        compile=False,
        seed=31,
    )
    second = GANTrainer(
        backend="cpp",
        device="cpu",
        n_events=30,
        grid_size=3,
        compile=False,
        seed=31,
    )

    start = first.params.detach().clone()
    for _ in range(2):
        d_first, g_first = first.step()
        d_second, g_second = second.step()
        torch.testing.assert_close(d_first, d_second, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(g_first, g_second, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(first.params, second.params, rtol=1e-12, atol=1e-12)
    assert not torch.equal(first.params, start)
