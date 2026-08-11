import torch

from loits import LOITS
from pytorch.loits import TorchLOITSCore
from pytorch.theory import TorchProxyTheoryLite


def inputs(grid=3):
    cfg = {
        "n_points_x": grid,
        "n_points_y": grid,
        "n_cdf_points_x": 5,
        "n_cdf_points_y": 5,
        "average": False,
    }
    theory = TorchProxyTheoryLite(cfg, "cpu")
    params = torch.tensor(
        [[0.67, 0.20, 0.23, 0.67, 0.50]],
        dtype=torch.float64,
        requires_grad=True,
    )
    return theory(params), params


def test_uniform_backend_api():
    theory_outputs, _ = inputs()
    sampler = LOITS(backend="torch", device="cpu", compile=False)
    torch.manual_seed(1)
    events = sampler(theory_outputs, 20)
    assert events.ndim == 2
    assert events.shape[1] == 2
    assert events.shape[0] > 0


def test_core_is_single_graph_and_aot_backward_runs():
    theory_outputs, params = inputs()
    core = TorchLOITSCore()
    explanation = torch._dynamo.explain(core)(theory_outputs, 20)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert not explanation.break_reasons

    compiled = torch.compile(core, backend="aot_eager", fullgraph=True)
    torch.manual_seed(2)
    events = compiled(theory_outputs, 20)
    events.square().mean().backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def test_regions_compile_fullgraph():
    theory_outputs, _ = inputs()
    x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = theory_outputs[:6]
    core = TorchLOITSCore()
    grid0, grid1 = weights.shape[1:]
    mask = core.allocation(weights, 20)
    rho_x = core.rho_x(x_bins, xsec_x)
    rho_q2 = core.rho_q2(q2_bins, xsec_q2)
    cdf_x = core.cdf_x(x_bins, rho_x, acceptance, True)
    cdf_q2 = core.cdf_q2(q2_bins, rho_q2, acceptance, False)
    flat_x = core.flatten_x(x_bins, cdf_x, grid0, grid1, True)
    flat_q2 = core.flatten_q2(q2_bins, cdf_q2, grid0, grid1, False)
    u_x = core.random_x(mask, xsec_x)
    u_q2 = core.random_q2(mask, xsec_q2)

    cases = {
        "allocation": (weights, 20),
        "rho_x": (x_bins, xsec_x),
        "rho_q2": (q2_bins, xsec_q2),
        "cdf_x": (x_bins, rho_x, acceptance, True),
        "cdf_q2": (q2_bins, rho_q2, acceptance, False),
        "flatten_x": (x_bins, cdf_x, grid0, grid1, True),
        "flatten_q2": (q2_bins, cdf_q2, grid0, grid1, False),
        "random_x": (mask, xsec_x),
        "random_q2": (mask, xsec_q2),
        "interpolation_x": (u_x, *flat_x, mask),
        "interpolation_q2": (u_q2, *flat_q2, mask),
    }
    x = core.interpolation_x(u_x, *flat_x, mask)
    q2 = core.interpolation_q2(u_q2, *flat_q2, mask)
    cases["stream_compaction"] = (x, q2, mask)

    for name, args in cases.items():
        compiled = torch.compile(getattr(core, name), backend="eager", fullgraph=True)
        compiled(*args)
