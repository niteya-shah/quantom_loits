import argparse

import torch

from cpp.backend import load_extension
from pytorch.loits import TorchLOITSCore
from pytorch.theory import TorchProxyTheoryLite


def make_theory(grid):
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


def report(name, reference, candidate, preview=5):
    reference = reference.detach()
    candidate = candidate.detach()
    diff = (reference - candidate).abs()
    floor = max(torch.finfo(reference.dtype).eps, 1e-15)
    rel = diff / reference.abs().clamp_min(floor)
    ref_norm = torch.linalg.vector_norm(reference)
    rel_l2 = torch.linalg.vector_norm(reference - candidate) / ref_norm.clamp_min(floor)

    print(f"\n{name}")
    print(f"  shape:       {tuple(reference.shape)}")
    print(f"  max abs:     {diff.max().item():.6e}")
    print(f"  mean abs:    {diff.mean().item():.6e}")
    print(f"  max rel:     {rel.max().item():.6e}")
    print(f"  relative L2: {rel_l2.item():.6e}")
    print(f"  torch[:{preview}]: {reference.flatten()[:preview].tolist()}")
    print(f"  cpp[:{preview}]:   {candidate.flatten()[:preview].tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100000)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=2e-12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    theory = make_theory(args.grid)
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
        args.events,
        args.seed,
        0,
        False,
    )

    events_cpp, norm_x, norm_q, cdf_x, cdf_q, u_x, u_q, interval_x, interval_q, packed = state
    events_torch = torch_with_samples(outputs, args.events, u_x, u_q)

    report("forward events", events_torch, events_cpp)

    torch.manual_seed(args.seed + 1)
    upstream = torch.randn_like(events_cpp)

    grad_torch_x, grad_torch_q = torch.autograd.grad(
        (events_torch * upstream).sum(),
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

    report("VJP xsec_x", grad_torch_x, grad_cpp_x)
    report("VJP xsec_q2", grad_torch_q, grad_cpp_q)

    param_grad_torch = torch.autograd.grad(
        (xsec_x * grad_torch_x.detach()).sum() + (xsec_q2 * grad_torch_q.detach()).sum(),
        params,
        retain_graph=True,
    )[0]
    param_grad_cpp = torch.autograd.grad(
        (xsec_x * grad_cpp_x.detach()).sum() + (xsec_q2 * grad_cpp_q.detach()).sum(),
        params,
    )[0]

    report("end-to-end theory parameter VJP", param_grad_torch, param_grad_cpp)

    checks = [
        ("forward", events_cpp, events_torch, 1e-12, 1e-12),
        ("xsec_x VJP", grad_cpp_x, grad_torch_x, args.rtol, args.atol),
        ("xsec_q2 VJP", grad_cpp_q, grad_torch_q, args.rtol, args.atol),
        ("parameter VJP", param_grad_cpp, param_grad_torch, args.rtol, args.atol),
    ]

    failed = []
    for name, candidate, reference, rtol, atol in checks:
        if not torch.allclose(candidate, reference, rtol=rtol, atol=atol):
            failed.append(name)

    if failed:
        print("\nFAIL:", ", ".join(failed))
        raise SystemExit(1)

    print("\nPASS: C++ forward and reverse VJP match PyTorch numerically.")


if __name__ == "__main__":
    main()
