import argparse
import importlib

import torch

from loits import backend_status
from pytorch.loits import TorchLOITSCore
from pytorch.theory import TorchProxyTheoryLite


NATIVE_BACKENDS = ("cpp", "openmp", "sycl")

# These match the tolerances used by the backend correctness tests.  The
# script still prints the raw error metrics; these values only determine the
# final PASS/FAIL line.
TOLERANCES = {
    "cpp": {
        "forward": (1e-12, 1e-12),
        "backward": (1e-10, 2e-12),
    },
    "openmp": {
        "forward": (1e-12, 1e-12),
        "backward": (1e-10, 2e-12),
    },
    "sycl": {
        "forward": (2e-10, 2e-11),
        "backward": (2e-9, 2e-10),
    },
}


def make_theory(grid, device):
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
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def load_native_extension(backend):
    module = importlib.import_module(f"{backend}.backend")
    try:
        return module, module.load_extension()
    except AttributeError as exc:
        raise RuntimeError(f"backend {backend!r} does not expose load_extension()") from exc


def bind_native_runtime(module, extension, device):
    bind = getattr(module, "bind_torch_stream", None)
    if bind is not None:
        bind(extension, device)


def report(name, reference, candidate, backend, preview=5):
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
    print(f"  torch[:{preview}]:   {reference.flatten()[:preview].tolist()}")
    print(f"  {backend}[:{preview}]: {candidate.flatten()[:preview].tolist()}")


def resolve_device(parser, backend, requested):
    if requested is None:
        if backend == "sycl":
            parser.error("--device is required explicitly for --backend sycl")
        requested = "cpu"
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(
        description="Compare a native LOITS forward/reverse VJP against PyTorch using identical samples."
    )
    parser.add_argument("--backend", required=True, choices=NATIVE_BACKENDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--events", type=int, default=100000)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    device = resolve_device(parser, args.backend, args.device)
    ok, reason = backend_status(args.backend, device)
    if not ok:
        parser.error(f"backend {args.backend!r} is unavailable for {device}: {reason}")

    torch.manual_seed(args.seed)
    theory = make_theory(args.grid, device)
    params = make_params(device)
    outputs = theory(params)
    x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = outputs[:6]

    backend_module, extension = load_native_extension(args.backend)
    sync(device)
    bind_native_runtime(backend_module, extension, device)
    state = extension.forward(
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

    # All native backends return events first, followed by exactly the saved
    # state their own backward() expects.  OpenMP/SYCL currently carry one
    # extra row_offsets tensor compared with serial C++; keeping this generic
    # avoids encoding that implementation detail here.
    events_native, *saved = state
    u_x, u_q = saved[4], saved[5]
    events_torch = torch_with_samples(outputs, args.events, u_x, u_q)

    report("forward events", events_torch, events_native, args.backend)

    torch.manual_seed(args.seed + 1)
    upstream = torch.randn_like(events_native)

    grad_torch_x, grad_torch_q = torch.autograd.grad(
        (events_torch * upstream).sum(),
        (xsec_x, xsec_q2),
        retain_graph=True,
    )

    sync(device)
    bind_native_runtime(backend_module, extension, device)
    grad_native_x, grad_native_q = extension.backward(
        upstream.contiguous(),
        x_bins,
        xsec_x,
        q2_bins,
        xsec_q2,
        acceptance,
        *saved,
        False,
    )
    sync(device)

    report("VJP xsec_x", grad_torch_x, grad_native_x, args.backend)
    report("VJP xsec_q2", grad_torch_q, grad_native_q, args.backend)

    param_grad_torch = torch.autograd.grad(
        (xsec_x * grad_torch_x.detach()).sum()
        + (xsec_q2 * grad_torch_q.detach()).sum(),
        params,
        retain_graph=True,
    )[0]
    param_grad_native = torch.autograd.grad(
        (xsec_x * grad_native_x.detach()).sum()
        + (xsec_q2 * grad_native_q.detach()).sum(),
        params,
    )[0]

    report(
        "end-to-end theory parameter VJP",
        param_grad_torch,
        param_grad_native,
        args.backend,
    )

    forward_rtol, forward_atol = TOLERANCES[args.backend]["forward"]
    backward_rtol, backward_atol = TOLERANCES[args.backend]["backward"]
    checks = [
        ("forward", events_native, events_torch, forward_rtol, forward_atol),
        ("xsec_x VJP", grad_native_x, grad_torch_x, backward_rtol, backward_atol),
        ("xsec_q2 VJP", grad_native_q, grad_torch_q, backward_rtol, backward_atol),
        ("parameter VJP", param_grad_native, param_grad_torch, backward_rtol, backward_atol),
    ]

    failed = []
    for name, candidate, reference, rtol, atol in checks:
        if not torch.allclose(candidate, reference, rtol=rtol, atol=atol):
            failed.append(name)

    print(
        f"\nTolerance: forward rtol={forward_rtol:.1e}, atol={forward_atol:.1e}; "
        f"backward rtol={backward_rtol:.1e}, atol={backward_atol:.1e}"
    )

    if failed:
        print("FAIL:", ", ".join(failed))
        raise SystemExit(1)

    print(f"PASS: {args.backend} forward and reverse VJP match PyTorch numerically.")


if __name__ == "__main__":
    main()
