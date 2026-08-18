import argparse
import os
import re
from pathlib import Path

from loits import backend_status, registered_backends
from pytorch.gan import GANTrainer
from pytorch.profiler import RegionHooks, TrainingProfiler


def safe_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "run"


def implementation_name(backend):
    if backend != "sycl":
        return backend
    from sycl.backend import selected_variant

    return selected_variant()


def native_threads(backend):
    if backend == "openmp":
        return os.environ.get("OMP_NUM_THREADS", "")
    return ""


def run(args, profile_regions=False):
    trainer = GANTrainer(
        backend=args.backend,
        device=args.device,
        n_events=args.events,
        grid_size=args.grid_size,
        compile=True,
        profile_regions=profile_regions,
        seed=args.seed,
    )
    hooks = None
    if profile_regions and args.backend == "torch":
        hooks = RegionHooks(trainer.sampler.impl)
    profiler = TrainingProfiler(args.device)
    suffix = "regions" if profile_regions else "training"
    implementation = implementation_name(args.backend)
    threads = native_threads(args.backend)
    site = safe_component(args.site) if args.backend == "torch" and args.site else ""
    stem = "_".join(
        part
        for part in [
            suffix,
            safe_component(implementation),
            site,
            safe_component(args.device.replace(":", "-")),
            str(args.events),
            f"t{threads}" if threads else "",
        ]
        if part
    )
    trace = Path(args.output) / f"{stem}.json" if args.trace else None
    prof = profiler.run(trainer, args.warmup, args.iterations, trace)
    rows = profiler.rows(
        prof,
        args.backend,
        args.device,
        args.events,
        implementation=implementation,
        threads=threads,
    )
    profiler.write_csv(Path(args.output) / f"{stem}.csv", rows)
    if hooks:
        hooks.close()


def print_backends(device):
    for backend in registered_backends():
        ok, reason = backend_status(backend, device)
        state = "available" if ok else "unavailable"
        suffix = "" if ok or not reason else f": {reason}"
        print(f"{backend:8s} {state}{suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="torch", choices=registered_backends())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--events", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/training")
    parser.add_argument("--site", default=os.environ.get("QUANTOM_SITE", ""))
    parser.add_argument("--regions", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--list-backends", action="store_true")
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="print a skip message and exit successfully when the requested backend/device is unavailable",
    )
    args = parser.parse_args()

    if args.list_backends:
        print_backends(args.device)
        return

    ok, reason = backend_status(args.backend, args.device)
    if not ok:
        message = f"backend={args.backend!r} unavailable for device={args.device!r}: {reason}"
        if args.skip_unavailable:
            print(f"SKIP: {message}")
            return
        parser.error(message)

    run(args, False)
    if args.regions:
        run(args, True)


if __name__ == "__main__":
    main()
