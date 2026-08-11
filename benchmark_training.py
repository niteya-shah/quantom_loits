import argparse
from pathlib import Path

from pytorch.gan import GANTrainer
from pytorch.profiler import RegionHooks, TrainingProfiler


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
    trace = Path(args.output) / f"{args.backend}_{suffix}.json" if args.trace else None
    prof = profiler.run(trainer, args.warmup, args.iterations, trace)
    rows = profiler.rows(prof, args.backend, args.device, args.events)
    name = f"{suffix}_{args.backend}_{args.device.replace(':', '-')}_{args.events}.csv"
    profiler.write_csv(Path(args.output) / name, rows)
    if hooks:
        hooks.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="torch", choices=["torch", "cpp", "openmp", "sycl"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--events", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/training")
    parser.add_argument("--regions", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    run(args, False)
    if args.regions:
        run(args, True)


if __name__ == "__main__":
    main()
