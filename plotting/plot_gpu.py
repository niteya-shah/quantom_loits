#!/usr/bin/env python3
from plotting.plot_fixed_resource import generate


if __name__ == "__main__":
    if not generate(["results/training"], "results/plots/gpu_scaling.pdf", cpu=False):
        raise SystemExit("not enough GPU training + region data under results/training")
