#!/usr/bin/env python3
from plotting.plot_fixed_resource import generate


if __name__ == "__main__":
    if not generate(["results/training"], "results/plots/cpu_scaling.pdf", cpu=True):
        raise SystemExit("not enough CPU training + region data under results/training")
