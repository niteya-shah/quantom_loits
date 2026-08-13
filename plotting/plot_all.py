#!/usr/bin/env python3
import argparse
from pathlib import Path

from plotting.plot_fixed_resource import generate as generate_fixed_resource
from plotting.plot_ss import generate as generate_ss
from plotting.plot_ws import generate as generate_ws


def main():
    parser = argparse.ArgumentParser(description="Generate the standard QuantOm publication plots.")
    parser.add_argument("--input", default="results/training")
    parser.add_argument("--output", default="results/plots")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    made = 0
    made += int(generate_fixed_resource([args.input], output / "cpu_scaling.pdf", cpu=True))
    made += int(generate_fixed_resource([args.input], output / "gpu_scaling.pdf", cpu=False))
    made += int(generate_ss(args.input, output / "strong_scaling.pdf"))
    made += int(generate_ws(args.input, output / "weak_scaling.pdf"))

    if not made:
        raise SystemExit(
            "no standard plots could be generated; run benchmarks with --regions so both training_*.csv and regions_*.csv exist"
        )


if __name__ == "__main__":
    main()
