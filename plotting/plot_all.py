#!/usr/bin/env python3
import argparse
from pathlib import Path

from plotting.common import available_sites, load_rows
from plotting.plot_fixed_resource import generate as generate_fixed
from plotting.plot_ss import generate as generate_strong
from plotting.plot_ws import generate as generate_weak


def main():
    parser = argparse.ArgumentParser(description="Generate the standard QuantOm publication plots.")
    parser.add_argument("--input", default="results/training")
    parser.add_argument("--output", default="results/plots")
    args = parser.parse_args()

    rows = load_rows([args.input])
    output = Path(args.output)
    made = 0

    if any(row.get("device") != "cpu" for row in rows):
        made += int(generate_fixed([args.input], output / "gpu" / "fixed.pdf", cpu=False))

    for site in available_sites(rows, cpu=True):
        site_output = output / site
        made += int(generate_fixed([args.input], site_output / "fixed.pdf", cpu=True, site=site))
        made += int(generate_strong(args.input, site_output / "strong.pdf", site=site))
        made += int(generate_weak(args.input, site_output / "weak.pdf", site=site))

    if not made:
        raise SystemExit("no standard plots could be generated from the canonical result tree")


if __name__ == "__main__":
    main()
