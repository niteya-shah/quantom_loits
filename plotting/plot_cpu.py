#!/usr/bin/env python3
from pathlib import Path

from plotting.common import available_sites, load_rows
from plotting.plot_fixed_resource import generate as generate_fixed
from plotting.plot_ss import generate as generate_strong
from plotting.plot_ws import generate as generate_weak


def main(input_root="results/training", output_root="results/plots"):
    rows = load_rows([input_root])
    sites = available_sites(rows, cpu=True)
    if not sites:
        raise SystemExit("no CPU benchmark data found")

    made = 0
    for site in sites:
        out = Path(output_root) / site
        made += int(generate_fixed([input_root], out / "fixed.pdf", cpu=True, site=site))
        made += int(generate_strong(input_root, out / "strong.pdf", site=site))
        made += int(generate_weak(input_root, out / "weak.pdf", site=site))
    if not made:
        raise SystemExit("not enough CPU benchmark data to generate plots")


if __name__ == "__main__":
    main()
