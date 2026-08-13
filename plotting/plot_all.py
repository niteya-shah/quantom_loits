#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

from plotting.common import load_rows


def run(module, *args):
    command = [sys.executable, "-m", module, *map(str, args)]
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate the standard QuantOm plots from available CSVs.")
    parser.add_argument("--input", default="results/training")
    parser.add_argument("--output", default="results/plots")
    args = parser.parse_args()

    rows = load_rows([args.input])
    if not rows:
        raise SystemExit(f"no CSV results found under {args.input}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run("plotting.plot_training", args.input, "--output", output / "training_scaling.pdf")

    regions = {row.get("region") for row in rows}
    if "loits::autograd::forward" in regions:
        run("plotting.plot_loits", args.input, "--scope", "autograd-forward", "--output", output / "loits_forward.pdf")
        implementations = {row.get("implementation") or row.get("backend") for row in rows}
        if "cpp" in implementations:
            run("plotting.plot_speedup", args.input, "--reference", "cpp", "--output", output / "loits_forward_speedup.pdf")
    if "loits::autograd::backward" in regions:
        run("plotting.plot_loits", args.input, "--scope", "autograd-backward", "--output", output / "loits_backward.pdf")

    event_counts = sorted({row.get("events") for row in rows if row.get("region", "").startswith("loits::forward::")})
    for events in event_counts:
        run("plotting.plot_breakdown", args.input, "--events", events, "--output", output / f"loits_breakdown_{events}.pdf")


if __name__ == "__main__":
    main()
