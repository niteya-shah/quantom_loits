#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import load_rows, percentile, require_rows, series_label


def main():
    parser = argparse.ArgumentParser(description="Plot end-to-end GAN training scaling.")
    parser.add_argument("inputs", nargs="*", default=["results/training"])
    parser.add_argument("--region", default="gan::training_iteration")
    parser.add_argument("--output", default="results/plots/training_scaling.pdf")
    parser.add_argument("--log-y", action="store_true")
    args = parser.parse_args()

    rows = [row for row in load_rows(args.inputs) if row.get("region") == args.region]
    require_rows(rows, args.region)

    groups = defaultdict(lambda: defaultdict(list))
    metadata = {}
    for row in rows:
        label = series_label(row)
        groups[label][row["events"]].append(float(row["cpu_ms"]))
        metadata[label] = row

    fig, ax = plt.subplots()
    for label, by_events in sorted(groups.items()):
        xs = sorted(by_events)
        medians = [percentile(by_events[x], 0.50) for x in xs]
        q1 = [percentile(by_events[x], 0.25) for x in xs]
        q3 = [percentile(by_events[x], 0.75) for x in xs]
        lower = [m - lo for m, lo in zip(medians, q1)]
        upper = [hi - m for m, hi in zip(medians, q3)]
        ax.errorbar(xs, medians, yerr=[lower, upper], marker="o", capsize=3, label=label)

    ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Requested events")
    ax.set_ylabel("Training iteration time (ms)")
    ax.set_title("End-to-end GAN training")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
