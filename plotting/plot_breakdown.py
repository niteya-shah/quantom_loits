#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import load_rows, metric_value, percentile, require_rows, series_label


FORWARD = [
    ("allocation", ["loits::forward::allocation"]),
    ("density", ["loits::forward::rho_x", "loits::forward::rho_q2"]),
    ("CDF", ["loits::forward::cdf_x", "loits::forward::cdf_q2"]),
    ("RNG", ["loits::forward::random_x", "loits::forward::random_q2"]),
    ("interpolation", ["loits::forward::interpolation_x", "loits::forward::interpolation_q2"]),
    ("compaction", ["loits::forward::stream_compaction"]),
]

BACKWARD = [
    ("interpolation VJP", ["loits::backward::interpolation_x", "loits::backward::interpolation_q2"]),
    ("CDF VJP", ["loits::backward::cdf_x", "loits::backward::cdf_q2"]),
    ("density VJP", ["loits::backward::rho_x", "loits::backward::rho_q2"]),
]


def stage_median(rows, regions, metric):
    by_occurrence = defaultdict(float)
    seen = defaultdict(set)
    expected = set(regions)
    for row in rows:
        region = row.get("region")
        if region not in expected:
            continue
        occurrence = int(row.get("occurrence") or 0)
        by_occurrence[occurrence] += metric_value(row, metric)
        seen[occurrence].add(region)
    complete = [value for occurrence, value in by_occurrence.items() if seen[occurrence] == expected]
    return percentile(complete, 0.50) if complete else 0.0


def main():
    parser = argparse.ArgumentParser(description="Plot native LOITS forward/backward stage breakdown.")
    parser.add_argument("inputs", nargs="*", default=["results/training"])
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metric", choices=["auto", "cpu_ms", "device_ms"], default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = [row for row in load_rows(args.inputs) if row.get("events") == args.events]
    if args.device:
        rows = [row for row in rows if row.get("device") == args.device]
    rows = [row for row in rows if row.get("region", "").startswith("loits::")]
    require_rows(rows, f"LOITS regions at events={args.events}")

    groups = defaultdict(list)
    for row in rows:
        groups[series_label(row)].append(row)

    implementations = sorted(groups)
    bar_labels = []
    bar_groups = []
    for label in implementations:
        bar_labels.extend([f"{label} F", f"{label} B"])
        bar_groups.extend([(label, "forward"), (label, "backward")])

    fig, ax = plt.subplots()
    bottom = [0.0] * len(bar_groups)
    for stage, regions, direction in [
        *[(stage, regions, "forward") for stage, regions in FORWARD],
        *[(stage, regions, "backward") for stage, regions in BACKWARD],
    ]:
        values = []
        for label, bar_direction in bar_groups:
            values.append(
                stage_median(groups[label], regions, args.metric)
                if bar_direction == direction
                else 0.0
            )
        if not any(values):
            continue
        ax.bar(bar_labels, values, bottom=bottom, label=stage)
        bottom = [base + value for base, value in zip(bottom, values)]

    ax.set_ylabel(f"Median per-call region time ({args.metric}, ms)")
    ax.set_title(f"LOITS native stage breakdown, {args.events:,} requested events")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()

    output = Path(args.output or f"results/plots/loits_breakdown_{args.events}.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
