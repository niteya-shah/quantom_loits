#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import metric_value, load_rows, percentile, require_rows, series_label


REGIONS = {
    "autograd-forward": "loits::autograd::forward",
    "autograd-backward": "loits::autograd::backward",
    "binding-forward": "loits::binding::forward",
    "binding-backward": "loits::binding::backward",
    "native-forward": "loits::forward",
    "native-backward": "loits::backward",
}


def main():
    parser = argparse.ArgumentParser(description="Plot LOITS forward/backward scaling.")
    parser.add_argument("inputs", nargs="*", default=["results/training"])
    parser.add_argument("--scope", choices=sorted(REGIONS), default="autograd-forward")
    parser.add_argument("--region", help="override the semantic region name")
    parser.add_argument("--metric", choices=["auto", "cpu_ms", "device_ms"], default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-y", action="store_true")
    args = parser.parse_args()

    region = args.region or REGIONS[args.scope]
    all_rows = load_rows(args.inputs)
    rows = [row for row in all_rows if row.get("region") == region]
    require_rows(rows, region)

    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[series_label(row)][row["events"]].append(metric_value(row, args.metric))

    fig, ax = plt.subplots()
    for label, by_events in sorted(groups.items()):
        xs = sorted(by_events)
        medians = [percentile(by_events[x], 0.50) for x in xs]
        q1 = [percentile(by_events[x], 0.25) for x in xs]
        q3 = [percentile(by_events[x], 0.75) for x in xs]
        ax.errorbar(
            xs,
            medians,
            yerr=[
                [m - lo for m, lo in zip(medians, q1)],
                [hi - m for m, hi in zip(medians, q3)],
            ],
            marker="o",
            capsize=3,
            label=label,
        )

    ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Requested events")
    ax.set_ylabel(f"{region} ({args.metric}, ms)")
    ax.set_title(region)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = Path(args.output or f"results/plots/{args.scope}.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
