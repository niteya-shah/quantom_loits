#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import load_rows, percentile, require_rows, series_label


def main():
    parser = argparse.ArgumentParser(description="Plot speedup relative to a reference implementation.")
    parser.add_argument("inputs", nargs="*", default=["results/training"])
    parser.add_argument("--region", default="loits::autograd::forward")
    parser.add_argument("--reference", default="cpp")
    parser.add_argument("--output", default="results/plots/loits_speedup.pdf")
    args = parser.parse_args()

    rows = [row for row in load_rows(args.inputs) if row.get("region") == args.region]
    require_rows(rows, args.region)

    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[series_label(row)][row["events"]].append(float(row["cpu_ms"]))

    if args.reference not in groups:
        candidates = ", ".join(sorted(groups))
        raise SystemExit(f"reference {args.reference!r} not present; available series: {candidates}")

    reference = {events: percentile(values, 0.50) for events, values in groups[args.reference].items()}
    fig, ax = plt.subplots()
    for label, by_events in sorted(groups.items()):
        xs = sorted(set(by_events) & set(reference))
        if not xs:
            continue
        speedup = [reference[x] / percentile(by_events[x], 0.50) for x in xs]
        ax.plot(xs, speedup, marker="o", label=label)

    ax.axhline(1.0, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Requested events")
    ax.set_ylabel(f"Speedup vs {args.reference}")
    ax.set_title(args.region)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
