#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from plotting.common import (
    STAGES,
    fixed_resource_events,
    fixed_resource_series,
    event_hatch,
    load_rows,
    mean_std,
    per_iteration_stage_samples,
    rows_for_device,
    series_label,
    training_samples,
)
from plotting.plotting_tools import clustered_positions, event_legend, plot_runtime_bars, stage_legend


def generate(inputs, output, cpu):
    rows = load_rows(inputs)
    rows = rows_for_device(rows, cpu)
    series = fixed_resource_series(rows, cpu)
    if not series:
        return False

    stage_samples = per_iteration_stage_samples(rows)
    training = training_samples(rows)
    if not stage_samples or not training:
        return False

    # The results directory also contains strong/weak-scaling runs.  Only use
    # event sizes that form a complete fixed-resource comparison across every
    # selected implementation; otherwise weak-scaling sizes (20k, 40k, ...)
    # create sparse/misaligned bars and an incorrect event legend.
    events = fixed_resource_events(series, stage_samples, training)
    if not events:
        return False

    mpl.rcParams["hatch.linewidth"] = 0.15
    fig, ax = plt.subplots(2, sharex="col", figsize=(12, 6), dpi=200)

    positions, width = clustered_positions(len(series), len(events))
    colors = [plt.get_cmap("tab20")(idx) for idx in range(len(STAGES))]

    for sidx, key in enumerate(series):
        for eidx, event_count in enumerate(events):
            stats = stage_samples.get((key, event_count))
            if not stats:
                continue
            stage_means = [mean_std(stats.get(stage, []))[0] if stats.get(stage) else 0.0 for stage, _ in STAGES]
            total = sum(stage_means)
            if total <= 0.0:
                continue
            bottom = 0.0
            xpos = positions[(sidx, eidx)]
            for (stage, _regions), value, color in zip(STAGES, stage_means, colors):
                height = 100.0 * value / total
                if height <= 0.0:
                    continue
                ax[0].bar(
                    xpos,
                    height,
                    bottom=bottom,
                    width=width * 0.92,
                    color=color,
                    edgecolor="black",
                    linewidth=0.25,
                    hatch=event_hatch(event_count, events),
                )
                bottom += height

    ax[0].set_ylim(0, 100)
    ax[0].set_ylabel("% of LOITS Runtime\n(Forward + Backward)")

    plot_runtime_bars(ax[1], series, events, training, "GAN Iteration Time (s)", show_threads=False)

    fig.subplots_adjust(wspace=0.10, hspace=0.12, top=0.96, bottom=0.14, right=0.82, left=0.08)
    stage_handles = stage_legend([stage for stage, _ in STAGES], colors)
    event_handles = event_legend(events)
    ax[0].legend(stage_handles, [handle.get_label() for handle in stage_handles], fontsize=7.5, bbox_to_anchor=(1.01, 0.5), loc="center left")
    ax[1].legend(event_handles, [handle.get_label() for handle in event_handles], fontsize=8, bbox_to_anchor=(1.01, 0.5), loc="center left")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate the old-style fixed-resource stacked/bar plot from current benchmark CSVs.")
    parser.add_argument("--input", default="results/training")
    parser.add_argument("--output", default="results/plots/cpu_scaling.pdf")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not generate([args.input], args.output, cpu=not args.gpu):
        raise SystemExit("not enough matching training + region data to generate this plot")


if __name__ == "__main__":
    main()
