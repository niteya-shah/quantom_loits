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
    rows_for_experiment,
    rows_for_site,
    training_samples,
)
from plotting.plotting_tools import clustered_positions, event_legend, plot_runtime_bars, stage_legend


GPU_DEVICE_LABELS = {
    "athena": "NVIDIA A100",
    "polaris": "NVIDIA A100",
    "illyad": "NVIDIA H100",
    "odyssey": "AMD MI300A",
    "aurora": "Intel PVC",
}
GPU_DEVICE_ORDER = {
    "athena": 0,
    "polaris": 0,
    "illyad": 1,
    "odyssey": 2,
    "aurora": 3,
}


def _gpu_implementation_label(key):
    _site, backend, implementation, _device, _threads = key
    if backend == "sycl":
        if implementation.startswith("acpp-"):
            return "AdaptiveCPP"
        if implementation.startswith("dpcpp-"):
            return "DPC++"
    if backend == "torch":
        return "Torch"
    if backend == "triton":
        return "Triton"
    return implementation


def _gpu_implementation_rank(key):
    label = _gpu_implementation_label(key)
    order = {"AdaptiveCPP": 0, "DPC++": 1, "Torch": 2, "Triton": 3}
    return (order.get(label, 99), label)


def _gpu_device_columns(series):
    grouped = {}
    for key in series:
        grouped.setdefault(key[0], []).append(key)
    sites = sorted(grouped, key=lambda value: (GPU_DEVICE_ORDER.get(value, 99), value))
    return [(site, sorted(grouped[site], key=_gpu_implementation_rank)) for site in sites]


def _plot_breakdown(ax, series, events, stage_samples, colors):
    positions, width = clustered_positions(len(series), len(events))
    for sidx, key in enumerate(series):
        for eidx, event_count in enumerate(events):
            stats = stage_samples.get((key, event_count))
            if not stats:
                continue
            stage_means = [
                mean_std(stats.get(stage, []))[0] if stats.get(stage) else 0.0
                for stage, _ in STAGES
            ]
            total = sum(stage_means)
            if total <= 0.0:
                continue
            bottom = 0.0
            xpos = positions[(sidx, eidx)]
            for (stage, _regions), value, color in zip(STAGES, stage_means, colors):
                height = 100.0 * value / total
                if height <= 0.0:
                    continue
                ax.bar(
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
    ax.set_ylim(0, 100)


def _gpu_device_title(site, series):
    if site in GPU_DEVICE_LABELS:
        return GPU_DEVICE_LABELS[site]
    device = series[0][3] if series else "GPU"
    return device.upper()


def generate(inputs, output, cpu, site=None):
    rows = rows_for_experiment(load_rows(inputs), "fixed")
    if site is not None:
        rows = rows_for_site(rows, site)
    rows = rows_for_device(rows, cpu)
    series = fixed_resource_series(rows, cpu)
    if not series:
        return False

    stage_samples = per_iteration_stage_samples(rows)
    training = training_samples(rows)
    if not stage_samples or not training:
        return False

    # Keep only event sizes with complete timing and region data across every
    # selected fixed-resource implementation.
    events = fixed_resource_events(series, stage_samples, training)
    if not events:
        return False

    mpl.rcParams["hatch.linewidth"] = 0.15
    colors = [plt.get_cmap("tab20")(idx) for idx in range(len(STAGES))]
    stage_handles = stage_legend([stage for stage, _ in STAGES], colors)
    event_handles = event_legend(events)
    forward_handles = [
        (handle, handle.get_label().removeprefix("F: "))
        for handle in stage_handles
        if handle.get_label().startswith("F: ")
    ][::-1]
    backward_handles = [
        (handle, handle.get_label().removeprefix("B: "))
        for handle in stage_handles
        if handle.get_label().startswith("B: ")
    ][::-1]

    if not cpu and site is None:
        columns = _gpu_device_columns(series)
        fig, ax = plt.subplots(
            2,
            len(columns),
            sharex="col",
            figsize=(4.0 * len(columns), 6),
            dpi=300,
            squeeze=False,
        )

        for col, (device_site, device_series) in enumerate(columns):
            ax[1, col].set_axisbelow(True)
            ax[1, col].yaxis.grid(True, which="major", linewidth=0.5, alpha=0.85)
            ax[1, col].yaxis.grid(True, which="minor", linewidth=0.3, alpha=0.65)

            _plot_breakdown(ax[0, col], device_series, events, stage_samples, colors)
            ax[0, col].set_title(_gpu_device_title(device_site, device_series), fontsize=11)
            plot_runtime_bars(
                ax[1, col],
                device_series,
                events,
                training,
                "GAN Iteration Time (s)",
                show_threads=False,
            )
            ax[1, col].set_xticklabels(
                [_gpu_implementation_label(key) for key in device_series],
                rotation=30,
                ha="center",
            )
            if col != 0:
                ax[0, col].sharey(ax[0, 0])
                ax[0, col].tick_params(axis="y", labelleft=False)
                ax[0, col].set_ylabel("")
                ax[1, col].set_ylabel("")

        ax[0, 0].set_ylabel("% of LOITS Runtime\n(Forward + Backward)")
        ax[1, 0].set_ylabel("GAN Iteration Time (s)")
        fig.subplots_adjust(
            wspace=0.15,
            hspace=0.10,
            top=0.95,
            bottom=0.12,
            right=0.88,
            left=0.065,
        )
        backward_legend = ax[0, -1].legend(
            [handle for handle, _label in backward_handles],
            [label for _handle, label in backward_handles],
            title="Backward",
            fontsize=7.5,
            title_fontsize=8,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
        )
        ax[0, -1].add_artist(backward_legend)
        ax[0, -1].legend(
            [handle for handle, _label in forward_handles],
            [label for _handle, label in forward_handles],
            title="Forward",
            fontsize=7.5,
            title_fontsize=8,
            bbox_to_anchor=(1.02, 0.52),
            loc="upper left",
        )
        ax[1, -1].legend(
            event_handles,
            [handle.get_label() for handle in event_handles],
            fontsize=8,
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
        )
    else:
        fig, ax = plt.subplots(2, sharex="col", figsize=(12, 6), dpi=200)
        _plot_breakdown(ax[0], series, events, stage_samples, colors)
        ax[0].set_ylabel("% of LOITS Runtime\n(Forward + Backward)")
        plot_runtime_bars(
            ax[1],
            series,
            events,
            training,
            "GAN Iteration Time (s)",
            show_threads=False,
        )
        fig.subplots_adjust(
            wspace=0.10,
            hspace=0.12,
            top=0.96,
            bottom=0.14,
            right=0.82,
            left=0.08,
        )
        ax[0].legend(
            stage_handles,
            [handle.get_label() for handle in stage_handles],
            fontsize=7.5,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
        )
        ax[1].legend(
            event_handles,
            [handle.get_label() for handle in event_handles],
            fontsize=8,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate a fixed-resource stacked/runtime plot.")
    parser.add_argument("--input", default="results/training")
    parser.add_argument("--output", default="results/plots/fixed.png")
    parser.add_argument("--site")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not generate([args.input], args.output, cpu=not args.gpu, site=args.site):
        raise SystemExit("not enough matching training + region data to generate this plot")


if __name__ == "__main__":
    main()
