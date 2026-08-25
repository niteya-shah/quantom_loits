#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import load_rows, mean_std, rows_for_experiment, rows_for_site, training_samples



def _cpu_label(backend, implementation):
    if backend == "openmp":
        return "OpenMP"
    if backend == "torch":
        return "Torch"
    if backend == "sycl":
        if implementation.startswith("acpp-"):
            return "AdaptiveCPP"
        if implementation.startswith("dpcpp-"):
            return "DPC++"
    return implementation

def generate(input_root="results/training", output="results/plots/weak.png", site=None):
    rows = rows_for_experiment(load_rows([input_root]), "weak")
    if site is not None:
        rows = rows_for_site(rows, site)
    raw = training_samples(rows)

    by_series_ratio = defaultdict(lambda: defaultdict(dict))
    for (key, events), values in raw.items():
        row_site, backend, implementation, device, threads = key
        if device != "cpu" or not str(threads).isdigit():
            continue
        threads = int(threads)
        if threads <= 0 or events % threads:
            continue
        ratio = events // threads
        by_series_ratio[(row_site, backend, implementation)][ratio][threads] = values

    selected = []
    for series, ratios in by_series_ratio.items():
        viable = [(ratio, values) for ratio, values in ratios.items() if len(values) >= 2]
        if not viable:
            continue
        ratio, values = max(viable, key=lambda item: (len(item[1]), item[0]))
        selected.append((series, ratio, values))

    if not selected:
        return False

    all_threads = sorted({thread for _series, _ratio, values in selected for thread in values})
    total_width = 0.75
    width = total_width / len(selected)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    for sidx, (series, _ratio, values_by_thread) in enumerate(sorted(selected)):
        _site, backend, implementation = series
        xs = []
        ys = []
        errors = []
        for tidx, threads in enumerate(all_threads):
            values = values_by_thread.get(threads)
            if not values:
                continue
            mean, std = mean_std(values)
            xs.append(tidx - total_width / 2 + width / 2 + sidx * width)
            ys.append(mean)
            errors.append(std)
        label = _cpu_label(backend, implementation)
        ax.bar(xs, ys, width=width * 0.92, yerr=errors, capsize=2, edgecolor="black", label=label)

    ax.set_xticks(range(len(all_threads)))
    ax.set_xticklabels([str(value) for value in all_threads])
    ax.set_xlabel("Number of Hardware Threads")
    ax.set_ylabel("GAN Iteration Time (s)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    fig.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)
    return True


if __name__ == "__main__":
    if not generate():
        raise SystemExit("not enough weak-scaling wall-clock data")
