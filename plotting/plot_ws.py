#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import load_rows, mean_std, per_iteration_stage_samples


def generate(input_root="results/training", output="results/plots/weak_scaling.pdf"):
    rows = load_rows([input_root])
    raw = per_iteration_stage_samples(rows)

    by_series_ratio = defaultdict(lambda: defaultdict(dict))
    for (key, events), values in raw.items():
        backend, implementation, device, threads = key
        if device != "cpu" or not str(threads).isdigit():
            continue
        threads = int(threads)
        if threads <= 0 or events % threads:
            continue
        ratio = events // threads
        by_series_ratio[(backend, implementation)][ratio][threads] = values["total"]

    selected = []
    for series, ratios in by_series_ratio.items():
        viable = [(ratio, samples) for ratio, samples in ratios.items() if len(samples) >= 2]
        if not viable:
            continue
        ratio, samples = max(viable, key=lambda item: (len(item[1]), item[0]))
        selected.append((series, ratio, samples))

    if not selected:
        return False

    all_threads = sorted({thread for _series, _ratio, samples in selected for thread in samples})
    total_width = 0.75
    width = total_width / len(selected)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    for sidx, (series, ratio, samples) in enumerate(sorted(selected)):
        backend, implementation = series
        xs = []
        ys = []
        errors = []
        for tidx, threads in enumerate(all_threads):
            values = samples.get(threads)
            if not values:
                continue
            mean, std = mean_std([value / 1000.0 for value in values])
            xs.append(tidx - total_width / 2 + width / 2 + sidx * width)
            ys.append(mean)
            errors.append(std)
        label = "C++ (OpenMP)" if backend == "openmp" else implementation
        ax.bar(xs, ys, width=width * 0.92, yerr=errors, capsize=2, edgecolor="black", label=label)

    ax.set_xticks(range(len(all_threads)))
    ax.set_xticklabels([str(value) for value in all_threads])
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("LOITS Forward + Backward Time (s)")
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
        raise SystemExit("not enough weak-scaling CPU region data (constant events/thread) to generate weak scaling")
