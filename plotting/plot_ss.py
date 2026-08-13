#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import load_rows, mean_std, per_iteration_stage_samples, series_key


def _threaded_samples(rows):
    raw = per_iteration_stage_samples(rows)
    grouped = defaultdict(dict)
    for (key, events), values in raw.items():
        backend, implementation, device, threads = key
        if device != "cpu" or not str(threads).isdigit():
            continue
        grouped[(backend, implementation, events)][int(threads)] = values["total"]
    return grouped, raw


def generate(input_root="results/training", output="results/plots/strong_scaling.pdf"):
    rows = load_rows([input_root])
    threaded, raw = _threaded_samples(rows)
    candidates = [(key, samples) for key, samples in threaded.items() if len(samples) >= 2]
    if not candidates:
        return False

    event_count = max(key[2] for key, _ in candidates)
    candidates = [(key, samples) for key, samples in candidates if key[2] == event_count]

    cpp_values = None
    for (key, events), values in raw.items():
        if key[0] == "cpp" and key[2] == "cpu" and events == event_count:
            cpp_values = values["total"]
            break
    if not cpp_values:
        return False

    ref_mean, ref_std = mean_std(cpp_values)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    all_threads = sorted({thread for _key, samples in candidates for thread in samples})
    total_width = 0.75
    nseries = len(candidates)
    width = total_width / max(1, nseries)

    for sidx, (key, samples) in enumerate(sorted(candidates)):
        backend, implementation, _events = key
        xs = []
        ys = []
        errors = []
        for tidx, threads in enumerate(all_threads):
            values = samples.get(threads)
            if not values:
                continue
            mean, std = mean_std(values)
            speedup = ref_mean / mean
            rel_ref = ref_std / ref_mean if ref_mean else 0.0
            rel_cur = std / mean if mean else 0.0
            error = speedup * (rel_ref * rel_ref + rel_cur * rel_cur) ** 0.5
            xs.append(tidx - total_width / 2 + width / 2 + sidx * width)
            ys.append(speedup)
            errors.append(error)
        label = "C++ (OpenMP)" if backend == "openmp" else implementation
        ax.bar(xs, ys, width=width * 0.92, yerr=errors, capsize=2, edgecolor="black", label=label)

    ax.set_xticks(range(len(all_threads)))
    ax.set_xticklabels([str(value) for value in all_threads])
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("LOITS Speedup over Serial C++")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.7)
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
        raise SystemExit("not enough threaded CPU region data to generate strong scaling")
