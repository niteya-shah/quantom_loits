#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import load_rows, mean_std, rows_for_experiment, rows_for_site, training_samples


def generate(input_root="results/training", output="results/plots/strong.png", site=None):
    rows = rows_for_experiment(load_rows([input_root]), "strong")
    if site is not None:
        rows = rows_for_site(rows, site)
    samples = training_samples(rows)

    threaded = defaultdict(dict)
    cpp = {}
    for (key, events), values in samples.items():
        row_site, backend, implementation, device, threads = key
        if device != "cpu":
            continue
        if backend == "cpp":
            cpp[events] = values
            continue
        if not str(threads).isdigit():
            continue
        threaded[(row_site, backend, implementation, events)][int(threads)] = values

    candidates = [(key, values) for key, values in threaded.items() if len(values) >= 2]
    if not candidates:
        return False

    event_count = max(key[3] for key, _ in candidates)
    candidates = [(key, values) for key, values in candidates if key[3] == event_count]
    cpp_values = cpp.get(event_count)
    if not cpp_values:
        return False

    ref_mean, ref_std = mean_std(cpp_values)
    all_threads = sorted({thread for _key, values in candidates for thread in values})
    total_width = 0.75
    width = total_width / len(candidates)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    for sidx, (key, values_by_thread) in enumerate(sorted(candidates)):
        _site, backend, implementation, _events = key
        xs = []
        ys = []
        errors = []
        for tidx, threads in enumerate(all_threads):
            values = values_by_thread.get(threads)
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
    ax.set_ylabel("GAN Iteration Speedup over Serial C++")
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
        raise SystemExit("not enough strong-scaling wall-clock data")
