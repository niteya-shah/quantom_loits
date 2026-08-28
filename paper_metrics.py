#!/usr/bin/env python3
import argparse
import ast
import csv
import math
import re
import statistics
import tokenize
from collections import defaultdict
from io import StringIO
from pathlib import Path

PAPER_SITES = ("instinct", "athena", "odyssey", "aurora")

STAGES = (
    ("F: Allocation", ("loits::forward::allocation",)),
    ("F: Density", ("loits::forward::rho_x", "loits::forward::rho_q2")),
    ("F: CDF", ("loits::forward::cdf_x", "loits::forward::cdf_q2")),
    ("F: Flatten", ("loits::forward::flatten_x", "loits::forward::flatten_q2")),
    ("F: RNG", ("loits::forward::random_x", "loits::forward::random_q2")),
    ("F: Interpolation", ("loits::forward::interpolation_x", "loits::forward::interpolation_q2")),
    ("F: Compaction", ("loits::forward::stream_compaction",)),
    (
        "B: Interpolation VJP",
        (
            "loits::backward::stream_compaction",
            "loits::backward::interpolation_x",
            "loits::backward::interpolation_q2",
        ),
    ),
    ("B: Flatten", ("loits::backward::flatten_x", "loits::backward::flatten_q2")),
    ("B: CDF VJP", ("loits::backward::cdf_x", "loits::backward::cdf_q2")),
    ("B: Density VJP", ("loits::backward::rho_x", "loits::backward::rho_q2")),
)

PYTORCH_CORE_SYMBOLS = (
    "EventAllocation",
    "Density",
    "CDF",
    "FlattenObservable",
    "RandomSamples",
    "LinearInterpolation",
    "StreamCompaction",
    "TorchLOITSCore",
)

SLOC_FILES = {
    "C++": {
        "core": ("cpp/loits_core.cpp", "cpp/loits_core.hpp"),
        "interface": ("cpp/bindings.cpp", "cpp/backend.py"),
        "shared": ("rng/philox.hpp",),
    },
    "OpenMP": {
        "core": ("openmp/loits_core.cpp", "openmp/loits_core.hpp"),
        "interface": ("openmp/bindings.cpp", "openmp/backend.py"),
        "shared": ("rng/philox.hpp",),
    },
    "SYCL": {
        "core": ("sycl/loits_core.cpp", "sycl/loits_core.hpp"),
        "interface": ("sycl/bindings.cpp", "sycl/backend.py", "sycl/runtime.cpp", "sycl/runtime.hpp"),
        "shared": ("rng/philox.hpp",),
    },
    "Triton": {
        "core": ("triton_backend/kernels.py",),
        "interface": ("triton_backend/backend.py",),
        "shared": (),
    },
}


def csv_paths(root):
    return sorted(Path(root).rglob("*.csv"))


def parse_value(key, value):
    if value in (None, ""):
        return value
    if key in {"events", "grid_size", "warmup", "iterations", "seed", "occurrence"}:
        return int(value)
    if key in {"wall_ms", "cpu_ms", "device_ms", "self_cpu_ms", "self_device_ms", "start_us", "end_us"}:
        return float(value)
    return value


def load_rows(root, sites):
    rows = []
    for path in csv_paths(root):
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                row = {key: parse_value(key, value) for key, value in raw.items()}
                row["_path"] = str(path)
                if row.get("site") in sites:
                    rows.append(row)
    return rows


def is_training(row):
    return Path(row["_path"]).name.startswith("training_")


def is_regions(row):
    return Path(row["_path"]).name.startswith("regions_")


def series_key(row, include_threads=True):
    backend = row.get("backend", "")
    return (
        row.get("site", ""),
        backend,
        row.get("implementation") or backend or "unknown",
        row.get("device", ""),
        str(row.get("threads", "")) if include_threads else "",
    )


def impl_label(key):
    _site, backend, implementation, _device, _threads = key
    if backend == "cpp":
        return "C++"
    if backend == "openmp":
        return "OpenMP"
    if backend == "torch":
        return "Torch"
    if backend == "triton":
        return "Triton"
    if backend == "sycl":
        if implementation.startswith("acpp-"):
            return "AdaptiveCPP"
        if implementation.startswith("dpcpp-"):
            return "DPC++"
    return implementation


def mean_std(values):
    values = [float(value) for value in values]
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def training_samples(rows, experiment=None):
    grouped = defaultdict(list)
    for row in rows:
        if not is_training(row) or row.get("region") != "gan::training_iteration":
            continue
        if experiment is not None and row.get("experiment") != experiment:
            continue
        wall_ms = row.get("wall_ms")
        if wall_ms in (None, ""):
            continue
        grouped[(series_key(row), row["events"])].append(float(wall_ms) / 1000.0)
    return grouped


def fixed_series(rows):
    grouped = defaultdict(set)
    for row in rows:
        if row.get("experiment") != "fixed":
            continue
        full = series_key(row)
        base = series_key(row, include_threads=False)
        grouped[base].add(full)
    selected = []
    for candidates in grouped.values():
        def rank(key):
            threads = key[4]
            return (1, int(threads)) if threads.isdigit() else (0, 0)
        selected.append(max(candidates, key=rank))
    return sorted(selected)


def event_maps(rows):
    by_id = {}
    for row in rows:
        event_id = str(row.get("event_id") or "")
        if not event_id:
            continue
        key = (row["_path"], event_id)
        current = by_id.get(key)
        if current is None or (not current.get("parent_event_id") and row.get("parent_event_id")):
            by_id[key] = row
    return by_id


def training_intervals(rows):
    intervals = defaultdict(list)
    for row in rows:
        if row.get("region") != "gan::training_iteration":
            continue
        start = row.get("start_us")
        end = row.get("end_us")
        if start in (None, "") or end in (None, ""):
            continue
        root = (row["_path"], str(row.get("event_id") or row.get("occurrence") or "0"))
        intervals[row["_path"]].append((float(start), float(end), root))
    return intervals


def training_root(row, by_id, intervals):
    if row.get("region") == "gan::training_iteration":
        return (row["_path"], str(row.get("event_id") or row.get("occurrence") or "0"))
    current = row
    visited = set()
    while current is not None:
        parent_id = str(current.get("parent_event_id") or "")
        if not parent_id:
            break
        key = (current["_path"], parent_id)
        if key in visited:
            break
        visited.add(key)
        parent = by_id.get(key)
        if parent is None:
            break
        if parent.get("region") == "gan::training_iteration":
            return key
        current = parent
    start = row.get("start_us")
    end = row.get("end_us")
    if start in (None, "") or end in (None, ""):
        return None
    candidates = [
        item for item in intervals.get(row["_path"], ())
        if item[0] <= float(start) and float(end) <= item[1]
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1] - item[0])[2]


def stage_samples(rows):
    region_rows = [
        row for row in rows
        if is_regions(row)
        and row.get("experiment") == "fixed"
        and (row.get("region_timing") == "synchronized_wall" or row.get("device") == "cpu")
    ]
    by_id = event_maps(region_rows)
    intervals = training_intervals(region_rows)
    region_to_stage = {region: stage for stage, regions in STAGES for region in regions}
    by_iteration = defaultdict(lambda: defaultdict(float))
    for row in region_rows:
        stage = region_to_stage.get(row.get("region"))
        if stage is None:
            continue
        event_id = str(row.get("event_id") or "")
        if event_id and by_id.get((row["_path"], event_id)) is not row:
            continue
        root = training_root(row, by_id, intervals)
        if root is None:
            continue
        by_iteration[(series_key(row), row["events"], root)][stage] += float(row.get("cpu_ms") or 0.0)
    grouped = defaultdict(lambda: defaultdict(list))
    for (series, events, _root), values in by_iteration.items():
        for stage, _regions in STAGES:
            grouped[(series, events)][stage].append(values.get(stage, 0.0))
    return grouped


def runtime_rows(rows):
    samples = training_samples(rows, "fixed")
    selected = set(fixed_series(rows))
    output = []
    for (key, events), values in sorted(samples.items()):
        if key not in selected:
            continue
        mean, std = mean_std(values)
        output.append({
            "site": key[0], "device": key[3], "implementation": impl_label(key),
            "backend": key[1], "variant": key[2], "threads": key[4], "events": events,
            "mean_s": mean, "std_s": std, "samples": len(values),
        })
    return output


def pairwise_rows(runtime, experiment):
    grouped = defaultdict(list)
    for row in runtime:
        extra = (row.get("threads", ""),) if experiment in {"strong", "weak"} else ()
        grouped[(row["site"], row["device"], row["events"], *extra)].append(row)
    output = []
    for _group, values in sorted(grouped.items()):
        for baseline in values:
            for candidate in values:
                if baseline["implementation"] == candidate["implementation"]:
                    continue
                output.append({
                    "experiment": experiment, "site": baseline["site"], "device": baseline["device"],
                    "events": baseline["events"], "threads": candidate.get("threads", ""),
                    "baseline": baseline["implementation"], "candidate": candidate["implementation"],
                    "speedup": baseline["mean_s"] / candidate["mean_s"],
                })
    return output


def strong_rows(rows):
    samples = training_samples(rows, "strong")
    cpp_by_site_event = {}
    one_thread = {}
    for (key, events), values in samples.items():
        if key[1] == "cpp":
            cpp_by_site_event[(key[0], events)] = mean_std(values)[0]
        elif key[3] == "cpu" and key[4].isdigit() and int(key[4]) == 1:
            one_thread[(key[0], impl_label(key), events)] = mean_std(values)[0]
    output = []
    for (key, events), values in sorted(samples.items()):
        if key[1] == "cpp" or key[3] != "cpu" or not key[4].isdigit():
            continue
        threads = int(key[4])
        mean, std = mean_std(values)
        reference = cpp_by_site_event.get((key[0], events))
        t1 = one_thread.get((key[0], impl_label(key), events))
        output.append({
            "site": key[0], "device": "cpu", "implementation": impl_label(key),
            "backend": key[1], "variant": key[2], "events": events, "threads": threads,
            "mean_s": mean, "std_s": std, "samples": len(values),
            "speedup_over_serial_cpp": reference / mean if reference else math.nan,
            "speedup_over_1thread": t1 / mean if t1 else math.nan,
            "parallel_efficiency": (t1 / mean / threads) if t1 else math.nan,
        })
    return output


def weak_rows(rows):
    samples = training_samples(rows, "weak")
    one_thread = {}
    for (key, events), values in samples.items():
        if key[3] == "cpu" and key[4].isdigit() and int(key[4]) == 1:
            one_thread[(key[0], impl_label(key), events)] = mean_std(values)[0]
    output = []
    for (key, events), values in sorted(samples.items()):
        if key[3] != "cpu" or not key[4].isdigit():
            continue
        threads = int(key[4])
        if threads <= 0 or events % threads:
            continue
        ratio = events // threads
        mean, std = mean_std(values)
        t1 = one_thread.get((key[0], impl_label(key), ratio))
        output.append({
            "site": key[0], "device": "cpu", "implementation": impl_label(key),
            "backend": key[1], "variant": key[2], "events": events,
            "events_per_thread": ratio, "threads": threads, "mean_s": mean,
            "std_s": std, "samples": len(values),
            "weak_efficiency": t1 / mean if t1 else math.nan,
            "runtime_growth_over_1thread": mean / t1 if t1 else math.nan,
        })
    return output


def stage_rows(rows):
    grouped = stage_samples(rows)
    selected = set(fixed_series(rows))
    output = []
    for (key, events), stages in sorted(grouped.items()):
        if key not in selected:
            continue
        means, stds = {}, {}
        for stage, _regions in STAGES:
            mean, std = mean_std(stages.get(stage, []))
            means[stage] = 0.0 if math.isnan(mean) else mean
            stds[stage] = 0.0 if math.isnan(std) else std
        total = sum(means.values())
        forward = sum(v for name, v in means.items() if name.startswith("F: "))
        backward = sum(v for name, v in means.items() if name.startswith("B: "))
        for stage, _regions in STAGES:
            mean = means[stage]
            output.append({
                "site": key[0], "device": key[3], "implementation": impl_label(key),
                "backend": key[1], "variant": key[2], "threads": key[4], "events": events,
                "direction": "forward" if stage.startswith("F: ") else "backward",
                "stage": stage.removeprefix("F: ").removeprefix("B: "),
                "mean_ms": mean, "std_ms": stds[stage],
                "fraction_pct": 100.0 * mean / total if total else math.nan,
                "profiled_total_ms": total, "forward_total_ms": forward, "backward_total_ms": backward,
                "forward_fraction_pct": 100.0 * forward / total if total else math.nan,
                "backward_fraction_pct": 100.0 * backward / total if total else math.nan,
            })
    return output


def stage_pairwise_rows(stages):
    grouped = defaultdict(list)
    for row in stages:
        grouped[(row["site"], row["device"], row["events"], row["direction"], row["stage"])].append(row)
    output = []
    for _group, values in sorted(grouped.items()):
        for baseline in values:
            for candidate in values:
                if baseline["implementation"] == candidate["implementation"] or candidate["mean_ms"] <= 0:
                    continue
                output.append({
                    "site": baseline["site"], "device": baseline["device"], "events": baseline["events"],
                    "direction": baseline["direction"], "stage": baseline["stage"],
                    "baseline": baseline["implementation"], "candidate": candidate["implementation"],
                    "speedup": baseline["mean_ms"] / candidate["mean_ms"],
                })
    return output


def python_code_lines(text):
    code = set()
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT,
               tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT}
    for token in tokenize.generate_tokens(StringIO(text).readline):
        if token.type not in ignored:
            code.update(range(token.start[0], token.end[0] + 1))
    return code


def cpp_code_lines(text):
    code = set()
    in_block = False
    for lineno, line in enumerate(text.splitlines(), 1):
        i, has_code, quote, escape = 0, False, None, False
        while i < len(line):
            ch = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if in_block:
                if ch == "*" and nxt == "/":
                    in_block = False
                    i += 2
                else:
                    i += 1
                continue
            if quote is not None:
                has_code = True
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                i += 1
                continue
            if ch in {'"', "'"}:
                quote, has_code = ch, True
                i += 1
                continue
            if ch == "/" and nxt == "*":
                in_block = True
                i += 2
                continue
            if ch == "/" and nxt == "/":
                break
            if not ch.isspace():
                has_code = True
            i += 1
        if has_code:
            code.add(lineno)
    return code


def code_lines(path):
    text = path.read_text()
    return python_code_lines(text) if path.suffix == ".py" else cpp_code_lines(text)


def count_files(repo, paths):
    total = 0
    for relative in paths:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"SLOC source file is missing: {relative}")
        total += len(code_lines(path))
    return total


def pytorch_sloc(repo):
    path = repo / "pytorch/loits.py"
    text = path.read_text()
    all_code = python_code_lines(text)
    tree = ast.parse(text)
    symbol_ranges = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbol_ranges[node.name] = set(range(node.lineno, node.end_lineno + 1))
    missing = [name for name in PYTORCH_CORE_SYMBOLS if name not in symbol_ranges]
    if missing:
        raise RuntimeError(f"PyTorch SLOC symbols missing: {', '.join(missing)}")
    core_lines = set()
    for name in PYTORCH_CORE_SYMBOLS:
        core_lines |= symbol_ranges[name]
    core_lines &= all_code
    interface_lines = all_code - core_lines
    return {
        "implementation": "PyTorch", "core_sloc": len(core_lines),
        "interface_sloc": len(interface_lines), "shared_sloc": 0,
        "total_sloc": len(all_code), "total_with_shared_sloc": len(all_code),
    }


def sloc_rows(repo):
    rows = [pytorch_sloc(repo)]
    for implementation, groups in SLOC_FILES.items():
        core = count_files(repo, groups["core"])
        interface = count_files(repo, groups["interface"])
        shared = count_files(repo, groups["shared"])
        rows.append({
            "implementation": implementation, "core_sloc": core, "interface_sloc": interface,
            "shared_sloc": shared, "total_sloc": core + interface,
            "total_with_shared_sloc": core + interface + shared,
        })
    torch_total = next(row["total_with_shared_sloc"] for row in rows if row["implementation"] == "PyTorch")
    torch_core = next(row["core_sloc"] for row in rows if row["implementation"] == "PyTorch")
    for row in rows:
        row["total_sloc_ratio_vs_pytorch"] = row["total_with_shared_sloc"] / torch_total
        row["core_sloc_ratio_vs_pytorch"] = row["core_sloc"] / torch_core
        row["interface_fraction_pct"] = 100.0 * row["interface_sloc"] / row["total_sloc"] if row["total_sloc"] else math.nan
    return rows


def inventory_rows(rows):
    grouped = defaultdict(lambda: {"paths": set(), "samples": 0})
    for row in rows:
        kind = "training" if is_training(row) else "regions" if is_regions(row) else "other"
        key = (row.get("site", ""), row.get("experiment", ""), row.get("backend", ""),
               row.get("implementation", ""), row.get("device", ""), str(row.get("threads", "")),
               row.get("events", ""), row.get("warmup", ""), row.get("iterations", ""),
               row.get("seed", ""), kind)
        grouped[key]["paths"].add(row["_path"])
        grouped[key]["samples"] += 1
    output = []
    for key, value in sorted(grouped.items()):
        output.append({
            "site": key[0], "experiment": key[1], "backend": key[2], "variant": key[3],
            "device": key[4], "threads": key[5], "events": key[6], "warmup": key[7],
            "iterations": key[8], "seed": key[9], "kind": key[10],
            "rows": value["samples"], "files": len(value["paths"]),
        })
    return output


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def summary_markdown(sloc, fixed, fixed_pairs, strong, weak, stages):
    lines = ["# JPDC paper metrics", "", "Speedup is always defined as baseline runtime / candidate runtime.", ""]
    lines += ["## SLOC", "", "| Implementation | Core | Interface | Shared | Total incl. shared | vs. PyTorch |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in sloc:
        lines.append(f"| {row['implementation']} | {row['core_sloc']} | {row['interface_sloc']} | {row['shared_sloc']} | {row['total_with_shared_sloc']} | {fmt(row['total_sloc_ratio_vs_pytorch'])}x |")
    lines += ["", "## Fixed-resource headline comparisons", ""]
    for site in PAPER_SITES:
        site_rows = [row for row in fixed if row["site"] == site]
        if not site_rows:
            continue
        max_events = max(row["events"] for row in site_rows)
        current = [row for row in site_rows if row["events"] == max_events]
        lines += [f"### {site} — {max_events:,} events", ""]
        for row in sorted(current, key=lambda item: item["mean_s"]):
            lines.append(f"- {row['implementation']}: {row['mean_s']:.6g} s (n={row['samples']})")
        pair_index = {(row["baseline"], row["candidate"]): row["speedup"] for row in fixed_pairs if row["site"] == site and row["events"] == max_events}
        if any(row["implementation"] == "Torch" for row in current):
            lines.append("- Torch relative to other implementations:")
            for row in sorted(current, key=lambda item: item["implementation"]):
                other = row["implementation"]
                if other != "Torch" and (other, "Torch") in pair_index:
                    lines.append(f"  - vs. {other}: {pair_index[(other, 'Torch')]:.3f}x")
        lines.append("")
    if strong:
        lines += ["## Strong scaling at maximum hardware-thread count", ""]
        for site in sorted({row["site"] for row in strong}):
            site_rows = [row for row in strong if row["site"] == site]
            max_threads = max(row["threads"] for row in site_rows)
            lines += [f"### {site} — {max_threads} hardware threads", ""]
            for row in site_rows:
                if row["threads"] == max_threads:
                    lines.append(f"- {row['implementation']}: {fmt(row['speedup_over_serial_cpp'])}x vs serial C++, {fmt(row['speedup_over_1thread'])}x self-speedup")
            lines.append("")
    if weak:
        lines += ["## Weak scaling at maximum hardware-thread count", ""]
        for site in sorted({row["site"] for row in weak}):
            site_rows = [row for row in weak if row["site"] == site]
            max_threads = max(row["threads"] for row in site_rows)
            lines += [f"### {site} — {max_threads} hardware threads", ""]
            for row in site_rows:
                if row["threads"] == max_threads:
                    lines.append(f"- {row['implementation']}: weak efficiency={fmt(row['weak_efficiency'])}, runtime growth={fmt(row['runtime_growth_over_1thread'])}x")
            lines.append("")
    if stages:
        lines += ["## Largest fixed-resource stage breakdown", ""]
        for site in PAPER_SITES:
            site_rows = [row for row in stages if row["site"] == site]
            if not site_rows:
                continue
            max_events = max(row["events"] for row in site_rows)
            current = [row for row in site_rows if row["events"] == max_events]
            lines += [f"### {site} — {max_events:,} events", ""]
            for implementation in sorted({row["implementation"] for row in current}):
                impl_rows = [row for row in current if row["implementation"] == implementation]
                dominant = max(impl_rows, key=lambda row: row["fraction_pct"])
                lines.append(f"- {implementation}: dominant stage = {dominant['direction']} {dominant['stage']} ({dominant['fraction_pct']:.1f}%)")
            lines.append("")
    return "\n".join(lines) + "\n"


def latex_macros(sloc, fixed, fixed_pairs):
    lines = ["% Generated by paper_metrics.py. Do not edit by hand."]
    for row in sloc:
        name = re.sub(r"[^A-Za-z]", "", row["implementation"])
        lines.append(rf"\newcommand{{\Sloc{name}Core}}{{{row['core_sloc']}}}")
        lines.append(rf"\newcommand{{\Sloc{name}Interface}}{{{row['interface_sloc']}}}")
        lines.append(rf"\newcommand{{\Sloc{name}Total}}{{{row['total_with_shared_sloc']}}}")
    for site in PAPER_SITES:
        site_rows = [row for row in fixed if row["site"] == site]
        if not site_rows:
            continue
        max_events = max(row["events"] for row in site_rows)
        site_name = re.sub(r"[^A-Za-z]", "", site.title())
        for row in fixed_pairs:
            if row["site"] == site and row["events"] == max_events and row["candidate"] == "Torch":
                baseline = re.sub(r"[^A-Za-z]", "", row["baseline"])
                lines.append(rf"\newcommand{{\{site_name}MaxTorchVs{baseline}}}{{{row['speedup']:.2f}}}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible JPDC paper metrics from benchmark CSVs and source.")
    parser.add_argument("--input", default="results/training")
    parser.add_argument("--output", default="results/paper_metrics")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--sites", nargs="+", default=list(PAPER_SITES))
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    output = Path(args.output)
    rows = load_rows(args.input, set(args.sites))
    sloc = sloc_rows(repo)
    fixed = runtime_rows(rows)
    fixed_pairs = pairwise_rows(fixed, "fixed")
    strong = strong_rows(rows)
    weak = weak_rows(rows)
    strong_pairs = pairwise_rows(strong, "strong")
    weak_pairs = pairwise_rows(weak, "weak")
    stages = stage_rows(rows)
    stage_pairs = stage_pairwise_rows(stages)
    inventory = inventory_rows(rows)

    write_csv(output / "sloc.csv", sloc)
    write_csv(output / "fixed_runtime.csv", fixed)
    write_csv(output / "fixed_speedups.csv", fixed_pairs)
    write_csv(output / "strong_scaling.csv", strong)
    write_csv(output / "strong_speedups.csv", strong_pairs)
    write_csv(output / "weak_scaling.csv", weak)
    write_csv(output / "weak_speedups.csv", weak_pairs)
    write_csv(output / "stage_breakdown.csv", stages)
    write_csv(output / "stage_speedups.csv", stage_pairs)
    write_csv(output / "experiment_inventory.csv", inventory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text(summary_markdown(sloc, fixed, fixed_pairs, strong, weak, stages))
    (output / "paper_metrics.tex").write_text(latex_macros(sloc, fixed, fixed_pairs))

    for name in (
        "summary.md", "sloc.csv", "fixed_runtime.csv", "fixed_speedups.csv",
        "strong_scaling.csv", "strong_speedups.csv", "weak_scaling.csv",
        "weak_speedups.csv", "stage_breakdown.csv", "stage_speedups.csv",
        "experiment_inventory.csv", "paper_metrics.tex",
    ):
        path = output / name
        if path.exists():
            print(path)


if __name__ == "__main__":
    main()
