import csv
import math
import re
from collections import defaultdict
from pathlib import Path


FORWARD_STAGES = [
    ("F: Allocation", {"loits::forward::allocation"}),
    ("F: Density", {"loits::forward::rho_x", "loits::forward::rho_q2"}),
    ("F: CDF", {"loits::forward::cdf_x", "loits::forward::cdf_q2"}),
    ("F: Flatten", {"loits::forward::flatten_x", "loits::forward::flatten_q2"}),
    ("F: RNG", {"loits::forward::random_x", "loits::forward::random_q2"}),
    (
        "F: Interpolation",
        {"loits::forward::interpolation_x", "loits::forward::interpolation_q2"},
    ),
    ("F: Compaction", {"loits::forward::stream_compaction"}),
]

BACKWARD_STAGES = [
    (
        "B: Interpolation VJP",
        {
            "loits::backward::stream_compaction",
            "loits::backward::interpolation_x",
            "loits::backward::interpolation_q2",
        },
    ),
    ("B: Flatten", {"loits::backward::flatten_x", "loits::backward::flatten_q2"}),
    ("B: CDF VJP", {"loits::backward::cdf_x", "loits::backward::cdf_q2"}),
    ("B: Density VJP", {"loits::backward::rho_x", "loits::backward::rho_q2"}),
]

STAGES = FORWARD_STAGES + BACKWARD_STAGES

EVENT_HATCHES = ("", "////", "+++", "..", "\\\\", "xx", "oo", "**")


def csv_paths(inputs):
    paths = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.csv")))
        elif path.is_file() and path.suffix == ".csv":
            paths.append(path)
    return sorted(set(paths))


def load_rows(inputs):
    rows = []
    for path in csv_paths(inputs):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["_path"] = str(path)
                if row.get("events"):
                    row["events"] = int(row["events"])
                if row.get("occurrence") not in (None, ""):
                    row["occurrence"] = int(row["occurrence"])
                for key in ("cpu_ms", "device_ms", "self_cpu_ms", "self_device_ms"):
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                rows.append(row)
    return rows


def mean_std(values):
    values = [float(value) for value in values]
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def metric_value(row):
    device = float(row.get("device_ms") or 0.0)
    if device > 0.0:
        return device
    return float(row.get("cpu_ms") or 0.0)


def is_training_file(row):
    return Path(row.get("_path", "")).name.startswith("training_")


def is_regions_file(row):
    return Path(row.get("_path", "")).name.startswith("regions_")


def series_key(row, include_threads=True):
    return (
        row.get("backend", ""),
        row.get("implementation") or row.get("backend", "unknown"),
        row.get("device", ""),
        row.get("threads", "") if include_threads else "",
    )


def series_label(key, show_threads=True):
    backend, implementation, device, threads = key
    if backend == "cpp":
        label = "C++"
    elif backend == "openmp":
        label = "C++\n(OpenMP)"
    elif backend == "torch":
        dev = {"cpu": "CPU", "cuda": "CUDA", "xpu": "XPU"}.get(device, device.upper())
        label = f"PyTorch\n({dev})"
    elif backend == "sycl":
        label = f"SYCL\n{implementation}"
    else:
        label = implementation

    if threads and show_threads:
        label += f"\n{threads} threads"
    return label


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "plot"


def event_hatch(events, all_events):
    index = list(sorted(all_events)).index(events)
    return EVENT_HATCHES[index % len(EVENT_HATCHES)]


def rows_for_device(rows, cpu):
    if cpu:
        return [row for row in rows if row.get("device") == "cpu"]
    return [row for row in rows if row.get("device") != "cpu"]


def _event_maps(rows):
    by_id = {}
    for row in rows:
        event_id = str(row.get("event_id") or "")
        if event_id:
            by_id[(row.get("_path", ""), event_id)] = row
    return by_id


def training_root(row, by_id):
    if row.get("region") == "gan::training_iteration":
        return (row.get("_path", ""), str(row.get("event_id") or row.get("occurrence") or "0"))

    current = row
    visited = set()
    while current is not None:
        parent_id = str(current.get("parent_event_id") or "")
        if not parent_id:
            return None
        key = (current.get("_path", ""), parent_id)
        if key in visited:
            return None
        visited.add(key)
        parent = by_id.get(key)
        if parent is None:
            return None
        if parent.get("region") == "gan::training_iteration":
            return key
        current = parent
    return None


def per_iteration_stage_samples(rows):
    """Return {(series_key, events): {stage: [per-iteration ms], total: [...]}}.

    Only semantic leaf regions are summed. This makes the breakdown common to
    PyTorch, C++, OpenMP, and SYCL without double-counting nested profiler ranges.
    Two LOITS forwards and the generator backward naturally contribute to the
    same GAN-iteration sample through the profiler parent hierarchy.
    """
    rows = [row for row in rows if is_regions_file(row)]
    by_id = _event_maps(rows)
    stage_regions = {name: regions for name, regions in STAGES}
    region_to_stage = {}
    for stage, regions in stage_regions.items():
        for region in regions:
            region_to_stage[region] = stage

    by_iteration = defaultdict(lambda: defaultdict(float))
    for row in rows:
        stage = region_to_stage.get(row.get("region"))
        if stage is None:
            continue
        root = training_root(row, by_id)
        if root is None:
            continue
        key = (series_key(row), row["events"], root)
        by_iteration[key][stage] += metric_value(row)

    grouped = defaultdict(lambda: defaultdict(list))
    for (series, events, _root), stage_values in by_iteration.items():
        total = 0.0
        for stage, _regions in STAGES:
            value = stage_values.get(stage, 0.0)
            grouped[(series, events)][stage].append(value)
            total += value
        grouped[(series, events)]["total"].append(total)
    return grouped


def training_samples(rows):
    grouped = defaultdict(list)
    for row in rows:
        if not is_training_file(row):
            continue
        if row.get("region") != "gan::training_iteration":
            continue
        grouped[(series_key(row), row["events"])].append(float(row["cpu_ms"]) / 1000.0)
    return grouped



def fixed_resource_series(rows, cpu):
    """Select one fixed-resource series per implementation.

    If multiple explicit thread-count builds are present in the same results
    directory, the largest thread count is used for the fixed-resource plot.
    The complete set remains available to the strong/weak scaling plots.
    """
    subset = rows_for_device(rows, cpu)
    grouped = defaultdict(set)
    for row in subset:
        full = series_key(row)
        base = series_key(row, include_threads=False)
        grouped[base].add(full)

    selected = []
    for base, candidates in grouped.items():
        def rank(key):
            threads = str(key[3])
            return (1, int(threads)) if threads.isdigit() else (0, 0)

        selected.append(max(candidates, key=rank))
    return sorted(selected, key=lambda key: (key[0], key[1], key[2]))

def available_series(rows, cpu):
    subset = rows_for_device(rows, cpu)
    keys = {series_key(row) for row in subset}
    return sorted(keys, key=lambda key: (key[0], key[1], key[2], int(key[3]) if str(key[3]).isdigit() else 0))


def available_events(rows, cpu):
    return sorted({row["events"] for row in rows_for_device(rows, cpu) if row.get("events")})


def fixed_resource_events(series, stage_samples, training):
    """Return event counts shared by every selected fixed-resource series.

    Strong- and weak-scaling runs live in the same results directory as the
    fixed-resource experiment.  Their event counts must not leak into the
    fixed-resource figure.  A fixed-resource event size is therefore one for
    which every selected implementation has both detailed region samples and
    end-to-end training samples at its selected resource level.
    """
    if not series:
        return []

    common = None
    for key in series:
        stage_events = {events for (sample_key, events) in stage_samples if sample_key == key}
        training_events = {events for (sample_key, events) in training if sample_key == key}
        complete = stage_events & training_events
        common = complete if common is None else common & complete

    return sorted(common or ())
