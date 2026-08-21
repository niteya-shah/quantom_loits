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
    integer_fields = ("events", "grid_size", "warmup", "iterations", "seed")
    float_fields = (
        "wall_ms",
        "cpu_ms",
        "device_ms",
        "self_cpu_ms",
        "self_device_ms",
        "start_us",
        "end_us",
    )
    for path in csv_paths(inputs):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["_path"] = str(path)
                for key in integer_fields:
                    if row.get(key) not in (None, ""):
                        row[key] = int(row[key])
                if row.get("occurrence") not in (None, ""):
                    row["occurrence"] = int(row["occurrence"])
                for key in float_fields:
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


def metric_value(row, related_rows=None):
    related_rows = tuple(related_rows or (row,))
    device_values = [
        float(candidate.get("device_ms") or 0.0)
        for candidate in related_rows
        if float(candidate.get("device_ms") or 0.0) > 0.0
    ]
    if device_values:
        # torch.profiler can emit both the nested CPU annotation and a
        # parentless device-side row for the same logical event_id. Use the
        # device activity when it exists so host synchronization inside a
        # record_function scope is not charged to the GPU kernel stage. Use
        # max rather than sum because these rows are duplicate views of the
        # same logical profiler event.
        return max(device_values)
    return float(row.get("cpu_ms") or 0.0)


def is_training_file(row):
    return Path(row.get("_path", "")).name.startswith("training_")


def is_regions_file(row):
    return Path(row.get("_path", "")).name.startswith("regions_")


def rows_for_experiment(rows, experiment):
    return [row for row in rows if row.get("experiment") == experiment]


def rows_for_site(rows, site):
    return [row for row in rows if row.get("site") == site]


def available_sites(rows, *, cpu=None, experiment=None):
    subset = rows
    if experiment is not None:
        subset = rows_for_experiment(subset, experiment)
    if cpu is not None:
        subset = rows_for_device(subset, cpu)
    return sorted({row.get("site", "") for row in subset if row.get("site")})


def series_key(row, include_threads=True):
    backend = row.get("backend", "")
    return (
        row.get("site", ""),
        backend,
        row.get("implementation") or backend or "unknown",
        row.get("device", ""),
        row.get("threads", "") if include_threads else "",
    )


def series_label(key, show_threads=True):
    site, backend, implementation, device, threads = key
    if backend == "cpp":
        label = "C++"
    elif backend == "openmp":
        label = "C++\n(OpenMP)"
    elif backend == "torch":
        dev = {"cpu": "CPU", "cuda": "CUDA", "xpu": "XPU"}.get(device, device.upper())
        label = f"PyTorch\n({dev})"
        if device != "cpu" and site:
            label += f"\n{site}"
    elif backend == "triton":
        dev = {"cuda": "CUDA", "xpu": "XPU"}.get(device, device.upper())
        label = f"Triton\n({dev})"
        if site:
            label += f"\n{site}"
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
        if not event_id:
            continue
        key = (row.get("_path", ""), event_id)
        current = by_id.get(key)
        if current is None or (
            not current.get("parent_event_id") and row.get("parent_event_id")
        ):
            by_id[key] = row
    return by_id


def _training_intervals(rows):
    intervals = defaultdict(list)
    for row in rows:
        if row.get("region") != "gan::training_iteration":
            continue
        start = row.get("start_us")
        end = row.get("end_us")
        if start in (None, "") or end in (None, ""):
            continue
        root = (
            row.get("_path", ""),
            str(row.get("event_id") or row.get("occurrence") or "0"),
        )
        intervals[row.get("_path", "")].append((float(start), float(end), root))
    return intervals


def _enclosing_training_root(row, intervals):
    start = row.get("start_us")
    end = row.get("end_us")
    if start in (None, "") or end in (None, ""):
        return None
    candidates = [
        interval
        for interval in intervals.get(row.get("_path", ""), ())
        if interval[0] <= float(start) and float(end) <= interval[1]
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda interval: interval[1] - interval[0])[2]


def training_root(row, by_id, intervals=None):
    if row.get("region") == "gan::training_iteration":
        return (row.get("_path", ""), str(row.get("event_id") or row.get("occurrence") or "0"))

    current = row
    visited = set()
    while current is not None:
        parent_id = str(current.get("parent_event_id") or "")
        if not parent_id:
            break
        key = (current.get("_path", ""), parent_id)
        if key in visited:
            break
        visited.add(key)
        parent = by_id.get(key)
        if parent is None:
            break
        if parent.get("region") == "gan::training_iteration":
            return key
        current = parent
    return _enclosing_training_root(row, intervals or {})


def per_iteration_stage_samples(rows):
    rows = [row for row in rows if is_regions_file(row)]
    by_id = _event_maps(rows)
    event_rows = defaultdict(list)
    for row in rows:
        event_id = str(row.get("event_id") or "")
        if event_id:
            event_rows[(row.get("_path", ""), event_id)].append(row)

    intervals = _training_intervals(rows)
    region_to_stage = {
        region: stage
        for stage, regions in STAGES
        for region in regions
    }

    by_iteration = defaultdict(lambda: defaultdict(float))
    for row in rows:
        stage = region_to_stage.get(row.get("region"))
        if stage is None:
            continue
        event_id = str(row.get("event_id") or "")
        if event_id and by_id.get((row.get("_path", ""), event_id)) is not row:
            continue
        root = training_root(row, by_id, intervals)
        if root is None:
            continue
        key = (series_key(row), row["events"], root)
        related = event_rows.get((row.get("_path", ""), event_id), (row,))
        by_iteration[key][stage] += metric_value(row, related)

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
        if not is_training_file(row) or row.get("region") != "gan::training_iteration":
            continue
        wall_ms = row.get("wall_ms")
        if wall_ms in (None, ""):
            continue
        grouped[(series_key(row), row["events"])].append(float(wall_ms) / 1000.0)
    return grouped


def fixed_resource_series(rows, cpu):
    subset = rows_for_device(rows_for_experiment(rows, "fixed"), cpu)
    grouped = defaultdict(set)
    for row in subset:
        full = series_key(row)
        base = series_key(row, include_threads=False)
        grouped[base].add(full)

    selected = []
    for candidates in grouped.values():
        def rank(key):
            threads = str(key[4])
            return (1, int(threads)) if threads.isdigit() else (0, 0)

        selected.append(max(candidates, key=rank))
    return sorted(selected, key=lambda key: (key[0], key[1], key[2], key[3]))


def available_series(rows, cpu, experiment=None):
    subset = rows_for_device(rows, cpu)
    if experiment is not None:
        subset = rows_for_experiment(subset, experiment)
    keys = {series_key(row) for row in subset}
    return sorted(
        keys,
        key=lambda key: (
            key[0], key[1], key[2], key[3],
            int(key[4]) if str(key[4]).isdigit() else 0,
        ),
    )


def available_events(rows, cpu, experiment=None):
    subset = rows_for_device(rows, cpu)
    if experiment is not None:
        subset = rows_for_experiment(subset, experiment)
    return sorted({row["events"] for row in subset if row.get("events")})


def fixed_resource_events(series, stage_samples, training):
    if not series:
        return []

    common = None
    for key in series:
        stage_events = {events for (sample_key, events) in stage_samples if sample_key == key}
        training_events = {events for (sample_key, events) in training if sample_key == key}
        complete = stage_events & training_events
        common = complete if common is None else common & complete
    return sorted(common or ())
