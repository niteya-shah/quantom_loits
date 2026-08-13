import csv
import math
import re
from collections import defaultdict
from pathlib import Path


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
                for key in ("cpu_ms", "device_ms", "self_cpu_ms", "self_device_ms"):
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                rows.append(row)
    return rows


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize(rows, keys, metric="cpu_ms"):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        groups[tuple(row.get(key, "") for key in keys)].append(float(value))

    result = []
    for key_values, values in sorted(groups.items(), key=lambda item: item[0]):
        record = dict(zip(keys, key_values))
        record.update(
            {
                "count": len(values),
                "median": percentile(values, 0.50),
                "q1": percentile(values, 0.25),
                "q3": percentile(values, 0.75),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        )
        result.append(record)
    return result


def implementation(row):
    return row.get("implementation") or row.get("backend") or "unknown"


def series_label(row):
    impl = implementation(row)
    device = row.get("device", "")
    threads = row.get("threads", "")
    label = impl
    if device and device != "cpu":
        label += f"/{device}"
    if threads:
        label += f"/{threads}t"
    return label


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "plot"


def metric_value(row, requested="auto"):
    if requested == "auto":
        device = float(row.get("device_ms") or 0.0)
        return device if device > 0.0 else float(row.get("cpu_ms") or 0.0)
    return float(row.get(requested) or 0.0)


def require_rows(rows, description):
    if rows:
        return
    raise SystemExit(f"no matching benchmark rows found for {description}")
