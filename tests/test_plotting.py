import csv

from plotting.common import (
    fixed_resource_events,
    load_rows,
    mean_std,
    per_iteration_stage_samples,
    series_key,
)


def _write(path, rows):
    fieldnames = [
        "backend",
        "implementation",
        "device",
        "events",
        "threads",
        "region",
        "occurrence",
        "event_id",
        "parent_event_id",
        "parent_region",
        "cpu_ms",
        "device_ms",
        "self_cpu_ms",
        "self_device_ms",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_plotting_loader_and_mean_std(tmp_path):
    path = tmp_path / "training_cpp_cpu_100000.csv"
    _write(
        path,
        [
            {
                "backend": "cpp",
                "implementation": "cpp",
                "device": "cpu",
                "events": 100000,
                "threads": "",
                "region": "gan::training_iteration",
                "occurrence": 0,
                "event_id": 1,
                "parent_event_id": "",
                "parent_region": "",
                "cpu_ms": 5.0,
                "device_ms": 0.0,
                "self_cpu_ms": 0.0,
                "self_device_ms": 0.0,
            }
        ],
    )
    rows = load_rows([path])
    assert rows[0]["events"] == 100000
    assert rows[0]["cpu_ms"] == 5.0
    assert series_key(rows[0])[0] == "cpp"
    assert mean_std([1.0, 2.0, 3.0])[0] == 2.0


def test_forward_and_backward_are_accumulated_per_training_iteration(tmp_path):
    path = tmp_path / "regions_cpp_cpu_100000.csv"
    base = {
        "backend": "cpp",
        "implementation": "cpp",
        "device": "cpu",
        "events": 100000,
        "threads": "",
        "device_ms": 0.0,
        "self_cpu_ms": 0.0,
        "self_device_ms": 0.0,
    }
    rows = [
        base | {"region": "gan::training_iteration", "occurrence": 0, "event_id": 1, "parent_event_id": "", "parent_region": "", "cpu_ms": 100.0},
        base | {"region": "gan::discriminator_step", "occurrence": 0, "event_id": 2, "parent_event_id": 1, "parent_region": "gan::training_iteration", "cpu_ms": 60.0},
        base | {"region": "loits::forward::random_x", "occurrence": 0, "event_id": 3, "parent_event_id": 2, "parent_region": "gan::discriminator_step", "cpu_ms": 1.0},
        base | {"region": "loits::forward::random_q2", "occurrence": 0, "event_id": 4, "parent_event_id": 2, "parent_region": "gan::discriminator_step", "cpu_ms": 2.0},
        base | {"region": "gan::generator_step", "occurrence": 0, "event_id": 5, "parent_event_id": 1, "parent_region": "gan::training_iteration", "cpu_ms": 40.0},
        base | {"region": "loits::forward::random_x", "occurrence": 1, "event_id": 6, "parent_event_id": 5, "parent_region": "gan::generator_step", "cpu_ms": 1.5},
        base | {"region": "loits::forward::random_q2", "occurrence": 1, "event_id": 7, "parent_event_id": 5, "parent_region": "gan::generator_step", "cpu_ms": 2.5},
        base | {"region": "loits::backward::interpolation_x", "occurrence": 0, "event_id": 8, "parent_event_id": 5, "parent_region": "gan::generator_step", "cpu_ms": 3.0},
        base | {"region": "loits::backward::interpolation_q2", "occurrence": 0, "event_id": 9, "parent_event_id": 5, "parent_region": "gan::generator_step", "cpu_ms": 4.0},
    ]
    _write(path, rows)
    loaded = load_rows([path])
    samples = per_iteration_stage_samples(loaded)
    key = next(iter(samples))
    data = samples[key]
    assert data["F: RNG"] == [7.0]
    assert data["B: Interpolation VJP"] == [7.0]
    assert data["total"] == [14.0]


def test_fixed_resource_events_exclude_strong_and_weak_scaling_sizes():
    cpp = ("cpp", "cpp", "cpu", "")
    omp = ("openmp", "openmp", "cpu", "64")
    series = [cpp, omp]

    # 10k/100k/1M are the fixed-resource experiment.  The OpenMP-only sizes
    # are weak-scaling points sharing the same results directory.
    fixed = [10_000, 100_000, 1_000_000]
    weak_only = [20_000, 40_000, 80_000, 160_000, 320_000, 640_000]

    stage_samples = {}
    training = {}
    for events in fixed:
        stage_samples[(cpp, events)] = {"total": [1.0]}
        training[(cpp, events)] = [1.0]
        stage_samples[(omp, events)] = {"total": [1.0]}
        training[(omp, events)] = [1.0]

    for events in weak_only:
        stage_samples[(omp, events)] = {"total": [1.0]}
        training[(omp, events)] = [1.0]

    assert fixed_resource_events(series, stage_samples, training) == fixed
