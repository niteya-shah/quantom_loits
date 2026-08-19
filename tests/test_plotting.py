import csv

from plotting.common import (
    fixed_resource_events,
    load_rows,
    mean_std,
    per_iteration_stage_samples,
    series_key,
    training_samples,
)


def _write(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base(**overrides):
    row = {
        "site": "test",
        "experiment": "fixed",
        "backend": "cpp",
        "implementation": "cpp",
        "device": "cpu",
        "events": 100000,
        "threads": "",
        "grid_size": 100,
        "warmup": 5,
        "iterations": 20,
        "seed": 0,
        "vjp_case": "",
        "compact_case": "",
    }
    row.update(overrides)
    return row


def test_plotting_loader_and_wall_clock_samples(tmp_path):
    path = tmp_path / "training_cpp_cpu_e100000.csv"
    _write(path, [_base(region="gan::training_iteration", occurrence=0, wall_ms=5.0)])
    rows = load_rows([path])
    assert rows[0]["events"] == 100000
    assert rows[0]["wall_ms"] == 5.0
    assert series_key(rows[0]) == ("test", "cpp", "cpp", "cpu", "")
    assert training_samples(rows)[(series_key(rows[0]), 100000)] == [0.005]
    assert mean_std([1.0, 2.0, 3.0])[0] == 2.0


def test_forward_and_backward_are_accumulated_per_training_iteration(tmp_path):
    path = tmp_path / "regions_cpp_cpu_e100000.csv"
    base = _base(device_ms=0.0, self_cpu_ms=0.0, self_device_ms=0.0)
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
    data = next(iter(per_iteration_stage_samples(load_rows([path])).values()))
    assert data["F: RNG"] == [7.0]
    assert data["B: Interpolation VJP"] == [7.0]
    assert data["total"] == [14.0]


def test_fixed_resource_events_use_complete_series():
    cpp = ("test", "cpp", "cpp", "cpu", "")
    omp = ("test", "openmp", "openmp", "cpu", "64")
    series = [cpp, omp]
    stage_samples = {}
    training = {}
    for events in [10_000, 100_000, 1_000_000]:
        stage_samples[(cpp, events)] = {"total": [1.0]}
        training[(cpp, events)] = [1.0]
        stage_samples[(omp, events)] = {"total": [1.0]}
        training[(omp, events)] = [1.0]
    stage_samples[(omp, 20_000)] = {"total": [1.0]}
    training[(omp, 20_000)] = [1.0]
    assert fixed_resource_events(series, stage_samples, training) == [10_000, 100_000, 1_000_000]


def test_duplicate_profiler_event_ids_prefer_nested_row(tmp_path):
    path = tmp_path / "regions_acpp-odyssey-mi300a_cuda_e100000.csv"
    base = _base(
        site="odyssey",
        backend="sycl",
        implementation="acpp-odyssey-mi300a",
        device="cuda",
        vjp_case=4,
        compact_case=4,
        device_ms=0.0,
        self_cpu_ms=0.0,
        self_device_ms=0.0,
    )
    rows = [
        base | {"region": "gan::training_iteration", "occurrence": 0, "event_id": 1, "parent_event_id": "", "parent_region": "", "start_us": 0.0, "end_us": 100.0, "cpu_ms": 100.0},
        base | {"region": "gan::generator_step", "occurrence": 0, "event_id": 2, "parent_event_id": 1, "parent_region": "gan::training_iteration", "start_us": 10.0, "end_us": 90.0, "cpu_ms": 80.0},
        base | {"region": "loits::forward", "occurrence": 0, "event_id": 3, "parent_event_id": 2, "parent_region": "gan::generator_step", "start_us": 20.0, "end_us": 40.0, "cpu_ms": 20.0},
        base | {"region": "loits::forward::random_x", "occurrence": 0, "event_id": 4, "parent_event_id": 3, "parent_region": "loits::forward", "start_us": 25.0, "end_us": 30.0, "cpu_ms": 1.0},
        base | {"region": "loits::forward::random_x", "occurrence": 1, "event_id": 4, "parent_event_id": "", "parent_region": "", "start_us": 25.0, "end_us": 30.0, "cpu_ms": 99.0},
    ]
    _write(path, rows)
    data = next(iter(per_iteration_stage_samples(load_rows([path])).values()))
    assert data["F: RNG"] == [1.0]


def test_parentless_native_backward_uses_training_interval(tmp_path):
    path = tmp_path / "regions_dpcpp-aurora-pvc_xpu_e100000.csv"
    base = _base(
        site="aurora",
        backend="sycl",
        implementation="dpcpp-aurora-pvc",
        device="xpu",
        vjp_case=4,
        compact_case=4,
        device_ms=0.0,
        self_cpu_ms=0.0,
        self_device_ms=0.0,
    )
    rows = [
        base | {"region": "gan::training_iteration", "occurrence": 0, "event_id": 1, "parent_event_id": "", "parent_region": "", "start_us": 0.0, "end_us": 100.0, "cpu_ms": 100.0},
        base | {"region": "loits::autograd::backward", "occurrence": 0, "event_id": 2, "parent_event_id": "", "parent_region": "", "start_us": 60.0, "end_us": 90.0, "cpu_ms": 30.0},
        base | {"region": "loits::backward", "occurrence": 0, "event_id": 3, "parent_event_id": 2, "parent_region": "loits::autograd::backward", "start_us": 65.0, "end_us": 85.0, "cpu_ms": 20.0},
        base | {"region": "loits::backward::interpolation_x", "occurrence": 0, "event_id": 4, "parent_event_id": 3, "parent_region": "loits::backward", "start_us": 70.0, "end_us": 75.0, "cpu_ms": 3.0},
    ]
    _write(path, rows)
    data = next(iter(per_iteration_stage_samples(load_rows([path])).values()))
    assert data["B: Interpolation VJP"] == [3.0]


def test_site_is_metadata_not_filename(tmp_path):
    base = _base(
        backend="torch",
        implementation="torch",
        device="cuda",
        region="gan::training_iteration",
        occurrence=0,
        wall_ms=1.0,
    )
    odyssey = tmp_path / "training_torch_cuda_e100000.csv"
    illyad_dir = tmp_path / "other"
    illyad_dir.mkdir()
    illyad = illyad_dir / "training_torch_cuda_e100000.csv"
    _write(odyssey, [base | {"site": "odyssey"}])
    _write(illyad, [base | {"site": "illyad"}])
    keys = {series_key(row) for row in load_rows([tmp_path])}
    assert ("odyssey", "torch", "torch", "cuda", "") in keys
    assert ("illyad", "torch", "torch", "cuda", "") in keys
