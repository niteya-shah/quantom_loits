import csv

from plotting.common import load_rows, percentile, series_label


def test_plotting_loader_accepts_current_and_older_csv_schema(tmp_path):
    path = tmp_path / "regions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["backend", "device", "events", "region", "occurrence", "cpu_ms", "device_ms"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "backend": "cpp",
                "device": "cpu",
                "events": 100000,
                "region": "loits::forward",
                "occurrence": 0,
                "cpu_ms": 5.0,
                "device_ms": 0.0,
            }
        )

    rows = load_rows([path])
    assert rows[0]["events"] == 100000
    assert rows[0]["cpu_ms"] == 5.0
    assert series_label(rows[0]) == "cpp"
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
