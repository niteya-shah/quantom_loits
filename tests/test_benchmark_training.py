from types import SimpleNamespace

from benchmark_training import result_directory, result_stem, row_metadata


def _args(**overrides):
    values = {
        "output": "results/training",
        "site": "odyssey",
        "experiment": "fixed",
        "backend": "sycl",
        "device": "cuda",
        "events": 100000,
        "grid_size": 100,
        "warmup": 5,
        "iterations": 20,
        "seed": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_result_path_and_stem():
    args = _args()
    assert str(result_directory(args)) == "results/training/odyssey/fixed"
    assert result_stem("training", "acpp-odyssey-mi300a", "cuda", 100000) == (
        "training_acpp-odyssey-mi300a_cuda_e100000"
    )


def test_metadata_contains_reproducibility_fields():
    args = _args()
    metadata = row_metadata(args, "dpcpp-odyssey-mi300a", "")
    assert metadata["site"] == "odyssey"
    assert metadata["experiment"] == "fixed"
    assert metadata["grid_size"] == 100
    assert metadata["warmup"] == 5
    assert metadata["iterations"] == 20
    assert metadata["seed"] == 0
