from pathlib import Path

import pytest

import sycl.backend as sycl_backend
from loits import BackendUnavailableError, LOITS, backend_status


def test_unbuilt_sycl_is_reported_without_loading_extension(monkeypatch, tmp_path):
    missing = tmp_path / "missing" / "libquantom_loits_sycl.so"
    monkeypatch.setattr(sycl_backend, "core_library_path", lambda: missing)

    ok, reason = backend_status("sycl", "cpu")
    assert not ok
    assert "not built" in reason

    with pytest.raises(BackendUnavailableError, match="SYCL core is not built"):
        LOITS(backend="sycl", device="cpu")


def test_cpu_backend_device_constraints_are_reported():
    assert backend_status("cpp", "cpu")[0]
    assert backend_status("openmp", "cpu")[0]
    assert not backend_status("cpp", "cuda")[0]
    assert not backend_status("openmp", "xpu")[0]
