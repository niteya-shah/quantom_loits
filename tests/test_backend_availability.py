from pathlib import Path

import pytest

import sycl.backend as sycl_backend
from loits import BackendUnavailableError, LOITS, backend_status


def make_fake_sycl_build(root, variant, device="cpu", toolchain="acpp", target="cpu"):
    build = root / "build" / variant
    build.mkdir(parents=True)
    (build / "libquantom_loits_sycl.so").touch()
    metadata = {
        "toolchain": toolchain,
        "target": target,
        "torch_device": device,
        "architecture": None,
    }
    (build / "variant.py").write_text(f"METADATA = {metadata!r}\n")
    return build


def test_sycl_requires_explicit_variant_even_when_one_build_exists(monkeypatch, tmp_path):
    root = tmp_path / "sycl"
    make_fake_sycl_build(root, "acpp-cpu")
    monkeypatch.setattr(sycl_backend, "_root", lambda: root)
    monkeypatch.delenv("QUANTOM_SYCL_VARIANT", raising=False)

    ok, reason = backend_status("sycl", "cpu")
    assert not ok
    assert "QUANTOM_SYCL_VARIANT must be set explicitly" in reason
    assert "acpp-cpu" in reason

    with pytest.raises(BackendUnavailableError, match="QUANTOM_SYCL_VARIANT must be set explicitly"):
        LOITS(backend="sycl", device="cpu")


def test_selected_unbuilt_sycl_variant_is_reported(monkeypatch, tmp_path):
    root = tmp_path / "sycl"
    make_fake_sycl_build(root, "acpp-cpu")
    monkeypatch.setattr(sycl_backend, "_root", lambda: root)
    monkeypatch.setenv("QUANTOM_SYCL_VARIANT", "missing")

    ok, reason = backend_status("sycl", "cpu")
    assert not ok
    assert "selected SYCL variant 'missing' is not built" in reason
    assert "acpp-cpu" in reason


def test_selected_sycl_variant_uses_its_device_marker(monkeypatch, tmp_path):
    root = tmp_path / "sycl"
    make_fake_sycl_build(root, "acpp-cpu", device="cpu")
    make_fake_sycl_build(root, "dpcpp-xpu", device="xpu", toolchain="dpcpp", target="xpu")
    monkeypatch.setattr(sycl_backend, "_root", lambda: root)

    assert sycl_backend.built_variants() == ("acpp-cpu", "dpcpp-xpu")

    monkeypatch.setenv("QUANTOM_SYCL_VARIANT", "acpp-cpu")
    assert backend_status("sycl", "cpu")[0]
    ok, reason = backend_status("sycl", "xpu")
    assert not ok
    assert "expects torch device 'cpu'" in reason


def test_cpu_backend_device_constraints_are_reported():
    assert backend_status("cpp", "cpu")[0]
    assert backend_status("openmp", "cpu")[0]
    assert not backend_status("cpp", "cuda")[0]
    assert not backend_status("openmp", "xpu")[0]
