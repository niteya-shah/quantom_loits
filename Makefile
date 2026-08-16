SHELL := /usr/bin/env bash

PYTHON ?= python
BACKEND ?= torch
DEVICE ?= cpu
EVENTS ?= 100000
GRID_SIZE ?= 10
WARMUP ?= 5
ITERATIONS ?= 10
SEED ?= 0
SYCL_VARIANT ?=
SYCL_TARGET ?=
SYCL_ARCH ?=
LLVM_PREFIX ?=
LLVM_TARGETS ?=
LLVM_VERSION ?=
LLVM_JOBS ?= 4
ACPP_PREFIX ?=
ACPP_REF ?=
ACPP_JOBS ?= 4
DPCPP_PREFIX ?=
DPCPP_TARGETS ?=
DPCPP_REF ?=
DPCPP_JOBS ?= 4

.PHONY: all \
        build build-all build-cpp build-openmp \
        install-llvm install-acpp install-dpcpp build-sycl-acpp build-sycl-dpcpp list-sycl-builds \
        test test-pytorch test-cpp test-openmp test-sycl test-rng \
        benchmark profile list-backends compile-check \
        plots plot-cpu plot-gpu plot-ss plot-ws \
        clean clean-cpp clean-openmp clean-sycl clean-results

all: test

# C++ and OpenMP use torch.utils.cpp_extension and build on demand.
build: build-cpp build-openmp

# SYCL is never selected or built implicitly. Every SYCL build requires an
# explicit variant name and target.
build-all: build-cpp build-openmp
	@echo "SYCL not built automatically; use build-sycl-acpp or build-sycl-dpcpp with explicit SYCL_VARIANT/SYCL_TARGET."

build-cpp:
	$(PYTHON) -m cpp.build

build-openmp:
	$(PYTHON) -m openmp.build

# AdaptiveCpp uses a controlled source-built LLVM toolchain, as in the
# original artifact, but without any hostname/module assumptions. LLVM target
# selection is explicit.
install-llvm:
	@test -n "$(LLVM_PREFIX)" || { echo "LLVM_PREFIX is required" >&2; exit 2; }
	@test -n "$(LLVM_TARGETS)" || { echo "LLVM_TARGETS is required (cpu,cuda,hip or a comma-separated combination)" >&2; exit 2; }
	@test -n "$(LLVM_VERSION)" || { echo "LLVM_VERSION is required" >&2; exit 2; }
	LLVM_JOBS="$(LLVM_JOBS)" ./sycl/install-llvm.sh "$(LLVM_PREFIX)" "$(LLVM_TARGETS)" "$(LLVM_VERSION)"

install-acpp:
	@test -n "$(ACPP_PREFIX)" || { echo "ACPP_PREFIX is required" >&2; exit 2; }
	@test -n "$(LLVM_PREFIX)" || { echo "LLVM_PREFIX is required; AdaptiveCpp does not fall back to system LLVM" >&2; exit 2; }
	ACPP_REF="$(ACPP_REF)" ACPP_JOBS="$(ACPP_JOBS)" ./sycl/install-acpp.sh "$(ACPP_PREFIX)" "$(LLVM_PREFIX)"

# DPC++ follows Intel's current source-build workflow. It builds its own
# LLVM-based SYCL toolchain with buildbot/configure.py + compile.py; it does
# not consume LLVM_PREFIX and no longer requires the old external oneTBB setup.
install-dpcpp:
	@test -n "$(DPCPP_PREFIX)" || { echo "DPCPP_PREFIX is required" >&2; exit 2; }
	@test -n "$(DPCPP_TARGETS)" || { echo "DPCPP_TARGETS is required (cpu,xpu,cuda,hip or a comma-separated combination)" >&2; exit 2; }
	DPCPP_REF="$(DPCPP_REF)" DPCPP_JOBS="$(DPCPP_JOBS)" ./sycl/install-dpcpp.sh "$(DPCPP_PREFIX)" "$(DPCPP_TARGETS)"

# Examples:
#   make build-sycl-acpp SYCL_VARIANT=acpp-a100 SYCL_TARGET=cuda SYCL_ARCH=sm_80
#   make build-sycl-acpp SYCL_VARIANT=acpp-mi250 SYCL_TARGET=hip SYCL_ARCH=gfx90a
#   make build-sycl-dpcpp SYCL_VARIANT=dpcpp-xpu SYCL_TARGET=xpu
build-sycl-acpp:
	@test -n "$(SYCL_VARIANT)" || { echo "SYCL_VARIANT is required" >&2; exit 2; }
	@test -n "$(SYCL_TARGET)" || { echo "SYCL_TARGET is required" >&2; exit 2; }
	@if [[ "$(SYCL_TARGET)" == "cuda" || "$(SYCL_TARGET)" == "hip" ]]; then \
		test -n "$(SYCL_ARCH)" || { echo "SYCL_ARCH is required for $(SYCL_TARGET)" >&2; exit 2; }; \
		./sycl/build-acpp.sh "$(SYCL_VARIANT)" "$(SYCL_TARGET)" "$(SYCL_ARCH)"; \
	else \
		test -z "$(SYCL_ARCH)" || { echo "SYCL_ARCH is only valid for cuda/hip targets" >&2; exit 2; }; \
		./sycl/build-acpp.sh "$(SYCL_VARIANT)" "$(SYCL_TARGET)"; \
	fi

build-sycl-dpcpp:
	@test -n "$(SYCL_VARIANT)" || { echo "SYCL_VARIANT is required" >&2; exit 2; }
	@test -n "$(SYCL_TARGET)" || { echo "SYCL_TARGET is required" >&2; exit 2; }
	@if [[ "$(SYCL_TARGET)" == "cuda" || "$(SYCL_TARGET)" == "hip" ]]; then \
		test -n "$(SYCL_ARCH)" || { echo "SYCL_ARCH is required for $(SYCL_TARGET)" >&2; exit 2; }; \
		./sycl/build-dpcpp.sh "$(SYCL_VARIANT)" "$(SYCL_TARGET)" "$(SYCL_ARCH)"; \
	else \
		test -z "$(SYCL_ARCH)" || { echo "SYCL_ARCH is only valid for cuda/hip targets" >&2; exit 2; }; \
		./sycl/build-dpcpp.sh "$(SYCL_VARIANT)" "$(SYCL_TARGET)"; \
	fi

list-sycl-builds:
	@$(PYTHON) -c 'from sycl.backend import built_variants; print("\\n".join(built_variants()) or "(none)")'

# test_sycl_loits.py skips from the complete suite when no explicit SYCL
# variant is selected. The dedicated target requires explicit selection.
test:
	$(PYTHON) -m pytest -q tests

test-pytorch:
	$(PYTHON) -m pytest -q tests/test_pytorch_loits.py tests/test_profiler.py

test-cpp:
	$(PYTHON) -m pytest -q tests/test_cpp_loits.py

test-openmp:
	$(PYTHON) -m pytest -q tests/test_openmp_loits.py

test-sycl:
	@test -n "$${QUANTOM_SYCL_VARIANT:-}" || { echo "QUANTOM_SYCL_VARIANT must be set explicitly" >&2; exit 2; }
	$(PYTHON) -m pytest -q tests/test_sycl_loits.py

test-rng:
	$(PYTHON) -m pytest -q tests/test_native_rng.py

benchmark:
	$(PYTHON) benchmark_training.py \
		--backend $(BACKEND) \
		--device $(DEVICE) \
		--events $(EVENTS) \
		--grid-size $(GRID_SIZE) \
		--warmup $(WARMUP) \
		--iterations $(ITERATIONS) \
		--seed $(SEED)

profile:
	$(PYTHON) benchmark_training.py \
		--backend $(BACKEND) \
		--device $(DEVICE) \
		--events $(EVENTS) \
		--grid-size $(GRID_SIZE) \
		--warmup $(WARMUP) \
		--iterations $(ITERATIONS) \
		--seed $(SEED) \
		--regions \
		--trace

list-backends:
	$(PYTHON) benchmark_training.py --device $(DEVICE) --list-backends

plots:
	$(PYTHON) -m plotting.plot_all --input results/training

plot-cpu:
	$(PYTHON) -m plotting.plot_cpu

plot-gpu:
	$(PYTHON) -m plotting.plot_gpu

plot-ss:
	$(PYTHON) -m plotting.plot_ss

plot-ws:
	$(PYTHON) -m plotting.plot_ws

compile-check:
	$(PYTHON) -m pytorch.compile_check

clean: clean-cpp clean-openmp clean-sycl
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

clean-cpp:
	rm -rf cpp/build

clean-openmp:
	rm -rf openmp/build

clean-sycl:
	rm -rf sycl/build

# Results are intentionally not removed by `make clean`.
clean-results:
	rm -rf results
