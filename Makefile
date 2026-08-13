SHELL := /usr/bin/env bash

PYTHON ?= python
BACKEND ?= torch
DEVICE ?= cpu
EVENTS ?= 100000
GRID_SIZE ?= 10
WARMUP ?= 5
ITERATIONS ?= 10
SEED ?= 0
SYCL_TARGET ?= generic

.PHONY: all \
        build build-all build-cpp build-openmp \
        build-sycl-acpp build-sycl-dpcpp \
        build-sycl-acpp-generic build-sycl-acpp-cpu build-sycl-acpp-cuda build-sycl-acpp-hip \
        build-sycl-dpcpp-cpu build-sycl-dpcpp-xpu build-sycl-dpcpp-cuda build-sycl-dpcpp-hip \
        test test-pytorch test-cpp test-openmp test-sycl test-rng \
        benchmark profile list-backends compile-check \
        plots plot-training plot-loits plot-breakdown \
        clean clean-cpp clean-openmp clean-sycl clean-results

all: test

# C++ and OpenMP use torch.utils.cpp_extension and build on demand.
build: build-cpp build-openmp

# SYCL requires an explicitly selected external compiler/target, so it is not
# part of the portable build-all target. Use one of the build-sycl-* targets.
build-all: build-cpp build-openmp
	@echo "SYCL not built automatically; select an explicit build-sycl-* target if available."

build-cpp:
	$(PYTHON) -m cpp.build

build-openmp:
	$(PYTHON) -m openmp.build

# Generic SYCL entry points. Override SYCL_TARGET as needed, e.g.
#   make build-sycl-acpp SYCL_TARGET=cuda
#   make build-sycl-dpcpp SYCL_TARGET=xpu
build-sycl-acpp:
	./sycl/build-acpp.sh $(SYCL_TARGET)

build-sycl-dpcpp:
	./sycl/build-dpcpp.sh $(SYCL_TARGET)

build-sycl-acpp-generic:
	./sycl/build-acpp.sh generic

build-sycl-acpp-cpu:
	./sycl/build-acpp.sh cpu

build-sycl-acpp-cuda:
	./sycl/build-acpp.sh cuda

build-sycl-acpp-hip:
	./sycl/build-acpp.sh hip

build-sycl-dpcpp-cpu:
	./sycl/build-dpcpp.sh cpu

build-sycl-dpcpp-xpu:
	./sycl/build-dpcpp.sh xpu

build-sycl-dpcpp-cuda:
	./sycl/build-dpcpp.sh cuda

build-sycl-dpcpp-hip:
	./sycl/build-dpcpp.sh hip

# test_sycl_loits.py skips when no SYCL core has been built.
test:
	$(PYTHON) -m pytest -q tests

test-pytorch:
	$(PYTHON) -m pytest -q tests/test_pytorch_loits.py tests/test_profiler.py

test-cpp:
	$(PYTHON) -m pytest -q tests/test_cpp_loits.py

test-openmp:
	$(PYTHON) -m pytest -q tests/test_openmp_loits.py

test-sycl:
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

plot-training:
	$(PYTHON) -m plotting.plot_training results/training

plot-loits:
	$(PYTHON) -m plotting.plot_loits results/training --scope autograd-forward

plot-breakdown:
	$(PYTHON) -m plotting.plot_breakdown results/training --events $(EVENTS)

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
