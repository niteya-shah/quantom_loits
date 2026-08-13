.PHONY: all plot plot-strong plot-weak plot-frs \
        install reinstall \
        env env-pytorch env-check bootstrap bootstrap-all \
        preflight rebuild rebuild-cpp rebuild-omp rebuild-sycl \
        build build-cpp build-omp build-openmp build-sycl \
        build-sycl-acpp build-sycl-tbb build-sycl-cuda build-sycl-hip build-sycl-xpu \
        rerun rerun-strong rerun-weak rerun-frs \
        rerun-frs-cpu rerun-frs-cuda rerun-frs-hip rerun-frs-xpu \
        clean clean-cpp clean-omp clean-openmp clean-sycl \
        smoke scaling profile debug \
        test test-pytorch test-cpp test-omp test-openmp test-sycl \
        test-omp-basic test-omp-instrument test-omp-flto test-omp-instrument-flto \
        test-sycl-acpp test-sycl-acpp-omp test-sycl-acpp-cuda \
        test-sycl-dpcpp-cuda test-sycl-dpcpp-hip test-sycl-usy \
        benchmark-training profile-training compile-check

SHELL := /usr/bin/env bash
PIXI_RUN := ./setup-pixi.sh run
BACKEND ?= torch
DEVICE ?= cpu
EVENTS ?= 10000
ITERATIONS ?= 10

# Default: regenerate figures from results/ only
all: plot

# -------------------------
# Plotting
# -------------------------

plot: plot-strong plot-weak plot-frs

plot-strong:
	./plot_strong_scaling.sh

plot-weak:
	./plot_weak_scaling.sh

plot-frs:
	./plot_fixed_resource_and_stacked.sh

# -------------------------
# Pixi / Python environment
# -------------------------

env:
	./setup-pixi.sh install

env-pytorch: env
	./setup-pytorch.sh

env-check: env-pytorch
	./setup-pixi.sh run python -c "import sys, torch; print(sys.executable); print(torch.__version__)"

bootstrap: env-pytorch build

bootstrap-all: env-pytorch build plot

# -------------------------
# Toolchain install / preflight
# -------------------------

install:
	cd sycl && ./install.sh

reinstall:
	cd sycl && REINSTALL=1 ./install.sh

# Run before any rerun experiment:
# - ensure pixi exists
# - ensure torch wheels are installed
# - ensure SYCL toolchains/backends are installed for this host
preflight: env-pytorch install

# Force a fresh rebuild from scratch after install.
rebuild: preflight clean build

rebuild-cpp: preflight clean-cpp build-cpp
rebuild-omp: preflight clean-omp build-omp
rebuild-sycl: preflight clean-sycl build-sycl

# -------------------------
# Build Python wrappers
# -------------------------

build: build-cpp build-omp build-sycl

build-cpp:
	$(PIXI_RUN) python -m cpp.build

build-omp: build-openmp

build-openmp:
	$(PIXI_RUN) python -m openmp.build

build-sycl: build-sycl-acpp

build-sycl-acpp:
	$(PIXI_RUN) ./sycl/build-acpp.sh generic

build-sycl-tbb:
	$(PIXI_RUN) ./sycl/build-dpcpp.sh cpu

build-sycl-cuda:
	$(PIXI_RUN) ./sycl/build-dpcpp.sh cuda

build-sycl-hip:
	$(PIXI_RUN) ./sycl/build-dpcpp.sh hip

build-sycl-xpu:
	$(PIXI_RUN) ./sycl/build-dpcpp.sh xpu

# -------------------------
# Rerun experiments into results/
# -------------------------

rerun: rerun-strong rerun-weak rerun-frs

# Strong/weak scaling are CPU-side paths.
# Use rebuild so we do:
#   install -> clean -> rebuild -> rerun
rerun-strong: rebuild
	DORUN=1 ./plot_strong_scaling.sh

rerun-weak: rebuild
	DORUN=1 ./plot_weak_scaling.sh

# FRS may additionally rebuild backend-specific DPC++ wrappers in-script,
# but we still want a clean installed baseline first.
rerun-frs: rebuild
	DORUN=1 ./plot_fixed_resource_and_stacked.sh

# Optional explicit per-backend entry points
rerun-frs-cpu: rebuild
	DORUN=1 ./plot_fixed_resource_and_stacked.sh

rerun-frs-cuda: rebuild
	DORUN=1 FORCE_CUDA=1 ./plot_fixed_resource_and_stacked.sh

rerun-frs-hip: rebuild
	DORUN=1 FORCE_HIP=1 ./plot_fixed_resource_and_stacked.sh

rerun-frs-xpu: rebuild
	DORUN=1 FORCE_XPU=1 ./plot_fixed_resource_and_stacked.sh


# -------------------------
# Differentiable training benchmark
# -------------------------

benchmark-training:
	$(PIXI_RUN) python benchmark_training.py --backend $(BACKEND) --device $(DEVICE) --events $(EVENTS) --iterations $(ITERATIONS)

profile-training:
	$(PIXI_RUN) python benchmark_training.py --backend $(BACKEND) --device $(DEVICE) --events $(EVENTS) --iterations $(ITERATIONS) --regions --trace

compile-check:
	$(PIXI_RUN) python -m pytorch.compile_check

# -------------------------
# Tests
# -------------------------

test: env test-pytorch test-cpp test-omp test-sycl

test-pytorch:
	$(PIXI_RUN) python -m pytest -q tests

test-cpp:
	$(PIXI_RUN) python -m pytest -q tests/test_cpp_loits.py

test-omp: test-openmp

test-openmp:
	$(PIXI_RUN) python -m pytest -q tests/test_openmp_loits.py

test-omp-basic:
	$(PIXI_RUN) make -C omp omp_test

test-omp-instrument:
	$(PIXI_RUN) make -C omp omp_instrument

test-omp-flto:
	$(PIXI_RUN) make -C omp omp_test_flto

test-omp-instrument-flto:
	$(PIXI_RUN) make -C omp omp_instrument_flto

test-sycl:
	$(PIXI_RUN) python -m pytest -q tests/test_sycl_loits.py

test-sycl-acpp: test-sycl

test-sycl-acpp-omp: test-sycl

test-sycl-acpp-cuda: test-sycl

test-sycl-dpcpp-cuda: test-sycl

test-sycl-dpcpp-hip: test-sycl

test-sycl-usy: test-sycl

# -------------------------
# Legacy helper scripts
# -------------------------

smoke:
	./run.sh --smoke

scaling:
	./run_omp_thread_scaling.sh

profile:
	./profile_2dloits_performance.sh

debug:
	./debug.sh

# -------------------------
# Cleanup
# -------------------------

clean: clean-cpp clean-omp clean-sycl

clean-cpp:
	rm -rf cpp/build

clean-omp: clean-openmp

clean-openmp:
	rm -rf openmp/build

clean-sycl:
	rm -rf sycl/build
