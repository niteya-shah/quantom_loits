# QuantOm LOITS

This repository contains the active differentiable LOITS implementations used
for the PyTorch/C++/OpenMP/SYCL performance study. The current codebase focuses
on end-to-end training, native forward/backward execution, correctness, and
profiling. Historical conference-artifact implementations and archived results are
not kept in the active tree. Machine-specific toolchain logic was removed, but
the controlled source-built LLVM -> AdaptiveCpp installation path is retained.
The active plotting tools have been rewritten for the current training/profiler
CSV format.

## Backends

The common Python interface is:

```python
from loits import LOITS

sampler = LOITS(backend="torch", device="cpu")
events = sampler(theory_outputs, n_events)
```

Available backends are:

- `torch`: PyTorch implementation with autograd and `torch.compile` support.
- `cpp`: serial C++17 CPU implementation with a hand-written reverse VJP.
- `openmp`: OpenMP CPU implementation with a hand-written reverse VJP.
- `sycl`: one SYCL implementation compiled for different SYCL toolchains and
  targets, also with a hand-written reverse VJP.

C++, OpenMP, and SYCL use the same Philox4x32-10 counter-based RNG mapping so
native random streams can be compared exactly for a fixed seed and sequence.

## Repository layout

```text
benchmark_training.py       end-to-end GAN benchmark/profiler
loits.py                    backend selection
rng/philox.hpp              shared native Philox implementation

pytorch/                    PyTorch LOITS, theory, GAN, profiler
cpp/                        serial C++ backend
openmp/                     OpenMP backend
sycl/                       single-source SYCL backend and build scripts
tests/                      correctness, autograd, RNG, profiler tests
plotting/                   current scaling, speedup, and breakdown plots
```

Generated native extensions live under each backend's `build/` directory.
Benchmark CSVs and traces are written under `results/`. Both are ignored by
Git.

## Environment

A minimal Pixi environment is provided for Python build/test dependencies:

```bash
pixi install
pixi shell
```

Install the PyTorch distribution appropriate for the machine separately. C++
and OpenMP also require a system compiler. SYCL requires either AdaptiveCpp or
an appropriate DPC++/oneAPI/LLVM SYCL toolchain.

### SYCL toolchain installation

The repository keeps source-build helpers for both SYCL implementations. They
follow the current upstream build procedures rather than restoring the old
machine-specific installer.

AdaptiveCpp uses a controlled official LLVM release because that was required
for the original cluster setup. AdaptiveCpp currently documents support for
official LLVM releases 15--21; its generic CUDA path needs NVPTX and its AMD
path needs AMDGPU code generation. Build only the targets required by the
installation:

```bash
make install-llvm \
    LLVM_PREFIX=/shared/toolchains/llvm-21.1.0 \
    LLVM_TARGETS=cpu,cuda \
    LLVM_VERSION=21.1.0

make install-acpp \
    LLVM_PREFIX=/shared/toolchains/llvm-21.1.0 \
    ACPP_PREFIX=/shared/toolchains/adaptivecpp \
    ACPP_REF=<tag-or-commit>
```

The LLVM helper uses the current AdaptiveCpp source-LLVM configuration: Clang,
LLD and OpenMP projects, `compiler-rt`, a shared `libLLVM`, assertions/tests
disabled, and libomptarget disabled. It deliberately does not build the old
libc++/libc++abi/libunwind/offload stack. AdaptiveCpp is configured with the
full compiler feature profile plus the documented `LLVM_DIR` and
`CLANG_EXECUTABLE_PATH` settings. CUDA/ROCm paths remain explicit environment
inputs when those stacks are not in standard locations.

DPC++ is independent of that AdaptiveCpp LLVM prefix. Current Intel DPC++
documentation builds the compiler from the `intel/llvm` `sycl` branch with
`buildbot/configure.py` followed by `buildbot/compile.py`; Native CPU, CUDA and
HIP are enabled with `--native_cpu`, `--cuda` and `--hip`. The build directory
itself is the toolchain used at runtime:

```bash
make install-dpcpp \
    DPCPP_PREFIX=/shared/toolchains/dpcpp-xpu-cuda \
    DPCPP_TARGETS=xpu,cuda \
    DPCPP_REF=<commit-or-tag>
```

For a non-standard CUDA location set `CUDA_PATH`; for a non-standard ROCm
location set `ROCM_PATH`. The DPC++ helper passes the current documented
`CUDAToolkit_ROOT` and `UR_HIP_ROCM_DIR` configuration. XPU execution still
requires a Level Zero implementation on the target node. See `sycl/README.md`
for the complete interfaces.

## Build

Serial C++ and OpenMP:

```bash
make build-cpp
make build-openmp
```

The SYCL backend has one `sycl/loits_core.cpp`. Multiple compiled variants can
coexist under `sycl/build/<variant>/` on a shared filesystem; every variant
compiles that same source. SYCL builds are fully explicit: provide a variant
name, target, and an architecture for CUDA/HIP.

AdaptiveCpp examples:

```bash
./sycl/build-acpp.sh acpp-a100 cuda sm_80
./sycl/build-acpp.sh acpp-mi250 hip gfx90a
./sycl/build-acpp.sh acpp-cpu cpu
```

DPC++ examples:

```bash
./sycl/build-dpcpp.sh dpcpp-xpu xpu
./sycl/build-dpcpp.sh dpcpp-a100 cuda sm_80
```

Equivalent top-level Make targets are:

```bash
make build-sycl-acpp SYCL_VARIANT=acpp-a100 SYCL_TARGET=cuda SYCL_ARCH=sm_80
make build-sycl-dpcpp SYCL_VARIANT=dpcpp-xpu SYCL_TARGET=xpu
```

There are deliberately no default SYCL target, architecture, or runtime build.
See `sycl/README.md` for compiler overrides and cluster usage.

## Backend availability

SYCL is optional. `make build` and `make build-all` build only the portable
PyTorch/C++/OpenMP side. A SYCL runtime variant must always be selected
explicitly, even if exactly one build exists:

```bash
export QUANTOM_SYCL_VARIANT=acpp-a100
python benchmark_training.py --device cuda --list-backends
```

Without `QUANTOM_SYCL_VARIANT`, `sycl` is reported as unavailable and the
message lists any complete variants found under `sycl/build/`. There is no
automatic hardware or single-build fallback.

`make test` remains valid on machines with no selected SYCL variant: the SYCL
execution tests skip. `make test-sycl` requires `QUANTOM_SYCL_VARIANT`. An
explicitly requested unavailable benchmark reports a short error before trainer
construction; automation can use `--skip-unavailable` to turn that case into a
successful skip instead.

## Correctness tests

Run the complete suite:

```bash
python -m pytest -q tests
```

or individual backends:

```bash
make test-pytorch
make test-cpp
make test-openmp
make test-sycl
make test-rng
```

The native-backend correctness tests compare forward results and the
hand-written reverse VJP against the PyTorch reference while holding the
stochastic samples fixed. RNG tests additionally compare native Philox streams
exactly. SYCL tests skip when a SYCL core has not been built.

To print the actual forward and backward error metrics for a native backend:

```bash
python compare_backend_torch.py --backend cpp --events 100000
OMP_NUM_THREADS=32 python compare_backend_torch.py --backend openmp --events 100000
QUANTOM_SYCL_VARIANT=acpp-a100 \
python compare_backend_torch.py --backend sycl --device cuda --events 100000
```

The comparison holds the random samples and upstream gradient fixed between
PyTorch and the selected native backend. SYCL requires both an explicit
`QUANTOM_SYCL_VARIANT` and an explicit `--device`.

## Training benchmark

Examples:

```bash
python benchmark_training.py --backend torch --device cpu --events 100000
python benchmark_training.py --backend cpp --device cpu --events 100000
python benchmark_training.py --backend openmp --device cpu --events 100000
QUANTOM_SYCL_VARIANT=acpp-a100 \
python benchmark_training.py --backend sycl --device cuda --events 100000
```

The Makefile exposes the same benchmark with configurable variables:

```bash
make benchmark BACKEND=openmp DEVICE=cpu EVENTS=1000000 ITERATIONS=10
```

For the detailed semantic region trace:

```bash
make profile BACKEND=openmp DEVICE=cpu EVENTS=1000000 ITERATIONS=10
```

The native backends expose the same hierarchical `loits::*` profiler names so
forward/backward stages can be compared directly.

## Plotting

Current plotting tools live under `plotting/` and consume the CSVs emitted by
`benchmark_training.py`. They discover only the series that are present, so a
CPU-only machine does not need placeholder SYCL/CUDA/XPU results.

Generate the standard available plots with:

```bash
make plots
```

or individually:

```bash
python -m plotting.plot_training results/training
python -m plotting.plot_loits results/training --scope autograd-forward
python -m plotting.plot_loits results/training --scope autograd-backward
python -m plotting.plot_speedup results/training --reference cpp
python -m plotting.plot_breakdown results/training --events 1000000
```

Scaling plots use medians and interquartile ranges. New CSVs also record an
`implementation` field. For SYCL this is the explicit `QUANTOM_SYCL_VARIANT`
name, such as `acpp-a100` or `dpcpp-xpu`; this prevents measurements from
different SYCL builds from overwriting or being merged while still retaining
`backend=sycl`.

## SYCL interop model

PyTorch owns the input, output, and saved-autograd tensors. The SYCL binding
passes their raw device pointers to the single SYCL implementation. The current
interop path intentionally synchronizes PyTorch before entering SYCL and waits
for the SYCL queue before returning to PyTorch. This prioritizes correctness
and zero-copy tensor storage over asynchronous stream overlap and avoids the
host/NumPy staging used by the retired implementation.

The selected SYCL device must correspond to the PyTorch tensor device. Each
explicit SYCL build stores its build provenance in `sycl/build/<variant>/variant.py`.
See `sycl/README.md` for runtime selector, shared-filesystem, and toolchain build
details.
