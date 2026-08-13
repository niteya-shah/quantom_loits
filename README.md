# QuantOm LOITS

This repository contains the active differentiable LOITS implementations used
for the PyTorch/C++/OpenMP/SYCL performance study. The current codebase focuses
on end-to-end training, native forward/backward execution, correctness, and
profiling. Historical conference-artifact implementations, archived results, and old
toolchain installers are intentionally not kept in the active tree; they
remain available through Git history. The active plotting tools have been
rewritten for the current training/profiler CSV format.

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

## Build

Serial C++ and OpenMP:

```bash
make build-cpp
make build-openmp
```

The SYCL backend has one `sycl/loits_core.cpp`. The Bash build scripts compile
that same source for the requested implementation/target.

AdaptiveCpp:

```bash
./sycl/build-acpp.sh generic
./sycl/build-acpp.sh cpu
./sycl/build-acpp.sh cuda
./sycl/build-acpp.sh hip
```

DPC++:

```bash
./sycl/build-dpcpp.sh cpu
./sycl/build-dpcpp.sh xpu
./sycl/build-dpcpp.sh cuda
./sycl/build-dpcpp.sh hip
```

Equivalent top-level Make targets are available, for example:

```bash
make build-sycl-acpp-cuda
make build-sycl-dpcpp-xpu
```

Compiler paths and device architecture flags can be overridden using the
environment variables documented in `sycl/README.md`.

## Backend availability

SYCL is optional. `make build` and `make build-all` build only the portable
PyTorch/C++/OpenMP side; SYCL builds are always explicit because the compiler
and target differ by machine. Inspect the current device/backend status with:

```bash
python benchmark_training.py --device cpu --list-backends
python benchmark_training.py --device cuda --list-backends
```

`make test` remains valid on machines with no SYCL installation: the SYCL tests
skip when no SYCL core has been built. An explicitly requested unavailable
benchmark reports a short error before trainer construction. Automation can use
`--skip-unavailable` to turn that case into a successful skip instead.

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

## Training benchmark

Examples:

```bash
python benchmark_training.py --backend torch --device cpu --events 100000
python benchmark_training.py --backend cpp --device cpu --events 100000
python benchmark_training.py --backend openmp --device cpu --events 100000
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
`implementation` field. For SYCL this contains the selected build variant such
as `acpp:cuda` or `dpcpp:xpu`; this prevents measurements from different SYCL
toolchains from overwriting or being merged while still retaining `backend=sycl`.

## SYCL interop model

PyTorch owns the input, output, and saved-autograd tensors. The SYCL binding
passes their raw device pointers to the single SYCL implementation. The current
interop path intentionally synchronizes PyTorch before entering SYCL and waits
for the SYCL queue before returning to PyTorch. This prioritizes correctness
and zero-copy tensor storage over asynchronous stream overlap and avoids the
host/NumPy staging used by the retired implementation.

The selected SYCL device must correspond to the PyTorch tensor device. See
`sycl/README.md` for runtime selector and build details.
