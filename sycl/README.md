# SYCL LOITS backend

This directory contains one differentiable SYCL LOITS implementation. The same
`loits_core.cpp` is compiled with AdaptiveCpp or DPC++ for CPU, NVIDIA, AMD, or
Intel GPU targets. There are no target-specific LOITS kernels.

The Torch binding remains ordinary host C++. PyTorch owns every input, output,
and saved-autograd tensor. The SYCL core receives their raw pointers. To keep
interop simple and correct, the Python backend synchronizes the PyTorch device
before entering SYCL and the native binding waits for the SYCL queue before it
returns to PyTorch. This intentionally gives up stream overlap; it does not
stage tensors through CPU/NumPy or copy the event arrays between frameworks.

The SYCL runtime chooses the queue/device. Use the implementation's normal
runtime selector variables (`ONEAPI_DEVICE_SELECTOR`, `ACPP_VISIBILITY_MASK`,
etc.) so that the selected SYCL device matches the PyTorch tensor device.

## Build

AdaptiveCpp:

    ./sycl/build-acpp.sh generic
    ./sycl/build-acpp.sh cpu
    ./sycl/build-acpp.sh cuda
    ./sycl/build-acpp.sh hip

DPC++:

    ./sycl/build-dpcpp.sh cpu
    ./sycl/build-dpcpp.sh xpu
    ./sycl/build-dpcpp.sh cuda
    ./sycl/build-dpcpp.sh hip

Compiler paths and architecture flags are overrideable with environment
variables such as `ACPP_CXX`, `DPCPP_CXX`, `DPCPP_CUDA_CXX`,
`CUDA_DEV_TARGET`, `HIP_DEV_TARGET`, and `SYCL_EXTRA_FLAGS`.

For ROCm PyTorch, the Torch device string is still `cuda`.

## Test

Set the runtime selector so SYCL and PyTorch refer to the same physical device,
then run:

    python -m pytest -q tests/test_sycl_loits.py

For an AdaptiveCpp `generic` build, set `QUANTOM_SYCL_TEST_DEVICE` to the
matching Torch device (`cpu`, `cuda`, or `xpu`) when the automatic marker is
not specific enough.

The tests compare the SYCL forward result and hand-written reverse VJP against
the PyTorch implementation using exactly the same Philox samples. They also
check the standard Philox4x32-10 known-answer values.

## Profiling

The semantic `loits::*` regions match the C++ and OpenMP backends. With
`profile_regions=True`, the binding waits at each native region so its CPU wall
time includes that SYCL kernel. The normal benchmark path waits only where a
host scalar is required and at the end of forward/backward.

## Optional backend behavior

A SYCL build is intentionally optional. The backend is considered available
only after the core library and its successful-build markers exist. Failed or
partial rebuilds remove those markers, so `make test` will skip SYCL instead of
attempting to load a stale core. Use:

    python benchmark_training.py --device <cpu|cuda|xpu> --list-backends

to inspect availability without compiling or loading the SYCL extension.
