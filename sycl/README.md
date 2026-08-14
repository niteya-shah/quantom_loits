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

## Multiple builds on one filesystem

Each SYCL build is stored in its own directory:

    sycl/build/<variant>/

The variant name is user supplied. Nothing is selected automatically and one
build never overwrites another.

For example, a shared cluster checkout can contain:

    sycl/build/acpp-a100/
    sycl/build/acpp-mi250/
    sycl/build/dpcpp-xpu/
    sycl/build/dpcpp-a100/

All variants compile the same `loits_core.cpp`.

At runtime a SYCL variant is mandatory:

    export QUANTOM_SYCL_VARIANT=acpp-a100

There is deliberately no fallback to the only build present, no hardware-based
automatic selection, and no default toolchain. This allows multiple SYCL
implementations to coexist and makes benchmark provenance explicit.

List complete builds with:

    make list-sycl-builds

## Install AdaptiveCpp

The repository includes a small general AdaptiveCpp source installer:

    ./sycl/install-acpp.sh /shared/toolchains/adaptivecpp

To pin the toolchain for reproducible experiments, pass a tag or commit:

    ./sycl/install-acpp.sh /shared/toolchains/adaptivecpp <tag-or-commit>

The Makefile wrapper is equivalent:

    make install-acpp \
        ACPP_PREFIX=/shared/toolchains/adaptivecpp \
        ACPP_REF=<tag-or-commit>

The installer intentionally contains no package-manager commands, module loads,
hardware detection, or machine-specific compiler paths. It performs the normal
AdaptiveCpp CMake configure/build/install sequence only. The machine must already
provide AdaptiveCpp's build dependencies and whichever vendor stacks are needed
for the targets that installation should support.

Useful environment overrides are:

    ACPP_WORKDIR=/scratch/me/acpp-build
    ACPP_SOURCE_DIR=/path/to/existing/AdaptiveCpp
    ACPP_JOBS=8
    ACPP_BUILD_TYPE=Release
    ACPP_CMAKE_ARGS="-DLLVM_DIR=/path/to/llvm/lib/cmake/llvm ..."

`ACPP_SOURCE_DIR` is useful on clusters where compute nodes do not have outbound
network access. `ACPP_CMAKE_ARGS` is the escape hatch for the LLVM/CUDA/ROCm
configuration we determine for a particular cluster; those settings are not
hard-coded into this repository.

After installation either put `bin/` on `PATH`, set `ACPP_CXX` directly, or set:

    export ACPP_PREFIX=/shared/toolchains/adaptivecpp

`build-acpp.sh` recognizes `ACPP_PREFIX` and uses `$ACPP_PREFIX/bin/acpp`.

## Build

The build scripts require an explicit variant name and target. CUDA/HIP builds
also require an explicit offload architecture.

AdaptiveCpp:

    ./sycl/build-acpp.sh acpp-generic generic
    ./sycl/build-acpp.sh acpp-cpu cpu
    ./sycl/build-acpp.sh acpp-a100 cuda sm_80
    ./sycl/build-acpp.sh acpp-h100 cuda sm_90
    ./sycl/build-acpp.sh acpp-mi250 hip gfx90a

DPC++:

    ./sycl/build-dpcpp.sh dpcpp-cpu cpu
    ./sycl/build-dpcpp.sh dpcpp-xpu xpu
    ./sycl/build-dpcpp.sh dpcpp-a100 cuda sm_80
    ./sycl/build-dpcpp.sh dpcpp-mi250 hip gfx90a

The Makefile exposes the same explicit interface:

    make build-sycl-acpp \
        SYCL_VARIANT=acpp-a100 \
        SYCL_TARGET=cuda \
        SYCL_ARCH=sm_80

    make build-sycl-dpcpp \
        SYCL_VARIANT=dpcpp-xpu \
        SYCL_TARGET=xpu

Compiler paths remain overrideable with `ACPP_CXX`, `DPCPP_CXX`,
`DPCPP_CPU_CXX`, `DPCPP_XPU_CXX`, `DPCPP_CUDA_CXX`, and `DPCPP_HIP_CXX`.
Additional compiler flags can be supplied with `SYCL_EXTRA_FLAGS`.

For ROCm PyTorch, the Torch device string is still `cuda`.

A failed rebuild removes the selected variant's completeness markers before
compilation, but does not modify any other variant directory.

## Runtime selection

Select a built variant explicitly before using `backend=sycl`:

    export QUANTOM_SYCL_VARIANT=acpp-a100

    python benchmark_training.py \
        --backend sycl \
        --device cuda \
        --events 1000000

On another node using the same checkout:

    export QUANTOM_SYCL_VARIANT=dpcpp-xpu

    python benchmark_training.py \
        --backend sycl \
        --device xpu \
        --events 1000000

Changing `QUANTOM_SYCL_VARIANT` after a SYCL extension has already been loaded
in the same Python process is rejected. Start a new process to use another
variant.

Without `QUANTOM_SYCL_VARIANT`, SYCL is reported as unavailable even if one or
more builds exist:

    python benchmark_training.py --device cuda --list-backends

This is intentional.

## Test

Set the runtime selector so SYCL and PyTorch refer to the same physical device,
select the variant, then run:

    export QUANTOM_SYCL_VARIANT=acpp-a100
    python -m pytest -q tests/test_sycl_loits.py

or:

    export QUANTOM_SYCL_VARIANT=acpp-a100
    make test-sycl

For an AdaptiveCpp `generic` build, also set `QUANTOM_SYCL_TEST_DEVICE`
explicitly to `cpu`, `cuda`, or `xpu`.

The tests compare the SYCL forward result and hand-written reverse VJP against
the PyTorch implementation using exactly the same Philox samples. They also
check the standard Philox4x32-10 known-answer values.

The complete repository test suite still works when SYCL is not selected:
SYCL execution tests skip. If `QUANTOM_SYCL_VARIANT` is set, that selection is
expected to refer to a complete build.

## Profiling

The semantic `loits::*` regions match the C++ and OpenMP backends. With
`profile_regions=True`, the binding waits at each native region so its CPU wall
time includes that SYCL kernel. The normal benchmark path waits only where a
host scalar is required and at the end of forward/backward.

Benchmark CSVs record the explicit variant name in the `implementation` field,
so results from `acpp-a100`, `dpcpp-a100`, `acpp-mi250`, etc. remain separate.
