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

    sycl/build/acpp-polaris-a100/
    sycl/build/acpp-illyad-h100/
    sycl/build/acpp-odyssey-mi300a/
    sycl/build/acpp-aurora-cpu/
    sycl/build/dpcpp-illyad-h100/
    sycl/build/dpcpp-odyssey-mi300a/

All variants compile the same `loits_core.cpp` and `runtime.cpp`.

At runtime a SYCL variant is mandatory:

    export QUANTOM_SYCL_VARIANT=acpp-a100

There is deliberately no fallback to the only build present, no hardware-based
automatic selection, and no default toolchain. This allows multiple SYCL
implementations to coexist and makes benchmark provenance explicit.

Each complete variant contains the native library plus one generated metadata file:

    libquantom_loits_sycl.so
    variant.py

`variant.py` contains a single `METADATA` dictionary recording the toolchain,
target, Torch device type, and architecture that were actually built.

List complete builds with:

    make list-sycl-builds

The cluster helper keeps compiler installs and work trees site-specific under
`$TOOLCHAIN_ROOT/<site>` and `$TOOLCHAIN_WORK_ROOT/<site>`. For example:

    ./sycl/setup-cluster-sycl.sh illyad fetch acpp
    ./sycl/setup-cluster-sycl.sh illyad build acpp
    ./sycl/setup-cluster-sycl.sh illyad fetch dpcpp
    ./sycl/setup-cluster-sycl.sh illyad build dpcpp

Illyad targets its NVIDIA H100 with `sm_90`. Odyssey remains isolated under its
own site directories even when both machines share the same filesystem. On
Aurora, DPC++ remains site-provided for PVC/XPU while AdaptiveCpp is built only
for its OpenMP CPU backend:

    ./sycl/setup-cluster-sycl.sh aurora fetch acpp
    ./sycl/setup-cluster-sycl.sh aurora build acpp

The Aurora AdaptiveCpp build uses LLVM's host CPU target only and forces CUDA,
ROCm, OpenCL, Level Zero, and Vulkan runtime backends off. The resulting QuantOm
variant is `acpp-aurora-cpu`.

## Install the SYCL toolchains

The source-build helpers are intentionally generic: no hostnames, module loads,
or vendor-specific filesystem paths are encoded. They follow the current
upstream build instructions and leave cluster-specific CUDA, ROCm, Level Zero,
and compiler module selection to the environment.

### LLVM for AdaptiveCpp

The original artifact needed a controlled source-built LLVM for AdaptiveCpp. We
retain that reproducibility path, even though current AdaptiveCpp documentation
says a sufficiently recent distribution LLVM is usually easier. The source-build
recipe itself has been updated to the current AdaptiveCpp LLVM documentation.
AdaptiveCpp currently supports official LLVM releases 15--21 and does not
support development LLVM snapshots.

Build only the code-generation targets required by the AdaptiveCpp installation:

    make install-llvm \
        LLVM_PREFIX=/shared/toolchains/llvm-21.1.0 \
        LLVM_TARGETS=cpu,cuda \
        LLVM_VERSION=21.1.0

or directly:

    ./sycl/install-llvm.sh \
        /shared/toolchains/llvm-21.1.0 \
        cpu,cuda \
        21.1.0

`cpu` enables X86, `cuda` adds NVPTX, and `hip` adds AMDGPU. The current source
recipe builds `clang;lld;openmp`, `compiler-rt`, and shared `libLLVM`; disables
LLVM assertions/dumps/tests/examples/benchmarks; and disables libomptarget. It
does **not** build the old `clang-tools-extra`, libc++, libc++abi, libunwind, or
OpenMP offload runtime stack because AdaptiveCpp's current instructions do not
require those for this LLVM dependency.

Builds use all CPUs reported by `nproc` by default. Override the job count only
when needed. Useful overrides:

    LLVM_WORKDIR=/scratch/me/llvm-build
    LLVM_SOURCE_DIR=/path/to/existing/llvm-project
    LLVM_REF=llvmorg-21.1.0
    LLVM_JOBS=16
    LLVM_LINK_JOBS=2
    LLVM_C_COMPILER=/path/to/gcc
    LLVM_CXX_COMPILER=/path/to/g++
    LLVM_CMAKE_ARGS="..."

The installer rejects LLVM major versions outside AdaptiveCpp's currently
documented 15--21 window unless `LLVM_ALLOW_UNSUPPORTED=1` is set deliberately.

### AdaptiveCpp

AdaptiveCpp's current source prerequisites include Python 3, CMake, and Boost;
the installer intentionally does not install system packages. Build AdaptiveCpp
against the controlled LLVM prefix:

    make install-acpp \
        LLVM_PREFIX=/shared/toolchains/llvm-21.1.0 \
        ACPP_PREFIX=/shared/toolchains/adaptivecpp \
        ACPP_REF=<tag-or-commit>

or directly:

    ./sycl/install-acpp.sh \
        /shared/toolchains/adaptivecpp \
        /shared/toolchains/llvm-21.1.0 \
        <tag-or-commit>

The CMake configuration uses AdaptiveCpp's current recommended `full` compiler
feature profile and points it to the selected LLVM with `LLVM_DIR` and
`CLANG_EXECUTABLE_PATH`. If `CUDA_PATH` is supplied, the installer passes
`CUDA_TOOLKIT_ROOT_DIR` and enables the CUDA backend. If `ROCM_PATH` is supplied,
it passes `ROCM_PATH` and enables the ROCm backend.

For the generic AMD path, AdaptiveCpp currently requires its LLVM version to be
no newer than the LLVM version bundled with ROCm. That compatibility must be
checked when we inspect the cluster's ROCm module.

After installation:

    export ACPP_PREFIX=/shared/toolchains/adaptivecpp

`build-acpp.sh` then resolves `$ACPP_PREFIX/bin/acpp`.

### DPC++

Intel's current Linux source-build prerequisites include Git, CMake, Python,
Ninja, and hwloc >= 2.3 (with zstd optional). The installer checks the basic
command-line tools but intentionally does not install system packages.

DPC++ does **not** use the AdaptiveCpp LLVM prefix. Current Intel documentation
builds DPC++'s own LLVM-based SYCL toolchain from the `intel/llvm` `sycl` branch
with `buildbot/configure.py` followed by `buildbot/compile.py`. The old QuantOm
requirements for a pre-built custom LLVM and external oneTBB have therefore been
removed from the DPC++ installer.

Build a toolchain with an explicit target set:

    make install-dpcpp \
        DPCPP_PREFIX=/shared/toolchains/dpcpp-xpu-cuda \
        DPCPP_TARGETS=xpu,cuda \
        DPCPP_REF=<commit-or-tag>

or directly:

    ./sycl/install-dpcpp.sh \
        /shared/toolchains/dpcpp-xpu-cuda \
        xpu,cuda \
        <commit-or-tag>

Targets map to Intel's current configure interface:

    cpu   -> --native_cpu
    xpu   -> normal SPIR-capable DPC++ build
    cuda  -> --cuda
    hip   -> --hip

The target list can contain several entries, so a single source-built DPC++
toolchain can support, for example, `xpu,cuda`. There is no default target.

For non-standard vendor stacks:

    CUDA_PATH=/path/to/cuda
    ROCM_PATH=/path/to/rocm

The installer passes `CUDAToolkit_ROOT` for CUDA and `UR_HIP_ROCM_DIR` for HIP,
matching the current DPC++ guide. For CUDA builds it also validates CMake's CUPTI
library selection and repairs a bad generated path from the toolkit's real CUPTI
location before compilation starts. `DPCPP_CUPTI_LIBRARY` can override that path.
Native CPU no longer requires a separate TBB installation. Level Zero remains an
external runtime requirement on Intel GPU nodes.

Intel's source-build guide currently uses the build tree directly as the
toolchain (`bin/` and `lib/`) and marks deployment instructions as incomplete.
For that reason `DPCPP_PREFIX` is the final DPC++ **build directory**, not a
relocated copy of it. After building:

    export DPCPP_PREFIX=/shared/toolchains/dpcpp-xpu-cuda

`build-dpcpp.sh` then resolves `$DPCPP_PREFIX/bin/clang++`.

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

Installed toolchain roots can be selected with `ACPP_PREFIX` and
`DPCPP_PREFIX`. Compiler paths remain overrideable with `ACPP_CXX`,
`DPCPP_CXX`, `DPCPP_CPU_CXX`, `DPCPP_XPU_CXX`, `DPCPP_CUDA_CXX`, and
`DPCPP_HIP_CXX`. Additional compiler flags can be supplied with
`SYCL_EXTRA_FLAGS`.

For ROCm PyTorch, the Torch device string is still `cuda`.

A failed rebuild removes the selected variant's `variant.py` completeness metadata
before compilation, but does not modify any other variant directory.

If only the active Python/PyTorch ABI changes, rebuild just the host binding and
leave the native SYCL core untouched:

    export QUANTOM_SYCL_VARIANT=acpp-illyad-h100
    make rebuild-sycl-binding

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
