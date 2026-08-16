#!/usr/bin/env bash
set -euo pipefail

# QuantOm SYCL cluster toolchain helper
#
# Intended targets:
#   polaris  - NVIDIA A100 (CUDA, sm_80)
#   odyssey  - AMD MI300A (ROCm/HIP, gfx942)
#   aurora   - Intel PVC (use the system DPC++ toolchain; do not build AdaptiveCpp)
#
# Edit ONLY the module-loading functions below once the correct modules are known.
#
# Typical usage:
#   ./sycl/setup-cluster-sycl.sh polaris
#   ./sycl/setup-cluster-sycl.sh odyssey
#   ./sycl/setup-cluster-sycl.sh aurora
#
# Or build one implementation only:
#   ./sycl/setup-cluster-sycl.sh polaris acpp
#   ./sycl/setup-cluster-sycl.sh polaris dpcpp
#   ./sycl/setup-cluster-sycl.sh odyssey acpp
#   ./sycl/setup-cluster-sycl.sh odyssey dpcpp
#
# Defaults may be overridden from the environment:
#   TOOLCHAIN_ROOT=/shared/path/toolchains
#   TOOLCHAIN_WORK_ROOT=/scratch/$USER/quantom-toolchains
#   JOBS=16
#   LLVM_VERSION=21.1.8
#   ACPP_REF=v25.10.0
#   DPCPP_REF=sycl
#   FORCE_TOOLCHAIN=1
#
# This helper assumes patches 0015 and 0016 are applied, providing:
#   make install-llvm
#   make install-acpp
#   make install-dpcpp
#   make build-sycl-acpp
#   make build-sycl-dpcpp

###############################################################################
# SITE MODULES
#
# Fill these in after determining the correct module stack on each machine.
# Keep CUDA_PATH / ROCM_PATH assignments here if the module does not export
# CUDA_HOME/CUDA_ROOT or ROCM_HOME.
###############################################################################

modules_polaris() {
    # Example structure only -- replace with the actual Polaris modules.
    #
    # module purge
    # module load <gcc-module>
    # module load <cmake-module>
    # module load <ninja-module>
    # module load <boost-module>
    # module load <cuda-module>
    #
    # export CUDA_PATH=/path/exported/by/the/cuda/module
    :
}

modules_odyssey() {
    # Example structure only -- replace with the actual Odyssey modules.
    #
    # module purge
    # module load <gcc-module>
    # module load <cmake-module>
    # module load <ninja-module>
    # module load <boost-module>
    # module load <rocm-module>
    #
    # export ROCM_PATH=/path/exported/by/the/rocm/module
    :
}

modules_aurora() {
    # Aurora already provides DPC++/oneAPI. Do not build AdaptiveCpp here.
    #
    # Load/source the site-supported Intel environment, for example:
    #
    # module purge
    # module load <intel-oneapi/dpcpp-module>
    #
    # or:
    # source <site-oneapi-setvars-script>
    :
}

###############################################################################
# END SITE MODULES
###############################################################################

usage() {
    cat >&2 <<'EOF'
usage: ./sycl/setup-cluster-sycl.sh <polaris|odyssey|aurora> [all|acpp|dpcpp]

Examples:
  ./sycl/setup-cluster-sycl.sh polaris
  ./sycl/setup-cluster-sycl.sh odyssey
  ./sycl/setup-cluster-sycl.sh aurora
  ./sycl/setup-cluster-sycl.sh polaris acpp
  ./sycl/setup-cluster-sycl.sh odyssey dpcpp

Environment:
  TOOLCHAIN_ROOT       installed compiler/toolchain root
                       default: $HOME/.local/quantom-toolchains
  TOOLCHAIN_WORK_ROOT  source/build workspace
                       default: $TOOLCHAIN_ROOT/work
  JOBS                 parallel build jobs; default: 8
  LLVM_VERSION         LLVM used for AdaptiveCpp; default: 21.1.8
  ACPP_REF             AdaptiveCpp git tag/ref; default: v25.10.0
  DPCPP_REF            intel/llvm git ref; default: sycl
  FORCE_TOOLCHAIN=1    rebuild an already-present compiler toolchain
EOF
}

[[ $# -ge 1 && $# -le 2 ]] || {
    usage
    exit 2
}

SITE="$1"
REQUESTED="${2:-all}"

case "$REQUESTED" in
    all|acpp|dpcpp) ;;
    *)
        echo "ERROR: implementation must be all, acpp, or dpcpp" >&2
        usage
        exit 2
        ;;
esac

case "$SITE" in
    polaris)
        GPU_BACKEND=cuda
        GPU_ARCH=sm_80
        LLVM_TARGETS=cpu,cuda
        DPCPP_TARGETS=cuda
        ACPP_VARIANT=acpp-polaris-a100
        DPCPP_VARIANT=dpcpp-polaris-a100
        modules_polaris
        ;;
    odyssey)
        GPU_BACKEND=hip
        GPU_ARCH=gfx942
        LLVM_TARGETS=cpu,hip
        DPCPP_TARGETS=hip
        ACPP_VARIANT=acpp-odyssey-mi300a
        DPCPP_VARIANT=dpcpp-odyssey-mi300a
        modules_odyssey
        ;;
    aurora)
        GPU_BACKEND=xpu
        GPU_ARCH=
        LLVM_TARGETS=
        DPCPP_TARGETS=xpu
        ACPP_VARIANT=
        DPCPP_VARIANT=dpcpp-aurora-pvc
        modules_aurora

        if [[ "$REQUESTED" == "acpp" ]]; then
            echo "ERROR: AdaptiveCpp is intentionally disabled for Aurora." >&2
            exit 2
        fi
        REQUESTED=dpcpp
        ;;
    *)
        echo "ERROR: unknown site '$SITE'" >&2
        usage
        exit 2
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$HOME/.local/quantom-toolchains}"
TOOLCHAIN_WORK_ROOT="${TOOLCHAIN_WORK_ROOT:-$TOOLCHAIN_ROOT/work}"
JOBS="${JOBS:-8}"
LLVM_VERSION="${LLVM_VERSION:-21.1.8}"
ACPP_REF="${ACPP_REF:-v25.10.0}"
DPCPP_REF="${DPCPP_REF:-sycl}"
FORCE_TOOLCHAIN="${FORCE_TOOLCHAIN:-0}"

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: JOBS must be a positive integer" >&2
    exit 2
}

mkdir -p "$TOOLCHAIN_ROOT" "$TOOLCHAIN_WORK_ROOT"

SITE_ROOT="$TOOLCHAIN_ROOT/$SITE"
mkdir -p "$SITE_ROOT"

LLVM_PREFIX="$SITE_ROOT/llvm-$LLVM_VERSION"
ACPP_PREFIX="$SITE_ROOT/adaptivecpp-${ACPP_REF//\//_}"
DPCPP_PREFIX="$SITE_ROOT/dpcpp"

export LLVM_WORKDIR="${LLVM_WORKDIR:-$TOOLCHAIN_WORK_ROOT/$SITE/llvm}"
export ACPP_WORKDIR="${ACPP_WORKDIR:-$TOOLCHAIN_WORK_ROOT/$SITE/acpp}"
export DPCPP_WORKDIR="${DPCPP_WORKDIR:-$TOOLCHAIN_WORK_ROOT/$SITE/dpcpp}"

resolve_vendor_paths() {
    if [[ "$GPU_BACKEND" == "cuda" ]]; then
        if [[ -z "${CUDA_PATH:-}" ]]; then
            if [[ -n "${CUDA_HOME:-}" ]]; then
                export CUDA_PATH="$CUDA_HOME"
            elif [[ -n "${CUDA_ROOT:-}" ]]; then
                export CUDA_PATH="$CUDA_ROOT"
            elif command -v nvcc >/dev/null 2>&1; then
                export CUDA_PATH="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
            fi
        fi

        if [[ -z "${CUDA_PATH:-}" || ! -d "$CUDA_PATH" ]]; then
            cat >&2 <<'EOF'
ERROR: CUDA_PATH could not be resolved.

Update modules_polaris() so the CUDA module is loaded and either:
  * CUDA_HOME/CUDA_ROOT is exported, or
  * CUDA_PATH is set explicitly.

Example:
  export CUDA_PATH=/path/to/cuda
EOF
            exit 2
        fi
    elif [[ "$GPU_BACKEND" == "hip" ]]; then
        if [[ -z "${ROCM_PATH:-}" ]]; then
            if [[ -n "${ROCM_HOME:-}" ]]; then
                export ROCM_PATH="$ROCM_HOME"
            elif command -v hipcc >/dev/null 2>&1; then
                export ROCM_PATH="$(cd "$(dirname "$(command -v hipcc)")/.." && pwd)"
            fi
        fi

        if [[ -z "${ROCM_PATH:-}" || ! -d "$ROCM_PATH" ]]; then
            cat >&2 <<'EOF'
ERROR: ROCM_PATH could not be resolved.

Update modules_odyssey() so the ROCm module is loaded and either:
  * ROCM_HOME is exported, or
  * ROCM_PATH is set explicitly.

Example:
  export ROCM_PATH=/path/to/rocm
EOF
            exit 2
        fi
    fi
}

require_common_build_tools() {
    local missing=0
    for tool in git cmake ninja python3; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "ERROR: required build tool not found: $tool" >&2
            missing=1
        fi
    done
    (( missing == 0 )) || exit 127
}

print_config() {
    echo "============================================================"
    echo "QuantOm SYCL setup"
    echo "============================================================"
    echo "site:                $SITE"
    echo "requested:           $REQUESTED"
    echo "backend:             $GPU_BACKEND"
    [[ -z "$GPU_ARCH" ]] || echo "architecture:        $GPU_ARCH"
    echo "toolchain root:      $TOOLCHAIN_ROOT"
    echo "toolchain work root: $TOOLCHAIN_WORK_ROOT"
    echo "jobs:                $JOBS"

    if [[ "$SITE" != "aurora" ]]; then
        echo "LLVM:                $LLVM_VERSION"
        echo "LLVM prefix:         $LLVM_PREFIX"
        echo "AdaptiveCpp ref:     $ACPP_REF"
        echo "AdaptiveCpp prefix:  $ACPP_PREFIX"
        echo "DPC++ ref:           $DPCPP_REF"
        echo "DPC++ prefix:        $DPCPP_PREFIX"
    else
        echo "DPC++:               site-provided"
    fi

    [[ -z "${CUDA_PATH:-}" ]] || echo "CUDA_PATH:           $CUDA_PATH"
    [[ -z "${ROCM_PATH:-}" ]] || echo "ROCM_PATH:           $ROCM_PATH"
    echo "============================================================"
}

install_llvm_for_acpp() {
    if [[ "$FORCE_TOOLCHAIN" != "1" && -x "$LLVM_PREFIX/bin/clang++" && -x "$LLVM_PREFIX/bin/llvm-config" ]]; then
        echo "LLVM already present: $LLVM_PREFIX"
        return
    fi

    echo
    echo "=== Building LLVM $LLVM_VERSION for AdaptiveCpp ==="
    make install-llvm \
        LLVM_PREFIX="$LLVM_PREFIX" \
        LLVM_TARGETS="$LLVM_TARGETS" \
        LLVM_VERSION="$LLVM_VERSION" \
        LLVM_JOBS="$JOBS"
}

install_acpp() {
    if [[ "$FORCE_TOOLCHAIN" != "1" && -x "$ACPP_PREFIX/bin/acpp" ]]; then
        echo "AdaptiveCpp already present: $ACPP_PREFIX"
        return
    fi

    echo
    echo "=== Building AdaptiveCpp $ACPP_REF ==="
    make install-acpp \
        LLVM_PREFIX="$LLVM_PREFIX" \
        ACPP_PREFIX="$ACPP_PREFIX" \
        ACPP_REF="$ACPP_REF" \
        ACPP_JOBS="$JOBS"
}

install_dpcpp() {
    if [[ "$FORCE_TOOLCHAIN" != "1" &&
          -x "$DPCPP_PREFIX/bin/clang++" &&
          -d "$DPCPP_PREFIX/lib" ]]; then
        echo "DPC++ already present: $DPCPP_PREFIX"
        return
    fi

    echo
    echo "=== Building DPC++ ($DPCPP_TARGETS) ==="
    make install-dpcpp \
        DPCPP_PREFIX="$DPCPP_PREFIX" \
        DPCPP_TARGETS="$DPCPP_TARGETS" \
        DPCPP_REF="$DPCPP_REF" \
        DPCPP_JOBS="$JOBS"
}

build_acpp_backend() {
    echo
    echo "=== Building QuantOm AdaptiveCpp backend: $ACPP_VARIANT ==="

    if [[ "$GPU_BACKEND" == "cuda" ]]; then
        ACPP_PREFIX="$ACPP_PREFIX" \
        make build-sycl-acpp \
            SYCL_VARIANT="$ACPP_VARIANT" \
            SYCL_TARGET=cuda \
            SYCL_ARCH="$GPU_ARCH"
    elif [[ "$GPU_BACKEND" == "hip" ]]; then
        ACPP_PREFIX="$ACPP_PREFIX" \
        make build-sycl-acpp \
            SYCL_VARIANT="$ACPP_VARIANT" \
            SYCL_TARGET=hip \
            SYCL_ARCH="$GPU_ARCH"
    else
        echo "ERROR: unsupported AdaptiveCpp site backend: $GPU_BACKEND" >&2
        exit 2
    fi
}

build_dpcpp_backend() {
    echo
    echo "=== Building QuantOm DPC++ backend: $DPCPP_VARIANT ==="

    if [[ "$SITE" == "aurora" ]]; then
        # build-dpcpp.sh falls back to icpx for xpu. The Aurora module block
        # should put the site-supported DPC++ compiler on PATH.
        if ! command -v "${DPCPP_XPU_CXX:-${DPCPP_CXX:-icpx}}" >/dev/null 2>&1; then
            echo "ERROR: Aurora DPC++ compiler not found." >&2
            echo "Update modules_aurora() or set DPCPP_CXX/DPCPP_XPU_CXX." >&2
            exit 127
        fi

        make build-sycl-dpcpp \
            SYCL_VARIANT="$DPCPP_VARIANT" \
            SYCL_TARGET=xpu
        return
    fi

    if [[ "$GPU_BACKEND" == "cuda" ]]; then
        DPCPP_PREFIX="$DPCPP_PREFIX" \
        make build-sycl-dpcpp \
            SYCL_VARIANT="$DPCPP_VARIANT" \
            SYCL_TARGET=cuda \
            SYCL_ARCH="$GPU_ARCH"
    elif [[ "$GPU_BACKEND" == "hip" ]]; then
        DPCPP_PREFIX="$DPCPP_PREFIX" \
        make build-sycl-dpcpp \
            SYCL_VARIANT="$DPCPP_VARIANT" \
            SYCL_TARGET=hip \
            SYCL_ARCH="$GPU_ARCH"
    else
        echo "ERROR: unsupported DPC++ site backend: $GPU_BACKEND" >&2
        exit 2
    fi
}

# Vendor paths are needed while building both the compiler toolchains and the
# QuantOm backend. Aurora's site DPC++ path is handled separately.
if [[ "$SITE" != "aurora" ]]; then
    resolve_vendor_paths
    require_common_build_tools
fi

print_config

case "$REQUESTED" in
    acpp)
        install_llvm_for_acpp
        install_acpp
        build_acpp_backend
        ;;
    dpcpp)
        if [[ "$SITE" != "aurora" ]]; then
            install_dpcpp
        fi
        build_dpcpp_backend
        ;;
    all)
        # Aurora is normalized to dpcpp above, so this only runs on Polaris
        # and Odyssey.
        install_llvm_for_acpp
        install_acpp
        build_acpp_backend

        install_dpcpp
        build_dpcpp_backend
        ;;
esac

echo
echo "============================================================"
echo "Completed"
echo "============================================================"

if [[ "$REQUESTED" == "acpp" || "$REQUESTED" == "all" ]]; then
    echo "AdaptiveCpp variant: $ACPP_VARIANT"
    echo "  export QUANTOM_SYCL_VARIANT=$ACPP_VARIANT"
fi

if [[ "$REQUESTED" == "dpcpp" || "$REQUESTED" == "all" ]]; then
    echo "DPC++ variant:       $DPCPP_VARIANT"
    echo "  export QUANTOM_SYCL_VARIANT=$DPCPP_VARIANT"
fi

echo
echo "Available builds:"
make list-sycl-builds
