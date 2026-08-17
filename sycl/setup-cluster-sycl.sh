#!/usr/bin/env bash
set -euo pipefail

# QuantOm SYCL cluster helper.
#
# Source download and compiler builds are deliberately separate:
#
#   ./sycl/setup-cluster-sycl.sh polaris fetch all
#   ./sycl/setup-cluster-sycl.sh polaris build all
#
#   ./sycl/setup-cluster-sycl.sh odyssey fetch all
#   ./sycl/setup-cluster-sycl.sh odyssey build all
#
# Fetching is network-facing and idempotent. Building performs no git fetch or
# clone. LLVM and AdaptiveCpp build directories are preserved, so rerunning an
# interrupted build resumes the existing Ninja/CMake build instead of deleting it.
#
# Aurora uses the site DPC++ toolchain and therefore only needs:
#
#   ./sycl/setup-cluster-sycl.sh aurora build dpcpp

###############################################################################
# SITE MODULES
#
# Fill these in once the correct module stacks are known. Fetch mode does not
# invoke these functions, so source can be downloaded independently of the GPU
# software environment.
###############################################################################

modules_polaris() {
    # module purge
    # module load <gcc-module>
    # module load <cmake-module>
    # module load <ninja/module or conda environment>
    # module load <boost-module>
    # module load <cuda-module>
    # export CUDA_PATH=/path/to/cuda   # only if the module does not export it
    :
}

modules_odyssey() {
+       module load gcc/13.2
+       module load cmake/3.31.1
+       module load boost/1.75
+       module load hwloc/2.4.0
+       module load rocm/7.2.4
   :
}

modules_aurora() {
    # Load/source Aurora's site-supported oneAPI/DPC++ environment here.
    # AdaptiveCpp is intentionally not built on Aurora.
    :
}

###############################################################################
# END SITE MODULES
###############################################################################

usage() {
    cat >&2 <<'USAGE'
usage: ./sycl/setup-cluster-sycl.sh <polaris|odyssey|aurora> <fetch|build> [all|acpp|dpcpp]

Examples:
  ./sycl/setup-cluster-sycl.sh polaris fetch all
  ./sycl/setup-cluster-sycl.sh polaris build all
  ./sycl/setup-cluster-sycl.sh odyssey fetch acpp
  ./sycl/setup-cluster-sycl.sh odyssey build acpp
  ./sycl/setup-cluster-sycl.sh aurora build dpcpp

fetch:
  Downloads/checks out source only. It does not load site GPU modules or build.
  Re-running fetch does not contact the network when the requested ref is
  already cached locally. Set FETCH_UPDATE=1 to explicitly refresh it.

build:
  Never clones or fetches source. It uses the cached source trees and preserves
  build directories so interrupted LLVM/AdaptiveCpp/DPC++ builds can resume.

Environment:
  TOOLCHAIN_ROOT        installed toolchain root
                        default: $HOME/.local/quantom-toolchains
  TOOLCHAIN_SOURCE_ROOT cached source root
                        default: $TOOLCHAIN_ROOT/sources
  TOOLCHAIN_WORK_ROOT   persistent build workspace
                        default: $TOOLCHAIN_ROOT/work
  JOBS                  parallel build jobs; default: 8
  LLVM_VERSION          LLVM for AdaptiveCpp; default: 20.1.8
  ACPP_REF              AdaptiveCpp ref; default: v25.10.0
  DPCPP_REF             intel/llvm ref; default: sycl
  FETCH_UPDATE=1        explicitly update an already-cached source ref
  FORCE_TOOLCHAIN=1     rerun compiler build even if install looks complete;
                        this remains incremental and does not delete build dirs
USAGE
}

[[ $# -ge 2 && $# -le 3 ]] || {
    usage
    exit 2
}

SITE="$1"
ACTION="$2"
REQUESTED="${3:-all}"

case "$ACTION" in
    fetch|build) ;;
    *)
        echo "ERROR: action must be fetch or build" >&2
        usage
        exit 2
        ;;
esac

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
        ;;
    odyssey)
        GPU_BACKEND=hip
        GPU_ARCH=gfx942
        LLVM_TARGETS=cpu,hip
        DPCPP_TARGETS=hip
        ACPP_VARIANT=acpp-odyssey-mi300a
        DPCPP_VARIANT=dpcpp-odyssey-mi300a
        ;;
    aurora)
        GPU_BACKEND=xpu
        GPU_ARCH=
        LLVM_TARGETS=
        DPCPP_TARGETS=xpu
        ACPP_VARIANT=
        DPCPP_VARIANT=dpcpp-aurora-pvc
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
TOOLCHAIN_SOURCE_ROOT="${TOOLCHAIN_SOURCE_ROOT:-$TOOLCHAIN_ROOT/sources}"
TOOLCHAIN_WORK_ROOT="${TOOLCHAIN_WORK_ROOT:-$TOOLCHAIN_ROOT/work}"
JOBS="${JOBS:-8}"
LLVM_VERSION="${LLVM_VERSION:-20.1.8}"
ACPP_REF="${ACPP_REF:-v25.10.0}"
DPCPP_REF="${DPCPP_REF:-sycl}"
FORCE_TOOLCHAIN="${FORCE_TOOLCHAIN:-0}"

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: JOBS must be a positive integer" >&2
    exit 2
}

mkdir -p "$TOOLCHAIN_ROOT" "$TOOLCHAIN_SOURCE_ROOT" "$TOOLCHAIN_WORK_ROOT"

SITE_ROOT="$TOOLCHAIN_ROOT/$SITE"
mkdir -p "$SITE_ROOT"

sanitize_ref() {
    printf '%s' "$1" | tr '/:@ ' '____'
}

LLVM_PREFIX="$SITE_ROOT/llvm-$LLVM_VERSION"
ACPP_PREFIX="$SITE_ROOT/adaptivecpp-$(sanitize_ref "$ACPP_REF")"
DPCPP_PREFIX="$SITE_ROOT/dpcpp"

export LLVM_WORKDIR="${LLVM_WORKDIR:-$TOOLCHAIN_WORK_ROOT/$SITE/llvm-$LLVM_VERSION}"
export ACPP_WORKDIR="${ACPP_WORKDIR:-$TOOLCHAIN_WORK_ROOT/$SITE/acpp-$(sanitize_ref "$ACPP_REF")-llvm-$LLVM_VERSION}"
export DPCPP_WORKDIR="${DPCPP_WORKDIR:-$TOOLCHAIN_WORK_ROOT/$SITE/dpcpp-$(sanitize_ref "$DPCPP_REF")}"

# Prefer the new shared source cache, but reuse checkouts created by the older
# combined download/build helper when they already exist.
default_source_or_legacy() {
    local preferred="$1"
    local legacy="$2"
    local marker="$3"
    if [[ -e "$preferred/$marker" || -d "$preferred/.git" ]]; then
        printf '%s' "$preferred"
    elif [[ -e "$legacy/$marker" || -d "$legacy/.git" ]]; then
        printf '%s' "$legacy"
    else
        printf '%s' "$preferred"
    fi
}

export LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-$(default_source_or_legacy \
    "$TOOLCHAIN_SOURCE_ROOT/llvm-project-$LLVM_VERSION" \
    "$TOOLCHAIN_WORK_ROOT/$SITE/llvm/source" \
    'llvm/CMakeLists.txt')}"
export ACPP_SOURCE_DIR="${ACPP_SOURCE_DIR:-$(default_source_or_legacy \
    "$TOOLCHAIN_SOURCE_ROOT/AdaptiveCpp-$(sanitize_ref "$ACPP_REF")" \
    "$TOOLCHAIN_WORK_ROOT/$SITE/acpp/source" \
    'CMakeLists.txt')}"
export DPCPP_SOURCE_DIR="${DPCPP_SOURCE_DIR:-$(default_source_or_legacy \
    "$TOOLCHAIN_SOURCE_ROOT/intel-llvm-$(sanitize_ref "$DPCPP_REF")" \
    "$TOOLCHAIN_WORK_ROOT/$SITE/dpcpp/source" \
    'buildbot/configure.py')}"


configure_host_gcc() {
    local host_cc host_cxx gcc_crt gcc_libstdcpp gcc_libdir

    host_cc="${LLVM_C_COMPILER:-$(command -v gcc || true)}"
    host_cxx="${LLVM_CXX_COMPILER:-$(command -v g++ || true)}"

    [[ -n "$host_cc" && -x "$host_cc" ]] || {
        echo "ERROR: gcc not found after loading the site modules." >&2
        exit 127
    }
    [[ -n "$host_cxx" && -x "$host_cxx" ]] || {
        echo "ERROR: g++ not found after loading the site modules." >&2
        exit 127
    }

    gcc_crt="$("$host_cxx" -print-file-name=crtbegin.o)"
    gcc_libstdcpp="$("$host_cxx" -print-file-name=libstdc++.so.6)"

    [[ "$gcc_crt" = /* && -f "$gcc_crt" ]] || {
        echo "ERROR: could not resolve crtbegin.o from $host_cxx" >&2
        exit 2
    }
    [[ "$gcc_libstdcpp" = /* && -f "$gcc_libstdcpp" ]] || {
        echo "ERROR: could not resolve libstdc++.so.6 from $host_cxx" >&2
        exit 2
    }

    export LLVM_C_COMPILER="${LLVM_C_COMPILER:-$host_cc}"
    export LLVM_CXX_COMPILER="${LLVM_CXX_COMPILER:-$host_cxx}"
    export DPCPP_HOST_CC="${DPCPP_HOST_CC:-$host_cc}"
    export DPCPP_HOST_CXX="${DPCPP_HOST_CXX:-$host_cxx}"
    export ACPP_GCC_INSTALL_DIR="${ACPP_GCC_INSTALL_DIR:-$(dirname "$gcc_crt")}"

    gcc_libdir="$(dirname "$gcc_libstdcpp")"
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$gcc_libdir:"*) ;;
        *) export LD_LIBRARY_PATH="$gcc_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}

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
        [[ -n "${CUDA_PATH:-}" && -d "$CUDA_PATH" ]] || {
            echo "ERROR: CUDA_PATH could not be resolved; update modules_polaris()." >&2
            exit 2
        }
    elif [[ "$GPU_BACKEND" == "hip" ]]; then
        if [[ -z "${ROCM_PATH:-}" ]]; then
            if [[ -n "${ROCM_HOME:-}" ]]; then
                export ROCM_PATH="$ROCM_HOME"
            elif command -v hipcc >/dev/null 2>&1; then
                export ROCM_PATH="$(cd "$(dirname "$(command -v hipcc)")/.." && pwd)"
            fi
        fi
        [[ -n "${ROCM_PATH:-}" && -d "$ROCM_PATH" ]] || {
            echo "ERROR: ROCM_PATH could not be resolved; update modules_odyssey()." >&2
            exit 2
        }
    fi
}

print_config() {
    echo "============================================================"
    echo "QuantOm SYCL setup"
    echo "============================================================"
    echo "site:                $SITE"
    echo "action:              $ACTION"
    echo "requested:           $REQUESTED"
    echo "backend:             $GPU_BACKEND"
    [[ -z "$GPU_ARCH" ]] || echo "architecture:        $GPU_ARCH"
    echo "toolchain root:      $TOOLCHAIN_ROOT"
    echo "source root:         $TOOLCHAIN_SOURCE_ROOT"
    echo "work root:           $TOOLCHAIN_WORK_ROOT"
    if [[ "$SITE" != "aurora" ]]; then
        echo "LLVM:                $LLVM_VERSION"
        echo "LLVM source:         $LLVM_SOURCE_DIR"
        echo "LLVM prefix:         $LLVM_PREFIX"
        echo "AdaptiveCpp ref:     $ACPP_REF"
        echo "AdaptiveCpp source:  $ACPP_SOURCE_DIR"
        echo "AdaptiveCpp prefix:  $ACPP_PREFIX"
        echo "DPC++ ref:           $DPCPP_REF"
        echo "DPC++ source:        $DPCPP_SOURCE_DIR"
        echo "DPC++ prefix:        $DPCPP_PREFIX"
    else
        echo "DPC++:               site-provided"
    fi
    [[ -z "${CUDA_PATH:-}" ]] || echo "CUDA_PATH:           $CUDA_PATH"
    [[ -z "${ROCM_PATH:-}" ]] || echo "ROCM_PATH:           $ROCM_PATH"
    echo "============================================================"
}

fetch_acpp_sources() {
    ./sycl/fetch-llvm.sh "$LLVM_SOURCE_DIR" "$LLVM_VERSION"
    ./sycl/fetch-acpp.sh "$ACPP_SOURCE_DIR" "$ACPP_REF"
}

fetch_dpcpp_source() {
    ./sycl/fetch-dpcpp.sh "$DPCPP_SOURCE_DIR" "$DPCPP_REF"
}

install_llvm_for_acpp() {
    if [[ "$FORCE_TOOLCHAIN" != "1" && -x "$LLVM_PREFIX/bin/clang++" && -x "$LLVM_PREFIX/bin/llvm-config" ]]; then
        echo "LLVM already installed: $LLVM_PREFIX"
        return
    fi

    echo
    echo "=== Building/resuming LLVM $LLVM_VERSION for AdaptiveCpp ==="
    LLVM_JOBS="$JOBS" ./sycl/install-llvm.sh \
        "$LLVM_PREFIX" "$LLVM_TARGETS" "$LLVM_VERSION"
}

install_acpp() {
    if [[ "$FORCE_TOOLCHAIN" != "1" && -x "$ACPP_PREFIX/bin/acpp" ]]; then
        echo "AdaptiveCpp already installed: $ACPP_PREFIX"
        return
    fi

    echo
    echo "=== Building/resuming AdaptiveCpp $ACPP_REF ==="
    ACPP_REF="$ACPP_REF" ACPP_JOBS="$JOBS" ./sycl/install-acpp.sh \
        "$ACPP_PREFIX" "$LLVM_PREFIX" "$ACPP_REF"
}

install_dpcpp() {
    if [[ "$FORCE_TOOLCHAIN" != "1" && -x "$DPCPP_PREFIX/bin/clang++" && -d "$DPCPP_PREFIX/lib" ]]; then
        echo "DPC++ already built: $DPCPP_PREFIX"
        return
    fi

    echo
    echo "=== Building/resuming DPC++ ($DPCPP_TARGETS) ==="
    DPCPP_REF="$DPCPP_REF" DPCPP_JOBS="$JOBS" ./sycl/install-dpcpp.sh \
        "$DPCPP_PREFIX" "$DPCPP_TARGETS" "$DPCPP_REF"
}

build_acpp_backend() {
    echo
    echo "=== Building QuantOm AdaptiveCpp backend: $ACPP_VARIANT ==="
    ACPP_PREFIX="$ACPP_PREFIX" make build-sycl-acpp \
        SYCL_VARIANT="$ACPP_VARIANT" \
        SYCL_TARGET="$GPU_BACKEND" \
        SYCL_ARCH="$GPU_ARCH"
}

build_dpcpp_backend() {
    echo
    echo "=== Building QuantOm DPC++ backend: $DPCPP_VARIANT ==="

    if [[ "$SITE" == "aurora" ]]; then
        if ! command -v "${DPCPP_XPU_CXX:-${DPCPP_CXX:-icpx}}" >/dev/null 2>&1; then
            echo "ERROR: Aurora DPC++ compiler not found; update modules_aurora()." >&2
            exit 127
        fi
        make build-sycl-dpcpp SYCL_VARIANT="$DPCPP_VARIANT" SYCL_TARGET=xpu
    else
        DPCPP_PREFIX="$DPCPP_PREFIX" make build-sycl-dpcpp \
            SYCL_VARIANT="$DPCPP_VARIANT" \
            SYCL_TARGET="$GPU_BACKEND" \
            SYCL_ARCH="$GPU_ARCH"
    fi
}

if [[ "$ACTION" == "fetch" ]]; then
    command -v git >/dev/null 2>&1 || { echo "ERROR: git is required for fetch mode" >&2; exit 127; }
    print_config

    if [[ "$SITE" == "aurora" ]]; then
        echo "Aurora uses the site DPC++ installation; there is no compiler source to fetch."
        exit 0
    fi

    case "$REQUESTED" in
        acpp)  fetch_acpp_sources ;;
        dpcpp) fetch_dpcpp_source ;;
        all)
            fetch_acpp_sources
            fetch_dpcpp_source
            ;;
    esac

    echo
    echo "Source download complete. Build later with:"
    echo "  ./sycl/setup-cluster-sycl.sh $SITE build $REQUESTED"
    exit 0
fi

# Build mode starts here. There are deliberately no git clone/fetch operations
# below this point.
case "$SITE" in
    polaris) modules_polaris ;;
    odyssey) modules_odyssey ;;
    aurora)  modules_aurora ;;
esac

if [[ "$SITE" != "aurora" ]]; then
    resolve_vendor_paths
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
