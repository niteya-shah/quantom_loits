#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: install-dpcpp.sh <toolchain-build-dir> <targets> [git-ref]

Build the open-source Intel DPC++ toolchain using Intel's current buildbot
configure.py / compile.py workflow. The build directory is the toolchain: its
bin/ and lib/ directories are used directly, matching Intel's source-build
instructions (their deployment/install documentation is currently incomplete).

<targets> is a comma-separated list containing one or more of:
  cpu   enable the Native CPU backend (--native_cpu)
  xpu   build the normal SPIR/Level Zero-capable DPC++ toolchain
  cuda  enable the CUDA backend (--cuda)
  hip   enable the AMD HIP backend (--hip)

No target is selected by default.

Examples:
  ./sycl/install-dpcpp.sh /shared/toolchains/dpcpp-xpu xpu
  ./sycl/install-dpcpp.sh /shared/toolchains/dpcpp-cuda xpu,cuda <commit-or-tag>
  ./sycl/install-dpcpp.sh /shared/toolchains/dpcpp-hip xpu,hip <commit-or-tag>

Environment overrides:
  DPCPP_REF             git ref if the third positional argument is omitted
  DPCPP_WORKDIR         build/work root
  DPCPP_SOURCE_DIR      intel/llvm checkout prepared by fetch-dpcpp.sh (required)
  DPCPP_JOBS            parallel build jobs (default: all available cores)
  DPCPP_BUILD_TYPE      Debug or Release (default: Release)
  DPCPP_HOST_CC         host C compiler (default: gcc from PATH)
  DPCPP_HOST_CXX        host C++ compiler (default: g++ from PATH)
  DPCPP_CONFIGURE_ARGS  extra whitespace-separated configure.py arguments
  DPCPP_CUPTI_LIBRARY   explicit CUPTI shared library override for CUDA builds
  CUDA_PATH             CUDA toolkit root for a non-default CUDA installation
  ROCM_PATH             ROCm root for a non-default ROCm installation
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

BUILD="$1"
TARGET_SPEC="$2"
REF="${3:-${DPCPP_REF:-}}"
JOBS="${DPCPP_JOBS:-$(nproc)}"
BUILD_TYPE="${DPCPP_BUILD_TYPE:-Release}"
WORK_ROOT="${DPCPP_WORKDIR:-${TMPDIR:-/tmp}/quantom-dpcpp-${USER:-user}}"
SOURCE="${DPCPP_SOURCE_DIR:-}"
CC_BIN="${DPCPP_HOST_CC:-$(command -v gcc || true)}"
CXX_BIN="${DPCPP_HOST_CXX:-$(command -v g++ || true)}"

[[ -n "$BUILD" ]] || { echo "DPC++ toolchain build directory must not be empty" >&2; exit 2; }
[[ -n "$TARGET_SPEC" ]] || { echo "DPC++ targets must not be empty" >&2; exit 2; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "DPCPP_JOBS must be a positive integer" >&2; exit 2; }
case "$BUILD_TYPE" in Debug|Release) ;; *) echo "DPCPP_BUILD_TYPE must be Debug or Release" >&2; exit 2;; esac
[[ -n "$CC_BIN" && -x "$CC_BIN" ]] || { echo "host C compiler not found; set DPCPP_HOST_CC" >&2; exit 127; }
[[ -n "$CXX_BIN" && -x "$CXX_BIN" ]] || { echo "host C++ compiler not found; set DPCPP_HOST_CXX" >&2; exit 127; }

for tool in python cmake ninja; do
  command -v "$tool" >/dev/null 2>&1 || { echo "required tool not found: $tool" >&2; exit 127; }
done

IFS=',' read -r -a REQUESTED_TARGETS <<< "$TARGET_SPEC"
WANT_CPU=0
WANT_XPU=0
WANT_CUDA=0
WANT_HIP=0
for target in "${REQUESTED_TARGETS[@]}"; do
  case "$target" in
    cpu) WANT_CPU=1 ;;
    xpu) WANT_XPU=1 ;;
    cuda) WANT_CUDA=1 ;;
    hip) WANT_HIP=1 ;;
    *)
      echo "invalid DPC++ target '$target'; expected cpu,xpu,cuda,hip" >&2
      exit 2
      ;;
  esac
done

if (( WANT_CUDA )) && [[ -n "${CUDA_PATH:-}" && ! -d "$CUDA_PATH" ]]; then
  echo "CUDA_PATH does not exist: $CUDA_PATH" >&2
  exit 2
fi
if (( WANT_HIP )) && [[ -n "${ROCM_PATH:-}" && ! -d "$ROCM_PATH" ]]; then
  echo "ROCM_PATH does not exist: $ROCM_PATH" >&2
  exit 2
fi

[[ -n "$SOURCE" ]] || {
  echo "DPCPP_SOURCE_DIR is required for the build step." >&2
  echo "Download once with ./sycl/fetch-dpcpp.sh <source-dir> <git-ref>" >&2
  exit 2
}
[[ -f "$SOURCE/buildbot/configure.py" && -f "$SOURCE/buildbot/compile.py" ]] || {
  echo "DPCPP_SOURCE_DIR is not an intel/llvm source tree: $SOURCE" >&2
  echo "Run ./sycl/fetch-dpcpp.sh before building." >&2
  exit 2
}
if [[ -n "$REF" && -f "$SOURCE/.quantom-requested-ref" ]]; then
  CACHED_REF="$(cat "$SOURCE/.quantom-requested-ref")"
  if [[ "$CACHED_REF" != "$REF" ]]; then
    echo "DPC++ source cache contains ref '$CACHED_REF', requested '$REF'." >&2
    echo "Run ./sycl/fetch-dpcpp.sh '$SOURCE' '$REF' before building." >&2
    exit 2
  fi
fi

COMMIT="unknown"
DESCRIBE=""
if [[ -d "$SOURCE/.git" ]] && command -v git >/dev/null 2>&1; then
  COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
  DESCRIBE="$(git -C "$SOURCE" describe --always --dirty --tags 2>/dev/null || true)"
fi

echo "DPC++ source commit: $COMMIT${DESCRIBE:+ ($DESCRIBE)}"

mkdir -p "$BUILD"
read -r -a EXTRA_CONFIGURE <<< "${DPCPP_CONFIGURE_ARGS:-}"

CONFIGURE_CMD=(
  python "$SOURCE/buildbot/configure.py"
  -t "$BUILD_TYPE"
  -o "$BUILD"
  --cmake-gen Ninja
)

(( WANT_CPU )) && CONFIGURE_CMD+=(--native_cpu)
(( WANT_CUDA )) && CONFIGURE_CMD+=(--cuda)
(( WANT_HIP )) && CONFIGURE_CMD+=(--hip)

# XPU/Level Zero uses the normal SPIR-capable DPC++ build; no special
# configure.py flag is required. The Level Zero implementation remains a
# machine/runtime dependency rather than something this source build installs.
if (( WANT_CUDA )) && [[ -n "${CUDA_PATH:-}" ]]; then
  CONFIGURE_CMD+=("-DCUDAToolkit_ROOT=$CUDA_PATH")
fi
if (( WANT_HIP )) && [[ -n "${ROCM_PATH:-}" ]]; then
  CONFIGURE_CMD+=("-DUR_HIP_ROCM_DIR=$ROCM_PATH")
fi
CONFIGURE_CMD+=("${EXTRA_CONFIGURE[@]}")

printf 'Configuring DPC++:'
printf ' %q' "${CONFIGURE_CMD[@]}"
printf '\n'
CC="$CC_BIN" CXX="$CXX_BIN" "${CONFIGURE_CMD[@]}"

repair_cuda_cupti() {
  local cache="$BUILD/CMakeCache.txt"
  local cached=""
  local cupti="${DPCPP_CUPTI_LIBRARY:-}"
  local cupti_static=""

  [[ -f "$cache" ]] || {
    echo "DPC++ configure did not create $cache" >&2
    exit 1
  }

  cached="$(sed -n 's/^CUDA_cupti_LIBRARY:FILEPATH=//p' "$cache" | head -n 1)"
  if [[ -z "$cupti" && -n "$cached" && -f "$cached" ]]; then
    return
  fi

  if [[ -z "$cupti" ]]; then
    for candidate in \
      "$CUDA_PATH/extras/CUPTI/lib64/libcupti.so" \
      "$CUDA_PATH/targets/x86_64-linux/lib/libcupti.so" \
      "$CUDA_PATH/lib64/libcupti.so"; do
      if [[ -f "$candidate" ]]; then
        cupti="$candidate"
        break
      fi
    done
  fi

  [[ -n "$cupti" && -f "$cupti" ]] || {
    echo "DPC++ configured an invalid CUPTI library path: ${cached:-<unset>}" >&2
    echo "Set DPCPP_CUPTI_LIBRARY to the full path to libcupti.so." >&2
    exit 2
  }

  cupti_static="$(dirname "$cupti")/libcupti_static.a"
  CMAKE_REPAIR=(
    cmake
    -S "$SOURCE/llvm"
    -B "$BUILD"
    "-DCUDA_cupti_LIBRARY:FILEPATH=$cupti"
  )
  if [[ -f "$cupti_static" ]]; then
    CMAKE_REPAIR+=("-DCUDA_cupti_static_LIBRARY:FILEPATH=$cupti_static")
  fi

  echo "Repairing DPC++ CUPTI path: $cupti"
  CC="$CC_BIN" CXX="$CXX_BIN" "${CMAKE_REPAIR[@]}"

  cached="$(sed -n 's/^CUDA_cupti_LIBRARY:FILEPATH=//p' "$cache" | head -n 1)"
  [[ -n "$cached" && -f "$cached" ]] || {
    echo "DPC++ CUPTI path remains invalid after CMake repair: ${cached:-<unset>}" >&2
    exit 2
  }
}

if (( WANT_CUDA )) && [[ -n "${CUDA_PATH:-}" ]]; then
  repair_cuda_cupti
fi

COMPILE_CMD=(
  python "$SOURCE/buildbot/compile.py"
  -o "$BUILD"
  -j "$JOBS"
)
printf 'Building DPC++:'
printf ' %q' "${COMPILE_CMD[@]}"
printf '\n'
CC="$CC_BIN" CXX="$CXX_BIN" "${COMPILE_CMD[@]}"

for required in "$BUILD/bin/clang" "$BUILD/bin/clang++"; do
  [[ -x "$required" ]] || { echo "DPC++ build missing required compiler: $required" >&2; exit 1; }
done

if ! compgen -G "$BUILD/lib/libsycl.so*" >/dev/null; then
  echo "DPC++ build missing libsycl under $BUILD/lib" >&2
  exit 1
fi

cat > "$BUILD/quantom-dpcpp-info.txt" <<INFO
ref=${REF:-sycl}
commit=$COMMIT
targets=$TARGET_SPEC
build_type=$BUILD_TYPE
host_cc=$CC_BIN
host_cxx=$CXX_BIN
cuda_path=${CUDA_PATH:-}
rocm_path=${ROCM_PATH:-}
INFO

echo
echo "DPC++ toolchain built successfully"
echo "  build/toolchain dir: $BUILD"
echo "  compiler:            $BUILD/bin/clang++"
echo "  targets:             $TARGET_SPEC"
echo
echo "Use this toolchain for QuantOm with:"
echo "  export DPCPP_PREFIX='$BUILD'"
