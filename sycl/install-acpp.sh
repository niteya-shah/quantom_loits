#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: install-acpp.sh <install-prefix> <llvm-prefix> [git-ref]

Build and install AdaptiveCpp against an explicit LLVM installation.
QuantOm intentionally does not fall back to a system LLVM here: the old
artifact required a controlled LLVM toolchain, and this installer preserves
that behavior without machine-specific paths.

Examples:
  ./sycl/install-acpp.sh /shared/toolchains/adaptivecpp /shared/toolchains/llvm-20.1.8
  ./sycl/install-acpp.sh /shared/toolchains/adaptivecpp /shared/toolchains/llvm-20.1.8 <tag-or-commit>

Environment overrides:
  ACPP_REF          git ref if the third positional argument is omitted
  ACPP_WORKDIR      build root; preserved across restarts
  ACPP_SOURCE_DIR   AdaptiveCpp checkout prepared by fetch-acpp.sh (required)
  ACPP_JOBS         parallel build jobs (default: all available cores)
  ACPP_BUILD_TYPE   CMake build type (default: Release)
  ACPP_CMAKE_ARGS   additional whitespace-separated CMake arguments
  ACPP_CPU_ONLY     set to 1 to build only the OpenMP CPU runtime/backend;
                    CUDA, ROCm, OpenCL, Level Zero, and Vulkan are forced off
  ACPP_GCC_INSTALL_DIR
                    GCC installation selected for Clang's host C/C++ runtime.
                    The cluster helper derives this from g++ on PATH.
  CUDA_PATH         optional CUDA toolkit root passed to AdaptiveCpp
  ROCM_PATH         optional ROCm root passed to AdaptiveCpp
  ACPP_EXPERIMENTAL_LLVM set to 1 to pass -DACPP_EXPERIMENTAL_LLVM=ON for LLVM >20
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

PREFIX="$1"
LLVM_PREFIX="$2"
REF="${3:-${ACPP_REF:-}}"
JOBS="${ACPP_JOBS:-$(nproc)}"
BUILD_TYPE="${ACPP_BUILD_TYPE:-Release}"
CPU_ONLY="${ACPP_CPU_ONLY:-0}"
WORK_ROOT="${ACPP_WORKDIR:-${TMPDIR:-/tmp}/quantom-adaptivecpp-${USER:-user}}"
SOURCE="${ACPP_SOURCE_DIR:-}"
BUILD="$WORK_ROOT/build"

[[ -n "$PREFIX" ]] || { echo "install prefix must not be empty" >&2; exit 2; }
[[ -n "$LLVM_PREFIX" ]] || { echo "LLVM prefix must not be empty" >&2; exit 2; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "ACPP_JOBS must be a positive integer" >&2; exit 2; }
[[ "$CPU_ONLY" == "0" || "$CPU_ONLY" == "1" ]] || { echo "ACPP_CPU_ONLY must be 0 or 1" >&2; exit 2; }

LLVM_CLANG="$LLVM_PREFIX/bin/clang"
LLVM_CLANGXX="$LLVM_PREFIX/bin/clang++"
LLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
CLANG_DIR="$LLVM_PREFIX/lib/cmake/clang"

for required in \
  "$LLVM_CLANG" \
  "$LLVM_CLANGXX" \
  "$LLVM_DIR/LLVMConfig.cmake" \
  "$CLANG_DIR/ClangConfig.cmake"; do
  [[ -e "$required" ]] || {
    echo "custom LLVM installation is incomplete: missing $required" >&2
    echo "build it first with ./sycl/install-llvm.sh" >&2
    exit 2
  }
done

if ! compgen -G "$LLVM_PREFIX/lib/libLLVM.so*" >/dev/null; then
  echo "custom LLVM installation is incomplete: no libLLVM.so under $LLVM_PREFIX/lib" >&2
  exit 2
fi

LLVM_CONFIG="$LLVM_PREFIX/bin/llvm-config"
if [[ -x "$LLVM_CONFIG" ]]; then
  LLVM_VERSION_ACTUAL="$($LLVM_CONFIG --version)"
  LLVM_MAJOR="${LLVM_VERSION_ACTUAL%%.*}"
  if [[ "$LLVM_MAJOR" =~ ^[0-9]+$ ]] && (( LLVM_MAJOR < 15 )); then
    echo "LLVM $LLVM_VERSION_ACTUAL is too old for this AdaptiveCpp configuration." >&2
    exit 2
  fi
  if [[ "$LLVM_MAJOR" =~ ^[0-9]+$ ]] && (( LLVM_MAJOR > 20 )) && [[ "${ACPP_EXPERIMENTAL_LLVM:-0}" != "1" ]]; then
    echo "AdaptiveCpp v25.10 rejects LLVM >20 by default (found $LLVM_VERSION_ACTUAL)." >&2
    echo "Use LLVM 20 for the supported path, or set ACPP_EXPERIMENTAL_LLVM=1 intentionally." >&2
    exit 2
  fi
fi

for tool in cmake python3; do
  command -v "$tool" >/dev/null 2>&1 || { echo "required tool not found: $tool" >&2; exit 127; }
done

[[ -n "$SOURCE" ]] || {
  echo "ACPP_SOURCE_DIR is required for the build step." >&2
  echo "Download once with ./sycl/fetch-acpp.sh <source-dir> <git-ref>" >&2
  exit 2
}
[[ -f "$SOURCE/CMakeLists.txt" ]] || {
  echo "ACPP_SOURCE_DIR is not an AdaptiveCpp source tree: $SOURCE" >&2
  echo "Run ./sycl/fetch-acpp.sh before building." >&2
  exit 2
}

if [[ -n "$REF" && -f "$SOURCE/.quantom-requested-ref" ]]; then
  CACHED_REF="$(cat "$SOURCE/.quantom-requested-ref")"
  if [[ "$CACHED_REF" != "$REF" ]]; then
    echo "AdaptiveCpp source cache contains ref '$CACHED_REF', requested '$REF'." >&2
    echo "Run ./sycl/fetch-acpp.sh '$SOURCE' '$REF' before building." >&2
    exit 2
  fi
fi

COMMIT="unknown"
DESCRIBE=""
if [[ -d "$SOURCE/.git" ]] && command -v git >/dev/null 2>&1; then
  COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
  DESCRIBE="$(git -C "$SOURCE" describe --always --dirty --tags 2>/dev/null || true)"
  echo "AdaptiveCpp source commit: $COMMIT${DESCRIBE:+ ($DESCRIBE)}"
fi

mkdir -p "$PREFIX" "$BUILD"
if [[ -f "$BUILD/CMakeCache.txt" ]]; then
  echo "Resuming existing AdaptiveCpp build directory: $BUILD"
else
  echo "Creating AdaptiveCpp build directory: $BUILD"
fi

# Preserve the important part of the old working setup: AdaptiveCpp is built
# by, and linked against, the same controlled LLVM installation.
export PATH="$LLVM_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -n "${ACPP_GCC_INSTALL_DIR:-}" && ! -d "$ACPP_GCC_INSTALL_DIR" ]]; then
  echo "ACPP_GCC_INSTALL_DIR does not exist: $ACPP_GCC_INSTALL_DIR" >&2
  exit 2
fi

read -r -a EXTRA_CMAKE <<< "${ACPP_CMAKE_ARGS:-}"

CMAKE_CMD=(
  cmake
  -S "$SOURCE"
  -B "$BUILD"
  "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
  "-DCMAKE_INSTALL_PREFIX=$PREFIX"
  "-DCMAKE_C_COMPILER=$LLVM_CLANG"
  "-DCMAKE_CXX_COMPILER=$LLVM_CLANGXX"
  "-DLLVM_DIR=$LLVM_DIR"
  "-DCLANG_EXECUTABLE_PATH=$LLVM_CLANGXX"
  -DACPP_COMPILER_FEATURE_PROFILE=full
)

# The custom Clang used to build AdaptiveCpp can otherwise discover a different
# system GCC than the one selected by the cluster module environment. Pin the
# GCC installation explicitly so crt objects, libgcc, libstdc++, and headers
# all come from the intended host GCC.
if [[ -n "${ACPP_GCC_INSTALL_DIR:-}" ]]; then
  GCC_DRIVER_FLAG="--gcc-install-dir=$ACPP_GCC_INSTALL_DIR"
  CMAKE_CMD+=(
    "-DCMAKE_C_FLAGS=$GCC_DRIVER_FLAG"
    "-DCMAKE_CXX_FLAGS=$GCC_DRIVER_FLAG"
    "-DCMAKE_EXE_LINKER_FLAGS=$GCC_DRIVER_FLAG"
    "-DCMAKE_SHARED_LINKER_FLAGS=$GCC_DRIVER_FLAG"
  )
fi

if [[ "${ACPP_EXPERIMENTAL_LLVM:-0}" == "1" ]]; then
  CMAKE_CMD+=(-DACPP_EXPERIMENTAL_LLVM=ON)
fi

if [[ "$CPU_ONLY" != "1" ]]; then
  if [[ -n "${CUDA_PATH:-}" ]]; then
    CMAKE_CMD+=("-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH" -DWITH_CUDA_BACKEND=ON)
  fi
  if [[ -n "${ROCM_PATH:-}" ]]; then
    CMAKE_CMD+=("-DROCM_PATH=$ROCM_PATH" -DWITH_ROCM_BACKEND=ON)
  fi
fi

CMAKE_CMD+=("${EXTRA_CMAKE[@]}")

if [[ "$CPU_ONLY" == "1" ]]; then
  CMAKE_CMD+=(
    -DWITH_CUDA_BACKEND=OFF
    -DWITH_ROCM_BACKEND=OFF
    -DWITH_OPENCL_BACKEND=OFF
    -DWITH_LEVEL_ZERO_BACKEND=OFF
    -DWITH_VULKAN_BACKEND=OFF
  )
fi

printf 'Configuring AdaptiveCpp:'
printf ' %q' "${CMAKE_CMD[@]}"
printf '\n'
"${CMAKE_CMD[@]}"

if [[ -n "${ACPP_GCC_INSTALL_DIR:-}" ]]; then
  for cache_var in CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS; do
    if ! grep -Fq "${cache_var}:STRING=--gcc-install-dir=$ACPP_GCC_INSTALL_DIR" "$BUILD/CMakeCache.txt"; then
      echo "ERROR: CMake did not retain the requested GCC selection in $cache_var." >&2
      echo "Expected --gcc-install-dir=$ACPP_GCC_INSTALL_DIR" >&2
      exit 2
    fi
  done
fi

if [[ "$CPU_ONLY" == "1" ]]; then
  for backend in WITH_CUDA_BACKEND WITH_ROCM_BACKEND WITH_OPENCL_BACKEND WITH_LEVEL_ZERO_BACKEND WITH_VULKAN_BACKEND; do
    value="$(sed -n "s/^${backend}:[^=]*=//p" "$BUILD/CMakeCache.txt" | tail -n 1)"
    case "$value" in
      ON|TRUE|1)
        echo "ERROR: $backend is enabled in a CPU-only AdaptiveCpp build." >&2
        exit 2
        ;;
    esac
  done
fi

cmake --build "$BUILD" --target install --parallel "$JOBS"

# AdaptiveCpp v25.10.0 has a host-only rootn implementation that Clang 20
# rejects because T{y} performs narrowing from int to float/double. Keep this
# compatibility fix local to CPU-only installs instead of modifying the shared
# source cache used by accelerator toolchains.
if [[ "$CPU_ONLY" == "1" ]]; then
  HOST_BUILTINS="$PREFIX/include/AdaptiveCpp/hipSYCL/sycl/libkernel/host/builtins.hpp"
  if [[ -f "$HOST_BUILTINS" ]] && grep -Fq 'std::pow(x, T{1}/T{y})' "$HOST_BUILTINS"; then
    sed -i 's/std::pow(x, T{1}\/T{y})/std::pow(x, T{1}\/static_cast<T>(y))/g' "$HOST_BUILTINS"
  fi
fi

ACPP="$PREFIX/bin/acpp"
if [[ ! -x "$ACPP" ]]; then
  echo "AdaptiveCpp install completed but compiler driver was not found: $ACPP" >&2
  exit 1
fi

cat > "$PREFIX/quantom-acpp-info.txt" <<INFO
llvm_prefix=$LLVM_PREFIX
adaptivecpp_ref=${REF:-un-pinned}
adaptivecpp_commit=${COMMIT:-unknown}
cuda_path=${CUDA_PATH:-}
rocm_path=${ROCM_PATH:-}
cpu_only=$CPU_ONLY
gcc_install_dir=${ACPP_GCC_INSTALL_DIR:-}
INFO

echo
echo "AdaptiveCpp installed successfully"
echo "  prefix:   $PREFIX"
echo "  compiler: $ACPP"
echo "  LLVM:     $LLVM_PREFIX"
"$ACPP" --acpp-version || true

echo
echo "Use this installation for QuantOm with:"
echo "  export ACPP_PREFIX='$PREFIX'"
echo "or:"
echo "  export ACPP_CXX='$ACPP'"
