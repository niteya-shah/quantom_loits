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
  ACPP_JOBS         parallel build jobs (default: 4)
  ACPP_BUILD_TYPE   CMake build type (default: Release)
  ACPP_CMAKE_ARGS   additional whitespace-separated CMake arguments
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
JOBS="${ACPP_JOBS:-4}"
BUILD_TYPE="${ACPP_BUILD_TYPE:-Release}"
WORK_ROOT="${ACPP_WORKDIR:-${TMPDIR:-/tmp}/quantom-adaptivecpp-${USER:-user}}"
SOURCE="${ACPP_SOURCE_DIR:-}"
BUILD="$WORK_ROOT/build"

[[ -n "$PREFIX" ]] || { echo "install prefix must not be empty" >&2; exit 2; }
[[ -n "$LLVM_PREFIX" ]] || { echo "LLVM prefix must not be empty" >&2; exit 2; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "ACPP_JOBS must be a positive integer" >&2; exit 2; }

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

if [[ "${ACPP_EXPERIMENTAL_LLVM:-0}" == "1" ]]; then
  CMAKE_CMD+=(-DACPP_EXPERIMENTAL_LLVM=ON)
fi

if [[ -n "${CUDA_PATH:-}" ]]; then
  CMAKE_CMD+=("-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH" -DWITH_CUDA_BACKEND=ON)
fi
if [[ -n "${ROCM_PATH:-}" ]]; then
  CMAKE_CMD+=("-DROCM_PATH=$ROCM_PATH" -DWITH_ROCM_BACKEND=ON)
fi

CMAKE_CMD+=("${EXTRA_CMAKE[@]}")

printf 'Configuring AdaptiveCpp:'
printf ' %q' "${CMAKE_CMD[@]}"
printf '\n'
"${CMAKE_CMD[@]}"

cmake --build "$BUILD" --target install --parallel "$JOBS"

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
