#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: install-llvm.sh <install-prefix> <targets> <llvm-version>

Build the released LLVM toolchain used to build AdaptiveCpp.

<targets> is a comma-separated list containing one or more of:
  cpu   build the X86 target
  cuda  additionally build the NVPTX target
  hip   additionally build the AMDGPU target

Examples:
  ./sycl/install-llvm.sh /shared/toolchains/llvm-19.0.1 cpu
  ./sycl/install-llvm.sh /shared/toolchains/llvm-19.0.1 cpu,cuda
  ./sycl/install-llvm.sh /shared/toolchains/llvm-19.0.1 cpu,hip
  ./sycl/install-llvm.sh /shared/toolchains/llvm-19.0.1 cpu,cuda,hip 19.0.1

Environment overrides:
  LLVM_WORKDIR          source/build root; use cluster scratch for large builds
  LLVM_SOURCE_DIR       existing llvm-project checkout/source tree
  LLVM_JOBS             parallel build jobs (default: 4)
  LLVM_LINK_JOBS        optional LLVM_PARALLEL_LINK_JOBS value
  LLVM_BUILD_TYPE       CMake build type (default: Release)
  LLVM_C_COMPILER       host C compiler (default: gcc from PATH)
  LLVM_CXX_COMPILER     host C++ compiler (default: g++ from PATH)
  LLVM_CMAKE_ARGS       additional whitespace-separated CMake arguments
  LLVM_REPO             llvm-project repository URL
  CUDA_PATH             optional CUDA toolkit root for CUDA/offload discovery
  ROCM_PATH             optional ROCm root for HIP/offload discovery
  CUDA_ARCH             optional libomptarget CUDA architecture (e.g. sm_80)
  HIP_ARCH              optional libomptarget HIP architecture (e.g. gfx90a)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

PREFIX="$1"
TARGET_SPEC="$2"
VERSION="$3"
JOBS="${LLVM_JOBS:-4}"
BUILD_TYPE="${LLVM_BUILD_TYPE:-Release}"
REPO="${LLVM_REPO:-https://github.com/llvm/llvm-project.git}"
WORK_ROOT="${LLVM_WORKDIR:-${TMPDIR:-/tmp}/quantom-llvm-${VERSION}-${USER:-user}}"
SOURCE="${LLVM_SOURCE_DIR:-$WORK_ROOT/source}"
BUILD="$WORK_ROOT/build"
CC_BIN="${LLVM_C_COMPILER:-$(command -v gcc || true)}"
CXX_BIN="${LLVM_CXX_COMPILER:-$(command -v g++ || true)}"

[[ -n "$PREFIX" ]] || { echo "install prefix must not be empty" >&2; exit 2; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "LLVM_JOBS must be a positive integer" >&2; exit 2; }
[[ -n "$CC_BIN" && -x "$CC_BIN" ]] || { echo "host C compiler not found; set LLVM_C_COMPILER" >&2; exit 127; }
[[ -n "$CXX_BIN" && -x "$CXX_BIN" ]] || { echo "host C++ compiler not found; set LLVM_CXX_COMPILER" >&2; exit 127; }

for tool in cmake ninja; do
  command -v "$tool" >/dev/null 2>&1 || { echo "required tool not found: $tool" >&2; exit 127; }
done

IFS=',' read -r -a REQUESTED_TARGETS <<< "$TARGET_SPEC"
LLVM_TARGETS=(X86)
WANT_CUDA=0
WANT_HIP=0

for target in "${REQUESTED_TARGETS[@]}"; do
  case "$target" in
    cpu)
      ;;
    cuda)
      WANT_CUDA=1
      ;;
    hip)
      WANT_HIP=1
      ;;
    *)
      echo "invalid LLVM target '$target'; expected cpu,cuda,hip" >&2
      exit 2
      ;;
  esac
done

if (( WANT_CUDA )); then
  LLVM_TARGETS+=(NVPTX)
fi
if (( WANT_HIP )); then
  LLVM_TARGETS+=(AMDGPU)
fi

# Remove duplicates while preserving order.
UNIQUE_TARGETS=()
for target in "${LLVM_TARGETS[@]}"; do
  seen=0
  for existing in "${UNIQUE_TARGETS[@]:-}"; do
    [[ "$existing" == "$target" ]] && seen=1
  done
  (( seen )) || UNIQUE_TARGETS+=("$target")
done
LLVM_TARGETS_CMAKE="$(IFS=';'; echo "${UNIQUE_TARGETS[*]}")"

if [[ -z "${LLVM_SOURCE_DIR:-}" ]]; then
  command -v git >/dev/null 2>&1 || { echo "required tool not found: git" >&2; exit 127; }
  if [[ ! -d "$SOURCE/.git" ]]; then
    mkdir -p "$(dirname "$SOURCE")"
    echo "Cloning LLVM ${VERSION} from $REPO"
    git clone --depth 1 --branch "llvmorg-${VERSION}" "$REPO" "$SOURCE"
  else
    echo "Using existing LLVM checkout: $SOURCE"
  fi
else
  [[ -f "$SOURCE/llvm/CMakeLists.txt" ]] || {
    echo "LLVM_SOURCE_DIR is not an llvm-project source tree: $SOURCE" >&2
    exit 2
  }
fi

mkdir -p "$PREFIX"
rm -rf "$BUILD"
mkdir -p "$BUILD"

read -r -a EXTRA_CMAKE <<< "${LLVM_CMAKE_ARGS:-}"

CMAKE_CMD=(
  cmake
  -S "$SOURCE/llvm"
  -B "$BUILD"
  -G Ninja
  "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
  "-DCMAKE_INSTALL_PREFIX=$PREFIX"
  "-DCMAKE_C_COMPILER=$CC_BIN"
  "-DCMAKE_CXX_COMPILER=$CXX_BIN"
  "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld"
  "-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi;openmp;offload;libunwind;compiler-rt"
  "-DLLVM_TARGETS_TO_BUILD=$LLVM_TARGETS_CMAKE"
  -DLLVM_BUILD_LLVM_DYLIB=ON
  -DLLVM_LINK_LLVM_DYLIB=ON
  -DLLVM_ENABLE_RTTI=ON
)

if [[ -n "${LLVM_LINK_JOBS:-}" ]]; then
  [[ "$LLVM_LINK_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "LLVM_LINK_JOBS must be a positive integer" >&2; exit 2; }
  CMAKE_CMD+=("-DLLVM_PARALLEL_LINK_JOBS=$LLVM_LINK_JOBS")
fi

OFFLOAD_ARCHES=()
if (( WANT_CUDA )); then
  if [[ -z "${CUDA_PATH:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    CUDA_PATH="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  fi
  if [[ -n "${CUDA_PATH:-}" ]]; then
    CMAKE_CMD+=("-DCUDAToolkit_ROOT=$CUDA_PATH" "-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH")
  fi
  [[ -n "${CUDA_ARCH:-}" ]] && OFFLOAD_ARCHES+=("$CUDA_ARCH")
fi

if (( WANT_HIP )); then
  if [[ -z "${ROCM_PATH:-}" ]] && command -v hipcc >/dev/null 2>&1; then
    ROCM_PATH="$(cd "$(dirname "$(command -v hipcc)")/.." && pwd)"
  fi
  [[ -n "${ROCM_PATH:-}" ]] && CMAKE_CMD+=("-DROCM_PATH=$ROCM_PATH")
  [[ -n "${HIP_ARCH:-}" ]] && OFFLOAD_ARCHES+=("$HIP_ARCH")
fi

if (( ${#OFFLOAD_ARCHES[@]} > 0 )); then
  OFFLOAD_ARCHES_CMAKE="$(IFS=';'; echo "${OFFLOAD_ARCHES[*]}")"
  CMAKE_CMD+=("-DLIBOMPTARGET_DEVICE_ARCHITECTURES=$OFFLOAD_ARCHES_CMAKE")
fi

CMAKE_CMD+=("${EXTRA_CMAKE[@]}")

printf 'Configuring LLVM:'
printf ' %q' "${CMAKE_CMD[@]}"
printf '\n'
"${CMAKE_CMD[@]}"

cmake --build "$BUILD" --target install --parallel "$JOBS"

for required in \
  "$PREFIX/bin/clang" \
  "$PREFIX/bin/clang++" \
  "$PREFIX/bin/ld.lld" \
  "$PREFIX/lib/cmake/llvm/LLVMConfig.cmake" \
  "$PREFIX/lib/cmake/clang/ClangConfig.cmake"; do
  [[ -e "$required" ]] || { echo "LLVM install missing required file: $required" >&2; exit 1; }
done

if ! compgen -G "$PREFIX/lib/libLLVM.so*" >/dev/null; then
  echo "LLVM install missing shared libLLVM under $PREFIX/lib" >&2
  exit 1
fi

cat > "$PREFIX/quantom-llvm-info.txt" <<INFO
version=$VERSION
requested_targets=$TARGET_SPEC
llvm_targets=$LLVM_TARGETS_CMAKE
host_cc=$CC_BIN
host_cxx=$CXX_BIN
cuda_path=${CUDA_PATH:-}
rocm_path=${ROCM_PATH:-}
cuda_arch=${CUDA_ARCH:-}
hip_arch=${HIP_ARCH:-}
INFO

echo
echo "LLVM installed successfully"
echo "  prefix:  $PREFIX"
echo "  clang:   $PREFIX/bin/clang"
echo "  targets: $LLVM_TARGETS_CMAKE"
echo
echo "Use this LLVM to build AdaptiveCpp with:"
echo "  make install-acpp ACPP_PREFIX=/path/to/adaptivecpp LLVM_PREFIX='$PREFIX'"
