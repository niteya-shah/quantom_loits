#!/usr/bin/env bash


usage() {
  cat >&2 <<'USAGE'
usage: install-llvm.sh <install-prefix> <targets> <llvm-version>

Build an official released LLVM toolchain for AdaptiveCpp.

<targets> is a comma-separated list containing one or more of:
  cpu   build the X86 target
  cuda  additionally build the NVPTX target
  hip   additionally build the AMDGPU target

The version must name an official LLVM release, e.g. 20.1.8. Source download is
separate from this build step; set LLVM_SOURCE_DIR to a checkout prepared by
fetch-llvm.sh. The build directory is preserved so interrupted Ninja builds resume.

Environment overrides:
  LLVM_WORKDIR          persistent build root; use cluster scratch if desired
  LLVM_SOURCE_DIR       llvm-project checkout prepared by fetch-llvm.sh (required)
  LLVM_REF              source ref recorded in metadata (default llvmorg-<version>)
  LLVM_JOBS             parallel build jobs (default: all available cores)
  LLVM_LINK_JOBS        optional LLVM_PARALLEL_LINK_JOBS value
  LLVM_BUILD_TYPE       CMake build type (default: Release)
  LLVM_C_COMPILER       host C compiler (default: gcc from PATH)
  LLVM_CXX_COMPILER     host C++ compiler (default: g++ from PATH)
  LLVM_CMAKE_ARGS       additional whitespace-separated CMake arguments
  LLVM_ALLOW_UNSUPPORTED allow a release outside the AdaptiveCpp 25.10 LLVM 15--20 window
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
REF="${LLVM_REF:-llvmorg-${VERSION}}"
JOBS="${LLVM_JOBS:-$(nproc)}"
BUILD_TYPE="${LLVM_BUILD_TYPE:-Release}"
WORK_ROOT="${LLVM_WORKDIR:-${TMPDIR:-/tmp}/quantom-llvm-${VERSION}-${USER:-user}}"
SOURCE="${LLVM_SOURCE_DIR:-}"
BUILD="$WORK_ROOT/build"
CC_BIN="${LLVM_C_COMPILER:-$(command -v gcc || true)}"
CXX_BIN="${LLVM_CXX_COMPILER:-$(command -v g++ || true)}"

[[ -n "$PREFIX" ]] || { echo "install prefix must not be empty" >&2; exit 2; }
[[ -n "$TARGET_SPEC" ]] || { echo "LLVM targets must not be empty" >&2; exit 2; }
[[ -n "$VERSION" ]] || { echo "LLVM version must not be empty" >&2; exit 2; }
LLVM_MAJOR="${VERSION%%.*}"
if [[ ! "$LLVM_MAJOR" =~ ^[0-9]+$ ]]; then
  echo "LLVM version must begin with a numeric major release: $VERSION" >&2
  exit 2
fi
# AdaptiveCpp v25.10.0 rejects LLVM >20 unless ACPP_EXPERIMENTAL_LLVM is enabled.
# QuantOm therefore defaults to the released/supported LLVM 20 line.
if (( LLVM_MAJOR < 15 || LLVM_MAJOR > 20 )) && [[ "${LLVM_ALLOW_UNSUPPORTED:-0}" != "1" ]]; then
  echo "LLVM $VERSION is outside the AdaptiveCpp v25.10 supported LLVM 15--20 window." >&2
  echo "Set LLVM_ALLOW_UNSUPPORTED=1 only for an intentional experimental build." >&2
  exit 2
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "LLVM_JOBS must be a positive integer" >&2; exit 2; }
[[ -n "$CC_BIN" && -x "$CC_BIN" ]] || { echo "host C compiler not found; set LLVM_C_COMPILER" >&2; exit 127; }
[[ -n "$CXX_BIN" && -x "$CXX_BIN" ]] || { echo "host C++ compiler not found; set LLVM_CXX_COMPILER" >&2; exit 127; }

for tool in cmake ninja; do
  command -v "$tool" >/dev/null 2>&1 || { echo "required tool not found: $tool" >&2; exit 127; }
done

IFS=',' read -r -a REQUESTED_TARGETS <<< "$TARGET_SPEC"
LLVM_TARGETS=(X86)
for target in "${REQUESTED_TARGETS[@]}"; do
  case "$target" in
    cpu) ;;
    cuda) LLVM_TARGETS+=(NVPTX) ;;
    hip) LLVM_TARGETS+=(AMDGPU) ;;
    *)
      echo "invalid LLVM target '$target'; expected cpu,cuda,hip" >&2
      exit 2
      ;;
  esac
done

UNIQUE_TARGETS=()
for target in "${LLVM_TARGETS[@]}"; do
  seen=0
  for existing in "${UNIQUE_TARGETS[@]:-}"; do
    [[ "$existing" == "$target" ]] && seen=1
  done
  (( seen )) || UNIQUE_TARGETS+=("$target")
done
LLVM_TARGETS_CMAKE="$(IFS=';'; echo "${UNIQUE_TARGETS[*]}")"

[[ -n "$SOURCE" ]] || {
  echo "LLVM_SOURCE_DIR is required for the build step." >&2
  echo "Download once with ./sycl/fetch-llvm.sh <source-dir> $VERSION" >&2
  exit 2
}
[[ -f "$SOURCE/llvm/CMakeLists.txt" ]] || {
  echo "LLVM_SOURCE_DIR is not an llvm-project source tree: $SOURCE" >&2
  echo "Run ./sycl/fetch-llvm.sh before building." >&2
  exit 2
}

COMMIT="unknown"
if [[ -d "$SOURCE/.git" ]] && command -v git >/dev/null 2>&1; then
  COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
fi

mkdir -p "$PREFIX" "$BUILD"

if [[ -f "$BUILD/CMakeCache.txt" ]]; then
  echo "Resuming existing LLVM build directory: $BUILD"
else
  echo "Creating LLVM build directory: $BUILD"
fi

read -r -a EXTRA_CMAKE <<< "${LLVM_CMAKE_ARGS:-}"

# This intentionally follows AdaptiveCpp's current source-LLVM recipe rather
# than the older QuantOm artifact recipe. In particular we do not build libc++,
# libc++abi, libunwind, clang-tools-extra, or libomptarget/offload runtimes.
CMAKE_CMD=(
  cmake
  -S "$SOURCE/llvm"
  -B "$BUILD"
  -G Ninja
  "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
  "-DCMAKE_INSTALL_PREFIX=$PREFIX"
  "-DCMAKE_C_COMPILER=$CC_BIN"
  "-DCMAKE_CXX_COMPILER=$CXX_BIN"
  "-DLLVM_ENABLE_PROJECTS=clang;lld;openmp"
  "-DLLVM_ENABLE_RUNTIMES=compiler-rt"
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF
  -DLLVM_ENABLE_ASSERTIONS=OFF
  -DLLVM_ENABLE_DUMP=OFF
  "-DLLVM_TARGETS_TO_BUILD=$LLVM_TARGETS_CMAKE"
  -DLLVM_INCLUDE_BENCHMARKS=OFF
  -DLLVM_INCLUDE_EXAMPLES=OFF
  -DLLVM_INCLUDE_TESTS=OFF
  -DLLVM_ENABLE_OCAMLDOC=OFF
  -DLLVM_ENABLE_BINDINGS=OFF
  -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF
  -DLLVM_BUILD_LLVM_DYLIB=ON
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
  "-DCMAKE_INSTALL_RPATH=$PREFIX/lib"
)

if [[ -n "${LLVM_LINK_JOBS:-}" ]]; then
  [[ "$LLVM_LINK_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "LLVM_LINK_JOBS must be a positive integer" >&2; exit 2; }
  CMAKE_CMD+=("-DLLVM_PARALLEL_LINK_JOBS=$LLVM_LINK_JOBS")
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
ref=$REF
commit=$COMMIT
requested_targets=$TARGET_SPEC
llvm_targets=$LLVM_TARGETS_CMAKE
host_cc=$CC_BIN
host_cxx=$CXX_BIN
INFO

echo
echo "LLVM installed successfully"
echo "  prefix:  $PREFIX"
echo "  clang:   $PREFIX/bin/clang"
echo "  targets: $LLVM_TARGETS_CMAKE"
echo
echo "Use this LLVM to build AdaptiveCpp with:"
echo "  make install-acpp ACPP_PREFIX=/path/to/adaptivecpp LLVM_PREFIX='$PREFIX'"
