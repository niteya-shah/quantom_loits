#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD_ROOT="$HERE/build"

usage() {
  echo "usage: $0 <variant> <cpu|xpu|cuda|hip> [architecture]" >&2
  echo "examples:" >&2
  echo "  $0 dpcpp-xpu xpu" >&2
  echo "  $0 dpcpp-a100 cuda sm_80" >&2
  echo "  $0 dpcpp-mi250 hip gfx90a" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

VARIANT="$1"
MODE="$2"
ARCH="${3:-}"

# A source-built DPC++ toolchain is used directly from its build directory,
# matching Intel's current source-build documentation. Explicit per-mode
# compiler overrides still take precedence.
DPCPP_PREFIX="${DPCPP_PREFIX:-}"

resolve_cxx() {
  local specific="$1"
  local fallback="$2"
  if [[ -n "$specific" ]]; then
    printf '%s\n' "$specific"
  elif [[ -n "${DPCPP_CXX:-}" ]]; then
    printf '%s\n' "$DPCPP_CXX"
  elif [[ -n "$DPCPP_PREFIX" ]]; then
    printf '%s\n' "$DPCPP_PREFIX/bin/clang++"
  else
    printf '%s\n' "$fallback"
  fi
}

if [[ ! "$VARIANT" =~ ^[A-Za-z0-9][A-Za-z0-9_.+-]*$ ]]; then
  echo "invalid SYCL variant name: $VARIANT" >&2
  exit 2
fi

case "$MODE" in
  cpu)
    [[ -z "$ARCH" ]] || { echo "cpu target does not take an architecture" >&2; exit 2; }
    CXX="$(resolve_cxx "${DPCPP_CPU_CXX:-}" icpx)"
    TARGET_FLAGS=(-fsycl -fsycl-targets=native_cpu)
    TORCH_DEVICE="cpu"
    ;;
  xpu)
    [[ -z "$ARCH" ]] || { echo "xpu target does not take an architecture" >&2; exit 2; }
    CXX="$(resolve_cxx "${DPCPP_XPU_CXX:-}" icpx)"
    TARGET_FLAGS=(-fsycl -fsycl-targets=spir64)
    TORCH_DEVICE="xpu"
    ;;
  cuda)
    [[ -n "$ARCH" ]] || { echo "CUDA architecture is required, e.g. sm_80" >&2; usage; exit 2; }
    CXX="$(resolve_cxx "${DPCPP_CUDA_CXX:-}" clang++)"
    TARGET="nvptx64-nvidia-cuda"
    TARGET_FLAGS=(-fsycl -fsycl-targets="$TARGET" -Xsycl-target-backend "--cuda-gpu-arch=$ARCH")
    CUDA_ROOT="${DPCPP_CUDA_PATH:-${CUDA_PATH:-}}"
    [[ -z "$CUDA_ROOT" ]] || TARGET_FLAGS+=("--cuda-path=$CUDA_ROOT")
    TORCH_DEVICE="cuda"
    ;;
  hip)
    [[ -n "$ARCH" ]] || { echo "HIP architecture is required, e.g. gfx90a" >&2; usage; exit 2; }
    CXX="$(resolve_cxx "${DPCPP_HIP_CXX:-}" clang++)"
    TARGET="amdgcn-amd-amdhsa"
    TARGET_FLAGS=(-fsycl -fsycl-targets="$TARGET" -Xsycl-target-backend "--offload-arch=$ARCH")
    TORCH_DEVICE="cuda"
    ;;
  *)
    usage
    exit 2
    ;;
esac

BUILD="$BUILD_ROOT/$VARIANT"
mkdir -p "$BUILD"
if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "SYCL compiler not found: $CXX" >&2
  exit 127
fi

LIBSPIRV_FLAGS=()
if [[ "$MODE" == "hip" ]]; then
  if [[ -n "${DPCPP_LIBSPIRV_PATH:-}" ]]; then
    LIBSPIRV="$DPCPP_LIBSPIRV_PATH"
  else
    RESOURCE_DIR="$("$CXX" -print-resource-dir)"
    LIBSPIRV="$RESOURCE_DIR/lib/amdgcn-amd-amdhsa-llvm/libspirv.l64.signed_char.bc"
  fi

  if [[ ! -f "$LIBSPIRV" ]]; then
    echo "DPC++ HIP libspirv not found: $LIBSPIRV" >&2
    echo "compiler: $CXX" >&2
    echo "resource dir: ${RESOURCE_DIR:-<DPCPP_LIBSPIRV_PATH override>}" >&2
    echo "set DPCPP_LIBSPIRV_PATH to the full libspirv.l64.signed_char.bc path if needed" >&2
    exit 2
  fi

  echo "DPC++ HIP libspirv: $LIBSPIRV"
  LIBSPIRV_FLAGS+=("-fsycl-libspirv-path=$LIBSPIRV")
fi

rm -f \
  "$BUILD/libquantom_loits_sycl.so" \
  "$BUILD/toolchain.txt" \
  "$BUILD/torch_device.txt" \
  "$BUILD/target.txt" \
  "$BUILD/architecture.txt"
read -r -a EXTRA <<< "${SYCL_EXTRA_FLAGS:-}"
RPATH_FLAGS=()
if [[ -n "$DPCPP_PREFIX" ]]; then
  RPATH_FLAGS+=("-Wl,-rpath,$DPCPP_PREFIX/lib")
fi

"$CXX" \
  -O3 -std=c++17 -DNDEBUG -fPIC -shared \
  "${TARGET_FLAGS[@]}" \
  "${LIBSPIRV_FLAGS[@]}" \
  -I"$ROOT" \
  "${EXTRA[@]}" \
  "${RPATH_FLAGS[@]}" \
  "$HERE/loits_core.cpp" \
  -Wl,-soname,libquantom_loits_sycl.so \
  -o "$BUILD/libquantom_loits_sycl.so"

rm -f "$BUILD"/quantom_loits_sycl_binding*.so "$BUILD"/bindings.o "$BUILD"/build.ninja "$BUILD"/lock
(cd "$ROOT" && QUANTOM_SYCL_VARIANT="$VARIANT" python -m sycl.build)

printf '%s\n' "dpcpp" > "$BUILD/toolchain.txt"
printf '%s\n' "$MODE" > "$BUILD/target.txt"
printf '%s\n' "$TORCH_DEVICE" > "$BUILD/torch_device.txt"
if [[ -n "$ARCH" ]]; then
  printf '%s\n' "$ARCH" > "$BUILD/architecture.txt"
else
  rm -f "$BUILD/architecture.txt"
fi

echo "built SYCL variant '$VARIANT' in $BUILD"
echo "use it with: export QUANTOM_SYCL_VARIANT='$VARIANT'"
