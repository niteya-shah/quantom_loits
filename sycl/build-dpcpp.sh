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

if [[ ! "$VARIANT" =~ ^[A-Za-z0-9][A-Za-z0-9_.+-]*$ ]]; then
  echo "invalid SYCL variant name: $VARIANT" >&2
  exit 2
fi

case "$MODE" in
  cpu)
    [[ -z "$ARCH" ]] || { echo "cpu target does not take an architecture" >&2; exit 2; }
    CXX="${DPCPP_CPU_CXX:-${DPCPP_CXX:-icpx}}"
    TARGET_FLAGS=(-fsycl -fsycl-targets=native_cpu)
    TORCH_DEVICE="cpu"
    ;;
  xpu)
    [[ -z "$ARCH" ]] || { echo "xpu target does not take an architecture" >&2; exit 2; }
    CXX="${DPCPP_XPU_CXX:-${DPCPP_CXX:-icpx}}"
    TARGET_FLAGS=(-fsycl -fsycl-targets=spir64)
    TORCH_DEVICE="xpu"
    ;;
  cuda)
    [[ -n "$ARCH" ]] || { echo "CUDA architecture is required, e.g. sm_80" >&2; usage; exit 2; }
    CXX="${DPCPP_CUDA_CXX:-${DPCPP_CXX:-clang++}}"
    TARGET="nvptx64-nvidia-cuda"
    TARGET_FLAGS=(-fsycl -fsycl-targets="$TARGET" -Xsycl-target-backend="$TARGET" "--offload-arch=$ARCH")
    TORCH_DEVICE="cuda"
    ;;
  hip)
    [[ -n "$ARCH" ]] || { echo "HIP architecture is required, e.g. gfx90a" >&2; usage; exit 2; }
    CXX="${DPCPP_HIP_CXX:-${DPCPP_CXX:-clang++}}"
    TARGET="amdgcn-amd-amdhsa"
    TARGET_FLAGS=(-fsycl -fsycl-targets="$TARGET" -Xsycl-target-backend="$TARGET" "--offload-arch=$ARCH")
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

rm -f \
  "$BUILD/libquantom_loits_sycl.so" \
  "$BUILD/toolchain.txt" \
  "$BUILD/torch_device.txt" \
  "$BUILD/target.txt" \
  "$BUILD/architecture.txt"
read -r -a EXTRA <<< "${SYCL_EXTRA_FLAGS:-}"

"$CXX" \
  -O3 -std=c++17 -DNDEBUG -fPIC -shared \
  "${TARGET_FLAGS[@]}" \
  -I"$ROOT" \
  "${EXTRA[@]}" \
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
