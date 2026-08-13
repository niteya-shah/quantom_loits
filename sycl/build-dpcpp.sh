#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="$HERE/build"
MODE="${1:-xpu}"

case "$MODE" in
  cpu)
    CXX="${DPCPP_CPU_CXX:-${DPCPP_CXX:-icpx}}"
    TARGET_FLAGS=(-fsycl -fsycl-targets=native_cpu)
    TORCH_DEVICE="cpu"
    ;;
  xpu)
    CXX="${DPCPP_XPU_CXX:-${DPCPP_CXX:-icpx}}"
    TARGET_FLAGS=(-fsycl -fsycl-targets=spir64)
    TORCH_DEVICE="xpu"
    ;;
  cuda)
    CXX="${DPCPP_CUDA_CXX:-${DPCPP_CXX:-clang++}}"
    TARGET="nvptx64-nvidia-cuda"
    TARGET_FLAGS=(-fsycl -fsycl-targets="$TARGET" -Xsycl-target-backend="$TARGET" "--offload-arch=${CUDA_DEV_TARGET:-sm_80}")
    TORCH_DEVICE="cuda"
    ;;
  hip)
    CXX="${DPCPP_HIP_CXX:-${DPCPP_CXX:-clang++}}"
    TARGET="amdgcn-amd-amdhsa"
    TARGET_FLAGS=(-fsycl -fsycl-targets="$TARGET" -Xsycl-target-backend="$TARGET" "--offload-arch=${HIP_DEV_TARGET:-gfx90a}")
    TORCH_DEVICE="cuda"
    ;;
  *)
    echo "usage: $0 [cpu|xpu|cuda|hip]" >&2
    exit 2
    ;;
esac

mkdir -p "$BUILD"
if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "SYCL compiler not found: $CXX" >&2
  exit 127
fi
rm -f "$BUILD/libquantom_loits_sycl.so" "$BUILD/toolchain.txt" "$BUILD/torch_device.txt"
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
(cd "$ROOT" && python -m sycl.build)
printf '%s\n' "dpcpp:$MODE" > "$BUILD/toolchain.txt"
printf '%s\n' "$TORCH_DEVICE" > "$BUILD/torch_device.txt"
