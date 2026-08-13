#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="$HERE/build"
MODE="${1:-generic}"
CXX="${ACPP_CXX:-${ADAPTIVECPP:-acpp}}"

case "$MODE" in
  generic)
    TARGETS="generic"
    TORCH_DEVICE="auto"
    ;;
  cpu|omp)
    TARGETS="omp"
    TORCH_DEVICE="cpu"
    ;;
  cuda)
    TARGETS="cuda:${CUDA_DEV_TARGET:-sm_80}"
    TORCH_DEVICE="cuda"
    ;;
  hip)
    TARGETS="hip:${HIP_DEV_TARGET:-gfx90a}"
    TORCH_DEVICE="cuda"
    ;;
  *)
    echo "usage: $0 [generic|cpu|omp|cuda|hip]" >&2
    exit 2
    ;;
esac

mkdir -p "$BUILD"
read -r -a EXTRA <<< "${SYCL_EXTRA_FLAGS:-}"

"$CXX" \
  -O3 -std=c++17 -DNDEBUG -fPIC -shared \
  --acpp-targets="$TARGETS" \
  -I"$ROOT" \
  "${EXTRA[@]}" \
  "$HERE/loits_core.cpp" \
  -Wl,-soname,libquantom_loits_sycl.so \
  -o "$BUILD/libquantom_loits_sycl.so"

printf '%s\n' "acpp:$MODE" > "$BUILD/toolchain.txt"
printf '%s\n' "$TORCH_DEVICE" > "$BUILD/torch_device.txt"
rm -f "$BUILD"/quantom_loits_sycl_binding*.so "$BUILD"/bindings.o "$BUILD"/build.ninja "$BUILD"/lock
(cd "$ROOT" && python -m sycl.build)
