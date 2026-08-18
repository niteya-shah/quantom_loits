#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD_ROOT="$HERE/build"

usage() {
  echo "usage: $0 <variant> <generic|cpu|omp|cuda|hip> [architecture]" >&2
  echo "examples:" >&2
  echo "  $0 acpp-a100 cuda sm_80" >&2
  echo "  $0 acpp-mi250 hip gfx90a" >&2
  echo "  $0 acpp-cpu cpu" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

VARIANT="$1"
MODE="$2"
ARCH="${3:-}"
if [[ -n "${ACPP_CXX:-}" ]]; then
  CXX="$ACPP_CXX"
elif [[ -n "${ACPP_PREFIX:-}" ]]; then
  CXX="$ACPP_PREFIX/bin/acpp"
elif [[ -n "${ADAPTIVECPP:-}" ]]; then
  CXX="$ADAPTIVECPP"
else
  CXX="acpp"
fi

if [[ ! "$VARIANT" =~ ^[A-Za-z0-9][A-Za-z0-9_.+-]*$ ]]; then
  echo "invalid SYCL variant name: $VARIANT" >&2
  exit 2
fi

BACKEND_FLAGS=()

case "$MODE" in
  generic)
    [[ -z "$ARCH" ]] || { echo "generic target does not take an architecture" >&2; exit 2; }
    TARGETS="generic"
    [[ -z "${CUDA_PATH:-}" ]] || BACKEND_FLAGS+=("--acpp-cuda-path=$CUDA_PATH")
    [[ -z "${ROCM_PATH:-}" ]] || BACKEND_FLAGS+=("--acpp-rocm-path=$ROCM_PATH")
    TORCH_DEVICE="auto"
    ;;
  cpu|omp)
    [[ -z "$ARCH" ]] || { echo "$MODE target does not take an architecture" >&2; exit 2; }
    TARGETS="omp"
    TORCH_DEVICE="cpu"
    ;;
  cuda)
    [[ -n "$ARCH" ]] || { echo "CUDA architecture is required, e.g. sm_80" >&2; usage; exit 2; }
    TARGETS="cuda:$ARCH"
    [[ -z "${CUDA_PATH:-}" ]] || BACKEND_FLAGS+=("--acpp-cuda-path=$CUDA_PATH")
    TORCH_DEVICE="cuda"
    ;;
  hip)
    [[ -n "$ARCH" ]] || { echo "HIP architecture is required, e.g. gfx90a" >&2; usage; exit 2; }
    TARGETS="hip:$ARCH"
    [[ -z "${ROCM_PATH:-}" ]] || BACKEND_FLAGS+=("--acpp-rocm-path=$ROCM_PATH")
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

# Remove completeness markers before rebuilding so a failed build is never
# reported as available. Other variants are untouched.
rm -f \
  "$BUILD/libquantom_loits_sycl.so" \
  "$BUILD/variant.py" \
  "$BUILD/toolchain.txt" \
  "$BUILD/torch_device.txt" \
  "$BUILD/target.txt" \
  "$BUILD/architecture.txt"
read -r -a EXTRA <<< "${SYCL_EXTRA_FLAGS:-}"
RPATH_FLAGS=()
if [[ -n "${ACPP_PREFIX:-}" ]]; then
  RPATH_FLAGS+=("-Wl,-rpath,$ACPP_PREFIX/lib")
fi

"$CXX" \
  -O3 -std=c++17 -DNDEBUG -DQUANTOM_SYCL_NATIVE=1 -fPIC -shared \
  --acpp-targets="$TARGETS" \
  "${BACKEND_FLAGS[@]}" \
  -I"$ROOT" \
  "${EXTRA[@]}" \
  "${RPATH_FLAGS[@]}" \
  "$HERE/loits_core.cpp" "$HERE/bindings.cpp" \
  -Wl,-soname,libquantom_loits_sycl.so \
  -o "$BUILD/libquantom_loits_sycl.so"

rm -f "$BUILD"/quantom_loits_sycl_binding*.so "$BUILD"/bindings.o "$BUILD"/build.ninja "$BUILD"/lock
(cd "$ROOT" && QUANTOM_SYCL_VARIANT="$VARIANT" python -m sycl.build)

python - "$BUILD/variant.py" "acpp" "$MODE" "$TORCH_DEVICE" "$ARCH" <<'PY'
from pathlib import Path
import sys

path, toolchain, target, torch_device, architecture = sys.argv[1:]
metadata = {
    "toolchain": toolchain,
    "target": target,
    "torch_device": torch_device,
    "architecture": architecture or None,
}
Path(path).write_text("METADATA = " + repr(metadata) + "\n")
PY

echo "built SYCL variant '$VARIANT' in $BUILD"
echo "use it with: export QUANTOM_SYCL_VARIANT='$VARIANT'"
