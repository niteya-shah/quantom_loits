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

case "$MODE" in
  generic)
    [[ -z "$ARCH" ]] || { echo "generic target does not take an architecture" >&2; exit 2; }
    TARGETS="generic"
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
    TORCH_DEVICE="cuda"
    ;;
  hip)
    [[ -n "$ARCH" ]] || { echo "HIP architecture is required, e.g. gfx90a" >&2; usage; exit 2; }
    TARGETS="hip:$ARCH"
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
  "$BUILD/toolchain.txt" \
  "$BUILD/torch_device.txt" \
  "$BUILD/target.txt" \
  "$BUILD/architecture.txt"
read -r -a EXTRA <<< "${SYCL_EXTRA_FLAGS:-}"

"$CXX" \
  -O3 -std=c++17 -DNDEBUG -fPIC -shared \
  --acpp-targets="$TARGETS" \
  -I"$ROOT" \
  "${EXTRA[@]}" \
  "$HERE/loits_core.cpp" \
  -Wl,-soname,libquantom_loits_sycl.so \
  -o "$BUILD/libquantom_loits_sycl.so"

rm -f "$BUILD"/quantom_loits_sycl_binding*.so "$BUILD"/bindings.o "$BUILD"/build.ninja "$BUILD"/lock
(cd "$ROOT" && QUANTOM_SYCL_VARIANT="$VARIANT" python -m sycl.build)

printf '%s\n' "acpp" > "$BUILD/toolchain.txt"
printf '%s\n' "$MODE" > "$BUILD/target.txt"
printf '%s\n' "$TORCH_DEVICE" > "$BUILD/torch_device.txt"
if [[ -n "$ARCH" ]]; then
  printf '%s\n' "$ARCH" > "$BUILD/architecture.txt"
else
  rm -f "$BUILD/architecture.txt"
fi

echo "built SYCL variant '$VARIANT' in $BUILD"
echo "use it with: export QUANTOM_SYCL_VARIANT='$VARIANT'"
