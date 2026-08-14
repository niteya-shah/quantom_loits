#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: install-acpp.sh <install-prefix> [git-ref]

Build and install AdaptiveCpp from source using the upstream CMake flow.
This script intentionally does not install OS packages, LLVM, CUDA, ROCm,
or make machine-specific backend choices.

Examples:
  ./sycl/install-acpp.sh /shared/toolchains/adaptivecpp
  ./sycl/install-acpp.sh /shared/toolchains/adaptivecpp <tag-or-commit>

Environment overrides:
  ACPP_REF          git ref if the second positional argument is omitted
  ACPP_WORKDIR      temporary source/build root
  ACPP_SOURCE_DIR   use an existing AdaptiveCpp checkout instead of cloning
  ACPP_JOBS         parallel build jobs (default: 4)
  ACPP_BUILD_TYPE   CMake build type (default: Release)
  ACPP_CMAKE_ARGS   additional whitespace-separated CMake arguments
  ACPP_REPO         repository URL
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

PREFIX="$1"
REF="${2:-${ACPP_REF:-}}"
JOBS="${ACPP_JOBS:-4}"
BUILD_TYPE="${ACPP_BUILD_TYPE:-Release}"
REPO="${ACPP_REPO:-https://github.com/AdaptiveCpp/AdaptiveCpp.git}"
WORK_ROOT="${ACPP_WORKDIR:-${TMPDIR:-/tmp}/quantom-adaptivecpp-${USER:-user}}"
SOURCE="${ACPP_SOURCE_DIR:-$WORK_ROOT/source}"
BUILD="$WORK_ROOT/build"

if [[ -z "$PREFIX" ]]; then
  echo "install prefix must not be empty" >&2
  exit 2
fi

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACPP_JOBS must be a positive integer" >&2
  exit 2
fi

for tool in cmake python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "required tool not found: $tool" >&2
    exit 127
  fi
done

if [[ -z "${ACPP_SOURCE_DIR:-}" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "required tool not found: git" >&2
    exit 127
  fi

  if [[ ! -d "$SOURCE/.git" ]]; then
    mkdir -p "$(dirname "$SOURCE")"
    echo "Cloning AdaptiveCpp from $REPO"
    git clone "$REPO" "$SOURCE"
  else
    echo "Using existing AdaptiveCpp checkout: $SOURCE"
    git -C "$SOURCE" fetch --tags origin
  fi
else
  if [[ ! -f "$SOURCE/CMakeLists.txt" ]]; then
    echo "ACPP_SOURCE_DIR is not an AdaptiveCpp source tree: $SOURCE" >&2
    exit 2
  fi
fi

if [[ -n "$REF" ]]; then
  if [[ ! -d "$SOURCE/.git" ]]; then
    echo "a git ref was requested, but ACPP_SOURCE_DIR is not a git checkout" >&2
    exit 2
  fi

  if ! git -C "$SOURCE" rev-parse --verify --quiet "${REF}^{commit}" >/dev/null; then
    echo "AdaptiveCpp git ref not found locally: $REF" >&2
    echo "If using ACPP_SOURCE_DIR, fetch the ref first." >&2
    exit 2
  fi

  echo "Checking out AdaptiveCpp ref: $REF"
  git -C "$SOURCE" checkout --detach "$REF"
fi

if [[ -d "$SOURCE/.git" ]]; then
  COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
  DESCRIBE="$(git -C "$SOURCE" describe --always --dirty --tags 2>/dev/null || true)"
  echo "AdaptiveCpp source commit: $COMMIT${DESCRIBE:+ ($DESCRIBE)}"
fi

mkdir -p "$PREFIX"
rm -rf "$BUILD"
mkdir -p "$BUILD"

read -r -a EXTRA_CMAKE <<< "${ACPP_CMAKE_ARGS:-}"

CMAKE_CMD=(
  cmake
  -S "$SOURCE"
  -B "$BUILD"
  "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
  "-DCMAKE_INSTALL_PREFIX=$PREFIX"
)
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

echo
echo "AdaptiveCpp installed successfully"
echo "  prefix:   $PREFIX"
echo "  compiler: $ACPP"
"$ACPP" --acpp-version || true

echo
echo "Use this installation for QuantOm with:"
echo "  export ACPP_PREFIX='$PREFIX'"
echo "or:"
echo "  export ACPP_CXX='$ACPP'"
