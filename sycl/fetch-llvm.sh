#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: fetch-llvm.sh <source-dir> <llvm-version>

Download/check out llvm-project source for an official LLVM release without
building it. Re-running this command is offline/idempotent when the requested
ref is already present locally. Set FETCH_UPDATE=1 to force a network refresh.

Environment overrides:
  LLVM_REF      explicit git ref instead of llvmorg-<version>
  LLVM_REPO     repository URL (default: https://github.com/llvm/llvm-project.git)
  FETCH_UPDATE  set to 1 to refresh the requested ref from origin
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -eq 2 ]] || { usage; exit 2; }

SOURCE="$1"
VERSION="$2"
REF="${LLVM_REF:-llvmorg-${VERSION}}"
REPO="${LLVM_REPO:-https://github.com/llvm/llvm-project.git}"
UPDATE="${FETCH_UPDATE:-0}"

command -v git >/dev/null 2>&1 || { echo "required tool not found: git" >&2; exit 127; }
mkdir -p "$(dirname "$SOURCE")"

if [[ ! -d "$SOURCE/.git" ]]; then
  if [[ -e "$SOURCE" ]] && [[ -n "$(find "$SOURCE" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "source directory exists but is not a git checkout: $SOURCE" >&2
    exit 2
  fi
  echo "Cloning llvm-project repository metadata into $SOURCE"
  git clone --filter=blob:none --no-checkout "$REPO" "$SOURCE"
fi

# If this exact requested ref/commit was already prepared, do not contact origin.
if [[ "$UPDATE" != "1" && -f "$SOURCE/.quantom-requested-ref" && -f "$SOURCE/.quantom-requested-commit" ]]; then
  cached_ref="$(cat "$SOURCE/.quantom-requested-ref")"
  cached_commit="$(cat "$SOURCE/.quantom-requested-commit")"
  if [[ "$cached_ref" == "$REF" ]] && git -C "$SOURCE" cat-file -e "${cached_commit}^{commit}" 2>/dev/null; then
    git -C "$SOURCE" checkout --detach "$cached_commit" >/dev/null
    echo "LLVM source already cached: $REF ($cached_commit)"
    exit 0
  fi
fi

# Also reuse an already-local tag/commit even if it predates our metadata files.
if [[ "$UPDATE" != "1" ]] && commit="$(git -C "$SOURCE" rev-parse --verify "${REF}^{commit}" 2>/dev/null)"; then
  git -C "$SOURCE" checkout --detach "$commit" >/dev/null
  printf '%s\n' "$REF" > "$SOURCE/.quantom-requested-ref"
  printf '%s\n' "$commit" > "$SOURCE/.quantom-requested-commit"
  echo "LLVM source ref already present locally: $REF ($commit)"
  exit 0
fi

echo "Fetching LLVM ref $REF"
git -C "$SOURCE" fetch --depth 1 origin "$REF"
git -C "$SOURCE" checkout --detach FETCH_HEAD >/dev/null
commit="$(git -C "$SOURCE" rev-parse HEAD)"
printf '%s\n' "$REF" > "$SOURCE/.quantom-requested-ref"
printf '%s\n' "$commit" > "$SOURCE/.quantom-requested-commit"
echo "LLVM source ready: $REF ($commit)"
