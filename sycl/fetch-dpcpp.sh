#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: fetch-dpcpp.sh <source-dir> <git-ref>

Download/check out Intel's llvm/DPC++ source without building it. Re-running
this command is offline/idempotent when the requested ref is already present
locally. Set FETCH_UPDATE=1 to refresh a moving ref such as 'sycl'.

Environment overrides:
  DPCPP_REPO    repository URL (default: https://github.com/intel/llvm.git)
  FETCH_UPDATE  set to 1 to refresh the requested ref from origin
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -eq 2 ]] || { usage; exit 2; }

SOURCE="$1"
REF="$2"
REPO="${DPCPP_REPO:-https://github.com/intel/llvm.git}"
UPDATE="${FETCH_UPDATE:-0}"

command -v git >/dev/null 2>&1 || { echo "required tool not found: git" >&2; exit 127; }
mkdir -p "$(dirname "$SOURCE")"

if [[ ! -d "$SOURCE/.git" ]]; then
  if [[ -e "$SOURCE" ]] && [[ -n "$(find "$SOURCE" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "source directory exists but is not a git checkout: $SOURCE" >&2
    exit 2
  fi
  echo "Cloning Intel llvm repository metadata into $SOURCE"
  git clone --filter=blob:none --no-checkout "$REPO" "$SOURCE"
fi

if [[ "$UPDATE" != "1" && -f "$SOURCE/.quantom-requested-ref" && -f "$SOURCE/.quantom-requested-commit" ]]; then
  cached_ref="$(cat "$SOURCE/.quantom-requested-ref")"
  cached_commit="$(cat "$SOURCE/.quantom-requested-commit")"
  if [[ "$cached_ref" == "$REF" ]] && git -C "$SOURCE" cat-file -e "${cached_commit}^{commit}" 2>/dev/null; then
    git -C "$SOURCE" checkout --detach "$cached_commit" >/dev/null
    echo "DPC++ source already cached: $REF ($cached_commit)"
    exit 0
  fi
fi

if [[ "$UPDATE" != "1" ]] && commit="$(git -C "$SOURCE" rev-parse --verify "${REF}^{commit}" 2>/dev/null)"; then
  git -C "$SOURCE" checkout --detach "$commit" >/dev/null
  printf '%s\n' "$REF" > "$SOURCE/.quantom-requested-ref"
  printf '%s\n' "$commit" > "$SOURCE/.quantom-requested-commit"
  echo "DPC++ source ref already present locally: $REF ($commit)"
  exit 0
fi

echo "Fetching DPC++ ref $REF"
git -C "$SOURCE" fetch --depth 1 origin "$REF"
git -C "$SOURCE" checkout --detach FETCH_HEAD >/dev/null
commit="$(git -C "$SOURCE" rev-parse HEAD)"
printf '%s\n' "$REF" > "$SOURCE/.quantom-requested-ref"
printf '%s\n' "$commit" > "$SOURCE/.quantom-requested-commit"
echo "DPC++ source ready: $REF ($commit)"
