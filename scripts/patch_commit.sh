#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$(dirname "$0")/../logs"
timestamp=$(date +%Y%m%d_%H%M%S)
exec > >(tee "$(dirname "$0")/../logs/patch_commit_${timestamp}.log") 2>&1

# Apply a specific commit from one local repo clone to another using
# git diff + git apply. Useful when the two repos have divergent
# history and standard cherry-pick or format-patch won't work.
#
# Usage:
#   ./scripts/patch_commit.sh <commit> <source_repo_path> [target_repo_path]
#
# Examples:
#   # Apply faabfee from the GitHub clone to the current repo
#   ./scripts/patch_commit.sh faabfee <path_current_repo>
#
#   # Apply from another path explicitly
#   ./scripts/patch_commit.sh abc1234 /path/to/source /path/to/target
#
# Files changed in the commit are discovered automatically via git diff-tree.

COMMIT="${1:?Usage: $0 <commit> <source_repo> [target_repo]}"
SRC="${2:?Usage: $0 <commit> <source_repo> [target_repo]}"
TGT="${3:-$(pwd)}"

# Discover files changed in the commit
mapfile -t FILES < <(git -C "$SRC" diff-tree --no-commit-id -r --name-only "$COMMIT")

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No files changed in commit $COMMIT"
  exit 1
fi

echo "Commit: $COMMIT"
echo "Source: $SRC"
echo "Target: $TGT"
echo "Files: ${#FILES[@]}"
echo ""

TMPDIR="/tmp/patch_commit_${COMMIT}"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

# Generate patches from source repo
for f in "${FILES[@]}"; do
  patch="$TMPDIR/$(echo "$f" | tr '/' '_').patch"
  dir=$(dirname "$patch")
  mkdir -p "$dir"
  echo "Generating patch for $f..."
  git -C "$SRC" diff "$COMMIT^..$COMMIT" -- "$f" > "$patch" || true
done

# Apply patches in target repo
echo ""
echo "=== Applying patches ==="
cd "$TGT"
for f in "${FILES[@]}"; do
  patch="$TMPDIR/$(echo "$f" | tr '/' '_').patch"
  if [ ! -s "$patch" ]; then
    continue
  fi
  echo "Applying $f..."
  if ! git apply "$patch" 2>/dev/null; then
    echo "  CONFLICT: $f — retrying with --reject"
    git apply "$patch" --reject 2>&1 || echo "  FAILED: $f"
  fi
done

echo ""
echo "=== Done ==="
echo "Check for *.rej files to resolve conflicts:"
find "$TGT" -name "*.rej" 2>/dev/null
