#!/usr/bin/env bash
# Download the owner-approved public bundle without checking out the data branch.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${1:-$ROOT_DIR/../training_assets}"
ARCHIVE_NAME=auv_twophase_gpu_phase1_30ep.tar.gz
cd "$ROOT_DIR"
git lfs version >/dev/null 2>&1 || { echo 'Git LFS is required. On Ubuntu install it with: apt-get update && apt-get install -y git-lfs' >&2; exit 1; }
git lfs install --local
git fetch origin refs/heads/training-data:refs/remotes/origin/training-data
mkdir -p "$ASSET_DIR"
ASSET_DIR="$(cd -- "$ASSET_DIR" && pwd)"
CHECKSUM_FILE="$ASSET_DIR/$ARCHIVE_NAME.sha256"
git show "origin/training-data:$ARCHIVE_NAME.sha256" > "$CHECKSUM_FILE"
EXPECTED="$(awk 'NR==1 {print $1}' "$CHECKSUM_FILE")"
[[ "$EXPECTED" =~ ^[0-9a-f]{64}$ ]] || { echo 'Invalid checksum metadata.' >&2; exit 1; }
TARGET="$ASSET_DIR/$ARCHIVE_NAME"
if [[ -e "$TARGET" ]]; then
    ACTUAL="$(sha256sum "$TARGET" | awk '{print $1}')"
    [[ "$ACTUAL" == "$EXPECTED" ]] || { echo "Different existing archive: $TARGET; refusing to overwrite." >&2; exit 1; }
    echo "Verified existing archive: $TARGET"
else
    TEMP_FILE="$TARGET.download.tmp"
    [[ ! -L "$TEMP_FILE" ]] || { echo 'Refusing symlink temporary file.' >&2; exit 1; }
    git show "origin/training-data:$ARCHIVE_NAME" | GIT_LFS_SKIP_SMUDGE=0 GIT_LFS_SKIP_DOWNLOAD_ERRORS=0 git lfs smudge -- "$ARCHIVE_NAME" > "$TEMP_FILE"
    ACTUAL="$(sha256sum "$TEMP_FILE" | awk '{print $1}')"
    [[ "$ACTUAL" == "$EXPECTED" ]] || { echo 'Downloaded archive checksum mismatch.' >&2; exit 1; }
    mv -- "$TEMP_FILE" "$TARGET"
fi
echo "Download verified: $TARGET"
echo 'Import data with:'
printf 'python server/import_data.py --archive %q\n' "$TARGET"
