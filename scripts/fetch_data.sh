#!/usr/bin/env bash
# Manual helper (run_full_rerun.sh uses scripts/inputs.py data, which first reuses a verified copy in the cache).
# Download CivilComments-WILDS v1.0 (CC0) and verify the archive and CSV sha256. Usage: fetch_data.sh <dest_dir>
set -euo pipefail
# Some network filesystems refuse chown/chmod/utime: tar must not try to restore owner, mode or mtime.
export TAR_OPTIONS="${TAR_OPTIONS:---no-same-owner --no-same-permissions -m --no-overwrite-dir}"
DEST="${1:?destination directory}"
URL="https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/"
ARCH_SHA=cfc65ff0cfc9e91f9afd2856d2212076ebb94deebfb6c5a1b2a32ccd97149b5f
CSV_SHA=5fbf41b9903e6a8f78a3c2e5ea3a659621336538a48c1be8d051d50082b5af79
mkdir -p "$DEST"
if [ -f "$DEST/all_data_with_identities.csv" ] && echo "$CSV_SHA  $DEST/all_data_with_identities.csv" | sha256sum -c --quiet; then
  echo "CSV present and verified"; exit 0
fi
if [ ! -f "$DEST/civilcomments_v1.0.tar.gz" ]; then
  curl -fL --retry 3 -o "$DEST/civilcomments_v1.0.tar.gz" "$URL"
fi
echo "$ARCH_SHA  $DEST/civilcomments_v1.0.tar.gz" | sha256sum -c --quiet || { echo "archive sha256 mismatch"; exit 2; }
# extract only the named files: the archive's own "./" entry would make tar chmod the existing folder
tar -xzf "$DEST/civilcomments_v1.0.tar.gz" -C "$DEST" --wildcards '*all_data_with_identities.csv' '*RELEASE_v1.0.txt'
CSV=$(find "$DEST" -name all_data_with_identities.csv | head -1)
[ "$CSV" = "$DEST/all_data_with_identities.csv" ] || mv "$CSV" "$DEST/all_data_with_identities.csv"
echo "$CSV_SHA  $DEST/all_data_with_identities.csv" | sha256sum -c --quiet || { echo "CSV sha256 mismatch"; exit 3; }
echo "data verified"
