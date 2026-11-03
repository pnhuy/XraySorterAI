#!/usr/bin/env bash
# Upload ./dist to the Hugging Face model repo.
#
#   ./upload.sh                         # uploads to pnhuy/xraysorterai
#   ./upload.sh -m "Update inference"   # custom commit message
#   HF_REPO=user/other ./upload.sh      # different repo
#   ./upload.sh --dry-run               # show what would be uploaded, upload nothing
#
# Requires: huggingface_hub (pip install -U huggingface_hub) and a login (hf auth login)
# or an HF_TOKEN environment variable with write access.
set -euo pipefail

REPO="${HF_REPO:-pnhuy/xraysorterai}"
DIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dist"
MSG="Upload XraySorterAI ONNX models"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message) MSG="$2"; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --- sanity checks -----------------------------------------------------------
for f in onnx/projection.onnx onnx/side.onnx config.json inference.py README.md; do
  [[ -f "$DIST/$f" ]] || { echo "error: $DIST/$f missing - run 'python export.py' first" >&2; exit 1; }
done

find "$DIST" -name '__pycache__' -type d -prune -exec rm -rf {} +

# samples/ (local test images) are never uploaded.
EXCLUDES=(--exclude "__pycache__/*" --exclude "*.pyc" --exclude ".DS_Store" --exclude "samples/*")

echo "repo   : https://huggingface.co/$REPO"
echo "folder : $DIST"
du -sh "$DIST" | awk '{print "size   : " $1}'
if [[ $DRY -eq 1 ]]; then
  echo "--dry-run: would run: hf upload $REPO $DIST . ${EXCLUDES[*]}"
  exit 0
fi

if   command -v hf >/dev/null 2>&1;              then HF=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then HF=(huggingface-cli)
else
  echo "error: Hugging Face CLI ('hf') not found in this environment." >&2
  echo "       uv venv:  uv pip install -U huggingface_hub     (a bare 'pip' may install elsewhere)" >&2
  echo "       otherwise: python -m pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]] && ! "${HF[@]}" auth whoami >/dev/null 2>&1 \
   && ! "${HF[@]}" whoami >/dev/null 2>&1; then
  echo "error: not logged in. Run '${HF[0]} auth login' (or 'huggingface-cli login'), or set HF_TOKEN." >&2
  exit 1
fi

# --- upload (dist/ becomes the repo root) -------------------------------------
"${HF[@]}" upload "$REPO" "$DIST" . \
  --repo-type model \
  "${EXCLUDES[@]}" \
  --commit-message "$MSG"

echo "done: https://huggingface.co/$REPO"
