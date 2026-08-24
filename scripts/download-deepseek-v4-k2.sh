#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

readonly REPO="wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1"
readonly REVISION="68eaca43e99bfbfd697a5559c7796b983deb38f8"
readonly DEST="${1:-/home/flocka/models/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1}"

command -v hf >/dev/null || {
  echo "hf CLI is required (pip install huggingface_hub)" >&2
  exit 1
}
mkdir -p "$DEST"
exec hf download "$REPO" --revision "$REVISION" --local-dir "$DEST"
