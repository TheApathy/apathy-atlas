#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Run from the checkout root. Includes dirty/untracked build inputs and aliases.
set -euo pipefail
git ls-files -c -o --exclude-standard -z -- \
  Cargo.toml Cargo.lock rust-toolchain.toml .cargo crates kernels |
  sort -zu |
  while IFS= read -r -d '' candidate; do
    if [[ -d "$candidate" ]]; then
      find -L "$candidate" -type f -print0
    elif [[ -f "$candidate" ]]; then
      printf '%s\0' "$candidate"
    else
      printf 'Missing build input: %s\n' "$candidate" >&2
      exit 1
    fi
  done |
  sort -zu |
  xargs -0 -r sha256sum |
  sha256sum
