#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build or verify the exact release binary used by exl3-prefill-max.sh.
set -euo pipefail
export LC_ALL=C

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary=$repo_root/target/release/spark
receipt=$repo_root/target/release/spark.prefill-max-build-receipt
build_inputs=(Cargo.toml Cargo.lock rust-toolchain.toml .cargo crates kernels vendor)

source_sha256() {
  {
    git -C "$repo_root" rev-parse HEAD
    git -C "$repo_root" diff --binary HEAD -- "${build_inputs[@]}"
    while IFS= read -r -d '' relative; do
      printf 'untracked\0%s\0' "$relative"
      if [ -L "$repo_root/$relative" ]; then
        printf 'symlink\0%s\0' "$(readlink "$repo_root/$relative")"
      elif [ -f "$repo_root/$relative" ]; then
        sha256sum "$repo_root/$relative"
      else
        echo "unsupported untracked build input: $relative" >&2
        return 1
      fi
    done < <(
      git -C "$repo_root" ls-files --others --exclude-standard -z -- \
        "${build_inputs[@]}" | LC_ALL=C sort -z
    )
  } | sha256sum | awk '{print $1}'
}

receipt_value() {
  local key=$1
  local count
  count=$(awk -v key="$key" '$1 == key { count++ } END { print count + 0 }' "$receipt")
  if [ "$count" -ne 1 ]; then
    echo "invalid max-prefill build receipt field: $key" >&2
    return 1
  fi
  awk -v key="$key" '$1 == key { print $2 }' "$receipt"
}

verify() {
  if [ ! -f "$receipt" ] || [ -L "$receipt" ]; then
    echo "max-prefill build receipt is missing or unsafe: $receipt" >&2
    echo "run scripts/build-exl3-prefill-max.sh from the intended checkout" >&2
    return 1
  fi
  if [ ! -f "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
    echo "max-prefill release binary is missing, non-executable, or a symlink: $binary" >&2
    return 1
  fi

  local expected_schema expected_commit expected_hw expected_model expected_quant
  local expected_source expected_binary actual_source actual_binary actual_commit
  if [ "$(wc -l <"$receipt")" -ne 7 ]; then
    echo "max-prefill build receipt has unexpected fields" >&2
    return 1
  fi
  expected_schema=$(receipt_value schema)
  expected_commit=$(receipt_value git_commit)
  expected_hw=$(receipt_value target_hw)
  expected_model=$(receipt_value target_model)
  expected_quant=$(receipt_value target_quant)
  expected_source=$(receipt_value source_sha256)
  expected_binary=$(receipt_value binary_sha256)
  actual_commit=$(git -C "$repo_root" rev-parse HEAD)
  if [ "$expected_schema" != atlas-prefill-max-build-v1 ] || \
    [ "$expected_commit" != "$actual_commit" ] || \
    [ "$expected_hw" != gb10 ] || \
    [ "$expected_model" != deepseek-v4-flash ] || \
    [ "$expected_quant" != nvfp4 ]; then
    echo "max-prefill receipt has the wrong schema, commit, or build target" >&2
    return 1
  fi
  actual_source=$(source_sha256)
  actual_binary=$(sha256sum "$binary" | awk '{print $1}')
  if [ "$actual_source" != "$expected_source" ]; then
    echo "max-prefill build inputs changed after the release binary was built" >&2
    echo "run scripts/build-exl3-prefill-max.sh again" >&2
    return 1
  fi
  if [ "$actual_binary" != "$expected_binary" ]; then
    echo "max-prefill release binary does not match its build receipt" >&2
    return 1
  fi
  if ! "$binary" serve --help | grep -Fq \
    'Omitting the option keeps the MODEL.toml default.'; then
    echo "max-prefill release binary has stale FP8 KV calibration semantics" >&2
    return 1
  fi
}

case "${1:-}" in
  --verify-only)
    [ "$#" -eq 1 ] || { echo "usage: $0 [--verify-only]" >&2; exit 2; }
    verify
    echo "verified max-prefill binary: $binary"
    exit 0
    ;;
  '') ;;
  *) echo "usage: $0 [--verify-only]" >&2; exit 2 ;;
esac

source_before=$(source_sha256)
build_target_dir=$repo_root/target/prefill-max-build/$source_before
env -u ATLAS_SKIP_BUILD -u SKIP_ATLAS_BUILD -u ATLAS_EXTRA_NVCC_FLAGS \
  -u NVCC_PREPEND_FLAGS -u NVCC_APPEND_FLAGS \
  CUDA_ROOT=/usr/local/cuda-13.0 \
  CUDA_ROOT_PATH=/usr/local/cuda-13.0 \
  CUDARC_CUDA_VERSION=13000 \
  ATLAS_TARGET_HW=gb10 \
  ATLAS_TARGET_MODEL=deepseek-v4-flash \
  ATLAS_TARGET_QUANT=nvfp4 \
  CARGO_TARGET_DIR="$build_target_dir" \
  cargo build --release --locked --package spark-server --bin spark \
    --manifest-path "$repo_root/Cargo.toml"
source_after=$(source_sha256)
if [ "$source_before" != "$source_after" ]; then
  echo "max-prefill build inputs changed during cargo build; refusing a receipt" >&2
  exit 1
fi
built_binary=$build_target_dir/release/spark
if [ ! -f "$built_binary" ] || [ ! -x "$built_binary" ] || [ -L "$built_binary" ]; then
  echo "cargo did not produce a safe release binary: $built_binary" >&2
  exit 1
fi
if ! "$built_binary" serve --help | grep -Fq \
  'Omitting the option keeps the MODEL.toml default.'; then
  echo "new release binary does not implement MODEL.toml FP8 calibration fallback" >&2
  exit 1
fi

mkdir -p "$repo_root/target/release"
binary_tmp=$(mktemp "$repo_root/target/release/.spark-prefill-binary.XXXXXX")
trap 'rm -f "$binary_tmp"' EXIT
install -m 0755 "$built_binary" "$binary_tmp"
mv -f "$binary_tmp" "$binary"
trap - EXIT

binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
commit=$(git -C "$repo_root" rev-parse HEAD)
receipt_tmp=$(mktemp "$repo_root/target/release/.spark-prefill-receipt.XXXXXX")
trap 'rm -f "$receipt_tmp"' EXIT
{
  printf 'schema atlas-prefill-max-build-v1\n'
  printf 'target_hw gb10\n'
  printf 'target_model deepseek-v4-flash\n'
  printf 'target_quant nvfp4\n'
  printf 'git_commit %s\n' "$commit"
  printf 'source_sha256 %s\n' "$source_after"
  printf 'binary_sha256 %s\n' "$binary_sha256"
} >"$receipt_tmp"
chmod 0444 "$receipt_tmp"
mv -f "$receipt_tmp" "$receipt"
trap - EXIT

verify
echo "wrote max-prefill build receipt: $receipt"
