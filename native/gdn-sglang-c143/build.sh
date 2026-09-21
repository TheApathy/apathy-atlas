#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd "$here/../.." && pwd -P)
wy32="$repo/kernels/gb10/common/gated_delta_rule_wy32_gatecache.cu"
gdn_source="$here/src/gated_delta_rule_sglang_c143.cu"
atlas_source_sha=$(sha256sum "$wy32" | awk '{print $1}')
[[ "$atlas_source_sha" == f73a7071aa8160960e9221bb584caa1e9f733856c0d82bcc39085e9e881d6aed ]] || {
  echo "current Atlas WY32 source drift" >&2; exit 2;
}
gdn_source_sha=$(sha256sum "$gdn_source" | awk '{print $1}')
atlas_bridge="$here/src/atlas_wy32_bridge.cu"
atlas_bridge_sha=$(sha256sum "$atlas_bridge" | awk '{print $1}')
mkdir -p "$here/build"
nvcc=(/usr/local/cuda/bin/nvcc -std=c++17 -O3 --use_fast_math -shared -Xcompiler=-fPIC
      -gencode arch=compute_121a,code=sm_121a)
strip_bin=/usr/bin/strip
[[ -x "$strip_bin" ]] || {
  echo "required deterministic strip tool is missing: $strip_bin" >&2; exit 2;
}
"${nvcc[@]}" -DATLAS_GDN_C143_SOURCE_SHA256=\"$gdn_source_sha\" \
  "$gdn_source" -o "$here/build/libatlas_gdn_c143_sm121.so"
"${nvcc[@]}" -DATLAS_WY32_SOURCE_SHA256=\"$atlas_source_sha\" \
  -DATLAS_WY32_BRIDGE_SHA256=\"$atlas_bridge_sha\" \
  "$atlas_bridge" -o "$here/build/libatlas_gdn_wy32_sm121.so"

# nvcc embeds a random tmpxft_* translation-unit name in non-runtime ELF
# symbol/string tables. Remove only unneeded symbols before full-file hashing;
# dynamic symbols and the loadable CUDA fatbin remain intact for dlopen/dlsym.
"$strip_bin" --strip-unneeded "$here/build/libatlas_gdn_c143_sm121.so"
"$strip_bin" --strip-unneeded "$here/build/libatlas_gdn_wy32_sm121.so"
