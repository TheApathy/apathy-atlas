#!/usr/bin/env bash
set -euo pipefail

native_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${BUILD_DIR:-${native_dir}/build}
cuda_root=${CUDA_HOME:-/usr/local/cuda}
library=${build_dir}/libatlas_fi_fp4_sm121.so

bash "${native_dir}/verify-deps.sh"
cc -x c -fsyntax-only "${native_dir}/include/atlas_fi_fp4.h"
test -s "${library}"

actual_exports=$(
  nm -D --defined-only "${library}" |
    awk '$2 == "T" {print $3}' |
    sed 's/@@ATLAS_FI_FP4_0$//' |
    sort
)
expected_exports=$'atlas_fi_nvfp4_sm121_bf16\natlas_fi_nvfp4_sm121_last_error\natlas_fi_nvfp4_sm121_workspace_size'
[[ "${actual_exports}" == "${expected_exports}" ]]

actual_needed=$(
  readelf -d "${library}" |
    sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' |
    sort
)
expected_needed=$'ld-linux-aarch64.so.1\nlibc.so.6\nlibcudart.so.13\nlibgcc_s.so.1\nlibstdc++.so.6'
[[ "${actual_needed}" == "${expected_needed}" ]]
readelf -d "${library}" |
  rg -q 'Library soname: \[libatlas_fi_fp4_sm121.so.0\]'
if readelf -Ws "${library}" | rg -qi 'tvm|torch|python'; then
  echo "forbidden runtime dependency symbol" >&2
  exit 1
fi

elf_listing=$("${cuda_root}/bin/cuobjdump" --list-elf "${library}")
[[ $(rg -c 'sm_121a\.cubin' <<<"${elf_listing}") -eq 4 ]]
[[ $(rg -c '^ELF file' <<<"${elf_listing}") -eq 4 ]]

ptx_listing=$("${cuda_root}/bin/cuobjdump" --list-ptx "${library}" 2>&1)
rg -q 'No PTX file found' <<<"${ptx_listing}"

resources=$("${cuda_root}/bin/cuobjdump" --dump-resource-usage "${library}")
resource_rows=$(rg '^  REG:' <<<"${resources}")
[[ $(wc -l <<<"${resource_rows}") -eq 6 ]]
[[ $(rg -c 'REG:168' <<<"${resource_rows}") -eq 6 ]]
[[ $(rg -c 'SHARED:1024' <<<"${resource_rows}") -eq 6 ]]
[[ $(rg -c 'LOCAL:0' <<<"${resource_rows}") -eq 6 ]]
[[ $(rg -c 'STACK:72' <<<"${resource_rows}") -eq 1 ]]
[[ $(rg -c 'STACK:0' <<<"${resource_rows}") -eq 5 ]]

echo "PASS: FlashInfer SM121 C ABI static contract"
