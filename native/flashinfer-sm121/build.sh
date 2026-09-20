#!/usr/bin/env bash
set -euo pipefail

native_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=${native_dir}/src
include_dir=${native_dir}/include
build_dir=${BUILD_DIR:-${native_dir}/build}
fi_data_root=${FLASHINFER_DATA_ROOT:-/home/flocka/.local/lib/python3.12/site-packages/flashinfer/data}
cuda_root=${CUDA_HOME:-/usr/local/cuda}

if [[ "${ATLAS_FI_SKIP_DEP_VERIFY:-0}" != "1" ]]; then
  bash "${native_dir}/verify-deps.sh"
fi
mkdir -p "${build_dir}"

common=(
  --compiler-options=-fPIC,-fvisibility=hidden
  --expt-relaxed-constexpr
  -static-global-template-stub=false
  -std=c++17
  --threads=1
  -use_fast_math
  -DNDEBUG
  -O3
  -gencode=arch=compute_121a,code=sm_121a
  -DFLASHINFER_ENABLE_FP8_E8M0
  -DFLASHINFER_ENABLE_FP4_E2M1
  -DFLASHINFER_ENABLE_F16
  -DFLASHINFER_ENABLE_BF16
  -DFLASHINFER_ENABLE_FP8_E4M3
  -DFLASHINFER_ENABLE_FP8_E5M2
  -DENABLE_BF16
  -DENABLE_FP4
  -I"${include_dir}"
  -I"${fi_data_root}/include"
  -I"${fi_data_root}/cutlass/include"
  -I"${fi_data_root}/cutlass/tools/util/include"
  -I"${cuda_root}/include"
)

pids=()
for unit in atlas_fi_fp4 inst_128_128_128 inst_128_128_256 inst_256_128_128; do
  "${cuda_root}/bin/nvcc" "${common[@]}" -c "${source_dir}/${unit}.cu" \
    -o "${build_dir}/${unit}.o" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

c++ -shared \
  "${build_dir}/atlas_fi_fp4.o" \
  "${build_dir}/inst_128_128_128.o" \
  "${build_dir}/inst_128_128_256.o" \
  "${build_dir}/inst_256_128_128.o" \
  -L"${cuda_root}/lib64" -L"${cuda_root}/lib64/stubs" \
  -lcudart -lcuda \
  -Wl,-soname,libatlas_fi_fp4_sm121.so.0 \
  -Wl,--version-script="${native_dir}/exports.map" \
  -o "${build_dir}/libatlas_fi_fp4_sm121.so"
