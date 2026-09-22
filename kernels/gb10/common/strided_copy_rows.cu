// SPDX-License-Identifier: AGPL-3.0-only
// Byte-exact strided row copy: rows x row_bytes from src (stride src_stride)
// to dst (stride dst_stride). Replaces per-row cuMemcpyDtoDAsync loops
// (e.g. the DFlash prefill capture: 2048 copies per captured layer).
// Requires 16-byte aligned base pointers, strides and row_bytes.
extern "C" __global__ void strided_copy_rows_16(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int rows,
    unsigned int row_bytes,
    unsigned long long src_stride,
    unsigned long long dst_stride
) {
    const unsigned int row = blockIdx.y;
    if (row >= rows) return;
    const unsigned int chunks = row_bytes >> 4;
    const uint4* s = (const uint4*)(src + (unsigned long long)row * src_stride);
    uint4* d = (uint4*)(dst + (unsigned long long)row * dst_stride);
    for (unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += gridDim.x * blockDim.x) {
        d[c] = s[c];
    }
}
