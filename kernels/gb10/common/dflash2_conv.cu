// SPDX-License-Identifier: AGPL-3.0-only
//
// DFlash2 `GroupedDynamicCausalConv`, ported from Avarok-Cybersecurity/atlas
// PR #648 (`kernels/gb10/common/dflash2.cu`, kernel `dflash2_conv2`).
//
// The upstream kernel is ONE entry point that selects the stage with an
// `app_off` argument and is launched grid=(rows), block=(threads over h).
// This tree's Rust ops (layers/ops/dflash2_conv.rs) predate that and expect
// TWO entry points with a FLAT element-parallel launch
// (grid = ceil(n_attn*h/256), block = 256). Ported to this ABI rather than
// changing the Rust, because `examples/dflash2_conv_selector_microtest.rs`
// already validates that ABI against CPU references of the reference math
// (z-lab/dflash `dflash/model.py`) — keeping it means the port has a gate.
//
// Math:
//   out[l,g,s] = Σ_o (base[stage][o][g*GS+s] + dyn[l,stage,o,g]) * x[l-o,g,s]
// with x[-1] = 0 (causal pad at the block start), GROUP_SIZE 16, KERNEL_SIZE 2.

#include <cuda_bf16.h>

#define DF2_GS 16u
#define DF2_K  2u

// Stage 0 over `hidden`, plus export of the stage-1 dynamic rows for `finish`.
//   x        [n_attn, h]                 h = groups*16
//   dyn      [n_attn, 2*K*groups]        both stages
//   base     [2, K, h]                   stage 0 is the first K*h
//   out      [n_attn, h]
//   dyn1_out [n_attn, K*groups]          stage-1 slice, consumed by finish
extern "C" __global__ void dflash2_conv_prepare(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dyn,
    const __nv_bfloat16* __restrict__ base,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ dyn1_out,
    unsigned int n_attn,
    unsigned int groups
) {
    const unsigned int h = groups * DF2_GS;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_attn * h) return;

    const unsigned int t = idx / h;
    const unsigned int c = idx - t * h;
    const unsigned int g = c / DF2_GS;

    const unsigned int dyn_stride = 2u * DF2_K * groups;
    const __nv_bfloat16* dr = dyn + (size_t)t * dyn_stride;   // stage 0 at +0

    float acc = (__bfloat162float(base[c]) + __bfloat162float(dr[g]))
              * __bfloat162float(x[(size_t)t * h + c]);
    if (t > 0) {
        acc += (__bfloat162float(base[h + c]) + __bfloat162float(dr[groups + g]))
             * __bfloat162float(x[(size_t)(t - 1) * h + c]);
    }
    out[(size_t)t * h + c] = __float2bfloat16(acc);

    // Export stage-1 dyn. K*groups (= 2*groups) is always < h (= 16*groups),
    // so the low threads of each row cover the slice exactly once.
    const unsigned int d1 = DF2_K * groups;
    if (c < d1) {
        dyn1_out[(size_t)t * d1 + c] = dr[d1 + c];
    }
}

// Stage 1 over the sublayer output.
//   dyn  [n_attn, K*groups]   the slice `prepare` exported
//   base [2, K, h]            stage 1 begins at K*h
extern "C" __global__ void dflash2_conv_finish(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dyn,
    const __nv_bfloat16* __restrict__ base,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_attn,
    unsigned int groups
) {
    const unsigned int h = groups * DF2_GS;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_attn * h) return;

    const unsigned int t = idx / h;
    const unsigned int c = idx - t * h;
    const unsigned int g = c / DF2_GS;

    const unsigned int dyn_stride = DF2_K * groups;
    const __nv_bfloat16* dr = dyn + (size_t)t * dyn_stride;
    const __nv_bfloat16* b1 = base + (size_t)DF2_K * h;       // stage 1

    float acc = (__bfloat162float(b1[c]) + __bfloat162float(dr[g]))
              * __bfloat162float(x[(size_t)t * h + c]);
    if (t > 0) {
        acc += (__bfloat162float(b1[h + c]) + __bfloat162float(dr[groups + g]))
             * __bfloat162float(x[(size_t)(t - 1) * h + c]);
    }
    out[(size_t)t * h + c] = __float2bfloat16(acc);
}
