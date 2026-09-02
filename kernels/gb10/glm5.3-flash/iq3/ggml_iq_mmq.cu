// SPDX-License-Identifier: AGPL-3.0-only
// Dense GGML IQ/Q MMQ primitives for the exact pinned GLM-5.3 IQ3 recipe.

#include <cuda_bf16.h>
#include "../../qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"
#include "../../qwen3.6-27b/nvfp4/q4k_vendor/quantize_impl.cuh"

template <ggml_type type, int mmq_x, bool need_check>
static __device__ __forceinline__ void atlas_ggml_iq_tile(
        const char * __restrict__ weights, const int * __restrict__ activations,
        __nv_bfloat16 * __restrict__ output, const int output_rows,
        const int batch_rows, const int inner, const int weight_row_stride,
        const int activation_cols, const int output_row_stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y = get_mmq_y_device();

    if (output_rows <= 0 || batch_rows <= 0 || inner <= 0 || inner % qk != 0 ||
        inner % (4 * QK8_1) != 0 ||
        weight_row_stride != inner / qk || activation_cols != batch_rows ||
        output_row_stride != output_rows) {
        return;
    }

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > mmq_x && j >= mmq_x) {
            break;
        }
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int output_tile = blockIdx.x;
    const int batch_tile = blockIdx.y;
    const int activation_offset =
        batch_tile * mmq_x * (int) (sizeof(block_q8_1_mmq) / sizeof(int));
    const int output_offset =
        batch_tile * mmq_x * output_row_stride + output_tile * mmq_y;
    const int weight_offset = output_tile * mmq_y * weight_row_stride;
    const int max_output_row = output_rows - output_tile * mmq_y - 1;
    const int max_batch_row = batch_rows - batch_tile * mmq_x - 1;

    mul_mat_q_process_tile<type, mmq_x, need_check, false, __nv_bfloat16>(
        weights, weight_offset, activations + activation_offset, ids_dst_shared,
        output + output_offset, nullptr, weight_row_stride, activation_cols,
        output_row_stride, max_output_row, max_batch_row, 0, inner / qk);
}

#define ATLAS_DEFINE_GGML_IQ_MMQ(tag, type)                                      \
extern "C" __global__ void __launch_bounds__(256, 1)                            \
atlas_##tag##_mmq128_nc(                                                         \
        const char * weights, const int * activations, __nv_bfloat16 * output,   \
        int output_rows, int batch_rows, int inner, int weight_row_stride,       \
        int activation_cols, int output_row_stride) {                            \
    atlas_ggml_iq_tile<type, 128, false>(                                         \
        weights, activations, output, output_rows, batch_rows, inner,             \
        weight_row_stride, activation_cols, output_row_stride);                   \
}                                                                                 \
extern "C" __global__ void __launch_bounds__(256, 1)                            \
atlas_##tag##_mmq128_wc(                                                         \
        const char * weights, const int * activations, __nv_bfloat16 * output,   \
        int output_rows, int batch_rows, int inner, int weight_row_stride,       \
        int activation_cols, int output_row_stride) {                            \
    atlas_ggml_iq_tile<type, 128, true>(                                          \
        weights, activations, output, output_rows, batch_rows, inner,             \
        weight_row_stride, activation_cols, output_row_stride);                   \
}

ATLAS_DEFINE_GGML_IQ_MMQ(q2_k, GGML_TYPE_Q2_K)
ATLAS_DEFINE_GGML_IQ_MMQ(q3_k, GGML_TYPE_Q3_K)
ATLAS_DEFINE_GGML_IQ_MMQ(q4_k, GGML_TYPE_Q4_K)
ATLAS_DEFINE_GGML_IQ_MMQ(q5_k, GGML_TYPE_Q5_K)
ATLAS_DEFINE_GGML_IQ_MMQ(q6_k, GGML_TYPE_Q6_K)
ATLAS_DEFINE_GGML_IQ_MMQ(q8_0, GGML_TYPE_Q8_0)
ATLAS_DEFINE_GGML_IQ_MMQ(iq2_xxs, GGML_TYPE_IQ2_XXS)
ATLAS_DEFINE_GGML_IQ_MMQ(iq2_xs, GGML_TYPE_IQ2_XS)
ATLAS_DEFINE_GGML_IQ_MMQ(iq2_s, GGML_TYPE_IQ2_S)
ATLAS_DEFINE_GGML_IQ_MMQ(iq3_xxs, GGML_TYPE_IQ3_XXS)
ATLAS_DEFINE_GGML_IQ_MMQ(iq3_s, GGML_TYPE_IQ3_S)
ATLAS_DEFINE_GGML_IQ_MMQ(iq4_xs, GGML_TYPE_IQ4_XS)

template <mmq_q8_1_ds_layout layout>
static __device__ __forceinline__ void atlas_quantize_bf16_q8(
        const __nv_bfloat16 * input, void * output, long row_width,
        long row_stride, long padded_width, int rows) {
    quantize_mmq_q8_1_worker<layout, __nv_bfloat16>(
        input, nullptr, output, row_width, row_stride, 0, 0, padded_width, rows, 1);
}

extern "C" __global__ void atlas_q8_1_quantize_d4_bf16(
        const __nv_bfloat16 * input, void * output, long row_width,
        long row_stride, long padded_width, int rows) {
    atlas_quantize_bf16_q8<MMQ_Q8_1_DS_LAYOUT_D4>(
        input, output, row_width, row_stride, padded_width, rows);
}

extern "C" __global__ void atlas_q8_1_quantize_ds4_bf16(
        const __nv_bfloat16 * input, void * output, long row_width,
        long row_stride, long padded_width, int rows) {
    atlas_quantize_bf16_q8<MMQ_Q8_1_DS_LAYOUT_DS4>(
        input, output, row_width, row_stride, padded_width, rows);
}

extern "C" __global__ void atlas_q8_1_quantize_d2s6_bf16(
        const __nv_bfloat16 * input, void * output, long row_width,
        long row_stride, long padded_width, int rows) {
    atlas_quantize_bf16_q8<MMQ_Q8_1_DS_LAYOUT_D2S6>(
        input, output, row_width, row_stride, padded_width, rows);
}

// ---------------------------------------------------------------------------
// GROUPED MoE MMQ: one launch for all TOP_K expert slots, weights resolved
// ON DEVICE from a route-id array that never leaves the GPU.
//
// The serial seam this replaces did, per MoE layer: copy 32 bytes of route ids
// to the host with cuMemcpyDtoHAsync_v2 + cuStreamSynchronize (a full pipeline
// drain), then use those ids on the CPU to compute `base + id*expert_bytes`
// and issue one launch per slot. Measured, that drain is 91.5% of decode wall
// time -- 42 of them per token at 3.91 ms each.
//
// It is removable because the expert bank is a single packed tensor with a
// UNIFORM stride: glm53_gguf.rs derives `expert_bytes` once per bank and
// refuses to load a tensor whose length is not `expert_bytes * experts`. So
// the host round trip existed only to perform an integer multiply-add that a
// kernel can do itself.
//
// blockIdx.z is the (row, slot) pair: slot = z % top_k, row = z / top_k.
// Grouping over BOTH is deliberate -- rows are what prefill and speculative
// verify need, and a slot-only grouping would fix decode and leave those
// exactly where they are.
template <ggml_type type, int mmq_x, bool need_check>
static __device__ __forceinline__ void atlas_ggml_iq_grouped_tile(
        const char * __restrict__ expert_base, const int expert_bytes,
        const int * __restrict__ route_ids, const int top_k, const int experts,
        const int * __restrict__ activations,
        __nv_bfloat16 * __restrict__ output,
        const int output_rows, const int batch_rows, const int inner,
        const int weight_row_stride, const int activation_cols,
        const int output_row_stride) {
    const int pair = blockIdx.z;
    const int slot = pair % top_k;
    const int row  = pair / top_k;

    // The route id is read from DEVICE memory. This is the whole point: the
    // host never sees it, so nothing has to synchronize to learn it.
    const int expert_id = route_ids[row * top_k + slot];
    // Fail closed rather than address outside the bank. A corrupt id would
    // otherwise read arbitrary weights and produce fluent, wrong output.
    if (expert_id < 0 || expert_id >= experts) {
        return;
    }

    const char * __restrict__ weights =
        expert_base + (size_t) expert_id * (size_t) expert_bytes;
    // Each (row, slot) writes its own output slice; slices never overlap, so
    // no reduction or atomics are needed here.
    __nv_bfloat16 * __restrict__ out_slice =
        output + (size_t) pair * (size_t) output_rows;

    atlas_ggml_iq_tile<type, mmq_x, need_check>(
        weights, activations, out_slice, output_rows, batch_rows, inner,
        weight_row_stride, activation_cols, output_row_stride);
}

#define ATLAS_DEFINE_GROUPED_MOE_MMQ(tag, type)                                  \
extern "C" __global__ void __launch_bounds__(256, 1)                            \
atlas_##tag##_moe_grouped_nc(                                                    \
        const char * expert_base, int expert_bytes,                              \
        const int * route_ids, int top_k, int experts,                           \
        const int * activations, __nv_bfloat16 * output,                         \
        int output_rows, int batch_rows, int inner, int weight_row_stride,       \
        int activation_cols, int output_row_stride) {                            \
    atlas_ggml_iq_grouped_tile<type, 128, false>(                  \
        expert_base, expert_bytes, route_ids, top_k, experts, activations,        \
        output, output_rows, batch_rows, inner, weight_row_stride,                \
        activation_cols, output_row_stride);                                      \
}                                                                                 \
extern "C" __global__ void __launch_bounds__(256, 1)                            \
atlas_##tag##_moe_grouped_wc(                                                    \
        const char * expert_base, int expert_bytes,                              \
        const int * route_ids, int top_k, int experts,                           \
        const int * activations, __nv_bfloat16 * output,                         \
        int output_rows, int batch_rows, int inner, int weight_row_stride,       \
        int activation_cols, int output_row_stride) {                            \
    atlas_ggml_iq_grouped_tile<type, 128, true>(                   \
        expert_base, expert_bytes, route_ids, top_k, experts, activations,        \
        output, output_rows, batch_rows, inner, weight_row_stride,                \
        activation_cols, output_row_stride);                                      \
}

ATLAS_DEFINE_GROUPED_MOE_MMQ(q2_k, GGML_TYPE_Q2_K)
ATLAS_DEFINE_GROUPED_MOE_MMQ(q4_k, GGML_TYPE_Q4_K)
ATLAS_DEFINE_GROUPED_MOE_MMQ(q5_k, GGML_TYPE_Q5_K)
ATLAS_DEFINE_GROUPED_MOE_MMQ(q6_k, GGML_TYPE_Q6_K)
ATLAS_DEFINE_GROUPED_MOE_MMQ(q8_0, GGML_TYPE_Q8_0)
ATLAS_DEFINE_GROUPED_MOE_MMQ(iq2_xxs, GGML_TYPE_IQ2_XXS)
ATLAS_DEFINE_GROUPED_MOE_MMQ(iq2_xs, GGML_TYPE_IQ2_XS)
ATLAS_DEFINE_GROUPED_MOE_MMQ(iq2_s, GGML_TYPE_IQ2_S)
ATLAS_DEFINE_GROUPED_MOE_MMQ(iq3_xxs, GGML_TYPE_IQ3_XXS)
ATLAS_DEFINE_GROUPED_MOE_MMQ(iq3_s, GGML_TYPE_IQ3_S)
ATLAS_DEFINE_GROUPED_MOE_MMQ(iq4_xs, GGML_TYPE_IQ4_XS)
