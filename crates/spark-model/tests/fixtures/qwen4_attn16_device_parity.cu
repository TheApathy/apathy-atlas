// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../../../kernels/gb10/common/rope_mrope_interleaved.cu"
#include "../../../../kernels/gb10/common/w4a16_gemv_rt.cu"
#include "../../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_attn16_device.cu"

#define CUDA_OK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::exit(2); \
    } \
} while (0)

static constexpr unsigned ROWS = 32, HALF = 16, STRIDE = 13312, K_OFFSET = 12288;
static constexpr unsigned NQ = 24, NKV = 2, HD = 256, ROTARY = 64;

static bool test_mrope(bool aliased) {
    const size_t words = (size_t)ROWS * STRIDE;
    std::vector<unsigned short> input(words), reference(words), candidate(words);
    for (size_t i = 0; i < words; ++i)
        input[i] = (unsigned short)(((i & 1) << 15) | 0x3f00 | (i & 0x7f));
    unsigned short *d_ref, *d_candidate;
    unsigned *d_t, *d_h, *d_w;
    CUDA_OK(cudaMalloc(&d_ref, words * 2));
    CUDA_OK(cudaMalloc(&d_candidate, words * 2));
    CUDA_OK(cudaMalloc(&d_t, ROWS * 4));
    CUDA_OK(cudaMalloc(&d_h, ROWS * 4));
    CUDA_OK(cudaMalloc(&d_w, ROWS * 4));
    CUDA_OK(cudaMemcpy(d_ref, input.data(), words * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_candidate, input.data(), words * 2, cudaMemcpyHostToDevice));
    std::vector<unsigned> t(ROWS), h(ROWS), w(ROWS);
    for (unsigned row = 0; row < ROWS; ++row) {
        t[row] = row * 7 + 3;
        h[row] = aliased ? t[row] : row * 5 + 11;
        w[row] = aliased ? t[row] : row * 3 + 19;
    }
    CUDA_OK(cudaMemcpy(d_t, t.data(), ROWS * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_h, h.data(), ROWS * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_w, w.data(), ROWS * 4, cudaMemcpyHostToDevice));
    unsigned *h_ptr = aliased ? d_t : d_h;
    unsigned *w_ptr = aliased ? d_t : d_w;
    for (unsigned start : {0u, HALF})
        rope_forward_mrope_interleaved_strided<<<dim3(NQ + NKV, 4, 1), 128>>>(
            (__nv_bfloat16*)d_ref + (size_t)start * STRIDE,
            d_t + start, h_ptr + start, w_ptr + start, HALF, STRIDE, K_OFFSET,
            NQ, NKV, HD, ROTARY, 1000000.0f);
    rope_forward_mrope_interleaved_strided<<<dim3(NQ + NKV, 8, 1), 128>>>(
        (__nv_bfloat16*)d_candidate, d_t, h_ptr, w_ptr, ROWS, STRIDE, K_OFFSET,
        NQ, NKV, HD, ROTARY, 1000000.0f);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(reference.data(), d_ref, words * 2, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(candidate.data(), d_candidate, words * 2, cudaMemcpyDeviceToHost));
    bool ok = reference == candidate;
    if (!ok) for (size_t i = 0; i < words; ++i) if (reference[i] != candidate[i]) {
        std::fprintf(stderr, "MRoPE mismatch alias=%d word=%zu ref=%04x got=%04x\n",
                     aliased, i, reference[i], candidate[i]);
        break;
    }
    CUDA_OK(cudaFree(d_ref)); CUDA_OK(cudaFree(d_candidate));
    CUDA_OK(cudaFree(d_t)); CUDA_OK(cudaFree(d_h)); CUDA_OK(cudaFree(d_w));
    return ok;
}

static bool test_metadata(unsigned blocks, unsigned tile_start) {
    const int canary = 0x5a17c0de;
    const size_t guard = 8, table_words = (size_t)ROWS * blocks;
    std::vector<int> source(blocks + 2 * guard, canary);
    std::vector<int> reference_tables(table_words + 2 * guard, canary);
    std::vector<int> tables(table_words + 2 * guard, canary);
    std::vector<unsigned> reference_lengths(ROWS + 2 * guard, (unsigned)canary);
    std::vector<unsigned> lengths(ROWS + 2 * guard, (unsigned)canary);
    std::vector<unsigned> reference_final(1 + 2 * guard, (unsigned)canary);
    std::vector<unsigned> final_length(1 + 2 * guard, (unsigned)canary);
    for (unsigned i = 0; i < blocks; ++i) source[guard + i] = 100003 - (int)i * 37;
    int *d_source, *d_ref_tables, *d_tables;
    unsigned *d_ref_lengths, *d_lengths, *d_ref_final, *d_final;
    CUDA_OK(cudaMalloc(&d_source, source.size() * 4));
    CUDA_OK(cudaMalloc(&d_ref_tables, reference_tables.size() * 4));
    CUDA_OK(cudaMalloc(&d_tables, tables.size() * 4));
    CUDA_OK(cudaMalloc(&d_ref_lengths, reference_lengths.size() * 4));
    CUDA_OK(cudaMalloc(&d_lengths, lengths.size() * 4));
    CUDA_OK(cudaMalloc(&d_ref_final, reference_final.size() * 4));
    CUDA_OK(cudaMalloc(&d_final, final_length.size() * 4));
    CUDA_OK(cudaMemcpy(d_source, source.data(), source.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ref_tables, reference_tables.data(), reference_tables.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tables, tables.data(), tables.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ref_lengths, reference_lengths.data(), reference_lengths.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_lengths, lengths.data(), lengths.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ref_final, reference_final.data(), reference_final.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_final, final_length.data(), final_length.size() * 4, cudaMemcpyHostToDevice));
    unsigned grid = (unsigned)((table_words + 255) / 256);
    for (unsigned start : {0u, HALF})
        qwen4_attn16_expand_meta<<<grid, 256>>>(
            d_source + guard, d_ref_tables + guard + (size_t)start * blocks,
            d_ref_lengths + guard + start, d_ref_final + guard, blocks, tile_start + start);
    qwen4_attn32_expand_meta<<<grid, 256>>>(
        d_source + guard, d_tables + guard, d_lengths + guard, d_final + guard,
        blocks, tile_start);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int> source_after(source.size());
    CUDA_OK(cudaMemcpy(source_after.data(), d_source, source.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(reference_tables.data(), d_ref_tables, reference_tables.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(tables.data(), d_tables, tables.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(reference_lengths.data(), d_ref_lengths, reference_lengths.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(lengths.data(), d_lengths, lengths.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(reference_final.data(), d_ref_final, reference_final.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(final_length.data(), d_final, final_length.size() * 4, cudaMemcpyDeviceToHost));
    bool ok = source == source_after && reference_tables == tables &&
        reference_lengths == lengths && reference_final == final_length;
    for (size_t i = 0; i < table_words; ++i)
        ok &= tables[guard + i] == source[guard + i % blocks];
    for (unsigned row = 0; row < ROWS; ++row)
        ok &= lengths[guard + row] == tile_start + row + 1;
    ok &= final_length[guard] == tile_start + ROWS;
    for (size_t i = 0; i < guard; ++i) {
        ok &= tables[i] == canary && tables[guard + table_words + i] == canary;
        ok &= lengths[i] == (unsigned)canary && lengths[guard + ROWS + i] == (unsigned)canary;
        ok &= final_length[i] == (unsigned)canary && final_length[guard + 1 + i] == (unsigned)canary;
    }
    if (!ok) std::fprintf(stderr, "metadata mismatch blocks=%u start=%u\n", blocks, tile_start);
    CUDA_OK(cudaFree(d_source)); CUDA_OK(cudaFree(d_ref_tables)); CUDA_OK(cudaFree(d_tables));
    CUDA_OK(cudaFree(d_ref_lengths)); CUDA_OK(cudaFree(d_lengths));
    CUDA_OK(cudaFree(d_ref_final)); CUDA_OK(cudaFree(d_final));
    return ok;
}

static bool test_o32() {
    constexpr unsigned N = 2560, K = 6144;
    constexpr size_t GUARD = 64;
    const size_t a_words = (size_t)ROWS * K;
    const size_t packed_bytes = (size_t)N * K / 2;
    const size_t scale_bytes = (size_t)N * K / 16;
    const size_t c_words = (size_t)ROWS * N;
    std::vector<unsigned short> a(a_words), a_after(a_words);
    std::vector<unsigned char> packed(packed_bytes), packed_after(packed_bytes);
    std::vector<unsigned char> scales(scale_bytes, 0x38), scales_after(scale_bytes);
    std::vector<unsigned short> reference(c_words + 2 * GUARD, 0x5a17);
    std::vector<unsigned short> candidate(c_words + 2 * GUARD, 0x5a17);
    for (size_t i = 0; i < a_words; ++i) a[i] = 0x3e00u + (i % 127);
    for (size_t i = 0; i < packed_bytes; ++i)
        packed[i] = (unsigned char)(((i * 7) & 7) | (((i * 11) & 7) << 4));
    unsigned short *d_a, *d_reference, *d_candidate;
    unsigned char *d_packed, *d_scales;
    CUDA_OK(cudaMalloc(&d_a, a_words * 2));
    CUDA_OK(cudaMalloc(&d_packed, packed_bytes));
    CUDA_OK(cudaMalloc(&d_scales, scale_bytes));
    CUDA_OK(cudaMalloc(&d_reference, reference.size() * 2));
    CUDA_OK(cudaMalloc(&d_candidate, candidate.size() * 2));
    CUDA_OK(cudaMemcpy(d_a, a.data(), a_words * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_scales, scales.data(), scale_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_reference, reference.data(), reference.size() * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_candidate, candidate.data(), candidate.size() * 2, cudaMemcpyHostToDevice));
    const unsigned grid = (N + 7) / 8;
    for (unsigned start : {0u, HALF})
        w4a16_gemv_batch_logits_exact_rt2_m17<<<grid, 256>>>(
            (const __nv_bfloat16*)d_a + (size_t)start * K, d_packed, d_scales, 0.5f,
            (__nv_bfloat16*)(d_reference + GUARD + (size_t)start * N), HALF, N, K);
    w4a16_gemv_batch_logits_exact_rt2_m32<<<grid, 256>>>(
        (const __nv_bfloat16*)d_a, d_packed, d_scales, 0.5f,
        (__nv_bfloat16*)(d_candidate + GUARD), ROWS, N, K);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(a_after.data(), d_a, a_words * 2, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(packed_after.data(), d_packed, packed_bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(scales_after.data(), d_scales, scale_bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(reference.data(), d_reference, reference.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(candidate.data(), d_candidate, candidate.size() * 2, cudaMemcpyDeviceToHost));
    bool ok = a == a_after && packed == packed_after && scales == scales_after &&
        reference == candidate;
    if (!ok) std::fprintf(stderr, "exact O32 mismatch or mutation\n");
    CUDA_OK(cudaFree(d_a)); CUDA_OK(cudaFree(d_packed)); CUDA_OK(cudaFree(d_scales));
    CUDA_OK(cudaFree(d_reference)); CUDA_OK(cudaFree(d_candidate));
    return ok;
}

int main() {
    bool ok = test_mrope(true) && test_mrope(false) && test_o32();
    for (unsigned blocks : {1u, 9u, 128u})
        for (unsigned start : {0u, 16u, 2032u}) ok &= test_metadata(blocks, start);
    std::printf("F40 raw 32-vs-2x16 parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
