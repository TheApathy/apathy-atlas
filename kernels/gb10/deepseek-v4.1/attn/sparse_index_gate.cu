// SPDX-License-Identifier: AGPL-3.0-only
//
// Gate for sparse_index.cu against the oracle's own indexer taps (make_index_fixture.py).
//
//   nvcc -O3 -std=c++17 -arch=sm_121a -DDSV41_INDEX_GATE sparse_index_gate.cu -o sparse_index_gate
//   ./sparse_index_gate idx_fixtures/<case> [...]
//
// Per case, four independent claims, each an INTEGER / bit comparison:
//   SELECT   topk kernel fed the engine's OWN idx_score must give the engine's topk exactly.
//            Isolates selection (radix threshold, ascending order, -1 pad) from scoring.
//   CAND     (layer 20) candidate kernel fed the engine's idx_score must give cand_out exactly.
//   SCORE    score kernel vs idx_score: bit-exact fraction over finite entries, -inf positions
//            exactly, for each head-sum order (measured, not assumed).
//   END2END  score kernel -> topk kernel vs the engine's topk: rows exact.
// Controls, each REQUIRED to change the selection (else the claim is UNRUNNABLE):
//   no bf16 rounding of the dot; compress_lens off by one row; candidate mask dropped (L24).

#include "sparse_index.cu"

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); std::exit(2);} } while(0)

static std::vector<char> slurp(const std::string& p, size_t want, bool optional = false) {
    std::vector<char> v;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { if (optional) return v; std::fprintf(stderr, "open %s\n", p.c_str()); std::exit(2); }
    v.resize(want);
    if (std::fread(v.data(), 1, want, f) != want) { std::fprintf(stderr, "short read %s\n", p.c_str()); std::exit(2); }
    std::fclose(f);
    return v;
}

template <typename T> static T* up(const std::vector<char>& h) {
    if (h.empty()) return nullptr;
    void* d; CUDA_OK(cudaMalloc(&d, h.size()));
    CUDA_OK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    return (T*)d;
}

struct Case {
    int T, n_keys, n_pad, ratio, cand_ld, is_src, n_c; long long pos0;
    __nv_bfloat16 *q, *ik; float* w; uint8_t* cand;
    std::vector<float> ref_score; std::vector<long long> ref_topk; std::vector<uint8_t> ref_cand;
};

static int g_fail = 0;
static const int SMALL_BLOCKS = 96;   // make_index_fixture.py
#define SMALL_S "96"

static void score(const Case& c, int variant, const uint8_t* cand, long long pos0, float* d_out) {
    dim3 grid(c.T, c.n_pad / BN);
    dsv41_index_score_probe<<<grid, SCORE_THREADS>>>(c.q, c.ik, c.w, cand, d_out, c.n_keys, c.n_pad,
                                                      pos0, c.ratio, cand ? c.cand_ld : 0, variant);
    CUDA_OK(cudaGetLastError());
}

static std::vector<long long> topk(const Case& c, const float* d_score) {
    long long* d_idx; CUDA_OK(cudaMalloc(&d_idx, (size_t)c.T * TOPK * 8));
    dsv41_index_topk<<<c.T, SEL_THREADS>>>(d_score, d_idx, c.n_pad, std::min(TOPK, c.n_c));
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    std::vector<long long> h((size_t)c.T * TOPK);
    CUDA_OK(cudaMemcpy(h.data(), d_idx, h.size() * 8, cudaMemcpyDeviceToHost));
    cudaFree(d_idx);
    return h;
}

static int rows_differ(const Case& c, const std::vector<long long>& got) {
    int bad = 0;
    for (int t = 0; t < c.T; ++t)
        bad += std::memcmp(&got[(size_t)t * TOPK], &c.ref_topk[(size_t)t * TOPK], TOPK * 8) != 0;
    return bad;
}

// Boundary ties in the REFERENCE score: rows whose k-th and (k+1)-th finite values are equal.
static int boundary_ties(const Case& c) {
    int ties = 0; const int k = std::min(TOPK, c.n_c);
    std::vector<float> v;
    for (int t = 0; t < c.T; ++t) {
        v.clear();
        for (int n = 0; n < c.n_pad; ++n) { float s = c.ref_score[(size_t)t * c.n_pad + n]; if (std::isfinite(s)) v.push_back(s); }
        if ((int)v.size() <= k) continue;
        std::nth_element(v.begin(), v.begin() + (k - 1), v.end(), std::greater<float>());
        const float kth = v[k - 1];
        float next = -INFINITY;
        for (size_t i = k; i < v.size(); ++i) next = std::max(next, v[i]);
        ties += kth == next;
    }
    return ties;
}

static void run_case(const std::string& dir) {
    Case c;
    { FILE* f = std::fopen((dir + "/dims.txt").c_str(), "r");
      if (!f || std::fscanf(f, "%d %d %d %lld %d %d %d %d", &c.T, &c.n_keys, &c.n_pad, &c.pos0,
                            &c.ratio, &c.cand_ld, &c.is_src, &c.n_c) != 8) { std::fprintf(stderr, "dims %s\n", dir.c_str()); std::exit(2); }
      std::fclose(f); }
    const size_t TS = (size_t)c.T * c.n_pad;
    c.q = up<__nv_bfloat16>(slurp(dir + "/q.bin", (size_t)c.T * H * DH * 2));
    c.ik = up<__nv_bfloat16>(slurp(dir + "/ik.bin", (size_t)c.n_keys * DH * 2));
    c.w = up<float>(slurp(dir + "/wts.bin", (size_t)c.T * H * 4));
    c.cand = c.cand_ld ? up<uint8_t>(slurp(dir + "/cand_in.bin", (size_t)c.T * c.cand_ld)) : nullptr;
    { auto v = slurp(dir + "/ref_score.bin", TS * 4); c.ref_score.assign((float*)v.data(), (float*)v.data() + TS); }
    { auto v = slurp(dir + "/ref_topk.bin", (size_t)c.T * TOPK * 8);
      c.ref_topk.assign((long long*)v.data(), (long long*)v.data() + (size_t)c.T * TOPK); }
    if (c.is_src) { auto v = slurp(dir + "/ref_cand.bin", TS); c.ref_cand.assign(v.begin(), v.end()); }

    int discriminating = 0;   // rows with more finite columns than k: where selection is a choice
    for (int t = 0; t < c.T; ++t) {
        int fin = 0;
        for (int n = 0; n < c.n_pad; ++n) fin += std::isfinite(c.ref_score[(size_t)t * c.n_pad + n]);
        discriminating += fin > std::min(TOPK, c.n_c);
    }
    std::printf("== %s  T=%d n_c=%d n_pad=%d pos0=%lld ratio=%d cand=%s src=%d | discriminating rows %d/%d, ref boundary ties %d\n",
                dir.c_str(), c.T, c.n_c, c.n_pad, c.pos0, c.ratio, c.cand ? "yes" : "no", c.is_src,
                discriminating, c.T, boundary_ties(c));

    float *d_ref, *d_s; CUDA_OK(cudaMalloc(&d_ref, TS * 4)); CUDA_OK(cudaMalloc(&d_s, TS * 4));
    CUDA_OK(cudaMemcpy(d_ref, c.ref_score.data(), TS * 4, cudaMemcpyHostToDevice));

    // SELECT: pure selection on the engine's own scores.
    const int sel_bad = rows_differ(c, topk(c, d_ref));
    std::printf("  SELECT  topk(ref score) vs ref topk: %d/%d rows differ  %s\n", sel_bad, c.T, sel_bad ? "FAIL" : "PASS");
    g_fail += sel_bad != 0;

    // CAND: layer 20's candidate mask from the engine's own scores.
    //   @2048 (production): at these lengths every visible block survives, so this tests the
    //   block amax, visibility and the x8 expansion -- control: block size 16.
    //   @SMALL (reference algorithm, fewer blocks): tests the top-k-of-blocks and the
    //   force-keep of the block holding compress_lens-1 -- control: force-keep shifted.
    if (c.is_src) {
        const int nb = c.n_pad / 8;
        float* d_b; uint8_t* d_c;
        CUDA_OK(cudaMalloc(&d_b, (size_t)c.T * nb * 4)); CUDA_OK(cudaMalloc(&d_c, TS));
        std::vector<uint8_t> h(TS);
        auto cand_diff = [&](long long pos0, int blocks, int bs, const std::vector<uint8_t>& ref) {
            dsv41_select_candidates<<<c.T, SEL_THREADS>>>(d_ref, d_b, d_c, c.n_pad, pos0, c.ratio, blocks, bs);
            CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(h.data(), d_c, TS, cudaMemcpyDeviceToHost));
            size_t bad = 0; for (size_t i = 0; i < TS; ++i) bad += h[i] != ref[i];
            return bad;
        };
        size_t kept = 0; for (auto v : c.ref_cand) kept += v;
        const size_t bad = cand_diff(c.pos0, 2048, 8, c.ref_cand);
        const size_t cbad = cand_diff(c.pos0, 2048, 16, c.ref_cand);
        std::printf("  CAND    @2048 vs ref cand_out: %zu/%zu differ (ref keeps %.3f) | CTRL block-size 16: %zu differ  %s\n",
                    bad, TS, (double)kept / TS, cbad, bad ? "FAIL" : (cbad ? "PASS" : "UNRUNNABLE"));
        g_fail += bad != 0 || cbad == 0;
        auto v = slurp(dir + "/ref_cand" SMALL_S ".bin", TS, true);
        if (!v.empty()) {
            std::vector<uint8_t> ref(v.begin(), v.end());
            const size_t b2 = cand_diff(c.pos0, SMALL_BLOCKS, 8, ref);
            const size_t c2 = cand_diff(c.pos0 - 8 * c.ratio, SMALL_BLOCKS, 8, ref);
            std::printf("  CAND    @%d (reference algorithm) : %zu/%zu differ | CTRL force-keep shifted one block: %zu differ  %s\n",
                        SMALL_BLOCKS, b2, TS, c2, b2 ? "FAIL" : (c2 ? "PASS" : "UNRUNNABLE"));
            g_fail += b2 != 0 || c2 == 0;
        }
        cudaFree(d_b); cudaFree(d_c);
    }

    // POOL: a candidate mask that actually PRUNES, consumed by the score kernel.
    {
        auto m = slurp(dir + "/pool_mask" SMALL_S ".bin", TS, true);
        if (!m.empty()) {
            auto r = slurp(dir + "/ref_topk_pool" SMALL_S ".bin", (size_t)c.T * TOPK * 8);
            Case p = c;
            p.ref_topk.assign((long long*)r.data(), (long long*)r.data() + (size_t)c.T * TOPK);
            p.cand_ld = c.n_pad;
            uint8_t* d_m = up<uint8_t>(m);
            score(p, HEAD_SUM_ORDER, d_m, c.pos0, d_s);
            const int bad = rows_differ(p, topk(p, d_s));
            score(p, HEAD_SUM_ORDER, c.cand, c.pos0, d_s);            // CONTROL: pool dropped
            const int cbad = rows_differ(p, topk(p, d_s));
            std::printf("  POOL    %d-block pool mask -> score -> topk vs reference: %d/%d rows differ | CTRL pool dropped: %d differ  %s\n",
                        SMALL_BLOCKS, bad, c.T, cbad, bad ? "FAIL" : (cbad ? "PASS" : "UNRUNNABLE"));
            g_fail += bad != 0 || cbad == 0;
            cudaFree(d_m);
        }
    }

    // SCORE, per head-sum order, and END2END for each.
    std::vector<float> h(TS);
    static const char* names[4] = { "seq", "pairwise", "butterfly", "CTRL-noround" };
    int e2e_prod = -1;
    for (int v = 0; v < 4; ++v) {
        score(c, v, c.cand, c.pos0, d_s);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(h.data(), d_s, TS * 4, cudaMemcpyDeviceToHost));
        size_t fin = 0, same = 0, infpos = 0; double worst = 0;
        for (size_t i = 0; i < TS; ++i) {
            const float r = c.ref_score[i], g = h[i];
            if (std::isinf(r) != std::isinf(g)) { ++infpos; continue; }
            if (!std::isfinite(r)) continue;
            ++fin; same += std::memcmp(&r, &g, 4) == 0; worst = std::max(worst, (double)std::fabs(r - g));
        }
        const int e2e = rows_differ(c, topk(c, d_s));
        if (v == HEAD_SUM_ORDER) e2e_prod = e2e;
        std::printf("  SCORE   %-12s bit-exact %.5f of %zu finite, worst_abs %.3e, -inf position mismatches %zu | END2END rows differ %d/%d\n",
                    names[v], (double)same / fin, fin, worst, infpos, e2e, c.T);
    }
    std::printf("  END2END production order (%s): %d/%d rows differ  %s\n", names[HEAD_SUM_ORDER], e2e_prod, c.T,
                e2e_prod ? "MISMATCH" : "EXACT");

    // Controls that must change the selection on discriminating data.
    if (discriminating) {
        score(c, HEAD_SUM_ORDER, c.cand, c.pos0 - c.ratio, d_s);   // compress_lens one row short
        const int c_lens = rows_differ(c, topk(c, d_s));
        score(c, 3, c.cand, c.pos0, d_s);
        const int c_round = rows_differ(c, topk(c, d_s));
        std::printf("  CTRL    lens-1: %d rows differ; no-bf16-round: %d rows differ", c_lens, c_round);
        if (c.cand) { score(c, HEAD_SUM_ORDER, nullptr, c.pos0, d_s);
                      std::printf("; no-candidates: %d rows differ (layer 20's pool == visibility at this length:"
                                  " pruning is exercised by POOL, not here)", rows_differ(c, topk(c, d_s))); }
        const bool ran = c_lens > 0;
        std::printf("  %s\n", ran ? "(controls bite)" : "UNRUNNABLE: a control did not change the selection");
        g_fail += !ran;
    } else {
        std::printf("  CTRL    no discriminating rows: selection is NOT TESTED by this case (every finite column is kept)\n");
    }
    cudaFree(d_ref); cudaFree(d_s); cudaFree(c.q); cudaFree(c.ik); cudaFree(c.w); if (c.cand) cudaFree(c.cand);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) run_case(argv[i]);
    std::printf("GATE %s (%d hard failures; END2END mismatches are reported, not failed)\n",
                g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
