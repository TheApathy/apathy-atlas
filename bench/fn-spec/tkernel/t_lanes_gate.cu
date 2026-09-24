// Standalone gate for the lane-parallel transposed MoE decode kernels.
// Builds random experts in the [N, K/2] layout and their transposed copies,
// runs the untransposed decode kernel (reference), the old one-thread-per-
// output _t kernel, and the new _t_lanes kernel, and compares outputs byte for
// byte. A positive control flips one weight byte in the transposed copy and
// must be reported. Times each kernel with CUDA events.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

typedef const __nv_bfloat16* cbf;
typedef __nv_bfloat16* bf;
typedef const unsigned long long* cptrs;
typedef const unsigned char* cub;

// Single translation unit: the reference and transposed kernel sources are
// included directly (they only disagree on BLOCK_SIZE).
#include "../../../kernels/gb10/common/moe_shared_expert_fused.cu"
#undef BLOCK_SIZE
#include "../../../kernels/gb10/common/moe_shared_expert_fused_t.cu"

struct Mat { std::vector<unsigned char> packed, scale, packed_t, scale_t; float s2; };

static Mat make(int N, int K, std::mt19937& rng) {
    Mat m; m.s2 = 0.0173f;
    int hk = K / 2, g = K / 16;
    m.packed.resize((size_t)N * hk); m.scale.resize((size_t)N * g);
    for (auto& b : m.packed) b = rng() & 0xFF;
    // FP8 E4M3 scales in a sane range (exponent 5..9, any mantissa, positive).
    for (auto& s : m.scale) s = (unsigned char)(((5 + rng() % 5) << 3) | (rng() & 7));
    m.packed_t.resize(m.packed.size()); m.scale_t.resize(m.scale.size());
    for (int n = 0; n < N; n++) {
        for (int kh = 0; kh < hk; kh++) m.packed_t[(size_t)kh * N + n] = m.packed[(size_t)n * hk + kh];
        for (int sg = 0; sg < g; sg++) m.scale_t[(size_t)sg * N + n] = m.scale[(size_t)n * g + sg];
    }
    return m;
}

template <class T> static T* up(const std::vector<T>& v) { T* d; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }

static std::vector<unsigned short> down16(const void* d, size_t n) { std::vector<unsigned short> h(n); CK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost)); return h; }
static size_t diff(const std::vector<unsigned short>& a, const std::vector<unsigned short>& b) { size_t c = 0; for (size_t i = 0; i < a.size(); i++) c += a[i] != b[i]; return c; }

typedef void (*GuFn)(cbf, cptrs, cptrs, const float*, bf, cptrs, cptrs, const float*, bf, const unsigned int*,
                     cub, cub, float, bf, cub, cub, float, bf, unsigned int, unsigned int, unsigned int);
typedef void (*DnFn)(cbf, cbf, cptrs, cptrs, const float*, bf, const unsigned int*, cbf, cbf, cub, cub, float, bf,
                     unsigned int, unsigned int, unsigned int);
struct Variant { const char* name; GuFn gu; DnFn dn; int opt, tpl; bool transposed; };
#define V(O, T, U) {"lanes_o" #O "t" #T "u" #U, moe_expert_gate_up_shared_t_lanes_o##O##t##T##u##U, \
                    moe_expert_silu_down_shared_t_lanes_o##O##t##T##u##U, O, T, true}

int main(int argc, char** argv) {
    // FN decode MoE: hidden 2560, expert intermediate 640, top-10 + 1 shared.
    // NSETS disjoint routings are cycled when timing so the working set
    // (NSETS * 11 experts * ~2.7 MB) is far larger than L2, as in decode.
    const int NSETS = 8, TOPK = 10, E = NSETS * TOPK, H = 2560, I = 640;
    std::mt19937 rng(1234);
    // gate/up: N=I, K=H.  down: N=H, K=I.
    std::vector<Mat> gate, upm, dn;
    for (int e = 0; e < E + 1; e++) { gate.push_back(make(I, H, rng)); upm.push_back(make(I, H, rng)); dn.push_back(make(H, I, rng)); }
    std::vector<unsigned int> idx((size_t)NSETS * TOPK);
    for (int s = 0; s < NSETS; s++) for (int t = 0; t < TOPK; t++) idx[s * TOPK + t] = s * TOPK + (t * 7 + 3) % TOPK;
    std::vector<unsigned short> a(H); for (auto& x : a) { float f = ((int)(rng() % 2001) - 1000) / 400.0f; __nv_bfloat16 b = __float2bfloat16(f); memcpy(&x, &b, 2); }

    auto ptrs = [&](std::vector<Mat>& v, bool t, bool scale, int corrupt_expert) {
        std::vector<unsigned long long> p(E);
        for (int e = 0; e < E; e++) {
            auto src = scale ? (t ? v[e].scale_t : v[e].scale) : (t ? v[e].packed_t : v[e].packed);
            if (!scale && e == corrupt_expert) src[src.size() / 3] ^= 0x11;
            p[e] = (unsigned long long)up(src);
        }
        return up(p);
    };
    std::vector<float> s2g(E), s2u(E), s2d(E); for (int e = 0; e < E; e++) { s2g[e] = gate[e].s2; s2u[e] = upm[e].s2; s2d[e] = dn[e].s2; }
    float *ds2g = up(s2g), *ds2u = up(s2u), *ds2d = up(s2d);
    unsigned int* didx = up(idx);
    unsigned short* dA = up(a);
    Mat& sg = gate[E]; Mat& su = upm[E]; Mat& sd = dn[E];

    struct Layout { unsigned long long *gp, *gs, *upk, *us, *dp, *ds; unsigned char *sgp, *sgs, *sup, *sus, *sdp, *sds; };
    auto layout = [&](bool t, int corrupt) {
        Layout L;
        L.gp = ptrs(gate, t, false, corrupt); L.gs = ptrs(gate, t, true, -1);
        L.upk = ptrs(upm, t, false, -1); L.us = ptrs(upm, t, true, -1);
        L.dp = ptrs(dn, t, false, corrupt); L.ds = ptrs(dn, t, true, -1);
        L.sgp = up(t ? sg.packed_t : sg.packed); L.sgs = up(t ? sg.scale_t : sg.scale);
        L.sup = up(t ? su.packed_t : su.packed); L.sus = up(t ? su.scale_t : su.scale);
        L.sdp = up(t ? sd.packed_t : sd.packed); L.sds = up(t ? sd.scale_t : sd.scale);
        return L;
    };
    // The control corrupts one byte of a routed expert of set 0 (gate and down).
    Layout O = layout(false, -1), T = layout(true, -1), X = layout(true, (int)idx[4]);

    bf go, uo, sgo, suo, dout, sdo;
    CK(cudaMalloc(&go, TOPK * I * 2)); CK(cudaMalloc(&uo, TOPK * I * 2)); CK(cudaMalloc(&sgo, I * 2)); CK(cudaMalloc(&suo, I * 2));
    CK(cudaMalloc(&dout, TOPK * H * 2)); CK(cudaMalloc(&sdo, H * 2));
    const size_t sm = (size_t)I * 4;

    std::vector<Variant> vs = {
        {"REF [N,K/2]", moe_expert_gate_up_shared, moe_expert_silu_down_shared, 8, 128, false},
        {"OLD _t", moe_expert_gate_up_shared_t, moe_expert_silu_down_shared_t, 32, 32, true},
        V(4, 16, 1), V(4, 16, 2), V(4, 32, 1), V(2, 32, 1), V(2, 32, 2),
    };
    // REF: 8 outputs per 128-thread block; OLD _t: 32 outputs per 32-thread block.
    auto launch = [&](const Variant& v, Layout& L, const unsigned int* ix) {
        int ob = v.transposed && v.opt != 32 ? v.opt * v.tpl : v.opt, threads = v.transposed && v.opt != 32 ? 32 * v.tpl : v.tpl;
        v.gu<<<dim3((I + ob - 1) / ob, TOPK + 1, 2), threads>>>((cbf)dA, L.gp, L.gs, ds2g, go, L.upk, L.us, ds2u, uo, ix,
                                                                 L.sgp, L.sgs, sg.s2, sgo, L.sup, L.sus, su.s2, suo, I, H, TOPK);
        v.dn<<<dim3((H + ob - 1) / ob, TOPK + 1, 1), threads, sm>>>(go, uo, L.dp, L.ds, ds2d, dout, ix, sgo, suo,
                                                                     L.sdp, L.sds, sd.s2, sdo, H, I, TOPK);
    };
    auto run = [&](const Variant& v, Layout& L) {
        CK(cudaMemset(go, 0xFF, TOPK * I * 2)); CK(cudaMemset(dout, 0xFF, TOPK * H * 2));
        launch(v, L, didx);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        auto g = down16(go, TOPK * I); auto u = down16(uo, TOPK * I); auto x = down16(sgo, I); auto y = down16(suo, I);
        auto z = down16(dout, TOPK * H); auto q = down16(sdo, H);
        g.insert(g.end(), u.begin(), u.end()); g.insert(g.end(), x.begin(), x.end()); g.insert(g.end(), y.begin(), y.end());
        z.insert(z.end(), q.begin(), q.end());
        return std::make_pair(g, z);
    };
    auto ref = run(vs[0], O);
    printf("ref twice: gu=%zu dn=%zu differing\n", diff(ref.first, run(vs[0], O).first), diff(ref.second, run(vs[0], O).second));
    int fails = 0;
    for (size_t i = 1; i < vs.size(); i++) {
        auto r = run(vs[i], T), c = run(vs[i], X);
        size_t dg = diff(ref.first, r.first), dd = diff(ref.second, r.second);
        size_t cg = diff(ref.first, c.first), cd = diff(ref.second, c.second);
        bool exact = dg == 0 && dd == 0, control = cg + cd > 0;
        if (i >= 2 && !(exact && control)) fails++;
        printf("%-16s differing gu=%zu/%zu dn=%zu/%zu  control gu=%zu dn=%zu  %s\n", vs[i].name, dg, ref.first.size(), dd,
               ref.second.size(), cg, cd, i == 1 ? "(known inexact)" : exact && control ? "EXACT" : "FAIL");
    }

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int ITERS = 400;
    for (int rep = 0; rep < 2; rep++) {
        for (auto& v : vs) {
            Layout& L = v.transposed ? T : O;
            for (int w = 0; w < 16; w++) launch(v, L, didx + (w % NSETS) * TOPK);
            cudaEventRecord(e0);
            for (int it = 0; it < ITERS; it++) launch(v, L, didx + (it % NSETS) * TOPK);
            cudaEventRecord(e1); cudaEventSynchronize(e1); CK(cudaGetLastError());
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            printf("rep%d %-16s %.1f us per layer (gate_up + silu_down)\n", rep, v.name, ms * 1000 / ITERS);
        }
    }
    printf("RESULT %s\n", fails ? "FAIL" : "ALL EXACT");
    return fails ? 1 : 0;
}
