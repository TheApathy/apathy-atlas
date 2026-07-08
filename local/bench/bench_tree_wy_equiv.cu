// TASK #40 — bit-equivalence proof: gated_delta_rule_tree_wy vs wy17 chain.
//
// Loads TWO PTX modules (wy17 reference + tree_wy under test) and asserts,
// via bad-byte counting on identical inputs (wy17_lazy_check pattern):
//
//  T1  Linear chain K=17: tree_wy(parents=[-1,0,1,..,15]) output rows,
//      inter slots 0..15, and inter[16] (tree's final state) are byte-equal
//      to wy17's output, Hi[0..15], and final h_state. tree_wy's h_state
//      must be untouched (RO contract).
//  T1b Linear chain K=32: rows 0..16 + inter 0..15 byte-equal to wy17 K=17
//      (a WY row depends only on its token prefix, so the T=32 chain's
//      first 17 rows must equal the T=17 window's rows bit-for-bit).
//  T2  Fork LEAF (general ancestor path): spine 0..16 + fork@parent7 at
//      slot 17. Fork row must be byte-equal to row 8 of the pure chain
//      [tok0..tok7, fork] (T=9) — general path == fast path == wy17 order.
//  T3  Branch TAIL (the fast-path predicate bug): spine + fork@7 + 3-token
//      tail at contiguous slots. Tail rows have parent == t-1 but ancestors
//      != [0..t-1]; pre-fix they routed through the linear fast path and
//      summed cross terms over NON-ancestors (algebraic corruption). Rows
//      17..20 must be byte-equal to rows 8..11 of the T=12 chain.
//  T4  Realistic K=32 free-slots payload (3 branches, tails, cliffs 4/9/13):
//      every spine row byte-equal to wy17; every branch row byte-equal to
//      its equivalent chain window.
//  T5  WY-form vs direct-form floor (characterization, NOT a gate): rows of
//      wy17 (WY-corrected at depth t) vs a sequential T=1 direct recompute
//      from materialized states. These are mathematically equal but
//      differently rounded — this is the irreducible residual the
//      cross-step BRANCH_AUDIT oracle measures, and it affects the FLAT
//      wy17 path identically. Reported as max-ULP / element-diff stats.
//
// Usage: bench_tree_wy_equiv <wy17.ptx> <tree_wy.ptx>
// Build: nvcc -o /tmp/bench_tree_wy_equiv bench_tree_wy_equiv.cu -lcuda
// PTX (exact engine flags): nvcc --ptx -arch=sm_121f -O3 --use_fast_math
//   -Xptxas -O3 --fmad=false <kernel.cu> -I kernels/gb10/common
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <unistd.h>

#define CK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("CUDA err %d: %s @%d\n",r,s,__LINE__);exit(1);} } while(0)

static unsigned short f_to_bf16(float f){ unsigned u; memcpy(&u,&f,4); return (unsigned short)(u>>16); }
static float bf16_to_f(unsigned short h){ unsigned u = ((unsigned)h)<<16; float f; memcpy(&f,&u,4); return f; }

// Production shapes (qwen3.6-27b GDN): nk=16 nv=48 kd=vd=128.
static const unsigned NK=16, NV=48, KD=128, VD=128;
static const unsigned CONV = 2*NK*KD + NV*VD;   // qk/v row stride (deinterleaved buffer)
static const unsigned GB   = NV*2;              // gate/beta row stride
static const size_t   HV   = (size_t)KD*VD;
static const size_t   HFLOATS = (size_t)NV*HV;
static const size_t   HBYTES  = HFLOATS*4;
static const int      NTOK = 48;                // master token pool

static int g_failures = 0;

struct Pool {
    std::vector<unsigned short> q, k, v;  // [NTOK][CONV]
    std::vector<float> g, b;              // [NTOK][GB]
    std::vector<float> h0;                // root state
};

static Pool make_pool() {
    Pool p;
    unsigned seed=987654321u;
    auto rnd=[&](){ seed=seed*1664525u+1013904223u; return seed>>16; };
    p.q.resize((size_t)NTOK*CONV); p.k.resize((size_t)NTOK*CONV); p.v.resize((size_t)NTOK*CONV);
    for (auto& x:p.q) x=f_to_bf16(((int)(rnd()&255)-128)/256.0f);
    for (auto& x:p.k) x=f_to_bf16(((int)(rnd()&255)-128)/256.0f);
    for (auto& x:p.v) x=f_to_bf16(((int)(rnd()&255)-128)/256.0f);
    p.g.resize((size_t)NTOK*GB); p.b.resize((size_t)NTOK*GB);
    for (auto& x:p.g) x=(rnd()&1023)/1024.0f;          // gate in [0,1)
    for (auto& x:p.b) x=0.5f + (rnd()&1023)/2048.0f;   // beta in (0.5,1)
    p.h0.resize(HFLOATS);
    for (auto& x:p.h0) x=((int)(rnd()&255)-128)/512.0f;
    return p;
}

static CUdeviceptr alloc_retry(size_t bytes){
    CUdeviceptr d; CUresult r;
    for(int i=0;i<600;i++){ r=cuMemAlloc(&d,bytes); if(r==CUDA_SUCCESS) return d;
        if(r!=CUDA_ERROR_OUT_OF_MEMORY){const char*s;cuGetErrorString(r,&s);printf("alloc err %s\n",s);exit(1);}
        usleep(200000); }
    printf("alloc OOM after retries (%.1fMB)\n",bytes/1e6); exit(2);
}

// One verify window: token selection sel[T] (indices into the pool) + parents.
struct Window {
    std::vector<int> sel;
    std::vector<int> parents;
    // Device buffers (rebuilt per run)
    CUdeviceptr Q=0,K=0,V=0,G=0,B=0,P=0,H=0,O=0,HI=0;
    // Host results
    std::vector<unsigned char> out, hi, h_after;
    size_t T() const { return sel.size(); }
};

static void upload_window(Window& w, const Pool& p){
    size_t T = w.T();
    std::vector<unsigned short> hq(T*CONV), hk(T*CONV), hv(T*CONV);
    std::vector<float> hg(T*GB), hb(T*GB);
    for (size_t t=0;t<T;t++){
        int s = w.sel[t];
        memcpy(&hq[t*CONV], &p.q[(size_t)s*CONV], CONV*2);
        memcpy(&hk[t*CONV], &p.k[(size_t)s*CONV], CONV*2);
        memcpy(&hv[t*CONV], &p.v[(size_t)s*CONV], CONV*2);
        memcpy(&hg[t*GB],   &p.g[(size_t)s*GB],   GB*4);
        memcpy(&hb[t*GB],   &p.b[(size_t)s*GB],   GB*4);
    }
    auto up=[&](const void* h, size_t bytes){ CUdeviceptr d=alloc_retry(bytes); CK(cuMemcpyHtoD(d,h,bytes)); return d; };
    w.Q=up(hq.data(),hq.size()*2); w.K=up(hk.data(),hk.size()*2); w.V=up(hv.data(),hv.size()*2);
    w.G=up(hg.data(),hg.size()*4); w.B=up(hb.data(),hb.size()*4);
    w.P=up(w.parents.data(), w.parents.size()*4);
    w.H=alloc_retry(HBYTES);
    w.O=alloc_retry(T*NV*VD*2);
    w.HI=alloc_retry(T*HBYTES);
}
static void free_window(Window& w){
    for (CUdeviceptr d : {w.Q,w.K,w.V,w.G,w.B,w.P,w.H,w.O,w.HI}) if (d) cuMemFree(d);
    w.Q=w.K=w.V=w.G=w.B=w.P=w.H=w.O=w.HI=0;
}

static void d2h(std::vector<unsigned char>& v, CUdeviceptr d, size_t bytes){
    v.resize(bytes); CK(cuMemcpyDtoH(v.data(), d, bytes));
}

// Run tree_wy on a window (h_state reset to pool root; inter poisoned 0xEE).
static void run_tree(CUfunction f, Window& w, const Pool& p){
    unsigned T=(unsigned)w.T(), batch=1, nk=NK, nv=NV, kd=KD, vd=VD, conv=CONV, gb=GB;
    unsigned isf=(unsigned)HFLOATS;
    CK(cuMemcpyHtoD(w.H, p.h0.data(), HBYTES));
    CK(cuMemsetD8(w.HI, 0xEE, w.T()*HBYTES));
    CK(cuMemsetD8(w.O, 0xEE, w.T()*NV*VD*2));
    void* a[]={&w.H,&w.Q,&w.K,&w.V,&w.G,&w.B,&w.P,&w.O,&w.HI,&isf,&T,&batch,&nk,&nv,&kd,&vd,&conv,&conv,&gb};
    CK(cuLaunchKernel(f, nv, batch, 1, 128,1,1, 0,0,a,0));
    CK(cuCtxSynchronize());
    d2h(w.out, w.O, w.T()*NV*VD*2);
    d2h(w.hi,  w.HI, w.T()*HBYTES);
    d2h(w.h_after, w.H, HBYTES);
}

// Run wy17 (K must be 17). Writes Hi[0..15] + final h_state.
static void run_wy17(CUfunction f, Window& w, const Pool& p){
    if (w.T()!=17){ printf("wy17 needs T=17\n"); exit(1); }
    unsigned batch=1, nk=NK, nv=NV, kd=KD, vd=VD, conv=CONV, gb=GB;
    unsigned isf=(unsigned)HFLOATS;
    CK(cuMemcpyHtoD(w.H, p.h0.data(), HBYTES));
    CK(cuMemsetD8(w.HI, 0xEE, w.T()*HBYTES));
    CK(cuMemsetD8(w.O, 0xEE, w.T()*NV*VD*2));
    void* a[]={&w.H,&w.Q,&w.K,&w.V,&w.G,&w.B,&w.O,&w.HI,&isf,&batch,&nk,&nv,&kd,&vd,&conv,&conv,&gb};
    CK(cuLaunchKernel(f, nv, batch, 1, 128,1,1, 0,0,a,0));
    CK(cuCtxSynchronize());
    d2h(w.out, w.O, w.T()*NV*VD*2);
    d2h(w.hi,  w.HI, w.T()*HBYTES);
    d2h(w.h_after, w.H, HBYTES);
}

static size_t diff_bytes(const unsigned char* a, const unsigned char* b, size_t n){
    size_t d=0; for (size_t i=0;i<n;i++) d += (a[i]!=b[i]); return d;
}
// Compare one output ROW (nv*vd bf16) + one inter slot between two windows.
static size_t cmp_row(const Window& a, size_t ra, const Window& b, size_t rb, size_t* hi_bad){
    size_t row_bytes = (size_t)NV*VD*2;
    size_t d = diff_bytes(&a.out[ra*row_bytes], &b.out[rb*row_bytes], row_bytes);
    *hi_bad = diff_bytes(&a.hi[ra*HBYTES], &b.hi[rb*HBYTES], HBYTES);
    return d;
}
static void report(const char* name, size_t out_bad, size_t hi_bad){
    bool ok = (out_bad==0 && hi_bad==0);
    if (!ok) g_failures++;
    printf("%-52s out-bad=%zu  state-bad=%zu  %s\n", name, out_bad, hi_bad,
           ok ? "BIT-IDENTICAL ✓" : "*** MISMATCH ***");
}

// Build a linear chain window over pool tokens sel[0..T).
static Window chain_window(const std::vector<int>& sel){
    Window w; w.sel = sel;
    w.parents.resize(sel.size());
    for (size_t t=0;t<sel.size();t++) w.parents[t] = (int)t-1;
    return w;
}

int main(int argc, char** argv){
    if (argc < 3){ printf("usage: %s <wy17.ptx> <tree_wy.ptx>\n", argv[0]); return 1; }
    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev,0));
    CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
    CUmodule mW, mT; CUresult r;
    for(int i=0;i<600;i++){ r=cuModuleLoad(&mW,argv[1]); if(r==CUDA_SUCCESS)break;
        if(r!=CUDA_ERROR_OUT_OF_MEMORY){const char*s;cuGetErrorString(r,&s);printf("wy17 module: %s\n",s);return 1;} usleep(200000);}
    for(int i=0;i<600;i++){ r=cuModuleLoad(&mT,argv[2]); if(r==CUDA_SUCCESS)break;
        if(r!=CUDA_ERROR_OUT_OF_MEMORY){const char*s;cuGetErrorString(r,&s);printf("tree module: %s\n",s);return 1;} usleep(200000);}
    CUfunction fW, fT;
    CK(cuModuleGetFunction(&fW, mW, "gated_delta_rule_wy17"));
    CK(cuModuleGetFunction(&fT, mT, "gated_delta_rule_tree_wy"));

    Pool p = make_pool();

    // Reference: wy17 on tokens 0..16.
    std::vector<int> spine17; for (int i=0;i<17;i++) spine17.push_back(i);
    Window ref = chain_window(spine17);
    upload_window(ref, p);
    run_wy17(fW, ref, p);

    // ── T1: tree_wy linear chain K=17 vs wy17 ──
    {
        Window w = chain_window(spine17);
        upload_window(w, p);
        run_tree(fT, w, p);
        size_t out_bad = diff_bytes(w.out.data(), ref.out.data(), w.out.size());
        // wy17: Hi[0..15] then final state in h_state. tree: inter[0..16].
        size_t hi_bad = diff_bytes(w.hi.data(), ref.hi.data(), 16*HBYTES);
        size_t fin_bad = diff_bytes(&w.hi[16*HBYTES], ref.h_after.data(), HBYTES);
        report("T1  chain K=17 out",            out_bad, hi_bad);
        report("T1  chain K=17 final state",    0, fin_bad);
        // RO contract: tree_wy must not touch h_state.
        std::vector<unsigned char> h0b(HBYTES); memcpy(h0b.data(), p.h0.data(), HBYTES);
        report("T1  chain K=17 h_state RO",     0, diff_bytes(w.h_after.data(), h0b.data(), HBYTES));
        free_window(w);
    }

    // ── T1b: tree_wy linear chain K=32, rows 0..16 vs wy17 ──
    {
        std::vector<int> sel; for (int i=0;i<32;i++) sel.push_back(i%NTOK);
        Window w = chain_window(sel);
        upload_window(w, p);
        run_tree(fT, w, p);
        size_t row_bytes=(size_t)NV*VD*2;
        size_t out_bad = diff_bytes(w.out.data(), ref.out.data(), 17*row_bytes);
        size_t hi_bad  = diff_bytes(w.hi.data(),  ref.hi.data(),  16*HBYTES);
        size_t fin_bad = diff_bytes(&w.hi[16*HBYTES], ref.h_after.data(), HBYTES);
        report("T1b chain K=32 rows0-16 vs wy17",  out_bad, hi_bad+fin_bad);
        free_window(w);
    }

    // Cross-window helper: window A branch rows vs equivalent chain window.
    auto cmp_branch=[&](const char* name, Window& A, const std::vector<size_t>& rowsA,
                        const std::vector<int>& chain_sel, size_t first_ref_row){
        Window B = chain_window(chain_sel);
        upload_window(B, p);
        run_tree(fT, B, p);
        size_t out_bad=0, hi_bad=0;
        for (size_t i=0;i<rowsA.size();i++){
            size_t hb=0;
            out_bad += cmp_row(A, rowsA[i], B, first_ref_row+i, &hb);
            hi_bad  += hb;
        }
        report(name, out_bad, hi_bad);
        free_window(B);
    };

    // ── T2: fork LEAF at slot 17, parent 7 (general ancestor path) ──
    {
        std::vector<int> sel = spine17; sel.push_back(20);
        Window A = chain_window(sel);
        A.parents[17] = 7;
        upload_window(A, p);
        run_tree(fT, A, p);
        // spine rows must stay byte-identical to wy17 despite the extra fork
        size_t row_bytes=(size_t)NV*VD*2;
        size_t sp_out = diff_bytes(A.out.data(), ref.out.data(), 17*row_bytes);
        size_t sp_hi  = diff_bytes(A.hi.data(),  ref.hi.data(),  16*HBYTES);
        report("T2  spine rows with fork present",  sp_out, sp_hi);
        std::vector<int> csel; for (int i=0;i<=7;i++) csel.push_back(i); csel.push_back(20);
        cmp_branch("T2  fork LEAF row vs chain (general path)", A, {17}, csel, 8);
        free_window(A);
    }

    // ── T3: fork + contiguous 3-token TAIL (fast-path predicate bug) ──
    {
        std::vector<int> sel = spine17;
        sel.push_back(20); sel.push_back(21); sel.push_back(22); sel.push_back(23);
        Window A = chain_window(sel);           // tail parents 18,19,20 == t-1 (contiguous!)
        A.parents[17] = 7;                      // fork
        upload_window(A, p);
        run_tree(fT, A, p);
        std::vector<int> csel; for (int i=0;i<=7;i++) csel.push_back(i);
        csel.push_back(20); csel.push_back(21); csel.push_back(22); csel.push_back(23);
        cmp_branch("T3  fork+TAIL rows vs chain (tail bug)", A, {17,18,19,20}, csel, 8);
        free_window(A);
    }

    // ── T4: realistic K=32 free-slots payload — 3 branches, cliffs 4/9/13 ──
    {
        std::vector<int> sel = spine17;
        struct Br { int cliff; std::vector<int> toks; size_t slot0; };
        std::vector<Br> brs = {
            {4,  {20,21,22,23,24}, 0},   // fork+4 tail
            {9,  {25,26,27,28,29}, 0},
            {13, {30,31,32,33,34}, 0},
        };
        Window A = chain_window(sel);   // parents rebuilt below
        for (auto& b : brs){
            b.slot0 = A.sel.size();
            for (size_t i=0;i<b.toks.size();i++){
                A.sel.push_back(b.toks[i]);
                A.parents.push_back(i==0 ? b.cliff-1 : (int)A.sel.size()-2);
            }
        }
        if (A.T()!=32){ printf("T4 layout error T=%zu\n", A.T()); return 1; }
        upload_window(A, p);
        run_tree(fT, A, p);
        size_t row_bytes=(size_t)NV*VD*2;
        size_t sp_out = diff_bytes(A.out.data(), ref.out.data(), 17*row_bytes);
        size_t sp_hi  = diff_bytes(A.hi.data(),  ref.hi.data(),  16*HBYTES);
        report("T4  K=32 spine rows vs wy17", sp_out, sp_hi);
        for (auto& b : brs){
            std::vector<int> csel; for (int i=0;i<b.cliff;i++) csel.push_back(i);
            for (int t : b.toks) csel.push_back(t);
            std::vector<size_t> rows; for (size_t i=0;i<b.toks.size();i++) rows.push_back(b.slot0+i);
            char nm[80]; snprintf(nm,80,"T4  K=32 branch@cliff%d rows vs chain", b.cliff);
            cmp_branch(nm, A, rows, csel, (size_t)b.cliff);
        }
        free_window(A);
    }

    // ── T6: portfolio 2-root forest (ATLAS_DFLASH_PORTFOLIO topology) ──
    // Root-B chain laid contiguously at slots 8..15 behind a re-root
    // (parent=-1) at slot 8. Pre-fix, slots 9..15 hit the linear fast path
    // and summed cross terms over chain A — the "portfolio deep-B" blocker.
    // Every chain-B row must equal the pure chain window [tok20..tok27].
    {
        std::vector<int> sel; for (int i=0;i<8;i++) sel.push_back(i);      // chain A
        for (int i=0;i<8;i++) sel.push_back(20+i);                          // chain B
        Window A = chain_window(sel);
        A.parents[8] = -1;                                                  // root B
        upload_window(A, p);
        run_tree(fT, A, p);
        std::vector<int> csel; for (int i=0;i<8;i++) csel.push_back(20+i);
        cmp_branch("T6  portfolio 2-root chain-B rows vs chain", A, {8,9,10,11,12,13,14,15}, csel, 0);
        free_window(A);
    }

    // ── T5: WY-form vs direct-form floor (characterization only) ──
    // Sequential T=1 windows == the direct per-token form (corrected = H·k).
    // wy17's row t uses the WY-corrected form. Mathematically equal,
    // differently rounded — the irreducible cross-step audit residual.
    {
        Window s1 = chain_window({0});
        upload_window(s1, p);
        std::vector<unsigned char> hcur(HBYTES);
        memcpy(hcur.data(), p.h0.data(), HBYTES);
        size_t row_bytes=(size_t)NV*VD*2;
        size_t tot=0, diff_el=0; int max_ulp=0; double max_abs=0;
        for (int t=0;t<17;t++){
            // rebuild inputs for token t
            free_window(s1);
            s1 = chain_window({t});
            upload_window(s1, p);
            CK(cuMemcpyHtoD(s1.H, hcur.data(), HBYTES));
            unsigned T=1, batch=1, nk=NK, nv=NV, kd=KD, vd=VD, conv=CONV, gb=GB, isf=(unsigned)HFLOATS;
            CK(cuMemsetD8(s1.HI, 0xEE, HBYTES));
            void* a[]={&s1.H,&s1.Q,&s1.K,&s1.V,&s1.G,&s1.B,&s1.P,&s1.O,&s1.HI,&isf,&T,&batch,&nk,&nv,&kd,&vd,&conv,&conv,&gb};
            CK(cuLaunchKernel(fT, nv, batch, 1, 128,1,1, 0,0,a,0));
            CK(cuCtxSynchronize());
            d2h(s1.out, s1.O, row_bytes);
            d2h(s1.hi, s1.HI, HBYTES);
            memcpy(hcur.data(), s1.hi.data(), HBYTES);   // feed state forward
            const unsigned short* d1=(const unsigned short*)s1.out.data();
            const unsigned short* d2=(const unsigned short*)&ref.out[(size_t)t*row_bytes];
            for (size_t i=0;i<(size_t)NV*VD;i++){
                tot++;
                if (d1[i]!=d2[i]){
                    diff_el++;
                    int u = abs((int)d1[i]-(int)d2[i]); if (u>max_ulp) max_ulp=u;
                    double ad = fabs((double)bf16_to_f(d1[i])-(double)bf16_to_f(d2[i]));
                    if (ad>max_abs) max_abs=ad;
                }
            }
        }
        printf("T5  WY-form vs direct-form (NOT a gate): %zu/%zu bf16 elements differ "
               "(%.2f%%), max bf16-ULP=%d, max |abs|=%.3e\n",
               diff_el, tot, 100.0*diff_el/tot, max_ulp, max_abs);
        free_window(s1);
    }

    // ── Timing (informational): wy17 K=17 vs tree_wy T=17 / T=32 ──
    // Quantifies the SSM-kernel share of the free-slots K=32 step cost.
    {
        auto time_min=[&](CUfunction f, Window& w, bool tree)->double{
            unsigned T=(unsigned)w.T(), batch=1, nk=NK, nv=NV, kd=KD, vd=VD,
                     conv=CONV, gb=GB, isf=(unsigned)HFLOATS;
            void* at[]={&w.H,&w.Q,&w.K,&w.V,&w.G,&w.B,&w.P,&w.O,&w.HI,&isf,&T,&batch,&nk,&nv,&kd,&vd,&conv,&conv,&gb};
            void* aw[]={&w.H,&w.Q,&w.K,&w.V,&w.G,&w.B,&w.O,&w.HI,&isf,&batch,&nk,&nv,&kd,&vd,&conv,&conv,&gb};
            void** a = tree ? at : aw;
            for (int i=0;i<8;i++) CK(cuLaunchKernel(f, nv,1,1, 128,1,1, 0,0,a,0));
            CK(cuCtxSynchronize());
            CUevent e0,e1; cuEventCreate(&e0,0); cuEventCreate(&e1,0);
            double best=1e30;
            for (int i=0;i<100;i++){
                cuEventRecord(e0,0);
                CK(cuLaunchKernel(f, nv,1,1, 128,1,1, 0,0,a,0));
                cuEventRecord(e1,0);
                CK(cuEventSynchronize(e1));
                float ms=0; cuEventElapsedTime(&ms,e0,e1);
                if (ms*1000.0<best) best=ms*1000.0;
            }
            cuEventDestroy(e0); cuEventDestroy(e1);
            return best;
        };
        Window w17 = chain_window(spine17);
        upload_window(w17, p);
        CK(cuMemcpyHtoD(w17.H, p.h0.data(), HBYTES));
        std::vector<int> sel32; for (int i=0;i<32;i++) sel32.push_back(i%NTOK);
        Window w32 = chain_window(sel32);
        upload_window(w32, p);
        CK(cuMemcpyHtoD(w32.H, p.h0.data(), HBYTES));
        printf("\n--- µs/launch (min-of-100, informational) ---\n");
        printf("wy17     K=17: %7.1f us\n", time_min(fW, w17, false));
        printf("tree_wy  T=17: %7.1f us\n", time_min(fT, w17, true));
        printf("tree_wy  T=32: %7.1f us\n", time_min(fT, w32, true));
        free_window(w17); free_window(w32);
    }

    free_window(ref);
    printf(g_failures==0 ? "ALL EQUIVALENCE GATES PASS\n"
                         : "*** %d EQUIVALENCE GATE(S) FAILED ***\n", g_failures);
    return g_failures==0 ? 0 : 3;
}
