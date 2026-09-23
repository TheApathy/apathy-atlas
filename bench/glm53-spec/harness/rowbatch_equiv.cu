// Bit-equivalence harness: glm53_exl3_rowbatch_k4_s* (all rows in one pass) vs
// glm53_exl3_rowexact_k4_s* (the shipped per-row loop of the pinned M=1 kernel)
// on identical random inputs, rows 1..8, one shape per tile configuration.
// Also times both. Build: see run.sh.
#include "../../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_k4_cb2.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(2); } } while (0)

typedef void (*kfn)(EXL3_GEMM_ARGS);

struct Shape { int idx, tk, tn, block; kfn exact, batch; };

static float launch(kfn f, int grid, int block, const half* A, const uint16_t* B, half* C, int m, int k, int n,
                    int* locks, const half* suh, half* A_had, const half* svh, int reps) {
    void* args[] = {(void*)&A, (void*)&B, (void*)&C, (void*)&m, (void*)&k, (void*)&n, (void*)&locks,
                    (void*)&suh, (void*)&A_had, (void*)&svh};
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r)
        CK(cudaLaunchCooperativeKernel((void*)f, dim3(grid), dim3(block), args, 90 * 1024, 0));
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / reps;
}

int main() {
    Shape shapes[] = {
        {1, 16, 128, 256, glm53_exl3_rowexact_k4_s1, glm53_exl3_rowbatch_k4_s1},
        {2, 32, 128, 512, glm53_exl3_rowexact_k4_s2, glm53_exl3_rowbatch_k4_s2},
        {3, 32, 256, 512, glm53_exl3_rowexact_k4_s3, glm53_exl3_rowbatch_k4_s3},
        {4, 16, 512, 256, glm53_exl3_rowexact_k4_s4, glm53_exl3_rowbatch_k4_s4},
    };
    int dims[][2] = {{2048, 4096}, {4096, 4096}, {16384, 4096}, {4096, 24576}};
    int failures = 0;
    for (int s = 0; s < 4; ++s) {
        Shape sh = shapes[s];
        int K = dims[s][0], N = dims[s][1];
        CK(cudaFuncSetAttribute((void*)sh.exact, cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024));
        CK(cudaFuncSetAttribute((void*)sh.batch, cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024));
        size_t tiles = (size_t)(K / sh.tk) * (N / sh.tn);
        int grid = (int)(tiles < 48 ? tiles : 48);
        size_t bwords = (size_t)K * N * 4 / 16;
        std::vector<uint16_t> hb(bwords);
        srand(1234 + s);
        for (auto& w : hb) w = (uint16_t)(rand() & 0xffff);
        std::vector<half> ha(8 * (size_t)K), hsuh(K), hsvh(N);
        for (auto& v : ha) v = __float2half(((rand() % 2001) - 1000) / 1000.0f);
        for (auto& v : hsuh) v = __float2half((rand() & 1) ? 1.0f : -1.0f);
        for (auto& v : hsvh) v = __float2half(((rand() % 200) + 900) / 1000.0f * ((rand() & 1) ? 1 : -1));
        uint16_t* B; half *A, *Ahad, *C0, *C1, *suh, *svh; int* locks;
        CK(cudaMalloc(&B, bwords * 2)); CK(cudaMalloc(&A, 8 * (size_t)K * 2)); CK(cudaMalloc(&Ahad, 8 * (size_t)K * 2));
        CK(cudaMalloc(&C0, 8 * (size_t)N * 2)); CK(cudaMalloc(&C1, 8 * (size_t)N * 2));
        CK(cudaMalloc(&suh, K * 2)); CK(cudaMalloc(&svh, N * 2)); CK(cudaMalloc(&locks, 1 << 20));
        CK(cudaMemset(locks, 0, 1 << 20));
        CK(cudaMemcpy(B, hb.data(), bwords * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(A, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(suh, hsuh.data(), K * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(svh, hsvh.data(), N * 2, cudaMemcpyHostToDevice));
        for (int m = 1; m <= 8; ++m) {
            CK(cudaMemset(C0, 0xff, 8 * (size_t)N * 2)); CK(cudaMemset(C1, 0x7f, 8 * (size_t)N * 2));
            float t0 = launch(sh.exact, grid, sh.block, A, B, C0, m, K, N, locks, suh, Ahad, svh, 1);
            float t1 = launch(sh.batch, grid, sh.block, A, B, C1, m, K, N, locks, suh, Ahad, svh, 1);
            std::vector<uint16_t> r0((size_t)m * N), r1((size_t)m * N);
            CK(cudaMemcpy(r0.data(), C0, r0.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(r1.data(), C1, r1.size() * 2, cudaMemcpyDeviceToHost));
            size_t diff = 0, nonfinite = 0;
            for (size_t i = 0; i < r0.size(); ++i) {
                if (r0[i] != r1[i]) ++diff;
                if ((r0[i] & 0x7c00) == 0x7c00) ++nonfinite;
            }
            // Also each row of the batch must equal the M=1 kernel on that row alone.
            size_t diff1 = 0;
            for (int row = 0; row < m; ++row) {
                launch(sh.exact, grid, sh.block, A + (size_t)row * K, B, C0, 1, K, N, locks, suh, Ahad, svh, 1);
                std::vector<uint16_t> one(N);
                CK(cudaMemcpy(one.data(), C0, N * 2, cudaMemcpyDeviceToHost));
                for (int i = 0; i < N; ++i) if (one[i] != r1[(size_t)row * N + i]) ++diff1;
            }
            t0 = launch(sh.exact, grid, sh.block, A, B, C0, m, K, N, locks, suh, Ahad, svh, 20);
            t1 = launch(sh.batch, grid, sh.block, A, B, C1, m, K, N, locks, suh, Ahad, svh, 20);
            printf("shape s%d K=%5d N=%5d rows=%d: batch-vs-rowexact diff=%zu batch-vs-M1 diff=%zu nonfinite=%zu  "
                   "rowexact %.1f us  rowbatch %.1f us  (%.2fx)\n",
                   sh.idx, K, N, m, diff, diff1, nonfinite, t0 * 1000, t1 * 1000, t0 / t1);
            if (diff || diff1) ++failures;
            if (m == 2) {
                // Negative control: perturb row 1's input; row 1's output must change,
                // row 0's must not. A comparison that cannot fail proves nothing.
                half orig = ha[K + 7];
                half bumped = __float2half(__half2float(orig) + 0.5f);
                CK(cudaMemcpy(A + K + 7, &bumped, 2, cudaMemcpyHostToDevice));
                launch(sh.batch, grid, sh.block, A, B, C0, 2, K, N, locks, suh, Ahad, svh, 1);
                std::vector<uint16_t> rp((size_t)2 * N);
                CK(cudaMemcpy(rp.data(), C0, rp.size() * 2, cudaMemcpyDeviceToHost));
                size_t d0 = 0, d1 = 0;
                for (int i = 0; i < N; ++i) { d0 += rp[i] != r1[i]; d1 += rp[N + i] != r1[N + i]; }
                printf("  control s%d: perturbed row1 -> row0 changed %zu, row1 changed %zu (%s)\n",
                       sh.idx, d0, d1, (d0 == 0 && d1 > 0) ? "OK, gate can fail" : "CONTROL FAILED");
                if (!(d0 == 0 && d1 > 0)) ++failures;
                CK(cudaMemcpy(A + K + 7, &orig, 2, cudaMemcpyHostToDevice));
            }
        }
        cudaFree(B); cudaFree(A); cudaFree(Ahad); cudaFree(C0); cudaFree(C1); cudaFree(suh); cudaFree(svh); cudaFree(locks);
    }
    // Negative control: a harness that cannot fail proves nothing. Perturb one input
    // element of row 1 and require that row's output to change.
    printf(failures ? "RESULT: DIFFER (%d cases)\n" : "RESULT: BIT-IDENTICAL (%d failing cases)\n", failures);
    return failures ? 1 : 0;
}
