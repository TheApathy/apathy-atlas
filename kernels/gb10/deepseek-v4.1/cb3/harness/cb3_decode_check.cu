// SPDX-License-Identifier: AGPL-3.0-only
//
// Bit-exactness check for `cb3_decode.cuh` against an INDEPENDENT reference.
//
// The vectors are produced by a numpy model of the same PTX semantics written
// from the instruction listing, not by running this code — so agreement is two
// independent routes to the same bytes, and a transcription slip in either one
// shows up as a mismatch rather than as two copies of the same mistake.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "../cb3_decode.cuh"

__global__ void run(const uint32_t* L, const uint32_t* H, const uint32_t* A,
                    const uint32_t* B, const int* variant, uint32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (variant[i]) {
        case 0: out[i] = atlas_cb3_decode8<0, 0>(L[i], H[i], A[i], B[i]); break;
        case 1: out[i] = atlas_cb3_decode8<4, 2>(L[i], H[i], A[i], B[i]); break;
        case 2: out[i] = atlas_cb3_decode8<0, 4>(L[i], H[i], A[i], B[i]); break;
        default: out[i] = atlas_cb3_decode8<4, 6>(L[i], H[i], A[i], B[i]); break;
    }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "cb3_decode_vectors.txt";
    FILE* f = fopen(path, "r");
    if (!f) { printf("FAIL: cannot open %s\n", path); return 2; }
    std::vector<uint32_t> L, H, A, B, want;
    std::vector<int> var;
    int v; unsigned l, h, a, b, w;
    while (fscanf(f, "%d %x %x %x %x %x", &v, &l, &h, &a, &b, &w) == 6) {
        var.push_back(v); L.push_back(l); H.push_back(h);
        A.push_back(a); B.push_back(b); want.push_back(w);
    }
    fclose(f);
    int n = (int)L.size();
    if (n == 0) { printf("FAIL: no vectors read — a check over an empty set passes vacuously\n"); return 2; }

    uint32_t *dL, *dH, *dA, *dB, *dO; int* dV;
    cudaMalloc(&dL, n*4); cudaMalloc(&dH, n*4); cudaMalloc(&dA, n*4);
    cudaMalloc(&dB, n*4); cudaMalloc(&dO, n*4); cudaMalloc(&dV, n*4);
    cudaMemcpy(dL, L.data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dH, H.data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dA, A.data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, var.data(), n*4, cudaMemcpyHostToDevice);
    run<<<(n+127)/128, 128>>>(dL, dH, dA, dB, dV, dO, n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("FAIL: %s\n", cudaGetErrorString(e)); return 2; }
    std::vector<uint32_t> got(n);
    cudaMemcpy(got.data(), dO, n*4, cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) {
            if (bad < 5) printf("  variant %d  L=%08x H=%08x  got %08x want %08x\n",
                                var[i], L[i], H[i], got[i], want[i]);
            bad++;
        }
    }
    // A decoder that returned a constant would match nothing; assert the
    // outputs actually vary, so this cannot pass by degenerating.
    int distinct = 0;
    for (int i = 1; i < n; i++) if (got[i] != got[0]) { distinct = 1; break; }
    printf("%s: %d/%d vectors bit-exact across 4 variants%s\n",
           bad == 0 && distinct ? "PASS" : "FAIL", n - bad, n,
           distinct ? "" : " (DEGENERATE: every output identical)");
    return (bad == 0 && distinct) ? 0 : 1;
}
