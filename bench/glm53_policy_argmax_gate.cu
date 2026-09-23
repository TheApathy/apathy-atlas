// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <cstring>
#include <cstdio>
#include <vector>

extern "C" __global__ void atlas_glm53_policy_argmax_bf16_rows(
    const __nv_bfloat16 *, unsigned int *, float *, unsigned int, unsigned int,
    unsigned int, unsigned int);

int main() {
    constexpr unsigned int rows = 4;
    constexpr unsigned int columns = 154880;
    std::vector<__nv_bfloat16> input((size_t) rows * columns, __float2bfloat16(-INFINITY));
    input[10] = __float2bfloat16(7.0f);
    input[11] = __float2bfloat16(6.0f);
    input[(size_t) columns + 5] = __float2bfloat16(9.0f);
    input[(size_t) columns + 100] = __float2bfloat16(9.0f);
    input[(size_t) 2 * columns + 20] = __float2bfloat16(5.0f);
    input[(size_t) 2 * columns + 21] = __float2bfloat16(NAN);
    const auto original = input;

    __nv_bfloat16 *device_input = nullptr;
    unsigned int *device_indices = nullptr;
    float *device_values = nullptr;
    if (cudaMalloc(&device_input, input.size() * sizeof(input[0])) != cudaSuccess ||
        cudaMalloc(&device_indices, rows * sizeof(unsigned int)) != cudaSuccess ||
        cudaMalloc(&device_values, rows * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(device_input, input.data(), input.size() * sizeof(input[0]),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return 2;
    }
    atlas_glm53_policy_argmax_bf16_rows<<<rows, 1024>>>(
        device_input, device_indices, device_values, rows, columns, 10, UINT_MAX);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    unsigned int indices[rows] = {};
    float values[rows] = {};
    if (cudaMemcpy(indices, device_indices, sizeof(indices), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(values, device_values, sizeof(values), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(input.data(), device_input, input.size() * sizeof(input[0]),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 4;
    }
    const bool pass = indices[0] == 11 && values[0] == 6.0f &&
        indices[1] == 100 && values[1] == 9.0f &&
        indices[2] == 20 && std::isnan(values[2]) &&
        indices[3] == columns - 1 && std::isinf(values[3]) && values[3] < 0.0f &&
        std::memcmp(input.data(), original.data(), input.size() * sizeof(input[0])) == 0;
    cudaFree(device_values);
    cudaFree(device_indices);
    cudaFree(device_input);
    std::printf("policy_argmax_gate=%s indices=%u,%u,%u,%u\n",
                pass ? "PASS" : "FAIL", indices[0], indices[1], indices[2], indices[3]);
    return pass ? 0 : 5;
}
