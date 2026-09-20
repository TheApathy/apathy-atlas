// SPDX-License-Identifier: MIT
// Derived from ExLlamaV3 e648f1a131365aae15920073e761a3fa5a527654.
/*
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

// Route-private equivalent of the donor's had_hf_r_128_d_inner. Every
// source-pair row is unique, so the final Hadamard values are ordinary stores
// rather than unordered atomic additions into a shared token row.
inline __device__
void atlas_glm53_had_hf_r_128_d_private(
    const half* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ post_scale,
    const float r_scale)
{
    int t = threadIdx.x & 31;
    half4 v = ((half4*) input_ptr)[t];

    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    shuffle_had_f4x32(h0, h1, h2, h3, t);
    h0 *= r_scale;
    h1 *= r_scale;
    h2 *= r_scale;
    h3 *= r_scale;

    int i = blockIdx.y * 32 + t;
    half4 scales = ((half4*) post_scale)[i];
    h0 *= __low2float(scales.x);
    h1 *= __high2float(scales.x);
    h2 *= __low2float(scales.y);
    h3 *= __high2float(scales.y);

    extern __shared__ float temp_shared[];
    int warp_id = threadIdx.x / 32;
    float* sh = temp_shared + warp_id * 128;
    sh[t * 4 + 0] = h0;
    sh[t * 4 + 1] = h1;
    sh[t * 4 + 2] = h2;
    sh[t * 4 + 3] = h3;
    __syncwarp();
    output_ptr[ 0 + t] = sh[ 0 + t];
    output_ptr[32 + t] = sh[32 + t];
    output_ptr[64 + t] = sh[64 + t];
    output_ptr[96 + t] = sh[96 + t];
}
