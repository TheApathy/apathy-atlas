// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

inline std::vector<unsigned short> run_case(const CaseSpec& spec, unsigned int n,
                                             const FixtureData& data) {
    const unsigned int stride = spec.strided ? spec.width : n;
    Guarded<__nv_bfloat16> out0(spec.rows * stride);
    Guarded<__nv_bfloat16> out1(spec.rows * stride);
    const dim3 block(spec.kind == Kind::V4 ? 128 :
                     spec.kind == Kind::DualBatch3Tuned ? 512 : 256);
    const dim3 grid((n + spec.width - 1) / spec.width, 1,
                    spec.projections == 2 && spec.kind != Kind::DualBatch3Tuned ? 2 : 1);

    switch (spec.kind) {
        case Kind::Batch2:
            w4a16_gemv_batch2<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                               out0.data, n, FIXTURE_K);
            break;
        case Kind::Qg:
            w4a16_gemv_qg<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                           out0.data, n, FIXTURE_K, 1, 2);
            break;
        case Kind::Qkvz:
            w4a16_gemv_qkvz<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                             out0.data, n, FIXTURE_K, 1, 1, 1, 1);
            break;
        case Kind::QgBatch2:
            w4a16_gemv_qg_batch2<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                                  out0.data, n, FIXTURE_K, 1, 2);
            break;
        case Kind::DualBatch2:
            w4a16_gemv_dual_batch2<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                out0.data, data.packed, data.scales, 1.0f, out1.data, n, FIXTURE_K);
            break;
        case Kind::Batch3:
            w4a16_gemv_batch3<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                               out0.data, n, FIXTURE_K);
            break;
        case Kind::QgBatch3:
            w4a16_gemv_qg_batch3<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                                  out0.data, n, FIXTURE_K, 1, 2);
            break;
        case Kind::DualBatch3:
            w4a16_gemv_dual_batch3<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                out0.data, data.packed, data.scales, 1.0f, out1.data, n, FIXTURE_K);
            break;
        case Kind::V1:
            w4a16_gemv_v1<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                           out0.data, n, FIXTURE_K);
            break;
        case Kind::V3:
            w4a16_gemv_v3<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                           out0.data, n, FIXTURE_K);
            break;
        case Kind::Batch3Logits:
            w4a16_gemv_batch3_logits<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                                      out0.data, n, FIXTURE_K);
            break;
        case Kind::V4:
            w4a16_gemv_v4<<<grid, block>>>(data.a, data.packed, data.scales, 1.0f,
                                           out0.data, n, FIXTURE_K);
            break;
        case Kind::QgBatch3Strided:
            w4a16_gemv_qg_batch3_strided<<<grid, block>>>(data.a, data.packed, data.scales,
                1.0f, out0.data, n, FIXTURE_K, 1, 2, stride);
            break;
        case Kind::DualBatch3Strided:
            w4a16_gemv_dual_batch3_strided<<<grid, block>>>(data.a, data.packed, data.scales,
                1.0f, out0.data, data.packed, data.scales, 1.0f, out1.data, n,
                FIXTURE_K, stride);
            break;
        case Kind::DualBatch3Tuned:
            w4a16_gemv_dual_batch3_tuned<<<grid, block>>>(data.a, data.packed, data.scales,
                1.0f, out0.data, data.packed, data.scales, 1.0f, out1.data, n, FIXTURE_K);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<bool> written(spec.rows * stride, false);
    for (unsigned int row = 0; row < spec.rows; ++row) {
        for (unsigned int col = 0; col < n; ++col) written[row * stride + col] = true;
    }
    out0.verify_canaries(written, std::string(spec.name) + "/out0/N=" + std::to_string(n));
    auto result = out0.normalize(spec.rows, stride, n);
    if (spec.projections == 2) {
        out1.verify_canaries(written, std::string(spec.name) + "/out1/N=" + std::to_string(n));
        auto second = out1.normalize(spec.rows, stride, n);
        result.insert(result.end(), second.begin(), second.end());
    }
    return result;
}

inline void compare_prefix(const CaseSpec& spec, unsigned int n,
                           const std::vector<unsigned short>& actual,
                           const std::vector<unsigned short>& oracle) {
    const size_t expected_full = spec.projections * spec.rows * spec.width;
    if (oracle.size() != expected_full) throw std::runtime_error(std::string(spec.name) + ": bad oracle size");
    for (unsigned int proj = 0; proj < spec.projections; ++proj) {
        for (unsigned int row = 0; row < spec.rows; ++row) {
            for (unsigned int col = 0; col < n; ++col) {
                const size_t got = (proj * spec.rows + row) * n + col;
                const size_t want = (proj * spec.rows + row) * spec.width + col;
                if (actual[got] != oracle[want]) {
                    throw std::runtime_error(std::string(spec.name) + "/N=" +
                        std::to_string(n) + ": byte mismatch");
                }
            }
        }
    }
}

