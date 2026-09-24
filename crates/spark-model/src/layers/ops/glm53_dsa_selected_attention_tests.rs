// SPDX-License-Identifier: AGPL-3.0-only

use half::bf16;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

const HOST_SOURCE: &str = include_str!("glm53_dsa_selected_attention.rs");
const CUDA_SOURCE: &str =
    include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/glm53_dsa_selected_attention.cu");

/// The per-(row, head) reference kernel's source: the file minus the
/// row-shared rewrite, which deliberately repeats the reference's signature
/// and arithmetic tokens.
fn reference_cuda() -> String {
    let start = CUDA_SOURCE.find("// Row-shared rewrite").unwrap();
    let end = CUDA_SOURCE.find("// Before the selector becomes sparse").unwrap();
    format!("{}{}", &CUDA_SOURCE[..start], &CUDA_SOURCE[end..])
}

fn bf(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn network_indices(selected: &[i32], sequence_length: u32, query_position: u32) -> Vec<u32> {
    let mut ordered = [u32::MAX; 4_096];
    for (slot, &raw) in selected.iter().take(2_051).enumerate() {
        if let Ok(candidate) = u32::try_from(raw)
            && candidate < sequence_length
            && candidate <= query_position
        {
            ordered[slot] = candidate;
        }
    }
    let mut width = 2;
    while width <= 4_096 {
        let mut stride = width / 2;
        while stride != 0 {
            for left in 0..4_096 {
                let right = left ^ stride;
                if right > left {
                    let ascending = left & width == 0;
                    if (ascending && ordered[left] > ordered[right])
                        || (!ascending && ordered[left] < ordered[right])
                    {
                        ordered.swap(left, right);
                    }
                }
            }
            stride /= 2;
        }
        width *= 2;
    }
    let mut result = Vec::new();
    for candidate in ordered {
        if candidate == u32::MAX {
            break;
        }
        if result.last() != Some(&candidate) {
            result.push(candidate);
        }
    }
    result
}

fn reference(
    query: &[f32],
    cache: &[f32],
    capacity: usize,
    selected: &[i32],
    sequence_length: usize,
    query_position: usize,
    query_valid: bool,
) -> Vec<f32> {
    let mut output = vec![0.0; 64 * 512];
    if !query_valid
        || sequence_length == 0
        || sequence_length > capacity
        || query_position >= sequence_length
    {
        return output;
    }
    let mut ordered: Vec<usize> = selected
        .iter()
        .filter_map(|&raw| usize::try_from(raw).ok())
        .filter(|&index| index < sequence_length && index <= query_position)
        .collect();
    ordered.sort_unstable();
    ordered.dedup();
    for head in 0..64 {
        let q = &query[head * 512..(head + 1) * 512];
        let mut logits = Vec::with_capacity(ordered.len());
        for &token in &ordered {
            let key = &cache[token * 512..(token + 1) * 512];
            let mut partial = [0.0f32; 256];
            for lane in 0..256 {
                partial[lane] = bf(q[lane]) * bf(key[lane]);
                partial[lane] += bf(q[lane + 256]) * bf(key[lane + 256]);
            }
            let mut stride = 128;
            while stride != 0 {
                for lane in 0..stride {
                    partial[lane] += partial[lane + stride];
                }
                stride >>= 1;
            }
            logits.push(partial[0] * 0.0625);
        }
        let maximum = logits.iter().fold(
            f32::MIN,
            |value, &next| if next > value { next } else { value },
        );
        let exponentials: Vec<f32> = logits
            .iter()
            .map(|&value| (value - maximum).exp())
            .collect();
        let denominator = exponentials
            .iter()
            .copied()
            .fold(0.0f32, |sum, value| sum + value);
        for channel in 0..512 {
            let mut sum = 0.0f32;
            for (&token, &exponential) in ordered.iter().zip(&exponentials) {
                let probability = bf(exponential / denominator);
                sum += probability * bf(cache[token * 512 + channel]);
            }
            output[head * 512 + channel] = bf(sum);
        }
    }
    output
}

#[test]
fn plan_pins_1m_extents_and_rejects_fp8_or_wrong_geometry() {
    // The historic default prompt chunk; MAX_QUERIES itself is now 8192.
    let plan = Glm53DsaSelectedAttentionPlan::new(
        1,
        2_048,
        1_048_576,
        64,
        512,
        2_051,
        Glm53DsaSelectedStorage::Bf16,
    )
    .unwrap();
    assert_eq!((plan.grid_y, plan.grid_z), (2_048, 1));
    assert_eq!(plan.query_bytes, 134_217_728);
    assert_eq!(plan.latent_cache_bytes, 1_073_741_824);
    assert_eq!(plan.selected_index_bytes, 16_801_792);
    assert_eq!(plan.output_bytes, 134_217_728);
    let split = Glm53DsaSelectedAttentionPlan::new(
        70_000,
        1,
        1,
        64,
        512,
        2_051,
        Glm53DsaSelectedStorage::Bf16,
    )
    .unwrap();
    assert_eq!((split.grid_y, split.grid_z), (65_535, 2));
    for hostile in [
        Glm53DsaSelectedAttentionPlan::new(
            1,
            1,
            1,
            64,
            512,
            2_051,
            Glm53DsaSelectedStorage::Fp8E4M3,
        ),
        Glm53DsaSelectedAttentionPlan::new(0, 1, 1, 64, 512, 2_051, Glm53DsaSelectedStorage::Bf16),
        Glm53DsaSelectedAttentionPlan::new(
            1,
            MAX_QUERIES + 1,
            1,
            64,
            512,
            2_051,
            Glm53DsaSelectedStorage::Bf16,
        ),
        Glm53DsaSelectedAttentionPlan::new(
            1,
            1,
            1_048_577,
            64,
            512,
            2_051,
            Glm53DsaSelectedStorage::Bf16,
        ),
        Glm53DsaSelectedAttentionPlan::new(1, 1, 1, 63, 512, 2_051, Glm53DsaSelectedStorage::Bf16),
    ] {
        assert!(hostile.is_err());
    }
}

#[test]
fn head_transpose_admits_m2048_and_rejects_m2049_before_enqueue() {
    assert!(CUDA_SOURCE.contains("#define GLM53_DSA_MAX_QUERIES 8192U"));
    assert!(CUDA_SOURCE.contains("rows > GLM53_DSA_MAX_QUERIES"));
    assert!(!CUDA_SOURCE.contains("rows > 8U"));
    let gpu = MockGpuBackend::new();
    let kernel = Glm53DsaSelectedAttentionKernel::load(&gpu).unwrap();
    let rows = GLM53_EXL3_MAX_WIDE_ROWS as u32;
    let bytes = rows as usize * 64 * 512 * 2;
    let buffers = Glm53DsaHeadTransposeBuffers {
        input_bf16: GgmlIqBuffer {
            ptr: DevicePtr(0x2000_0000),
            bytes,
        },
        output_bf16: GgmlIqBuffer {
            ptr: DevicePtr(0x2100_0000),
            bytes,
        },
    };
    kernel
        .transpose_heads(&gpu, rows, 512, true, buffers, 0)
        .unwrap();
    assert_eq!(gpu.launch_count(), 1);
    assert!(
        kernel
            .transpose_heads(&gpu, rows + 1, 512, true, buffers, 0)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 1);
}

#[test]
fn reference_normalizes_permutation_duplicates_and_future_indices() {
    let mut query = vec![0.0f32; 64 * 512];
    let mut cache = vec![0.0f32; 6 * 512];
    for head in 0..64 {
        query[head * 512] = 1.0;
    }
    for token in 0..6 {
        cache[token * 512] = token as f32 - 2.0;
        cache[token * 512 + 1] = (token * token) as f32;
    }
    let mut scrambled = vec![-1; 2_051];
    scrambled[..8].copy_from_slice(&[4, 2, 4, 5, 3, 7, -1, 2]);
    let mut canonical = vec![-1; 2_051];
    canonical[..3].copy_from_slice(&[2, 3, 4]);
    let left = reference(&query, &cache, 6, &scrambled, 6, 4, true);
    let right = reference(&query, &cache, 6, &canonical, 6, 4, true);
    assert_eq!(left, right);
    assert!(left.iter().any(|&value| value != 0.0));
    let mut single = vec![-1; 2_051];
    single[0] = 4;
    let one = reference(&query, &cache, 6, &single, 6, 4, true);
    for head in 0..64 {
        assert_eq!(one[head * 512], bf(cache[4 * 512]));
        assert_eq!(one[head * 512 + 1], bf(cache[4 * 512 + 1]));
    }
    assert!(
        reference(&query, &cache, 6, &canonical, 6, 4, false)
            .iter()
            .all(|&value| value == 0.0)
    );
}

#[test]
fn exact_4096_network_matches_scalar_unique_order_at_full_selected_width() {
    let selected: Vec<i32> = (0..2_051)
        .map(|slot| {
            if slot % 97 == 0 {
                -1
            } else {
                ((slot * 997) % 2_048) as i32
            }
        })
        .rev()
        .collect();
    let mut scalar: Vec<u32> = selected
        .iter()
        .filter_map(|&raw| u32::try_from(raw).ok())
        .filter(|&index| index < 2_048 && index <= 2_047)
        .collect();
    scalar.sort_unstable();
    scalar.dedup();
    assert_eq!(network_indices(&selected, 2_048, 2_047), scalar);
}

#[test]
fn forged_extent_alignment_alias_or_overflow_has_zero_launches() {
    let gpu = MockGpuBackend::new();
    let kernel = Glm53DsaSelectedAttentionKernel::load(&gpu).unwrap();
    let plan =
        Glm53DsaSelectedAttentionPlan::new(1, 1, 4, 64, 512, 2_051, Glm53DsaSelectedStorage::Bf16)
            .unwrap();
    let mut address = 0x20_0000u64;
    let mut next = |bytes: usize| {
        address = address.checked_add(3).unwrap() & !3;
        let buffer = GgmlIqBuffer {
            ptr: DevicePtr(address),
            bytes,
        };
        address += u64::try_from(bytes).unwrap() + 0x1000;
        buffer
    };
    let valid = Glm53DsaSelectedAttentionBuffers {
        absorbed_query_bf16: next(plan.query_bytes),
        latent_cache_bf16: next(plan.latent_cache_bytes),
        selected_indices_i32: next(plan.selected_index_bytes),
        sequence_lengths_u32: next(plan.sequence_length_bytes),
        query_positions_u32: next(plan.query_position_bytes),
        query_validity_u8: next(plan.query_validity_bytes),
        output_weighted_latent_bf16: next(plan.output_bytes),
    };
    let mut forged = plan;
    forged.grid_z += 1;
    assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
    assert!(
        kernel
            .launch(
                &gpu,
                plan,
                Glm53DsaSelectedAttentionBuffers {
                    selected_indices_i32: GgmlIqBuffer {
                        bytes: plan.selected_index_bytes + 4,
                        ..valid.selected_indices_i32
                    },
                    ..valid
                },
                0,
            )
            .is_err()
    );
    assert!(
        kernel
            .launch(
                &gpu,
                plan,
                Glm53DsaSelectedAttentionBuffers {
                    query_positions_u32: GgmlIqBuffer {
                        ptr: DevicePtr(valid.query_positions_u32.ptr.0 + 1),
                        ..valid.query_positions_u32
                    },
                    ..valid
                },
                0,
            )
            .is_err()
    );
    assert!(
        kernel
            .launch(
                &gpu,
                plan,
                Glm53DsaSelectedAttentionBuffers {
                    output_weighted_latent_bf16: GgmlIqBuffer {
                        ptr: valid.absorbed_query_bf16.ptr,
                        ..valid.output_weighted_latent_bf16
                    },
                    ..valid
                },
                0,
            )
            .is_err()
    );
    assert!(
        kernel
            .launch(
                &gpu,
                plan,
                Glm53DsaSelectedAttentionBuffers {
                    latent_cache_bf16: GgmlIqBuffer {
                        ptr: DevicePtr(u64::MAX - 1),
                        ..valid.latent_cache_bf16
                    },
                    ..valid
                },
                0,
            )
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
    kernel.launch(&gpu, plan, valid, 0).unwrap();
    assert_eq!(gpu.launch_count(), 1);
}

#[test]
fn dense_causal_launch_is_exactly_b1_full_prefix_and_m2051_bounded() {
    assert!(CUDA_SOURCE.contains("#define KERNEL_NAME atlas_glm53_dsa_dense_causal_bf16"));
    assert!(CUDA_SOURCE.contains("#include \"../../common/prefill_paged_compute_512.cuh\""));
    assert!(CUDA_SOURCE.contains("(unsigned long long)_pos * HDIM_512 + _col"));
    let gpu = MockGpuBackend::new();
    let kernel = Glm53DsaSelectedAttentionKernel::load(&gpu).unwrap();
    let plan = Glm53DsaSelectedAttentionPlan::new(
        1,
        32,
        64,
        64,
        512,
        2_051,
        Glm53DsaSelectedStorage::Bf16,
    )
    .unwrap();
    let mut address = 0x20_0000u64;
    let mut next = |bytes: usize| {
        address = address.checked_add(3).unwrap() & !3;
        let buffer = GgmlIqBuffer {
            ptr: DevicePtr(address),
            bytes,
        };
        address += u64::try_from(bytes).unwrap() + 0x1000;
        buffer
    };
    let buffers = Glm53DsaSelectedAttentionBuffers {
        absorbed_query_bf16: next(plan.query_bytes),
        latent_cache_bf16: next(plan.latent_cache_bytes),
        selected_indices_i32: next(plan.selected_index_bytes),
        sequence_lengths_u32: next(plan.sequence_length_bytes),
        query_positions_u32: next(plan.query_position_bytes),
        query_validity_u8: next(plan.query_validity_bytes),
        output_weighted_latent_bf16: next(plan.output_bytes),
    };
    kernel
        .launch_dense_causal(&gpu, plan, buffers, 16, 48, 0)
        .unwrap();
    assert_eq!(gpu.launch_count(), 1);
    assert!(
        kernel
            .launch_dense_causal(&gpu, plan, buffers, 16, 49, 0)
            .is_err()
    );
    assert!(
        kernel
            .launch_dense_causal(&gpu, plan, buffers, 2_020, 2_052, 0)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 1);
}

fn swap_unique(source: &str, left: &str, right: &str) -> String {
    assert_eq!(source.matches(left).count(), 1);
    assert_eq!(source.matches(right).count(), 1);
    source
        .replace(left, "__GLM53_DSA_ABI_SWAP__")
        .replace(right, left)
        .replace("__GLM53_DSA_ABI_SWAP__", right)
}

#[test]
fn launch_argument_chain_matches_cuda_signature_and_rejects_swaps() {
    let reference = reference_cuda();
    let cuda_source = reference.as_str();
    const HOST_ARG_CHAIN: &str = r#".arg_ptr(buffers.absorbed_query_bf16.ptr)
            .arg_ptr(buffers.latent_cache_bf16.ptr)
            .arg_ptr(buffers.selected_indices_i32.ptr)
            .arg_ptr(buffers.sequence_lengths_u32.ptr)
            .arg_ptr(buffers.query_positions_u32.ptr)
            .arg_ptr(buffers.query_validity_u8.ptr)
            .arg_ptr(buffers.output_weighted_latent_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.queries)
            .arg_u32(plan.kv_capacity)"#;
    const CUDA_SIGNATURE: &str = r#"const __nv_bfloat16 * __restrict__ absorbed_query,
        const __nv_bfloat16 * __restrict__ latent_cache,
        const int * __restrict__ selected_indices,
        const unsigned int * __restrict__ sequence_lengths,
        const unsigned int * __restrict__ query_positions,
        const unsigned char * __restrict__ query_validity,
        __nv_bfloat16 * __restrict__ output_weighted_latent,
        unsigned int batch, unsigned int query_count,
        unsigned int kv_capacity) {"#;
    fn abi_contract(host: &str, cuda: &str) -> bool {
        host.contains(HOST_ARG_CHAIN) && cuda.contains(CUDA_SIGNATURE)
    }
    // The reference launch: after the row-shared early return.
    let sparse_start = HOST_SOURCE.find("KernelLaunch::new(gpu, self.attention)").unwrap();
    let sparse_end = HOST_SOURCE[sparse_start..]
        .find("pub fn transpose_heads(")
        .map(|offset| sparse_start + offset)
        .unwrap();
    let sparse_host = &HOST_SOURCE[sparse_start..sparse_end];
    assert!(abi_contract(sparse_host, cuda_source));

    let host_pointer_swaps = [
        (
            ".arg_ptr(buffers.absorbed_query_bf16.ptr)",
            ".arg_ptr(buffers.latent_cache_bf16.ptr)",
        ),
        (
            ".arg_ptr(buffers.selected_indices_i32.ptr)",
            ".arg_ptr(buffers.output_weighted_latent_bf16.ptr)",
        ),
    ];
    for (left, right) in host_pointer_swaps {
        assert!(!abi_contract(
            &swap_unique(sparse_host, left, right),
            cuda_source,
        ));
    }
    for (left, right) in [
        (".arg_u32(plan.batch)", ".arg_u32(plan.queries)"),
        (".arg_u32(plan.batch)", ".arg_u32(plan.kv_capacity)"),
    ] {
        assert!(!abi_contract(
            &swap_unique(sparse_host, left, right),
            cuda_source,
        ));
    }

    let cuda_pointer_swaps = [
        (
            "const __nv_bfloat16 * __restrict__ absorbed_query,",
            "const __nv_bfloat16 * __restrict__ latent_cache,",
        ),
        (
            "const int * __restrict__ selected_indices,",
            "__nv_bfloat16 * __restrict__ output_weighted_latent,",
        ),
    ];
    for (left, right) in cuda_pointer_swaps {
        assert!(!abi_contract(
            sparse_host,
            &swap_unique(cuda_source, left, right),
        ));
    }
    for cuda_scalar_swap in [
        cuda_source.replacen(
            "unsigned int batch, unsigned int query_count,",
            "unsigned int query_count, unsigned int batch,",
            1,
        ),
        cuda_source.replacen(
            "unsigned int batch, unsigned int query_count,\n        unsigned int kv_capacity",
            "unsigned int kv_capacity, unsigned int query_count,\n        unsigned int batch",
            1,
        ),
    ] {
        assert!(!abi_contract(sparse_host, &cuda_scalar_swap));
    }
}

#[test]
fn cuda_source_pins_sort_unique_causal_and_bf16_probability_chronology() {
    let reference = reference_cuda();
    let cuda_source = reference.as_str();
    const REQUIRED: &[&str] = &[
        "slot < GLM53_DSA_SELECTED",
        "candidate < sequence_length",
        "candidate <= query_position",
        "query_position >= sequence_length",
        "width <= GLM53_DSA_SORT_WIDTH",
        "ordered[left] > ordered[right]",
        "ordered[left] < ordered[right]",
        "candidate != previous",
        "for (unsigned int item = 0U; item < unique_count; ++item) {\n        const unsigned long long latent_base =",
        "for (unsigned int item = 0U; item < unique_count; ++item) {\n            if (exponentials[item] > maximum) {",
        "for (unsigned int item = 0U; item < unique_count; ++item) {\n            exponentials[item] = expf(",
        "for (unsigned int item = 0U; item < unique_count; ++item) {\n        const float probability = __bfloat162float(",
        "GLM53_DSA_INV_SQRT_QK 0.0625f",
        "partial[lane], partial[lane + stride]",
        "__fsub_rn(exponentials[item], maximum)",
        "__float2bfloat16_rn(exponentials[item] / denominator)",
        "first_sum, __fmul_rn(probability, first_value)",
        "second_sum, __fmul_rn(probability, second_value)",
        "__float2bfloat16_rn(first_sum)",
        "__float2bfloat16_rn(second_sum)",
    ];
    fn source_contract(source: &str) -> bool {
        REQUIRED.iter().all(|token| source.contains(token))
            && !source.contains("malloc(")
            && !source.contains("for (unsigned int token = 0U; token < sequence_length")
    }
    assert_eq!(cuda_source.matches("item < unique_count").count(), 4);
    assert!(source_contract(cuda_source));
    for required in REQUIRED {
        assert!(!source_contract(&cuda_source.replacen(required, "", 1)));
    }
}
