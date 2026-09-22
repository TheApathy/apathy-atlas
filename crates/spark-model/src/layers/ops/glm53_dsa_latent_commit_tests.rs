// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

const HOST_SOURCE: &str = include_str!("glm53_dsa_latent_commit.rs");
const CUDA_SOURCE: &str =
    include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/glm53_dsa_latent_commit.cu");

#[allow(clippy::too_many_arguments)]
fn commit_reference(
    overlay: &[u16],
    cache: &mut [u16],
    ends: &mut [u32],
    nonces: &mut [u64],
    lengths: &mut [u32],
    queries: usize,
    accepted: usize,
    capacity: usize,
    start: usize,
    expected_end: u32,
    expected_nonce: u64,
) -> bool {
    let batch = lengths.len();
    assert_eq!(overlay.len(), batch * queries * 512);
    assert_eq!(cache.len(), batch * capacity * 512);
    if ends.len() != batch
        || nonces.len() != batch
        || ends.iter().any(|&value| value != expected_end)
        || nonces.iter().any(|&value| value != expected_nonce)
        || lengths.iter().any(|&value| value != start as u32)
    {
        return false;
    }
    for batch_index in 0..batch {
        for query in 0..accepted {
            let source = (batch_index * queries + query) * 512;
            let target = (batch_index * capacity + start + query) * 512;
            cache[target..target + 512].copy_from_slice(&overlay[source..source + 512]);
        }
    }
    lengths.fill((start + accepted) as u32);
    ends.fill(0);
    nonces.fill(0);
    true
}

#[test]
fn plan_pins_zero_to_full_accept_and_exact_1m_extents() {
    for accepted in [0, 8] {
        let plan = Glm53DsaLatentCommitPlan::new(
            1,
            8,
            accepted,
            1_048_576,
            1_048_568,
            1_048_576,
            512,
            0x1234,
            Glm53DsaLatentCommitStorage::Bf16,
        )
        .unwrap();
        assert_eq!(plan.overlay_bytes, 8_192);
        assert_eq!(plan.persistent_cache_bytes, 1_073_741_824);
        assert_eq!(plan.published_end_bytes, 4);
        assert_eq!(plan.published_nonce_bytes, 8);
        assert_eq!(plan.logical_length_bytes, 4);
    }
    for hostile in [
        Glm53DsaLatentCommitPlan::new(0, 1, 0, 1, 0, 1, 512, 1, Glm53DsaLatentCommitStorage::Bf16),
        Glm53DsaLatentCommitPlan::new(1, 0, 0, 1, 0, 0, 512, 1, Glm53DsaLatentCommitStorage::Bf16),
        Glm53DsaLatentCommitPlan::new(1, 9, 0, 9, 0, 9, 512, 1, Glm53DsaLatentCommitStorage::Bf16),
        Glm53DsaLatentCommitPlan::new(1, 1, 2, 2, 0, 1, 512, 1, Glm53DsaLatentCommitStorage::Bf16),
        Glm53DsaLatentCommitPlan::new(1, 1, 0, 0, 0, 1, 512, 1, Glm53DsaLatentCommitStorage::Bf16),
        Glm53DsaLatentCommitPlan::new(
            1,
            1,
            0,
            1_048_577,
            0,
            1,
            512,
            1,
            Glm53DsaLatentCommitStorage::Bf16,
        ),
        Glm53DsaLatentCommitPlan::new(1, 1, 0, 1, 0, 1, 511, 1, Glm53DsaLatentCommitStorage::Bf16),
        Glm53DsaLatentCommitPlan::new(1, 1, 0, 1, 0, 1, 512, 0, Glm53DsaLatentCommitStorage::Bf16),
        Glm53DsaLatentCommitPlan::new(
            1,
            1,
            0,
            1,
            0,
            1,
            512,
            1,
            Glm53DsaLatentCommitStorage::Fp8E4M3,
        ),
        Glm53DsaLatentCommitPlan::new(1, 2, 0, 2, 0, 1, 512, 1, Glm53DsaLatentCommitStorage::Bf16),
        Glm53DsaLatentCommitPlan::new(
            1,
            1,
            0,
            1,
            u32::MAX,
            0,
            512,
            1,
            Glm53DsaLatentCommitStorage::Bf16,
        ),
    ] {
        assert!(hostile.is_err());
    }
}

#[test]
fn oracle_prevalidates_all_batches_before_effects_and_commits_only_prefix() {
    let (batch, queries, accepted, capacity, start, end, nonce) = (3, 3, 2, 7, 2, 5, 0xabcdu64);
    let overlay: Vec<u16> = (0..batch * queries * 512)
        .map(|index| (index as u16).wrapping_mul(19).wrapping_add(7))
        .collect();
    let original_cache = vec![0x7badu16; batch * capacity * 512];
    for fault in 0..3 {
        let mut cache = original_cache.clone();
        let mut ends = vec![end; batch];
        let mut nonces = vec![nonce; batch];
        let mut lengths = vec![start as u32; batch];
        match fault {
            0 => ends[batch - 1] -= 1,
            1 => nonces[batch - 1] ^= 1,
            _ => lengths[batch - 1] += 1,
        }
        let before = (cache.clone(), ends.clone(), nonces.clone(), lengths.clone());
        assert!(!commit_reference(
            &overlay,
            &mut cache,
            &mut ends,
            &mut nonces,
            &mut lengths,
            queries,
            accepted,
            capacity,
            start,
            end,
            nonce,
        ));
        assert_eq!((cache, ends, nonces, lengths), before);
    }

    let mut cache = original_cache.clone();
    let mut ends = vec![end; batch];
    let mut nonces = vec![nonce; batch];
    let mut lengths = vec![start as u32; batch];
    assert!(commit_reference(
        &overlay,
        &mut cache,
        &mut ends,
        &mut nonces,
        &mut lengths,
        queries,
        accepted,
        capacity,
        start,
        end,
        nonce,
    ));
    assert_eq!(lengths, vec![4; batch]);
    assert_eq!(ends, vec![0; batch]);
    assert_eq!(nonces, vec![0; batch]);
    for batch_index in 0..batch {
        for query in 0..accepted {
            let source = (batch_index * queries + query) * 512;
            let target = (batch_index * capacity + start + query) * 512;
            assert_eq!(&cache[target..target + 512], &overlay[source..source + 512]);
        }
        let hidden = (batch_index * capacity + start + accepted) * 512;
        assert!(
            cache[hidden..hidden + 512]
                .iter()
                .all(|&value| value == 0x7bad)
        );
    }

    let before = original_cache.clone();
    let mut cache = original_cache;
    let mut ends = vec![end; batch];
    let mut nonces = vec![nonce; batch];
    let mut lengths = vec![start as u32; batch];
    assert!(commit_reference(
        &overlay,
        &mut cache,
        &mut ends,
        &mut nonces,
        &mut lengths,
        queries,
        0,
        capacity,
        start,
        end,
        nonce,
    ));
    assert_eq!(cache, before);
    assert_eq!(lengths, vec![start as u32; batch]);
    assert_eq!((ends, nonces), (vec![0; batch], vec![0; batch]));
}

fn fake_buffers(plan: Glm53DsaLatentCommitPlan) -> Glm53DsaLatentCommitBuffers {
    Glm53DsaLatentCommitBuffers {
        transaction_overlay_bf16: GgmlIqBuffer {
            ptr: DevicePtr(0x20_0000),
            bytes: plan.overlay_bytes,
        },
        persistent_cache_bf16: GgmlIqBuffer {
            ptr: DevicePtr(0x30_0000),
            bytes: plan.persistent_cache_bytes,
        },
        published_ends_u32: GgmlIqBuffer {
            ptr: DevicePtr(0x40_0000),
            bytes: plan.published_end_bytes,
        },
        published_nonces_u64: GgmlIqBuffer {
            ptr: DevicePtr(0x50_0000),
            bytes: plan.published_nonce_bytes,
        },
        logical_lengths_u32: GgmlIqBuffer {
            ptr: DevicePtr(0x60_0000),
            bytes: plan.logical_length_bytes,
        },
    }
}

#[test]
fn forged_extent_alignment_alias_or_address_overflow_has_zero_launches() {
    let gpu = MockGpuBackend::new();
    let kernel = Glm53DsaLatentCommitKernel::load(&gpu).unwrap();
    let plan =
        Glm53DsaLatentCommitPlan::new(1, 2, 1, 4, 2, 4, 512, 9, Glm53DsaLatentCommitStorage::Bf16)
            .unwrap();
    let valid = fake_buffers(plan);
    let mut forged = plan;
    forged.overlay_bytes += 2;
    assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
    for hostile in [
        Glm53DsaLatentCommitBuffers {
            persistent_cache_bf16: GgmlIqBuffer {
                bytes: plan.persistent_cache_bytes + 2,
                ..valid.persistent_cache_bf16
            },
            ..valid
        },
        Glm53DsaLatentCommitBuffers {
            published_nonces_u64: GgmlIqBuffer {
                ptr: DevicePtr(valid.published_nonces_u64.ptr.0 + 4),
                ..valid.published_nonces_u64
            },
            ..valid
        },
        Glm53DsaLatentCommitBuffers {
            logical_lengths_u32: GgmlIqBuffer {
                ptr: valid.published_ends_u32.ptr,
                ..valid.logical_lengths_u32
            },
            ..valid
        },
        Glm53DsaLatentCommitBuffers {
            transaction_overlay_bf16: GgmlIqBuffer {
                ptr: DevicePtr(u64::MAX - 1),
                ..valid.transaction_overlay_bf16
            },
            ..valid
        },
        Glm53DsaLatentCommitBuffers {
            published_ends_u32: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                ..valid.published_ends_u32
            },
            ..valid
        },
    ] {
        assert!(kernel.launch(&gpu, plan, hostile, 0).is_err());
    }
    assert_eq!(gpu.launch_count(), 0);
    kernel.launch(&gpu, plan, valid, 0).unwrap();
    assert_eq!(gpu.launch_count(), 1);
}

fn swap_unique(source: &str, left: &str, right: &str) -> String {
    assert_eq!(source.matches(left).count(), 1);
    assert_eq!(source.matches(right).count(), 1);
    source
        .replace(left, "__LEFT__")
        .replace(right, left)
        .replace("__LEFT__", right)
}

fn abi_contract(host: &str, cuda: &str) -> bool {
    const HOST: &str = ".arg_ptr(buffers.transaction_overlay_bf16.ptr)\n            .arg_ptr(buffers.persistent_cache_bf16.ptr)\n            .arg_ptr(buffers.published_ends_u32.ptr)\n            .arg_ptr(buffers.published_nonces_u64.ptr)\n            .arg_ptr(buffers.logical_lengths_u32.ptr)\n            .arg_u32(plan.batch)\n            .arg_u32(plan.queries)\n            .arg_u32(plan.accepted)\n            .arg_u32(plan.capacity)\n            .arg_u32(plan.start_position)\n            .arg_u32(plan.end_position)\n            .arg_u64(plan.transaction_nonce)";
    const CUDA: &str = "const __nv_bfloat16 * __restrict__ transaction_overlay,\n        __nv_bfloat16 * __restrict__ persistent_cache,\n        unsigned int * __restrict__ published_ends,\n        unsigned long long * __restrict__ published_nonces,\n        unsigned int * __restrict__ logical_lengths,\n        unsigned int batch, unsigned int query_count,\n        unsigned int accepted_count, unsigned int capacity,\n        unsigned int start_position, unsigned int end_position,\n        unsigned long long transaction_nonce";
    host.contains(HOST) && cuda.contains(CUDA)
}

#[test]
fn ordered_host_cuda_abi_rejects_pointer_and_scalar_swaps() {
    assert!(abi_contract(HOST_SOURCE, CUDA_SOURCE));
    for (left, right) in [
        (
            ".arg_ptr(buffers.transaction_overlay_bf16.ptr)",
            ".arg_ptr(buffers.published_ends_u32.ptr)",
        ),
        (
            ".arg_ptr(buffers.persistent_cache_bf16.ptr)",
            ".arg_ptr(buffers.logical_lengths_u32.ptr)",
        ),
        (".arg_u32(plan.batch)", ".arg_u32(plan.accepted)"),
        (".arg_u32(plan.queries)", ".arg_u32(plan.capacity)"),
        (
            ".arg_u32(plan.end_position)",
            ".arg_u64(plan.transaction_nonce)",
        ),
    ] {
        assert!(!abi_contract(
            &swap_unique(HOST_SOURCE, left, right),
            CUDA_SOURCE
        ));
    }
    for (left, right) in [
        (
            "const __nv_bfloat16 * __restrict__ transaction_overlay,",
            "unsigned int * __restrict__ published_ends,",
        ),
        (
            "__nv_bfloat16 * __restrict__ persistent_cache,",
            "unsigned int * __restrict__ logical_lengths,",
        ),
        ("unsigned int batch", "unsigned int accepted_count"),
        ("unsigned int query_count", "unsigned int capacity"),
        (
            "unsigned int end_position",
            "unsigned long long transaction_nonce",
        ),
    ] {
        assert!(!abi_contract(
            HOST_SOURCE,
            &swap_unique(CUDA_SOURCE, left, right)
        ));
    }
}

#[test]
fn source_contract_pins_global_prevalidation_copy_publish_and_nonce_last_retire() {
    const INVARIANTS: &[&str] = &[
        "accepted_count > query_count",
        "computed_end != end_position",
        "computed_end > capacity",
        "gridDim.x != 1U || gridDim.y != 1U || gridDim.z != 1U",
    ];
    const REQUIRED: &[&str] = &[
        "invalid_transaction = 0U;",
        "published_ends[batch_index] != end_position",
        "published_nonces[batch_index] != transaction_nonce",
        "logical_lengths[batch_index] != start_position",
        "atomicExch(&invalid_transaction, 1U);",
        "if (invalid_transaction != 0U) {\n        return;",
        "(unsigned long long)batch * accepted_count * GLM53_DSA_COMMIT_LATENT",
        "if (accepted_count != 0U) {",
        "element / GLM53_DSA_COMMIT_LATENT;",
        "element % GLM53_DSA_COMMIT_LATENT;",
        "accepted_row / accepted_count;",
        "accepted_row - batch_index * accepted_count;",
        "(batch_index * query_count + query_index) *",
        "(batch_index * capacity + start_position + query_index) *",
        "persistent_cache[cache_index] = transaction_overlay[overlay_index];",
        "const unsigned int committed_length = start_position + accepted_count;",
        "logical_lengths[batch_index] = committed_length;",
        "published_ends[batch_index] = 0U;",
        "atomicExch(published_nonces + batch_index, 0ULL);",
    ];
    const PHASES: &[&str] = &[
        "invalid_transaction = 0U;\n    }\n    __syncthreads();\n    for (unsigned long long batch_index = lane; batch_index < batch;",
        "atomicExch(&invalid_transaction, 1U);\n        }\n    }\n    __syncthreads();\n    if (invalid_transaction != 0U) {",
        "        }\n    }\n    __threadfence();\n    __syncthreads();\n\n    const unsigned int committed_length",
        "logical_lengths[batch_index] = committed_length;\n    }\n    __threadfence();\n    __syncthreads();",
        "published_ends[batch_index] = 0U;\n    }\n    __threadfence();\n    __syncthreads();",
    ];
    fn contract(source: &str) -> bool {
        let positions: Vec<_> = REQUIRED.iter().map(|token| source.find(token)).collect();
        REQUIRED
            .iter()
            .all(|token| source.matches(token).count() == 1)
            && INVARIANTS
                .iter()
                .all(|token| source.matches(token).count() == 1)
            && positions.iter().all(Option::is_some)
            && positions.windows(2).all(|pair| pair[0] < pair[1])
            && PHASES
                .iter()
                .all(|phase| source.matches(phase).count() == 1)
            && source.matches("__threadfence();").count() == 3
            && source.matches("__syncthreads();").count() == 5
            && !source.contains("__nv_fp8")
            && !source.contains("bfloat162float")
    }
    assert!(contract(CUDA_SOURCE));
    for required in REQUIRED {
        assert!(!contract(&CUDA_SOURCE.replacen(required, "", 1)));
    }
    for invariant in INVARIANTS {
        assert!(!contract(&CUDA_SOURCE.replacen(invariant, "", 1)));
    }
    for phase in PHASES {
        assert!(!contract(&CUDA_SOURCE.replacen(phase, "", 1)));
    }
    for (left, right) in [
        (REQUIRED[5], REQUIRED[7]),
        (REQUIRED[14], REQUIRED[16]),
        (REQUIRED[16], REQUIRED[17]),
        (REQUIRED[17], REQUIRED[18]),
    ] {
        assert!(!contract(&swap_unique(CUDA_SOURCE, left, right)));
    }
    assert!(!contract(&CUDA_SOURCE.replacen("__threadfence();", "", 1)));
    assert!(!contract(&CUDA_SOURCE.replacen("__syncthreads();", "", 1)));
}
