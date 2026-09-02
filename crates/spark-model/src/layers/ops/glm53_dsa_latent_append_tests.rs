// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

const HOST_SOURCE: &str = include_str!("glm53_dsa_latent_append.rs");
const CUDA_SOURCE: &str =
    include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/glm53_dsa_latent_append.cu");

#[derive(Clone)]
struct Staged {
    overlay: Vec<u16>,
    ends: Vec<u32>,
    nonces: Vec<u64>,
}

fn stage_reference(
    source: &[u16],
    batch: usize,
    queries: usize,
    end: u32,
    nonce: u64,
    copied_elements_before_failure: Option<usize>,
) -> Staged {
    let extent = batch * queries * 512;
    assert_eq!(source.len(), extent);
    let copied = copied_elements_before_failure.unwrap_or(extent).min(extent);
    let mut staged = Staged {
        overlay: vec![0xdead; extent],
        ends: vec![0; batch],
        nonces: vec![0; batch],
    };
    staged.overlay[..copied].copy_from_slice(&source[..copied]);
    if copied_elements_before_failure.is_none() {
        staged.ends.fill(end);
        staged.nonces.fill(nonce);
    }
    staged
}

#[allow(clippy::too_many_arguments)]
fn commit_reference(
    persistent: &mut [u16],
    logical_lengths: &mut [u32],
    staged: &Staged,
    capacity: usize,
    start: usize,
    end: u32,
    queries: usize,
    expected_nonce: u64,
    accepted: usize,
    transaction_active: bool,
) -> Result<()> {
    if !transaction_active
        || expected_nonce == 0
        || accepted > queries
        || staged.ends.iter().any(|&value| value != end)
        || staged.nonces.iter().any(|&value| value != expected_nonce)
    {
        bail!("overlay publication does not match the active transaction");
    }
    let batch = logical_lengths.len();
    assert_eq!(persistent.len(), batch * capacity * 512);
    for batch_index in 0..batch {
        for query in 0..accepted {
            let source = (batch_index * queries + query) * 512;
            let target = (batch_index * capacity + start + query) * 512;
            persistent[target..target + 512].copy_from_slice(&staged.overlay[source..source + 512]);
        }
        logical_lengths[batch_index] = u32::try_from(start + accepted)?;
    }
    Ok(())
}

#[test]
fn plan_pins_full_1m_boundary_overlay_extents_and_bf16_only() {
    let plan = Glm53DsaLatentAppendPlan::new(
        1,
        8,
        1_048_576,
        1_048_568,
        1_048_576,
        512,
        0x1234,
        Glm53DsaLatentAppendStorage::Bf16,
    )
    .unwrap();
    assert_eq!((plan.grid_y, plan.grid_z), (1, 1));
    assert_eq!(plan.source_bytes, 8_192);
    assert_eq!(plan.overlay_bytes, 8_192);
    assert_eq!(plan.published_end_bytes, 4);
    assert_eq!(plan.published_nonce_bytes, 8);

    let split = Glm53DsaLatentAppendPlan::new(
        70_000,
        1,
        1,
        0,
        1,
        512,
        1,
        Glm53DsaLatentAppendStorage::Bf16,
    )
    .unwrap();
    assert_eq!((split.grid_y, split.grid_z), (65_535, 2));

    for hostile in [
        Glm53DsaLatentAppendPlan::new(0, 1, 1, 0, 1, 512, 1, Glm53DsaLatentAppendStorage::Bf16),
        Glm53DsaLatentAppendPlan::new(1, 0, 1, 0, 0, 512, 1, Glm53DsaLatentAppendStorage::Bf16),
        Glm53DsaLatentAppendPlan::new(1, 1, 0, 0, 1, 512, 1, Glm53DsaLatentAppendStorage::Bf16),
        Glm53DsaLatentAppendPlan::new(1, 9, 9, 0, 9, 512, 1, Glm53DsaLatentAppendStorage::Bf16),
        Glm53DsaLatentAppendPlan::new(
            1,
            1,
            1_048_577,
            0,
            1,
            512,
            1,
            Glm53DsaLatentAppendStorage::Bf16,
        ),
        Glm53DsaLatentAppendPlan::new(1, 1, 1, 0, 1, 511, 1, Glm53DsaLatentAppendStorage::Bf16),
        Glm53DsaLatentAppendPlan::new(1, 1, 1, 0, 1, 512, 0, Glm53DsaLatentAppendStorage::Bf16),
        Glm53DsaLatentAppendPlan::new(1, 1, 1, 0, 1, 512, 1, Glm53DsaLatentAppendStorage::Fp8E4M3),
        Glm53DsaLatentAppendPlan::new(1, 2, 2, 0, 1, 512, 1, Glm53DsaLatentAppendStorage::Bf16),
        Glm53DsaLatentAppendPlan::new(
            1,
            1,
            1,
            u32::MAX,
            0,
            512,
            1,
            Glm53DsaLatentAppendStorage::Bf16,
        ),
        Glm53DsaLatentAppendPlan::new(
            u32::MAX,
            1,
            1,
            0,
            1,
            512,
            1,
            Glm53DsaLatentAppendStorage::Bf16,
        ),
    ] {
        assert!(hostile.is_err());
    }
}

#[test]
fn oracle_hides_failed_or_rolled_back_rows_and_commits_only_prefix() {
    let batch = 2;
    let queries = 3;
    let capacity = 7;
    let start = 2;
    let end = 5;
    let nonce = 0xabcdu64;
    let source: Vec<u16> = (0..batch * queries * 512)
        .map(|index| (index as u16).wrapping_mul(17).wrapping_add(3))
        .collect();
    let sentinel = 0x7badu16;
    let original = vec![sentinel; batch * capacity * 512];
    let original_lengths = vec![start as u32; batch];

    let failed = stage_reference(&source, batch, queries, end, nonce, Some(777));
    let mut persistent = original.clone();
    let mut lengths = original_lengths.clone();
    assert!(
        commit_reference(
            &mut persistent,
            &mut lengths,
            &failed,
            capacity,
            start,
            end,
            queries,
            nonce,
            queries,
            true,
        )
        .is_err()
    );
    assert_eq!(persistent, original);
    assert_eq!(lengths, original_lengths);

    let complete = stage_reference(&source, batch, queries, end, nonce, None);
    assert!(
        commit_reference(
            &mut persistent,
            &mut lengths,
            &complete,
            capacity,
            start,
            end,
            queries,
            nonce,
            queries,
            false,
        )
        .is_err()
    );
    assert_eq!(persistent, original);
    commit_reference(
        &mut persistent,
        &mut lengths,
        &complete,
        capacity,
        start,
        end,
        queries,
        nonce,
        2,
        true,
    )
    .unwrap();
    assert_eq!(lengths, vec![4, 4]);
    for batch_index in 0..batch {
        for query in 0..2 {
            let source_base = (batch_index * queries + query) * 512;
            let cache_base = (batch_index * capacity + start + query) * 512;
            assert_eq!(
                &persistent[cache_base..cache_base + 512],
                &source[source_base..source_base + 512]
            );
        }
        let hidden = (batch_index * capacity + start + 2) * 512;
        assert!(
            persistent[hidden..hidden + 512]
                .iter()
                .all(|&value| value == sentinel)
        );
    }
}

fn fake_buffers(plan: Glm53DsaLatentAppendPlan) -> Glm53DsaLatentAppendBuffers {
    Glm53DsaLatentAppendBuffers {
        source_bf16: GgmlIqBuffer {
            ptr: DevicePtr(0x20_0000),
            bytes: plan.source_bytes,
        },
        transaction_overlay_bf16: GgmlIqBuffer {
            ptr: DevicePtr(0x30_0000),
            bytes: plan.overlay_bytes,
        },
        published_ends_u32: GgmlIqBuffer {
            ptr: DevicePtr(0x40_0000),
            bytes: plan.published_end_bytes,
        },
        published_nonces_u64: GgmlIqBuffer {
            ptr: DevicePtr(0x50_0000),
            bytes: plan.published_nonce_bytes,
        },
    }
}

#[test]
fn forged_extent_alignment_alias_or_address_overflow_precedes_side_effects() {
    let gpu = MockGpuBackend::new();
    let kernel = Glm53DsaLatentAppendKernel::load(&gpu).unwrap();
    let plan =
        Glm53DsaLatentAppendPlan::new(1, 1, 4, 2, 3, 512, 9, Glm53DsaLatentAppendStorage::Bf16)
            .unwrap();
    let valid = fake_buffers(plan);
    let mut forged = plan;
    forged.grid_z += 1;
    assert!(kernel.launch_stage(&gpu, forged, valid, 0).is_err());
    for hostile in [
        Glm53DsaLatentAppendBuffers {
            source_bf16: GgmlIqBuffer {
                bytes: plan.source_bytes + 2,
                ..valid.source_bf16
            },
            ..valid
        },
        Glm53DsaLatentAppendBuffers {
            published_ends_u32: GgmlIqBuffer {
                ptr: DevicePtr(valid.published_ends_u32.ptr.0 + 1),
                ..valid.published_ends_u32
            },
            ..valid
        },
        Glm53DsaLatentAppendBuffers {
            transaction_overlay_bf16: GgmlIqBuffer {
                ptr: valid.source_bf16.ptr,
                ..valid.transaction_overlay_bf16
            },
            ..valid
        },
        Glm53DsaLatentAppendBuffers {
            source_bf16: GgmlIqBuffer {
                ptr: DevicePtr(u64::MAX - 1),
                ..valid.source_bf16
            },
            ..valid
        },
    ] {
        assert!(kernel.launch_stage(&gpu, plan, hostile, 0).is_err());
    }
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn valid_mock_launch_invalidates_publication_before_kernel_enqueue() {
    let gpu = MockGpuBackend::new();
    let kernel = Glm53DsaLatentAppendKernel::load(&gpu).unwrap();
    let plan =
        Glm53DsaLatentAppendPlan::new(2, 1, 4, 2, 3, 512, 9, Glm53DsaLatentAppendStorage::Bf16)
            .unwrap();
    let source = gpu.alloc(plan.source_bytes).unwrap();
    let overlay = gpu.alloc(plan.overlay_bytes).unwrap();
    let ends = gpu.alloc(plan.published_end_bytes).unwrap();
    let nonces = gpu.alloc(plan.published_nonce_bytes).unwrap();
    gpu.copy_h2d(&vec![0xff; plan.published_end_bytes], ends)
        .unwrap();
    gpu.copy_h2d(&vec![0xff; plan.published_nonce_bytes], nonces)
        .unwrap();
    kernel
        .launch_stage(
            &gpu,
            plan,
            Glm53DsaLatentAppendBuffers {
                source_bf16: GgmlIqBuffer {
                    ptr: source,
                    bytes: plan.source_bytes,
                },
                transaction_overlay_bf16: GgmlIqBuffer {
                    ptr: overlay,
                    bytes: plan.overlay_bytes,
                },
                published_ends_u32: GgmlIqBuffer {
                    ptr: ends,
                    bytes: plan.published_end_bytes,
                },
                published_nonces_u64: GgmlIqBuffer {
                    ptr: nonces,
                    bytes: plan.published_nonce_bytes,
                },
            },
            7,
        )
        .unwrap();
    assert_eq!(gpu.launch_count(), 1);
    assert_eq!(
        gpu.read_alloc(ends).unwrap(),
        vec![0; plan.published_end_bytes]
    );
    assert_eq!(
        gpu.read_alloc(nonces).unwrap(),
        vec![0; plan.published_nonce_bytes]
    );
    for ptr in [source, overlay, ends, nonces] {
        gpu.free(ptr).unwrap();
    }
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
    const HOST_ABI: &str = ".arg_ptr(buffers.source_bf16.ptr)\n            .arg_ptr(buffers.transaction_overlay_bf16.ptr)\n            .arg_ptr(buffers.published_ends_u32.ptr)\n            .arg_ptr(buffers.published_nonces_u64.ptr)\n            .arg_u32(plan.batch)\n            .arg_u32(plan.queries)\n            .arg_u32(plan.capacity)\n            .arg_u32(plan.start_position)\n            .arg_u32(plan.end_position)\n            .arg_u64(plan.transaction_nonce)";
    const CUDA_ABI: &str = "const __nv_bfloat16 * __restrict__ source,\n        __nv_bfloat16 * __restrict__ transaction_overlay,\n        unsigned int * __restrict__ published_ends,\n        unsigned long long * __restrict__ published_nonces,\n        unsigned int batch, unsigned int query_count,\n        unsigned int capacity, unsigned int start_position,\n        unsigned int end_position, unsigned long long transaction_nonce";
    host.contains(HOST_ABI) && cuda.contains(CUDA_ABI)
}

#[test]
fn ordered_host_cuda_abi_rejects_pointer_and_scalar_swaps() {
    assert!(abi_contract(HOST_SOURCE, CUDA_SOURCE));
    for (left, right) in [
        (
            ".arg_ptr(buffers.source_bf16.ptr)",
            ".arg_ptr(buffers.published_ends_u32.ptr)",
        ),
        (
            ".arg_ptr(buffers.transaction_overlay_bf16.ptr)",
            ".arg_ptr(buffers.published_nonces_u64.ptr)",
        ),
        (".arg_u32(plan.batch)", ".arg_u32(plan.queries)"),
        (".arg_u32(plan.capacity)", ".arg_u32(plan.end_position)"),
        (
            ".arg_u32(plan.start_position)",
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
            "const __nv_bfloat16 * __restrict__ source,",
            "unsigned int * __restrict__ published_ends,",
        ),
        (
            "__nv_bfloat16 * __restrict__ transaction_overlay,",
            "unsigned long long * __restrict__ published_nonces,",
        ),
        ("unsigned int batch", "unsigned int query_count"),
        ("unsigned int capacity", "unsigned int end_position"),
        (
            "unsigned int start_position",
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
fn source_contract_pins_fail_closed_bf16_copy_then_publish_chronology() {
    const REQUIRED: &[&str] = &[
        "gpu.memset_async(\n            buffers.published_nonces_u64.ptr",
        "gpu.memset_async(\n            buffers.published_ends_u32.ptr",
        "computed_end != end_position",
        "computed_end > capacity",
        "transaction_nonce == 0ULL",
        "for (unsigned long long element = lane; element < elements;",
        "transaction_overlay[base + element] = source[base + element];",
        "    }\n    __syncthreads();\n\n    if (lane == 0U) {",
        "published_ends[batch_index] = end_position;\n        __threadfence();\n        atomicExch(published_nonces + batch_index, transaction_nonce);",
    ];
    fn contract(host: &str, cuda: &str) -> bool {
        let nonce_clear = host.find(REQUIRED[0]);
        let end_clear = host.find(REQUIRED[1]);
        let launch = host.find("KernelLaunch::new(gpu, self.stage)");
        let copy = cuda.find(REQUIRED[6]);
        let barrier = cuda.find("__syncthreads();");
        let publish_end = cuda.find("published_ends[batch_index] = end_position;");
        let fence = cuda.find("__threadfence();");
        let publish_nonce =
            cuda.find("atomicExch(published_nonces + batch_index, transaction_nonce);");
        REQUIRED
            .iter()
            .all(|token| host.contains(token) || cuda.contains(token))
            && nonce_clear.is_some()
            && nonce_clear < end_clear
            && end_clear < launch
            && copy < barrier
            && barrier < publish_end
            && publish_end < fence
            && fence < publish_nonce
            && !cuda.contains("latent_cache")
            && !cuda.contains("__nv_fp8")
            && !cuda.contains("bfloat162float")
    }
    assert!(contract(HOST_SOURCE, CUDA_SOURCE));
    for token in REQUIRED {
        let host = HOST_SOURCE.replacen(token, "", 1);
        let cuda = CUDA_SOURCE.replacen(token, "", 1);
        assert!(!contract(&host, &cuda));
    }
    for (left, right) in [
        (
            "__syncthreads();",
            "published_ends[batch_index] = end_position;",
        ),
        (
            "__threadfence();",
            "atomicExch(published_nonces + batch_index, transaction_nonce);",
        ),
    ] {
        assert!(!contract(
            HOST_SOURCE,
            &swap_unique(CUDA_SOURCE, left, right)
        ));
    }
}
