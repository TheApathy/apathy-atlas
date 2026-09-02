// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

fn plan() -> Glm53Dflash2AttentionPlan {
    Glm53Dflash2AttentionPlan::new(2, 4, 10, 2047, 10_000, 258, 32, 8, 128, 1e-5, 1e4, 2048, 16)
        .unwrap()
}

fn buffers(p: Glm53Dflash2AttentionPlan) -> Glm53Dflash2AttentionBuffers {
    let mut address = 0x10_0000u64;
    let mut next = |bytes| {
        let result = GgmlIqBuffer {
            ptr: DevicePtr(address),
            bytes,
        };
        address += u64::try_from(bytes).unwrap().next_multiple_of(0x10_0000);
        result
    };
    Glm53Dflash2AttentionBuffers {
        q_noise_bf16: next(p.q_bytes),
        target_tail_k_bf16: next(p.target_kv_bytes),
        target_tail_v_bf16: next(p.target_kv_bytes),
        noise_k_bf16: next(p.noise_kv_bytes),
        noise_v_bf16: next(p.noise_kv_bytes),
        output_bf16: next(p.q_bytes),
        q_norm_weight_bf16: next(p.norm_weight_bytes),
        k_norm_weight_bf16: next(p.norm_weight_bytes),
        target_slots_i64: next(p.target_slots_bytes),
        noise_slots_i64: next(p.noise_slots_bytes),
        block_tables_u32: next(p.block_tables_bytes),
        k_cache_bf16: next(p.cache_pool_bytes),
        v_cache_bf16: next(p.cache_pool_bytes),
    }
}

#[test]
fn window_tail_absolute_boundary_and_cache_transitions_are_pinned() {
    let p = plan();
    assert_eq!((p.kept_past_tokens, p.past_drop_tokens), (2037, 10));
    assert_eq!(
        (p.local_context_tokens, p.provisional_cache_len),
        (2047, 2051)
    );
    assert_eq!((p.success_cache_len(), p.failure_cache_len()), (2047, 2037));
    for q in 0..p.noise_tokens {
        for k in 0..p.provisional_cache_len {
            let q_pos = p.local_context_tokens + q;
            let visible = k > q_pos || q_pos - k < SLIDING_WINDOW;
            assert_eq!(visible, !(q_pos >= k && q_pos - k >= 2048));
        }
    }
    let edge = Glm53Dflash2AttentionPlan::new(
        1, 1, 1_048_575, 0, 1_048_575, 129, 32, 8, 128, 1e-5, 1e4, 2048, 16,
    )
    .unwrap();
    assert_eq!(edge.source_context_skip_tokens, 1_046_528);
    assert_eq!(edge.absolute_context_end, 1_048_575);
    assert!(
        Glm53Dflash2AttentionPlan::new(
            1, 2, 1, 0, 1_048_575, 129, 32, 8, 128, 1e-5, 1e4, 2048, 16,
        )
        .is_err()
    );
}

#[test]
fn faults_are_effect_free_and_each_batch_has_an_independent_table() {
    let gpu = MockGpuBackend::new();
    let kernels = Glm53Dflash2AttentionKernels::load(&gpu).unwrap();
    let p = plan();
    let b = buffers(p);
    assert!(
        kernels
            .execute(&gpu, Glm53Dflash2AttentionPlan { q_bytes: 2, ..p }, b, 3,)
            .is_err()
    );
    assert!(
        kernels
            .execute(
                &gpu,
                p,
                Glm53Dflash2AttentionBuffers {
                    output_bf16: GgmlIqBuffer {
                        ptr: b.q_noise_bf16.ptr,
                        bytes: p.q_bytes,
                    },
                    ..b
                },
                3,
            )
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
    kernels.execute(&gpu, p, b, 3).unwrap();
    assert_eq!(gpu.launch_count(), 12);
}
