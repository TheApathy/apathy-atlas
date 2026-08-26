// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;

#[test]
fn test_buffer_sizes_qwen3() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let sizes = BufferSizes::from_config(&cfg, 1, 4096, 16);

    // hidden_states: 1 * 2048 * 2 = 4096 (BF16, 2 bytes/elem).
    // (Was FP32 = 8192 in earlier prototypes; NVFP4 path keeps the
    // residual stream in BF16, halving the buffer size.)
    assert_eq!(sizes.hidden_states, 4096);
    // qkv: 1 * (16*2 + 2*2) * 256 * 2 = 1 * 36 * 256 * 2 = 18432
    // Q+gate: 16*2*256, K: 2*256, V: 2*256
    assert_eq!(sizes.qkv_output, 18432);
    // attn: 1 * 16 * 256 * 2 = 8192
    assert_eq!(sizes.attn_output, 8192);
    // gate: 1 * 512 * 2 = 1024
    assert_eq!(sizes.gate_logits, 1024);
    // logits: 1 * 151936 * 2 = 303872
    assert_eq!(sizes.logits, 303872);
    // ssm_qkvz: 1 * 12288 * 2 = 24576
    // Q(16*128) + K(16*128) + V(32*128) + Z(32*128) = 12288
    assert_eq!(sizes.ssm_qkvz, 24576);
    // ssm_ba: max(1 * 64 * 2, 256) = 256 (minimum allocation)
    assert_eq!(sizes.ssm_ba, 256);
    // ssm_deinterleaved: same as ssm_qkvz = 24576
    assert_eq!(sizes.ssm_deinterleaved, 24576);
    // ssm_gates: 1 * 32 * 2 * 4 = 256 (FP32 gate + beta, scaled by M)
    assert_eq!(sizes.ssm_gates, 256);
}

#[test]
fn test_buffer_arena_alloc() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let arena = BufferArena::new(&cfg, 128, 4096, 16, &gpu).unwrap();

    assert!(!arena.hidden_states().is_null());
    assert!(!arena.logits().is_null());
    assert_eq!(arena.max_batch_tokens(), 128);
    // 18 allocations for 18 buffers (12 data + 1 scratch + 3 expert + 2 splitk).
    // Bump from 17 reflects an added split-K accumulator buffer.
    assert_eq!(gpu.alloc_count(), 18);
}

#[test]
fn test_buffer_sizes_scale_with_batch() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let s1 = BufferSizes::from_config(&cfg, 1, 4096, 16);
    let s128 = BufferSizes::from_config(&cfg, 128, 4096, 16);
    assert_eq!(s128.hidden_states, s1.hidden_states * 128);
    // logits is capped at 16 tokens; FP32 sampling buffer (4 bytes/elem),
    // so s128.logits = 16 * vocab * 4 (not 128× the unbatched value).
    assert_eq!(s128.logits, 16 * cfg.vocab_size * 4);
}

#[test]
fn qwen4_exp_expands_only_the_persistent_residual_buffers() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.hidden_size = 2560;
    cfg.hc_count = 4;
    let sizes = BufferSizes::from_config(&cfg, 3, 4096, 16);

    assert_eq!(sizes.hidden_states, 3 * 4 * 2560 * 2);
    assert_eq!(sizes.residual, 3 * 4 * 2560 * 2);
    assert_eq!(sizes.norm_output, 3 * 2560 * 2);
    assert_eq!(sizes.moe_output, 3 * 2560 * 2);
}

/// `ssm_qkvz` is the Q staging buffer for the multi-seq attention QKV routes,
/// at the full gated projection width. On Qwen3.8-27B that width (2*24*256 =
/// 12288) happens to equal `ssm_qkvz_size()`, so the fit was a coincidence of
/// this architecture rather than something the sizing guaranteed. Pin the
/// invariant directly so a future model with a wider gated q_proj_dim fails
/// here instead of overrunning the arena at runtime.
#[test]
fn ssm_qkvz_covers_gated_attention_q_staging() {
    for m in [1usize, 17, 128] {
        let cfg = ModelConfig::qwen3_next_80b_nvfp4();
        let sizes = BufferSizes::from_config(&cfg, m, 4096, 16);
        let q_proj_mul = if cfg.attn_gated { 2 } else { 1 };
        let q_staging = m * cfg.num_attention_heads * q_proj_mul * cfg.head_dim * 2;
        assert!(
            sizes.ssm_qkvz >= q_staging,
            "m={m}: ssm_qkvz={} < gated Q staging={q_staging}",
            sizes.ssm_qkvz
        );
        // The same buffer still has to hold the prefill K+V staging.
        let kv_staging = m * 2 * cfg.num_key_value_heads * cfg.head_dim * 2;
        assert!(
            sizes.ssm_qkvz >= kv_staging,
            "m={m}: ssm_qkvz < K+V staging"
        );
    }
}
