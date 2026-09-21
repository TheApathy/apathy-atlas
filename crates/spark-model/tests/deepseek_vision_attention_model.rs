// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free integration contracts for initial-chunk Vision raw visibility.

const CUDA: &str =
    include_str!("../../../kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu");
const DISPATCH: &str = include_str!("../src/layers/qwen3_attention/prefill/cache_skip_v4.rs");

#[test]
fn vision_tc2_is_explicit_and_wired_for_both_compressed_and_raw_only_layers() {
    assert!(DISPATCH.contains("self.deepseek_vision_prefill_attn_k"));
    assert_eq!(
        DISPATCH
            .matches("if let Some((token_ids, vocab)) = vision_tokens")
            .count(),
        2
    );
    assert_eq!(
        DISPATCH
            .matches("launch.arg_ptr(token_ids).arg_u32(vocab)")
            .count(),
        2
    );
}

#[test]
fn only_the_tc2_raw_arm_has_conditional_expanded_bounds() {
    let tc2 = CUDA
        .split("extern \"C\" __global__ void prefill_attn_compressed_tc2(")
        .nth(1)
        .unwrap();
    assert!(tc2.contains("#ifdef ATLAS_DEEPSEEK_VISION_TC2"));
    assert!(tc2.contains("deepseek_vision_raw_bounds("));
    assert!(tc2.contains("vision_lo[r0]"));
    assert!(tc2.contains("vision_hi[r1]"));
    assert!(tc2.contains("comp_same, 0u, cvis0, 0u, cvis1"));
    assert!(tc2.contains("(qrow0 + 1u) / ratio"));
}
