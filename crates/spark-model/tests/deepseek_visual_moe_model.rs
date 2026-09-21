// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free source contracts for the actual DeepSeek Vision mixed router.

const ASSEMBLE: &str = include_str!("../src/weight_loader/deepseek_v4/assemble.rs");
const DECODE: &str = include_str!("../src/layers/moe/forward.rs");
const BATCHED: &str = include_str!("../src/layers/moe/forward_batched.rs");
const PREFILL: &str = include_str!("../src/layers/moe/forward_prefill.rs");
const VERIFY: &str = include_str!("../src/layers/moe/forward_km.rs");
const ROUTER: &str = include_str!("../src/layers/moe/visual_routing.rs");
const CUDA: &str =
    include_str!("../../../kernels/gb10/deepseek-v4-flash/nvfp4/deepseek_visual_route.cu");
const OPS: &str = include_str!("../src/layers/ops/deepseek_visual_route.rs");

#[test]
fn actual_vision_admission_requires_its_own_bias_and_validated_hash_table() {
    assert!(ASSEMBLE.contains("config.deepseek_vision.is_some()"));
    assert!(ASSEMBLE.contains("vision_moe::load_bias"));
    assert!(ASSEMBLE.contains("vision_moe::validate_hash_table"));
    assert!(ASSEMBLE.contains("set_deepseek_visual_routing"));
}

#[test]
fn vision_ids_are_checked_before_hash_lookup_with_no_clamp() {
    let bound = CUDA.find("tok >= vocab_size + 5u").unwrap();
    let image = CUDA.find("tid2eid != nullptr && !image").unwrap();
    let lookup = CUDA.find("tid2eid + (size_t)tok * TOP_K").unwrap();
    assert!(bound < image && image < lookup);
    assert!(CUDA.contains("expert < 0 || expert >= EXPERTS"));
    assert!(CUDA.contains("asm volatile(\"trap;\")"));
    assert!(!CUDA.contains("expert = 0"));
}

#[test]
fn selection_bias_never_weights_experts() {
    assert!(CUDA.contains("image ? visual_bias[tid] : text_bias[tid]"));
    assert!(CUDA.contains("selection[tid] = raw + bias"));
    assert!(CUDA.contains("selected[k] = score[selected_ids[k]]"));
    assert!(CUDA.contains("selected[k] /= sum"));
    assert!(CUDA.contains("selected[k] * scaling_factor"));
    assert!(CUDA.contains("sum > 1e-20f"));
}

#[test]
fn routing_abi_is_bounded_and_missing_capability_does_not_launch() {
    assert!(ROUTER.contains("return Ok(false)"));
    assert!(ROUTER.contains("config.num_experts == 256 && config.num_experts_per_tok == 6"));
    assert!(ROUTER.contains("DeepSeek Vision MoE requires token IDs for every pass"));
    let args = [
        ".arg_ptr(logits)",
        ".arg_ptr(table)",
        ".arg_ptr(token_ids)",
        ".arg_ptr(text_bias)",
        ".arg_ptr(visual_bias)",
        ".arg_ptr(indices)",
        ".arg_ptr(weights)",
        ".arg_u32(vocab)",
        ".arg_f32(scale)",
    ];
    let offsets: Vec<_> = args.iter().map(|s| OPS.find(s).unwrap()).collect();
    assert!(offsets.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn alternative_prefill_and_verify_routes_do_not_bypass_visual_bias() {
    for source in [
        include_str!("../src/layers/moe/forward_k2.rs"),
        include_str!("../src/layers/moe/forward_k3.rs"),
        include_str!("../src/layers/moe/forward_kn.rs"),
        include_str!("../src/layers/moe/forward_token_major.rs"),
        include_str!("../src/layers/moe/forward_atomic_c4.rs"),
        include_str!("../src/layers/moe/forward_prefill_bf16.rs"),
        include_str!("../src/layers/moe/forward_prefill_fp8.rs"),
    ] {
        assert!(
            source.find("self.route_deepseek_visual(").unwrap()
                < source
                    .find("if let Some(bias) = self.correction_bias_dev")
                    .unwrap()
        );
    }
}

#[test]
fn every_hash_dispatch_is_preceded_by_visual_routing() {
    for (name, source) in [
        ("decode", DECODE),
        ("batched", BATCHED),
        ("grouped prefill", PREFILL),
        ("wide verify", VERIFY),
    ] {
        let mixed = source.find("self.route_deepseek_visual(").expect(name);
        let hash = source
            .find("if let Some(tid2eid) = self.tid2eid_dev")
            .expect(name);
        assert!(
            mixed < hash,
            "{name}: sentinels must not index the hash table"
        );
    }
}
