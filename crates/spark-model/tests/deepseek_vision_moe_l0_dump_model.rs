// SPDX-License-Identifier: AGPL-3.0-only

fn source(path: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap()
}

#[test]
fn observed_moe_brackets_real_shared_and_routed_work_without_global_context() {
    let hc = source("src/layers/qwen3_attention/trait_impl/prefill_inner.rs");
    assert!(hc.contains("MoeCapture::begin("));
    assert!(hc.contains("moe_capture.as_mut()"));
    let ffn = source("src/layers/mod.rs");
    assert!(ffn.contains("pub(crate) fn forward_prefill_observed("));
    let moe = source("src/layers/moe/forward_prefill.rs");
    assert!(
        source("src/layers/moe/forward_prefill_phase.rs")
            .contains("self.forward_prefill_observed(input, num_tokens, ctx, stream, None)")
    );
    assert!(
        moe.find("Stage::SharedAfterRouted").unwrap()
            > moe.find("ops::moe_unpermute_reduce_indexed(").unwrap()
    );
    assert!(moe.find("Stage::SharedAfterRouted").unwrap() < moe.find("Stage::RoutedOnly").unwrap());
    assert!(moe.find("Stage::RoutedOnly").unwrap() < moe.find("ops::moe_batched_blend(").unwrap());
    assert!(moe.find("Stage::MoeBlended").unwrap() > moe.find("ops::moe_batched_blend(").unwrap());
    assert!(!source("src/layer.rs").contains("MoeCapture"));
}

#[test]
fn native_observer_reads_real_weights_before_projection_and_exact_intermediates() {
    let native = source("src/layers/moe/native_shared_fp8.rs");
    let start = native.find("capture.native_weights(").unwrap();
    let gate = native
        .find("project(input, state.weights.gate_proj, gate)?;")
        .unwrap();
    let up = native
        .find("project(input, state.weights.up_proj, up)?;")
        .unwrap();
    let act = native.find("ops::silu_mul(ctx.gpu").unwrap();
    let down = native
        .find("project(gate, state.weights.down_proj, down)?;")
        .unwrap();
    assert!(start < gate && gate < native.find("Stage::SharedGate").unwrap());
    assert!(up < native.find("Stage::SharedUp").unwrap() && up < act);
    assert!(act < native.find("Stage::SharedActivation").unwrap());
    assert!(down < native.find("Stage::SharedDown").unwrap());
    let capture = source("src/layers/moe/vision_l0_dump.rs");
    assert!(!capture.contains("thread_local!"));
    assert!(!capture.contains("gpu.alloc("));
    assert!(!capture.contains("copy_h2d"));
}
