// SPDX-License-Identifier: AGPL-3.0-only
//! Selected-block CLI admission must precede backend and ordinary repeats.
const GPU: &str = include_str!("../examples/deepseek_vision_probe/gpu.rs");

#[test]
fn explicit_detail_block_is_rejected_before_probe_output_or_gpu_effects() {
    let source: String = GPU.split_whitespace().collect();
    let selected = source
        .find("ifletSome(block)=detail_block")
        .expect("explicit block preflight missing");
    let bound = source
        .find("block<config.num_hidden_layers")
        .expect("preflight must use admitted configuration depth");
    assert!(source[..selected].contains(".deepseek_vision.as_ref()"));
    assert!(source[..selected].contains("inspect(model)?"));
    assert!(selected < bound);
    for effect in [
        "std::fs::create_dir(out)",
        "AtlasCudaBackend::new(",
        "loader.load(",
        "encoder.forward(",
    ] {
        assert!(
            bound < source.find(effect).expect("existing probe effect missing"),
            "{effect} ran before selected-block validation"
        );
    }
}
