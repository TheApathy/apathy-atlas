// SPDX-License-Identifier: AGPL-3.0-only

const NOISE_PASS: &str = include_str!("../src/layers/dflash_head/noise_pass.rs");
const FORWARD_BLOCK_LAYER: &str = include_str!("../src/layers/dflash_head/forward_block_layer.rs");

fn successful_call_end<'a>(source: &'a str, call: &str) -> (&'a str, usize) {
    let call_start = source.find(call).expect("fallible call missing");
    let call_tail = &source[call_start..];
    let relative_end = call_tail.find(")?;").expect("fallible call must use ?") + 3;
    (source, call_start + relative_end)
}

#[test]
fn selector_walk_receipt_follows_successful_kernel_completion() {
    let selector_region = NOISE_PASS
        .split_once("// Greedy walk seeded at the last verified token")
        .expect("selector walk region missing")
        .1;
    let (_, call_end) = successful_call_end(selector_region, "ops::dflash2_selector_walk(");
    let receipt = selector_region
        .find("DFLASH2_EXEC: selector_walk RAN")
        .expect("selector execution receipt missing");
    assert!(receipt > call_end, "selector receipt must be success-only");
}

#[test]
fn nvfp4_attention_conv_receipt_follows_successful_prepare_completion() {
    let nvfp4_region = FORWARD_BLOCK_LAYER
        .split_once("// 3a'. DFlash 2 attention-conv `prepare` (NVFP4 path).")
        .expect("NVFP4 attention-conv region missing")
        .1;
    let (_, call_end) = successful_call_end(nvfp4_region, "self.dflash2_conv_prepare(");
    let receipt = nvfp4_region
        .find("DFLASH2_EXEC: nvfp4 attention_conv prepare RAN")
        .expect("NVFP4 attention-conv execution receipt missing");
    assert!(
        receipt > call_end,
        "NVFP4 conv receipt must be success-only"
    );
}
