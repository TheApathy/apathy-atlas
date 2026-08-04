// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn v4_decode_does_not_info_log_per_layer_per_token() {
    let source = include_str!("../src/layers/qwen3_attention/decode/run_paged_decode.rs");
    let message = "V4-Flash MLA decode (FP8)";
    let message_offset = source.find(message).expect("V4 decode audit message");
    let prefix = &source[..message_offset];
    let macro_offset = prefix
        .rfind("tracing::")
        .expect("tracing macro before V4 decode audit message");

    assert!(
        source[macro_offset..message_offset].starts_with("tracing::trace!"),
        "the per-layer, per-token V4 decode audit must stay below INFO"
    );
}
