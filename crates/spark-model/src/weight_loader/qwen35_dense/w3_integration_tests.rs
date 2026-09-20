// SPDX-License-Identifier: AGPL-3.0-only

const LOADER: &str = include_str!("../qwen35_dense.rs");
const FFN: &str = include_str!("ffn.rs");

fn ordered(source: &str, needles: &[&str]) {
    let mut cursor = 0;
    for needle in needles {
        let offset = source[cursor..]
            .find(needle)
            .unwrap_or_else(|| panic!("missing lifecycle token: {needle}"));
        cursor += offset + needle.len();
    }
}

fn function_body<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing function start: {start}"))
        .1
        .split_once(end)
        .unwrap_or_else(|| panic!("missing function end: {end}"))
        .0
}

#[test]
fn session_brackets_all_loader_side_effects_and_publication() {
    let load_layers = function_body(LOADER, "fn load_layers(", "fn load_embedding(");
    ordered(
        load_layers,
        &[
            "ffn::prepare_w3_session(config, gpu)?",
            "super::transform_cache::init(",
            "for (i, lt) in layer_types.iter().enumerate()",
            "ffn::finish_w3_session(w3_session)?",
            "super::transform_cache::finish();",
            "Ok(layers)",
        ],
    );
    assert!(!load_layers.contains("maybe_load_w3_ffn"));
}

#[test]
fn request_is_parsed_once_and_prevalidated_for_every_layer() {
    let prepare = function_body(
        FFN,
        "pub(super) fn prepare_w3_session(",
        "pub(super) fn finish_w3_session(",
    );
    assert_eq!(prepare.matches("W3SidecarRequest::from_env(").count(), 1);
    ordered(
        prepare,
        &[
            "W3SidecarRequest::from_env(config.num_hidden_layers)?",
            "return Ok(None)",
            "gpu.kernel(\"w3a16_gemm\", \"w3a16_gemm_t_m32_n64\")?",
            "(0..config.num_hidden_layers)",
            ".map(|layer| config.layer_prefix(layer))",
            "W3SidecarSession::prepare(",
        ],
    );
}

#[test]
fn requested_layer_upload_is_installed_then_accounted() {
    let load = function_body(
        FFN,
        "pub(super) fn load(",
        "pub(super) fn prepare_w3_session(",
    );
    ordered(
        load,
        &[
            "session.upload_layer(layer, self.gpu)?",
            "ffn_layer.set_w3_weights(gemv, gemm_t);",
            "session.mark_installed(layer)?;",
        ],
    );
    assert!(!FFN.contains("maybe_load_w3_ffn"));
    assert!(!FFN.contains("Fail-open"));
    assert!(!FFN.contains("fail-open"));
}

#[test]
fn finish_rechecks_exact_requested_validated_uploaded_installed_census() {
    let finish = FFN
        .split_once("pub(super) fn finish_w3_session(")
        .expect("missing finish function")
        .1;
    ordered(
        finish,
        &[
            "let receipt = session.finish()?;",
            "receipt.requested_layers == receipt.validated_layers",
            "receipt.requested_layers == receipt.uploaded_layers",
            "receipt.requested_layers == receipt.installed_layers",
            "receipt.requested_count == receipt.validated_count",
            "receipt.requested_count == receipt.uploaded_count",
            "receipt.requested_count == receipt.installed_count",
        ],
    );
}
