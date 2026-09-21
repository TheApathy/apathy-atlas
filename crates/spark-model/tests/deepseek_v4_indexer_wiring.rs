// SPDX-License-Identifier: AGPL-3.0-only
//! Production wiring checks supplement the actual parser/WeightStore tests.
#[test]
fn indexer_admission_precedes_load_layer_device_effects() {
    let source = include_str!("../src/weight_loader/deepseek_v4/load_layers.rs");
    let admission = source
        .find("super::indexer::admit_indexer_weights(store, config)?;")
        .expect("indexer metadata preflight must be in actual load_all_layers");
    assert!(admission < source.find("let n = config.num_hidden_layers;").unwrap());
    assert!(admission < source.find("load_hc_f32(").unwrap());
    assert!(admission < source.find("dense_auto(").unwrap());
}

#[test]
fn native_indexer_parser_precedes_numeric_null_sanitization() {
    let source = include_str!("../../atlas-core/src/config/parsers/deepseek_v4.rs");
    let admission = source
        .find("DeepSeekV4IndexerConfig::parse_flat(&raw)?")
        .expect("native indexer config preflight missing");
    assert!(
        admission
            < source
                .find("if let Some(obj) = raw.as_object_mut()")
                .unwrap()
    );
    assert!(source.contains("config.deepseek_v4_indexer = deepseek_v4_indexer;"));
}
