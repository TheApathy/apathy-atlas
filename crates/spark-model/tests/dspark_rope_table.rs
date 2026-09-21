// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only coverage of the exact production DSpark table builder.

#[path = "../src/layers/dspark_head/rope_table.rs"]
mod rope_table;

use atlas_core::config::{ModelConfig, parse_config};
use rope_table::build_rope_table;

fn config() -> ModelConfig {
    parse_config(
        r#"{
        "model_type":"deepseek_v4", "hidden_size":4096,
        "num_hidden_layers":43, "num_attention_heads":64,
        "num_key_value_heads":1, "head_dim":512, "vocab_size":129280,
        "o_lora_rank":1024, "q_lora_rank":1024,
        "rope_theta":10000, "compress_rope_theta":160000,
        "dspark_block_size":5
    }"#,
    )
    .unwrap()
}

#[test]
fn table_uses_declared_base_not_compressed_target_theta() {
    let config = config();
    assert_eq!(config.rope_theta, 160000.0);
    let table = build_rope_table(&config, 101, 64).unwrap();
    assert_eq!(table.len(), 101 * 64);
    for j in 0..32 {
        assert_eq!(table[j * 2], 1.0);
        assert_eq!(table[j * 2 + 1], 0.0);
    }
    // At dimension32 the exponent is1/2: angle100/sqrt(10000)=1.
    let slot = (100 * 32 + 16) * 2;
    assert!((table[slot] - 1.0f32.cos()).abs() < 1e-7);
    assert!((table[slot + 1] - 1.0f32.sin()).abs() < 1e-7);
    assert!((table[slot] - 0.25f32.cos()).abs() > 0.4);
}

#[test]
fn compressed_theta_changes_cannot_change_draft_table() {
    let original = config();
    let mut changed = original.clone();
    changed.rope_theta = 40000.0;
    assert_eq!(
        build_rope_table(&original, 129, 64).unwrap(),
        build_rope_table(&changed, 129, 64).unwrap()
    );
    changed.deepseek_main_rope_theta = Some(25000.0);
    assert_ne!(
        build_rope_table(&original, 129, 64).unwrap(),
        build_rope_table(&changed, 129, 64).unwrap()
    );
}

#[test]
fn constructor_boundary_rejects_absent_nonfinite_and_nonpositive_base() {
    for theta in [
        None,
        Some(0.0),
        Some(-1.0),
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
    ] {
        let mut config = config();
        config.deepseek_main_rope_theta = theta;
        assert!(build_rope_table(&config, 8, 64).is_err());
    }
}

#[test]
fn invalid_geometry_and_overflow_fail_before_allocation() {
    let config = config();
    for (rows, dim) in [(0, 64), (8, 0), (8, 63), (usize::MAX, 64), (8, usize::MAX)] {
        assert!(build_rope_table(&config, rows, dim).is_err());
    }
}

#[test]
fn production_constructor_uses_checked_builder_before_device_allocation() {
    let source = include_str!("../src/layers/dspark_head.rs");
    let constructor = source
        .split("pub fn new(")
        .nth(1)
        .unwrap()
        .split("pub fn set_capture(")
        .next()
        .unwrap();
    let plan = constructor
        .find("rope_table::build_rope_table(target_config, max_seq_len, ROPE_DIM as usize)")
        .unwrap();
    assert!(plan < constructor.find("gpu.alloc(").unwrap());
    assert!(!constructor.contains("target_config.rope_theta"));
}
