// SPDX-License-Identifier: AGPL-3.0-only

use deepseek_vision_p1::contract::{
    Arg, Grid, ReferenceMode, angles_launch, check_bf16, check_stage, fc2_launch, rope_launch,
    validate_reference,
};
use serde_json::json;

#[test]
fn only_the_retained_corpus_and_its_exact_shapes_are_admitted() {
    for (h, w, patches, rows) in [(3, 3, 9, 1), (4, 5, 20, 4), (54, 54, 2916, 324)] {
        let grid = Grid::new(h, w).unwrap();
        assert_eq!(grid.patches(), patches);
        assert_eq!(grid.aligned_rows(), rows);
        assert_eq!(grid.name(), format!("grid-{h}x{w}"));
    }
    for (h, w) in [(0, 3), (3, 0), (1, 1), (3, 4), (usize::MAX, 3), (54, 55)] {
        assert!(Grid::new(h, w).is_err());
    }
}

#[test]
fn bf16_inputs_reject_nonfinite_odd_or_empty_payloads() {
    check_bf16(&[0x80, 0x3f, 0x00, 0x80]).unwrap();
    for raw in [
        vec![],
        vec![1],
        vec![0x80, 0x7f],
        vec![0x81, 0x7f],
        vec![0x80, 0xff],
    ] {
        assert!(check_bf16(&raw).is_err());
    }
}

#[test]
fn exact_stage_schema_rejects_dtype_shape_bounds_and_paths() {
    let valid = json!({"name":"block-00-qkv","file":"block-00-qkv.bf16",
        "dtype":"bf16","shape":[20,3072],"bytes":122880,"sha256":"a".repeat(64)});
    check_stage(&valid, "block-00-qkv", [20, 3072]).unwrap();
    for (field, value) in [
        ("file", json!("../block-00-qkv.bf16")),
        ("dtype", json!("f32")),
        ("shape", json!([20.0, 3072])),
        ("shape", json!([20, 3071])),
        ("bytes", json!(122881)),
        ("bytes", json!(122880.0)),
        ("sha256", json!("a".repeat(63))),
        ("name", json!("query")),
    ] {
        let mut bad = valid.clone();
        bad[field] = value;
        assert!(
            check_stage(&bad, "block-00-qkv", [20, 3072]).is_err(),
            "{field}"
        );
    }
}

fn reference() -> serde_json::Value {
    json!({"diagnostic_only":true,
        "scope":"independent operators on shared native inputs, not chained-reference qualification",
        "official_sha256":"a4f089069310398d42ca17fd4496cec82da64cbbfde9b0230679ce1537cc0bb1",
        "payload_sha256":"fc9d54d790826fb7d8f60b06c8ae17a45b3cade7bfee44e7ae3f74dc948922b0",
        "torch_version":"2.10.0+cu130","bf16_reduced_precision_reduction":true})
}

#[test]
fn official_reference_identity_and_precision_cannot_be_silently_changed() {
    let original = reference();
    validate_reference(&original, ReferenceMode::Default).unwrap();
    assert!(validate_reference(&original, ReferenceMode::Full).is_err());
    for (field, value) in [
        ("official_sha256", json!("0".repeat(64))),
        ("payload_sha256", json!("1".repeat(64))),
        ("torch_version", json!("2.11.0")),
        ("bf16_reduced_precision_reduction", json!(1)),
        ("diagnostic_only", json!(false)),
        ("scope", json!("chained-reference qualification")),
    ] {
        let mut bad = original.clone();
        bad[field] = value;
        assert!(
            validate_reference(&bad, ReferenceMode::Default).is_err(),
            "{field}"
        );
    }
    let mut full = original;
    full["bf16_reduction"] = json!("full");
    full["bf16_reduced_precision_reduction"] = json!(false);
    validate_reference(&full, ReferenceMode::Full).unwrap();
}

#[test]
fn rope_abi_and_candidate_geometry_cover_all_retained_rows() {
    let grid = Grid::new(54, 54).unwrap();
    let launch = rope_launch(grid, [256, 512, 768, 1024, 1280]).unwrap();
    assert_eq!(launch.grid, [5832, 1, 1]);
    assert_eq!(launch.block, [256, 1, 1]);
    assert_eq!(
        launch.args,
        vec![
            Arg::Ptr(256),
            Arg::Ptr(512),
            Arg::Ptr(768),
            Arg::Ptr(1024),
            Arg::Ptr(1280),
            Arg::U32(2916),
            Arg::U32(16),
            Arg::U32(64)
        ]
    );
    let angles = angles_launch(grid, 1280).unwrap();
    assert_eq!(angles.grid, [365, 1, 1]);
    assert_eq!(
        angles.args,
        vec![Arg::Ptr(1280), Arg::U32(54), Arg::U32(54)]
    );
    for slot in 0..5 {
        let mut ptrs = [256, 512, 768, 1024, 1280];
        ptrs[slot] = 0;
        assert!(rope_launch(grid, ptrs).is_err());
    }
    assert!(angles_launch(grid, 0).is_err());
}

#[test]
fn fc2_uses_original_layout_and_only_bias_pointer_can_be_null() {
    let launch = fc2_launch(256, 512, 768).unwrap();
    assert_eq!(launch.grid, [32, 1, 1]);
    assert_eq!(launch.block, [128, 1, 1]);
    assert_eq!(
        launch.args,
        vec![
            Arg::Ptr(256),
            Arg::Ptr(512),
            Arg::Ptr(0),
            Arg::Ptr(768),
            Arg::U32(20),
            Arg::U32(1024),
            Arg::U32(2816),
            Arg::U32(1024)
        ]
    );
    assert!(fc2_launch(0, 512, 768).is_err());
    assert!(fc2_launch(256, 0, 768).is_err());
    assert!(fc2_launch(256, 512, 0).is_err());
}
