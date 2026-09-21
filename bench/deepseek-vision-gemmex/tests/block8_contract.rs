// SPDX-License-Identifier: AGPL-3.0-only
//! RED for the actual bounded block-8 adapter, not block-0 teacher admission.
#[path = "../src/block8/contract.rs"]
mod contract;
#[path = "../src/contract.rs"]
mod gemm_geometry;

use contract::{DeviceSpan, Fc1Plan, fc1_weight_span, owned_device_bytes, validate_capture};
use deepseek_vision_p1::{contract::Arg, driver::guarded_bytes, pins};
use serde_json::{Value, json};

const INPUT: &str = "bae82ae227edfcdf0ec81150cc2f7fc01371fd2da19bea87948c73e773480606";
const NATIVE: &str = "955e1d6608e55e175445f6e215a5c2ddb0058be26f10374fef5a1c8dddd11ef3";
const WEIGHT: &str = "vision.blocks.8.mlp.w1.weight";

fn capture() -> (Value, Value) {
    let top = json!({
        "checkpoint":{"model_revision":pins::MODEL_REV,"config_sha256":pins::CONFIG_SHA,
            "index_sha256":pins::INDEX_SHA,"selected_tensors":267,"selected_bytes":932786176},
        "binary_sha256":"ba5d7177ecd6d9955fd6c292b8a4829420d0c78804f375c3dd1a09ff568ce9d9",
        "diagnostic_stage_capture":true,"selected_detail_block":8,"scratch_bytes":206275584,
        "cases":[
            {"name":"grid-3x3","grid_h":3,"grid_w":3,"patches":9,"aligned_rows":1,
                "hidden_size":4096,"repeat_byte_equal":true},
            {"name":"grid-4x5","grid_h":4,"grid_w":5,"patches":20,"aligned_rows":4,
                "hidden_size":4096,"repeat_byte_equal":true,
                "input_sha256":"319cc2fac27587ad2e9e2ea03e2c9a6b13b481bdd183b9046735e40b70944e2f",
                "output_sha256":"27b00f915f5f6c34573dfccaff012d03c596227c77c55d5559f565d401602f9d"},
            {"name":"grid-54x54","grid_h":54,"grid_w":54,"patches":2916,"aligned_rows":324,
                "hidden_size":4096,"repeat_byte_equal":true}]});
    let stages = json!({"grid":[4,5],"selected_detail_block":8,"output_byte_equal":true,
        "stages":[
            {"name":"block-08-norm2","file":"block-08-norm2.bf16","shape":[20,1024],
                "dtype":"bf16","bytes":40960,"sha256":INPUT},
            {"name":"block-08-fc1","file":"block-08-fc1.bf16","shape":[20,5632],
                "dtype":"bf16","bytes":225280,"sha256":NATIVE}]});
    (top, stages)
}

#[test]
fn p14_capture_pins_are_native_not_invented_teacher_receipts() {
    assert_eq!(contract::INPUT_SHA, INPUT);
    assert_eq!(contract::NATIVE_SHA, NATIVE);
    assert_eq!(contract::WEIGHT, WEIGHT);
    assert_eq!(
        contract::MANIFEST_SHA,
        "06a0d7b7dc4f91c36b8657a846b1331031efb4a1c3533da407ab8e26da0a1870"
    );
    assert_eq!(
        contract::STAGES_SHA,
        "054ba4f31f606643c5ffa9fdd3e25a49e3237ccc6597eff3d6d36a07fa300e48"
    );
    let (top, stages) = capture();
    let before = (top.clone(), stages.clone());
    let receipt = validate_capture(&top, &stages).unwrap();
    assert_eq!(receipt.input_sha256, INPUT);
    assert_eq!(receipt.native_sha256, NATIVE);
    assert_eq!((top, stages), before);
}

#[test]
fn capture_requires_selected_block_eight_same_model_and_complete_unique_three_grid_corpus() {
    let (top, stages) = capture();
    for (key, value) in [
        ("selected_detail_block", json!(0)),
        ("selected_detail_block", json!(8.5)),
        ("selected_detail_block", Value::Null),
        ("diagnostic_stage_capture", json!(false)),
        ("binary_sha256", json!("a".repeat(64))),
        ("scratch_bytes", json!(206275583)),
    ] {
        let mut changed = top.clone();
        changed[key] = value;
        assert!(
            validate_capture(&changed, &stages).is_err(),
            "accepted {key}"
        );
    }
    for key in [
        "model_revision",
        "config_sha256",
        "index_sha256",
        "selected_bytes",
        "selected_tensors",
    ] {
        let mut changed = top.clone();
        changed["checkpoint"][key] = Value::Null;
        assert!(
            validate_capture(&changed, &stages).is_err(),
            "accepted {key}"
        );
    }
    let mut changed = top.clone();
    changed["cases"].as_array_mut().unwrap().pop();
    assert!(validate_capture(&changed, &stages).is_err());
    let mut changed = top.clone();
    changed["cases"][0] = top["cases"][1].clone();
    assert!(validate_capture(&changed, &stages).is_err());
    for (key, value) in [
        ("repeat_byte_equal", json!(false)),
        ("hidden_size", json!(4095)),
        ("grid_h", json!(5)),
        ("patches", json!(19)),
        ("aligned_rows", json!(1)),
        ("input_sha256", json!("a".repeat(64))),
        ("output_sha256", json!("a".repeat(64))),
    ] {
        let mut changed = top.clone();
        changed["cases"][1][key] = value;
        assert!(
            validate_capture(&changed, &stages).is_err(),
            "accepted {key}"
        );
    }
}

#[test]
fn missing_misbound_old_block_zero_or_malformed_stage_cannot_enter_adapter() {
    let (top, stages) = capture();
    for (key, value) in [
        ("selected_detail_block", json!(0)),
        ("grid", json!([5, 4])),
        ("output_byte_equal", json!(false)),
    ] {
        let mut changed = stages.clone();
        changed[key] = value;
        assert!(validate_capture(&top, &changed).is_err());
    }
    for (key, value) in [
        ("name", json!("block-00-norm2")),
        ("file", json!("../block-08-norm2.bf16")),
        ("shape", json!([20, 1023])),
        ("dtype", json!("f32")),
        ("bytes", json!(40962)),
        ("sha256", json!(gemm_geometry::INPUT_SHA)),
    ] {
        let mut changed = stages.clone();
        changed["stages"][0][key] = value;
        assert!(validate_capture(&top, &changed).is_err(), "accepted {key}");
    }
    let mut changed = stages.clone();
    changed["stages"][1]["sha256"] = json!(gemm_geometry::NATIVE_SHA);
    assert!(validate_capture(&top, &changed).is_err());
    let mut changed = stages.clone();
    changed["stages"].as_array_mut().unwrap().pop();
    assert!(validate_capture(&top, &changed).is_err());
    let mut changed = stages.clone();
    changed["stages"]
        .as_array_mut()
        .unwrap()
        .push(stages["stages"][0].clone());
    assert!(validate_capture(&top, &changed).is_err());
    let mut changed = stages.clone();
    changed["stages"] = json!(vec![stages["stages"][0].clone(); 65]);
    assert!(validate_capture(&top, &changed).is_err());
}

#[test]
fn selected_weight_uses_original_block_eight_key_and_checked_native_bf16_span() {
    // Actual shard10 layout: header1,326,456 + prefix8 + payload-relative offset.
    let lo = 7_557_430_244u64;
    let header = json!({WEIGHT:{"dtype":"BF16","shape":[5632,1024],
        "data_offsets":[lo,lo+11_534_336]}});
    assert_eq!(
        fc1_weight_span(&header, 1_326_456, 7_602_973_028).unwrap(),
        (7_558_756_708, 11_534_336)
    );
    for (key, value) in [
        ("dtype", json!("F16")),
        ("shape", json!([1024, 5632])),
        ("shape", json!([5632, 1024.5])),
        ("data_offsets", json!([lo + 1, lo + 11_534_336])),
        ("data_offsets", json!([u64::MAX - 11_534_336, u64::MAX])),
    ] {
        let mut changed = header.clone();
        changed[WEIGHT][key] = value;
        assert!(fc1_weight_span(&changed, 1_326_456, 7_602_973_028).is_err());
    }
    let old = json!({"vision.blocks.0.mlp.w1.weight":header[WEIGHT]});
    assert!(fc1_weight_span(&old, 1_326_456, 7_602_973_028).is_err());
    assert!(fc1_weight_span(&header, 1_326_456, 7_558_756_708 + 11_534_336 - 1).is_err());
    for bad in [0, u64::MAX] {
        assert!(fc1_weight_span(&header, bad, u64::MAX).is_err());
    }
}

fn spans() -> [DeviceSpan; 4] {
    [
        DeviceSpan {
            ptr: 0x1000,
            bytes: 40960,
        },
        DeviceSpan {
            ptr: 0x100000,
            bytes: 11_534_336,
        },
        DeviceSpan {
            ptr: 0x2000000,
            bytes: 225280,
        },
        DeviceSpan {
            ptr: 0x3000000,
            bytes: 8_519_680,
        },
    ]
}

#[test]
fn block_eight_reuses_exact_native_and_gemmex_geometry_not_reference_accessors() {
    let [x, w, y, workspace] = spans();
    let bound = Fc1Plan::new().bind(x, w, y, workspace).unwrap();
    let call = bound.call();
    assert_eq!(
        (call.transa, call.transb, call.m, call.n, call.k),
        (1, 0, 5632, 20, 1024)
    );
    assert_eq!((call.lda, call.ldb, call.ldc), (1024, 1024, 5632));
    assert_eq!((call.a, call.b, call.c), (w.ptr, x.ptr, y.ptr));
    assert_eq!(
        (
            call.a_type,
            call.b_type,
            call.c_type,
            call.compute_type,
            call.algorithm
        ),
        (14, 14, 14, 68, 99)
    );
    assert_eq!(
        (call.alpha.to_bits(), call.beta.to_bits()),
        (1f32.to_bits(), 0)
    );
    let native = bound.native_launch();
    assert_eq!((native.grid, native.block), ([176, 1, 1], [128, 1, 1]));
    assert_eq!(
        native.args,
        vec![
            Arg::Ptr(x.ptr),
            Arg::Ptr(w.ptr),
            Arg::Ptr(0),
            Arg::Ptr(y.ptr),
            Arg::U32(20),
            Arg::U32(5632),
            Arg::U32(1024),
            Arg::U32(5632)
        ]
    );
    for index in 0..4 {
        let mut changed = spans();
        changed[index].bytes -= 2;
        assert!(
            Fc1Plan::new()
                .bind(changed[0], changed[1], changed[2], changed[3])
                .is_err()
        );
        let mut changed = spans();
        changed[index].ptr = u64::MAX - 255;
        assert!(
            Fc1Plan::new()
                .bind(changed[0], changed[1], changed[2], changed[3])
                .is_err()
        );
    }
    for left in 0..4 {
        for right in left + 1..4 {
            let mut changed = spans();
            changed[right].ptr = changed[left].ptr + 256;
            assert!(
                Fc1Plan::new()
                    .bind(changed[0], changed[1], changed[2], changed[3])
                    .is_err()
            );
        }
    }
}

#[test]
fn nine_real_guarded_allocations_fit_explicit_96_mib_cap_without_hiding_workspace() {
    let mut total = 0;
    for bytes in [
        40960,
        11_534_336,
        225280,
        225280,
        225280,
        225280,
        225280,
        8_519_680,
        64 * 1024 * 1024,
    ] {
        total = guarded_bytes(bytes, total).unwrap().1;
    }
    assert_eq!(total, 88_334_848);
    assert_eq!(owned_device_bytes().unwrap(), total);
    assert_eq!(contract::OWNED_DEVICE_CAP, 96 * 1024 * 1024);
    assert!(total < contract::OWNED_DEVICE_CAP);
    assert!(guarded_bytes(usize::MAX, total).is_err());
}
