// SPDX-License-Identifier: AGPL-3.0-only
// RED: production contract module is deliberately absent until root executes RED.
#[path = "../src/contract.rs"]
mod contract;

use contract::{DeviceSpan, Fc1Plan, ReductionMode, fc1_weight_span, validate_references};
use deepseek_vision_p1::{contract::Arg, driver::guarded_bytes, pins};
use serde_json::{Value, json};

const INPUT: &str = "a74cf71f1f2f958da55bb2ad64961dc79a0e34c5e0ecfad94fbe974ea74144cf";
const NATIVE: &str = "e39b9e80a339988ba215e8f22c6af82e5cf53f69b9da482056e03c8cb1307e41";
const DEFAULT: &str = "5c4d0b299c710760fe95be6a1caa9cba3826ff4739081b2d679f682d413261ad";
const WEIGHT: &str = "vision.blocks.0.mlp.w1.weight";

fn spans() -> [DeviceSpan; 4] {
    [
        DeviceSpan {
            ptr: 0x1000,
            bytes: 40_960,
        },
        DeviceSpan {
            ptr: 0x100000,
            bytes: 11_534_336,
        },
        DeviceSpan {
            ptr: 0x2000000,
            bytes: 225_280,
        },
        DeviceSpan {
            ptr: 0x3000000,
            bytes: 8_519_680,
        },
    ]
}

#[test]
fn fc1_plan_is_the_pinned_torch_row_major_gemmex_not_lt_or_fc2() {
    let plan = Fc1Plan::new();
    assert_eq!((plan.rows, plan.outputs, plan.inner), (20, 5632, 1024));
    assert_eq!(plan.workspace_bytes, 8_519_680);
    let [x, w, y, workspace] = spans();
    let bound = plan.bind(x, w, y, workspace).unwrap();
    let call = bound.call();
    assert_eq!((call.transa, call.transb), (1, 0));
    assert_eq!((call.m, call.n, call.k), (5632, 20, 1024));
    assert_eq!((call.lda, call.ldb, call.ldc), (1024, 1024, 5632));
    assert_eq!((call.a, call.b, call.c), (w.ptr, x.ptr, y.ptr));
    assert_eq!((call.a_type, call.b_type, call.c_type), (14, 14, 14));
    assert_eq!(
        (call.alpha.to_bits(), call.beta.to_bits()),
        (1.0f32.to_bits(), 0)
    );
    assert_eq!(
        call.compute_type, 68,
        "C++ legacy CUDA_R_32F=0 must migrate before C ABI"
    );
    assert_eq!(call.algorithm, 99);
    let launch = bound.native_launch();
    assert_eq!(launch.grid, [176, 1, 1]);
    assert_eq!(launch.block, [128, 1, 1]);
    assert_eq!(
        launch.args,
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
}

#[test]
fn modes_bind_distinct_teacher_hashes_without_a_tolerance_knob() {
    assert_eq!(ReductionMode::Default.math_mode(), 0);
    assert_eq!(ReductionMode::Full.math_mode(), 16);
    assert_eq!(ReductionMode::Default.reference_sha256(), DEFAULT);
    assert_eq!(ReductionMode::Full.reference_sha256(), NATIVE);
    assert_eq!(ReductionMode::Default.label(), "default");
    assert_eq!(ReductionMode::Full.label(), "full");
}

#[test]
fn every_device_extent_and_pairwise_alias_is_rejected_before_io() {
    let plan = Fc1Plan::new();
    for index in 0..4 {
        for replacement in [
            DeviceSpan {
                ptr: 0,
                bytes: spans()[index].bytes,
            },
            DeviceSpan {
                ptr: u64::MAX - 255,
                bytes: spans()[index].bytes,
            },
            DeviceSpan {
                ptr: spans()[index].ptr + 1,
                bytes: spans()[index].bytes,
            },
            DeviceSpan {
                ptr: spans()[index].ptr,
                bytes: spans()[index].bytes - 2,
            },
            DeviceSpan {
                ptr: spans()[index].ptr,
                bytes: usize::MAX,
            },
        ] {
            let mut value = spans();
            value[index] = replacement;
            assert!(plan.bind(value[0], value[1], value[2], value[3]).is_err());
        }
    }
    for left in 0..4 {
        for right in left + 1..4 {
            let mut value = spans();
            value[right].ptr = value[left].ptr + 256;
            assert!(plan.bind(value[0], value[1], value[2], value[3]).is_err());
        }
    }
}

#[test]
fn real_reused_guard_accounting_bounds_control_and_two_mode_outputs() {
    let mut total = 0;
    for bytes in [40_960, 11_534_336, 225_280, 225_280, 225_280, 8_519_680] {
        total = guarded_bytes(bytes, total).unwrap().1;
    }
    assert_eq!(total, 20_773_888);
    assert!(total < 32 * 1024 * 1024);
    assert!(guarded_bytes(usize::MAX, total).is_err());
}

fn reference(full: bool) -> Value {
    let mut result = json!({"diagnostic_only":true,
        "scope":"independent operators on shared native inputs, not chained-reference qualification",
        "official_sha256":pins::OFFICIAL_SHA,"payload_sha256":pins::PAYLOAD_SHA,
        "torch_version":"2.10.0+cu130","bf16_reduced_precision_reduction":!full,
        "cases":[{"case":"grid-4x5","operators":[{"stage":"block-00-fc1",
            "shared_native_inputs":["block-00-norm2"],"dtype":"torch.bfloat16",
            "reference_sha256":if full {NATIVE} else {DEFAULT}}]}]});
    if full {
        result["bf16_reduction"] = json!("full");
    }
    result
}
fn stages() -> Value {
    json!({"grid":[4,5],"output_byte_equal":true,"stages":[
        {"name":"block-00-norm2","file":"block-00-norm2.bf16","shape":[20,1024],
            "dtype":"bf16","bytes":40960,"sha256":INPUT},
        {"name":"block-00-fc1","file":"block-00-fc1.bf16","shape":[20,5632],
            "dtype":"bf16","bytes":225280,"sha256":NATIVE}]})
}

#[test]
fn retained_reference_requires_exact_same_input_geometry_and_distinct_modes() {
    let receipt = validate_references(&stages(), &reference(false), &reference(true)).unwrap();
    assert_eq!(receipt.input_sha256, INPUT);
    assert_eq!(receipt.native_sha256, NATIVE);
    assert_eq!(receipt.default_sha256, DEFAULT);
    assert_eq!(receipt.full_sha256, NATIVE);
    for key in [
        "official_sha256",
        "payload_sha256",
        "torch_version",
        "scope",
    ] {
        let mut changed = reference(false);
        changed[key] = Value::Null;
        assert!(validate_references(&stages(), &changed, &reference(true)).is_err());
    }
    for (key, value) in [
        ("shared_native_inputs", json!(["block-00-swiglu"])),
        ("reference_sha256", json!(NATIVE)),
        ("dtype", json!("torch.float32")),
    ] {
        let mut changed = reference(false);
        changed["cases"][0]["operators"][0][key] = value;
        assert!(validate_references(&stages(), &changed, &reference(true)).is_err());
    }
    for (key, value) in [
        ("shape", json!([20, 1023])),
        ("bytes", json!(40962)),
        ("sha256", json!("a".repeat(64))),
        ("file", json!("../norm2.bf16")),
    ] {
        let mut changed = stages();
        changed["stages"][0][key] = value;
        assert!(validate_references(&changed, &reference(false), &reference(true)).is_err());
    }
    let mut changed = stages();
    let duplicate = changed["stages"][0].clone();
    changed["stages"].as_array_mut().unwrap().push(duplicate);
    assert!(validate_references(&changed, &reference(false), &reference(true)).is_err());
}

#[test]
fn selected_fc1_weight_is_native_bf16_and_checked_inside_the_pinned_shard() {
    let header = json!({WEIGHT:{"dtype":"BF16","shape":[5632,1024],
        "data_offsets":[128,128+11_534_336]}});
    assert_eq!(
        fc1_weight_span(&header, 64, 200 + 11_534_336).unwrap(),
        (200, 11_534_336)
    );
    for (key, value) in [
        ("dtype", json!("F16")),
        ("shape", json!([1024, 5632])),
        ("shape", json!([5632.0, 1024.5])),
        ("data_offsets", json!([129, 128 + 11_534_336])),
        ("data_offsets", json!([u64::MAX - 11_534_336, u64::MAX])),
    ] {
        let mut changed = header.clone();
        changed[WEIGHT][key] = value;
        assert!(fc1_weight_span(&changed, 64, u64::MAX).is_err());
    }
    assert!(fc1_weight_span(&header, 64, 200 + 11_534_336 - 1).is_err());
    assert!(fc1_weight_span(&header, u64::MAX, u64::MAX).is_err());
    assert!(fc1_weight_span(&header, 0, u64::MAX).is_err());
}
