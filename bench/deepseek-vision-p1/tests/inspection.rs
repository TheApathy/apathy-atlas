// SPDX-License-Identifier: AGPL-3.0-only
use deepseek_vision_p1::{
    admission::{reference_hash, validate_build},
    weight::tensor_span,
};
use serde_json::json;
#[test]
fn default_stream_operation_finishes_before_returning_to_a_consumer() {
    use deepseek_vision_p1::driver::complete_default_stream;
    let events = std::cell::RefCell::new(Vec::new());
    let value = complete_default_stream(
        || {
            events.borrow_mut().push("operation");
            Ok(7)
        },
        || {
            events.borrow_mut().push("context fence");
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(value, 7);
    assert_eq!(*events.borrow(), ["operation", "context fence"]);
}
#[test]
fn default_stream_fence_is_attempted_on_error_and_cannot_be_ignored() {
    use deepseek_vision_p1::driver::complete_default_stream;
    let fenced = std::cell::Cell::new(false);
    let error = complete_default_stream::<()>(
        || anyhow::bail!("operation failure"),
        || {
            fenced.set(true);
            anyhow::bail!("fence failure")
        },
    )
    .unwrap_err()
    .to_string();
    assert!(fenced.get());
    assert!(error.contains("operation failure") && error.contains("fence failure"));
    assert!(complete_default_stream(|| Ok(7), || anyhow::bail!("fence failure")).is_err());
}
#[test]
fn all_default_stream_memory_operations_and_cleanup_use_context_completion() {
    let source = include_str!("../src/driver.rs");
    for (start, end) in [
        ("pub fn allocate(", "pub fn upload("),
        ("pub fn write(", "fn read_span("),
        ("fn read_span(", "pub fn read("),
    ] {
        let body = source
            .split_once(start)
            .unwrap()
            .1
            .split_once(end)
            .unwrap()
            .0;
        assert!(body.contains("complete_default_stream("), "{start}");
        assert!(body.contains("self.context_complete()"), "{start}");
    }
    let cleanup = source.split_once("pub fn close(").unwrap().1;
    assert!(
        cleanup.find("self.api.context_sync").unwrap()
            < cleanup.find("self.api.module_unload").unwrap()
    );
    let abi = include_str!("../src/cuda_abi.rs");
    assert!(abi.contains("context_sync: s!(\"cuCtxSynchronize\")"));
}
#[test]
fn guarded_device_sizes_and_u64_spans_reject_wrap_and_cap_overrun() {
    use deepseek_vision_p1::driver::{guarded_bytes, guarded_span};
    const CAP: usize = 128 * 1024 * 1024;
    assert_eq!(guarded_bytes(1024, 512).unwrap(), (1536, 2048));
    assert_eq!(guarded_bytes(CAP - 512, 0).unwrap(), (CAP, CAP));
    for (bytes, current) in [
        (0, 0),
        (usize::MAX, 0),
        (CAP - 511, 0),
        (1, CAP),
        (1, usize::MAX),
    ] {
        assert!(guarded_bytes(bytes, current).is_err());
    }
    assert_eq!(
        guarded_span(0x1000, 1024).unwrap(),
        (0x1100, 0x1500, 0x1600)
    );
    // Check leading guard, payload, and trailing guard overflow separately.
    for base in [
        0,
        u64::MAX - 255,
        u64::MAX - 256,
        u64::MAX - 1279,
        u64::MAX - 1535,
    ] {
        assert!(guarded_span(base, 1024).is_err(), "{base}");
    }
    assert_eq!(guarded_span(u64::MAX - 1536, 1024).unwrap().2, u64::MAX);
}
#[test]
fn selected_tensor_span_is_checked_before_any_payload_read() {
    let good = json!({"weight":{"dtype":"BF16","shape":[1024,2816],"data_offsets":[32,5767200]}});
    assert_eq!(
        tensor_span(&good, "weight", 100, 5767308).unwrap(),
        (140, 5767168)
    );
    for value in [
        json!([32.0, 5767200]),
        json!([32, 5767201]),
        json!([5767200, 32]),
        json!([0, u64::MAX]),
    ] {
        let mut bad = good.clone();
        bad["weight"]["data_offsets"] = value;
        assert!(tensor_span(&bad, "weight", 100, 5767308).is_err());
    }
    assert!(tensor_span(&good, "weight", 100, 5767307).is_err());
    let mut bad = good.clone();
    bad["weight"]["shape"] = json!([1024.0, 2816]);
    assert!(tensor_span(&bad, "weight", 100, 5767308).is_err());
    bad = good.clone();
    bad["weight"]["dtype"] = json!("F32");
    assert!(tensor_span(&bad, "weight", 100, 5767308).is_err());
}
#[test]
fn reference_lookup_requires_unique_case_operator_and_exact_shared_input() {
    let good = json!({"cases":[{"case":"grid-4x5","operators":[{"stage":"block-00-fc2",
        "dtype":"torch.bfloat16","shared_native_inputs":["block-00-swiglu"],"reference_sha256":"a".repeat(64)}]}]});
    assert_eq!(
        reference_hash(&good, "grid-4x5", "block-00-fc2", "block-00-swiglu").unwrap(),
        "a".repeat(64)
    );
    let mut bad = good.clone();
    bad["cases"][0]["operators"][0]["shared_native_inputs"] = json!(["reference-swiglu"]);
    assert!(reference_hash(&bad, "grid-4x5", "block-00-fc2", "block-00-swiglu").is_err());
    bad = good.clone();
    bad["cases"]
        .as_array_mut()
        .unwrap()
        .push(good["cases"][0].clone());
    assert!(reference_hash(&bad, "grid-4x5", "block-00-fc2", "block-00-swiglu").is_err());
}
#[test]
fn candidate_build_is_explicit_no_fastmath_no_fusion_and_source_bound() {
    let good = json!({"schema":"atlas-dsv-p1-angle-build-v1","source_sha256":"a".repeat(64),
        "ptx_sha256":"b".repeat(64),"compiler_sha256":"c".repeat(64),
        "compiler_version":"Cuda compilation tools, release 13.0",
        "flags":["-ptx","-O3","-arch=sm_121f","--fmad=false","--ftz=false","--prec-div=true","--prec-sqrt=true"]});
    validate_build(&good, &"a".repeat(64), &"b".repeat(64)).unwrap();
    for field in ["source_sha256", "ptx_sha256", "compiler_sha256"] {
        let mut bad = good.clone();
        bad[field] = json!("x");
        assert!(validate_build(&bad, &"a".repeat(64), &"b".repeat(64)).is_err());
    }
    let mut bad = good.clone();
    bad["flags"]
        .as_array_mut()
        .unwrap()
        .push(json!("--use_fast_math"));
    assert!(validate_build(&bad, &"a".repeat(64), &"b".repeat(64)).is_err());
}
