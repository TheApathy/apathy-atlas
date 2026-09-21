// SPDX-License-Identifier: AGPL-3.0-only
//! Source seams supplement behavior tests; neither is GPU numerical proof.
const FORWARD: &str = include_str!("../src/layers/deepseek_vision/forward.rs");
const MODULE: &str = include_str!("../src/layers/deepseek_vision/mod.rs");

fn compact(s: &str) -> String {
    s.split_whitespace().collect()
}
fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split_once(start)
        .expect("missing start seam")
        .1
        .split_once(end)
        .expect("missing end seam")
        .0
}

#[test]
fn native_linear_kernel_and_its_single_bias_cast_remain_the_control() {
    let f = compact(FORWARD);
    let linear = section(&f, "fnlinear(", "fnnorm(");
    assert!(linear.contains("KernelLaunch::new(gpu,self.kernels.linear)"));
    assert!(linear.contains(".arg_ptr(weight.bias).arg_ptr(output)"));
    assert!(linear.contains(".arg_u32(weight.kasu32).arg_u32(ldcasu32)"));
    assert!(!linear.contains("cublaslt"));
}

#[test]
fn diagnostic_hook_is_explicit_and_normal_forward_cannot_select_it() {
    let f = compact(FORWARD);
    assert!(f.contains("pubfnforward_observed_with_linear_backend("));
    let ordinary = section(&f, "pubfnforward(", "pubfnforward_observed(");
    assert!(ordinary.contains("self.forward_inner(gpu,patches,grid_h,grid_w,None,None)"));
    assert!(
        !f.contains("std::env"),
        "a request-independent env knob must not change production"
    );
    let module = compact(MODULE);
    assert!(module.contains("modlinear_diagnostic_plan;"));
    assert!(module.contains("modlinear_diagnostic_completion;"));
}

#[test]
fn real_encoder_seven_call_sites_route_the_same_weights_into_borrowed_backend() {
    let f = compact(FORWARD);
    let run = section(&f, "fnrun(", "fnlinear(");
    assert!(run.contains("for(layer,block)inself.weights.blocks.iter().enumerate()"));
    assert_eq!(run.matches("self.linear(").count(), 7);
    for call in run.split("self.linear(").skip(1) {
        let body = call
            .split_once(")?;")
            .expect("linear call must propagate failure")
            .0;
        assert!(
            body.contains("linear_backend"),
            "real projection bypasses the diagnostic hook"
        );
    }
    let linear = section(&f, "fnlinear(", "fnnorm(");
    let custom = linear
        .find("backend.linear(")
        .expect("borrowed backend is never invoked");
    let native = linear
        .find("KernelLaunch::new(gpu,self.kernels.linear)")
        .unwrap();
    assert!(custom < native);
    assert!(
        linear[..native].contains("return"),
        "candidate must not fall through to native"
    );
}

#[test]
fn shared_forward_completion_covers_upload_and_run_not_just_launch_success() {
    let f = compact(FORWARD);
    let inner = section(&f, "fnforward_inner(", "fnrun(");
    let wrapper = inner
        .find("with_completion(")
        .expect("completion wrapper missing");
    let upload = inner
        .find("gpu.copy_h2d(")
        .expect("existing upload missing");
    let run = inner
        .find("self.run(")
        .expect("existing encoder body missing");
    assert!(wrapper < upload && upload < run);
    assert!(
        !inner.contains("result?;fence?;"),
        "simultaneous fence failure must not be hidden"
    );
}
