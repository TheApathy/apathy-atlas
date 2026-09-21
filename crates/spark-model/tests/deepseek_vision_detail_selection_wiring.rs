// SPDX-License-Identifier: AGPL-3.0-only
//! Real production/probe seams supplement selection behavior; not GPU parity.
const FORWARD: &str = include_str!("../src/layers/deepseek_vision/forward.rs");
const MODULE: &str = include_str!("../src/layers/deepseek_vision/mod.rs");
const OBSERVER: &str = include_str!("../src/layers/deepseek_vision/observer.rs");
const EXAMPLE: &str = include_str!("../examples/deepseek_vision_probe.rs");
const GPU: &str = include_str!("../examples/deepseek_vision_probe/gpu.rs");
const STAGES: &str = include_str!("../examples/deepseek_vision_probe/stages.rs");

fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

fn body<'a>(source: &'a str, signature: &str) -> &'a str {
    let tail = source
        .split_once(signature)
        .expect("required production method missing")
        .1;
    let start = tail.find('{').expect("method body missing");
    let mut depth = 0usize;
    for (index, ch) in tail[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &tail[start + 1..start + index];
                }
            }
            _ => (),
        }
    }
    panic!("unterminated production method");
}

#[test]
fn selected_block_admission_precedes_the_shared_upload_and_any_callback() {
    let source = compact(FORWARD);
    assert!(compact(MODULE).contains("moddetail_selection;"));
    let selected = body(&source, "pubfnforward_observed_block(");
    let check = selected
        .find("DetailSelection::new(block,self.weights.blocks.len())")
        .expect("selected index must use actual loaded block count");
    let shared = selected
        .find("self.forward_inner(")
        .expect("must use the real forward graph");
    assert!(check < shared);
    assert!(!selected[..check].contains("gpu."));
    assert!(!selected[..check].contains("observe("));
    assert!(!selected[..check].contains("observer("));
    let inner = body(&source, "fnforward_inner(");
    assert!(
        inner.contains("gpu.copy_h2d("),
        "existing upload seam was not preserved"
    );
    assert!(inner.contains("self.run("));
}

#[test]
fn ordinary_and_legacy_observed_entries_keep_their_original_contracts() {
    let source = compact(FORWARD);
    let ordinary = body(&source, "pubfnforward(");
    assert!(ordinary.contains("self.forward_inner(gpu,patches,grid_h,grid_w,None,None)"));
    assert!(!ordinary.contains("stage_name("));
    let legacy = body(&source, "pubfnforward_observed(");
    assert!(legacy.contains("self.forward_observed_block(gpu,patches,grid_h,grid_w,0,observer)"));
    assert!(!source.contains("std::env"));
    assert!(!source.contains("set_var("));
}

#[test]
fn detailed_taps_select_one_block_but_all_actual_block_exits_remain() {
    let source = compact(FORWARD);
    let run = body(&source, "fnrun(");
    assert!(run.contains(".is_selected(layer)"));
    assert!(
        run.contains("observer.is_some()"),
        "ordinary forward must not allocate tap names"
    );
    assert!(
        !run.contains("iflayer==0"),
        "block-zero-only observer remains"
    );
    assert!(
        !run.contains("\"block-00-"),
        "detail labels must derive from selected block"
    );
    assert!(run.contains(".stage_name("));
    for suffix in [
        "norm1",
        "qkv",
        "query",
        "key",
        "value",
        "head-00-scores",
        "head-00-probs",
        "attention",
        "projection",
        "residual1",
        "norm2",
        "fc1",
        "swiglu",
        "fc2",
    ] {
        assert!(
            run.contains(&format!("\"{suffix}\"")),
            "missing actual tap {suffix}"
        );
    }
    assert!(run.contains("self.weights.blocks.iter().enumerate()"));
    assert!(run.contains("\"block-{layer:02}-exit\""));
    assert!(
        run.contains("head==0"),
        "head-zero score/probability detail removed"
    );
}

#[test]
fn explicit_probe_command_preserves_bounds_and_records_selected_block() {
    let example = compact(EXAMPLE);
    assert!(example.contains("\"stages-block\""));
    assert!(example.contains("args.len()==5"));
    assert!(example.contains("parse::<usize>()"));
    assert!(
        example.contains("\"run\"|\"stages\""),
        "legacy commands removed"
    );
    let gpu = compact(GPU);
    let stages = compact(STAGES);
    assert!(gpu.contains("\"selected_detail_block\""));
    assert!(stages.contains("\"selected_detail_block\""));
    assert!(stages.contains("encoder.forward_observed_block("));
    assert!(stages.contains("entries.len()<64"));
    assert!(stages.contains("total<=512*1024*1024"));
    assert!(
        stages.contains("final_raw==expected"),
        "observation must not change final bytes"
    );
    assert!(stages.contains("\"output_byte_equal\":true"));
}

#[test]
fn selection_does_not_add_an_alternate_arithmetic_graph_or_drop_observer_fences() {
    let source = compact(FORWARD);
    let run = body(&source, "fnrun(");
    for kernel in [
        "angles",
        "rope",
        "scores",
        "softmax",
        "attention_value",
        "swiglu",
        "unfold",
        "gelu",
    ] {
        assert_eq!(
            run.matches(&format!("KernelLaunch::new(gpu,self.kernels.{kernel})"))
                .count(),
            1,
            "production kernel graph changed: {kernel}"
        );
    }
    assert!(source.contains("self.kernels.linear"));
    assert!(!source.contains("GemmEx"));
    assert!(!source.contains("linear_diagnostic"));
    let observer = compact(OBSERVER);
    let fence = observer
        .find("gpu.synchronize(gpu.default_stream())?")
        .unwrap();
    let callback = observer
        .find("callback(name,pointer,shape,dtype)?")
        .unwrap();
    assert!(fence < callback);
}
