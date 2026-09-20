// SPDX-License-Identifier: AGPL-3.0-only

//! Source seam checks only. Kernel dispatch and invalid-route GPU gates follow separately.

use std::{fs, path::PathBuf};

fn source(path: &str) -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join(path),
    )
    .unwrap_or_else(|error| panic!("read production source {path}: {error}"))
    .chars()
    .filter(|c| !c.is_whitespace())
    .collect()
}

fn body<'a>(text: &'a str, name: &str) -> &'a str {
    let start = text
        .find(&format!("fn{name}("))
        .unwrap_or_else(|| panic!("missing production method {name}"));
    let open = start + text[start..].find('{').expect("method body");
    let mut depth = 0usize;
    for (at, byte) in text.as_bytes().iter().enumerate().skip(open) {
        if *byte == b'{' {
            depth += 1;
        }
        if *byte == b'}' {
            depth -= 1;
            if depth == 0 {
                return &text[start..=at];
            }
        }
    }
    panic!("unclosed production method {name}");
}

fn before(text: &str, first: &str, second: &str) {
    let a = text
        .find(first)
        .unwrap_or_else(|| panic!("missing {first}"));
    let b = text
        .find(second)
        .unwrap_or_else(|| panic!("missing {second}"));
    assert!(a < b, "{first} must precede {second}");
}

#[test]
fn target_latches_policy_and_owned_scratch_extent_before_first_allocation() {
    let target = source("model/glm53/target_model_exl3.rs");
    let constructor = body(&target, "new_inner");
    assert!(constructor.contains("std::env::var_os(\"ATLAS_GLM53_EXL3_ROUTE_PRIVATE\")"));
    assert!(constructor.contains("std::env::var_os(\"ATLAS_GLM53_EXL3_ROUTE_PRIVATE_PREFILL\")"));
    before(
        constructor,
        "Glm53Exl3RoutePolicy::parse(",
        "build_moe_pointer_tables(",
    );
    before(
        constructor,
        ".validate_moe_mode(",
        "build_moe_pointer_tables(",
    );
    assert!(constructor.contains("Glm53WalkScratch::required_bytes_with_route_policy("));
    assert!(constructor.contains("Glm53WalkScratch::bind_with_route_policy("));
    assert!(target.contains("scratch_bytes:u64") || target.contains("scratch_bytes:usize"));
    let reset = source("model/glm53/target_vision_exl3.rs");
    let reset = body(&reset, "reset_sequence");
    assert!(reset.contains("self.scratch_bytes"));
    assert!(!reset.contains("Glm53WalkScratch::required_bytes()"));
    assert!(!reset.contains("std::env::"));
}

#[test]
fn dispatcher_passes_the_bound_policy_instead_of_resizing_from_live_environment() {
    let dispatch = source("model/glm53/dispatch.rs");
    let constructor = body(&dispatch, "new_exl3");
    assert!(constructor.contains("scratch.exl3_route_policy()"));
    assert!(constructor.contains("Glm53SerialMoeKernels::load_exl3_with_route_policy("));
    assert!(!constructor.contains("Glm53SerialMoeKernels::load_exl3(gpu)"));
    let serial = source("layers/glm53_moe_serial.rs");
    let loader = body(&serial, "load_exl3_with_route_policy");
    assert!(!loader.contains("std::env::"));
    assert!(loader.contains("Glm53Exl3MoeKernels::load_with_route_policy("));
}

#[test]
fn actual_executor_derives_selected_plan_and_checks_full_private_prefix_before_cast() {
    let serial = source("layers/glm53_moe_serial.rs");
    let execute = body(&serial, "execute_exl3_fused_rows");
    assert!(execute.contains("fused.plan(rows)?"));
    assert!(!execute.contains("Glm53Exl3MoePlan::new(rows)"));
    before(
        execute,
        "prefix(moe_scratch.route_private_f32,plan.route_private_f32_bytes)?",
        "casts.bf16_to_f16(",
    );
    before(execute, "casts.bf16_to_f16(", "fused.launch(");
    before(execute, "fused.launch(", "fused.combine_shared(");
    let kernels = source("layers/ops/glm53_exl3_moe.rs");
    let plan = body(&kernels, "plan");
    assert!(plan.contains(".private_bytes(rows)"));
    assert!(!plan.contains("std::env::"));
}

#[test]
fn launch_and_combine_do_not_reread_process_environment() {
    let kernels = source("layers/ops/glm53_exl3_moe.rs");
    for method in ["launch", "combine_shared"] {
        assert!(!body(&kernels, method).contains("std::env::"));
    }
}
