// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the exact-2410 M32xN256 fused W2A8 subordinate arm.

use std::fs;
use std::path::PathBuf;

const FLAG: &str = "ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN_N256";
const REQUESTED: &str = "w2a8_fused_gu_down_n256_requested";
const HANDLE: &str = "w2a8_fused_gu_down_emit_n256_k";
const SYMBOL: &str = "exl3_w2a8_fused_gu_down_emit_n256";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative))
        .unwrap_or_else(|error| panic!("missing {relative}: {error}"))
}

fn compact(value: &str) -> String {
    value.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn state_selector_is_exact_one_and_subordinate_to_the_n128_fusion() {
    let state = source("crates/spark-model/src/layers/moe/exl3_decode.rs");
    let flat = compact(&state);
    assert!(state.contains(&format!("pub(crate) {REQUESTED}: bool")));
    assert!(state.contains(&format!("pub(crate) {HANDLE}: KernelHandle")));
    assert!(flat.contains(&format!(
        "let{REQUESTED}=w2a8_fused_gu_down_requested&&std::env::var(\"{FLAG}\").as_deref()==Ok(\"1\");"
    )));
    assert!(state.contains(&format!("if {REQUESTED}")));
    assert!(state.matches(&format!("\"{SYMBOL}\"")).count() >= 2);
    assert!(state.contains("KernelHandle(0)"));
}

#[test]
fn dispatch_prevalidates_exact_2410_and_falls_back_n256_to_n128_to_incumbent() {
    let dispatch = source("crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs");
    let flat = compact(&dispatch);
    let mutation = dispatch.find("// MUTATION START").expect("mutation marker");
    let before = &dispatch[..mutation];
    assert!(before.contains(&format!("pf.{REQUESTED}")));
    assert!(before.contains(&format!("pf.{HANDLE}.0 != 0")));
    assert!(before.contains("total_expanded == W2A8_TARGET_TOTAL_EXPANDED"));
    assert!(before.contains("num_experts == 256"));
    assert!(before.contains("fused_ranges_disjoint"));
    assert!(before.contains("fused_pointers_aligned"));
    assert!(flat.contains("W2A8_GATE_UP_N/256"));

    let after = &dispatch[mutation..];
    let selector = after
        .find("plan.use_fused_gu_n256")
        .expect("prevalidated N256 selector");
    let n256 = after[selector..].find(HANDLE).expect("N256 launch handle") + selector;
    let n128 = after[n256..]
        .find("plan.use_fused_gu")
        .expect("N128 fallback selector")
        + n256;
    let incumbent = after[n128..]
        .find("pf.w2a8_grouped_gu_k")
        .expect("incumbent gate/up fallback")
        + n128;
    assert!(selector < n256 && n256 < n128 && n128 < incumbent);
}

#[test]
fn n256_launch_uses_exact_abi_and_precomputed_grid() {
    let dispatch = source("crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs");
    let start = dispatch
        .find(&format!("KernelLaunch::new(ctx.gpu, pf.{HANDLE})"))
        .expect("N256 launch");
    let tail = &dispatch[start..];
    let end = tail.find(".launch(stream)?;").expect("launch end") + ".launch(stream)?;".len();
    let launch = compact(&tail[..end]);
    for contract in [
        ".grid([plan.fused_gu_n256_grid,1,1])",
        ".block([256,1,1])",
        ".arg_ptr(plan.gate_fp8)",
        ".arg_ptr(plan.gate_scale)",
        ".arg_ptr(plan.up_fp8)",
        ".arg_ptr(plan.up_scale)",
        ".arg_ptr(st.gate.trellis_tab)",
        ".arg_ptr(st.up.trellis_tab)",
        ".arg_ptr(st.gate.svh_tab)",
        ".arg_ptr(st.up.svh_tab)",
        ".arg_ptr(st.down.suh_tab)",
        ".arg_ptr(plan.down_fp8)",
        ".arg_ptr(plan.down_scale)",
        ".arg_ptr(expert_offsets)",
        ".arg_u32(num_experts)",
        ".arg_u32(total_expanded)",
        ".arg_u32(W2A8_GATE_UP_N)",
        ".arg_u32(W2A8_GATE_UP_K)",
        ".arg_u32(2)",
        ".arg_u32(1)",
    ] {
        assert!(launch.contains(contract), "N256 launch omits `{contract}`");
    }
}

#[test]
fn component_and_host_share_the_exact_expanded_row_contract() {
    let component = source("kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n256.cu");
    let flat = compact(&component);
    assert!(component.contains("#define W2F_TOTAL_ROWS 14460"));
    assert!(flat.contains("total_rows!=W2F_TOTAL_ROWS"));
    assert!(flat.contains("route_end!=(int)total_rows"));
}
