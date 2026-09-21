// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the default-off fused EXL3 W2A8 gate/up production arm.

use std::fs;
use std::path::PathBuf;

const WRAPPER: &str = "kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_fused_gu_down_emit_n128.cu";
const COMPONENT: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu";
const STATE: &str = "crates/spark-model/src/layers/moe/exl3_decode.rs";
const DISPATCH: &str = "crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs";
const BUILD: &str = "crates/atlas-kernels/build.rs";
const FLAG: &str = "ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN";
const HANDLE: &str = "w2a8_fused_gu_down_emit_n128_k";
const SYMBOL: &str = "exl3_w2a8_fused_gu_down_emit_n128";
const CONTINUOUS_RING: &str = "W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative)).unwrap_or_else(|error| {
        panic!("required fused W2A8 integration source {relative}: {error}")
    })
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn canonical_wrapper_is_auto_registered_only_for_the_deepseek_target() {
    let wrapper = read(WRAPPER);
    assert_eq!(
        wrapper.lines().next(),
        Some("// SPDX-License-Identifier: AGPL-3.0-only")
    );
    for contract in [
        "#define W2A8_FIXED_N 2048",
        "#define W2A8_FIXED_K 4096",
        "#define W2A8_KERNEL_NAME exl3_w2a8_fused_gu_down_emit_n128",
        "#ifdef W2A8_PACKED_E4M3_CANDIDATE",
        "#undef W2A8_PACKED_E4M3_CANDIDATE",
        "#define W2A8_PACKED_E4M3_CANDIDATE 1",
        "#ifdef W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE",
        "#undef W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE",
        "#define W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE 1",
        "#ifdef W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE",
        "#undef W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE",
        "#define W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE 1",
        "#include \"../../experiments/exl3_w2a8_fused_gu_down_emit_n128.cu\"",
    ] {
        assert!(wrapper.contains(contract), "wrapper omits `{contract}`");
    }
    assert_eq!(wrapper.matches("#include").count(), 1);
    assert!(!wrapper.contains("#ifndef W2A8_PACKED_E4M3_CANDIDATE"));
    assert!(!wrapper.contains("#ifndef W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE"));
    assert!(!wrapper.contains(&format!("#ifndef {CONTINUOUS_RING}")));
    assert!(
        !wrapper.contains("__global__"),
        "wrapper must not fork the locked body"
    );
    assert!(workspace().join(COMPONENT).is_file());
    let component = read(COMPONENT);
    assert!(component.contains(&format!("#ifndef {CONTINUOUS_RING}")));
    assert!(component.contains(&format!("#define {CONTINUOUS_RING} 0")));
    assert!(
        !workspace()
            .join("kernels/gb10/common/exl3_w2a8_fused_gu_down_emit_n128.cu")
            .exists(),
        "the exact DeepSeek kernel must not leak into common targets"
    );

    let build = compact(&read(BUILD));
    assert!(build.contains("collect_cu_files("));
    assert!(
        build.contains("target.common_kernel_dir.as_deref(),&target.model_kernel_dir,source_ext,")
    );
    assert!(build.contains("forfinfind_cu_files(model_dir,source_ext)"));
    assert!(build.contains("unwrap_or_else(||stem.clone())"));
}

#[test]
fn state_gate_is_strict_default_off_and_subordinate_to_w2a8() {
    let state = read(STATE);
    let flat = compact(&state);
    assert!(flat.contains(&format!(
        "letw2a8_fused_gu_down_requested=w2a8_requested&&std::env::var(\"{FLAG}\").as_deref()==Ok(\"1\");"
    )));
    assert!(state.contains("pub(crate) w2a8_fused_gu_down_requested: bool"));
    assert!(state.contains(&format!("pub(crate) {HANDLE}: KernelHandle")));

    let conditional = state
        .find("if w2a8_fused_gu_down_requested")
        .expect("fused handle must be resolved only when its subordinate gate is active");
    let tail = &state[conditional..];
    let end = tail.find("let configured_chunk").unwrap_or(tail.len());
    let handle_init = &tail[..end];
    assert!(handle_init.contains("try_kernel("));
    assert!(handle_init.matches(&format!("\"{SYMBOL}\"")).count() >= 2);
    assert!(handle_init.contains("KernelHandle(0)"));
    assert!(
        state.find("let w2a8_requested").unwrap() < conditional,
        "the base W2A8 gate must be established before the fused sub-gate"
    );
}

#[test]
fn fused_dispatch_is_prevalidated_and_preserves_the_five_launch_fallback() {
    let dispatch = read(DISPATCH);
    let flat = compact(&dispatch);
    let mutation = dispatch.find("// MUTATION START").expect("mutation marker");
    let before = &dispatch[..mutation];
    assert!(before.contains("pf.w2a8_fused_gu_down_requested"));
    assert!(before.contains(HANDLE));
    assert!(before.contains("W2A8_GATE_UP_N / 128"));
    assert!(before.contains("fused_ranges_disjoint"));
    assert!(before.contains("fused_pointers_aligned"));
    assert!(before.contains("num_experts == 256"));
    assert!(before.contains("checked_range(gate_fp8, gu.total_bytes)"));
    assert!(before.contains("checked_range(up_fp8, gu.total_bytes)"));
    assert!(before.contains("checked_range(fused_down_fp8, down.total_bytes)"));
    assert!(before.contains("let fused_down_fp8 = expert_gate_out"));
    assert!(
        before.find("pf.w2a8_fused_gu_down_requested").unwrap()
            > before.find("!pf.w2a8_requested").unwrap(),
        "the fused selection must remain inside the accepted base W2A8 plan"
    );

    let launch_start = dispatch
        .find(&format!("KernelLaunch::new(ctx.gpu, pf.{HANDLE})"))
        .expect("fused production launch");
    let launch_tail = &dispatch[launch_start..];
    let launch_end = launch_tail
        .find(".launch(stream)?;")
        .expect("fused production launch end")
        + ".launch(stream)?;".len();
    let launch = compact(&launch_tail[..launch_end]);
    for contract in [
        ".grid([plan.fused_gu_grid,1,1])",
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
        assert!(launch.contains(contract), "fused launch omits `{contract}`");
    }

    assert_eq!(
        flat.matches("KernelLaunch::new(ctx.gpu,pf.w2a8_grouped_gu_k)")
            .count(),
        2
    );
    assert_eq!(
        flat.matches("KernelLaunch::new(ctx.gpu,pf.w2a8_post_silu_pre_emit_k)")
            .count(),
        1
    );
    assert_eq!(
        flat.matches("KernelLaunch::new(ctx.gpu,pf.w2a8_grouped_down_k)")
            .count(),
        1
    );
    // The eighth static site is the narrower-grid N256 fused GU rung. Both
    // fused GU arms still replace exactly the incumbent's three middle sites.
    assert_eq!(flat.matches(".launch(stream)?").count(), 8);
}
