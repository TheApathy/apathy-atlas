// SPDX-License-Identifier: AGPL-3.0-only

//! CPU and source contracts for the opt-in DeepSeek EXL3 W2A8 prefill chain.

use std::fs;
use std::path::{Path, PathBuf};

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative))
        .unwrap_or_else(|error| panic!("missing {relative}: {error}"))
}

fn sidecar_bytes(rows: usize, k: usize) -> Option<(usize, usize)> {
    let fp8 = rows.checked_mul(k)?;
    let scales = rows.checked_mul(k.checked_div(128)?)?.checked_mul(4)?;
    Some((fp8, fp8.checked_add(scales)?))
}

fn capacities_fit(rows: usize, gate_bytes: usize, up_bytes: usize, down_bytes: usize) -> bool {
    let Some((_, gu_sidecar)) = sidecar_bytes(rows, 4096) else {
        return false;
    };
    let Some((_, down_sidecar)) = sidecar_bytes(rows, 2048) else {
        return false;
    };
    let Some(gu_output) = rows
        .checked_mul(2048)
        .and_then(|value| value.checked_mul(2))
    else {
        return false;
    };
    let Some(down_output) = rows
        .checked_mul(4096)
        .and_then(|value| value.checked_mul(2))
    else {
        return false;
    };
    gate_bytes >= gu_output
        && up_bytes >= gu_sidecar.max(down_sidecar)
        && down_bytes >= gu_sidecar.max(gu_output).max(down_output)
}

#[derive(Clone, Copy)]
struct Eligibility {
    requested: bool,
    direct: bool,
    persistent: bool,
    fixed_shape: bool,
    fixed_k2: bool,
    direct_m128: bool,
    direct_n128: bool,
    direct_n256: bool,
    dual_pre: bool,
    fused_post: bool,
    bits: [u32; 3],
    gate: (u32, u32),
    up: (u32, u32),
    down: (u32, u32),
    top_k: u32,
    tp: u32,
    ep: u32,
    communicator_present: bool,
    graph_capture: bool,
    handles_present: bool,
    capacity_present: bool,
}

impl Eligibility {
    fn exact() -> Self {
        Self {
            requested: true,
            direct: true,
            persistent: true,
            fixed_shape: true,
            fixed_k2: true,
            direct_m128: false,
            direct_n128: false,
            direct_n256: false,
            dual_pre: true,
            fused_post: true,
            bits: [2; 3],
            gate: (2048, 4096),
            up: (2048, 4096),
            down: (4096, 2048),
            top_k: 6,
            tp: 1,
            ep: 1,
            communicator_present: false,
            graph_capture: false,
            handles_present: true,
            capacity_present: true,
        }
    }

    fn eligible(self) -> bool {
        self.requested
            && self.direct
            && self.persistent
            && self.fixed_shape
            && self.fixed_k2
            && !self.direct_m128
            && !self.direct_n128
            && !self.direct_n256
            && self.dual_pre
            && self.fused_post
            && self.bits == [2; 3]
            && self.gate == (2048, 4096)
            && self.up == (2048, 4096)
            && self.down == (4096, 2048)
            && self.top_k == 6
            && self.tp == 1
            && self.ep == 1
            && !self.communicator_present
            && !self.graph_capture
            && self.handles_present
            && self.capacity_present
    }
}

#[test]
fn eligibility_declines_every_partial_or_unsupported_chain() {
    assert!(Eligibility::exact().eligible());
    let mut cases = Vec::new();
    macro_rules! decline {
        ($field:ident, $value:expr) => {{
            let mut case = Eligibility::exact();
            case.$field = $value;
            cases.push(case);
        }};
    }
    decline!(requested, false);
    decline!(direct, false);
    decline!(persistent, false);
    decline!(fixed_shape, false);
    decline!(fixed_k2, false);
    decline!(direct_m128, true);
    decline!(direct_n128, true);
    decline!(direct_n256, true);
    decline!(dual_pre, false);
    decline!(fused_post, false);
    decline!(bits, [3, 3, 3]);
    decline!(gate, (2048, 3072));
    decline!(up, (2048, 3072));
    decline!(down, (4096, 3072));
    decline!(top_k, 5);
    decline!(tp, 2);
    decline!(ep, 2);
    decline!(communicator_present, true);
    decline!(graph_capture, true);
    decline!(handles_present, false);
    decline!(capacity_present, false);
    assert!(cases.into_iter().all(|case| !case.eligible()));
}

#[test]
fn checked_sidecar_sizes_cover_scheduler_boundary_and_production_rows() {
    assert_eq!(sidecar_bytes(6144, 4096), Some((25_165_824, 25_952_256)));
    assert_eq!(sidecar_bytes(6150, 4096), Some((25_190_400, 25_977_600)));
    assert_eq!(sidecar_bytes(14460, 4096), Some((59_228_160, 61_079_040)));
    assert_eq!(sidecar_bytes(6144, 2048), Some((12_582_912, 12_976_128)));
    assert_eq!(sidecar_bytes(6150, 2048), Some((12_595_200, 12_988_800)));
    assert_eq!(sidecar_bytes(14460, 2048), Some((29_614_080, 30_539_520)));
    assert_eq!(sidecar_bytes(usize::MAX, 4096), None);
}

#[test]
fn arena_capacity_model_accepts_exact_bounds_and_rejects_each_one_byte_short() {
    let rows = 14_460;
    let gate = 59_228_160;
    let up = 61_079_040;
    let down = 118_456_320;
    assert!(capacities_fit(rows, gate, up, down));
    assert!(!capacities_fit(rows, gate - 1, up, down));
    assert!(!capacities_fit(rows, gate, up - 1, down));
    assert!(!capacities_fit(rows, gate, up, down - 1));
    assert!(!capacities_fit(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX
    ));
}

#[test]
fn state_is_explicit_opt_in_and_resolves_all_handles_only_when_requested() {
    let state = source("crates/spark-model/src/layers/moe/exl3_decode.rs");
    assert!(state.contains("ATLAS_EXL3_PREFILL_W2A8"));
    assert!(state.contains("as_deref() == Ok(\"1\")"));
    assert!(state.contains("if w2a8_requested"));
    for symbol in [
        "exl3_w2a8_h128_pre_dual_emit_h4096",
        "exl3_w2a8_h128_post_silu_pre_emit_h2048",
        "exl3_w2a8_grouped_prefill_k2_gu",
        "exl3_w2a8_grouped_prefill_k2_down",
    ] {
        assert!(
            state.contains(symbol),
            "missing conditional handle {symbol}"
        );
    }
}

#[test]
fn registered_cuda_wrappers_are_deepseek_only_and_exact_shape() {
    let root = workspace();
    let model_dir = root.join("kernels/gb10/deepseek-v4-flash/nvfp4");
    for (name, n, k, symbol) in [
        (
            "exl3_w2a8_grouped_prefill_k2_gu.cu",
            "2048",
            "4096",
            "exl3_w2a8_grouped_prefill_k2_gu",
        ),
        (
            "exl3_w2a8_grouped_prefill_k2_down.cu",
            "4096",
            "2048",
            "exl3_w2a8_grouped_prefill_k2_down",
        ),
    ] {
        let wrapper = fs::read_to_string(model_dir.join(name)).expect("model-specific wrapper");
        assert!(wrapper.contains(&format!("#define W2A8_FIXED_N {n}")));
        assert!(wrapper.contains(&format!("#define W2A8_FIXED_K {k}")));
        assert!(wrapper.contains(&format!("#define W2A8_KERNEL_NAME {symbol}")));
    }
    assert!(model_dir.join("exl3_w2a8_h128_emit.cu").is_file());
    for name in [
        "exl3_w2a8_grouped_prefill_k2_gu.cu",
        "exl3_w2a8_grouped_prefill_k2_down.cu",
        "exl3_w2a8_h128_emit.cu",
    ] {
        assert!(!Path::new("kernels/gb10/common").join(name).exists());
    }
}

#[test]
fn dispatch_is_capacity_checked_then_keeps_exact_five_launch_fallback() {
    let dispatch = source("crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs");
    assert!(dispatch.contains("checked_sidecar_layout"));
    assert!(dispatch.contains("let sizes = ctx.buffers.sizes()"));
    assert!(dispatch.contains("sizes.expert_gate_out"));
    assert!(dispatch.contains("sizes.expert_up_out"));
    assert!(dispatch.contains("sizes.expert_down_out"));
    assert!(dispatch.contains("ctx.comm.is_none()"));
    assert!(dispatch.contains("!st.down.svh_tab.is_null()"));
    for incompatible in ["pf.direct_m128", "pf.direct_n128", "pf.direct_n256"] {
        assert!(dispatch.contains(incompatible));
    }
    assert!(!dispatch.contains("fp8_act"));
    let mutation = dispatch.find("// MUTATION START").expect("mutation marker");
    let before = &dispatch[..mutation];
    let after = &dispatch[mutation..];
    assert!(before.contains("return Ok(false)"));
    assert!(!after.contains("return Ok(false)"));
    // Eight textual launch sites encode the N128/N256 fused-GU and N256-down
    // arms plus their incumbent fallbacks. Runtime remains five launches by
    // default and three when one fused-GU plus the N256-down gates qualify.
    assert_eq!(after.matches(".launch(stream)?").count(), 8);

    let names = [
        "w2a8_pre_dual_emit_k",
        "w2a8_grouped_gu_k",
        "w2a8_grouped_gu_k",
        "w2a8_post_silu_pre_emit_k",
        "w2a8_grouped_down_k",
    ];
    let mut cursor = mutation;
    for name in names {
        let position = dispatch[cursor..].find(name).expect("launch order") + cursor;
        assert!(position >= cursor);
        cursor = position + name.len();
    }
    assert!(dispatch.contains("let gate_fp8 = expert_down_out"));
    assert!(dispatch.contains("let up_fp8 = expert_up_out"));
    assert!(dispatch.contains("let incumbent_down_fp8 = expert_up_out"));
    assert!(dispatch.contains("let fused_down_fp8 = expert_gate_out"));
    assert!(dispatch.contains("checked_offset(gate_fp8, gu.fp8_bytes)"));
    assert!(dispatch.contains("checked_offset(up_fp8, gu.fp8_bytes)"));
    assert!(dispatch.contains("checked_offset(incumbent_down_fp8, down.fp8_bytes)"));
    assert!(dispatch.contains("checked_offset(fused_down_fp8, down.fp8_bytes)"));
}

#[test]
fn incumbent_dispatch_declines_before_any_existing_alias_write() {
    let incumbent = source("crates/spark-model/src/layers/moe/forward_prefill_exl3.rs");
    let decision = incumbent
        .find("try_run_exl3_w2a8_prefill")
        .expect("W2A8 decision");
    let incumbent_buffers = incumbent
        .find("let expert_gate_out = ctx.buffers.expert_gate_out()")
        .expect("incumbent mutation setup");
    assert!(decision < incumbent_buffers);
}

#[test]
fn successful_core_runs_the_incumbent_down_post_before_return_when_not_fused() {
    let incumbent = source("crates/spark-model/src/layers/moe/forward_prefill_exl3.rs");
    let start = incumbent
        .find("if self.try_run_exl3_w2a8_prefill")
        .expect("W2A8 success arm");
    let success = &incumbent[start..];
    let conditional = success
        .find("if !pf.fused_unpermute")
        .expect("non-fused tail condition");
    let post = success[conditional..]
        .find("launch_h128_post(")
        .expect("incumbent down H128 post")
        + conditional;
    let return_after_post = success[post..]
        .find("return Ok(())")
        .expect("return after down post")
        + post;
    assert!(conditional < post && post < return_after_post);

    let dispatch = source("crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs");
    let mutation = dispatch.find("// MUTATION START").unwrap();
    assert_eq!(dispatch[mutation..].matches(".launch(stream)?").count(), 8);

    let outer = source("crates/spark-model/src/layers/moe/forward_prefill.rs");
    let routed = outer
        .find("self.run_routed_grouped_gemm(")
        .expect("routed projection call");
    let fused_tail = outer[routed..]
        .find("try_exl3_fused_post_unpermute(")
        .expect("outer fused post-unpermute")
        + routed;
    assert!(routed < fused_tail);
}
