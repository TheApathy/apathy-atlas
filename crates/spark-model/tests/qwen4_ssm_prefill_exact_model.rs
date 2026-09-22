// SPDX-License-Identifier: AGPL-3.0-only

use std::{fs, path::PathBuf};

fn source(name: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name))
        .expect("production source must exist")
}

fn ordered(code: &str, markers: &[&str]) {
    let mut tail = code;
    for marker in markers {
        let index = tail
            .find(marker)
            .unwrap_or_else(|| panic!("missing {marker}"));
        tail = &tail[index + marker.len()..];
    }
}

#[test]
fn preflight_precedes_every_effect_and_requires_both_hcs() {
    let forward = source("src/layers/qwen3_ssm/qwen4_prefill_exact_forward.rs");
    ordered(
        &forward,
        &[
            "self.preflight_qwen4_prefill_exact(",
            "attn.prepare_prefill_exact(",
            "self.project_prefill_exact_tiles(",
            "ops::dense_gemv_ba_gates_batchn(",
            "ops::conv1d_update_l2norm_f32_sequence(",
            "ops::gdn_decode_f32_sequence(",
            "ops::gated_rms_norm_f32_multi_seq(",
            "self.project_prefill_exact_tiles(",
            "attn.inject_saved_batched(",
        ],
    );
    let admission = source("src/layers/qwen3_ssm/qwen4_prefill_exact.rs");
    for marker in [
        "ATLAS_QWEN4_PREFILL_SSM_EXACT",
        "hyper_selected()?",
        "validate_prefill_exact(",
        "self.qwen4_attn_hyper",
        "self.qwen4_mlp_hyper",
        "validate_regions(",
        "self.qkvz_fp8w.is_none()",
        "self.out_proj_fp8w.is_none()",
        "self.out_proj_dense.is_none()",
    ] {
        assert!(admission.contains(marker), "missing {marker}");
    }
}

#[test]
fn staged_core_uses_no_snapshots_and_exact_projection_tiles() {
    let forward = source("src/layers/qwen3_ssm/qwen4_prefill_exact_forward.rs");
    assert!(forward.contains("self.gdn_f32_sequence_nosnap_k"));
    assert_eq!(forward.matches("DevicePtr::NULL").count(), 2);
    assert!(forward.contains("ops::w4a16_gemv_batch_logits_exact_with("));
    assert!(forward.contains("let use_rt2 = ops::w4a16_gemv_rt2_enabled();"));
    assert!(forward.contains("register-tiled exact family is explicitly selected"));
    assert!(forward.contains("SSM_PREFILL_EXACT_RT2_ENGAGED"));
    assert!(forward.contains("ops::w4a16_decode_gemv("));
    for marker in [
        "w4a16_gemm(",
        "gdn_f32_sequence_persistent",
        "gdn_f32_sequence_lazyfinal",
        "copy_d2d",
        "gpu.alloc(",
        "self.ssm_forward(",
    ] {
        assert!(
            !forward.contains(marker),
            "unexpected alternate path {marker}"
        );
    }
}

#[test]
fn default_serial_core_and_single_finish_are_preserved() {
    let code = source("src/layers/qwen3_ssm/qwen4_prefill_moe.rs");
    ordered(
        &code,
        &[
            "qwen4_prefill_exact::selected()?",
            "self.prefill_qwen4_ssm_exact(",
            "for row in 0..rows",
            "self.ssm_forward(mixed, ssm_state, ctx, stream, trace)?",
            "qwen4_prefill_moe::finish(",
            "PrefillPath::Ssm",
        ],
    );
    assert_eq!(code.matches("qwen4_prefill_moe::finish(").count(), 1);
    ordered(
        &code,
        &[
            "attn_hyper.validate_prefill_exact(",
            "mlp_hyper.validate_prefill_exact(",
            "attn_hyper.prepare_prefill_exact(",
            "for row in 0..rows",
        ],
    );
    assert!(code.contains("(residual_row, Some(residual_row.offset(inject_offset)))"));
    assert!(source("src/layers/qwen3_ssm/mod.rs").contains("pub(crate) mod qwen4_prefill_exact;"));
}
