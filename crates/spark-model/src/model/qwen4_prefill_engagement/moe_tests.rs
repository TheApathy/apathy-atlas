// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn moe_only_receipt_requires_all_48_layers_and_labels_serial_core() {
    let selectors = Selectors::from_values(None, None).with_moe_only(true);
    let mut capture = Capture::new(selectors, 256, tests::flash_geometry()).unwrap();
    for _ in 0..12 {
        capture.engage(PrefillPath::Attention, 256).unwrap();
    }
    for _ in 0..36 {
        capture.engage(PrefillPath::Ssm, 256).unwrap();
    }
    let lines = capture.finish().unwrap();
    assert_eq!(lines.len(), 2);
    for line in lines {
        assert!(line.contains("selector=ATLAS_QWEN4_PREFILL_MOE_BATCH value=1"));
        assert!(line.contains("ffn=grouped core=serial_token_ordered"));
        assert!(line.contains("attention_selector=1 ssm_selector=1"));
        assert!(!line.contains("parity=pass"));
    }
}

#[test]
fn moe_only_receipt_cannot_credit_just_attention_or_wrong_rows() {
    let selectors = Selectors::from_values(None, None).with_moe_only(true);
    let mut capture = Capture::new(selectors, 256, tests::flash_geometry()).unwrap();
    assert!(capture.engage(PrefillPath::Ssm, 255).is_err());
    for _ in 0..12 {
        capture.engage(PrefillPath::Attention, 256).unwrap();
    }
    assert!(capture.finish().is_err());
}

#[test]
fn absent_moe_only_mode_preserves_existing_selectors() {
    for attention in [None, Some("0"), Some("1")] {
        for ssm in [None, Some("0"), Some("1")] {
            let selectors = Selectors::from_values(attention, ssm);
            assert_eq!(selectors.with_moe_only(false), selectors);
        }
    }
}

#[test]
fn f12_receipt_distinguishes_recurrence_and_hyper_from_attention() {
    let mut selectors = Selectors::from_values(None, None).with_moe_only(true);
    selectors.hyper_exact = true;
    selectors.ssm_exact = true;
    selectors.moe_compact = true;
    let mut capture = Capture::new(selectors, 33, tests::flash_geometry()).unwrap();
    for _ in 0..12 {
        capture.engage(PrefillPath::Attention, 33).unwrap();
    }
    for _ in 0..36 {
        capture.engage(PrefillPath::Ssm, 33).unwrap();
    }
    let lines = capture.finish().unwrap();
    assert!(lines[0].contains("core=serial_token_ordered"));
    assert!(lines[1].contains("core=exact_projection_sequence_nosnap"));
    for line in lines {
        assert!(line.contains("hc=exact_m32"));
        assert!(line.contains("routed_schedule=original_compact"));
        assert!(!line.contains("parity=pass"));
    }
}

#[test]
fn f13_receipts_do_not_claim_exact_or_checked_tensorcore_projections() {
    for (mode, core) in [
        (Mode::Off, "exact_projection_sequence_nosnap"),
        (Mode::Out, "exact_qkvz_bf16_mma_output_sequence_nosnap"),
        (Mode::All, "bf16_mma_projections_sequence_nosnap"),
    ] {
        let mut selectors = Selectors::from_values(None, None).with_moe_only(true);
        selectors.hyper_exact = true;
        selectors.ssm_exact = true;
        selectors.moe_compact = true;
        selectors.ssm_gemm = mode;
        let mut capture = Capture::new(selectors, 65, tests::flash_geometry()).unwrap();
        for _ in 0..12 {
            capture.engage(PrefillPath::Attention, 65).unwrap();
        }
        for _ in 0..36 {
            capture.engage(PrefillPath::Ssm, 65).unwrap();
        }
        let lines = capture.finish().unwrap();
        assert!(lines[0].contains("core=serial_token_ordered"));
        assert!(lines[1].contains(&format!("core={core} ")));
        if mode != Mode::Off {
            assert!(!lines[1].contains("core=exact_projection_sequence_nosnap"));
        }
        for line in lines {
            assert!(line.contains(&format!("ssm_projection_gemm={}", mode.as_str())));
            assert!(line.contains("hc=exact_m32"));
            assert!(line.contains("routed_schedule=original_compact"));
            assert!(!line.contains("parity=pass"));
        }
    }
}
