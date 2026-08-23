// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Mutex;

use super::*;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn reset() {
    abort();
    completed_flag().store(false, Ordering::Relaxed);
}

#[test]
fn reports_manifest_and_first_exact_byte_divergence() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset();
    begin_serial(40, &[10, 11]).unwrap();
    for stages in [[vec![1, 2], vec![3, 4]], [vec![5, 6], vec![7, 8]]] {
        record_serial("embed", stages[0].clone()).unwrap();
        record_serial("layer_00", stages[1].clone()).unwrap();
        finish_serial_row().unwrap();
    }
    finish_serial().unwrap();
    begin_batch(40, &[10, 11]).unwrap();
    record_batch("embed", &[1, 2, 5, 6], 2).unwrap();
    record_batch("layer_00", &[3, 4, 7, 9], 2).unwrap();
    let report = finish_batch().unwrap();
    let first = report.first.unwrap();
    assert_eq!(
        (first.stage.as_str(), first.row, first.first_byte),
        ("layer_00", 1, 1)
    );
    assert_eq!(first.mismatch_rows, [1]);
    assert_eq!(report.manifest.absolute_seq_lens, [40, 41]);
    assert_eq!(report.manifest.family, BASELINE_FAMILY);
    assert_eq!(report.terminal_stage, "layer_00");
    assert!(!report.logits_compared);
    assert!(!selector_matches(40, &[10, 11], 40, &[10, 11], true));
    reset();
}

#[test]
fn permits_only_the_named_controlled_serial_family_overlaps() {
    validate_serial_control_overlap(false, Some("ffn")).unwrap();
    validate_serial_control_overlap(true, None).unwrap();
    validate_serial_control_overlap(true, Some(CONTROLLED_SERIAL_FAMILY)).unwrap();
    validate_serial_control_overlap(true, Some(CONTROLLED_LM_HEAD_FAMILY)).unwrap();
    for family in ["ffn", "layer_norms", "final_norm", "lm_head"] {
        let error = validate_serial_control_overlap(true, Some(family)).unwrap_err();
        assert!(error.to_string().contains(CONTROLLED_SERIAL_FAMILY));
        assert!(error.to_string().contains(CONTROLLED_LM_HEAD_FAMILY));
        assert!(error.to_string().contains(family));
    }
}

#[test]
fn lm_head_family_captures_exactly_through_final_norm() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset();
    begin_serial_with_family(7, &[3], Some(CONTROLLED_LM_HEAD_FAMILY)).unwrap();
    assert!(capture_stage("final_norm"));
    assert!(!capture_stage("logits"));
    record_serial("final_norm", vec![1, 2]).unwrap();
    finish_serial_row().unwrap();
    finish_serial().unwrap();
    begin_batch_with_family(7, &[3], Some(CONTROLLED_LM_HEAD_FAMILY)).unwrap();
    assert!(capture_stage("final_norm"));
    assert!(!capture_stage("logits"));
    record_batch("final_norm", &[1, 2], 2).unwrap();
    let report = finish_batch().unwrap();
    assert!(report.first.is_none());
    assert_eq!(report.manifest.family, CONTROLLED_LM_HEAD_FAMILY);
    assert_eq!(report.terminal_stage, "final_norm");
    assert!(!report.logits_compared);
    assert_eq!(report.stages, 1);
    reset();
}

#[test]
fn records_and_cross_checks_controlled_serial_family() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset();
    begin_serial_with_family(7, &[3], Some(CONTROLLED_SERIAL_FAMILY)).unwrap();
    record_serial("embed", vec![1, 2]).unwrap();
    finish_serial_row().unwrap();
    finish_serial().unwrap();
    begin_batch_with_family(7, &[3], Some(CONTROLLED_SERIAL_FAMILY)).unwrap();
    record_batch("embed", &[1, 2], 2).unwrap();
    let report = finish_batch().unwrap();
    assert_eq!(report.manifest.family, CONTROLLED_SERIAL_FAMILY);

    reset();
    begin_serial_with_family(7, &[3], Some(CONTROLLED_SERIAL_FAMILY)).unwrap();
    record_serial("embed", vec![1, 2]).unwrap();
    finish_serial_row().unwrap();
    finish_serial().unwrap();
    let error = begin_batch_with_family(7, &[3], None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("control-family identity mismatch")
    );
    reset();
}

#[test]
fn successful_match_disarms_the_selector() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset();
    begin_serial_with_family(9, &[4], None).unwrap();
    record_serial("embed", vec![1, 2]).unwrap();
    finish_serial_row().unwrap();
    finish_serial().unwrap();
    begin_batch_with_family(9, &[4], None).unwrap();
    record_batch("embed", &[1, 2], 2).unwrap();
    let report = finish_batch().unwrap();
    assert!(report.first.is_none());
    assert!(completed_flag().load(Ordering::Relaxed));
    assert!(!selector_matches(
        9,
        &[4],
        9,
        &[4],
        completed_flag().load(Ordering::Relaxed)
    ));
    reset();
}

#[test]
fn selector_is_strict_and_matches_only_requested_frame() {
    assert_eq!(parse_selector(Some("42")).unwrap(), 42);
    assert_eq!(
        parse_tokens_selector(Some("1,2,4294967295")).unwrap(),
        [1, 2, u32::MAX]
    );
    assert!(parse_selector(None).is_err());
    assert!(parse_selector(Some("")).is_err());
    assert!(parse_selector(Some(" 42")).is_err());
    assert!(parse_selector(Some("-1")).is_err());
    assert!(parse_tokens_selector(None).is_err());
    assert!(parse_tokens_selector(Some("")).is_err());
    assert!(parse_tokens_selector(Some("1, 2")).is_err());
    assert!(parse_tokens_selector(Some("1,-2")).is_err());
    assert!(parse_tokens_selector(Some("4294967296")).is_err());
    assert!(selector_matches(42, &[1, 2], 42, &[1, 2], false));
    assert!(!selector_matches(42, &[1, 2], 41, &[1, 2], false));
    assert!(!selector_matches(42, &[1, 2], 42, &[1, 3], false));
    assert!(!selector_matches(42, &[1, 2], 42, &[1, 2], true));
}

#[test]
fn rejects_incomplete_or_out_of_order_capture() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset();
    begin_serial(0, &[1]).unwrap();
    record_serial("embed", vec![1]).unwrap();
    assert!(finish_serial().is_err());
    reset();
    begin_serial(0, &[1]).unwrap();
    record_serial("embed", vec![1]).unwrap();
    finish_serial_row().unwrap();
    finish_serial().unwrap();
    begin_batch(0, &[1]).unwrap();
    assert!(record_batch("wrong", &[1], 1).is_err());
    reset();

    begin_serial(5, &[1]).unwrap();
    record_serial("embed", vec![1, 2]).unwrap();
    finish_serial_row().unwrap();
    finish_serial().unwrap();
    assert!(begin_batch(6, &[1]).is_err());
    reset();

    begin_serial(5, &[1]).unwrap();
    record_serial("embed", vec![1, 2]).unwrap();
    finish_serial_row().unwrap();
    finish_serial().unwrap();
    begin_batch(5, &[1]).unwrap();
    assert!(record_batch("embed", &[1], 1).is_err());
    reset();
}
