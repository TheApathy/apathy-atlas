// SPDX-License-Identifier: AGPL-3.0-only

#[path = "dflash_verify_receipt.rs"]
mod subject;
use subject::{DflashVerifyReceipt, ReceiptExit};

#[test]
fn terminal_inside_accepted_prefix_keeps_verified_and_emitted_counts_distinct() {
    let mut r = DflashVerifyReceipt::new(7, 7, 1);
    r.observe_draft(10, 1, &[99, 10]);
    r.observe_draft(11, 2, &[99, 10, 11]);
    let s = r.finish(ReceiptExit::TerminalDraft, 3);
    assert_eq!((s.proposed, s.accepted, s.rejected), (7, 7, 0));
    assert_eq!(
        (s.attempted_drafts, s.emitted_drafts, s.emitted_bonus),
        (2, 2, 0)
    );
    assert_eq!((s.output_tokens_added, s.injected_output_tokens), (2, 0));
    assert_eq!(s.error, None);
}

#[test]
fn terminal_bonus_is_counted_once_after_partial_acceptance() {
    let mut r = DflashVerifyReceipt::new(7, 1, 0);
    r.observe_draft(10, 0, &[10]);
    r.observe_bonus(20, 1, &[10, 20]);
    let s = r.finish(ReceiptExit::TerminalBonus, 2);
    assert_eq!(
        (s.accepted, s.rejected, s.emitted_drafts, s.emitted_bonus),
        (1, 6, 1, 1)
    );
    assert_eq!(s.output_tokens_added, 2);
    assert_eq!(s.error, None);
}

#[test]
fn no_accepted_draft_still_records_terminal_bonus() {
    let mut r = DflashVerifyReceipt::new(7, 0, 4);
    r.observe_bonus(20, 4, &[1, 2, 3, 4, 20]);
    let s = r.finish(ReceiptExit::TerminalBonus, 5);
    assert_eq!(
        (s.emitted_drafts, s.emitted_bonus, s.output_tokens_added),
        (0, 1, 1)
    );
    assert_eq!(s.error, None);
}

#[test]
fn cancel_before_append_is_not_reported_as_an_emitted_draft() {
    let mut r = DflashVerifyReceipt::new(7, 7, 1);
    r.observe_draft(10, 1, &[99]);
    let s = r.finish(ReceiptExit::TerminalDraft, 1);
    assert_eq!(
        (s.attempted_drafts, s.emitted_drafts, s.output_tokens_added),
        (1, 0, 0)
    );
    assert_eq!(s.error, None);
}

#[test]
fn suppressed_control_tokens_do_not_inflate_emitted_counts() {
    let mut r = DflashVerifyReceipt::new(7, 2, 1);
    r.observe_draft(10, 1, &[99]);
    r.observe_draft(11, 1, &[99, 11]);
    r.observe_bonus(12, 2, &[99, 11, 12]);
    let s = r.finish(ReceiptExit::Continue, 3);
    assert_eq!(
        (s.attempted_drafts, s.emitted_drafts, s.emitted_bonus),
        (2, 1, 1)
    );
    assert_eq!(s.output_tokens_added, 2);
    assert_eq!(s.error, None);
}

#[test]
fn injected_grammar_close_is_separate_from_the_original_token() {
    let mut r = DflashVerifyReceipt::new(7, 7, 0);
    r.observe_draft(10, 0, &[10, 80, 81]);
    let s = r.finish(ReceiptExit::TerminalDraft, 3);
    assert_eq!(
        (s.emitted_drafts, s.emitted_bonus, s.output_tokens_added),
        (1, 0, 3)
    );
    assert_eq!(s.injected_output_tokens, 2);
    assert_eq!(s.error, None);
}

#[test]
fn complete_acceptance_matches_output_ledger_including_eos() {
    let mut r = DflashVerifyReceipt::new(7, 7, 0);
    let mut output = Vec::new();
    for token in 0..7 {
        let before = output.len();
        output.push(token);
        r.observe_draft(token, before, &output);
    }
    output.push(99); // EOS is counted by output_tokens even when not sent as text.
    r.observe_bonus(99, 7, &output);
    let s = r.finish(ReceiptExit::TerminalBonus, 8);
    assert_eq!(
        (s.emitted_drafts, s.emitted_bonus, s.output_tokens_added),
        (7, 1, 8)
    );
    assert_eq!(s.error, None);
}

#[test]
fn malformed_accounting_never_produces_a_valid_receipt() {
    assert!(
        DflashVerifyReceipt::new(0, 0, 0)
            .finish(ReceiptExit::Continue, 0)
            .error
            .is_some()
    );
    assert!(
        DflashVerifyReceipt::new(7, 8, 0)
            .finish(ReceiptExit::Continue, 0)
            .error
            .is_some()
    );
    let mut extra = DflashVerifyReceipt::new(7, 0, 0);
    extra.observe_draft(1, 0, &[1]);
    assert!(extra.finish(ReceiptExit::TerminalDraft, 1).error.is_some());
    let mut gap = DflashVerifyReceipt::new(7, 1, 0);
    gap.observe_draft(1, 1, &[9, 1]);
    assert!(gap.finish(ReceiptExit::TerminalDraft, 2).error.is_some());
    let mut wrong = DflashVerifyReceipt::new(7, 1, 0);
    wrong.observe_draft(1, 0, &[2]);
    assert!(wrong.finish(ReceiptExit::TerminalDraft, 1).error.is_some());
    let mut shrunk = DflashVerifyReceipt::new(7, 1, 1);
    shrunk.observe_draft(1, 1, &[]);
    assert!(shrunk.finish(ReceiptExit::TerminalDraft, 0).error.is_some());
}

#[test]
fn duplicate_bonus_and_missing_bonus_are_not_valid_continuations() {
    let mut r = DflashVerifyReceipt::new(7, 0, 0);
    r.observe_bonus(1, 0, &[1]);
    r.observe_bonus(2, 1, &[1, 2]);
    assert!(r.finish(ReceiptExit::Continue, 2).error.is_some());
    assert!(
        DflashVerifyReceipt::new(7, 0, 0)
            .finish(ReceiptExit::Continue, 0)
            .error
            .is_some()
    );
}
