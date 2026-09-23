// SPDX-License-Identifier: AGPL-3.0-only

const VERIFY: &str = include_str!("verify_dflash_step.rs");

#[test]
fn terminal_draft_and_bonus_emissions_each_close_the_receipt() {
    let drafts = VERIFY.split("// Emit accepted drafts.").nth(1).unwrap();
    let drafts = drafts.split("// Bonus token").next().unwrap();
    assert!(drafts.contains("receipt.observe_draft("));
    assert!(drafts.contains("ReceiptExit::TerminalDraft"));
    assert!(drafts.find("receipt.observe_draft(").unwrap() < drafts.find("if a.finished").unwrap());
    assert!(drafts.find("ReceiptExit::TerminalDraft").unwrap() < drafts.find("return;").unwrap());

    let bonus = VERIFY.split("// Bonus token =").nth(1).unwrap();
    let bonus = bonus.split("// Item #2").next().unwrap();
    assert!(bonus.contains("receipt.observe_bonus("));
    assert!(bonus.contains("ReceiptExit::TerminalBonus"));
    assert!(bonus.find("receipt.observe_bonus(").unwrap() < bonus.find("if a.finished").unwrap());
    assert!(bonus.find("ReceiptExit::TerminalBonus").unwrap() < bonus.find("return;").unwrap());
    assert!(bonus.contains("ReceiptExit::Continue"));
}

#[test]
fn receipt_is_request_bound_and_keeps_exact_output_accounting() {
    for field in [
        "request_start = ?a.request_start",
        "session_hash = a.session_hash",
        "pre_verify_len",
        "emitted_drafts",
        "emitted_bonus",
        "output_tokens_added",
        "injected_output_tokens",
        "DFLASH VERIFY_RECEIPT",
    ] {
        assert!(VERIFY.contains(field), "missing receipt field {field}");
    }
    assert_eq!(
        VERIFY
            .matches("let before_emit = a.output_tokens.len();")
            .count(),
        2
    );
    let compact: String = VERIFY.split_whitespace().collect();
    assert!(
        compact
            .contains("DflashVerifyReceipt::new(drafts.len(),num_accepted,a.output_tokens.len())")
    );
}

#[test]
fn existing_emit_and_state_commit_order_is_preserved() {
    assert_eq!(VERIFY.matches("emit_token(a, drafts[i], None);").count(), 1);
    assert_eq!(VERIFY.matches("emit_token(a, bonus, None);").count(), 1);
    let draft = VERIFY.find("emit_token(a, drafts[i], None);").unwrap();
    let bonus = VERIFY.find("emit_token(a, bonus, None);").unwrap();
    let last = VERIFY.find("a.last_token = bonus;").unwrap();
    let commit = VERIFY.find("model.commit_accepted_prefix(").unwrap();
    assert!(draft < bonus && bonus < last && last < commit);
    assert_eq!(VERIFY.matches("\"DFLASH K=γ verify:").count(), 1);
    assert_eq!(
        VERIFY.matches("crate::metrics::SPEC_DECODE_VERIFY").count(),
        1
    );
    let logger = VERIFY.split("fn log_dflash_receipt(").nth(1).unwrap();
    for forbidden in [
        "model.",
        "synchronize(",
        "emit_token(",
        "a.finished =",
        "a.last_token =",
    ] {
        assert!(
            !logger.contains(forbidden),
            "receipt logger has inference effect {forbidden}"
        );
    }
}
