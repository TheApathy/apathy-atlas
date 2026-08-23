// SPDX-License-Identifier: AGPL-3.0-only

//! Source-level guards on retrieval pre-emption (static store and
//! self-context).
//!
//! # Why these are source greps
//!
//! The property that matters is negative — *retrieval cannot change which
//! tokens are emitted* — and it holds because retrieval touches only the
//! proposal half of the step. Verification, acceptance and emission never
//! read a retrieval value. A behavioural test cannot show the absence of
//! an influence; the seam can, and these tests fail the build the moment
//! someone adds one.
//!
//! # End-to-end verification procedure
//!
//! The bit-identity claim is checked against a live server like this:
//!
//! 1. serve the target at temperature 0 with `ATLAS_REST_STORE` and
//!    `ATLAS_SELF_CONTEXT_DRAFT` both unset, and record the completions
//!    for a fixed prompt set;
//! 2. restart with `ATLAS_SELF_CONTEXT_DRAFT=1` (then again with
//!    `ATLAS_REST_STORE=<store>`), same build, same seed, same prompts;
//! 3. every token stream must be byte-identical across all three runs.
//!    Only step time and the retrieval counters may differ.
//!
//! A divergence means something in the verify path started reading the
//! proposal's provenance, which is exactly what the greps below forbid.

/// The verify path may touch retrieval for accounting and nothing else.
#[test]
fn verification_reads_no_retrieval_state() {
    let verify = include_str!("verify_dflash_step.rs");
    assert!(
        verify.contains("crate::rest_store::record_accepted(num_accepted)"),
        "accepted-from-store must be reported from the count the verifier already computed"
    );
    assert!(
        verify.contains("crate::rest_store::self_context::record_accepted(num_accepted)"),
        "accepted-from-self-context must be reported the same way"
    );
    assert_eq!(
        verify.matches("rest_store::").count(),
        2,
        "the verifier's only direct store references are the two accepted-token counters"
    );
    assert_eq!(
        verify
            .matches("std::mem::take(&mut a.draft_origin)")
            .count(),
        1,
        "provenance is read from the sequence exactly once, at entry"
    );
    assert_eq!(
        verify.matches("match draft_origin {").count(),
        1,
        "and consumed exactly once, at accounting"
    );

    // The verifier also proposes the NEXT frame, so it does mention
    // retrieval — but only after this frame's acceptance is settled.
    // Ordering is the invariant that matters: nothing retrieval-related
    // may be read before `num_accepted` exists.
    let accepted_at = verify
        .find("let (num_accepted, tree_last_inter_slot)")
        .expect("acceptance is computed in this file");
    let accounting_at = verify
        .find("match draft_origin {")
        .expect("accounting follows acceptance");
    assert!(accepted_at < accounting_at);
    assert_eq!(
        verify
            .matches("retrieval_chain(a, proposal_token, num_drafts)")
            .count(),
        1,
        "the verifier pre-empts the NEXT frame from exactly one place"
    );
    let retrieval_at = verify
        .find("retrieval_chain(a, proposal_token, num_drafts)")
        .expect("the re-propose hook");
    assert!(
        accounting_at < retrieval_at,
        "retrieval may only run after this frame's acceptance is decided"
    );
    assert!(
        verify.contains(
            "let should_propose = !skip_propose && !skip_repropose_diag && retrieved.is_none();"
        ),
        "a retrieved chain must skip the drafter's forward pass, not run alongside it"
    );
    assert!(
        verify.contains("let draft_origin = std::mem::take(&mut a.draft_origin);"),
        "provenance must be consumed by value so no later frame inherits it"
    );
    for forbidden in [
        "rest_store::propose",
        "rest_store::preempt",
        "rest_store::store",
        "self_context.propose",
        "self_context::enabled",
    ] {
        assert!(
            !verify.contains(forbidden),
            "verification must never consult a retrieval tier: {forbidden}"
        );
    }
}

/// Pre-emption installs a flat chain through the pairing authority, and
/// only inside the branch the grammar gate already guards.
#[test]
fn preemption_installs_a_flat_chain_through_the_pairing_authority() {
    let source = include_str!("mtp_step.rs");
    assert_eq!(
        source
            .matches("retrieval_chain(a, tok, num_drafts)")
            .count(),
        1,
        "retrieval has exactly one call site: the Phase A bootstrap propose"
    );
    assert!(
        source.contains("proposal_lifecycle::install_external_flat(model, a, chain)"),
        "a retrieval chain must be published through the pairing authority, never assigned"
    );
    assert!(
        !source.contains("a.pending_drafts = "),
        "the ngram path's direct assignment must not be copied here"
    );

    let gate = "if !dflash_grammar_skip_propose(model, a) {";
    let gate_at = source.find(gate).expect("bootstrap grammar gate");
    let call_at = source
        .find("retrieval_chain(a, tok, num_drafts)")
        .expect("retrieval call site");
    assert!(
        gate_at < call_at,
        "retrieval must not draft while the grammar constrains output"
    );
    let branch_end = source[call_at..]
        .find("let _mtp_grammar_mask = mtp_grammar_mask_for(a);")
        .expect("DFlash fallthrough follows the retrieval branch");
    assert!(
        !source[call_at..call_at + branch_end].contains("pending_tree_payload"),
        "a retrieval frame is flat: it must never set tree topology"
    );
}

/// The drafter runs whenever every retrieval tier declines.
#[test]
fn the_drafter_runs_whenever_retrieval_declines() {
    let source = include_str!("mtp_step.rs");
    let call_at = source
        .find("retrieval_chain(a, tok, num_drafts)")
        .expect("retrieval call site");
    let tail = &source[call_at..];
    let else_at = tail.find("} else {").expect("decline branch");
    let propose_at = tail
        .find("match propose_and_install(model, a, tok, num_drafts,")
        .expect("unchanged DFlash proposal");
    assert!(
        else_at < propose_at,
        "the unchanged DFlash proposal must be the decline branch, not a sibling"
    );
}

/// Self-context is consulted before the static store, and both are gated.
#[test]
fn the_cascade_asks_self_context_first_and_both_tiers_are_gated() {
    let source = include_str!("mtp_step.rs");
    let helper = source
        .split("fn retrieval_chain(")
        .nth(1)
        .expect("the cascade helper");
    let body = helper
        .split("\n}\n")
        .next()
        .expect("the helper's body ends at its closing brace");

    let self_ctx = body
        .find("self_context::enabled()")
        .expect("self-context is gated by its own env flag");
    let store = body
        .find("crate::rest_store::preempt(")
        .expect("the static store is the second tier");
    assert!(
        self_ctx < store,
        "self-context must be asked before the static store: it is the tier specific to this sequence"
    );
    assert!(
        body.contains("a.self_context.propose(&a.seq.tokens, tok, num_drafts)"),
        "self-context must draft from the sequence's own committed history"
    );
    assert!(
        body.contains("num_drafts < crate::rest_store::MIN_PREEMPT_WIDTH"),
        "both tiers share the width gate that keeps frames on the generic verifier"
    );
}

/// Neither tier may displace a drafter that is currently winning.
#[test]
fn a_saturated_drafter_is_never_pre_empted() {
    let source = include_str!("mtp_step.rs");
    let body = source
        .split("fn retrieval_chain(")
        .nth(1)
        .expect("the cascade helper")
        .split("\n}\n")
        .next()
        .expect("its body");
    assert!(
        body.contains("let drafter_recent = usize::from(a.last_verify_accepted);"),
        "the gate must read what the drafter actually just did on this sequence"
    );
    assert!(
        body.contains("drafter_recent < crate::rest_store::self_context::max_drafter_accept()"),
        "self-context must check the drafter before drafting"
    );
    assert!(
        body.contains("drafter_recent >= crate::rest_store::max_drafter_accept()"),
        "the static store must check it too, at its own threshold"
    );

    // The evidence has to be recorded, and only after acceptance exists.
    let verify = include_str!("verify_dflash_step.rs");
    assert_eq!(
        verify
            .matches("a.last_verify_accepted = u16::try_from(num_accepted)")
            .count(),
        1,
        "the drafter's acceptance is recorded exactly once per verify"
    );
}
