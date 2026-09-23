// SPDX-License-Identifier: AGPL-3.0-only
//! Equal fixed inputs through ordinary decode and the actual policy transaction.
use super::{
    artifacts::record_comparison,
    forced_policy::ForcedPrefixPolicy,
    session::Session,
    snapshots::{self, Raw, capture_rows},
};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use spark_model::model::glm53::verify_policy_transaction::VerifyOutcome;
use spark_model::traits::Model;
use std::path::Path;

fn nonce(frame: &Value) -> Result<u64> {
    frame["nonce"]
        .as_u64()
        .context("missing numeric probe nonce")
}

fn continuation(session: &mut Session, token: u32, anchor: u32) -> Result<(Raw, Vec<u32>)> {
    let seq = session
        .seq
        .as_mut()
        .context("missing continuation sequence")?;
    session.logits = session.model.decode(token, seq, session.stream)?;
    let logits = snapshots::next_logits(&mut session.model, session.stream)?;
    let drafts =
        session
            .model
            .run_mtp_propose_multi(anchor, seq.seq_len, 7, seq, session.stream, None)?;
    ensure!(
        drafts.len() == 7,
        "actual proposer did not return seven draft IDs"
    );
    Ok((logits, drafts))
}

pub fn run_case(
    session: &mut Session,
    corpus: &[u32],
    start: usize,
    rows: usize,
    output: &Path,
) -> Result<Value> {
    ensure!(
        (1..=8).contains(&rows) && start > 0,
        "invalid forced case geometry"
    );
    ensure!(
        start
            .checked_add(8)
            .is_some_and(|end| end <= 2047 && end <= corpus.len())
            && start
                .checked_add(rows + 1)
                .is_some_and(|end| end <= 2047 && end <= corpus.len()),
        "case must fit full staging and a fresh continuation within drafter capacity"
    );
    let inputs = corpus[start..start + 8].to_vec();
    let next_token = corpus[start + rows];
    // This identical anchor isolates context effects; it is not a sampled quality claim.
    let proposal_anchor = next_token;

    session.prefix(&corpus[..start])?;
    let before_scalar = snapshots::persistent(&mut session.model, session.stream)?;
    let mut scalar_captures = Vec::new();
    for &token in &inputs[..rows] {
        let seq = session.seq.as_mut().context("missing scalar sequence")?;
        session.logits = session.model.decode(token, seq, session.stream)?;
        scalar_captures.push(
            capture_rows(&mut session.model, session.stream, 1)?
                .pop()
                .context("missing scalar capture")?,
        );
    }
    let scalar = snapshots::persistent(&mut session.model, session.stream)?;
    let (scalar_next, scalar_drafts) = continuation(session, next_token, proposal_anchor)?;

    session.prefix(&corpus[..start])?;
    let before_candidate = snapshots::persistent(&mut session.model, session.stream)?;
    let before = record_comparison(
        output,
        "before",
        &before_scalar.regions,
        &before_candidate.regions,
    )?;
    let before_frames = json!({"scalar":before_scalar.frame,"candidate":before_candidate.frame});
    drop(before_scalar.regions);
    drop(before_candidate.regions);
    let seq = session.seq.as_mut().context("missing candidate sequence")?;
    let bound = session
        .model
        .bind_verify_policy_request(&inputs, seq, session.stream)?;
    let mut policy = ForcedPrefixPolicy::new(inputs.clone(), rows)?;
    let outcome = session
        .model
        .decode_verify_with_policy(bound, &mut policy)?;
    let VerifyOutcome::Committed(receipt) = outcome else {
        bail!("forced policy requested ordinary replay");
    };
    ensure!(
        receipt.accepted_drafts() == rows - 1,
        "forced accepted prefix changed"
    );
    let published = receipt.publish(&mut seq.tokens, &mut seq.seq_len, &mut seq.kv_valid_tokens)?;
    let emitted = published.emitted_tokens().to_vec();
    let publication_exact = policy.publication_matches(&emitted, published.terminal());
    super::json_file(
        &output.join("publication.json"),
        &json!({
            "exact":publication_exact,"emitted":emitted,"terminal":published.terminal(),
            "seq_len":seq.seq_len,"kv_valid_tokens":seq.kv_valid_tokens,
            "host_prefix_exact":seq.tokens == corpus[..start + rows]
        }),
    )?;
    ensure!(
        seq.seq_len == start + rows
            && seq.kv_valid_tokens == seq.seq_len
            && seq.tokens == corpus[..start + rows],
        "host committed prefix differs from fixed inputs"
    );
    let candidate_captures = capture_rows(&mut session.model, session.stream, rows as u32)?;
    let candidate = snapshots::persistent(&mut session.model, session.stream)?;
    let state = record_comparison(output, "state", &scalar.regions, &candidate.regions)?;
    let capture = record_comparison(
        output,
        "capture",
        &snapshots::capture_map(scalar_captures),
        &snapshots::capture_map(candidate_captures),
    )?;
    drop(scalar.regions);
    drop(candidate.regions);
    let frames_match = scalar.frame["position"] == candidate.frame["position"]
        && scalar.frame["context"] == candidate.frame["context"];
    let fresh = nonce(&scalar.frame)? > nonce(&before_frames["scalar"])?
        && nonce(&candidate.frame)? > nonce(&before_frames["candidate"])?;
    super::json_file(
        &output.join("frames.json"),
        &json!({
            "exact":frames_match && fresh,"frames_match":frames_match,"nonce_fresh":fresh,
            "before":before_frames,"scalar":scalar.frame,"candidate":candidate.frame
        }),
    )?;
    let (candidate_next, candidate_drafts) = continuation(session, next_token, proposal_anchor)?;
    let next = record_comparison(output, "next_logits", &scalar_next, &candidate_next)?;
    let drafts_exact = scalar_drafts == candidate_drafts;
    let exact = [&before, &state, &capture, &next]
        .iter()
        .all(|gate| gate["exact"] == true)
        && drafts_exact
        && frames_match
        && fresh
        && publication_exact;
    Ok(
        json!({"kind":"forced-prefix-state-parity","natural_acceptance":false,"exact":exact,
        "start":start,"rows":rows,"inputs":inputs,"next_token":next_token,"proposal_anchor":proposal_anchor,
        "emitted":emitted,"terminal":published.terminal(),"publication_exact":publication_exact,
        "before_frames":before_frames,"frames_match":frames_match,"nonce_fresh":fresh,
        "scalar_frame":scalar.frame,"candidate_frame":candidate.frame,
        "before":before,"state":state,"capture":capture,"next_logits":next,
        "drafts_exact":drafts_exact,"scalar_drafts":scalar_drafts,"candidate_drafts":candidate_drafts}),
    )
}
