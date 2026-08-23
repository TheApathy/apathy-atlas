// SPDX-License-Identifier: AGPL-3.0-only

//! Full sampler-policy oracle for speculative content tokens.
//!
//! A speculative verifier produces raw target argmaxes, but normal serving
//! samples after grammar masks, think masks, penalties, logit bias, and the
//! adaptive sampler.  Those are not optional decorations: Qwen presets ship
//! with non-neutral presence/LZ/DRY penalties, so comparing drafts to raw
//! argmaxes changes the output from the very first bootstrap step.

use super::*;

/// Element layout of the resident target verify-logits buffer.
///
/// Keep this explicit instead of inferring the width from a row length: both
/// BF16 and FP32 rows are byte slices, and interpreting an FP32 row as BF16
/// changes the sampler oracle rather than producing a useful error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VerifyLogitsFormat {
    Bf16,
    Fp32,
}

impl VerifyLogitsFormat {
    fn elem_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp32 => 4,
        }
    }

    fn is_fp32(self) -> bool {
        matches!(self, Self::Fp32)
    }
}

/// Whether raw greedy argmax is insufficient to reproduce the plain sampler.
pub(super) fn content_policy_required(a: &ActiveSeq, ctx: &ThinkSpecCtx<'_>) -> bool {
    ctx.adaptive_sampling
        || !fast_path_seq_eligible(a)
        || a.think_ended
        || a.think_just_ended
        || a.suppress_tool_call
        || a.grammar_state.is_some()
}

/// Whether a non-thinking DDTree may safely use the target's per-row greedy
/// argmaxes instead of the linear host policy walk.
///
/// `content_policy_required` is deliberately true for every post-thinking
/// sequence because think tags remain masked. Post-thinking trees stay
/// fail-closed even for neutral greedy requests: the current tree-row mask
/// cannot distinguish a safe no-op from FP32 logits, a disabled mask, D2H
/// failure, or an unresolved compact-row layout. `mtp_step` therefore drops
/// only the topology and preserves the top-1 spine for the BF16/FP32-capable
/// linear policy walk. Any other history-dependent or request-specific policy
/// remains fail-closed because its result depends on the chosen branch prefix.
pub(super) fn tree_content_raw_argmax_eligible(a: &ActiveSeq, ctx: &ThinkSpecCtx<'_>) -> bool {
    !ctx.adaptive_sampling
        && fast_path_seq_eligible(a)
        && !a.think_ended
        && !a.think_just_ended
        && !a.suppress_tool_call
        && !a.require_tool_call
        && a.output_tokens.len() >= a.min_tokens
        && a.grammar_state.is_none()
}

/// Sample and commit one non-thinking MTP bootstrap token through the same
/// host-side policy pipeline as `process_decode_logits`.
pub(super) fn bootstrap_content_token(
    model: &dyn Model,
    a: &mut ActiveSeq,
    logits: DevicePtr,
    ctx: &ThinkSpecCtx<'_>,
) -> Option<u32> {
    debug_assert!(!a.inside_thinking);
    let vocab = model.vocab_size();
    let logits_fp32 = model.decode_logits_fp32();
    let elem_bytes = if logits_fp32 { 4 } else { 2 };
    let mut buf = vec![0u8; vocab * elem_bytes];
    if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
        tracing::error!("spec-policy bootstrap: logits D2H failed: {e:#}");
        a.finished = true;
        return None;
    }
    let trace = std::env::var("ATLAS_SPEC_BOOTSTRAP_TRACE").ok().as_deref() == Some("1");
    let raw_argmax = trace.then(|| {
        (0..vocab)
            .max_by(|&x, &y| {
                let read = |j: usize| {
                    if logits_fp32 {
                        let off = j * 4;
                        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
                    } else {
                        bf16_to_f32(buf[j * 2], buf[j * 2 + 1])
                    }
                };
                read(x).total_cmp(&read(y))
            })
            .unwrap_or(0) as u32
    });
    let (tok, lp) = process_seq_logits(
        a,
        &buf,
        0,
        vocab,
        elem_bytes,
        logits_fp32,
        a.think_end_token,
        a.think_start_token,
        a.tool_call_start_token,
        a.tool_call_end_token,
        ctx.reflection_suppress_ids,
        ctx.adaptive_sampling,
    );
    let output_len = a.output_tokens.len();
    if trace {
        tracing::info!(
            raw_argmax = raw_argmax.unwrap_or(0),
            sampled = tok,
            think_ended = a.think_ended,
            think_start = ?a.think_start_token,
            think_end = ?a.think_end_token,
            "SPEC_POLICY_BOOTSTRAP"
        );
    }
    emit_token(a, tok, lp);
    // Plain single-token decode snapshots the recurrent state after a
    // non-EOS content boundary.  This bootstrap has advanced exactly one
    // input position, so taking the same snapshot is safe (unlike a
    // multi-row speculative verify, whose state is still over-extended).
    if a.output_tokens.len() > output_len && !a.inside_thinking && !a.eos_tokens.contains(&tok) {
        rollback::snapshot_boundary_if_ssm(a, model);
    }
    Some(tok)
}

/// Environment kill-switch for the raw-BF16 accept fast path.
///
/// Default ON. Set `ATLAS_ACCEPT_FAST_ARGMAX=0` to force every row through
/// the full `process_seq_logits` walk (use only when debugging a drift the
/// fast path might have introduced — the path is intended to be
/// byte-identical by construction).
fn accept_fast_argmax_enabled() -> bool {
    std::env::var("ATLAS_ACCEPT_FAST_ARGMAX").as_deref() != Ok("0")
}

/// Argmax over a host-copied BF16 logits row with **last-wins** tie
/// semantics, skipping a set of token indices.
///
/// The full walk's greedy branch is `raw_logits.iter().enumerate().max_by(
/// |a, b| a.1.partial_cmp(b.1).unwrap_or(Equal))`, which returns the LAST
/// maximal element on ties. A strict `>` scan would return the first, so a
/// byte-identical fast path must use `>=` to match `max_by`. BF16-as-i16
/// ordering is valid for all finite values (same trick as
/// `argmax_bf16_skip_tokens` in `verify_dflash_step.rs`).
fn argmax_bf16_skip_last_wins(bytes: &[u8], skip_toks: &[u32], vocab: usize) -> Option<u32> {
    let mut best_tok: Option<u32> = None;
    let mut best_val = i16::MIN;
    for tok in 0..vocab {
        if skip_toks.contains(&(tok as u32)) {
            continue;
        }
        let signed = u16::from_le_bytes([bytes[2 * tok], bytes[2 * tok + 1]]) as i16;
        if best_tok.is_none() || signed >= best_val {
            best_val = signed;
            best_tok = Some(tok as u32);
        }
    }
    best_tok
}

/// Per-row gate for the raw-BF16 accept fast path.
///
/// When all of these hold, the full walk reduces to a masked argmax over
/// the raw BF16 row:
///
/// - `fast_path_seq_eligible(a)`: temp 0, all penalties neutral, no
///   logit_bias, no logprobs — the sampler's greedy bypass is a pure
///   post-intervention argmax.
/// - no adaptive sampling (entropy/zone side effects must stay on the full
///   path).
/// - not inside thinking: reflection suppression, think-efficiency wave,
///   F2 confidence early-stop, and tool-call hard masks all apply only
///   while thinking.
/// - `!suppress_tool_call` / `!require_tool_call`: the −12 tool bias and
///   the one-shot pin-to-tool-call-start would re-order the argmax.
/// - no grammar: the grammar bitmask would re-order logits.
/// - BF16 rows only: the FP32 path needs an f32 scan, not the i16 trick.
///
/// The `</think>`/`<think>` masks applied by the full walk when
/// `think_ended` are reproduced here as a skip list; masking to −inf and
/// skipping are equivalent for argmax selection.
fn accept_row_fast_path_ok(
    a: &ActiveSeq,
    ctx: &ThinkSpecCtx<'_>,
    logits_format: VerifyLogitsFormat,
) -> bool {
    accept_fast_argmax_enabled()
        && !logits_format.is_fp32()
        && fast_path_seq_eligible(a)
        && !ctx.adaptive_sampling
        && !a.inside_thinking
        && !a.suppress_tool_call
        && !a.require_tool_call
        && a.grammar_state.is_none()
}

/// Walk flat DFlash verify rows using the actual plain-serving sampler.
///
/// Each processed token is committed immediately so history penalties,
/// grammar state, adaptive state, and seeded sampling see exactly the same
/// prefix they would see during serial decode.  Acceptance stops at the first
/// draft mismatch; that target token is the bonus and has not yet been fed to
/// the model.  Phase-transition tokens also remain bonuses so the next step
/// feeds them under the new phase exactly once.
///
/// Fast path: when `accept_row_fast_path_ok` holds, each row's target is a
/// raw-BF16 masked argmax instead of the full `process_seq_logits` walk.
/// This skips the per-row ~1 MB `Vec<f32>` allocation, the BF16→FP32
/// expansion, and the penalty scan — measured at ~6 ms/cycle for K=13 on
/// the champion (13 rows × 248,320 vocab). The result is byte-identical by
/// construction: BF16→FP32 is a monotone injection, all interventions that
/// could re-order the argmax are gated out, and the masks become a skip
/// list with matching last-wins tie semantics.
pub(super) fn dflash_content_accept(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    logits_format: VerifyLogitsFormat,
    mut fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
) -> SpecAcceptOutcome {
    debug_assert!(!a.inside_thinking);
    let mut row_buf = Vec::new();
    let mut num_accepted = 0usize;
    let elem_bytes = logits_format.elem_bytes();
    let mut fast_engaged_logged = false;

    for i in 0..verified.len() {
        if !fetch_row(i, &mut row_buf) {
            tracing::error!("spec-policy verify: logits row {i} unavailable; finishing seq");
            a.finished = true;
            return SpecAcceptOutcome {
                num_accepted,
                bonus: None,
            };
        }
        if row_buf.is_empty() || !row_buf.len().is_multiple_of(elem_bytes) {
            tracing::error!(
                row = i,
                bytes = row_buf.len(),
                elem_bytes,
                "spec-policy verify: malformed logits row; finishing seq"
            );
            a.finished = true;
            return SpecAcceptOutcome {
                num_accepted,
                bonus: None,
            };
        }
        let vocab = row_buf.len() / elem_bytes;
        let (target, lp) = if accept_row_fast_path_ok(a, ctx, logits_format) {
            if !fast_engaged_logged {
                fast_engaged_logged = true;
                tracing::info!("spec-policy accept: raw-BF16 masked-argmax fast path engaged");
            }
            let mut skip: Vec<u32> = Vec::with_capacity(2);
            if a.think_ended {
                if let Some(t) = a.think_start_token {
                    skip.push(t);
                }
                if let Some(t) = a.think_end_token {
                    skip.push(t);
                }
            }
            let tok = argmax_bf16_skip_last_wins(&row_buf, &skip, vocab).unwrap_or(0);
            (tok, None)
        } else {
            process_seq_logits(
                a,
                &row_buf,
                0,
                vocab,
                elem_bytes,
                logits_format.is_fp32(),
                a.think_end_token,
                a.think_start_token,
                a.tool_call_start_token,
                a.tool_call_end_token,
                ctx.reflection_suppress_ids,
                ctx.adaptive_sampling,
            )
        };

        let was_inside_thinking = a.inside_thinking;
        let draft_match = i < drafts.len() && i + 1 < verified.len() && drafts[i] == target;
        emit_token(a, target, lp);
        let phase_boundary = was_inside_thinking != a.inside_thinking
            || a.think_start_token == Some(target)
            || a.think_end_token == Some(target);

        if phase_boundary || !draft_match {
            if !a.finished {
                a.last_token = target;
            }
            return SpecAcceptOutcome {
                num_accepted,
                bonus: Some(target),
            };
        }
        if a.finished {
            return SpecAcceptOutcome {
                num_accepted: num_accepted + 1,
                bonus: None,
            };
        }
        num_accepted += 1;
    }

    debug_assert!(verified.is_empty());
    a.finished = true;
    SpecAcceptOutcome {
        num_accepted,
        bonus: None,
    }
}

/// Bind the content-policy walk to the resident BF16 or FP32 verify-logits rows.
pub(super) fn run_dflash_content_accept(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
) -> Option<SpecAcceptOutcome> {
    if verified.is_empty() {
        tracing::error!("spec-policy verify: empty target result; finishing seq");
        a.finished = true;
        return Some(SpecAcceptOutcome {
            num_accepted: 0,
            bonus: None,
        });
    }
    let logits_base = model.logits_buffer_ptr();
    let logits_format = if model.logits_ptr_is_fp32(logits_base) {
        VerifyLogitsFormat::Fp32
    } else {
        VerifyLogitsFormat::Bf16
    };
    let mut bulk: Option<Vec<u8>> = None;
    let mut bulk_rows: usize = 0;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            "DFlash sampler-policy filter active (penalties, masks, grammar, adaptive state)"
        );
    });
    let vocab = model.vocab_size();
    let elem_bytes = logits_format.elem_bytes();
    let rows_total = verified.len();
    let row_bytes = vocab * elem_bytes;
    Some(dflash_content_accept(
        a,
        drafts,
        verified,
        ctx,
        logits_format,
        // Bulk D2H: the verify-logits rows are contiguous in the resident
        // buffer (row i at `logits_base + i*row_bytes`), so ONE copy of
        // `rows_total * row_bytes` replaces N separate copy_logits_to_host
        // calls — the measured accept phase was dominated by the per-row
        // copy launches + per-row 0.5 MB resize, not by the walk itself.
        // Bytes delivered to the walk are identical, so this is bit-exact.
        |i, buf| {
            if bulk.is_none() {
                let mut all = vec![0u8; rows_total * row_bytes];
                match model.copy_logits_to_host(logits_base, &mut all) {
                    Ok(()) => {
                        bulk = Some(all);
                        bulk_rows = rows_total;
                    }
                    Err(e) => {
                        tracing::error!(
                            "spec-policy verify: bulk logits D2H failed ({} rows): {e:#}",
                            rows_total
                        );
                        return false;
                    }
                }
            }
            let Some(all) = bulk.as_ref() else {
                return false;
            };
            if i >= bulk_rows {
                return false;
            }
            let off = i * row_bytes;
            buf.clear();
            buf.extend_from_slice(&all[off..off + row_bytes]);
            true
        },
    ))
}

#[cfg(test)]
mod fast_argmax_tests {
    use super::argmax_bf16_skip_last_wins;

    fn bf16(f: f32) -> u16 {
        // Round-to-nearest-even BF16 conversion: keep the top 16 bits and
        // round based on the low 16 bits of the f32 mantissa.
        let bits = f.to_bits();
        let lsb = (bits >> 16) & 1;
        let round = ((bits & 0xFFFF) > 0x8000) || ((bits & 0xFFFF) == 0x8000 && lsb == 1);
        ((bits >> 16) + round as u32) as u16
    }

    /// Reference: exactly what `process_seq_logits`'s greedy branch does —
    /// expand BF16→FP32, apply −inf masks, `max_by` (last-wins ties).
    fn ref_walk(row: &[u8], skip: &[u32], vocab: usize) -> u32 {
        let f32s: Vec<f32> = (0..vocab)
            .map(|j| {
                let lo = row[2 * j];
                let hi = row[2 * j + 1];
                let u = u16::from_le_bytes([lo, hi]);
                let f = f32::from_bits((u as u32) << 16);
                if skip.contains(&(j as u32)) {
                    f32::NEG_INFINITY
                } else {
                    f
                }
            })
            .collect();
        f32s.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    #[test]
    fn last_wins_tie_semantics_match_max_by() {
        // Two distinct tokens with identical BF16 max values: `max_by`
        // returns the LAST; a strict `>` scan would return the first.
        let mut row = vec![0u8; 6 * 2];
        for (i, v) in [1.0f32, -2.0, 3.0, 3.0, 0.5, -1.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip_last_wins(&row, &[], 6), Some(3));
        assert_eq!(ref_walk(&row, &[], 6), 3);
    }

    #[test]
    fn negative_and_positive_values_ordered_correctly() {
        let mut row = vec![0u8; 6 * 2];
        // Largest magnitude positive must win; the most-negative value must
        // never be confused with a large unsigned pattern.
        for (i, v) in [0.0f32, -100.0, 5.5, -0.25, 1e3, -1e3].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip_last_wins(&row, &[], 6), Some(4));
        assert_eq!(ref_walk(&row, &[], 6), 4);
    }

    #[test]
    fn masked_tokens_never_win() {
        // Token 2 is the raw max but is masked (think token); the runner-up
        // must win, identically in both paths.
        let mut row = vec![0u8; 6 * 2];
        for (i, v) in [1.0f32, 2.0, 9.0, 3.0, 0.5, 4.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip_last_wins(&row, &[2], 6), Some(5));
        assert_eq!(ref_walk(&row, &[2], 6), 5);
    }

    #[test]
    fn masked_tie_breaks_toward_last_unmasked() {
        let mut row = vec![0u8; 6 * 2];
        for (i, v) in [7.0f32, 7.0, 7.0, 1.0, 2.0, 3.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        // Mask the first two 7.0s: last 7.0 (index 2) wins in both paths.
        assert_eq!(argmax_bf16_skip_last_wins(&row, &[0, 1], 6), Some(2));
        assert_eq!(ref_walk(&row, &[0, 1], 6), 2);
    }

    #[test]
    fn all_tokens_masked_falls_back_to_zero() {
        let mut row = vec![0u8; 3 * 2];
        for (i, v) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip_last_wins(&row, &[0, 1, 2], 3), None);
    }

    #[test]
    fn random_rows_match_reference() {
        // Deterministic pseudo-random sweep: many rows, various skip sets.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as f32 / u64::MAX as f32
        };
        for trial in 0..500 {
            let vocab = 32 + (trial % 200);
            let mut row = vec![0u8; vocab * 2];
            for j in 0..vocab {
                // Mix of signs and magnitudes; includes exact ties via the
                // quantized pool.
                let pool = [
                    (next() * 200.0 - 100.0) as f32,
                    ((next() * 20.0) as i32 as f32),
                ];
                let v = if trial % 3 == 0 { pool[0] } else { pool[1] };
                row[2 * j..2 * j + 2].copy_from_slice(&bf16(v).to_le_bytes());
            }
            let skip: Vec<u32> = (0..vocab)
                .filter(|_| next() < 0.1)
                .map(|j| j as u32)
                .collect();
            let fast = argmax_bf16_skip_last_wins(&row, &skip, vocab);
            let reference = ref_walk(&row, &skip, vocab);
            if skip.len() == vocab {
                assert_eq!(fast, None, "trial {trial}: all-skipped must be None");
            } else {
                assert_eq!(fast, Some(reference), "trial {trial}: fast path diverged");
            }
        }
    }
}
