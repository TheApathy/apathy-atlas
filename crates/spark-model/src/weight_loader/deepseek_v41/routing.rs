// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 MoE **routing**: scores -> mask -> top-k -> weights.
//!
//! Transcribed from the serving engine's `Model.moe`
//! (`dsv41-prefill-work/engine/model.py:559-575`) and then CHECKED against its taps, not
//! against my reading of it. [`tests::routing_replays_the_oracle_exactly`] replays
//! `route_scores` from the oracle capture through [`select_experts`] and requires
//! `route_idx` to match **exactly, including order** on layers 0, 2 and 14.
//!
//! ## Five details that are easy to get wrong, and were
//! 1. **The router GEMM is fp32.** The reference is `R.mm(y.float(), gate_w)`, not a bf16
//!    matmul. Routing decisions turn on ulp differences, so this is not a free choice.
//! 2. **Weights come from `scores`, NOT from `logits`.** The bias affects SELECTION only.
//!    Gathering the weights from the biased logits instead is a plausible misreading that
//!    produces finite, correctly-shaped, wrong weights — it is the negative control in
//!    [`tests::weights_must_come_from_scores_not_logits`], measured 0.41 away.
//! 3. **`route_scale` is `routed_scaling_factor` = 1.5**, applied AFTER normalisation.
//!    Recovered from the capture to 7 significant figures on three layers before it was
//!    found in the config; the config spells it `routed_scaling_factor`, the engine calls
//!    it `route_scale`, and nothing named `route_scale` exists in `config.json`.
//! 4. **`+ 1e-20` in the denominator**, not a bare divide.
//! 5. **The mask is applied to the logits as `-inf` BEFORE top-k**, per layer. The oracle
//!    shows exactly 124 finite entries per row out of 384 — that is `packed_keep = 124`
//!    pruning, and it is correct, not corruption. A port that skips it selects experts
//!    that are not resident.
//!
//! ## Why this is host-side and pure
//! Routing is where a wrong answer is least visible: it produces a *different, valid*
//! expert and the model keeps talking. Making the rule a pure function over slices means
//! it can be replayed against the working engine with no GPU, no weights and no attention
//! — which is how all five details above were settled.

use anyhow::{Result, ensure};

/// One token's routing decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    /// Selected expert ids in the 384-space router id space, `[num_tokens, k]`, in the
    /// order top-k returned them.
    pub indices: Vec<i64>,
    /// Their weights, `[num_tokens, k]`, normalised then scaled.
    pub weights: Vec<f32>,
    pub k: usize,
}

/// `sqrt(softplus(x))`, the V4.1 score function.
///
/// `softplus(x) = ln(1 + e^x)`, computed in the numerically stable form so a large `x`
/// does not overflow `exp`. The reference runs this in fp32 on an fp32 matmul result.
#[inline]
pub fn score_of(logit: f32) -> f32 {
    let softplus = if logit > 20.0 {
        // ln(1+e^x) -> x within fp32 resolution well before this point.
        logit
    } else {
        logit.exp().ln_1p()
    };
    softplus.sqrt()
}

/// Select experts and compute their weights.
///
/// * `scores` — `[num_tokens, num_experts]`, already `sqrt(softplus(y @ gate_w))`.
/// * `bias` — `[num_experts]`, the noaux_tc correction bias (`gate.bias`, or `gate.bias_vl`
///   on the multimodal path — the caller chooses, because that is a forward-path decision).
/// * `resident` — `[num_experts]` allow-list from `Cb3ExpertArena::routing_mask`. Applied
///   as `-inf` on the logits before top-k.
/// * `route_scale` — `config.routed_scaling_factor`, 1.5 for this checkpoint.
pub fn select_experts(
    scores: &[f32],
    bias: &[f32],
    resident: &[bool],
    num_tokens: usize,
    num_experts: usize,
    k: usize,
    route_scale: f32,
) -> Result<Routing> {
    ensure!(
        scores.len() == num_tokens * num_experts,
        "scores is {} long, expected {num_tokens} x {num_experts}",
        scores.len()
    );
    ensure!(bias.len() == num_experts, "bias must be {num_experts} wide");
    ensure!(
        resident.len() == num_experts,
        "the residency mask must be {num_experts} wide"
    );
    let resident_count = resident.iter().filter(|keep| **keep).count();
    ensure!(
        resident_count >= k,
        "only {resident_count} experts are resident but top-{k} is requested — the mask and \
         packed_keep disagree, and selecting from fewer than k would silently repeat experts"
    );

    let mut indices = Vec::with_capacity(num_tokens * k);
    let mut weights = Vec::with_capacity(num_tokens * k);
    // Reused across tokens so routing does not allocate per token.
    let mut order: Vec<u32> = Vec::with_capacity(num_experts);

    for token in 0..num_tokens {
        let row = &scores[token * num_experts..(token + 1) * num_experts];

        order.clear();
        order.extend((0..num_experts as u32).filter(|expert| resident[*expert as usize]));
        // Descending by logit. STABLE, and the tie-break is ascending expert id, which is
        // what `torch.topk` yields for this input — checked by the oracle replay, which
        // requires exact ORDER agreement, not just set agreement.
        order.sort_by(|a, b| {
            let la = row[*a as usize] + bias[*a as usize];
            let lb = row[*b as usize] + bias[*b as usize];
            lb.partial_cmp(&la)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });

        // Weights come from SCORES, not from the biased logits. See the module note.
        let mut sum = 0.0f32;
        for expert in &order[..k] {
            sum += row[*expert as usize];
        }
        let denominator = sum + 1e-20;
        for expert in &order[..k] {
            indices.push(*expert as i64);
            weights.push(row[*expert as usize] / denominator * route_scale);
        }
    }

    Ok(Routing { indices, weights, k })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    const REF_ROOT: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref";
    const NUM_EXPERTS: usize = 384;
    const K: usize = 6;
    const ROUTE_SCALE: f32 = 1.5;

    /// Every capture under `ref/`, sorted. Scanned rather than hardcoded so a new run
    /// (runB, runC, ...) is picked up the moment the oracle lane drops it — a test that
    /// named runA would have gone on passing while ignoring every later capture.
    fn ref_dirs() -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(REF_ROOT) else {
            return Vec::new();
        };
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                path.join("manifest.json").is_file().then_some(path)
            })
            .collect();
        dirs.sort();
        dirs
    }

    fn read_f32(dir: &Path, name: &str) -> Option<Vec<f32>> {
        let bytes = std::fs::read(dir.join(name)).ok()?;
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    fn read_i64(dir: &Path, name: &str) -> Option<Vec<i64>> {
        let bytes = std::fs::read(dir.join(name)).ok()?;
        Some(
            bytes
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                })
                .collect(),
        )
    }

    /// Recover `(bias, resident_mask)` from the capture's own `logits - scores`.
    ///
    /// Derived from the taps rather than loaded from the checkpoint ON PURPOSE: this test
    /// is about the routing RULE, and pulling the real bias tensor in would couple it to
    /// weight loading. `-inf` logits mark the pruned experts.
    fn bias_and_mask(scores: &[f32], logits: &[f32]) -> (Vec<f32>, Vec<bool>) {
        let mut bias = vec![0.0f32; NUM_EXPERTS];
        let mut mask = vec![false; NUM_EXPERTS];
        let tokens = scores.len() / NUM_EXPERTS;
        for expert in 0..NUM_EXPERTS {
            for token in 0..tokens {
                let logit = logits[token * NUM_EXPERTS + expert];
                if logit.is_finite() {
                    bias[expert] = logit - scores[token * NUM_EXPERTS + expert];
                    mask[expert] = true;
                    break;
                }
            }
        }
        (bias, mask)
    }

    /// THE TEST THAT MATTERS: replay the working engine's own router scores and require
    /// the selection to match EXACTLY.
    ///
    /// `route_idx` is an integer tap, so this is a mismatch COUNT, not a float tolerance —
    /// which is the right comparison for expert selection. Order is compared too, not just
    /// the set.
    #[test]
    fn routing_replays_the_oracle_exactly() {
        let dirs = ref_dirs();
        if dirs.is_empty() {
            eprintln!("skipping: no captures under {REF_ROOT}");
            return;
        }
        let mut layers_checked = 0;
        let mut rows_checked = 0usize;
        for dir in &dirs {
            for layer in ["L00", "L01", "L02", "L14"] {
                let (Some(scores), Some(logits), Some(want_idx), Some(want_w)) = (
                    read_f32(dir, &format!("{layer}.route_scores.000.bin")),
                    read_f32(dir, &format!("{layer}.route_logits.000.bin")),
                    read_i64(dir, &format!("{layer}.route_idx.000.bin")),
                    read_f32(dir, &format!("{layer}.route_w.000.bin")),
                ) else {
                    continue;
                };
                let tokens = scores.len() / NUM_EXPERTS;
                let tag = format!("{}/{layer}", dir.file_name().unwrap().to_string_lossy());

                let (bias, mask) = bias_and_mask(&scores, &logits);
                let resident = mask.iter().filter(|keep| **keep).count();
                assert_eq!(
                    resident, 124,
                    "{tag}: expected packed_keep=124 pruning, got {resident}"
                );

                let got = select_experts(
                    &scores, &bias, &mask, tokens, NUM_EXPERTS, K, ROUTE_SCALE,
                )
                .expect("routing must succeed");

                let mismatches = got
                    .indices
                    .iter()
                    .zip(&want_idx)
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(
                    mismatches, 0,
                    "{tag}: {mismatches} of {} selected experts differ from the engine",
                    want_idx.len()
                );

                let worst = got
                    .weights
                    .iter()
                    .zip(&want_w)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(worst < 1e-5, "{tag}: worst route_w error {worst:.3e} exceeds 1e-5");
                layers_checked += 1;
                rows_checked += tokens;
            }
        }
        assert!(layers_checked > 0, "the replay must have run on at least one layer");
        eprintln!(
            "routing replay: {layers_checked} layer-captures, {rows_checked} token-rows, \
             across {} run(s)",
            dirs.len()
        );
    }

    /// NEGATIVE CONTROL 1: taking the weights from the LOGITS instead of the scores must
    /// FAIL. This is the plausible misreading, and it is finite and correctly shaped.
    ///
    /// Measured ~0.41 away, so the 1e-5 tolerance above genuinely separates them.
    #[test]
    fn weights_must_come_from_scores_not_logits() {
        let Some(dir) = ref_dirs().into_iter().next() else {
            eprintln!("skipping: no captures under {REF_ROOT}");
            return;
        };
        let (Some(scores), Some(logits), Some(want_idx), Some(want_w)) = (
            read_f32(&dir, "L00.route_scores.000.bin"),
            read_f32(&dir, "L00.route_logits.000.bin"),
            read_i64(&dir, "L00.route_idx.000.bin"),
            read_f32(&dir, "L00.route_w.000.bin"),
        ) else {
            eprintln!("skipping: taps not present");
            return;
        };
        let tokens = scores.len() / NUM_EXPERTS;
        // Gather from logits, normalise the same way.
        let mut wrong = Vec::with_capacity(want_w.len());
        for token in 0..tokens {
            let picks = &want_idx[token * K..(token + 1) * K];
            let sum: f32 = picks
                .iter()
                .map(|e| logits[token * NUM_EXPERTS + *e as usize])
                .sum();
            for e in picks {
                wrong.push(logits[token * NUM_EXPERTS + *e as usize] / (sum + 1e-20) * ROUTE_SCALE);
            }
        }
        let worst = wrong
            .iter()
            .zip(&want_w)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst > 0.1,
            "the wrong rule landed within {worst:.3e} of the reference — the weight check \
             in `routing_replays_the_oracle_exactly` cannot distinguish them and is not \
             evidence"
        );
    }

    /// **The oracle replay is BLIND to the residency mask.** Measured, not assumed, and
    /// re-measured over EVERY capture rather than the one this was first written against.
    ///
    /// Across runA (prose) and runB (Python source) — two unrelated prompts, layers 0/1/2/14,
    /// 296 token-rows — the top-6 by UNMASKED logits are ALWAYS already resident, so masked
    /// and unmasked selection are identical. A port that skipped the mask entirely would
    /// pass `routing_replays_the_oracle_exactly` on both.
    ///
    /// That is expected rather than alarming — `packed_keep = 124` is a RANKED PREFIX of
    /// the most-used experts, so the top-6 for ordinary tokens nearly always fall inside
    /// it. But it means the replay is a gate that cannot fail on this particular property,
    /// and recording that is worth more than a control that quietly passes. The mask is
    /// therefore tested separately, on a constructed input where it MUST bite.
    ///
    /// If this ever starts failing, the capture has gained a token whose routing the mask
    /// actually changes — that is good news, and the replay becomes a real test of it.
    #[test]
    fn the_oracle_captures_do_not_exercise_the_residency_mask() {
        let dirs = ref_dirs();
        if dirs.is_empty() {
            eprintln!("skipping: no captures under {REF_ROOT}");
            return;
        }
        let mut biting_rows = 0usize;
        let mut total_rows = 0usize;
        for dir in &dirs {
            for layer in ["L00", "L01", "L02", "L14"] {
                let (Some(scores), Some(logits)) = (
                    read_f32(dir, &format!("{layer}.route_scores.000.bin")),
                    read_f32(dir, &format!("{layer}.route_logits.000.bin")),
                ) else {
                    continue;
                };
                let tokens = scores.len() / NUM_EXPERTS;
                let (bias, mask) = bias_and_mask(&scores, &logits);
                let all_resident = vec![true; NUM_EXPERTS];
                let unmasked = select_experts(
                    &scores, &bias, &all_resident, tokens, NUM_EXPERTS, K, ROUTE_SCALE,
                )
                .unwrap();
                for token in 0..tokens {
                    let picks = &unmasked.indices[token * K..(token + 1) * K];
                    if picks.iter().any(|e| !mask[*e as usize]) {
                        biting_rows += 1;
                    }
                    total_rows += 1;
                }
            }
        }
        eprintln!(
            "residency mask bites on {biting_rows}/{total_rows} token-rows across {} capture(s)",
            dirs.len()
        );
        assert_eq!(
            biting_rows, 0,
            "the captures NOW exercise the mask ({biting_rows}/{total_rows} rows). Good — \
             delete this test; `routing_replays_the_oracle_exactly` now covers it for real."
        );
    }

    /// The mask on an input where it MUST bite — the real test of it.
    ///
    /// Scores are rigged so the top-6 unmasked are all NON-resident. A port that skips the
    /// mask then selects six experts it does not have resident, and `slot_of` would have
    /// to invent slots for them. Constructed rather than captured because, per the test
    /// above, the capture does not contain such a token.
    #[test]
    fn the_mask_excludes_non_resident_experts_when_it_bites() {
        let mut scores = vec![0.1f32; NUM_EXPERTS];
        let bias = vec![0.0f32; NUM_EXPERTS];
        let mut mask = vec![false; NUM_EXPERTS];
        // Experts 0..6 score highest but are NOT resident; 100..106 are resident.
        for expert in 0..K {
            scores[expert] = 10.0;
        }
        for expert in 100..100 + K {
            mask[expert] = true;
            scores[expert] = 1.0;
        }
        let routed =
            select_experts(&scores, &bias, &mask, 1, NUM_EXPERTS, K, ROUTE_SCALE).unwrap();

        for expert in &routed.indices {
            assert!(
                mask[*expert as usize],
                "expert {expert} is NOT resident but was selected — the mask was not applied"
            );
        }
        let mut got = routed.indices.clone();
        got.sort_unstable();
        assert_eq!(got, (100..100 + K as i64).collect::<Vec<_>>());

        // NEGATIVE CONTROL for this control: without the mask, the same scores select the
        // non-resident 0..6. So the assertion above is discriminating, not vacuous.
        let all_resident = vec![true; NUM_EXPERTS];
        let unmasked =
            select_experts(&scores, &bias, &all_resident, 1, NUM_EXPERTS, K, ROUTE_SCALE)
                .unwrap();
        let mut unmasked_ids = unmasked.indices.clone();
        unmasked_ids.sort_unstable();
        assert_eq!(unmasked_ids, (0..K as i64).collect::<Vec<_>>());

        // Weights must be renormalised over the RESIDENT six and scaled.
        let sum: f32 = routed.weights.iter().sum();
        assert!(
            (sum - ROUTE_SCALE).abs() < 1e-5,
            "weights must sum to route_scale, got {sum}"
        );
    }

    /// Fewer resident experts than `k` must be an error, not a silent repeat.
    #[test]
    fn a_mask_narrower_than_k_is_refused() {
        let scores = vec![0.5f32; NUM_EXPERTS];
        let bias = vec![0.0f32; NUM_EXPERTS];
        let mut mask = vec![false; NUM_EXPERTS];
        for slot in mask.iter_mut().take(K - 1) {
            *slot = true;
        }
        let err = select_experts(&scores, &bias, &mask, 1, NUM_EXPERTS, K, ROUTE_SCALE)
            .expect_err("5 resident experts cannot satisfy top-6");
        assert!(err.to_string().contains("resident"), "{err}");
        // And exactly k IS allowed.
        mask[K - 1] = true;
        assert!(select_experts(&scores, &bias, &mask, 1, NUM_EXPERTS, K, ROUTE_SCALE).is_ok());
    }

    /// `sqrt(softplus(x))` must be the score function, and must not overflow.
    #[test]
    fn score_is_sqrt_softplus_and_is_stable_at_large_inputs() {
        // softplus(0) = ln 2
        assert!((score_of(0.0) - 2.0f32.ln().sqrt()).abs() < 1e-6);
        // Large input degenerates to sqrt(x), finite rather than NaN/inf.
        let big = score_of(100.0);
        assert!(big.is_finite(), "score_of(100) must not overflow, got {big}");
        assert!((big - 100.0f32.sqrt()).abs() < 1e-3);
        // NEGATIVE CONTROL: the naive form DOES overflow there, which is why the branch
        // exists. If this ever stops being true the branch is dead weight.
        //
        // The threshold matters and an earlier draft of this test got it wrong: f32 `exp`
        // overflows near x = 88.7, not at 80 — `80f32.exp()` is 5.54e34, comfortably
        // finite. So 80 would have made this control unable to fail.
        assert!(80.0f32.exp().is_finite(), "80 is BELOW the f32 exp overflow point");
        assert!(!(100.0f32.exp().ln_1p()).is_finite());
    }
}
