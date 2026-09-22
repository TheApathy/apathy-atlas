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

/// Token ids whose router row uses `gate.bias_vl` instead of `gate.bias`.
///
/// From the engine's `vision.image_sentinel_mask`: the image sentinel AND the image pad.
pub const IMAGE_SENTINEL_ID: i64 = 129_264;
pub const IMAGE_PAD_ID: i64 = 129_265;

/// Which rows are image positions, per `vision.image_sentinel_mask`.
pub fn image_rows(token_ids: &[i64]) -> Vec<bool> {
    token_ids
        .iter()
        .map(|id| *id == IMAGE_SENTINEL_ID || *id == IMAGE_PAD_ID)
        .collect()
}

/// Proof that a batch has NO image rows, so the text-only [`select_experts`] is correct for it.
///
/// Image rows route with `gate.bias_vl`. A text-only router applied to a multimodal batch
/// selects a different, valid expert set and raises nothing — measured on runE_image: 97
/// picks wrong across 4 layers from 7 image rows. So the text-only entry point cannot be
/// called without first checking the token ids; a multimodal caller gets an error pointing
/// at [`select_experts_multimodal`] instead of silently wrong routing.
#[derive(Clone, Copy, Debug)]
pub struct TextOnly {
    rows: usize,
}

impl TextOnly {
    pub fn verify(token_ids: &[i64]) -> Result<Self> {
        let image = image_rows(token_ids).iter().filter(|row| **row).count();
        ensure!(
            image == 0,
            "{image} of {} rows are image positions; they route with gate.bias_vl — use \
             select_experts_multimodal",
            token_ids.len()
        );
        Ok(Self { rows: token_ids.len() })
    }
}

/// Select experts and compute their weights, for a batch PROVEN text-only.
///
/// * `scores` — `[num_tokens, num_experts]`, already `sqrt(softplus(y @ gate_w))`.
/// * `bias` — `[num_experts]`, `gate.bias`.
/// * `resident` — `[num_experts]` allow-list from `Cb3ExpertArena::routing_mask`. Applied
///   as `-inf` on the logits before top-k.
/// * `route_scale` — `config.routed_scaling_factor`, 1.5 for this checkpoint.
#[allow(clippy::too_many_arguments)]
pub fn select_experts(
    scores: &[f32],
    bias: &[f32],
    text: TextOnly,
    resident: &[bool],
    num_tokens: usize,
    num_experts: usize,
    k: usize,
    route_scale: f32,
) -> Result<Routing> {
    ensure!(
        text.rows == num_tokens,
        "TextOnly was verified for {} rows, this batch has {num_tokens}",
        text.rows
    );
    let no_image = vec![false; num_tokens];
    select_experts_multimodal(
        scores, bias, bias, &no_image, resident, num_tokens, num_experts, k, route_scale,
    )
}

/// The general router: `gate.bias_vl` on `image_rows`, `gate.bias` elsewhere, per row —
/// the engine's `vision.router_bias`. Both biases and the row mask are always required.
#[allow(clippy::too_many_arguments)]
pub fn select_experts_multimodal(
    scores: &[f32],
    bias: &[f32],
    bias_vl: &[f32],
    image_rows: &[bool],
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
    ensure!(bias_vl.len() == num_experts, "bias_vl must be {num_experts} wide");
    ensure!(
        image_rows.len() == num_tokens,
        "image_rows is {} long for {num_tokens} tokens",
        image_rows.len()
    );
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
        let bias = if image_rows[token] { bias_vl } else { bias };

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

    /// A one-row text-only proof for synthetic single-token tests.
    fn text1() -> TextOnly {
        TextOnly::verify(&[0]).unwrap()
    }
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

    /// The TRUE `layers.N.ffn.gate.bias`, read from the checkpoint.
    ///
    /// Needed because the capture CANNOT supply it for pruned experts: their logits are all
    /// `-inf`, so `logits - scores` recovers nothing. Any question about what routing would
    /// do WITHOUT the mask has to read the real tensor.
    fn checkpoint_gate_bias(layer: usize) -> Option<Vec<f32>> {
        checkpoint_gate_f32(layer, "bias")
    }

    /// `layers.N.ffn.gate.<which>` as F32 straight from the shard (`bias` or `bias_vl`).
    fn checkpoint_gate_f32(layer: usize, which: &str) -> Option<Vec<f32>> {
        const MODEL: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(
                format!("{MODEL}/model.safetensors.index.json"),
            ).ok()?).ok()?;
        let key = format!("layers.{layer}.ffn.gate.{which}");
        let shard = index["weight_map"][&key].as_str()?;
        let bytes = std::fs::read(format!("{MODEL}/{shard}")).ok()?;
        let header_len = u64::from_le_bytes(bytes[..8].try_into().ok()?) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + header_len]).ok()?;
        let entry = &header[&key];
        if entry["dtype"].as_str()? != "F32" {
            return None;
        }
        let begin = 8 + header_len + entry["data_offsets"][0].as_u64()? as usize;
        let end = 8 + header_len + entry["data_offsets"][1].as_u64()? as usize;
        Some(
            bytes[begin..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// Recover `(bias, resident_mask)` from the capture's own `logits - scores`.
    ///
    /// **Only valid for RESIDENT experts.** A pruned expert's logits are all `-inf`, so its
    /// bias is unrecoverable here and is left at 0.0 — which is NOT its real value (~9.85).
    /// Use this only where the mask is applied; for any unmasked question use
    /// [`checkpoint_gate_bias`]. Trusting the 0.0 is what produced a confidently wrong
    /// finding that stood for several hours.
    ///
    /// Derived from the taps rather than loaded from the checkpoint ON PURPOSE: this test
    /// is about the routing RULE, and pulling the real bias tensor in would couple it to
    /// weight loading. `-inf` logits mark the pruned experts.
    ///
    /// Only rows where `use_row` holds are read. Image rows carry `gate.bias_vl`, a
    /// DIFFERENT bias, so a text bias must be recovered from text rows only and vice versa —
    /// recovering from "the first finite row" silently mixed the two on runE_image.
    fn bias_and_mask(
        scores: &[f32],
        logits: &[f32],
        use_row: impl Fn(usize) -> bool,
    ) -> (Vec<f32>, Vec<bool>) {
        let mut bias = vec![0.0f32; NUM_EXPERTS];
        let mut mask = vec![false; NUM_EXPERTS];
        let tokens = scores.len() / NUM_EXPERTS;
        for expert in 0..NUM_EXPERTS {
            for token in (0..tokens).filter(|token| use_row(*token)) {
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

    /// Image rows of the capture's first `tokens` positions, from the manifest's token ids.
    ///
    /// Occurrence 000 of a chunked capture is the FIRST chunk, which starts at position 0,
    /// so its rows are `token_ids[..tokens]`.
    fn capture_image_rows(dir: &Path, tokens: usize) -> Vec<bool> {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("manifest.json")).expect("manifest readable"),
        )
        .expect("manifest parses");
        let ids: Vec<i64> = manifest["token_ids"]
            .as_array()
            .expect("manifest has token_ids")
            .iter()
            .map(|id| id.as_i64().expect("integer token id"))
            .collect();
        assert!(ids.len() >= tokens, "manifest has {} ids for {tokens} rows", ids.len());
        image_rows(&ids[..tokens])
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
        let mut image_rows_checked = 0usize;
        // Picks that differ when image rows are routed with the TEXT bias. Must be > 0
        // wherever image rows were checked, or the vl-bias branch was never exercised.
        let mut text_bias_on_image_mismatches = 0usize;
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

                let image = capture_image_rows(dir, tokens);
                let (_, mask) = bias_and_mask(&scores, &logits, |t| !image[t]);
                let resident = mask.iter().filter(|keep| **keep).count();
                assert_eq!(
                    resident, 124,
                    "{tag}: expected packed_keep=124 pruning, got {resident}"
                );
                let image_count = image.iter().filter(|row| **row).count();
                // BOTH biases from the CHECKPOINT, as F32 — not recovered from the capture.
                // This is also the end-to-end check that the F32 bias reproduces the engine.
                let li: usize = layer[1..].parse().unwrap();
                let (Some(true_bias), Some(true_vl)) =
                    (checkpoint_gate_f32(li, "bias"), checkpoint_gate_f32(li, "bias_vl"))
                else {
                    panic!("{tag}: checkpoint gate biases unreadable");
                };
                if image_count > 0 {
                    let (_, vl_mask) = bias_and_mask(&scores, &logits, |t| image[t]);
                    assert_eq!(vl_mask, mask, "{tag}: image rows see a different residency mask");
                }

                let got = select_experts_multimodal(
                    &scores, &true_bias, &true_vl, &image, &mask, tokens, NUM_EXPERTS, K,
                    ROUTE_SCALE,
                )
                .expect("routing must succeed");

                if image_count > 0 {
                    // NEGATIVE CONTROL: the text bias on image rows must NOT reproduce them.
                    let text_on_images = select_experts_multimodal(
                        &scores, &true_bias, &true_bias, &image, &mask, tokens, NUM_EXPERTS, K,
                        ROUTE_SCALE,
                    )
                    .unwrap();
                    let wrong = text_on_images
                        .indices
                        .iter()
                        .zip(&want_idx)
                        .filter(|(a, b)| a != b)
                        .count();
                    eprintln!("  {tag}: text bias on {image_count} image rows -> {wrong} picks wrong");
                    text_bias_on_image_mismatches += wrong;
                    image_rows_checked += image_count;
                }

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
        if image_rows_checked > 0 {
            assert!(
                text_bias_on_image_mismatches > 0,
                "routing {image_rows_checked} image rows with the TEXT bias changed nothing, so \
                 the bias_vl branch is untested by these captures"
            );
        }
        eprintln!(
            "routing replay: {layers_checked} layer-captures, {rows_checked} token-rows \
             ({image_rows_checked} image rows; text bias on them: \
             {text_bias_on_image_mismatches} picks wrong), across {} run(s)",
            dirs.len()
        );
    }

    /// The text-only entry point must REFUSE a batch with image rows, rather than route them
    /// with the wrong bias. Control: the same call on text ids succeeds.
    #[test]
    fn the_text_only_router_refuses_image_rows() {
        let err = TextOnly::verify(&[11, IMAGE_SENTINEL_ID, 12]).expect_err("sentinel row");
        assert!(err.to_string().contains("select_experts_multimodal"), "{err}");
        assert!(TextOnly::verify(&[11, IMAGE_PAD_ID]).is_err(), "the pad id is an image row too");
        let proof = TextOnly::verify(&[11, 12]).expect("text ids are text-only");

        // A proof for a different row count cannot be spent on this batch.
        let scores = vec![0.5f32; NUM_EXPERTS];
        let bias = vec![0.0f32; NUM_EXPERTS];
        let mask = vec![true; NUM_EXPERTS];
        assert!(select_experts(&scores, &bias, proof, &mask, 1, NUM_EXPERTS, K, ROUTE_SCALE).is_err());
        assert!(select_experts(&scores, &bias, text1(), &mask, 1, NUM_EXPERTS, K, ROUTE_SCALE).is_ok());
    }

    /// The router bias must stay F32. `dense_auto` narrows F32 to bf16, and at ~9.8 bf16's
    /// step is 0.0625 — coarser than the 0.036 spread across experts. The replay above uses
    /// the F32 checkpoint bias and is exact; THIS is the control that shows the narrowing is
    /// not harmless: the bf16-rounded bias must mis-route real captured rows.
    #[test]
    fn a_bf16_router_bias_misroutes() {
        let dirs = ref_dirs();
        if dirs.is_empty() {
            eprintln!("skipping: no captures under {REF_ROOT}");
            return;
        }
        let to_bf16 = |v: f32| {
            let bits = v.to_bits();
            f32::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
        };
        let (mut wrong, mut picks) = (0usize, 0usize);
        for dir in &dirs {
            for (layer, li) in [("L00", 0usize), ("L02", 2)] {
                let (Some(scores), Some(logits), Some(want_idx)) = (
                    read_f32(dir, &format!("{layer}.route_scores.000.bin")),
                    read_f32(dir, &format!("{layer}.route_logits.000.bin")),
                    read_i64(dir, &format!("{layer}.route_idx.000.bin")),
                ) else {
                    continue;
                };
                let tokens = scores.len() / NUM_EXPERTS;
                let image = capture_image_rows(dir, tokens);
                let (_, mask) = bias_and_mask(&scores, &logits, |t| !image[t]);
                let bias = checkpoint_gate_f32(li, "bias").expect("bias");
                let bias_vl = checkpoint_gate_f32(li, "bias_vl").expect("bias_vl");
                let distinct = {
                    let mut v: Vec<u32> = bias.iter().map(|b| to_bf16(*b).to_bits()).collect();
                    v.sort_unstable();
                    v.dedup();
                    v.len()
                };
                assert!(distinct < 10, "{layer}: bf16 should collapse 384 biases to a handful, got {distinct}");
                let narrow = |b: &[f32]| b.iter().map(|v| to_bf16(*v)).collect::<Vec<f32>>();
                let got = select_experts_multimodal(
                    &scores, &narrow(&bias), &narrow(&bias_vl), &image, &mask, tokens,
                    NUM_EXPERTS, K, ROUTE_SCALE,
                )
                .unwrap();
                wrong += got.indices.iter().zip(&want_idx).filter(|(a, b)| a != b).count();
                picks += want_idx.len();
            }
        }
        if picks == 0 {
            eprintln!("skipping: no L00/L02 route taps");
            return;
        }
        eprintln!("bf16-narrowed router bias: {wrong} of {picks} picks wrong");
        assert!(wrong > 0, "a bf16 bias reproduced every pick — the F32 requirement would be unfounded");
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

    /// The residency mask CHANGES SELECTION on most tokens — it is load-bearing.
    ///
    /// ## This test previously asserted the OPPOSITE, and was wrong
    /// It reported the mask never bites (0/296) and I relayed that to three people as
    /// evidence that `packed_keep = 124` was "effectively lossless on real text". It is
    /// not. The bug was in the test's own bias reconstruction, not in the router.
    ///
    /// [`bias_and_mask`] recovers the router bias from `logits - scores` on FINITE entries.
    /// For a pruned expert every logit is `-inf`, so its bias is **never observable** — 0 of
    /// 260 recovered — and the helper left it at 0.0 while resident experts recovered their
    /// true ~9.85. That manufactured a ~9.85 handicap for exactly the experts under test, so
    /// of course none of them ever reached the top-6.
    ///
    /// The real `layers.N.ffn.gate.bias` is near-CONSTANT across all 384 (min 9.59, max
    /// 9.88, std 0.036) while scores span 0.017..1.86. So the bias barely affects ranking at
    /// all, pruned experts compete on equal footing, and with the true bias the best
    /// non-resident expert has median rank 3 — frequently rank 0, the top expert overall.
    ///
    /// Measured with the true per-layer bias read from the checkpoint: the mask changes
    /// selection on **202 of 296 token-rows (68%)** across runA and runB.
    ///
    /// ## What follows from that
    /// - `packed_keep = 124` is NOT lossless. It substantially rewrites routing.
    /// - A port that skipped the mask would get `route_idx` wrong on ~68% of tokens, so the
    ///   oracle replay DOES test it. There was never a blind spot here.
    /// - The margin analysis is what caught it. A binary "does it bite" answered 0 and
    ///   looked clean; asking HOW CLOSE it came returned "rank exactly 124, every row,
    ///   every layer, both runs" — a suspiciously perfect number that could only come from
    ///   the reconstruction, not from the model.
    #[test]
    fn the_residency_mask_changes_selection_on_most_tokens() {
        let dirs = ref_dirs();
        if dirs.is_empty() {
            eprintln!("skipping: no captures under {REF_ROOT}");
            return;
        }
        let Some(true_bias) = checkpoint_gate_bias(0) else {
            eprintln!("skipping: checkpoint gate.bias not readable");
            return;
        };

        let mut biting = 0usize;
        let mut total = 0usize;
        for dir in &dirs {
            let (Some(scores), Some(logits)) = (
                read_f32(dir, "L00.route_scores.000.bin"),
                read_f32(dir, "L00.route_logits.000.bin"),
            ) else {
                continue;
            };
            let tokens = scores.len() / NUM_EXPERTS;
            // TEXT rows only: image rows carry `gate.bias_vl`, which is not the tensor
            // under test here.
            let image = capture_image_rows(dir, tokens);
            let (_, mask) = bias_and_mask(&scores, &logits, |t| !image[t]);

            // The TRUE bias must reproduce the capture's logits on resident entries. Without
            // this the comparison below could be measuring a mis-read tensor.
            for token in (0..tokens).filter(|t| !image[*t]) {
                for expert in 0..NUM_EXPERTS {
                    if mask[expert] {
                        let mine = scores[token * NUM_EXPERTS + expert] + true_bias[expert];
                        let theirs = logits[token * NUM_EXPERTS + expert];
                        assert!(
                            (mine - theirs).abs() < 1e-3,
                            "true bias disagrees with the capture at expert {expert}: \
                             {mine} vs {theirs}"
                        );
                    }
                }
            }

            let all_resident = vec![true; NUM_EXPERTS];
            let unmasked = select_experts_multimodal(
                &scores, &true_bias, &true_bias, &image, &all_resident, tokens, NUM_EXPERTS, K,
                ROUTE_SCALE,
            )
            .unwrap();
            for token in (0..tokens).filter(|t| !image[*t]) {
                let picks = &unmasked.indices[token * K..(token + 1) * K];
                if picks.iter().any(|e| !mask[*e as usize]) {
                    biting += 1;
                }
                total += 1;
            }
        }
        if total == 0 {
            eprintln!("skipping: no L00 taps found");
            return;
        }
        let fraction = biting as f64 / total as f64;
        eprintln!("residency mask changes selection on {biting}/{total} token-rows");
        assert!(
            fraction > 0.4,
            "the mask changed selection on only {biting}/{total} rows. Measured 68% on \
             runA+runB; a collapse to near zero means the bias is being reconstructed \
             rather than read from the checkpoint — the exact bug this test was rewritten \
             to fix."
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
            select_experts(&scores, &bias, text1(), &mask, 1, NUM_EXPERTS, K, ROUTE_SCALE).unwrap();

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
            select_experts(&scores, &bias, text1(), &all_resident, 1, NUM_EXPERTS, K, ROUTE_SCALE)
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
        let err = select_experts(&scores, &bias, text1(), &mask, 1, NUM_EXPERTS, K, ROUTE_SCALE)
            .expect_err("5 resident experts cannot satisfy top-6");
        assert!(err.to_string().contains("resident"), "{err}");
        // And exactly k IS allowed.
        mask[K - 1] = true;
        assert!(select_experts(&scores, &bias, text1(), &mask, 1, NUM_EXPERTS, K, ROUTE_SCALE).is_ok());
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
