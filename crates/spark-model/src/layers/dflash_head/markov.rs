// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark VanillaMarkov head — sequential low-rank bigram logit-bias repair.
//!
//! Vendored math from `models/DSpark-AEON-draft/markov_head.py`
//! (`VanillaMarkov.sample_block_tokens`, DeepSpec / DSpark). The parallel
//! block drafter predicts every block position in ONE forward from mask-token
//! inputs, so position `k` cannot see what was actually sampled at position
//! `k-1` — the "suffix decay" the DSpark authors measured. The Markov head
//! fixes that cheaply by adding a per-position logit bias conditioned on the
//! previous token only:
//!
//! ```text
//!   B(x_{k-1}, :) = markov_w2 @ markov_w1[x_{k-1}]     W1: [V, r], W2: [V, r]
//!   corrected_k   = U_k + B(x_{k-1}, :)                U_k = base logit row k
//!   token_k       = argmax(corrected_k)                (greedy; temp=0 here)
//! ```
//!
//! The block is sampled LEFT-TO-RIGHT: position 0's predecessor is the
//! verified `last_token` (the bonus that seeds the block), and position `k`'s
//! predecessor is the token this loop just chose at `k-1`. This is CHAIN mode
//! (single-block verify) — the corrected tokens are still just *proposals*, so
//! the correction is LOSSLESS w.r.t. committed output: the target verify
//! commits its own greedy token regardless of what the drafter proposed.
//!
//! ## On-GPU vs host
//!
//! The per-position bias `B` is a dense `[V]` vector (V ≈ 248 K), so an exact
//! argmax cannot be windowed to a top-k. We compute `B` on-GPU (a `[V, r] @
//! [r]` GEMV, r = 256 — tiny), D2H just the `[V]` bias, add it to the base
//! logit row on host, and argmax on host. The base logits are D2H'd once for
//! the whole block (they don't change); only the per-position bias round-trips.
//! Cost per block: 1 × (γ·V) base-logit D2H + γ × (V bias GEMV + V D2H +
//! host argmax). At γ=11, V=248 K that is ~5.5 MB + 11 × ~0.5 MB ≈ 11 MB D2H
//! and 11 × 248 K host comparisons — negligible next to the drafter forward.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::BlockDiffusionDraftHead;
use crate::layers::ops;

impl BlockDiffusionDraftHead {
    /// Re-sample the γ-block drafts LEFT-TO-RIGHT with the DSpark Markov bias.
    ///
    /// Preconditions:
    ///  * `self.markov` is `Some` (caller gates on this).
    ///  * `self.scratch.logits` holds the base logit rows `[gamma_eff, vocab]`
    ///    BF16, row-major, exactly as `forward_block` left them (before any
    ///    subsequent kernel overwrites the buffer).
    ///
    /// `base_drafts` is the plain per-row argmax `forward_block` already
    /// computed (used only to size the output and as a fallback on the empty
    /// path). Returns the Markov-corrected drafts, same length.
    pub(super) fn apply_markov_sequential(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        last_token: u32,
        base_drafts: &[u32],
    ) -> Result<Vec<u32>> {
        let markov = match self.markov.as_ref() {
            Some(m) => m,
            None => return Ok(base_drafts.to_vec()),
        };
        let gamma_eff = base_drafts.len();
        if gamma_eff == 0 {
            return Ok(Vec::new());
        }
        // The shared lm_head only spans `target_vocab_size` rows even when the
        // drafter's own `vocab_size` is larger, and `forward_block` argmaxes
        // over exactly `min(target_vocab, vocab)`. Match that span here so the
        // Markov argmax ranges over the same valid tokens (the tail rows of the
        // base logits are stale/zero). The Markov weights are full-vocab, so we
        // read the same `lm_vocab`-length prefix of the bias.
        let lm_vocab = self.target_vocab_size.min(self.vocab_size);
        let bf16 = 2usize;
        let rank = markov.rank as u32;
        let full_vocab = self.vocab_size as u32;

        // D2H the base logit block once (rows don't change between positions).
        // Sync first so the forward_block kernels have landed.
        gpu.synchronize(stream)?;
        let mut base_bytes = vec![0u8; gamma_eff * self.vocab_size * bf16];
        gpu.copy_d2h(self.scratch.logits, &mut base_bytes)?;
        let base_logits: Vec<f32> = base_bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();

        let mut corrected: Vec<u32> = Vec::with_capacity(gamma_eff);
        let mut prev = last_token;
        let mut bias_bytes = vec![0u8; self.vocab_size * bf16];
        for k in 0..gamma_eff {
            // ── B(prev) = markov_w2 @ markov_w1[prev] ──
            // 1) gather markov_w1[prev] → [rank] BF16 via batched_embed (one
            //    token, hidden_size = rank). W1 is [vocab, rank] so row `prev`
            //    is a contiguous `[rank]` slice — a plain embedding lookup.
            let prev_le = prev.to_le_bytes();
            gpu.copy_h2d(&prev_le, self.scratch.markov_prev_dev)?;
            ops::batched_embed(
                gpu,
                self.kernels.batched_embed,
                self.scratch.markov_prev_dev,
                markov.w1.weight,
                self.scratch.markov_w1_row,
                1,
                rank,
                stream,
            )?;
            // 2) bias = dense_gemv(w1_row, w2) → [vocab] BF16. W2 weight is
            //    [vocab, rank] = nn.Linear(rank, vocab) → C[n=vocab] = A[k=rank]
            //    @ B[vocab, rank]^T. Exactly the dense_gemv contract.
            ops::dense_gemv(
                gpu,
                self.kernels.dense_gemv,
                self.scratch.markov_w1_row,
                &markov.w2,
                self.scratch.markov_bias,
                full_vocab,
                rank,
                stream,
            )?;
            gpu.synchronize(stream)?;
            gpu.copy_d2h(self.scratch.markov_bias, &mut bias_bytes)?;

            // 3) argmax over U_k + B(prev) on host, ranging over lm_vocab.
            let row_base = k * self.vocab_size;
            let mut best_tok = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for v in 0..lm_vocab {
                let b = bf16_to_f32(u16::from_le_bytes([
                    bias_bytes[v * 2],
                    bias_bytes[v * 2 + 1],
                ]));
                let corrected_logit = base_logits[row_base + v] + b;
                if corrected_logit > best_val {
                    best_val = corrected_logit;
                    best_tok = v as u32;
                }
            }
            corrected.push(best_tok);
            prev = best_tok;
        }
        Ok(corrected)
    }
}

/// BF16 (stored as u16 bit pattern) → f32. Truncation form: BF16 is the high
/// 16 bits of an IEEE-754 f32, so we just shift left by 16. Matches the
/// `forward_block` dump helper and the drafter's BF16 storage exactly.
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::bf16_to_f32;

    /// Host-side reference of `VanillaMarkov.sample_block_tokens` (greedy,
    /// temperature=0). Replicates `markov_head.py` exactly on small f32
    /// fixtures so the sequential bias math is pinned independent of any GPU:
    ///   B(prev) = w2 @ w1[prev];  token_k = argmax(base[k] + B(prev));
    ///   prev feeds position k+1.
    fn markov_sample_block_ref(
        base_logits: &[Vec<f32>], // [M][V]
        w1: &[Vec<f32>],          // [V][r]
        w2: &[Vec<f32>],          // [V][r]  (nn.Linear(r, V) weight, row = out)
        first_prev: usize,
    ) -> Vec<usize> {
        let vocab = w1.len();
        let rank = if vocab == 0 { 0 } else { w1[0].len() };
        let mut out = Vec::with_capacity(base_logits.len());
        let mut prev = first_prev;
        for row in base_logits {
            // w1[prev] : [r]
            let w1row = &w1[prev];
            // bias[v] = sum_j w2[v][j] * w1row[j]
            let mut best_tok = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for v in 0..vocab {
                let mut bias = 0.0f32;
                for j in 0..rank {
                    bias += w2[v][j] * w1row[j];
                }
                let corrected = row[v] + bias;
                if corrected > best_val {
                    best_val = corrected;
                    best_tok = v;
                }
            }
            out.push(best_tok);
            prev = best_tok;
        }
        out
    }

    #[test]
    fn bf16_roundtrip_high_bits() {
        // 1.0 = 0x3F800000 → bf16 0x3F80. 2.0 → 0x4000. -1.0 → 0xBF80.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x4000), 2.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn bias_flips_argmax_left_to_right() {
        // V=3, r=2, M=2. Base logits alone would pick token 0 at both
        // positions. A bias that strongly rewards "the token != prev" makes
        // the sequential head chain 1 -> 2 (or similar), proving the bias is
        // (a) applied and (b) fed left-to-right.
        let base = vec![
            vec![0.10, 0.05, 0.00], // pos 0: argmax w/o bias = tok 0
            vec![0.10, 0.05, 0.00], // pos 1: argmax w/o bias = tok 0
        ];
        // w1[prev] selects a one-hot-ish rank vector per predecessor token.
        // r=2. Let w1[t] = [t==0? 1:0, t==0? 0:1-ish] — we just need distinct
        // rows so the bias depends on prev.
        let w1 = vec![
            vec![1.0, 0.0], // prev tok 0
            vec![0.0, 1.0], // prev tok 1
            vec![1.0, 1.0], // prev tok 2
        ];
        // w2[v] : make token 2 heavily favored when prev==0 (uses component 0),
        // and token 1 favored when prev==1 (uses component 1).
        let w2 = vec![
            vec![0.0, 0.0], // token 0 bias = 0 always
            vec![0.0, 5.0], // token 1 bias = 5 * w1row[1]
            vec![5.0, 0.0], // token 2 bias = 5 * w1row[0]
        ];
        // first_prev = 0 (the seed / bonus token).
        //  pos0: prev=0 → w1row=[1,0]; bias=[0, 5*0, 5*1]=[0,0,5];
        //        corrected=[0.10,0.05,5.00] → argmax = tok 2. prev := 2.
        //  pos1: prev=2 → w1row=[1,1]; bias=[0, 5*1, 5*1]=[0,5,5];
        //        corrected=[0.10,5.05,5.00] → argmax = tok 1.
        let out = markov_sample_block_ref(&base, &w1, &w2, 0);
        assert_eq!(out, vec![2, 1]);

        // Sanity: WITHOUT the sequential feed (if pos1 also used prev=0), pos1
        // would pick tok 2 not tok 1 — confirms left-to-right chaining matters.
        let out_seed1 = markov_sample_block_ref(&base, &w1, &w2, 1);
        //  pos0: prev=1 → w1row=[0,1]; bias=[0,5,0]; corrected=[0.10,5.05,0.00]
        //        → tok 1. prev := 1. pos1 same → tok 1.
        assert_eq!(out_seed1, vec![1, 1]);
    }

    #[test]
    fn zero_bias_is_plain_argmax() {
        // With w2 all zeros the bias is 0 everywhere and the corrected argmax
        // must equal the base argmax at every position (LOSSLESS no-op path).
        let base = vec![vec![0.1, 0.9, 0.3], vec![0.5, 0.2, 0.8]];
        let w1 = vec![vec![1.0, 2.0]; 3];
        let w2 = vec![vec![0.0, 0.0]; 3];
        let out = markov_sample_block_ref(&base, &w1, &w2, 0);
        assert_eq!(out, vec![1, 2]); // plain per-row argmax
    }
}
