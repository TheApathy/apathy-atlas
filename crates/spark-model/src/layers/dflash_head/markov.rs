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
//! ## Device-resident static path
//!
//! The per-position bias `B` is a dense `[V]` vector (V ≈ 248 K), so an exact
//! argmax cannot be approximated by a top-k window. Each position now stays on
//! the producer stream: gather W1[prev], GEMV W2, then run
//! `argmax_add_bf16(base_row, bias)` over the complete logical target vocab.
//! The selected u32 is both this row's output and the next row's predecessor.
//! Only the final gamma u32 IDs cross to the host. The previous implementation
//! downloaded gamma base-logit rows plus one full-vocab bias per position and
//! synchronized inside every iteration.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::BlockDiffusionDraftHead;
use crate::layers::ops;

impl BlockDiffusionDraftHead {
    /// Enqueue the gamma-block Markov correction LEFT-TO-RIGHT on `stream`.
    ///
    /// Preconditions:
    ///  * `self.markov` is `Some` (caller gates on this).
    ///  * `self.scratch.logits` holds the base logit rows `[gamma_eff, vocab]`
    ///    BF16, row-major, exactly as `forward_block` left them (before any
    ///    subsequent kernel overwrites the buffer).
    ///
    /// Writes the corrected token IDs to `self.scratch.draft_tokens_dev`.
    /// There is deliberately no fallback: resolving/launching this path is a
    /// requirement for a checkpoint whose Markov head was loaded.
    pub(super) fn enqueue_markov_sequential(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        last_token: u32,
        gamma_eff: usize,
    ) -> Result<()> {
        let markov = self.markov.as_ref().ok_or_else(|| {
            anyhow::anyhow!("DSpark Markov device dispatch requested without loaded weights")
        })?;
        if gamma_eff == 0 {
            return Ok(());
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

        // Seed row 0 once. This four-byte H2D completes before the stream
        // kernels are enqueued; unlike the former implementation it is not
        // repeated inside the per-position loop.
        let seed_bytes = last_token.to_le_bytes();
        gpu.copy_h2d(&seed_bytes, self.scratch.markov_prev_dev)?;
        let mut prev_token_dev = self.scratch.markov_prev_dev;
        for k in 0..gamma_eff {
            // ── B(prev) = markov_w2 @ markov_w1[prev] ──
            // 1) gather markov_w1[prev] → [rank] BF16 via batched_embed (one
            //    token, hidden_size = rank). W1 is [vocab, rank] so row `prev`
            //    is a contiguous `[rank]` slice — a plain embedding lookup.
            ops::batched_embed(
                gpu,
                self.kernels.batched_embed,
                prev_token_dev,
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

            // 3) Exact argmax over U_k + B(prev) across the logical target
            // vocab. `scratch.logits` is compact [gamma_eff, lm_vocab] even
            // though its allocation has padded drafter-vocab capacity.
            let base_row = self
                .scratch
                .logits
                .offset(compact_logit_row_base(k, lm_vocab) * bf16);
            let token_out = self.scratch.draft_tokens_dev.offset(k * 4);
            ops::argmax_add_bf16(
                gpu,
                self.kernels.argmax_add,
                base_row,
                self.scratch.markov_bias,
                token_out,
                lm_vocab as u32,
                stream,
            )?;
            prev_token_dev = token_out;
        }
        Ok(())
    }
}

/// Compact LM-head row offset. The allocation may be padded to drafter vocab,
/// but emitted rows are packed at the logical target-vocabulary stride.
#[inline]
fn compact_logit_row_base(row: usize, lm_vocab: usize) -> usize {
    row * lm_vocab
}

#[cfg(test)]
mod tests {
    use super::compact_logit_row_base;

    fn argmax_add_ref(base: &[f32], bias: &[f32], logical_vocab: usize) -> usize {
        assert!(base.len() >= logical_vocab);
        assert!(bias.len() >= logical_vocab);
        let mut best_token = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for token in 0..logical_vocab {
            let value = base[token] + bias[token];
            if value > best_value {
                best_value = value;
                best_token = token;
            }
        }
        best_token
    }

    #[test]
    fn shared_lm_head_rows_use_target_vocab_stride() {
        // Qwen3.8's DSpark config pads its drafter vocab to 248320, while the
        // shared target LM head emits only 248077 columns.  Row 1 must begin
        // immediately after those emitted columns, not after the allocation's
        // padded capacity.
        assert_eq!(compact_logit_row_base(1, 248_077), 248_077);
        assert_ne!(compact_logit_row_base(1, 248_077), 248_320);
        assert_eq!(compact_logit_row_base(6, 248_077), 1_488_462);
    }

    #[test]
    fn device_argmax_contract_keeps_host_tie_break_and_logical_vocab_crop() {
        // Tokens 1 and 2 tie after addition. The former left-to-right host
        // scan kept token 1, so the device reduction must use token ID as its
        // explicit secondary key. The padded drafter-only row is larger but
        // outside the target LM head's logical vocabulary.
        let base = [0.0, 2.0, 1.0, 100.0];
        let bias = [0.0, 0.0, 1.0, 100.0];
        assert_eq!(argmax_add_ref(&base, &bias, 3), 1);
        assert_eq!(argmax_add_ref(&base, &bias, 4), 3);
    }

    #[test]
    fn markov_module_has_no_full_vocab_host_readback_or_loop_sync() {
        let source = include_str!("markov.rs");
        let d2h_call = ["copy_", "d2h("].concat();
        let sync_call = ["synchro", "nize("].concat();
        assert!(!source.contains(&d2h_call));
        assert!(!source.contains(&sync_call));
        assert!(source.contains("ops::argmax_add_bf16"));
    }

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
