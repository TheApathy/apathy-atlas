// SPDX-License-Identifier: AGPL-3.0-only

//! The cross-layer schedule of DeepSeek-V4.1 sparse index attention.
//!
//! Pure index arithmetic, no device work. This is deliberately separated from
//! the kernel because every bug of the shape that has cost this team time —
//! GLM's `usable_pools` dropping three tokens at every `q % 4 == 3` — lives in
//! exactly these functions and is checkable on CPU with no GPU lock held.
//!
//! Derived from the Python reference engine; see
//! `kernels/gb10/deepseek-v4.1/attn/SPEC.md` sections 1, 4 and 8b for the
//! provenance of every constant and formula here.
//!
//! The thing to understand before reading further: "cross-layer" is THREE
//! independent inheritance chains, not one.
//!
//!  1. compressed KV + index keys — built on `kv_source_layers` {2,8,14,20}
//!  2. top-k selection           — built on `index_source_layers` {2,8,14,20,24,28,32,36}
//!  3. candidate block mask      — built on layer 20 alone, consumed by 24..36
//!
//! Every other layer *reuses* what the most recent source layer left behind.

/// Layer schedule for one checkpoint. Defaults are the V4.1-Flash-Next ones.
#[derive(Debug, Clone)]
pub struct SparseSchedule {
    pub num_layers: usize,
    /// Per-layer compression ratio. 0 = window-only (no compressed rows, no
    /// indexer); 1 = one compressed row per token; 2 = gated mean of pairs.
    pub compress_ratios: Vec<usize>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub candidate_source_layer: usize,
    pub index_topk: usize,
    pub window_size: usize,
    pub candidate_block_size: usize,
    pub candidate_topk_blocks: usize,
}

impl SparseSchedule {
    /// Build from layer sets supplied by the caller.
    ///
    /// THIS is the constructor production code must use, with the constants
    /// from `weight_loader::deepseek_v41::seams`
    /// (INDEX_SOURCE_LAYERS / KV_SOURCE_LAYERS / CANDIDATE_SOURCE_LAYER /
    /// INDEX_TOPK) or from engine's `IndexerAdmission`, which derives them from
    /// the weight store. Two drifting copies of {2,8,14,20,24,28,32,36} is
    /// exactly the mis-selection this module exists to prevent, so the layer
    /// sets are a PARAMETER here and are not declared in this file outside the
    /// test-only default below.
    pub fn new(
        num_layers: usize,
        compress_ratios: Vec<usize>,
        kv_source_layers: Vec<usize>,
        index_source_layers: Vec<usize>,
        candidate_source_layer: usize,
        index_topk: usize,
    ) -> Self {
        Self {
            num_layers,
            compress_ratios,
            kv_source_layers,
            index_source_layers,
            candidate_source_layer,
            index_topk,
            window_size: 128,
            candidate_block_size: 8,
            candidate_topk_blocks: 2048,
        }
    }

    /// PROVISIONAL / TEST-ONLY. The V4.1-Flash-Next layer sets, hardcoded so
    /// these tests can run before the attention and engine branches converge.
    /// Once `seams` is reachable from this crate path, this must be replaced by
    /// `SparseSchedule::new(..)` fed from it — the literals below are the
    /// second copy, and the only acceptable second copy is one a test pins.
    ///
    /// `compress_ratios = [0,0] + [2]*18 + [1]*20`, 40 layers.
    pub fn v41_flash_next() -> Self {
        let mut compress_ratios = vec![0usize; 2];
        compress_ratios.extend(std::iter::repeat(2).take(18));
        compress_ratios.extend(std::iter::repeat(1).take(20));
        Self {
            num_layers: 40,
            compress_ratios,
            kv_source_layers: vec![2, 8, 14, 20],
            index_source_layers: vec![2, 8, 14, 20, 24, 28, 32, 36],
            candidate_source_layer: 20,
            index_topk: 512,
            window_size: 128,
            candidate_block_size: 8,
            candidate_topk_blocks: 2048,
        }
    }

    pub fn ratio(&self, layer: usize) -> usize {
        self.compress_ratios[layer]
    }

    /// A ratio-0 layer attends to its sliding window only: no compressed rows,
    /// no indexer call, and — see [`uses_yarn_rope`] — the *other* RoPE table.
    pub fn is_window_only(&self, layer: usize) -> bool {
        self.ratio(layer) == 0
    }

    /// `freqs_c` (theta 160000, YaRN) for every layer with a nonzero ratio;
    /// `freqs_w` (theta 10000, no YaRN) for the rest. Reference:
    /// `freqs = self.freqs_c if w.ratio else self.freqs_w`.
    ///
    /// Getting this backwards yields a plausible-looking but wrong model on 38
    /// of 40 layers, which is why it is a named predicate and not an inline
    /// condition at the call site.
    pub fn uses_yarn_rope(&self, layer: usize) -> bool {
        !self.is_window_only(layer)
    }

    /// The layer whose compressed KV / index keys `layer` reads: the most
    /// recent kv-source layer at or below it. `None` for layers before the
    /// first source (0 and 1).
    pub fn kv_source_for(&self, layer: usize) -> Option<usize> {
        self.kv_source_layers
            .iter()
            .copied()
            .filter(|&s| s <= layer)
            .max()
    }

    /// The layer whose top-k selection `layer` attends with.
    pub fn index_source_for(&self, layer: usize) -> Option<usize> {
        self.index_source_layers
            .iter()
            .copied()
            .filter(|&s| s <= layer)
            .max()
    }

    pub fn runs_indexer(&self, layer: usize) -> bool {
        self.index_source_layers.contains(&layer)
    }

    /// Whether this layer's indexer masks its scores with the candidate table
    /// produced at `candidate_source_layer`. Reference condition:
    /// `(L != src) and 0 <= src < L and candidates is not None`.
    pub fn uses_candidate_mask(&self, layer: usize) -> bool {
        self.runs_indexer(layer) && layer > self.candidate_source_layer
    }

    /// Startup invariant from `fastdecode.py`: the candidate table is only wide
    /// enough for downstream indexers whose ratio is at least the source's. A
    /// narrower downstream ratio would silently slice a too-short mask.
    ///
    /// Empty for this checkpoint (ratio 1 at 20 and at every later indexer) —
    /// kept because a future config change is exactly how this goes wrong.
    pub fn bad_candidate_consumers(&self) -> Vec<usize> {
        let src_ratio = self.ratio(self.candidate_source_layer);
        self.index_source_layers
            .iter()
            .copied()
            .filter(|&l| l > self.candidate_source_layer && self.ratio(l) < src_ratio)
            .collect()
    }

    /// Absolute positions a query at `pos` may see in its sliding window, most
    /// recent last. `None` where the window reaches before the start of the
    /// sequence or below `win_lo`.
    pub fn window_positions(&self, pos: usize, win_lo: usize) -> Vec<Option<usize>> {
        (0..self.window_size)
            .map(|i| {
                let back = self.window_size - 1 - i;
                pos.checked_sub(back).filter(|&p| p >= win_lo)
            })
            .collect()
    }

    /// Compressed rows visible to a query at `pos`: `(pos + 1) / ratio`.
    pub fn compress_len(&self, pos: usize, ratio: usize) -> usize {
        if ratio == 0 { 0 } else { (pos + 1) / ratio }
    }
}

/// Rows an indexer may score inside one static decode bucket.
///
/// The `+ 1` is the pending unpaired compressed row that a ratio-2 compressor
/// carries across chunks. Dropping it is the V4.1 spelling of the GLM
/// `usable_pools` off-by-one.
pub fn indexer_score_rows(cache_rows: usize, context_cap: usize, ratio: usize) -> Option<usize> {
    if cache_rows == 0 || context_cap == 0 || ratio == 0 {
        return None;
    }
    Some(cache_rows.min(context_cap / ratio + 1))
}

/// Real score columns an indexer may select before padding back to
/// `index_topk`. A 512-position bucket over a ratio-2 cache has 257 score rows,
/// so an unclamped `topk(512)` is invalid.
pub fn indexer_topk_width(index_topk: usize, score_rows: usize) -> Option<usize> {
    if index_topk == 0 {
        return None;
    }
    Some(index_topk.min(score_rows))
}

/// Smallest power-of-two decode capacity covering a request, floored at
/// `bucket_min`. The backing caches stay `max_seq` long; this only avoids
/// scoring future-zero index rows.
pub fn context_bucket(required: usize, max_seq: usize, bucket_min: usize) -> Option<usize> {
    if required == 0 || max_seq == 0 || required > max_seq {
        return None;
    }
    let pow2 = required.next_power_of_two();
    Some(pow2.max(bucket_min).min(max_seq))
}

/// Pad a selection to exactly `index_topk` columns with `-1`.
///
/// NOT an optimisation to delete. The reference is explicit: a varying column
/// count makes cuBLAS pick a different kernel for the downstream score GEMM,
/// and the resulting ulp differences flip MoE router decisions. The extra
/// columns are fully masked and change nothing mathematically.
pub fn pad_topk(mut idx: Vec<i32>, index_topk: usize) -> Vec<i32> {
    if idx.len() < index_topk {
        idx.resize(index_topk, -1);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s() -> SparseSchedule {
        SparseSchedule::v41_flash_next()
    }

    #[test]
    fn ratios_match_the_checkpoint() {
        let s = s();
        assert_eq!(s.compress_ratios.len(), 40);
        assert_eq!(s.ratio(0), 0);
        assert_eq!(s.ratio(1), 0);
        assert_eq!(s.ratio(2), 2);
        assert_eq!(s.ratio(19), 2);
        assert_eq!(s.ratio(20), 1);
        assert_eq!(s.ratio(39), 1);
    }

    #[test]
    fn kv_cache_is_inherited_from_the_most_recent_source() {
        let s = s();
        assert_eq!(s.kv_source_for(0), None);
        assert_eq!(s.kv_source_for(1), None);
        assert_eq!(s.kv_source_for(2), Some(2));
        assert_eq!(s.kv_source_for(7), Some(2));
        assert_eq!(s.kv_source_for(8), Some(8));
        assert_eq!(s.kv_source_for(19), Some(14));
        assert_eq!(s.kv_source_for(20), Some(20));
        assert_eq!(s.kv_source_for(39), Some(20));
    }

    /// The reference asserts `sh.ratio == w.ratio` on every layer. That holds
    /// only because the ratio flips 2 -> 1 exactly AT a kv-source layer. If a
    /// future schedule moves the flip off a source boundary, a layer would
    /// inherit a cache built at the wrong ratio.
    #[test]
    fn ratio_never_changes_between_kv_sources() {
        let s = s();
        for l in 2..s.num_layers {
            let src = s.kv_source_for(l).expect("layer >= 2 has a kv source");
            assert_eq!(
                s.ratio(l),
                s.ratio(src),
                "layer {l} inherits layer {src}'s cache but has a different ratio"
            );
        }
    }

    #[test]
    fn only_eight_layers_run_an_indexer() {
        let s = s();
        let n = (0..s.num_layers).filter(|&l| s.runs_indexer(l)).count();
        assert_eq!(n, 8);
        assert_eq!(s.index_source_for(3), Some(2));
        assert_eq!(s.index_source_for(23), Some(20));
        assert_eq!(s.index_source_for(39), Some(36));
        assert_eq!(s.index_source_for(1), None);
    }

    /// Layers 2, 8 and 14 run BEFORE the candidate source and must not use it.
    #[test]
    fn candidate_mask_is_used_only_after_layer_20() {
        let s = s();
        for l in [2usize, 8, 14, 20] {
            assert!(!s.uses_candidate_mask(l), "layer {l} must not use the mask");
        }
        for l in [24usize, 28, 32, 36] {
            assert!(s.uses_candidate_mask(l), "layer {l} must use the mask");
        }
    }

    #[test]
    fn candidate_mask_is_wide_enough_for_every_consumer() {
        assert!(s().bad_candidate_consumers().is_empty());
    }

    #[test]
    fn yarn_rope_table_selection() {
        let s = s();
        assert!(!s.uses_yarn_rope(0));
        assert!(!s.uses_yarn_rope(1));
        for l in 2..40 {
            assert!(s.uses_yarn_rope(l), "layer {l} must use freqs_c");
        }
    }

    #[test]
    fn window_is_clipped_at_the_start_of_the_sequence() {
        let s = s();
        let w = s.window_positions(3, 0);
        assert_eq!(w.len(), 128);
        assert_eq!(w[127], Some(3));
        assert_eq!(w[124], Some(0));
        assert_eq!(w[123], None);
        assert!(w[..124].iter().all(Option::is_none));
    }

    #[test]
    fn win_lo_masks_below_the_replay_floor() {
        let s = s();
        let w = s.window_positions(200, 150);
        assert_eq!(w[127], Some(200));
        assert_eq!(w[77], Some(150));
        assert_eq!(w[76], None);
    }

    /// The exact off-by-one that cost GLM three tokens at every `q % 4 == 3`.
    /// A 512-position bucket over a ratio-2 cache has 257 rows, not 256.
    #[test]
    fn score_rows_keeps_the_pending_unpaired_row() {
        assert_eq!(indexer_score_rows(1_000_000, 512, 2), Some(257));
        assert_eq!(indexer_score_rows(1_000_000, 512, 1), Some(513));
        assert_eq!(indexer_score_rows(100, 512, 2), Some(100)); // cache is the binding limit
        assert_eq!(indexer_score_rows(0, 512, 2), None);
        assert_eq!(indexer_score_rows(10, 512, 0), None);
    }

    /// NEGATIVE CONTROL for the above: the without-pending spelling a careless
    /// port writes. It must disagree, or the test above proves nothing.
    #[test]
    fn score_rows_control_without_pending_row_disagrees() {
        let correct = indexer_score_rows(1_000_000, 512, 2).unwrap();
        let wrong = 1_000_000usize.min(512 / 2); // the dropped `+ 1`
        assert_ne!(correct, wrong);
        assert_eq!(correct - wrong, 1);
    }

    #[test]
    fn topk_width_clamps_to_available_rows() {
        assert_eq!(indexer_topk_width(512, 257), Some(257));
        assert_eq!(indexer_topk_width(512, 4224), Some(512));
        assert_eq!(indexer_topk_width(0, 10), None);
    }

    #[test]
    fn context_bucket_is_a_floored_power_of_two() {
        assert_eq!(context_bucket(1000, 1 << 20, 32768), Some(32768));
        assert_eq!(context_bucket(40000, 1 << 20, 32768), Some(65536));
        assert_eq!(context_bucket(65536, 1 << 20, 32768), Some(65536));
        assert_eq!(context_bucket(65537, 1 << 20, 32768), Some(131072));
        assert_eq!(context_bucket(1 << 20, 1 << 20, 32768), Some(1 << 20));
        assert_eq!(context_bucket((1 << 20) + 1, 1 << 20, 32768), None);
        assert_eq!(context_bucket(0, 1 << 20, 32768), None);
    }

    #[test]
    fn selection_is_always_padded_to_index_topk() {
        assert_eq!(pad_topk(vec![1, 5, 9], 8), vec![1, 5, 9, -1, -1, -1, -1, -1]);
        assert_eq!(pad_topk(vec![0; 512], 512).len(), 512);
        // already at width: untouched, never truncated
        assert_eq!(pad_topk(vec![7; 600], 512).len(), 600);
    }

    #[test]
    fn compress_len_counts_visible_rows() {
        let s = s();
        assert_eq!(s.compress_len(0, 1), 1);
        assert_eq!(s.compress_len(0, 2), 0); // position 0 alone is still pending
        assert_eq!(s.compress_len(1, 2), 1);
        assert_eq!(s.compress_len(4223, 1), 4224);
        assert_eq!(s.compress_len(9, 0), 0);
    }
}
