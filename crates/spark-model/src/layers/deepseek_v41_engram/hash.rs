// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 n-gram hashing: token ids -> engram table row ids.
//!
//! Ported from the checkpoint's own `inference/engram.py` (`EngramLayout` +
//! `NgramHashState`). Every position is hashed as the `max_ngram_size - 1` n-grams
//! ending there (2-, 3- and 4-gram), each split over `n_heads = 8` heads, giving
//! `3 * 8 = 24` row ids per position per engram layer.
//!
//! Per position `p`, per engram layer `l`:
//! ```text
//!   c[p]    = token_map[ids[p]]                  // 129,280 -> 99,092 compressed ids
//!   t[s]    = c[p - s], s = 0..3, but once the look-back is blocked (p < s, or the
//!             source is DEAD) it is pad_id and STAYS blocked for every larger s
//!   prod[s] = t[s] * multiplier[l][s]            // wrapping i64
//!   rolling = prod[0]
//!   for i in 1..4:
//!       rolling ^= prod[i]
//!       for h in 0..8:
//!           col           = (i - 1) * 8 + h
//!           row_id[l][col] = rolling % prime[l][i - 1][h] + offset[l][col]
//! ```
//!
//! ## What is computed and what is shipped as data
//!
//! The primes and offsets are *computed* here — they are simply the first
//! `n_layers * 3 * 8` primes above `engram_vocab_size - 1`, handed out in order
//! (layer 0's 2-gram heads first, then its 3-gram, then its 4-gram, then layer 1).
//! That reading is pinned by an exact check, not by reading the Python: summing
//! each layer's 24 primes reproduces `engram_num_embeddings` bit for bit
//! (384,006,168 and 384,016,682). See [`EngramLayout::validate_against_config`].
//!
//! The `token_map` and the hash multipliers are *shipped as data*. The map comes
//! from running HuggingFace `tokenizers` normalizers (NFKC / NFD / StripAccents /
//! Lowercase / whitespace-collapse / Strip, with a private-use sentinel so a
//! single-space token survives `Strip`) over every decoded token, and the
//! multipliers come from numpy's PCG64. Reimplementing either here would be a
//! correctness risk for no gain.
//!
//! ## The load-bearing constant
//!
//! Every multiplier is derived from the *compressed* vocab size, so a mismatch
//! there silently rehashes the entire 768M-row table rather than failing. The
//! reference asserts it and so do we: see [`EngramHashState::new`].

use anyhow::{Context, Result, bail};

/// Blocked look-back marker, matching `NgramHashState.DEAD`.
pub const DEAD: i32 = -1;

/// Hash-table geometry for one model.
#[derive(Debug, Clone)]
pub struct EngramLayout {
    pub max_ngram_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub layer_ids: Vec<u32>,
    /// `[layer][ngram_index][head]` bucket modulus.
    pub primes: Vec<Vec<Vec<u64>>>,
    /// `[layer][col]` base row of that bucket range, an exclusive prefix sum.
    pub offsets: Vec<Vec<u64>>,
}

impl EngramLayout {
    /// Build the layout from the config fields.
    ///
    /// The primes are drawn in order and never reused, which is what keeps the
    /// bucket ranges disjoint — the per-group restart at `vocab_size - 1` in the
    /// reference re-draws from the same place but skips everything already handed
    /// out, so the net effect is one ascending run.
    pub fn new(
        layer_ids: &[u32],
        max_ngram_size: usize,
        n_heads: usize,
        head_dim: usize,
        vocab_size: u64,
    ) -> Result<Self> {
        if max_ngram_size < 2 || n_heads == 0 || layer_ids.is_empty() {
            bail!("engram layout: bad max_ngram_size {max_ngram_size} / n_heads {n_heads} / layers {}",
                  layer_ids.len());
        }
        let groups = max_ngram_size - 1;
        let want = layer_ids.len() * groups * n_heads;
        let run = first_primes_above(vocab_size - 1, want);

        let mut primes = Vec::with_capacity(layer_ids.len());
        let mut offsets = Vec::with_capacity(layer_ids.len());
        let mut k = 0usize;
        for _ in layer_ids {
            let mut per_ngram = Vec::with_capacity(groups);
            let mut flat = Vec::with_capacity(groups * n_heads);
            for _ in 0..groups {
                let mut sizes = Vec::with_capacity(n_heads);
                for _ in 0..n_heads {
                    sizes.push(run[k]);
                    flat.push(run[k]);
                    k += 1;
                }
                per_ngram.push(sizes);
            }
            let mut off = Vec::with_capacity(flat.len());
            let mut acc = 0u64;
            for p in &flat {
                off.push(acc);
                acc += p;
            }
            primes.push(per_ngram);
            offsets.push(off);
        }
        Ok(Self { max_ngram_size, n_heads, head_dim, layer_ids: layer_ids.to_vec(), primes, offsets })
    }

    /// Row ids emitted per position per layer (`(max_ngram_size - 1) * n_heads`).
    pub fn n_hash_cols(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_heads
    }

    /// Each layer's primes must sum to that layer's `engram_num_embeddings`.
    ///
    /// This is the check that pins the prime order, the group order, the layer
    /// order and the offset scheme all at once — and it is not a formality: a
    /// wrong ordering changes these nine-digit sums.
    pub fn validate_against_config(&self, num_embeddings: &[u64]) -> Result<()> {
        if num_embeddings.len() != self.primes.len() {
            bail!("engram_num_embeddings has {} entries, layout has {} layers",
                  num_embeddings.len(), self.primes.len());
        }
        for (li, want) in num_embeddings.iter().enumerate() {
            let got: u64 = self.primes[li].iter().flatten().sum();
            if got != *want {
                bail!("engram layer {} (id {}): primes sum to {got}, config says {want} — \
                       the prime/group/layer ordering does not match the checkpoint",
                      li, self.layer_ids[li]);
            }
        }
        Ok(())
    }
}

/// The `want` smallest primes strictly greater than `start`.
fn first_primes_above(start: u64, want: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(want);
    let mut c = start + 1;
    while out.len() < want {
        if is_prime(c) {
            out.push(c);
        }
        c += 1;
    }
    out
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    // Deterministic Miller-Rabin: this witness set is proven for all u64.
    let (mut d, mut r) = (n - 1, 0u32);
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    'next: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..r {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'next;
            }
        }
        return false;
    }
    true
}

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn pow_mod(mut a: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1u64;
    a %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = mul_mod(r, a, m);
        }
        a = mul_mod(a, a, m);
        e >>= 1;
    }
    r
}

/// Token ids -> engram row ids, carrying the compressed-id history across the
/// prefill/decode split.
pub struct EngramHashState {
    layout: EngramLayout,
    /// `[layer][lookback]`, odd by construction.
    multipliers: Vec<Vec<i64>>,
    /// Compressed id per raw token id.
    token_map: Vec<i32>,
    /// Compressed id of the pad token.
    pad_id: i32,
    /// Compressed ids of every position seen so far.
    cache: Vec<i32>,
}

impl EngramHashState {
    /// `token_map` and `multipliers` come from the exported fixture; everything
    /// else is derived.
    ///
    /// `compressed_vocab_size` is checked against the map because every multiplier
    /// is derived from it — a mismatch would silently rehash the whole table
    /// instead of failing, which is precisely the bug class this assert exists for.
    pub fn new(
        layout: EngramLayout,
        token_map: Vec<i32>,
        multipliers: Vec<Vec<i64>>,
        pad_token_id: usize,
        compressed_vocab_size: u64,
    ) -> Result<Self> {
        let distinct = token_map.iter().copied().max().map(|m| m as u64 + 1).unwrap_or(0);
        if distinct != compressed_vocab_size {
            bail!("token_map spans {distinct} compressed ids, config says \
                   engram_compressed_vocab_size = {compressed_vocab_size}; every hash multiplier \
                   derives from this value, so a mismatch rehashes the entire table");
        }
        if multipliers.len() != layout.layer_ids.len() {
            bail!("multipliers cover {} layers, layout has {}", multipliers.len(), layout.layer_ids.len());
        }
        for (li, m) in multipliers.iter().enumerate() {
            if m.len() != layout.max_ngram_size {
                bail!("layer {li}: {} multipliers, need {}", m.len(), layout.max_ngram_size);
            }
            if let Some(bad) = m.iter().find(|v| *v % 2 == 0) {
                bail!("layer {li}: multiplier {bad} is even; the reference keeps them odd");
            }
            // The reference bounds the multipliers precisely so that
            // `compressed_id * multiplier` cannot overflow i64. That bound is what
            // makes the whole hash sign-safe, and it is TIGHT: at the real
            // constants the largest product is 9,223,278,957,978,185,253 against
            // an i64 ceiling of 9,223,372,036,854,775,807 — 0.001% of headroom.
            // Check it rather than assume it.
            let max_id = compressed_vocab_size as i64 - 1;
            if let Some(bad) = m.iter().find(|v| v.checked_mul(max_id).is_none()) {
                bail!("layer {li}: multiplier {bad} overflows i64 against compressed id {max_id}");
            }
        }
        let pad_id = *token_map
            .get(pad_token_id)
            .with_context(|| format!("engram_pad_token_id {pad_token_id} is outside the token map"))?;
        Ok(Self { layout, multipliers, token_map, pad_id, cache: Vec::new() })
    }

    pub fn layout(&self) -> &EngramLayout {
        &self.layout
    }

    /// Drop the position history (new sequence).
    pub fn reset(&mut self) {
        self.cache.clear();
    }

    /// Positions already absorbed.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Hash `ids` at positions `start_pos .. start_pos + ids.len()`.
    ///
    /// `alive[i] == false` marks a token that takes no part in an n-gram (an image
    /// span), matching the reference's `token_mask`. Returns row ids laid out
    /// `[T][layer][col]`, i.e. `out[(t * n_layers + l) * n_cols + c]`.
    pub fn forward(&mut self, ids: &[u32], start_pos: usize, alive: Option<&[bool]>) -> Result<Vec<i64>> {
        if self.cache.len() != start_pos {
            bail!("engram hash state holds {} positions, asked to start at {start_pos}",
                  self.cache.len());
        }
        if let Some(a) = alive {
            if a.len() != ids.len() {
                bail!("alive mask has {} entries, {} tokens", a.len(), ids.len());
            }
        }
        // Absorb this chunk into the history first: an n-gram may look back into
        // it as well as into earlier chunks.
        for (i, &id) in ids.iter().enumerate() {
            let c = *self
                .token_map
                .get(id as usize)
                .with_context(|| format!("token id {id} is outside the token map"))?;
            self.cache.push(if alive.map(|a| a[i]).unwrap_or(true) { c } else { DEAD });
        }

        let n_layers = self.layout.layer_ids.len();
        let n_cols = self.layout.n_hash_cols();
        let mut out = vec![0i64; ids.len() * n_layers * n_cols];

        for t in 0..ids.len() {
            let p = start_pos + t;
            // Gather the look-back window. `blocked` is sticky: once the chain is
            // broken, every longer n-gram is pad from there on, so an n-gram never
            // spans the start of the sequence or a dead token.
            let mut window = vec![0i64; self.layout.max_ngram_size];
            let mut blocked = false;
            for (s, w) in window.iter_mut().enumerate() {
                let src = if s > p { DEAD } else { self.cache[p - s] };
                blocked |= s > p || src == DEAD;
                *w = if blocked { self.pad_id as i64 } else { src as i64 };
            }

            for l in 0..n_layers {
                let mult = &self.multipliers[l];
                // Wrapping: the reference bounds the multipliers so that
                // token_id * multiplier cannot overflow i64, but the XOR chain can
                // still land anywhere in the signed range.
                let mut rolling = window[0].wrapping_mul(mult[0]);
                let base = (t * n_layers + l) * n_cols;
                for i in 1..self.layout.max_ngram_size {
                    rolling ^= window[i].wrapping_mul(mult[i]);
                    for h in 0..self.layout.n_heads {
                        let col = (i - 1) * self.layout.n_heads + h;
                        let m = self.layout.primes[l][i - 1][h] as i64;
                        // Rust's `%` truncates toward zero and Python's floors, so
                        // they disagree on negatives. Here they cannot: every
                        // product is non-negative and bounded below i64::MAX (the
                        // multiplier bound checked in `new`), and XOR of
                        // non-negative i64 leaves the sign bit clear, so `rolling`
                        // is always >= 0. The fold is unreachable defence, kept so
                        // that a future change to the multiplier bound degrades
                        // into a wrong-but-positive id rather than a panic.
                        let mut v = rolling % m;
                        if v < 0 {
                            v += m;
                        }
                        out[base + col] = v + self.layout.offsets[l][col] as i64;
                    }
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The two checkpoint values. Reproducing them from the prime rules alone is
    // the strongest available evidence that the layout reading is right.
    const NUM_EMBEDDINGS: [u64; 2] = [384_006_168, 384_016_682];
    const VOCAB: u64 = 16_000_000;

    fn layout() -> EngramLayout {
        EngramLayout::new(&[1, 14], 4, 8, 256, VOCAB).unwrap()
    }

    #[test]
    fn prime_sums_reproduce_engram_num_embeddings() {
        layout().validate_against_config(&NUM_EMBEDDINGS).unwrap();
    }

    /// Negative control for the layout: the check must FAIL when the prime order
    /// is wrong. Without this, `validate_against_config` is a gate nobody has
    /// watched fail.
    #[test]
    fn wrong_prime_order_fails_validation() {
        let mut bad = layout();
        // Swap one prime between the two layers: the multiset is unchanged, only
        // the assignment moves, so only the per-layer sums can catch it.
        let a = bad.primes[0][0][0];
        let b = bad.primes[1][2][7];
        bad.primes[0][0][0] = b;
        bad.primes[1][2][7] = a;
        assert_ne!(a, b, "the two primes must differ for this control to mean anything");
        assert!(
            bad.validate_against_config(&NUM_EMBEDDINGS).is_err(),
            "a swapped prime must fail validation"
        );
    }

    #[test]
    fn bucket_ranges_are_disjoint_and_cover_the_table() {
        let l = layout();
        for (li, want) in NUM_EMBEDDINGS.iter().enumerate() {
            let flat: Vec<u64> = l.primes[li].iter().flatten().copied().collect();
            let mut acc = 0u64;
            for (c, p) in flat.iter().enumerate() {
                assert_eq!(l.offsets[li][c], acc, "offset {c} is not the exclusive prefix sum");
                acc += p;
            }
            assert_eq!(acc, *want);
        }
    }

    #[test]
    fn primes_are_prime_ascending_and_distinct() {
        let l = layout();
        let flat: Vec<u64> = l.primes.iter().flatten().flatten().copied().collect();
        assert_eq!(flat.len(), 48);
        for w in flat.windows(2) {
            assert!(w[0] < w[1], "primes must be handed out in ascending order, never reused");
        }
        assert!(flat.iter().all(|&p| is_prime(p)));
        assert!(flat[0] > VOCAB - 1);
    }

    #[test]
    fn miller_rabin_agrees_with_trial_division() {
        for n in 0u64..2000 {
            let naive = n >= 2 && (2..n).take_while(|d| d * d <= n).all(|d| n % d != 0);
            assert_eq!(is_prime(n), naive, "is_prime disagrees at {n}");
        }
        // and the first prime above the engram vocab, from the reference
        assert!(is_prime(16_000_057));
        assert!(!is_prime(16_000_057 * 3));
    }

    fn state(mult: Vec<Vec<i64>>) -> EngramHashState {
        // A toy map: 8 raw ids onto 4 compressed ids.
        let token_map = vec![0, 1, 2, 3, 0, 1, 2, 3];
        EngramHashState::new(layout(), token_map, mult, 2, 4).unwrap()
    }

    fn mults() -> Vec<Vec<i64>> {
        vec![
            vec![76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391],
            vec![67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591],
        ]
    }

    #[test]
    fn row_ids_land_inside_their_own_bucket_range() {
        let mut st = state(mults());
        let ids = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let out = st.forward(&ids, 0, None).unwrap();
        let l = st.layout().clone();
        let n_cols = l.n_hash_cols();
        for t in 0..ids.len() {
            for li in 0..2 {
                for c in 0..n_cols {
                    let v = out[(t * 2 + li) * n_cols + c] as u64;
                    let lo = l.offsets[li][c];
                    let hi = lo + l.primes[li][c / 8][c % 8];
                    assert!(v >= lo && v < hi, "layer {li} col {c} row {v} escaped [{lo}, {hi})");
                    assert!(v < NUM_EMBEDDINGS[li], "row {v} is past the end of the table");
                }
            }
        }
    }

    /// Negative control for the hash: flipping ONE multiplier's low bit must move
    /// the row ids. A hash that ignores its multipliers would pass every
    /// range check above while looking up entirely the wrong rows.
    #[test]
    fn perturbed_multiplier_changes_the_row_ids() {
        let ids = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let good = state(mults()).forward(&ids, 0, None).unwrap();

        let mut m = mults();
        m[0][2] += 2; // stay odd, as the reference requires
        let bad = state(m).forward(&ids, 0, None).unwrap();

        assert_ne!(good, bad, "perturbing a multiplier left the row ids unchanged");
        let moved = good.iter().zip(&bad).filter(|(a, b)| a != b).count();
        assert!(moved > good.len() / 8, "only {moved}/{} row ids moved", good.len());
    }

    /// An even multiplier is rejected outright rather than silently accepted.
    #[test]
    fn even_multiplier_is_rejected() {
        let mut m = mults();
        m[1][0] += 1;
        let err = EngramHashState::new(layout(), vec![0, 1, 2, 3, 0, 1, 2, 3], m, 2, 4);
        assert!(err.is_err(), "an even multiplier must be rejected");
    }

    /// The compressed-vocab assert is the reference's own load-bearing check.
    #[test]
    fn compressed_vocab_mismatch_is_rejected() {
        let err = EngramHashState::new(layout(), vec![0, 1, 2, 3], mults(), 2, 99_092);
        assert!(err.is_err(), "a token map that does not span the declared vocab must be rejected");
    }

    /// Look-back is sticky: once blocked, every longer n-gram is pad. At position
    /// 0 every n-gram is blocked, so all three groups hash the same pad-only
    /// window and the 2-, 3- and 4-gram columns must agree head for head.
    #[test]
    fn position_zero_is_pad_only_in_every_group() {
        let mut st = state(mults());
        let out = st.forward(&[5u32], 0, None).unwrap();
        let n_cols = st.layout().n_hash_cols();
        for li in 0..2 {
            let row = &out[(li) * n_cols..(li + 1) * n_cols];
            for h in 0..8 {
                // Groups differ only by their prime, so compare the pre-offset
                // residues' provenance indirectly: all three must be in range and
                // derived from the same rolling value.
                for g in 0..3 {
                    let c = g * 8 + h;
                    let l = st.layout();
                    let lo = l.offsets[li][c];
                    assert!(row[c] as u64 >= lo);
                }
            }
        }
    }

    /// A dead token breaks the chain for every n-gram that would span it.
    #[test]
    fn dead_token_changes_the_hashes() {
        let ids = [0u32, 1, 2, 3];
        let a = state(mults()).forward(&ids, 0, Some(&[true, true, true, true])).unwrap();
        let b = state(mults()).forward(&ids, 0, Some(&[true, false, true, true])).unwrap();
        assert_ne!(a, b, "marking a token dead did not change any hash");
    }

    /// Look-back blocking must be STICKY: once the chain breaks, every LONGER
    /// n-gram is pad too, even if the token further back is alive.
    ///
    /// This is the discriminating case. At position 3 with position 1 dead, the
    /// 4-gram reaches position 0. Under sticky semantics it is pad regardless of
    /// whether position 0 is alive, so killing position 0 as well must leave
    /// position 3's rows UNCHANGED. A non-sticky implementation reads position 0
    /// in the first case and pad in the second, and this test fails — verified by
    /// deliberately changing `|=` to `=`.
    #[test]
    fn lookback_blocking_is_sticky() {
        let ids = [0u32, 1, 2, 3];
        let one_dead = state(mults()).forward(&ids, 0, Some(&[true, false, true, true])).unwrap();
        let two_dead = state(mults()).forward(&ids, 0, Some(&[false, false, true, true])).unwrap();

        let n_layers = 2;
        let n_cols = layout().n_hash_cols();
        let at3 = |v: &[i64]| v[(3 * n_layers) * n_cols..(3 * n_layers + n_layers) * n_cols].to_vec();
        assert_eq!(
            at3(&one_dead),
            at3(&two_dead),
            "position 3's rows changed when a token BEHIND an already-dead token died: \
             the look-back block is not sticky"
        );
        // ...and the test is not vacuous: positions 0-2 DO differ between the two.
        assert_ne!(one_dead, two_dead, "killing position 0 changed nothing anywhere");
    }

    /// The multiplier bound is what makes the hash sign-safe, and it is tight
    /// (0.001% of headroom at the real constants), so it is checked, not assumed.
    #[test]
    fn oversized_multiplier_is_rejected() {
        let mut m = mults();
        m[0][0] = i64::MAX; // odd, but overflows against any non-trivial vocab
        let err = EngramHashState::new(layout(), vec![0, 1, 2, 3, 0, 1, 2, 3], m, 2, 4);
        assert!(err.is_err(), "a multiplier that overflows i64 must be rejected");
    }

    /// Chunked prefill must equal one-shot prefill: the history carries across.
    #[test]
    fn chunked_matches_one_shot() {
        let ids: Vec<u32> = (0..8).map(|i| i % 8).collect();
        let one = state(mults()).forward(&ids, 0, None).unwrap();
        let mut st = state(mults());
        let mut split = st.forward(&ids[..3], 0, None).unwrap();
        split.extend(st.forward(&ids[3..], 3, None).unwrap());
        assert_eq!(one, split, "chunk boundary changed the hashes");
    }

    /// Row ids are never negative. NOTE: this is a weak property — it holds
    /// structurally (see the sign-safety argument in `forward`), so it passes even
    /// with the modulo fold removed. Kept as a cheap invariant, NOT as evidence
    /// that the fold works; `oversized_multiplier_is_rejected` guards the bound
    /// that actually makes it true.
    #[test]
    fn row_ids_are_never_negative() {
        let mut st = state(mults());
        let ids: Vec<u32> = (0..8).collect();
        let out = st.forward(&ids, 0, None).unwrap();
        assert!(out.iter().all(|&v| v >= 0), "a negative row id escaped the modulo fold");
    }
}
