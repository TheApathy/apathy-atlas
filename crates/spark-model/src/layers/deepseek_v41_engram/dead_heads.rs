// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 engram dead-head masking: which of the 24 hash columns an image
//! span kills at each position.
//!
//! Ported bit-for-bit from the checkpoint's own `engine/vision.py`
//! `engram_dead_heads`. This is a PURE function of token ids — no hash state, no
//! table, no per-layer data — and it is intentionally kept separate from
//! [`super::gather::EngramGather`]: the reference applies it AFTER the NVMe fetch
//! (`rows.masked_fill(dead_heads.unsqueeze(-1), 0)`), so folding it into the
//! gather would silently change the contract on the one path (multimodal
//! prompts) where nobody is currently watching for it.
//!
//! ## Semantics
//!
//! `dead[p, col]` is true when the n-gram window that column's group reads from
//! touches an image token. Columns are grouped exactly as [`super::hash`] lays
//! out `EngramHashState::forward`'s output: cols 0..8 are the 2-gram heads
//! (offsets 0,1), 8..16 the 3-gram heads (offsets 0,1,2), 16..24 the 4-gram heads
//! (offsets 0,1,2,3) — group `g` covers `ngram = g + 2` and looks back
//! `0..ngram` positions.
//!
//! Unlike [`super::hash::EngramHashState`], this does **not** carry history
//! across a chunk boundary: the reference is called fresh on each forward's local
//! `ids` (`engine/model.py:747`, `dead_heads = engram_dead_heads(ids)`), so a
//! chunk's own start is always treated as if nothing precedes it. Confirmed
//! empirically: `runD_L20_kernel`'s chunk 1 (S=512) mask is bit-identical to
//! chunk 0's on a text-only prompt (both all-False), and `runE_image`'s image
//! span sits entirely inside chunk 0 — chunk 1 restarts at all-False rather than
//! inheriting anything from chunk 0's tail.

/// `IMAGE_SENTINEL_ID` / `IMAGE_PAD_ID` from `engine/vision.py`. These are V4.1's
/// own multimodal token ids, unrelated to any other model's image-pad constants
/// in this crate.
pub const IMAGE_SENTINEL_ID: u32 = 129_264;
pub const IMAGE_PAD_ID: u32 = 129_265;

/// Row ids emitted per position ((`max_ngram_size` - 1) * `n_heads` = 3 * 8).
pub const N_HEAD_COLS: usize = 24;
const HEADS_PER_GROUP: usize = 8;
/// `(group index, n-gram size)`: group 0 is the 2-gram, group 1 the 3-gram, group
/// 2 the 4-gram — the same order [`super::hash`] emits columns in.
const NGRAM_GROUPS: [usize; 3] = [2, 3, 4];

/// `image_sentinel_mask` from `engine/vision.py`: true for either multimodal
/// marker id.
pub fn image_sentinel_mask(ids: &[u32]) -> Vec<bool> {
    ids.iter().map(|&id| id == IMAGE_SENTINEL_ID || id == IMAGE_PAD_ID).collect()
}

/// `engram_dead_heads`: `[T, N_HEAD_COLS]` row-major, `out[p * N_HEAD_COLS + col]`.
///
/// `ids` is one forward chunk's local token ids — see the module doc for why this
/// does not look further back than the start of `ids` itself.
pub fn engram_dead_heads(ids: &[u32]) -> Vec<bool> {
    let t = ids.len();
    let image = image_sentinel_mask(ids);
    let mut dead = vec![false; t * N_HEAD_COLS];
    for (group, &ngram) in NGRAM_GROUPS.iter().enumerate() {
        for offset in 0..ngram {
            for p in 0..t {
                // `shifted[offset:] = image[:-offset]`, zero elsewhere: position p
                // sees image[p - offset] once p >= offset, else false.
                if p >= offset && image[p - offset] {
                    let base = p * N_HEAD_COLS + group * HEADS_PER_GROUP;
                    for h in 0..HEADS_PER_GROUP {
                        dead[base + h] = true;
                    }
                }
            }
        }
    }
    dead
}

/// Apply the reference's `rows.masked_fill(dead_heads.unsqueeze(-1), 0)`: zero
/// every `[row_dim]` slice whose `(position, col)` is dead. `rows` is `[T,
/// N_HEAD_COLS, row_dim]` row-major, matching `engram_rows_premask` /
/// `engram_rows` in the oracle captures.
pub fn apply_dead_mask(rows: &mut [f32], dead: &[bool], t: usize, row_dim: usize) {
    assert_eq!(dead.len(), t * N_HEAD_COLS, "dead mask is not [T, {N_HEAD_COLS}]");
    assert_eq!(rows.len(), t * N_HEAD_COLS * row_dim, "rows is not [T, {N_HEAD_COLS}, {row_dim}]");
    for (i, &is_dead) in dead.iter().enumerate() {
        if is_dead {
            let base = i * row_dim;
            rows[base..base + row_dim].fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A prompt with no image tokens at all must come back entirely alive: the
    /// mask cannot invent a dead head from nothing.
    #[test]
    fn text_only_is_all_false() {
        let ids = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let dead = engram_dead_heads(&ids);
        assert!(dead.iter().all(|&d| !d), "a text-only prompt produced a dead head");
    }

    /// One image sentinel at position p kills columns 0..24 at p (offset 0 in
    /// every group), and 8..24 at p+1 (offset 1 reaches every group with
    /// ngram >= 2, i.e. all three), and 16..24 at p+2 (only the 4-gram group's
    /// offset 2 reaches), and nothing at p+3 (max offset in any group is 3, only
    /// reached by the 4-gram's own position, i.e. p+3 needs offset 3 which only
    /// the 4-gram group has, so col 16..24 dies there too) or beyond p+3.
    #[test]
    fn single_sentinel_kills_the_predicted_columns_forward_in_time() {
        let ids = [1u32, 2, 129_264, 4, 5, 6, 7, 8];
        let dead = engram_dead_heads(&ids);
        let row = |p: usize| &dead[p * N_HEAD_COLS..(p + 1) * N_HEAD_COLS];

        // p=2: the sentinel itself. offset 0 in every group -> all 24 columns.
        assert!(row(2).iter().all(|&d| d), "the sentinel position itself must be fully dead");
        // p=3: offset 1 from the sentinel. 2-gram (needs offset<2), 3-gram (<3),
        // 4-gram (<4) all include offset 1 -> all 24 again.
        assert!(row(3).iter().all(|&d| d), "offset 1 must reach every group (min ngram size 2)");
        // p=4: offset 2. Only 3-gram (offsets 0,1,2) and 4-gram (0,1,2,3) reach;
        // the 2-gram (offsets 0,1 only) does not.
        assert!(row(4)[0..8].iter().all(|&d| !d), "the 2-gram group must not see offset 2");
        assert!(row(4)[8..24].iter().all(|&d| d), "3- and 4-gram groups must see offset 2");
        // p=5: offset 3. Only the 4-gram (offsets 0..4) reaches.
        assert!(row(5)[0..16].iter().all(|&d| !d), "2- and 3-gram groups must not see offset 3");
        assert!(row(5)[16..24].iter().all(|&d| d), "the 4-gram group must see offset 3");
        // p=6: offset 4, past every group's reach.
        assert!(row(6).iter().all(|&d| !d), "no group reaches 4 positions back");
        // p=0,1 precede the sentinel entirely.
        assert!(row(0).iter().all(|&d| !d));
        assert!(row(1).iter().all(|&d| !d));
    }

    /// IMAGE_PAD_ID must kill columns exactly like IMAGE_SENTINEL_ID — the
    /// reference ORs both into one mask, so a port that only checks one id would
    /// silently under-mask every image span's pad run.
    #[test]
    fn image_pad_id_is_equivalent_to_sentinel_id() {
        let sentinel = engram_dead_heads(&[1u32, 129_264, 3]);
        let pad = engram_dead_heads(&[1u32, 129_265, 3]);
        assert_eq!(sentinel, pad, "IMAGE_PAD_ID did not kill the same columns as IMAGE_SENTINEL_ID");
    }

    /// Negative control: shifting the mask by one position (the off-by-one this
    /// module's own doc warns is easy to introduce) must NOT reproduce the real
    /// function's output on an input where the shift matters.
    ///
    /// The sentinel MUST be at position 0: `p >= offset` and `p > offset` only
    /// disagree at `p == offset`, i.e. only when the look-back reaches all the
    /// way to position 0. A sentinel anywhere else makes this control vacuous —
    /// found by running it once and getting a false pass.
    #[test]
    fn shifted_mask_is_a_different_function() {
        let ids = [129_264u32, 2, 3, 4, 5];
        let real = engram_dead_heads(&ids);

        // Deliberately wrong: offset the lookback by one extra position (as if
        // `p >= offset` had been written `p > offset`), which would make the
        // sentinel's own row look alive.
        let image = image_sentinel_mask(&ids);
        let t = ids.len();
        let mut wrong = vec![false; t * N_HEAD_COLS];
        for (group, &ngram) in NGRAM_GROUPS.iter().enumerate() {
            for offset in 0..ngram {
                for p in 0..t {
                    if p > offset && image[p - offset] {
                        let base = p * N_HEAD_COLS + group * HEADS_PER_GROUP;
                        for h in 0..HEADS_PER_GROUP {
                            wrong[base + h] = true;
                        }
                    }
                }
            }
        }
        assert_ne!(real, wrong, "an off-by-one lookback produced the same mask as the real one");
    }

    /// Cross-check against an INDEPENDENT oracle: dsv41-parity's own run of
    /// `engine/vision.py::engram_dead_heads` (not this lane's captures), on a
    /// 10-token fixture built to exercise PAD and SENTINEL together and every
    /// n-gram group's offsets in one short sequence.
    /// `dsv41-parity/crates/spark-server/tests/fixtures/dsv41/vision.json`.
    #[test]
    fn matches_dsv41_parity_independent_fixture() {
        let path = "/home/flocka/atlas/dsv41-parity/crates/spark-server/tests/fixtures/dsv41/vision.json";
        let Ok(text) = std::fs::read_to_string(path) else {
            eprintln!("skipping: {path} not present on this box");
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).expect("parse vision.json");
        let fixture = &v["dead_heads"];
        let ids: Vec<u32> = fixture["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let want: Vec<bool> = fixture["dead"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|b| b.as_u64().unwrap() != 0))
            .collect();

        let got = engram_dead_heads(&ids);
        assert_eq!(got.len(), want.len(), "column count mismatch against the parity fixture");
        let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(mismatches, 0, "{mismatches}/{} entries differ from dsv41-parity's fixture", got.len());
    }

    /// `apply_dead_mask` must zero exactly the dead `(position, col)` slices and
    /// leave every alive one untouched.
    #[test]
    fn apply_dead_mask_zeroes_only_dead_slices() {
        let t = 2;
        let row_dim = 4;
        let mut dead = vec![false; t * N_HEAD_COLS];
        dead[0 * N_HEAD_COLS + 0] = true; // (p=0, col=0) dead
        dead[1 * N_HEAD_COLS + 5] = true; // (p=1, col=5) dead
        let mut rows = vec![1.0f32; t * N_HEAD_COLS * row_dim];
        apply_dead_mask(&mut rows, &dead, t, row_dim);

        let slice = |p: usize, c: usize| &rows[(p * N_HEAD_COLS + c) * row_dim..(p * N_HEAD_COLS + c) * row_dim + row_dim];
        assert_eq!(slice(0, 0), &[0.0; 4], "dead slice (0,0) was not zeroed");
        assert_eq!(slice(1, 5), &[0.0; 4], "dead slice (1,5) was not zeroed");
        assert_eq!(slice(0, 1), &[1.0; 4], "alive slice (0,1) was zeroed");
        assert_eq!(slice(1, 0), &[1.0; 4], "alive slice (1,0) was zeroed");

        // Negative control: a mask that is all-false must be a no-op.
        let mut untouched = vec![2.0f32; t * N_HEAD_COLS * row_dim];
        apply_dead_mask(&mut untouched, &vec![false; t * N_HEAD_COLS], t, row_dim);
        assert!(untouched.iter().all(|&v| v == 2.0), "an all-false mask must not change any row");
    }
}
