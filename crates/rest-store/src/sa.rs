// SPDX-License-Identifier: AGPL-3.0-only

//! Suffix array over a `u32` token alphabet, plus prefix range search.
//!
//! # Why a token-level SA rather than the byte-SA-plus-alignment trick
//!
//! A common shortcut is to encode each `u32` token as 4 big-endian bytes,
//! build a *byte* suffix array with an off-the-shelf divsufsort, and keep
//! only the positions `≡ 0 (mod 4)` — byte matches at aligned positions
//! are exactly token-aligned matches. That trick is correct, but here it
//! costs more than it saves:
//!
//! - It inflates the indexed sequence 4× (a 16 M-token corpus becomes a
//!   64 M-byte one), and the SA it produces is 4× larger before the
//!   alignment filter throws three quarters of it away.
//! - The filter has to run at *build* time to keep the on-disk SA small,
//!   so the peak build memory is the one that matters, and it is 4× worse.
//! - The workspace has no divsufsort/SA-IS dependency vendored, so the
//!   "proven crate" half of the argument does not apply — we would be
//!   adding a dependency to avoid writing an algorithm either way.
//!
//! So we sort token suffixes directly with prefix doubling (Manber–Myers)
//! and a stable counting sort per round: O(n log n), no alignment filter,
//! no 4× blowup, and every comparison is inherently token-aligned. The
//! implementation is short enough to verify exhaustively — [`tests`]
//! checks it against a brute-force suffix sort on random corpora *and*
//! against the byte-SA-with-alignment-filter construction, so the trick
//! we declined is still the oracle.

use std::cmp::Ordering;
use std::ops::Range;

/// Build the suffix array of `tokens`.
///
/// Returns start positions of every suffix in lexicographic order.
/// Suffixes are compared as token sequences; a proper prefix sorts before
/// the sequence it prefixes.
///
/// # Panics
///
/// If `tokens.len()` exceeds `u32::MAX` — the returned array indexes with
/// `u32` to keep the on-disk store half the size of a `u64` index.
pub fn build_suffix_array(tokens: &[u32]) -> Vec<u32> {
    let n = tokens.len();
    assert!(
        n <= u32::MAX as usize,
        "suffix array is u32-indexed; corpus of {n} tokens is too large"
    );
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Round 0: rank by the single token at each position.
    let mut sa: Vec<u32> = (0..n as u32).collect();
    sa.sort_unstable_by_key(|&i| tokens[i as usize]);
    let mut rank = vec![0u32; n];
    let mut max_rank = 0u32;
    for w in 1..n {
        if tokens[sa[w] as usize] != tokens[sa[w - 1] as usize] {
            max_rank += 1;
        }
        rank[sa[w] as usize] = max_rank;
    }

    let mut by_second = vec![0u32; n];
    let mut next_rank = vec![0u32; n];
    let mut k = 1usize;

    // Once every suffix has a distinct rank the order is final.
    while (max_rank as usize) + 1 < n && k < n {
        // Pass 1 — order positions by their SECOND key, rank[i + k].
        // Positions with no second key (i + k >= n) sort first. The rest
        // inherit the previous round's order, which is already sorted by
        // rank[i], i.e. by rank[(i - k) + k] for the shifted position.
        let mut p = 0usize;
        for i in (n - k)..n {
            by_second[p] = i as u32;
            p += 1;
        }
        for &s in &sa {
            if (s as usize) >= k {
                by_second[p] = s - k as u32;
                p += 1;
            }
        }
        debug_assert_eq!(p, n);

        // Pass 2 — stable counting sort by the FIRST key, rank[i].
        // Stability is what makes the pair (rank[i], rank[i+k]) sorted.
        let buckets = max_rank as usize + 1;
        let mut cursor = vec![0u32; buckets + 1];
        for &r in &rank {
            cursor[r as usize + 1] += 1;
        }
        for b in 0..buckets {
            cursor[b + 1] += cursor[b];
        }
        for &s in &by_second {
            let b = rank[s as usize] as usize;
            sa[cursor[b] as usize] = s;
            cursor[b] += 1;
        }

        // Re-rank: adjacent suffixes share a rank iff both keys match.
        let key =
            |i: usize| -> (u32, i64) { (rank[i], if i + k < n { rank[i + k] as i64 } else { -1 }) };
        next_rank[sa[0] as usize] = 0;
        let mut nr = 0u32;
        for w in 1..n {
            let (prev, cur) = (sa[w - 1] as usize, sa[w] as usize);
            if key(prev) != key(cur) {
                nr += 1;
            }
            next_rank[cur] = nr;
        }
        rank.copy_from_slice(&next_rank);
        max_rank = nr;
        k <<= 1;
    }

    sa
}

/// Compare the suffix starting at `start` against `pat`, looking only at
/// the first `pat.len()` tokens.
///
/// A suffix that runs out of tokens while still matching sorts *before*
/// `pat`, which is what makes [`prefix_range`] a half-open range over
/// exactly the suffixes having `pat` as a prefix.
fn cmp_prefix(tokens: &[u32], start: usize, pat: &[u32]) -> Ordering {
    let available = tokens.len() - start;
    let m = pat.len().min(available);
    match tokens[start..start + m].cmp(&pat[..m]) {
        Ordering::Equal if available < pat.len() => Ordering::Less,
        other => other,
    }
}

/// Half-open range of suffix-array slots whose suffix begins with `pat`.
///
/// Empty range means the pattern does not occur. `sa` must be the suffix
/// array of `tokens` as produced by [`build_suffix_array`].
pub fn prefix_range(tokens: &[u32], sa: &[u32], pat: &[u32]) -> Range<usize> {
    if pat.is_empty() || tokens.is_empty() {
        return 0..0;
    }
    let lo = partition_point(sa, |s| cmp_prefix(tokens, s, pat) == Ordering::Less);
    // Search only the tail — the upper bound cannot precede the lower.
    let hi = lo
        + partition_point(&sa[lo..], |s| {
            cmp_prefix(tokens, s, pat) != Ordering::Greater
        });
    lo..hi
}

/// `slice.partition_point` over suffix start positions, with the predicate
/// taking a `usize` position rather than a `&u32`.
fn partition_point(sa: &[u32], pred: impl Fn(usize) -> bool) -> usize {
    sa.partition_point(|&s| pred(s as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: sort every suffix by full comparison.
    fn brute_force_sa(tokens: &[u32]) -> Vec<u32> {
        let mut sa: Vec<u32> = (0..tokens.len() as u32).collect();
        sa.sort_by(|&a, &b| tokens[a as usize..].cmp(&tokens[b as usize..]));
        sa
    }

    /// The trick we declined, used here as an independent oracle: encode
    /// each token big-endian, brute-force the BYTE suffix array, keep the
    /// positions divisible by 4, and divide back into token positions.
    fn byte_sa_aligned(tokens: &[u32]) -> Vec<u32> {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_be_bytes()).collect();
        let mut bsa: Vec<u32> = (0..bytes.len() as u32).collect();
        bsa.sort_by(|&a, &b| bytes[a as usize..].cmp(&bytes[b as usize..]));
        bsa.into_iter()
            .filter(|p| p % 4 == 0)
            .map(|p| p / 4)
            .collect()
    }

    /// Deterministic xorshift so failures reproduce without a rand dep.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn empty_and_singleton() {
        assert!(build_suffix_array(&[]).is_empty());
        assert_eq!(build_suffix_array(&[42]), vec![0]);
    }

    #[test]
    fn matches_brute_force_on_random_corpora() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for alphabet in [2u64, 3, 5, 17, 1000] {
            for len in 1..=64usize {
                let tokens: Vec<u32> = (0..len).map(|_| rng.below(alphabet) as u32).collect();
                assert_eq!(
                    build_suffix_array(&tokens),
                    brute_force_sa(&tokens),
                    "alphabet={alphabet} tokens={tokens:?}"
                );
            }
        }
    }

    #[test]
    fn matches_byte_sa_with_alignment_filter() {
        // The 4-byte-BE + keep-positions-mod-4 construction must agree
        // with the direct token-level sort, since big-endian byte order
        // is order-isomorphic to u32 order.
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..40 {
            let len = 1 + rng.below(48) as usize;
            let tokens: Vec<u32> = (0..len).map(|_| rng.below(6) as u32).collect();
            assert_eq!(
                build_suffix_array(&tokens),
                byte_sa_aligned(&tokens),
                "tokens={tokens:?}"
            );
        }
    }

    #[test]
    fn handles_highly_repetitive_input() {
        // Worst case for prefix doubling: every suffix shares long prefixes.
        let tokens = vec![7u32; 200];
        assert_eq!(build_suffix_array(&tokens), brute_force_sa(&tokens));
        let mut abab: Vec<u32> = Vec::new();
        for i in 0..300 {
            abab.push((i % 2) as u32);
        }
        assert_eq!(build_suffix_array(&abab), brute_force_sa(&abab));
    }

    #[test]
    fn prefix_range_finds_every_occurrence() {
        let tokens: Vec<u32> = vec![1, 2, 3, 1, 2, 4, 1, 2, 3, 9];
        let sa = build_suffix_array(&tokens);

        let mut found: Vec<u32> = prefix_range(&tokens, &sa, &[1, 2]).map(|i| sa[i]).collect();
        found.sort_unstable();
        assert_eq!(found, vec![0, 3, 6]);

        let mut found: Vec<u32> = prefix_range(&tokens, &sa, &[1, 2, 3])
            .map(|i| sa[i])
            .collect();
        found.sort_unstable();
        assert_eq!(found, vec![0, 6]);

        assert!(prefix_range(&tokens, &sa, &[5, 5]).is_empty());
        assert!(prefix_range(&tokens, &sa, &[]).is_empty());
    }

    #[test]
    fn prefix_range_excludes_truncated_tail_match() {
        // [9] occurs at the very end; [9, 9] must not match it.
        let tokens: Vec<u32> = vec![1, 2, 9];
        let sa = build_suffix_array(&tokens);
        assert_eq!(prefix_range(&tokens, &sa, &[9]).len(), 1);
        assert!(prefix_range(&tokens, &sa, &[9, 9]).is_empty());
    }

    #[test]
    fn prefix_range_agrees_with_naive_scan() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE_1234);
        for _ in 0..50 {
            let len = 1 + rng.below(80) as usize;
            let tokens: Vec<u32> = (0..len).map(|_| rng.below(4) as u32).collect();
            let sa = build_suffix_array(&tokens);
            for plen in 1..=4usize {
                let pat: Vec<u32> = (0..plen).map(|_| rng.below(4) as u32).collect();
                let mut expect: Vec<u32> = (0..len)
                    .filter(|&i| i + plen <= len && tokens[i..i + plen] == pat[..])
                    .map(|i| i as u32)
                    .collect();
                let mut got: Vec<u32> = prefix_range(&tokens, &sa, &pat).map(|i| sa[i]).collect();
                got.sort_unstable();
                expect.sort_unstable();
                assert_eq!(got, expect, "tokens={tokens:?} pat={pat:?}");
            }
        }
    }
}
