// SPDX-License-Identifier: AGPL-3.0-only

//! mmap-backed reader for a built store.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use crate::format::StoreHeader;
use crate::sa::prefix_range;
use crate::trie::{DraftTree, TrieParams, build_draft_trie};

/// The result of a longest-suffix lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchSet {
    /// Length of the matched context suffix, in tokens.
    pub match_len: usize,
    /// Corpus start positions of (a sample of) the matching occurrences.
    /// The continuation of each begins at `position + match_len`.
    pub positions: Vec<u32>,
    /// How many occurrences existed before the scan cap was applied.
    pub total_occurrences: usize,
}

/// A memory-mapped REST draft store.
///
/// The file is mapped once and never copied; `tokens` and `suffix_array`
/// are views straight into the mapping, so opening a multi-gigabyte store
/// is a few syscalls and costs no resident memory until pages are touched.
pub struct RestStore {
    // Field order matters: the slices borrow from `_mmap`, so `_mmap` must
    // outlive them. Rust drops fields in declaration order, so it is
    // declared last.
    header: StoreHeader,
    path: PathBuf,
    tokens: &'static [u32],
    suffix_array: &'static [u32],
    doc_starts: &'static [u64],
    _mmap: Mmap,
}

/// Hand-written so that debug output stays a one-liner: deriving would
/// try to format the entire corpus and suffix array.
impl std::fmt::Debug for RestStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestStore")
            .field("path", &self.path)
            .field("n_tokens", &self.header.n_tokens)
            .field("n_docs", &self.header.n_docs)
            .field("sep_token", &self.header.sep_token)
            .finish()
    }
}

impl RestStore {
    /// Open a store and validate its header against `expected_tokenizer_fp`.
    ///
    /// Passing `None` skips the tokenizer check — appropriate only for
    /// offline inspection tools. The server always passes the fingerprint
    /// of the tokenizer it actually loaded: token ids from a different
    /// tokenizer are not merely lower quality, they are meaningless, and
    /// would silently poison every draft.
    pub fn open(path: impl AsRef<Path>, expected_tokenizer_fp: Option<u64>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !cfg!(target_endian = "little") {
            bail!("REST store format is little-endian; this target is big-endian");
        }
        let file =
            File::open(&path).with_context(|| format!("opening REST store {}", path.display()))?;
        // SAFETY: we map the file read-only. The usual mmap caveat applies
        // — concurrent truncation by another process would fault — which is
        // why the builder writes to a temp path and renames into place.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap of REST store {}", path.display()))?;

        let header = StoreHeader::decode(&mmap)
            .with_context(|| format!("REST store header in {}", path.display()))?;
        if mmap.len() < header.total_bytes() {
            bail!(
                "REST store {} is truncated: {} bytes on disk, header implies {}",
                path.display(),
                mmap.len(),
                header.total_bytes()
            );
        }
        if let Some(expected) = expected_tokenizer_fp
            && expected != header.tokenizer_fp
        {
            bail!(
                "REST store {} was built with tokenizer fingerprint {:#018x}, but the \
                 loaded tokenizer is {:#018x}; rebuild the store against this tokenizer",
                path.display(),
                header.tokenizer_fp,
                expected
            );
        }

        let n = header.n_tokens as usize;
        let n_docs = header.n_docs as usize;
        let base = mmap.as_ptr();
        // Section offsets are multiples of 16 and the mapping is page
        // aligned, so every section satisfies its element alignment. Assert
        // it anyway rather than trust arithmetic done in another module.
        for (off, align) in [
            (header.tokens_offset(), align_of::<u32>()),
            (header.sa_offset(), align_of::<u32>()),
            (header.docs_offset(), align_of::<u64>()),
        ] {
            if !(base as usize + off).is_multiple_of(align) {
                bail!(
                    "REST store section at offset {off} is misaligned for a {align}-byte element"
                );
            }
        }

        // SAFETY: `mmap` covers `header.total_bytes()` (checked above), each
        // section start is in-bounds and correctly aligned (checked above),
        // and every bit pattern is a valid u32/u64. The `'static` lifetime is
        // a lie contained by this struct: the slices are only ever handed out
        // behind `&self`, and `_mmap` is dropped last, so no borrow outlives
        // the mapping.
        let (tokens, suffix_array, doc_starts) = unsafe {
            (
                std::slice::from_raw_parts(base.add(header.tokens_offset()).cast::<u32>(), n),
                std::slice::from_raw_parts(base.add(header.sa_offset()).cast::<u32>(), n),
                std::slice::from_raw_parts(
                    base.add(header.docs_offset()).cast::<u64>(),
                    n_docs + 1,
                ),
            )
        };

        tracing::info!(
            path = %path.display(),
            n_tokens = n,
            n_docs,
            sep_token = header.sep_token,
            tokenizer_fp = format_args!("{:#018x}", header.tokenizer_fp),
            bytes = mmap.len(),
            "REST draft store mapped"
        );

        Ok(Self {
            header,
            path,
            tokens,
            suffix_array,
            doc_starts,
            _mmap: mmap,
        })
    }

    /// Store header, for diagnostics.
    pub fn header(&self) -> &StoreHeader {
        &self.header
    }

    /// Path the store was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The concatenated corpus token stream.
    pub fn tokens(&self) -> &[u32] {
        self.tokens
    }

    /// The suffix array over [`RestStore::tokens`].
    pub fn suffix_array(&self) -> &[u32] {
        self.suffix_array
    }

    /// Document index containing corpus position `pos`.
    pub fn doc_of(&self, pos: u32) -> usize {
        // doc_starts is a sorted prefix-offset array with a trailing
        // sentinel, so the owning document is one partition_point away.
        self.doc_starts
            .partition_point(|&s| s <= pos as u64)
            .saturating_sub(1)
    }

    /// Find the longest suffix of `ctx` (capped at `max_k` tokens) that
    /// occurs in the corpus with at least one continuation.
    ///
    /// Searches longest-first and returns on the first length that yields
    /// a usable occurrence, so the result is the longest match, not merely
    /// *a* match. Occurrences are capped at `max_occurrences`.
    pub fn longest_suffix_match(
        &self,
        ctx: &[u32],
        max_k: usize,
        max_occurrences: usize,
    ) -> Option<MatchSet> {
        let k = max_k.min(ctx.len());
        if k == 0 || max_occurrences == 0 || self.tokens.is_empty() {
            return None;
        }
        for len in (1..=k).rev() {
            let pat = &ctx[ctx.len() - len..];
            let range = prefix_range(self.tokens, self.suffix_array, pat);
            if range.is_empty() {
                continue;
            }
            let total = range.len();
            let positions = self.sample_positions(range, len, max_occurrences);
            if !positions.is_empty() {
                return Some(MatchSet {
                    match_len: len,
                    positions,
                    total_occurrences: total,
                });
            }
            // Every occurrence at this length sat at a document or corpus
            // boundary — fall through and try a shorter, more common suffix.
        }
        None
    }

    /// Collect at most `cap` usable occurrences from a suffix-array range.
    ///
    /// A short suffix can occur tens of thousands of times, and the range
    /// is ordered lexicographically *by continuation*. Taking the first
    /// `cap` slots would therefore sample only continuations beginning
    /// with the smallest token ids, which biases the trie's frequency
    /// estimates toward whatever token happens to sort first. Striding
    /// evenly across the range samples the continuation distribution
    /// instead.
    fn sample_positions(
        &self,
        range: std::ops::Range<usize>,
        match_len: usize,
        cap: usize,
    ) -> Vec<u32> {
        let n = self.tokens.len();
        let sep = self.header.sep_token;
        let usable = |&slot: &usize| -> Option<u32> {
            let pos = self.suffix_array[slot];
            let next = pos as usize + match_len;
            (next < n && self.tokens[next] != sep).then_some(pos)
        };
        let total = range.len();
        if total <= cap {
            return range.filter_map(|s| usable(&s)).collect();
        }
        let start = range.start;
        // Stride so the `cap` samples span the whole range.
        (0..cap)
            .map(|i| start + (i * total) / cap)
            .filter_map(|s| usable(&s))
            .collect()
    }

    /// Lookup plus trie construction in one call.
    ///
    /// Returns `None` when no suffix matched, when the longest match is
    /// shorter than `min_match`, or when no occurrence had a continuation.
    pub fn propose(
        &self,
        ctx: &[u32],
        max_k: usize,
        min_match: usize,
        max_occurrences: usize,
        depth: usize,
        max_nodes: usize,
    ) -> Option<DraftTree> {
        let m = self.longest_suffix_match(ctx, max_k, max_occurrences)?;
        if m.match_len < min_match {
            return None;
        }
        build_draft_trie(
            self.tokens,
            &m.positions,
            m.match_len,
            TrieParams {
                depth,
                max_nodes,
                sep_token: self.header.sep_token,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::write_store;

    /// Build a tiny store on disk and map it back.
    fn round_trip(tokens: &[u32], docs: &[u64], sep: u32) -> (tempfile::TempDir, RestStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rest");
        write_store(&path, tokens, docs, sep, 0xABCD).unwrap();
        let store = RestStore::open(&path, Some(0xABCD)).unwrap();
        (dir, store)
    }

    #[test]
    fn maps_back_what_was_written() {
        let tokens: Vec<u32> = vec![5, 6, 7, 0, 5, 6, 8];
        let (_d, store) = round_trip(&tokens, &[0, 4], 0);
        assert_eq!(store.tokens(), &tokens[..]);
        assert_eq!(store.header().n_tokens, 7);
        assert_eq!(store.header().n_docs, 2);
        assert_eq!(store.doc_of(0), 0);
        assert_eq!(store.doc_of(3), 0);
        assert_eq!(store.doc_of(4), 1);
        assert_eq!(store.doc_of(6), 1);
    }

    #[test]
    fn rejects_tokenizer_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.rest");
        write_store(&path, &[1, 2, 3], &[0], 0, 0x1111).unwrap();
        let err = RestStore::open(&path, Some(0x2222))
            .unwrap_err()
            .to_string();
        assert!(err.contains("tokenizer fingerprint"), "got: {err}");
        // No expectation means no check.
        assert!(RestStore::open(&path, None).is_ok());
    }

    #[test]
    fn finds_the_longest_suffix_not_merely_a_suffix() {
        // [1,2,3] occurs once (continuation 9); [2,3] occurs twice.
        let tokens: Vec<u32> = vec![1, 2, 3, 9, 4, 2, 3, 8];
        let (_d, store) = round_trip(&tokens, &[0], 0);
        let m = store
            .longest_suffix_match(&[7, 7, 1, 2, 3], 16, 64)
            .unwrap();
        assert_eq!(m.match_len, 3);
        assert_eq!(m.positions, vec![0]);

        // Context ending [4,2,3] -> longest match is [4,2,3] at pos 4, but
        // that continuation is 8.
        let m = store.longest_suffix_match(&[4, 2, 3], 16, 64).unwrap();
        assert_eq!(m.match_len, 3);
        assert_eq!(m.positions, vec![4]);
    }

    #[test]
    fn falls_back_when_the_longest_match_has_no_continuation() {
        // [7,8] occurs only at the very end, so it has no continuation;
        // the lookup must fall back to [8]... which also has none, then
        // give up. Add an earlier [8] with a continuation to prove the
        // fallback fires rather than returning the unusable long match.
        let tokens: Vec<u32> = vec![8, 5, 1, 7, 8];
        let (_d, store) = round_trip(&tokens, &[0], 0);
        let m = store.longest_suffix_match(&[7, 8], 16, 64).unwrap();
        assert_eq!(m.match_len, 1, "should fall back from [7,8] to [8]");
        assert_eq!(m.positions, vec![0]);
    }

    #[test]
    fn skips_occurrences_whose_continuation_is_a_separator() {
        // [5,6] occurs twice; the first is followed by the separator.
        let tokens: Vec<u32> = vec![5, 6, 0, 5, 6, 7];
        let (_d, store) = round_trip(&tokens, &[0, 3], 0);
        let m = store.longest_suffix_match(&[5, 6], 16, 64).unwrap();
        assert_eq!(m.match_len, 2);
        assert_eq!(m.positions, vec![3]);
    }

    #[test]
    fn no_match_returns_none() {
        let tokens: Vec<u32> = vec![1, 2, 3];
        let (_d, store) = round_trip(&tokens, &[0], 0);
        assert!(store.longest_suffix_match(&[77, 88], 16, 64).is_none());
        assert!(store.longest_suffix_match(&[], 16, 64).is_none());
        assert!(store.longest_suffix_match(&[1], 0, 64).is_none());
    }

    #[test]
    fn max_k_caps_the_match_length() {
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 9];
        let (_d, store) = round_trip(&tokens, &[0], 0);
        let m = store.longest_suffix_match(&[1, 2, 3, 4, 5], 2, 64).unwrap();
        assert_eq!(m.match_len, 2);
    }

    #[test]
    fn propose_honours_the_engage_gate() {
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 42, 43];
        let (_d, store) = round_trip(&tokens, &[0], 0);
        let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        // match_len 8 >= min_match 8 -> engages.
        let tree = store.propose(&ctx, 16, 8, 64, 16, 16).unwrap();
        assert_eq!(tree.match_len, 8);
        assert_eq!(tree.spine(), vec![42, 43]);
        // min_match 9 -> declines.
        assert!(store.propose(&ctx, 16, 9, 64, 16, 16).is_none());
    }

    #[test]
    fn occurrence_cap_is_respected() {
        // 200 copies of [1,2] each followed by a distinct token.
        let mut tokens: Vec<u32> = Vec::new();
        for i in 0..200u32 {
            tokens.extend_from_slice(&[1, 2, 1000 + i]);
        }
        let (_d, store) = round_trip(&tokens, &[0], 0);
        let m = store.longest_suffix_match(&[1, 2], 16, 64).unwrap();
        assert_eq!(m.match_len, 2);
        assert!(m.positions.len() <= 64, "got {}", m.positions.len());
        assert!(m.total_occurrences >= 200);
    }
}
