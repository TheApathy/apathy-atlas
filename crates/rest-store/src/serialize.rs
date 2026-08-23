// SPDX-License-Identifier: AGPL-3.0-only

//! Store serialization — turning an indexed corpus into the on-disk file.
//!
//! Split out of [`crate::build`] so the corpus-assembly half (walking,
//! tokenizing, JSONL ingest, holdout) and the byte-layout half stay
//! independently readable.

use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::format::{ALIGN, StoreHeader, align_up};
use crate::sa::build_suffix_array;

/// Serialize a store: header, tokens, suffix array, document offsets.
///
/// `doc_starts` holds one entry per document, without the trailing
/// sentinel — [`write_store`] appends it.
///
/// Writes to `<path>.tmp` and renames into place, so a reader never
/// mmaps a half-written file.
pub fn write_store(
    path: &Path,
    tokens: &[u32],
    doc_starts: &[u64],
    sep_token: u32,
    tokenizer_fp: u64,
) -> Result<u64> {
    if !cfg!(target_endian = "little") {
        bail!("REST store format is little-endian; this target is big-endian");
    }
    let sa = build_suffix_array(tokens);
    write_store_with_sa(path, tokens, &sa, doc_starts, sep_token, tokenizer_fp)
}

/// Write zero bytes until `written` reaches `target`.
fn pad_to(w: &mut BufWriter<std::fs::File>, written: &mut usize, target: usize) -> Result<()> {
    while *written < target {
        let chunk = (target - *written).min(ALIGN);
        w.write_all(&[0u8; ALIGN][..chunk])?;
        *written += chunk;
    }
    Ok(())
}

/// [`write_store`] with a caller-supplied suffix array, so a build that
/// already timed SA construction separately does not redo it.
pub fn write_store_with_sa(
    path: &Path,
    tokens: &[u32],
    sa: &[u32],
    doc_starts: &[u64],
    sep_token: u32,
    tokenizer_fp: u64,
) -> Result<u64> {
    if sa.len() != tokens.len() {
        bail!(
            "suffix array length {} != token count {}",
            sa.len(),
            tokens.len()
        );
    }
    let header = StoreHeader {
        format_version: crate::format::CACHE_FORMAT_VERSION,
        tokenizer_fp,
        n_tokens: tokens.len() as u64,
        n_docs: doc_starts.len() as u64,
        sep_token,
        built_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };

    let tmp = path.with_extension("tmp");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let file =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    let mut written = 0usize;

    w.write_all(&header.encode())?;
    written += crate::format::HEADER_BYTES;

    pad_to(&mut w, &mut written, header.tokens_offset())?;
    w.write_all(as_bytes_u32(tokens))?;
    written += tokens.len() * 4;

    pad_to(&mut w, &mut written, header.sa_offset())?;
    w.write_all(as_bytes_u32(sa))?;
    written += sa.len() * 4;

    pad_to(&mut w, &mut written, header.docs_offset())?;
    let mut docs: Vec<u64> = doc_starts.to_vec();
    docs.push(tokens.len() as u64); // sentinel
    w.write_all(as_bytes_u64(&docs))?;
    written += docs.len() * 8;

    let end = align_up(written);
    pad_to(&mut w, &mut written, end)?;
    w.flush()?;
    w.into_inner()
        .context("flushing REST store")?
        .sync_all()
        .context("fsync of REST store")?;

    debug_assert_eq!(written, header.total_bytes());
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(written as u64)
}

/// Reinterpret a `u32` slice as its native little-endian bytes.
fn as_bytes_u32(v: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and no invalid bit patterns, and u8 has
    // alignment 1, so any u32 slice is a valid u8 slice of 4x the length.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Reinterpret a `u64` slice as its native little-endian bytes.
fn as_bytes_u64(v: &[u64]) -> &[u8] {
    // SAFETY: as above, for u64.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::RestStore;

    #[test]
    fn store_size_matches_the_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rest");
        let tokens: Vec<u32> = (0..1000u32).map(|i| i % 37).collect();
        let n = write_store(&path, &tokens, &[0, 500], 0, 0x77).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), n);
        let store = RestStore::open(&path, Some(0x77)).unwrap();
        assert_eq!(store.header().total_bytes() as u64, n);
        assert_eq!(store.tokens(), &tokens[..]);
    }

    #[test]
    fn suffix_array_in_the_store_is_sorted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rest");
        let tokens: Vec<u32> = (0..500u32).map(|i| (i * 7919) % 11).collect();
        write_store(&path, &tokens, &[0], 0, 1).unwrap();
        let store = RestStore::open(&path, Some(1)).unwrap();
        let sa = store.suffix_array();
        assert_eq!(sa.len(), tokens.len());
        for w in sa.windows(2) {
            assert!(
                tokens[w[0] as usize..] < tokens[w[1] as usize..],
                "suffix array out of order at {w:?}"
            );
        }
    }

    #[test]
    fn rejects_mismatched_suffix_array_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rest");
        assert!(write_store_with_sa(&path, &[1, 2, 3], &[0, 1], &[0], 0, 1).is_err());
    }
}
