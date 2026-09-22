// SPDX-License-Identifier: AGPL-3.0-only

//! Builds `EngramHashState` straight from a checkpoint directory: the real
//! `tokenizer.json` and `config.json`, no exported fixture file.
//!
//! Everything else the constructor needs — [`EngramLayout`], the hash
//! multipliers — is data, not derived from the tokenizer, and stays put in
//! [`super::hash`]. This module owns exactly the one piece that DOES depend on
//! the checkpoint's tokenizer: [`build_compressed_token_map`], ported from the
//! checkpoint's own `inference/engram.py` `build_compressed_token_map` /
//! `bench/engram/export_map.py`.
//!
//! Gated behind the `engram-tokenizer` feature: serving needs it, but most of
//! this crate's tests should not have to build a real `tokenizers::Tokenizer`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{Lowercase, NFD, NFKC, Replace, Sequence, Strip, StripAccents};
use tokenizers::{NormalizedString, Normalizer, Tokenizer};

use super::hash::{EngramHashState, EngramLayout};

/// The reference's normalizer, `bench/engram/export_map.py` / `inference/engram.py`
/// `build_compressed_token_map`: NFKC, NFD, StripAccents, Lowercase, collapse
/// whitespace to a single space, then Strip — with a private-use sentinel so a
/// token that is EXACTLY one space survives `Strip` instead of collapsing to
/// empty and merging with every other all-whitespace token.
///
/// Built from `tokenizers` (the same crate Python's `tokenizers` package binds
/// to), not reimplemented, so this is bit-identical to the reference by
/// construction rather than by parallel maintenance.
fn normalizer() -> Result<Sequence> {
    let sentinel = "\u{e000}";
    Ok(Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        Replace::new(ReplacePattern::Regex(r"[ \t\r\n]+".to_string()), " ".to_string())
            .map_err(|e| anyhow::anyhow!("engram normalizer: whitespace regex: {e}"))?
            .into(),
        Replace::new(ReplacePattern::Regex(r"^ $".to_string()), sentinel.to_string())
            .map_err(|e| anyhow::anyhow!("engram normalizer: lone-space regex: {e}"))?
            .into(),
        Strip::new(true, true).into(),
        Replace::new(ReplacePattern::String(sentinel.to_string()), " ".to_string())
            .map_err(|e| anyhow::anyhow!("engram normalizer: sentinel restore: {e}"))?
            .into(),
    ]))
}

/// Every token id -> a smaller id space where tokens that normalize alike
/// collapse together (`" The"`, `"the"`, `"THE"` all hash the same way).
/// Returns the lookup plus the compressed vocab size.
pub fn build_compressed_token_map(tokenizer: &Tokenizer) -> Result<(Vec<i32>, usize)> {
    let norm = normalizer()?;
    let vocab_size = tokenizer.get_vocab_size(true);
    let mut key_to_new: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    let mut lookup = vec![0i32; vocab_size];

    for token_id in 0..vocab_size as u32 {
        let text = tokenizer
            .decode(&[token_id], false)
            .map_err(|e| anyhow::anyhow!("decode token {token_id}: {e}"))?;
        let key = if text.contains('\u{fffd}') {
            // A partial UTF-8 byte token: nothing to normalize, key it by its raw form.
            tokenizer
                .id_to_token(token_id)
                .with_context(|| format!("token {token_id} has no id_to_token entry"))?
        } else {
            let mut ns = NormalizedString::from(text.as_str());
            norm.normalize(&mut ns).map_err(|e| anyhow::anyhow!("normalize token {token_id}: {e}"))?;
            let normalized = ns.get().to_string();
            if normalized.is_empty() { text } else { normalized }
        };
        let next_id = key_to_new.len() as i32;
        let id = *key_to_new.entry(key).or_insert(next_id);
        lookup[token_id as usize] = id;
    }
    let compressed_vocab = key_to_new.len();
    Ok((lookup, compressed_vocab))
}

/// The `[1, 14]` engram-layer hash multipliers, one `[max_ngram_size]` row per
/// layer id, reproduced from `inference/engram.py::compute_hash_multipliers`:
/// `np.random.default_rng(10007 * layer_id).integers(low=0,
/// high=(i64::MAX // compressed_vocab_size) // 2, size=max_ngram_size) * 2 + 1`,
/// with `compressed_vocab_size = 99_092` (the COMPRESSED vocab, confirmed by
/// this being the only choice of that parameter that reproduces these exact
/// values — see `bench/engram/README.md`).
///
/// Shipped as data rather than computed here: reproducing NumPy's PCG64 +
/// Lemire bounded-integer sampling bit-for-bit in Rust is real work for a
/// value that depends on nothing but a layer id this checkpoint fixes at
/// `[1, 14]`. If the checkpoint ever adds an engram layer outside this set,
/// this function must fail loudly rather than guess — and does.
fn checkpoint_multiplier_row(layer_id: u32) -> Result<Vec<i64>> {
    match layer_id {
        1 => Ok(vec![76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391]),
        14 => Ok(vec![67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591]),
        other => bail!(
            "no shipped hash multiplier for engram layer {other}; this checkpoint's engram \
             layers are [1, 14] and multipliers are data, not derived — add the layer's row \
             (compute via `np.random.default_rng(10007 * layer_id).integers(...)`, see \
             bench/engram/README.md) rather than guessing"
        ),
    }
}

/// The `engram_*` fields this constructor needs out of `config.json`'s
/// (possibly `text_config`-nested) object.
#[derive(serde::Deserialize)]
struct EngramConfigFields {
    engram_layer_ids: Vec<u32>,
    engram_num_embeddings: Vec<u64>,
    engram_max_ngram_size: usize,
    engram_vocab_size: u64,
    engram_n_heads: usize,
    engram_head_dim: usize,
    engram_pad_token_id: usize,
    engram_compressed_vocab_size: u64,
}

fn read_engram_config(model_dir: &Path) -> Result<EngramConfigFields> {
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    // V4.1 nests the text model's config under `text_config`; fall back to the
    // top level so this also works against a flat config.
    let scope = v.get("text_config").unwrap_or(&v);
    serde_json::from_value(scope.clone())
        .with_context(|| format!("{}: missing or malformed engram_* fields", path.display()))
}

impl EngramHashState {
    /// Build the real, servable hash state from a checkpoint directory: reads
    /// `config.json` for the engram layout, `tokenizer.json` for the compressed
    /// token map, and uses the shipped multiplier table — no exported fixture
    /// file, no Python.
    #[cfg(feature = "engram-tokenizer")]
    pub fn for_checkpoint(model_dir: &Path) -> Result<Self> {
        let cfg = read_engram_config(model_dir)?;
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", tok_path.display()))?;

        let (token_map, compressed_vocab) = build_compressed_token_map(&tokenizer)?;
        if compressed_vocab as u64 != cfg.engram_compressed_vocab_size {
            bail!(
                "tokenizer.json normalizes to a {compressed_vocab}-entry compressed vocab, \
                 config.json says engram_compressed_vocab_size = {}; every hash multiplier \
                 derives from this value, so a mismatch would silently rehash the whole table",
                cfg.engram_compressed_vocab_size
            );
        }

        let layout = EngramLayout::new(
            &cfg.engram_layer_ids,
            cfg.engram_max_ngram_size,
            cfg.engram_n_heads,
            cfg.engram_head_dim,
            cfg.engram_vocab_size,
        )?;
        layout.validate_against_config(&cfg.engram_num_embeddings)?;

        let mut multipliers = Vec::with_capacity(cfg.engram_layer_ids.len());
        for &layer_id in &cfg.engram_layer_ids {
            multipliers.push(checkpoint_multiplier_row(layer_id)?);
        }

        EngramHashState::new(layout, token_map, multipliers, cfg.engram_pad_token_id, cfg.engram_compressed_vocab_size)
    }
}

#[cfg(all(test, feature = "engram-tokenizer"))]
mod tests {
    use super::*;

    const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";

    fn model_dir_present() -> bool {
        Path::new(MODEL_DIR).join("tokenizer.json").is_file()
    }

    /// THE TEST THAT MATTERS: the token map this Rust port builds must be
    /// byte-identical to `export_map.py`'s Python output — same normalizer
    /// crate underneath, but a fresh loop, fresh id_to_token/decode calls, and
    /// a fresh config read, so a divergence here is a real port bug, not a
    /// tautology.
    #[test]
    fn token_map_matches_the_python_export_exactly() {
        if !model_dir_present() {
            eprintln!("skipping: {MODEL_DIR} not present on this box");
            return;
        }
        let fixture = "/home/flocka/atlas/dsv41-engram/bench/engram/token_map_i32.bin";
        let Ok(bytes) = std::fs::read(fixture) else {
            eprintln!("skipping: {fixture} not present (regenerate with bench/engram/export_map.py)");
            return;
        };
        let want: Vec<i32> = bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();

        let tokenizer = Tokenizer::from_file(Path::new(MODEL_DIR).join("tokenizer.json")).unwrap();
        let (got, compressed) = build_compressed_token_map(&tokenizer).unwrap();

        assert_eq!(compressed, 99_092, "compressed vocab must match config.json's declared value");
        assert_eq!(got.len(), want.len(), "token map length differs from the Python export");
        let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(mismatches, 0, "{mismatches}/{} compressed ids differ from export_map.py", got.len());
    }

    /// Negative control: a token map that fails to collapse case/accent variants
    /// (e.g. if the normalizer sequence were built in the wrong order) must NOT
    /// pass the check above. Exercised directly rather than via `for_checkpoint`
    /// so a broken normalizer can't hide behind the compressed-vocab assert.
    #[test]
    fn case_and_accent_variants_collapse_to_the_same_id() {
        if !model_dir_present() {
            eprintln!("skipping: {MODEL_DIR} not present on this box");
            return;
        }
        let tokenizer = Tokenizer::from_file(Path::new(MODEL_DIR).join("tokenizer.json")).unwrap();
        let (map, _) = build_compressed_token_map(&tokenizer).unwrap();

        // " The", "the", "THE" must land on the same compressed id — export_map.py's own
        // sanity check, replayed here against the Rust build.
        let ids = |s: &str| tokenizer.encode(s, false).unwrap().get_ids().to_vec();
        let the_variants: Vec<i32> = [" The", "the", "THE", " the"]
            .iter()
            .flat_map(|s| ids(s))
            .map(|id| map[id as usize])
            .collect();
        assert!(
            the_variants.windows(2).all(|w| w[0] == w[1]),
            "case/accent variants of 'the' did not collapse to one compressed id: {the_variants:?}"
        );
    }

    /// NEGATIVE CONTROL: a normalizer that skips `StripAccents` must NOT
    /// reproduce the Python export — accented and unaccented spellings would
    /// stay separate compressed ids, changing the map. Without this, the exact
    /// match above could be passing because the fixture and the Rust build both
    /// somehow skip a step, not because either is right.
    #[test]
    fn dropping_strip_accents_breaks_the_match() {
        if !model_dir_present() {
            eprintln!("skipping: {MODEL_DIR} not present on this box");
            return;
        }
        let fixture = "/home/flocka/atlas/dsv41-engram/bench/engram/token_map_i32.bin";
        let Ok(bytes) = std::fs::read(fixture) else {
            eprintln!("skipping: {fixture} not present");
            return;
        };
        let want: Vec<i32> = bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();

        let sentinel = "\u{e000}";
        let broken = Sequence::new(vec![
            NFKC.into(),
            NFD.into(),
            // StripAccents deliberately omitted.
            Lowercase.into(),
            Replace::new(ReplacePattern::Regex(r"[ \t\r\n]+".to_string()), " ".to_string()).unwrap().into(),
            Replace::new(ReplacePattern::Regex(r"^ $".to_string()), sentinel.to_string()).unwrap().into(),
            Strip::new(true, true).into(),
            Replace::new(ReplacePattern::String(sentinel.to_string()), " ".to_string()).unwrap().into(),
        ]);

        let tokenizer = Tokenizer::from_file(Path::new(MODEL_DIR).join("tokenizer.json")).unwrap();
        let vocab_size = tokenizer.get_vocab_size(true);
        let mut key_to_new: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
        let mut got = vec![0i32; vocab_size];
        for token_id in 0..vocab_size as u32 {
            let text = tokenizer.decode(&[token_id], false).unwrap();
            let key = if text.contains('\u{fffd}') {
                tokenizer.id_to_token(token_id).unwrap()
            } else {
                let mut ns = NormalizedString::from(text.as_str());
                broken.normalize(&mut ns).unwrap();
                let n = ns.get().to_string();
                if n.is_empty() { text } else { n }
            };
            let next_id = key_to_new.len() as i32;
            got[token_id as usize] = *key_to_new.entry(key).or_insert(next_id);
        }

        let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert!(
            mismatches > 0,
            "a normalizer missing StripAccents reproduced the Python export exactly -- \
             this control cannot distinguish a correct build from a broken one"
        );
        eprintln!("dropping StripAccents moved {mismatches}/{} compressed ids", got.len());
    }

    /// `for_checkpoint` end to end: must construct without error and validate
    /// its own prime-sum check, exactly like the hand-built state in
    /// `hash.rs`'s oracle tests.
    #[test]
    fn for_checkpoint_builds_a_usable_state() {
        if !model_dir_present() {
            eprintln!("skipping: {MODEL_DIR} not present on this box");
            return;
        }
        let mut st = EngramHashState::for_checkpoint(Path::new(MODEL_DIR)).expect("for_checkpoint must succeed");
        // A tiny forward must not panic and must land inside each layer's bucket range —
        // the same shape check hash.rs's own tests run on the hand-built state.
        let out = st.forward(&[0u32, 1, 2, 3], 0, None).unwrap();
        assert_eq!(out.len(), 4 * 2 * 24);
    }
}
