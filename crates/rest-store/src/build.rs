// SPDX-License-Identifier: AGPL-3.0-only

//! Offline store construction: walk a corpus, tokenize it, index it, write it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use tokenizers::Tokenizer;
use walkdir::WalkDir;

use crate::format::tokenizer_fingerprint;
use crate::jsonl::{Holdout, IngestStats, load_rows};
use crate::sa::build_suffix_array;
use crate::serialize::write_store_with_sa;

/// File extensions indexed by default.
///
/// Deliberately code-heavy: REST pays off where the target reproduces
/// verbatim spans, which is what code, configs, and structured prose do.
pub const DEFAULT_EXTENSIONS: &[&str] = &[
    "rs", "py", "sh", "toml", "md", "c", "h", "cc", "cpp", "hpp", "cu", "cuh", "go", "js", "ts",
    "tsx", "java", "kt", "rb", "yaml", "yml", "json", "sql", "proto",
];

/// Directory names never descended into.
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", "__pycache__", ".venv"];

/// Corpus selection knobs.
#[derive(Debug, Clone)]
pub struct CorpusOptions {
    /// Extensions to index, lowercase, without the dot.
    pub extensions: Vec<String>,
    /// Files larger than this are skipped — a single vendored blob can
    /// otherwise dominate the corpus and the suffix array.
    pub max_file_bytes: u64,
    /// Separator token id written between documents.
    pub sep_token: u32,
    /// JSONL generation files to ingest. Each usable row contributes one
    /// document: the assistant turn's content, verbatim.
    pub jsonl: Vec<PathBuf>,
    /// Deterministic holdout partition. Rows it selects are excluded from
    /// the store so `rest-store-eval --holdout-only` can score exactly
    /// those rows against a corpus that has never seen them.
    ///
    /// Applies to JSONL rows only — directory files have no stable row
    /// identity to partition on, and are always indexed in full.
    pub holdout: Holdout,
}

impl Default for CorpusOptions {
    fn default() -> Self {
        Self {
            extensions: DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect(),
            max_file_bytes: 4 * 1024 * 1024,
            sep_token: 0,
            jsonl: Vec::new(),
            holdout: Holdout::none(),
        }
    }
}

/// What a build produced.
#[derive(Debug, Clone)]
pub struct BuildStats {
    pub n_docs: usize,
    pub n_tokens: usize,
    pub corpus_bytes: u64,
    pub store_bytes: u64,
    pub tokenize_secs: f64,
    pub suffix_array_secs: f64,
    pub write_secs: f64,
    /// Documents contributed by directory roots.
    pub n_dir_docs: usize,
    /// Documents contributed by JSONL rows (after holdout exclusion).
    pub n_jsonl_docs: usize,
    /// JSONL rows withheld from the store by the holdout partition.
    pub n_holdout_excluded: usize,
    /// JSONL rows read but unusable, aggregated across all `--jsonl` files.
    pub jsonl_ingest: IngestStats,
}

/// Enumerate corpus files under `roots`, deterministically ordered.
///
/// Order matters: the store's token positions are only reproducible if the
/// document order is, and a reproducible store is what makes a fingerprint
/// mismatch mean something.
pub fn collect_files(roots: &[PathBuf], opts: &CorpusOptions) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for root in roots {
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                !e.file_type().is_dir()
                    || !e
                        .file_name()
                        .to_str()
                        .is_some_and(|n| SKIP_DIRS.contains(&n))
            })
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let ext = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase);
            let Some(ext) = ext else { continue };
            if !opts.extensions.contains(&ext) {
                continue;
            }
            if entry.metadata().map(|m| m.len()).unwrap_or(u64::MAX) > opts.max_file_bytes {
                continue;
            }
            files.push(entry.into_path());
        }
    }
    files.sort();
    files.dedup();
    files
}

/// Tokenize every document and concatenate into one stream separated by
/// `opts.sep_token`.
///
/// Documents come from two sources, in this order: the `files` walked from
/// the directory roots, then `extra_docs` (JSONL assistant turns). Order is
/// documented because it fixes token positions, and reproducible positions
/// are what make a store comparable across rebuilds.
///
/// Returns `(tokens, doc_starts, corpus_bytes)` where `doc_starts[i]` is
/// the token offset of document `i` (the separator belongs to the
/// *preceding* document, so a continuation that reaches a separator has
/// left the document it started in).
pub fn tokenize_corpus(
    files: &[PathBuf],
    extra_docs: &[String],
    tokenizer: &Tokenizer,
    opts: &CorpusOptions,
) -> Result<(Vec<u32>, Vec<u64>, u64)> {
    let encode = |text: &str, what: &dyn std::fmt::Display| -> (Vec<u32>, u64) {
        let bytes = text.len() as u64;
        match tokenizer.encode(text, false) {
            Ok(enc) => (enc.get_ids().to_vec(), bytes),
            Err(e) => {
                tracing::warn!(source = %what, error = %e, "tokenize failed; skipping");
                (Vec::new(), 0)
            }
        }
    };

    let mut per_doc: Vec<(Vec<u32>, u64)> = files
        .par_iter()
        .map(|path| -> (Vec<u32>, u64) {
            let Ok(text) = std::fs::read_to_string(path) else {
                // Binary or non-UTF8 file that slipped the extension
                // filter. Skipping is right: a mojibake decode would put
                // garbage token ids into the corpus.
                return (Vec::new(), 0);
            };
            encode(&text, &path.display())
        })
        .collect();

    per_doc.par_extend(
        extra_docs
            .par_iter()
            .enumerate()
            .map(|(i, text)| encode(text, &format_args!("jsonl row {i}"))),
    );

    let total: usize = per_doc.iter().map(|(t, _)| t.len() + 1).sum();
    if total > u32::MAX as usize {
        bail!(
            "corpus of {total} tokens exceeds the u32 suffix-array index limit; \
             narrow --ext or lower --max-file-bytes"
        );
    }

    let mut tokens: Vec<u32> = Vec::with_capacity(total);
    let mut doc_starts: Vec<u64> = Vec::with_capacity(per_doc.len());
    let mut corpus_bytes = 0u64;
    for (ids, bytes) in &per_doc {
        if ids.is_empty() {
            continue;
        }
        doc_starts.push(tokens.len() as u64);
        tokens.extend_from_slice(ids);
        tokens.push(opts.sep_token);
        corpus_bytes += bytes;
    }
    Ok((tokens, doc_starts, corpus_bytes))
}

/// End-to-end build: walk `roots`, tokenize with the tokenizer at
/// `tokenizer_json`, index, and write the store to `out`.
pub fn build_store(
    roots: &[PathBuf],
    tokenizer_json: &Path,
    out: &Path,
    opts: &CorpusOptions,
) -> Result<BuildStats> {
    let tok_bytes = std::fs::read(tokenizer_json)
        .with_context(|| format!("reading tokenizer {}", tokenizer_json.display()))?;
    let fp = tokenizer_fingerprint(&tok_bytes);
    let tokenizer = Tokenizer::from_file(tokenizer_json)
        .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", tokenizer_json.display()))?;

    let files = collect_files(roots, opts);

    // JSONL ingest. Each usable row contributes one document: the
    // assistant turn's content, verbatim. Rows the holdout partition
    // selects are excluded here and nowhere else, so a store can never
    // contain a row that `--holdout-only` will later score against it.
    let mut jsonl_docs: Vec<String> = Vec::new();
    let mut ingest = IngestStats::default();
    let mut n_holdout_excluded = 0usize;
    for path in &opts.jsonl {
        let (rows, stats) = load_rows(path)?;
        ingest.kept += stats.kept;
        ingest.malformed += stats.malformed;
        ingest.no_conversations += stats.no_conversations;
        ingest.no_assistant_turn += stats.no_assistant_turn;
        ingest.empty_assistant += stats.empty_assistant;
        let (store_rows, held) = opts.holdout.split(rows);
        n_holdout_excluded += held.len();
        jsonl_docs.extend(store_rows.into_iter().map(|r| r.completion));
    }
    if opts.holdout.is_active() {
        tracing::info!(
            frac = opts.holdout.frac,
            seed = opts.holdout.seed,
            excluded = n_holdout_excluded,
            indexed = jsonl_docs.len(),
            "holdout partition applied to JSONL rows"
        );
    }

    if files.is_empty() && jsonl_docs.is_empty() {
        bail!(
            "no corpus documents: no files matched under {:?} with extensions {:?}, \
             and no usable rows came from {:?}",
            roots,
            opts.extensions,
            opts.jsonl
        );
    }
    tracing::info!(
        files = files.len(),
        jsonl_docs = jsonl_docs.len(),
        tokenizer_fp = format_args!("{fp:#018x}"),
        "tokenizing corpus"
    );

    let t0 = std::time::Instant::now();
    let (tokens, doc_starts, corpus_bytes) =
        tokenize_corpus(&files, &jsonl_docs, &tokenizer, opts)?;
    let tokenize_secs = t0.elapsed().as_secs_f64();
    if tokens.is_empty() {
        bail!("corpus tokenized to zero tokens");
    }
    tracing::info!(
        n_tokens = tokens.len(),
        n_docs = doc_starts.len(),
        secs = tokenize_secs,
        "tokenized"
    );

    let t1 = std::time::Instant::now();
    let sa = build_suffix_array(&tokens);
    let suffix_array_secs = t1.elapsed().as_secs_f64();
    tracing::info!(secs = suffix_array_secs, "suffix array built");

    let t2 = std::time::Instant::now();
    let store_bytes = write_store_with_sa(out, &tokens, &sa, &doc_starts, opts.sep_token, fp)?;
    let write_secs = t2.elapsed().as_secs_f64();

    // Empty documents are dropped during assembly, so recover the split
    // from the JSONL side (which is exact) rather than assuming both.
    let n_docs = doc_starts.len();
    let n_jsonl_docs = jsonl_docs.len().min(n_docs);
    Ok(BuildStats {
        n_docs,
        n_tokens: tokens.len(),
        corpus_bytes,
        store_bytes,
        tokenize_secs,
        suffix_array_secs,
        write_secs,
        n_dir_docs: n_docs - n_jsonl_docs,
        n_jsonl_docs,
        n_holdout_excluded,
        jsonl_ingest: ingest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::RestStore;

    /// A tokenizer is needed for the end-to-end build tests; use the
    /// target's if present, otherwise skip rather than fail on a machine
    /// that has no model checkout.
    fn target_tokenizer() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from("/home/flocka/atlas/qwen38/normal-qwen/tokenizer.json");
        p.exists().then_some(p)
    }

    fn jsonl_line(id: &str, assistant: &str) -> String {
        serde_json::json!({
            "id": id,
            "conversations": [
                {"role": "user", "content": "prompt text"},
                {"role": "assistant", "content": assistant},
            ],
        })
        .to_string()
    }

    #[test]
    fn builds_from_dirs_and_jsonl_together() {
        let Some(tok) = target_tokenizer() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn alpha() { let x = 1; }").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn beta() { let y = 2; }").unwrap();
        let jsonl = dir.path().join("gen.jsonl");
        std::fs::write(
            &jsonl,
            format!(
                "{}\n{}\n{}\n",
                jsonl_line("g1", "<think>plan</think>def one(): return 1"),
                jsonl_line("g2", "def two(): return 2"),
                jsonl_line("g3", "def three(): return 3"),
            ),
        )
        .unwrap();

        let out = dir.path().join("mixed.rest");
        let opts = CorpusOptions {
            jsonl: vec![jsonl],
            ..Default::default()
        };
        let stats = build_store(&[dir.path().to_path_buf()], &tok, &out, &opts).unwrap();

        assert_eq!(stats.n_dir_docs, 2, "both .rs files should be indexed");
        assert_eq!(stats.n_jsonl_docs, 3, "all three rows should be indexed");
        assert_eq!(stats.n_docs, 5);
        assert_eq!(stats.n_holdout_excluded, 0);
        assert_eq!(stats.jsonl_ingest.kept, 3);

        // Content from BOTH sources must be retrievable from one store.
        // Needles are WHOLE documents: a substring can re-encode to
        // different ids than it carries inside the document, because the
        // tokenizer merges across the cut (e.g. `)` + `:` -> `):`).
        let store = RestStore::open(&out, None).unwrap();
        let tokenizer = tokenizers::Tokenizer::from_file(&tok).unwrap();
        for needle in [
            "fn alpha() { let x = 1; }",
            "<think>plan</think>def one(): return 1",
        ] {
            let ids = tokenizer.encode(needle, false).unwrap().get_ids().to_vec();
            assert!(
                !crate::sa::prefix_range(store.tokens(), store.suffix_array(), &ids).is_empty(),
                "{needle:?} missing from the mixed store"
            );
        }
    }

    #[test]
    fn holdout_excludes_rows_from_the_store() {
        let Some(tok) = target_tokenizer() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("gen.jsonl");
        let body: String = (0..200)
            .map(|i| jsonl_line(&format!("row{i}"), &format!("fn unique_marker_{i}() {{}}")) + "\n")
            .collect();
        std::fs::write(&jsonl, body).unwrap();

        let holdout = Holdout {
            frac: 0.25,
            seed: 11,
        };
        let out = dir.path().join("h.rest");
        let opts = CorpusOptions {
            jsonl: vec![jsonl.clone()],
            holdout,
            ..Default::default()
        };
        let stats = build_store(&[], &tok, &out, &opts).unwrap();
        assert!(stats.n_holdout_excluded > 0);
        assert_eq!(stats.n_holdout_excluded + stats.n_jsonl_docs, 200);
        assert_eq!(stats.n_dir_docs, 0);

        // The decisive property: every held-out row's marker must be ABSENT
        // from the store, and every indexed row's marker present.
        let store = RestStore::open(&out, None).unwrap();
        let tokenizer = tokenizers::Tokenizer::from_file(&tok).unwrap();
        let (rows, _) = crate::jsonl::load_rows(&jsonl).unwrap();
        let (indexed, held) = holdout.split(rows);
        let occurs = |text: &str| {
            let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
            !crate::sa::prefix_range(store.tokens(), store.suffix_array(), &ids).is_empty()
        };
        for r in held.iter().take(20) {
            assert!(!occurs(&r.completion), "held-out row leaked into the store");
        }
        for r in indexed.iter().take(20) {
            assert!(occurs(&r.completion), "indexed row missing from the store");
        }
    }

    #[test]
    fn build_with_no_sources_is_an_error() {
        let Some(tok) = target_tokenizer() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("empty.rest");
        let err = build_store(&[], &tok, &out, &CorpusOptions::default()).unwrap_err();
        assert!(
            err.to_string().contains("no corpus documents"),
            "got: {err}"
        );
    }

    #[test]
    fn collect_files_is_deterministic_and_filters() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        std::fs::write(dir.path().join("b.bin"), "nope").unwrap();
        std::fs::create_dir(dir.path().join("target")).unwrap();
        std::fs::write(dir.path().join("target/c.rs"), "fn c() {}").unwrap();
        let opts = CorpusOptions::default();
        let roots = vec![dir.path().to_path_buf()];
        let files = collect_files(&roots, &opts);
        assert_eq!(files.len(), 1, "got {files:?}");
        assert!(files[0].ends_with("a.rs"));
        assert_eq!(files, collect_files(&roots, &opts));
    }
}
