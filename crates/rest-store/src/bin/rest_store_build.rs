// SPDX-License-Identifier: AGPL-3.0-only

//! `rest-store-build` — build a REST draft store from a corpus of source files.
//!
//! ```text
//! rest-store-build \
//!   --tokenizer /path/to/normal-qwen/tokenizer.json \
//!   --out /path/to/code.rest \
//!   /path/to/corpus/root ...
//! ```

#![deny(warnings)]
#![deny(clippy::all)]

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use rest_store::Holdout;
use rest_store::build::{CorpusOptions, DEFAULT_EXTENSIONS, build_store};

#[derive(Parser, Debug)]
#[command(
    name = "rest-store-build",
    about = "Build a REST-style retrieval draft store from a source corpus"
)]
struct Args {
    /// Corpus roots to walk. Directories are searched recursively.
    ///
    /// Optional when `--jsonl` is given; at least one source is required.
    roots: Vec<PathBuf>,

    /// JSONL generation files to ingest. Each usable row contributes one
    /// document: the assistant turn's content, verbatim (including any
    /// `<think>` wrapper — that text is part of the emitted stream).
    ///
    /// May be combined with directory roots. Precedence: directory files
    /// are indexed first, then JSONL rows; both land in the same store and
    /// neither shadows the other.
    #[arg(long)]
    jsonl: Vec<PathBuf>,

    /// Fraction of JSONL rows to EXCLUDE from the store, so that
    /// `rest-store-eval --holdout-only` can score them against a corpus
    /// that has never seen them. Directory files are never held out.
    #[arg(long, default_value_t = 0.0)]
    holdout_frac: f64,

    /// Seed for the holdout partition. The eval must be given the same
    /// `--holdout-frac` and `--holdout-seed` or the split will not match.
    #[arg(long, default_value_t = 0)]
    holdout_seed: u64,

    /// Path to the TARGET model's `tokenizer.json`.
    #[arg(long)]
    tokenizer: PathBuf,

    /// Output store path.
    #[arg(long)]
    out: PathBuf,

    /// Comma-separated file extensions to index (default: a code-heavy set).
    #[arg(long, value_delimiter = ',')]
    ext: Option<Vec<String>>,

    /// Skip files larger than this many bytes.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    max_file_bytes: u64,

    /// Token id written between documents. Continuations never cross it.
    #[arg(long, default_value_t = 0)]
    sep_token: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    if args.roots.is_empty() && args.jsonl.is_empty() {
        anyhow::bail!("pass at least one corpus root or --jsonl <path>");
    }
    if !(0.0..=1.0).contains(&args.holdout_frac) {
        anyhow::bail!(
            "--holdout-frac must be in [0.0, 1.0], got {}",
            args.holdout_frac
        );
    }
    if args.holdout_frac > 0.0 && args.jsonl.is_empty() {
        anyhow::bail!("--holdout-frac only applies to --jsonl rows, but no --jsonl was given");
    }
    let opts = CorpusOptions {
        extensions: args.ext.unwrap_or_else(|| {
            DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect()
        }),
        max_file_bytes: args.max_file_bytes,
        sep_token: args.sep_token,
        jsonl: args.jsonl,
        holdout: Holdout {
            frac: args.holdout_frac,
            seed: args.holdout_seed,
        },
    };

    let t0 = std::time::Instant::now();
    let stats = build_store(&args.roots, &args.tokenizer, &args.out, &opts)?;
    let total = t0.elapsed().as_secs_f64();

    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!("REST store written to {}", args.out.display());
    println!(
        "  documents        {}  ({} from dirs, {} from jsonl)",
        stats.n_docs, stats.n_dir_docs, stats.n_jsonl_docs
    );
    if stats.jsonl_ingest.total() > 0 {
        let d = &stats.jsonl_ingest;
        println!(
            "  jsonl rows       {} kept, {} dropped (malformed {}, no-conversations {}, \
             no-assistant {}, empty-assistant {})",
            d.kept,
            d.total() - d.kept,
            d.malformed,
            d.no_conversations,
            d.no_assistant_turn,
            d.empty_assistant
        );
    }
    if stats.n_holdout_excluded > 0 {
        println!(
            "  holdout          {} rows EXCLUDED (frac={}, seed={}) — score them with \
             `rest-store-eval --holdout-only --holdout-frac {} --holdout-seed {}`",
            stats.n_holdout_excluded,
            args.holdout_frac,
            args.holdout_seed,
            args.holdout_frac,
            args.holdout_seed
        );
    }
    println!(
        "  corpus text      {:.1} MiB",
        mib(stats.corpus_bytes)
    );
    println!(
        "  tokens           {} ({:.2} bytes/token)",
        stats.n_tokens,
        stats.corpus_bytes as f64 / stats.n_tokens as f64
    );
    println!(
        "  store size       {:.1} MiB ({:.1} bytes/token)",
        mib(stats.store_bytes),
        stats.store_bytes as f64 / stats.n_tokens as f64
    );
    println!("  tokenize         {:.2} s", stats.tokenize_secs);
    println!("  suffix array     {:.2} s", stats.suffix_array_secs);
    println!("  write            {:.2} s", stats.write_secs);
    println!("  total            {total:.2} s");
    Ok(())
}
