// SPDX-License-Identifier: AGPL-3.0-only

//! `rest-store-eval` — replay real generations against a store and measure
//! what the drafter *would* have contributed.
//!
//! No GPU and no model: every generation in the JSONL is replayed token by
//! token, and at each step the store is asked what it would have drafted.
//! The draft is then scored against the tokens the target actually emitted,
//! which is exactly what tree-verify would accept. That upper-bounds the
//! speedup without running the target once.
//!
//! ```text
//! rest-store-eval --store code.rest --tokenizer tokenizer.json \
//!                 --jsonl regen_qwen38.jsonl
//! ```

#![deny(warnings)]
#![deny(clippy::all)]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use rest_store::{
    DEFAULT_MAX_K, DEFAULT_MAX_NODES, DEFAULT_MAX_OCCURRENCES, DEFAULT_MIN_MATCH, Holdout,
    RestStore, format::tokenizer_fingerprint, load_rows,
};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(
    name = "rest-store-eval",
    about = "Replay generations against a REST draft store"
)]
struct Args {
    /// Store built by `rest-store-build`.
    #[arg(long)]
    store: PathBuf,

    /// The TARGET model's `tokenizer.json` — must be the one the store was built with.
    #[arg(long)]
    tokenizer: PathBuf,

    /// JSONL of real generations (`conversations: [{role, content}]`).
    #[arg(long)]
    jsonl: Option<PathBuf>,

    /// Fallback corpus: score these files as pseudo-generations. Only use
    /// files that are NOT in the store, or the numbers are self-matches.
    #[arg(long)]
    files: Vec<PathBuf>,

    /// Cap on rows read from the JSONL.
    #[arg(long)]
    limit: Option<usize>,

    /// Score ONLY the rows that `rest-store-build --holdout-frac` excluded
    /// from the store. Pass the same `--holdout-frac` and `--holdout-seed`
    /// the build used; the partition is derived from the row's `id`, so
    /// both tools select the identical set.
    ///
    /// This is how a single JSONL file yields a decontaminated in-domain
    /// measurement: the scored rows are, by construction, absent from the
    /// corpus.
    #[arg(long)]
    holdout_only: bool,

    /// Holdout fraction — must match the value passed to the build.
    #[arg(long, default_value_t = 0.0)]
    holdout_frac: f64,

    /// Holdout seed — must match the value passed to the build.
    #[arg(long, default_value_t = 0)]
    holdout_seed: u64,

    /// Longest context suffix considered.
    #[arg(long, default_value_t = DEFAULT_MAX_K)]
    max_k: usize,

    /// Engage threshold: no draft below this match length.
    #[arg(long, default_value_t = DEFAULT_MIN_MATCH)]
    min_match: usize,

    /// Cap on suffix-array occurrences scanned per lookup.
    #[arg(long, default_value_t = DEFAULT_MAX_OCCURRENCES)]
    max_occurrences: usize,

    /// Continuation depth.
    #[arg(long, default_value_t = 16)]
    depth: usize,

    /// Node budget for the proposed tree.
    #[arg(long, default_value_t = DEFAULT_MAX_NODES)]
    max_nodes: usize,

    /// Exclude boilerplate token positions from scoring (`--files` only).
    ///
    /// Held-out-file replay flatters a repo-source store, because a
    /// held-out file's SPDX header, `use` preamble and `#[derive]` lines
    /// are near-verbatim copies of the files still in the corpus. With
    /// this flag those steps still form CONTEXT — a real generation
    /// emits them — but they no longer count as decode steps, so the
    /// reported figure describes only the novel body of each file.
    #[arg(long)]
    strip_boilerplate: bool,
}

/// One replayable stream: the full token context plus where scoring starts.
struct Sample {
    tokens: Vec<u32>,
    score_from: usize,
    /// Per-token flags; `true` means the position is context but not a
    /// scored decode step. Empty means score everything.
    skip: Vec<bool>,
}

/// Load replayable samples from a JSONL file.
///
/// Row extraction goes through `rest_store::jsonl`, the same function the
/// builder uses, so the two tools cannot disagree about which text is the
/// assistant's output.
///
/// When `holdout.is_active()`, only rows the partition selects are
/// returned — exactly the rows `rest-store-build --holdout-frac` left out
/// of the store.
fn load_jsonl(
    path: &Path,
    tok: &Tokenizer,
    limit: Option<usize>,
    holdout: Holdout,
    holdout_only: bool,
) -> Result<Vec<Sample>> {
    let (rows, stats) = load_rows(path)?;
    let rows: Vec<_> = if holdout_only {
        rows.into_iter().filter(|r| holdout.contains(r)).collect()
    } else {
        rows
    };
    if holdout_only {
        println!(
            "  holdout-only     scoring {} of {} rows (frac={}, seed={})",
            rows.len(),
            stats.kept,
            holdout.frac,
            holdout.seed
        );
    }

    let mut out = Vec::new();
    for row in rows {
        if limit.is_some_and(|l| out.len() >= l) {
            break;
        }
        // Prompt tokens form context but are not scored: only the
        // assistant's own output is a decode step.
        let enc_prompt = tok
            .encode(row.prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?;
        let enc_completion = tok
            .encode(row.completion.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizing completion: {e}"))?;
        let score_from = enc_prompt.get_ids().len();
        let mut tokens = enc_prompt.get_ids().to_vec();
        tokens.extend_from_slice(enc_completion.get_ids());
        if tokens.len() > score_from + 1 {
            out.push(Sample {
                tokens,
                score_from,
                skip: Vec::new(),
            });
        }
    }
    Ok(out)
}

fn load_files(paths: &[PathBuf], tok: &Tokenizer, strip: bool) -> Result<Vec<Sample>> {
    let mut out = Vec::new();
    for p in paths {
        let text =
            std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        let enc = tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizing {}: {e}", p.display()))?;
        let tokens = enc.get_ids().to_vec();
        let skip = if strip {
            let spans = rest_store::boilerplate::boilerplate_spans(&text);
            rest_store::boilerplate::token_skip_mask(enc.get_offsets(), &spans)
        } else {
            Vec::new()
        };
        if tokens.len() > 2 {
            out.push(Sample {
                tokens,
                score_from: 1,
                skip,
            });
        }
    }
    Ok(out)
}

#[derive(Default)]
struct Metrics {
    steps: u64,
    engaged: u64,
    /// Accepted lookahead summed over engaged steps, tree verify.
    tree_accepted: u64,
    /// Same, but if only the highest-count chain were drafted (flat verify).
    flat_accepted: u64,
    /// Engaged steps where the draft contributed nothing.
    wasted: u64,
    /// Nodes proposed, summed over engaged steps.
    nodes: u64,
    /// Lookup latency samples, nanoseconds.
    lat_ns: Vec<u32>,
    /// Count of steps per match length (index = match_len, 0 = no match).
    match_len_hist: Vec<u64>,
}

impl Metrics {
    fn percentile(sorted: &[u32], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[idx] as f64 / 1000.0
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let tok_bytes = std::fs::read(&args.tokenizer)
        .with_context(|| format!("reading {}", args.tokenizer.display()))?;
    let fp = tokenizer_fingerprint(&tok_bytes);
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let store = RestStore::open(&args.store, Some(fp))?;

    let holdout = Holdout {
        frac: args.holdout_frac,
        seed: args.holdout_seed,
    };
    if args.holdout_only && !holdout.is_active() {
        bail!("--holdout-only requires --holdout-frac > 0 (matching the build)");
    }
    if args.strip_boilerplate && args.jsonl.is_some() {
        bail!("--strip-boilerplate applies to --files (source replay), not --jsonl");
    }
    let samples = match (&args.jsonl, args.files.is_empty()) {
        (Some(p), _) => load_jsonl(p, &tokenizer, args.limit, holdout, args.holdout_only)?,
        (None, false) => {
            if args.holdout_only {
                bail!("--holdout-only applies to --jsonl input, not --files");
            }
            load_files(&args.files, &tokenizer, args.strip_boilerplate)?
        }
        (None, true) => bail!("pass --jsonl or --files"),
    };
    if samples.is_empty() {
        bail!("no replayable samples found");
    }

    let mut m = Metrics {
        match_len_hist: vec![0; args.max_k + 1],
        ..Default::default()
    };
    let total_steps: usize = samples
        .iter()
        .map(|s| s.tokens.len().saturating_sub(s.score_from))
        .sum();
    m.lat_ns.reserve(total_steps);

    let mut n_skipped = 0u64;
    for sample in &samples {
        for t in sample.score_from..sample.tokens.len() {
            // Boilerplate positions remain in `ctx` — the model emits
            // them — but are not themselves scored.
            if sample.skip.get(t).copied().unwrap_or(false) {
                n_skipped += 1;
                continue;
            }
            let ctx = &sample.tokens[..t];
            let actual = &sample.tokens[t..];
            m.steps += 1;

            let t0 = std::time::Instant::now();
            let hit = store.longest_suffix_match(ctx, args.max_k, args.max_occurrences);
            let elapsed = t0.elapsed().as_nanos().min(u32::MAX as u128) as u32;
            m.lat_ns.push(elapsed);

            let match_len = hit.as_ref().map(|h| h.match_len).unwrap_or(0);
            m.match_len_hist[match_len.min(args.max_k)] += 1;

            let Some(hit) = hit else { continue };
            if hit.match_len < args.min_match {
                continue;
            }
            let Some(tree) = rest_store::build_draft_trie(
                store.tokens(),
                &hit.positions,
                hit.match_len,
                rest_store::TrieParams {
                    depth: args.depth,
                    max_nodes: args.max_nodes,
                    sep_token: store.header().sep_token,
                },
            ) else {
                continue;
            };

            m.engaged += 1;
            m.nodes += tree.len() as u64;
            let tree_hit = tree.longest_accepted_path(actual);
            m.tree_accepted += tree_hit as u64;
            if tree_hit == 0 {
                m.wasted += 1;
            }
            let spine = tree.spine();
            let flat = spine
                .iter()
                .zip(actual.iter())
                .take_while(|(a, b)| a == b)
                .count();
            m.flat_accepted += flat as u64;
        }
    }

    m.lat_ns.sort_unstable();
    let engaged_f = m.engaged.max(1) as f64;
    let steps_f = m.steps.max(1) as f64;

    println!("REST store eval");
    println!("  store            {}", store.path().display());
    println!(
        "  corpus           {} tokens, {} docs",
        store.header().n_tokens,
        store.header().n_docs
    );
    println!(
        "  replayed         {} generations, {} decode steps",
        samples.len(),
        m.steps
    );
    if args.strip_boilerplate {
        let offered = m.steps + n_skipped;
        println!(
            "  boilerplate      {n_skipped} of {offered} held-out tokens EXCLUDED from scoring \
             ({:.2}%); {} scored",
            100.0 * n_skipped as f64 / offered.max(1) as f64,
            m.steps
        );
    }
    println!(
        "  config           max_k={} min_match={} depth={} max_nodes={} max_occ={}",
        args.max_k, args.min_match, args.depth, args.max_nodes, args.max_occurrences
    );
    println!();
    println!(
        "  engagement       {:.2}%  ({} / {} steps)",
        100.0 * m.engaged as f64 / steps_f,
        m.engaged,
        m.steps
    );
    println!(
        "  accepted lookahead (tree)   {:.3} tok/engaged step   {:.3} tok/decode step",
        m.tree_accepted as f64 / engaged_f,
        m.tree_accepted as f64 / steps_f
    );
    println!(
        "  accepted lookahead (spine)  {:.3} tok/engaged step   {:.3} tok/decode step",
        m.flat_accepted as f64 / engaged_f,
        m.flat_accepted as f64 / steps_f
    );
    println!(
        "  wasted engagements          {:.2}%  (drafted, accepted 0)",
        100.0 * m.wasted as f64 / engaged_f
    );
    println!(
        "  mean nodes proposed         {:.2}",
        m.nodes as f64 / engaged_f
    );
    println!();
    println!(
        "  lookup latency   p50 {:.2} µs   p99 {:.2} µs   max {:.2} µs",
        Metrics::percentile(&m.lat_ns, 0.50),
        Metrics::percentile(&m.lat_ns, 0.99),
        Metrics::percentile(&m.lat_ns, 1.0)
    );
    println!();
    println!("  match-length distribution (share of decode steps):");
    for (len, &count) in m.match_len_hist.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let label = if len == 0 {
            "  none".to_string()
        } else if len == args.max_k {
            format!(" >={len:3}")
        } else {
            format!("  {len:3} ")
        };
        println!(
            "    {label}  {:6.2}%  {count}",
            100.0 * count as f64 / steps_f
        );
    }
    Ok(())
}
