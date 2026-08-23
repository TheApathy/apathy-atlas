// SPDX-License-Identifier: AGPL-3.0-only

//! `rest-store-selfctx` — replay real generations against the SELF-CONTEXT
//! drafter and measure what it would have contributed.
//!
//! No store, no GPU, no model. Each generation is replayed token by token;
//! at every decode step the sequence's own history (prompt + what it has
//! emitted so far) is asked for a continuation, and the answer is scored
//! against the tokens the target actually emitted next.
//!
//! # On contamination
//!
//! There is none to control for, and that is worth stating rather than
//! leaving implicit. A held-out split exists to stop a corpus from having
//! memorised the very generation being scored. Self-context has no corpus:
//! at step `t` it can only see `tokens[..t]`, which the target had already
//! emitted before it produced `tokens[t]`. The replay reproduces exactly
//! the information the live drafter would have had, so a holdout would
//! hold nothing out.
//!
//! ```text
//! rest-store-selfctx --tokenizer tokenizer.json --jsonl aeon_dedup.jsonl \
//!                    --num-drafts 15 --min-match 6 --min-match 8 --min-match 10
//! ```

#![deny(warnings)]
#![deny(clippy::all)]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use rest_store::{load_rows, sam::SuffixAutomaton};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(
    name = "rest-store-selfctx",
    about = "Replay generations against the self-context drafter"
)]
struct Args {
    /// The target model's `tokenizer.json`.
    #[arg(long)]
    tokenizer: PathBuf,

    /// JSONL of real generations (`conversations: [{role, content}]`).
    #[arg(long)]
    jsonl: PathBuf,

    /// Cap on rows read.
    #[arg(long)]
    limit: Option<usize>,

    /// Verify width. A chain shorter than this is declined, mirroring the
    /// server's gate: a retrieval chain must fill the frame the neural
    /// drafter would have filled.
    #[arg(long, default_value_t = 15)]
    num_drafts: usize,

    /// Engage thresholds to report, one row each.
    #[arg(long = "min-match", default_values_t = [6usize, 8, 10])]
    min_match: Vec<usize>,

    /// Drop rows whose assistant turn is a harness error rather than model
    /// output. A repeated error template is trivially self-similar and
    /// would inflate every number here.
    #[arg(long, default_value_t = true)]
    drop_error_rows: bool,

    /// Per-token-acceptance of the neural drafter this would pre-empt.
    /// Used only to print the break-even the retrieval chain must beat.
    #[arg(long, default_value_t = 0.90)]
    drafter_p: f64,
}

/// One replayable stream.
struct Sample {
    tokens: Vec<u32>,
    /// Where the assistant's own output starts; only these are decode steps.
    score_from: usize,
}

impl Sample {
    fn output_len(&self) -> usize {
        self.tokens.len() - self.score_from
    }
}

/// Assistant turns that are harness failures, not generations.
fn is_error_row(completion: &str) -> bool {
    let head = completion.trim_start();
    // Char-boundary safe: completions are UTF-8 prose that may put a
    // multi-byte character across any byte offset.
    let probe: String = head.chars().take(300).collect();
    head.starts_with("{\"error\"") || probe.contains("AdapterError")
}

fn load(
    path: &Path,
    tok: &Tokenizer,
    limit: Option<usize>,
    drop_errors: bool,
) -> Result<(Vec<Sample>, usize)> {
    let (rows, stats) = load_rows(path)?;
    let mut out = Vec::new();
    let mut dropped = 0usize;
    for row in rows {
        if limit.is_some_and(|l| out.len() >= l) {
            break;
        }
        if drop_errors && is_error_row(&row.completion) {
            dropped += 1;
            continue;
        }
        let prompt = tok
            .encode(row.prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?;
        let completion = tok
            .encode(row.completion.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenizing completion: {e}"))?;
        let score_from = prompt.get_ids().len();
        let mut tokens = prompt.get_ids().to_vec();
        tokens.extend_from_slice(completion.get_ids());
        if tokens.len() > score_from + 1 {
            out.push(Sample { tokens, score_from });
        }
    }
    let _ = stats;
    Ok((out, dropped))
}

/// Counters for one bucket of generations.
#[derive(Default, Clone, Copy)]
struct Bucket {
    rows: u64,
    steps: u64,
    engaged: u64,
    accepted: u64,
    wasted: u64,
    /// Engagements and acceptances in the FIRST quarter of each generation.
    early_engaged: u64,
    early_accepted: u64,
    /// ...and in the LAST quarter.
    late_engaged: u64,
    late_accepted: u64,
    /// Engagements that filled the whole verify frame. These are the
    /// positions where retrieval is at its ceiling — and, by the same
    /// token, the positions where a neural drafter is most likely to have
    /// been saturated too.
    full_frame: u64,
    /// Engagements and acceptances bucketed by matched-suffix length.
    by_match: [MatchBucket; MATCH_BUCKETS.len()],
}

/// Engagements and acceptances at one matched-suffix-length band.
#[derive(Default, Clone, Copy)]
struct MatchBucket {
    engaged: u64,
    accepted: u64,
    full_frame: u64,
}

/// Matched-suffix-length bands, as `[lo, hi)`.
const MATCH_BUCKETS: [(&str, usize, usize); 5] = [
    ("gate..20", 0, 20),
    ("20..32", 20, 32),
    ("32..64", 32, 64),
    ("64..128", 64, 128),
    ("128+", 128, usize::MAX),
];

impl Bucket {
    fn add(&mut self, other: &Bucket) {
        self.rows += other.rows;
        self.steps += other.steps;
        self.engaged += other.engaged;
        self.accepted += other.accepted;
        self.wasted += other.wasted;
        self.early_engaged += other.early_engaged;
        self.early_accepted += other.early_accepted;
        self.late_engaged += other.late_engaged;
        self.late_accepted += other.late_accepted;
        self.full_frame += other.full_frame;
        for (mine, theirs) in self.by_match.iter_mut().zip(other.by_match.iter()) {
            mine.engaged += theirs.engaged;
            mine.accepted += theirs.accepted;
            mine.full_frame += theirs.full_frame;
        }
    }

    fn per_engaged(&self) -> f64 {
        if self.engaged == 0 {
            0.0
        } else {
            self.accepted as f64 / self.engaged as f64
        }
    }

    fn per_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.accepted as f64 / self.steps as f64
        }
    }

    fn engagement(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            100.0 * self.engaged as f64 / self.steps as f64
        }
    }
}

/// Replay one generation and score every decode step.
fn replay(
    sample: &Sample,
    min_match: usize,
    num_drafts: usize,
    lat: &mut Vec<u32>,
) -> (Bucket, usize) {
    let mut b = Bucket {
        rows: 1,
        ..Default::default()
    };
    let mut sam = SuffixAutomaton::new();
    let t = &sample.tokens;
    // The prompt is history the drafter has before the first decode step.
    for &token in &t[..sample.score_from] {
        sam.push(token);
    }
    let out_len = sample.output_len();
    let quarter = (out_len / 4).max(1);

    for i in sample.score_from..t.len() {
        // `t[i]` is the token just sampled; it is not committed history yet.
        let position = i - sample.score_from;
        b.steps += 1;

        let t0 = std::time::Instant::now();
        let (match_len, end) = sam.peek(t[i]);
        lat.push(t0.elapsed().as_nanos().min(u32::MAX as u128) as u32);

        // Commit the sampled token before scoring, so the next iteration
        // sees exactly what the live index would.
        sam.push(t[i]);

        // The chain may only read tokens the live drafter would have had:
        // the committed history t[..i]. `t[i]` was sampled this step and
        // is not in the index yet, and t[i+1..] is the future being scored.
        if match_len < min_match || end + num_drafts > i {
            continue;
        }
        // The chain continues the earlier occurrence; the target's actual
        // continuation is what follows the current position.
        let chain = &t[end..end + num_drafts];
        let actual = &t[i + 1..];
        let accepted = chain
            .iter()
            .zip(actual.iter())
            .take_while(|(a, b)| a == b)
            .count();

        b.engaged += 1;
        b.accepted += accepted as u64;
        if accepted == 0 {
            b.wasted += 1;
        }
        if accepted == num_drafts {
            b.full_frame += 1;
        }
        let band = MATCH_BUCKETS
            .iter()
            .position(|&(_, lo, hi)| match_len >= lo && match_len < hi)
            .unwrap_or(0);
        b.by_match[band].engaged += 1;
        b.by_match[band].accepted += accepted as u64;
        if accepted == num_drafts {
            b.by_match[band].full_frame += 1;
        }
        if position < quarter {
            b.early_engaged += 1;
            b.early_accepted += accepted as u64;
        } else if position >= out_len.saturating_sub(quarter) {
            b.late_engaged += 1;
            b.late_accepted += accepted as u64;
        }
    }
    (b, sam.heap_bytes())
}

fn percentile(sorted: &[u32], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx] as f64 / 1000.0
}

/// Expected accepted tokens from a geometric drafter over `k` slots.
fn drafter_expectation(p: f64, k: usize) -> f64 {
    if p >= 1.0 {
        return k as f64;
    }
    p * (1.0 - p.powi(k as i32)) / (1.0 - p)
}

const BUCKETS: [(&str, usize, usize); 4] = [
    ("<2k tokens", 0, 2_000),
    ("2k-8k tokens", 2_000, 8_000),
    ("8k-32k tokens", 8_000, 32_000),
    (">32k tokens", 32_000, usize::MAX),
];

fn main() -> Result<()> {
    let args = Args::parse();
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let (samples, dropped) = load(&args.jsonl, &tokenizer, args.limit, args.drop_error_rows)
        .with_context(|| format!("loading {}", args.jsonl.display()))?;
    if samples.is_empty() {
        bail!("no replayable samples in {}", args.jsonl.display());
    }

    let total_out: usize = samples.iter().map(Sample::output_len).sum();
    let mut lengths: Vec<usize> = samples.iter().map(Sample::output_len).collect();
    lengths.sort_unstable();

    println!("self-context drafter eval");
    println!("  jsonl            {}", args.jsonl.display());
    println!(
        "  replayed         {} generations, {} output tokens ({} error rows dropped)",
        samples.len(),
        total_out,
        dropped
    );
    println!(
        "  output tokens    median {}  p90 {}  max {}",
        lengths[lengths.len() / 2],
        lengths[(lengths.len() * 9) / 10],
        lengths[lengths.len() - 1]
    );
    println!("  verify width     {} drafts/frame", args.num_drafts);
    println!(
        "  drafter it pre-empts: p={:.2} over {} slots = {:.2} accepted tok/step (break-even)",
        args.drafter_p,
        args.num_drafts,
        drafter_expectation(args.drafter_p, args.num_drafts)
    );
    println!("  contamination    not applicable: a step sees only this sequence's own prefix");

    for &min_match in &args.min_match {
        let mut total = Bucket::default();
        let mut buckets = [Bucket::default(); BUCKETS.len()];
        let mut lat: Vec<u32> = Vec::with_capacity(total_out);
        let mut index_bytes = 0usize;
        let mut indexed_tokens = 0usize;
        for sample in &samples {
            let (b, heap) = replay(sample, min_match, args.num_drafts, &mut lat);
            index_bytes += heap;
            indexed_tokens += sample.tokens.len();
            total.add(&b);
            let n = sample.output_len();
            for (slot, &(_, lo, hi)) in BUCKETS.iter().enumerate() {
                if n >= lo && n < hi {
                    buckets[slot].add(&b);
                }
            }
        }
        lat.sort_unstable();

        println!();
        println!("── min_match {min_match} ──");
        println!(
            "  engagement       {:.2}%  ({} / {} steps)",
            total.engagement(),
            total.engaged,
            total.steps
        );
        println!(
            "  accepted         {:.3} tok/engaged step   {:.4} tok/decode step",
            total.per_engaged(),
            total.per_step()
        );
        println!(
            "  wasted           {:.2}% of engagements accepted nothing",
            if total.engaged == 0 {
                0.0
            } else {
                100.0 * total.wasted as f64 / total.engaged as f64
            }
        );
        println!(
            "  lookup latency   p50 {:.3} µs   p99 {:.3} µs",
            percentile(&lat, 0.50),
            percentile(&lat, 0.99)
        );
        println!(
            "  index memory     {:.1} KB per 1k indexed tokens ({} tokens indexed)",
            index_bytes as f64 / indexed_tokens.max(1) as f64 * 1000.0 / 1024.0,
            indexed_tokens
        );
        println!("  by generation length:");
        for (slot, &(label, _, _)) in BUCKETS.iter().enumerate() {
            let b = &buckets[slot];
            if b.rows == 0 {
                println!("    {label:14}  (no generations in this range)");
                continue;
            }
            println!(
                "    {label:14}  {:3} gens  engage {:5.2}%  {:6.3} tok/engaged  {:6.4} tok/step",
                b.rows,
                b.engagement(),
                b.per_engaged(),
                b.per_step()
            );
        }
        println!("  by matched-suffix length:");
        for (slot, &(label, _, _)) in MATCH_BUCKETS.iter().enumerate() {
            let m = &total.by_match[slot];
            if m.engaged == 0 {
                continue;
            }
            println!(
                "    {label:9}  {:6} eng ({:5.1}% of engagements)  {:6.3} tok/engaged  {:5.1}% filled the frame",
                m.engaged,
                100.0 * m.engaged as f64 / total.engaged.max(1) as f64,
                m.accepted as f64 / m.engaged as f64,
                100.0 * m.full_frame as f64 / m.engaged as f64,
            );
        }
        // What pre-emption is actually worth depends on what the DRAFTER
        // would have accepted at the very same positions — which this
        // harness cannot observe, because it has no drafter. It can only
        // bound it. Retrieval engages on repetitive text, which is where
        // a neural drafter is strongest, so the drafter's acceptance at
        // these positions is at or ABOVE its unconditional mean.
        println!(
            "  frame-filling     {:.1}% of engagements accepted all {} drafts (retrieval at ceiling)",
            100.0 * total.full_frame as f64 / total.engaged.max(1) as f64,
            args.num_drafts
        );
        println!("  net gain per engaged step, under assumptions about the drafter it displaces:");
        let baseline = drafter_expectation(args.drafter_p, args.num_drafts);
        let engage_frac = total.engaged as f64 / total.steps.max(1) as f64;
        for (label, drafter_here) in [
            ("unconditional (p=0.90 everywhere)", baseline),
            (
                "elevated (drafter does better on repetitive text)",
                (baseline + args.num_drafts as f64) / 2.0,
            ),
            (
                "saturated (drafter already accepts the whole frame)",
                args.num_drafts as f64,
            ),
        ] {
            let per_engaged = total.per_engaged() - drafter_here;
            // Tokens per verify step, baseline vs retrieval-on-engaged.
            let delta = engage_frac * per_engaged / (1.0 + baseline);
            println!(
                "    {label:52}  drafter {:5.2} → {:+.2} tok/engagement  ({:+.2}% tok/step)",
                drafter_here,
                per_engaged,
                100.0 * delta
            );
        }
        println!(
            "  within a generation: first quarter {:.3} tok/engaged ({} eng), last quarter {:.3} tok/engaged ({} eng)",
            if total.early_engaged == 0 {
                0.0
            } else {
                total.early_accepted as f64 / total.early_engaged as f64
            },
            total.early_engaged,
            if total.late_engaged == 0 {
                0.0
            } else {
                total.late_accepted as f64 / total.late_engaged as f64
            },
            total.late_engaged,
        );
    }
    Ok(())
}
