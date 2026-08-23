// SPDX-License-Identifier: AGPL-3.0-only

//! Shared JSONL generation-row schema.
//!
//! Both `rest-store-build` (which indexes assistant turns) and
//! `rest-store-eval` (which replays them) parse the same rows. They read
//! them through this module and nothing else, so the two tools cannot
//! drift on what "the assistant's output" means — a drift that would show
//! up as a silently wrong engagement number rather than as an error.
//!
//! # Schema
//!
//! One JSON object per line:
//!
//! ```json
//! {"id": "...", "lang": null,
//!  "conversations": [{"role": "user", "content": "..."},
//!                    {"role": "assistant", "content": "..."}],
//!  "_finish": "...", "_tok": 123}
//! ```
//!
//! The assistant turn's `content` is taken **verbatim**, including any
//! `<think>...</think>` wrapper: that text is part of the token stream the
//! target actually emitted, so it is exactly what a retrieval drafter
//! needs to match against. Stripping it would build a store that cannot
//! draft the thinking span — which on this workload is most of the tokens.

use std::path::Path;

use anyhow::{Context, Result};

/// Roles whose `content` counts as model output rather than prompt.
const ASSISTANT_ROLES: &[&str] = &["assistant", "gpt"];

/// One generation row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRow {
    /// The row's `id` field, when present. Used as the holdout partition
    /// key so the partition survives reordering or filtering of the file.
    pub id: Option<String>,
    /// Concatenated non-assistant turns. Context for the eval; never
    /// indexed by the builder.
    pub prompt: String,
    /// Concatenated assistant turns, verbatim. This is the document the
    /// builder indexes and the span the eval scores.
    pub completion: String,
    /// Zero-based index of the line this row came from. The fallback
    /// holdout key when `id` is absent.
    pub line_index: usize,
}

impl GenerationRow {
    /// The key the holdout partition hashes.
    ///
    /// Prefers `id` so that adding, removing, or reordering rows does not
    /// reshuffle which rows are held out — otherwise a store and an eval
    /// built from slightly different files would silently disagree about
    /// the split, which is exactly the contamination this is meant to
    /// prevent.
    pub fn holdout_key(&self) -> String {
        match self.id.as_deref() {
            Some(id) if !id.is_empty() => id.to_string(),
            _ => format!("#line{}", self.line_index),
        }
    }
}

/// Rows that were read but not usable, counted rather than silently dropped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IngestStats {
    /// Rows successfully extracted.
    pub kept: usize,
    /// Lines that were not valid JSON.
    pub malformed: usize,
    /// Rows with no `conversations` array.
    pub no_conversations: usize,
    /// Rows whose `conversations` held no assistant-role turn at all.
    pub no_assistant_turn: usize,
    /// Rows with an assistant turn whose content was empty or whitespace.
    pub empty_assistant: usize,
}

impl IngestStats {
    /// Total lines considered (excluding blank lines).
    pub fn total(&self) -> usize {
        self.kept
            + self.malformed
            + self.no_conversations
            + self.no_assistant_turn
            + self.empty_assistant
    }

    /// Log a one-line summary, warning when anything was dropped.
    pub fn log(&self, path: &Path) {
        let dropped = self.total() - self.kept;
        if dropped > 0 {
            tracing::warn!(
                path = %path.display(),
                kept = self.kept,
                malformed = self.malformed,
                no_conversations = self.no_conversations,
                no_assistant_turn = self.no_assistant_turn,
                empty_assistant = self.empty_assistant,
                "JSONL rows dropped during ingest"
            );
        } else {
            tracing::info!(path = %path.display(), kept = self.kept, "JSONL rows ingested");
        }
    }
}

/// Extract one row from a JSONL line.
///
/// Returns `Ok(None)` for a line that parses but carries no usable
/// generation; `stats` records why. A blank line is not counted at all.
pub fn extract_row(
    line: &str,
    line_index: usize,
    stats: &mut IngestStats,
) -> Option<GenerationRow> {
    if line.trim().is_empty() {
        return None;
    }
    let Ok(row) = serde_json::from_str::<serde_json::Value>(line) else {
        stats.malformed += 1;
        return None;
    };
    let Some(turns) = row.get("conversations").and_then(|c| c.as_array()) else {
        stats.no_conversations += 1;
        return None;
    };

    let mut prompt = String::new();
    let mut completion = String::new();
    let mut saw_assistant = false;
    for turn in turns {
        let role = turn.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = turn.get("content").and_then(|c| c.as_str()).unwrap_or("");
        if ASSISTANT_ROLES.contains(&role) {
            saw_assistant = true;
            completion.push_str(content);
        } else {
            prompt.push_str(content);
        }
    }

    if !saw_assistant {
        stats.no_assistant_turn += 1;
        return None;
    }
    if completion.trim().is_empty() {
        stats.empty_assistant += 1;
        return None;
    }

    stats.kept += 1;
    Some(GenerationRow {
        id: row.get("id").and_then(|i| i.as_str()).map(str::to_string),
        prompt,
        completion,
        line_index,
    })
}

/// Read every usable row from a JSONL file.
pub fn load_rows(path: &Path) -> Result<(Vec<GenerationRow>, IngestStats)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading JSONL {}", path.display()))?;
    let mut stats = IngestStats::default();
    let rows: Vec<GenerationRow> = text
        .lines()
        .enumerate()
        .filter_map(|(i, line)| extract_row(line, i, &mut stats))
        .collect();
    stats.log(path);
    Ok((rows, stats))
}

/// A deterministic holdout partition over generation rows.
///
/// The builder excludes held-out rows from the store; the eval scores
/// exactly those rows. Because both derive membership from the same hash
/// of the same key, a single JSONL file yields a decontaminated in-domain
/// measurement *by construction* — there is no manual dedup step to
/// forget, which is how the Phase 1 measurement was inflated ~2x before it
/// was caught.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Holdout {
    /// Fraction of rows held out, in `[0.0, 1.0]`.
    pub frac: f64,
    /// Partition seed. Changing it reshuffles the split.
    pub seed: u64,
}

/// Resolution of the holdout hash. A row is held out when its hash lands
/// in the first `frac` of this many buckets.
const HOLDOUT_BUCKETS: u64 = 1_000_000;

impl Holdout {
    /// A partition that holds nothing out.
    pub fn none() -> Self {
        Self { frac: 0.0, seed: 0 }
    }

    /// Whether this partition excludes any rows at all.
    pub fn is_active(&self) -> bool {
        self.frac > 0.0
    }

    /// Whether `row` belongs to the held-out set.
    pub fn contains(&self, row: &GenerationRow) -> bool {
        if self.frac <= 0.0 {
            return false;
        }
        if self.frac >= 1.0 {
            return true;
        }
        let h = hash_key(self.seed, row.holdout_key().as_bytes());
        (h % HOLDOUT_BUCKETS) < (self.frac * HOLDOUT_BUCKETS as f64) as u64
    }

    /// Split rows into `(store_rows, holdout_rows)`.
    ///
    /// The two halves are disjoint and together are the input, so a store
    /// built from the first can never contain a row scored from the second.
    pub fn split(&self, rows: Vec<GenerationRow>) -> (Vec<GenerationRow>, Vec<GenerationRow>) {
        rows.into_iter().partition(|r| !self.contains(r))
    }
}

/// FNV-1a over the seed followed by the key.
///
/// Not cryptographic and does not need to be: it only has to spread keys
/// evenly and give the same answer in both binaries, forever. A stdlib
/// `DefaultHasher` would satisfy neither — its output is explicitly not
/// stable across Rust releases, which would silently re-split the corpus
/// on a toolchain bump.
fn hash_key(seed: u64, key: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for b in seed.to_le_bytes().iter().chain(key) {
        h ^= *b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, completion: &str) -> GenerationRow {
        GenerationRow {
            id: Some(id.to_string()),
            prompt: String::new(),
            completion: completion.to_string(),
            line_index: 0,
        }
    }

    fn line(id: &str, user: &str, assistant: &str) -> String {
        serde_json::json!({
            "id": id,
            "lang": serde_json::Value::Null,
            "conversations": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "_finish": "stop",
            "_tok": 42,
        })
        .to_string()
    }

    #[test]
    fn extracts_the_assistant_turn_verbatim() {
        let mut s = IngestStats::default();
        let l = line("g1", "write fib", "<think>recursion</think>def fib(n): ...");
        let r = extract_row(&l, 0, &mut s).unwrap();
        assert_eq!(r.id.as_deref(), Some("g1"));
        assert_eq!(r.prompt, "write fib");
        // The <think> wrapper is part of the emitted stream and must survive.
        assert_eq!(r.completion, "<think>recursion</think>def fib(n): ...");
        assert_eq!(s.kept, 1);
    }

    #[test]
    fn concatenates_multiple_assistant_turns() {
        let l = serde_json::json!({
            "id": "g2",
            "conversations": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "X"},
                {"role": "user", "content": "b"},
                {"role": "gpt", "content": "Y"},
            ],
        })
        .to_string();
        let mut s = IngestStats::default();
        let r = extract_row(&l, 0, &mut s).unwrap();
        assert_eq!(r.completion, "XY");
        assert_eq!(r.prompt, "ab");
    }

    #[test]
    fn skips_and_counts_every_unusable_shape() {
        let mut s = IngestStats::default();

        // Blank lines are not counted at all.
        assert!(extract_row("", 0, &mut s).is_none());
        assert!(extract_row("   ", 1, &mut s).is_none());
        assert_eq!(s.total(), 0);

        assert!(extract_row("{not json", 2, &mut s).is_none());
        assert_eq!(s.malformed, 1);

        assert!(extract_row(r#"{"id":"x"}"#, 3, &mut s).is_none());
        assert_eq!(s.no_conversations, 1);

        let no_asst = r#"{"conversations":[{"role":"user","content":"hi"}]}"#;
        assert!(extract_row(no_asst, 4, &mut s).is_none());
        assert_eq!(s.no_assistant_turn, 1);

        let empty = r#"{"conversations":[{"role":"assistant","content":"  \n "}]}"#;
        assert!(extract_row(empty, 5, &mut s).is_none());
        assert_eq!(s.empty_assistant, 1);

        assert_eq!(s.kept, 0);
        assert_eq!(s.total(), 4);
    }

    #[test]
    fn missing_id_falls_back_to_the_line_index() {
        let l = r#"{"conversations":[{"role":"assistant","content":"hi"}]}"#;
        let mut s = IngestStats::default();
        let r = extract_row(l, 7, &mut s).unwrap();
        assert_eq!(r.id, None);
        assert_eq!(r.holdout_key(), "#line7");
        // An empty id string must not be used as a key either.
        let mut r2 = r.clone();
        r2.id = Some(String::new());
        assert_eq!(r2.holdout_key(), "#line7");
    }

    #[test]
    fn load_rows_reads_a_file_and_reports_drops() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.jsonl");
        let body = format!(
            "{}\n\n{}\n{{bad\n{}\n",
            line("a", "u", "A"),
            line("b", "u", "B"),
            r#"{"conversations":[{"role":"user","content":"only user"}]}"#
        );
        std::fs::write(&path, body).unwrap();
        let (rows, stats) = load_rows(&path).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].completion, "A");
        assert_eq!(rows[1].completion, "B");
        assert_eq!(stats.kept, 2);
        assert_eq!(stats.malformed, 1);
        assert_eq!(stats.no_assistant_turn, 1);
    }

    #[test]
    fn holdout_is_deterministic_across_calls() {
        let h = Holdout {
            frac: 0.15,
            seed: 7,
        };
        let rows: Vec<GenerationRow> = (0..500).map(|i| row(&format!("id{i}"), "x")).collect();
        let first: Vec<bool> = rows.iter().map(|r| h.contains(r)).collect();
        let second: Vec<bool> = rows.iter().map(|r| h.contains(r)).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn holdout_split_is_disjoint_and_total() {
        let h = Holdout { frac: 0.2, seed: 1 };
        let rows: Vec<GenerationRow> = (0..1000).map(|i| row(&format!("id{i}"), "x")).collect();
        let (store, held) = h.split(rows.clone());
        assert_eq!(store.len() + held.len(), rows.len());

        let store_ids: std::collections::HashSet<_> =
            store.iter().map(|r| r.holdout_key()).collect();
        let held_ids: std::collections::HashSet<_> = held.iter().map(|r| r.holdout_key()).collect();
        assert!(
            store_ids.is_disjoint(&held_ids),
            "a row appeared in both the store and the holdout"
        );
        assert_eq!(store_ids.len() + held_ids.len(), rows.len());

        // Roughly the requested fraction — 1000 samples, so allow slack.
        let frac = held.len() as f64 / rows.len() as f64;
        assert!(
            (0.15..0.25).contains(&frac),
            "held out {frac:.3}, wanted ~0.20"
        );
    }

    #[test]
    fn holdout_key_is_the_id_not_the_position() {
        // The same row at a different line index must land in the same half.
        let h = Holdout { frac: 0.5, seed: 3 };
        let mut a = row("stable-id", "x");
        a.line_index = 0;
        let mut b = row("stable-id", "x");
        b.line_index = 999;
        assert_eq!(h.contains(&a), h.contains(&b));
    }

    #[test]
    fn different_seeds_give_different_splits() {
        let rows: Vec<GenerationRow> = (0..400).map(|i| row(&format!("id{i}"), "x")).collect();
        let a: Vec<bool> = rows
            .iter()
            .map(|r| Holdout { frac: 0.3, seed: 1 }.contains(r))
            .collect();
        let b: Vec<bool> = rows
            .iter()
            .map(|r| Holdout { frac: 0.3, seed: 2 }.contains(r))
            .collect();
        assert_ne!(a, b, "seed had no effect on the partition");
    }

    #[test]
    fn degenerate_fractions_behave() {
        let rows: Vec<GenerationRow> = (0..50).map(|i| row(&format!("id{i}"), "x")).collect();
        let (store, held) = Holdout::none().split(rows.clone());
        assert_eq!(store.len(), 50);
        assert!(held.is_empty());
        assert!(!Holdout::none().is_active());

        let (store, held) = Holdout { frac: 1.0, seed: 0 }.split(rows);
        assert!(store.is_empty());
        assert_eq!(held.len(), 50);
    }
}
