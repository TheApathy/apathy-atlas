// SPDX-License-Identifier: AGPL-3.0-only

//! Compact **DSpark draft** routed-expert selection (`REAP_K216_PLAN.json`).
//!
//! # Why this exists
//!
//! The DeepSeek-V4-Flash checkpoint ships its DSpark block drafter as
//! `mtp.{0,1,2}.*` — three full V4 stages whose routed MoE carries the SAME
//! expert count as the target (216 on the REAP-pruned reference checkpoint,
//! 256 on the unpruned 0731 one). That is 8.07 GiB of routed-expert bytes for
//! a drafter that only ever proposes 5 tokens.
//!
//! The reference SparkInfer stack does not serve that. It runs a **compact
//! 64-expert draft**, built offline by `scripts/build_dspark_draft.py` before
//! the server starts. The drafter architecture is UNCHANGED — the only
//! difference between the reference draft `config.json` and the target
//! `config.json` is `n_routed_experts: 64` vs `216`. Every other draft tensor
//! (MLA `wq_a/wq_b/wkv/wo_a/wo_b`, `attn_sink`, the mHC sites, the shared
//! expert, `main_proj`, `markov_head`, `confidence_head`) is byte-identical.
//!
//! # How the 64 are chosen
//!
//! NOT "the first 64", NOT "32 × 2 categories". The algorithm is:
//!
//! 1. `keep_maps.mtp_keep` is the sorted list of ORIGINAL (0..255) expert ids
//!    the REAP pass kept, so its position defines the CURRENT (0..215) id that
//!    the checkpoint tensors are actually named for.
//! 2. For each structured calibration category in order
//!    (`agentic_tool_trajectory`, then `tool_calling`), take the first
//!    `structured_per_category` (32) entries of
//!    `keep_maps.structured_ranked_by_category[cat][mtp_keep_from_layer]`,
//!    keeping only ids that survived REAP and are not already selected.
//! 3. Top the selection up to `experts` (64) from the global REAP ranking
//!    `keep_maps.mtp_ranked`, same availability + dedup rule.
//! 4. Map the chosen ORIGINAL ids back to CURRENT ids, **sort ascending**, and
//!    renumber densely 0..K-1. The sort is what makes a row-slice of
//!    `ffn.gate.{weight,bias}` line up with the renumbered experts.
//!
//! Step 2 does **not** yield 64: on the shipped plan the two structured
//! rankings overlap heavily and only **39** distinct experts survive it, so 25
//! of the 64 come from the global REAP fill in step 3. Any implementation that
//! assumes `2 × 32 = 64` picks a different expert set and silently drafts with
//! the wrong weights.
//!
//! This module reproduces the reference selection bit-exactly (locked by
//! [`tests::matches_shipped_reference_selection`] against the 64 ids in the
//! reference stack's own `DSPARK_DRAFT_PLAN.json`), so Atlas can subset the
//! drafter **at load time** instead of shipping a second 2.94 GiB checkpoint.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};

/// Reference expert count for the compact draft (`recipe.json: draft_experts`).
pub const DEFAULT_DRAFT_EXPERTS: usize = 64;
/// Reference per-category mandatory count
/// (`recipe.json: structured_experts_per_category`).
pub const DEFAULT_STRUCTURED_PER_CATEGORY: usize = 32;
/// Structured-output calibration categories, in the order the reference
/// builder walks them. Order is load-bearing: it decides which category wins
/// a slot when the two rankings overlap and the budget runs out.
pub const STRUCTURED_CATEGORIES: [&str; 2] = ["agentic_tool_trajectory", "tool_calling"];
/// Plan file the REAP pass writes next to the checkpoint shards.
pub const REAP_PLAN_FILE: &str = "REAP_K216_PLAN.json";

/// The subset of `REAP_K216_PLAN.json` the draft selection needs. Everything
/// else in the plan (`keep_by_layer`, `ranked_by_layer`, the observation
/// provenance) is target-side bookkeeping.
#[derive(Deserialize)]
struct ReapPlan {
    keep_maps: KeepMaps,
}

#[derive(Deserialize)]
struct KeepMaps {
    /// Sorted ORIGINAL expert ids kept by REAP; index = CURRENT id.
    mtp_keep: Vec<usize>,
    /// Which target layer's ranking the MTP stages inherit (a JSON *string*
    /// key, `"42"` on the reference plan — the observation dataset covers the
    /// 43 main routed-MoE layers, so `mtp.0` borrows layer 42's map).
    mtp_keep_from_layer: String,
    /// Global REAP ranking over ORIGINAL ids, best first (length 256 — it
    /// ranks the UNPRUNED set, so entries may be absent from `mtp_keep`).
    mtp_ranked: Vec<usize>,
    /// `[category][layer] -> ORIGINAL ids, best first`.
    structured_ranked_by_category: BTreeMap<String, BTreeMap<String, Vec<usize>>>,
}

/// A resolved compact-draft expert subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftExpertSubset {
    /// CURRENT (checkpoint) expert ids, ascending. Draft expert `n` reads the
    /// checkpoint's `…ffn.experts.{checkpoint_ids[n]}.*` tensors, and gate row
    /// `n` is gate row `checkpoint_ids[n]` of the full router.
    pub checkpoint_ids: Vec<usize>,
    /// How many of the slots came from the structured-category pass (the rest
    /// are global-REAP fill). Diagnostic only — logged at load.
    pub structured_count: usize,
    /// `mtp_keep.len()` — the expert count of the checkpoint this plan
    /// describes (216 on the REAP reference). Callers MUST check it against
    /// the drafter's actual gate rows: the ids are positions in THIS plan's
    /// kept set, so applying them to a differently-pruned checkpoint (e.g.
    /// the unpruned 256-expert 0731 drafter) would load real weights under
    /// wrong ids and silently draft garbage. Every id happens to be < 256, so
    /// a bounds check alone does not catch it.
    pub source_experts: usize,
}

impl DraftExpertSubset {
    pub fn len(&self) -> usize {
        self.checkpoint_ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.checkpoint_ids.is_empty()
    }
}

/// Append ids from `ranking` that are available and not already chosen, until
/// `out` reaches `limit`. Mirrors `add_ranked` in `build_dspark_draft.py`.
fn add_ranked(out: &mut Vec<usize>, ranking: &[usize], available: &HashSet<usize>, limit: usize) {
    let mut chosen: HashSet<usize> = out.iter().copied().collect();
    for &expert in ranking {
        if out.len() >= limit {
            return;
        }
        if available.contains(&expert) && chosen.insert(expert) {
            out.push(expert);
        }
    }
}

/// Select the compact draft's routed experts from a `REAP_K216_PLAN.json` body.
///
/// `experts` is the draft expert count (64 on the reference recipe) and
/// `structured_per_category` the mandatory top-N per structured category (32).
/// Setting `structured_per_category = 0` reduces this to a pure global-REAP
/// ranking, matching the reference builder's `--structured-per-category 0`.
pub fn select_draft_experts(
    plan_json: &str,
    experts: usize,
    structured_per_category: usize,
) -> Result<DraftExpertSubset> {
    let plan: ReapPlan =
        serde_json::from_str(plan_json).context("parsing REAP plan for DSpark draft selection")?;
    let km = plan.keep_maps;

    if km.mtp_keep.is_empty() {
        bail!("REAP plan has an empty keep_maps.mtp_keep");
    }
    if km.mtp_keep.windows(2).any(|w| w[0] >= w[1]) {
        bail!(
            "keep_maps.mtp_keep must be strictly ascending to reconstruct current expert ids \
             (it is the CURRENT-id -> ORIGINAL-id table)"
        );
    }
    if experts == 0 || experts >= km.mtp_keep.len() {
        bail!(
            "DSpark draft expert count {experts} must be in 1..{} (the checkpoint's kept set)",
            km.mtp_keep.len()
        );
    }

    let available: HashSet<usize> = km.mtp_keep.iter().copied().collect();
    let mut selected_original: Vec<usize> = Vec::with_capacity(experts);

    if structured_per_category > 0 {
        for category in STRUCTURED_CATEGORIES {
            let by_layer = km
                .structured_ranked_by_category
                .get(category)
                .with_context(|| {
                    format!("REAP plan has no structured_ranked_by_category.{category}")
                })?;
            let ranking = by_layer.get(&km.mtp_keep_from_layer).with_context(|| {
                format!(
                    "REAP plan has no structured_ranked_by_category.{category}.{} \
                     (mtp_keep_from_layer)",
                    km.mtp_keep_from_layer
                )
            })?;
            let head = &ranking[..structured_per_category.min(ranking.len())];
            add_ranked(&mut selected_original, head, &available, experts);
        }
    }
    let structured_count = selected_original.len();

    // The structured pass rarely fills the budget (39/64 on the shipped plan —
    // the two categories share most of their top specialists), so the global
    // REAP ranking supplies the remainder.
    add_ranked(&mut selected_original, &km.mtp_ranked, &available, experts);
    if selected_original.len() != experts {
        bail!(
            "DSpark draft selection produced {} experts, expected {experts} — the REAP plan's \
             rankings do not cover its own kept set",
            selected_original.len()
        );
    }

    // ORIGINAL -> CURRENT, then ASCENDING. The sort is required: gate rows are
    // sliced in this order, so the renumbering must be monotone in checkpoint
    // id or router scores land on the wrong experts.
    let original_to_current: BTreeMap<usize, usize> = km
        .mtp_keep
        .iter()
        .enumerate()
        .map(|(current, &original)| (original, current))
        .collect();
    let mut checkpoint_ids: Vec<usize> = selected_original
        .iter()
        .map(|o| original_to_current[o])
        .collect();
    checkpoint_ids.sort_unstable();

    Ok(DraftExpertSubset {
        checkpoint_ids,
        structured_count,
        source_experts: km.mtp_keep.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 64 CURRENT expert ids the reference stack's own
    /// `DSPARK_DRAFT_PLAN.json` records as `selected_current_expert_ids` for
    /// `--experts 64 --structured-per-category 32`. The shard built from them
    /// hashes to `d72dd9d92abe2cfd2d90931072ae3b920a8f0be09465a88c839072a16d7e5cd5`.
    const REFERENCE_SELECTED_CURRENT: [usize; 64] = [
        6, 7, 8, 12, 16, 17, 20, 22, 23, 26, 30, 32, 35, 47, 53, 59, 62, 67, 68, 69, 72, 82, 87,
        90, 96, 97, 101, 103, 105, 114, 119, 122, 123, 126, 128, 129, 137, 138, 139, 141, 146, 147,
        149, 153, 154, 155, 158, 161, 163, 172, 178, 181, 184, 185, 186, 188, 192, 196, 198, 200,
        202, 205, 206, 208,
    ];

    /// Golden lock against the real reference plan when this machine has the
    /// checkpoint mounted. Skips (rather than fails) elsewhere so CI without
    /// the 100 GB checkpoint stays green — the synthetic tests below cover the
    /// algorithm unconditionally.
    #[test]
    fn matches_shipped_reference_selection() {
        let path = std::env::var("ATLAS_REAP_PLAN_PATH")
            .unwrap_or_else(|_| format!("/home/flocka/sparkinfer-ref/data/tp1/{REAP_PLAN_FILE}"));
        let Ok(body) = std::fs::read_to_string(&path) else {
            eprintln!("skipping: no REAP plan at {path}");
            return;
        };
        let got = select_draft_experts(
            &body,
            DEFAULT_DRAFT_EXPERTS,
            DEFAULT_STRUCTURED_PER_CATEGORY,
        )
        .expect("reference plan selects cleanly");
        assert_eq!(got.checkpoint_ids, REFERENCE_SELECTED_CURRENT);
        // The documented headline: the structured pass alone does NOT fill 64.
        assert_eq!(got.structured_count, 39);
        // Guards the "wrong checkpoint" footgun at the loader.
        assert_eq!(got.source_experts, 216);
    }

    /// `mtp_keep` is CURRENT->ORIGINAL, so a selection expressed in original
    /// ids must come back re-indexed AND sorted.
    fn synthetic_plan() -> String {
        // Kept originals: 10,11,12,13,14,15,16,17 -> current 0..7.
        serde_json::json!({
            "keep_maps": {
                "mtp_keep": [10, 11, 12, 13, 14, 15, 16, 17],
                "mtp_keep_from_layer": "42",
                // Global ranking includes PRUNED ids (99) that must be ignored.
                "mtp_ranked": [99, 17, 10, 16, 11, 15, 12, 14, 13],
                "structured_ranked_by_category": {
                    "agentic_tool_trajectory": { "42": [15, 12, 99] },
                    // Overlaps the first category on 15 — must not double-count.
                    "tool_calling": { "42": [15, 13] }
                }
            }
        })
        .to_string()
    }

    #[test]
    fn structured_pass_dedups_across_categories_then_global_fills() {
        let got = select_draft_experts(&synthetic_plan(), 5, 2).unwrap();
        // Structured: cat1 top-2 = [15, 12]; cat2 top-2 = [15, 13] -> 15 is
        // already chosen, so only 13 is added. 3 distinct, not 4.
        assert_eq!(got.structured_count, 3);
        // Global fill (skipping pruned 99 and the already-chosen): 17, then 10.
        // Originals {15,12,13,17,10} -> currents {5,2,3,7,0} -> sorted.
        assert_eq!(got.checkpoint_ids, vec![0, 2, 3, 5, 7]);
    }

    #[test]
    fn zero_structured_is_pure_global_reap() {
        let got = select_draft_experts(&synthetic_plan(), 3, 0).unwrap();
        assert_eq!(got.structured_count, 0);
        // Global order skipping pruned 99: 17, 10, 16 -> currents 7, 0, 6.
        assert_eq!(got.checkpoint_ids, vec![0, 6, 7]);
    }

    #[test]
    fn structured_budget_never_exceeds_the_expert_count() {
        // Ask for 2 experts with a 3-deep structured head: the cap must bite
        // inside the structured pass, not after it.
        let got = select_draft_experts(&synthetic_plan(), 2, 3).unwrap();
        assert_eq!(got.structured_count, 2);
        assert_eq!(got.checkpoint_ids, vec![2, 5]); // originals 15, 12
    }

    #[test]
    fn rejects_counts_at_or_above_the_kept_set() {
        assert!(select_draft_experts(&synthetic_plan(), 8, 32).is_err());
        assert!(select_draft_experts(&synthetic_plan(), 0, 32).is_err());
    }

    #[test]
    fn rejects_unsorted_keep_map() {
        let body = serde_json::json!({
            "keep_maps": {
                "mtp_keep": [11, 10],
                "mtp_keep_from_layer": "42",
                "mtp_ranked": [10, 11],
                "structured_ranked_by_category": {
                    "agentic_tool_trajectory": { "42": [] },
                    "tool_calling": { "42": [] }
                }
            }
        })
        .to_string();
        assert!(select_draft_experts(&body, 1, 0).is_err());
    }
}
