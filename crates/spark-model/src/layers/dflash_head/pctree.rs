// SPDX-License-Identifier: AGPL-3.0-only

//! PCTree — parent-conditioned budgeted tree drafting.
//!
//! Implements Algorithm 1 of "From Chains to Trees: Parent-Conditioned Drafting
//! for Semi-Autoregressive Speculative Decoding" (arXiv 2608.02123) over our
//! block drafter.
//!
//! # The idea, and why it is not just "branching"
//!
//! A semi-autoregressive drafter produces shared per-position base logits
//! `L_0..L_{B-1}` in ONE backbone forward. The chain decoder walks them
//! left-to-right, so a single early mismatch invalidates the whole remaining
//! suffix — our measured failure mode (5.81 accepted of 15 at gamma 15).
//!
//! Naively branching on `L_d` alone gives every parent at depth `d-1` the SAME
//! child distribution ("parent-independent stitching", the paper's fig. 2), and
//! their own Table 4 control shows that branching *without* parent conditioning
//! is not where the gain comes from. PCTree instead re-scores the shared logits
//! per concrete parent,
//!
//! ```text
//!     z_d(p) = L_d + bias(p),      pi_d(. | p) = softmax(z_d(p))
//! ```
//!
//! so siblings differ through their parent's bias. That requires a
//! **parent-conditioned scorer** — see [`ParentConditionedScorer`] — which is
//! the one component this is gated on.
//!
//! # Checkpoint support (read this before enabling)
//!
//! The scorer is NOT part of the drafter backbone; it is a separate head that
//! must be present in the checkpoint:
//!
//! | checkpoint                          | scorer                                | PCTree |
//! |-------------------------------------|---------------------------------------|--------|
//! | `drafter-qwen38-v2` (**production**) | none — 69 tensors, `fc`/`layers`/`norm` | NO   |
//! | `drafter-dflash2-incoai`            | `candidate_selector.{predecessor,successor}_codebook` | yes |
//! | `DSpark-AEON-draft`, `dspark-drafter` | `markov_w1`/`markov_w2` (rank 256)  | yes    |
//!
//! Our production drafter carries neither head, so with it PCTree has nothing
//! to condition on and [`scorer_kind`] reports [`ScorerKind::None`]; the caller
//! must fall back to chain drafting. This is a checkpoint gap, not a code gap.
//!
//! # Losslessness
//!
//! Tree drafting changes WHICH tokens are proposed. It cannot change which are
//! committed: the target verifies with the strict greedy tree rule, accepting a
//! node only where the drafted token equals the target's own argmax along an
//! ancestor path. A different tree therefore yields a different acceptance
//! length, never different output. See `pctree_losslessness` in the tests.

// The tree constructor below is exercised only by this module's tests until the
// device-side scorer is bound (see `PcTreeDecision::Engaged`, which currently
// logs and falls back). `build_tree` and its supporting types are therefore
// unreachable from the release build. This allow is scoped to the module and
// MUST be deleted when the scorer binding lands, so that genuine dead code in
// here starts failing the build again.
#![allow(dead_code)]

use anyhow::Result;

use super::ddtree::TreePayload;

/// One scored child proposed for a concrete parent, under `pi_d(. | parent)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ScoredChild {
    pub token: u32,
    /// `log pi_d(token | parent)`. Must be <= 0; the joint-score ordering that
    /// makes the selected set prefix-closed depends on it.
    pub logprob: f32,
}

/// Supplies parent-conditioned child distributions over the shared block logits.
///
/// One call per depth, batched across the whole frontier — the paper's cost
/// argument (§4.3) is that this is `B` calls of batch `b_d <= k`, not `B*k`
/// calls of batch 1. The bias projection is bandwidth-bound on its weight, so
/// widening the batch is close to free.
pub(super) trait ParentConditionedScorer {
    /// Local top-`k` children of each parent in `parents`, at block position
    /// `depth`. Returns one row per parent, in the same order.
    fn top_k_children(
        &self,
        depth: usize,
        parents: &[u32],
        k: usize,
    ) -> Result<Vec<Vec<ScoredChild>>>;
}

/// Which parent-conditioned head the loaded checkpoint provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScorerKind {
    /// DSpark low-rank bigram: `bias(p) = markov_w2 @ markov_w1[p]`.
    Markov,
    /// DFlash2 candidate selector: predecessor/successor codebooks.
    Selector,
    /// Neither head present — PCTree cannot run on this checkpoint.
    None,
}

/// Tree shape parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PcTreeParams {
    /// Block length `B` — the number of expansion depths (our `gamma`).
    pub block: usize,
    /// Local branching factor `k`. `k = 1` collapses to the greedy chain.
    pub k: usize,
    /// Node budget, counting DRAFTED nodes only. The paper's `N` includes the
    /// root (the already-committed token that seeds the block), which we never
    /// emit — so paper `N = 32` is `nodes = 31` here, exactly our tree-WY cap.
    pub nodes: usize,
}

impl PcTreeParams {
    /// Upper bound on the candidate pool, `k + (B-1)k^2` (paper §4.1). Used to
    /// pre-size the pool and to keep a pathological `k` from exploding it.
    pub fn pool_bound(&self) -> usize {
        self.k + self.block.saturating_sub(1) * self.k * self.k
    }
}

/// A node in the candidate pool. The root is implicit (index `None`).
#[derive(Debug, Clone, Copy)]
struct PoolNode {
    token: u32,
    /// Pool index of the parent, or `None` when the parent is the root.
    parent: Option<usize>,
    depth: usize,
    /// Joint path score `s(c) = s(parent) + log pi(c | parent)`, root `s = 0`.
    score: f32,
}

/// Build a budgeted parent-conditioned draft tree.
///
/// `root` is the already-verified token that seeds the block; it is the first
/// conditioning parent and is never emitted.
///
/// Returns a [`TreePayload`] whose `parent_indices[i]` is `-1` or a strictly
/// smaller index — the invariant `DflashDraftBudget::validate_tree` enforces.
pub(super) fn build_tree<S: ParentConditionedScorer>(
    params: PcTreeParams,
    root: u32,
    scorer: &S,
) -> Result<TreePayload> {
    anyhow::ensure!(params.k >= 1, "PCTree branching factor must be >= 1");
    anyhow::ensure!(params.nodes >= 1, "PCTree node budget must be >= 1");
    anyhow::ensure!(params.block >= 1, "PCTree block length must be >= 1");

    let mut pool: Vec<PoolNode> = Vec::with_capacity(params.pool_bound());
    // Frontier holds pool indices; `None` is the implicit root, so depth 0
    // expands exactly one parent.
    let mut frontier: Vec<Option<usize>> = vec![None];

    for depth in 0..params.block {
        if frontier.is_empty() {
            break;
        }
        let parent_tokens: Vec<u32> = frontier
            .iter()
            .map(|f| f.map_or(root, |i| pool[i].token))
            .collect();
        let rows = scorer.top_k_children(depth, &parent_tokens, params.k)?;
        anyhow::ensure!(
            rows.len() == frontier.len(),
            "PCTree scorer returned {} child rows for {} frontier parents at depth {depth}",
            rows.len(),
            frontier.len()
        );

        // Every node added at this depth is a candidate for the next frontier.
        let mut added: Vec<usize> = Vec::with_capacity(frontier.len() * params.k);
        for (slot, children) in rows.iter().enumerate() {
            let parent = frontier[slot];
            let parent_score = parent.map_or(0.0, |i| pool[i].score);
            for child in children.iter().take(params.k) {
                anyhow::ensure!(
                    child.logprob <= 0.0 && child.logprob.is_finite(),
                    "PCTree child log-prob {} is not a finite value <= 0; the \
                     prefix-closure argument depends on scores being non-increasing \
                     along a path",
                    child.logprob
                );
                pool.push(PoolNode {
                    token: child.token,
                    parent,
                    depth,
                    score: parent_score + child.logprob,
                });
                added.push(pool.len() - 1);
            }
        }

        // Layer-wise pruning: only the k best of THIS depth's children are
        // expanded next. This bounds the frontier at k and the pool at
        // k + (B-1)k^2 instead of k^depth.
        added.sort_by(|&a, &b| cmp_nodes(&pool[a], a, &pool[b], b));
        added.truncate(params.k);
        frontier = added.into_iter().map(Some).collect();
    }

    Ok(select_budgeted(&pool, params.nodes))
}

/// Ordering used for both pruning and final selection: decreasing joint score,
/// ties broken by shallower depth, then by stable pool index.
///
/// The depth tiebreak is load-bearing, not cosmetic: a child with
/// `log pi = 0` (a probability-1 continuation) ties its parent exactly, and
/// without it the child could sort ahead of its own parent and break the
/// prefix-closure the payload contract requires.
fn cmp_nodes(a: &PoolNode, ai: usize, b: &PoolNode, bi: usize) -> std::cmp::Ordering {
    b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.depth.cmp(&b.depth))
        .then(ai.cmp(&bi))
}

/// Rank the whole pool and retain the best `budget` nodes, prefix-closed.
///
/// Because `s(c) = s(p) + log pi <= s(p)` and ties break toward the shallower
/// node, every ancestor sorts before its descendants — so emitting in sorted
/// order already satisfies `parent_index < child_index`. The explicit
/// `admitted` check is belt-and-braces: it makes prefix-closure a property of
/// the code rather than of a floating-point argument, so a node whose parent
/// was cut can never reach the payload.
fn select_budgeted(pool: &[PoolNode], budget: usize) -> TreePayload {
    let mut order: Vec<usize> = (0..pool.len()).collect();
    order.sort_by(|&a, &b| cmp_nodes(&pool[a], a, &pool[b], b));

    // pool index -> emitted index
    let mut emitted: Vec<Option<usize>> = vec![None; pool.len()];
    let mut tree_token_ids = Vec::with_capacity(budget.min(pool.len()));
    let mut parent_indices = Vec::with_capacity(budget.min(pool.len()));

    for &idx in &order {
        if tree_token_ids.len() >= budget {
            break;
        }
        let node = &pool[idx];
        let parent_emitted = match node.parent {
            // Parent is the implicit root: always present, emitted as -1.
            None => Some(-1i32),
            Some(p) => emitted[p].map(|e| e as i32),
        };
        let Some(parent_index) = parent_emitted else {
            // Parent was not admitted; skip to keep the tree prefix-closed.
            continue;
        };
        emitted[idx] = Some(tree_token_ids.len());
        tree_token_ids.push(node.token);
        parent_indices.push(parent_index);
    }

    TreePayload {
        tree_token_ids,
        parent_indices,
    }
}

/// PCTree configuration from the environment. `None` = disabled (the default),
/// in which case the caller must use chain drafting and the emitted tokens are
/// byte-identical to today.
///
/// * `ATLAS_PCTREE=1` — enable.
/// * `ATLAS_PCTREE_NODES` — drafted-node budget, default 16 (the pure
///   reallocation point: same verification budget as today's chain, so the
///   measured `verify(k) = 75.53 + 1.890k` cost is unchanged and only
///   acceptance can move).
/// * `ATLAS_PCTREE_K` — branching factor, default 4 (the paper's setting;
///   their Table 5 shows k=2 captures most of the gain and k=8 adds nothing).
pub(super) fn pctree_params_from_env(block: usize) -> Option<PcTreeParams> {
    static CACHED: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();
    let cfg = (*CACHED.get_or_init(|| {
        if std::env::var("ATLAS_PCTREE").ok().as_deref() != Some("1") {
            return None;
        }
        let num = |key: &str, default: usize| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v >= 1)
                .unwrap_or(default)
        };
        // 31 is the hard ceiling: the tree-WY verify kernel has K_MAX = 32
        // including the root (ddtree.rs, "max_nodes <= 31").
        let nodes = num("ATLAS_PCTREE_NODES", 16).min(31);
        let k = num("ATLAS_PCTREE_K", 4);
        tracing::info!(
            "PCTree ENABLED (ATLAS_PCTREE=1): nodes={nodes} k={k}. Parent-conditioned \
             tree drafting changes WHICH tokens are proposed; committed output is \
             unchanged because the target verifies by strict greedy tree argmax."
        );
        Some((nodes, k))
    }))?;
    Some(PcTreeParams {
        block,
        k: cfg.1,
        nodes: cfg.0,
    })
}

/// Report which parent-conditioned head the loaded checkpoint provides.
///
/// This is the gate that matters. Our production drafter
/// (`drafter-qwen38-v2-epoch4-step24852`) has neither head — verified against
/// the checkpoint, whose 69 tensors are only `fc`/`hidden_norm`/`layers`/`norm`
/// — so PCTree reports [`ScorerKind::None`] and the caller must stay on the
/// chain. Enabling the flag on such a checkpoint is a configuration error, not
/// a silent degradation, so it is logged once at warn level.
pub(super) fn scorer_kind(has_markov: bool, has_selector: bool) -> ScorerKind {
    if has_markov {
        ScorerKind::Markov
    } else if has_selector {
        ScorerKind::Selector
    } else {
        ScorerKind::None
    }
}

/// Whether PCTree can actually run, logging the reason once when it cannot.
pub(super) fn pctree_available(kind: ScorerKind) -> bool {
    if kind != ScorerKind::None {
        return true;
    }
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "ATLAS_PCTREE=1 but this drafter checkpoint has no parent-conditioned \
             scorer (no markov_w1/w2, no candidate_selector.*). PCTree re-scores \
             shared block logits per concrete parent, so without one of those \
             heads there is nothing to condition on — falling back to chain \
             drafting. Use a DFlash2 or DSpark checkpoint, or train a head."
        );
    });
    false
}

/// Why PCTree did or did not engage for this propose call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum PcTreeDecision {
    /// Flag unset — chain drafting, byte-identical to today.
    Disabled,
    /// Flag set but the checkpoint has no parent-conditioned head.
    NoScorer,
    /// Flag set and a head is present; tree drafting is applicable.
    Engaged(ScorerKind, PcTreeParams),
}

impl super::BlockDiffusionDraftHead {
    /// Resolve the PCTree decision for this drafter.
    ///
    /// Kept separate from the propose path so the whole policy is testable
    /// without a GPU: the inputs are the env flag and two booleans describing
    /// what the checkpoint loaded.
    pub(super) fn pctree_decision(&self) -> PcTreeDecision {
        let Some(params) = pctree_params_from_env(self.gamma) else {
            return PcTreeDecision::Disabled;
        };
        let kind = scorer_kind(self.markov.is_some(), self.selector.is_some());
        if !pctree_available(kind) {
            return PcTreeDecision::NoScorer;
        }
        PcTreeDecision::Engaged(kind, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic scorer: child tokens are `parent * 10 + rank`, and the
    /// rank-`r` child has log-prob `-0.5 * (r + 1)`. Parent-dependent by
    /// construction, so a bug that ignores the parent shows up as duplicate
    /// tokens across siblings-of-different-parents.
    ///
    /// The top child is deliberately NOT certain (`-0.5`, not `0.0`): with a
    /// certain top child the greedy chain dominates every sibling at every
    /// depth and best-first selection correctly returns a pure chain, which
    /// would make the branching assertions below vacuous rather than wrong.
    struct FakeScorer;
    impl ParentConditionedScorer for FakeScorer {
        fn top_k_children(
            &self,
            _depth: usize,
            parents: &[u32],
            k: usize,
        ) -> Result<Vec<Vec<ScoredChild>>> {
            Ok(parents
                .iter()
                .map(|p| {
                    (0..k)
                        .map(|r| ScoredChild {
                            token: p.wrapping_mul(10).wrapping_add(r as u32),
                            logprob: -0.5 * (r as f32 + 1.0),
                        })
                        .collect()
                })
                .collect())
        }
    }

    fn params(block: usize, k: usize, nodes: usize) -> PcTreeParams {
        PcTreeParams { block, k, nodes }
    }

    /// The payload contract `DflashDraftBudget::validate_tree` enforces.
    fn assert_valid(p: &TreePayload, budget: usize) {
        assert_eq!(p.tree_token_ids.len(), p.parent_indices.len());
        assert!(p.len() <= budget, "budget {budget} exceeded: {}", p.len());
        for (child, &parent) in p.parent_indices.iter().enumerate() {
            assert!(
                parent == -1 || (parent >= 0 && (parent as usize) < child),
                "node {child} has invalid parent {parent}"
            );
        }
    }

    #[test]
    fn k1_collapses_to_the_greedy_chain() {
        let t = build_tree(params(16, 1, 16), 7, &FakeScorer).unwrap();
        assert_valid(&t, 16);
        assert_eq!(t.len(), 16, "k=1 should fill the budget with one chain");
        // A chain: every node's parent is the previous node.
        assert_eq!(t.parent_indices[0], -1);
        for i in 1..t.len() {
            assert_eq!(t.parent_indices[i], (i - 1) as i32);
        }
    }

    #[test]
    fn the_budget_is_never_exceeded_and_the_tree_stays_prefix_closed() {
        for k in 1..=8 {
            for nodes in [1usize, 4, 15, 16, 24, 31, 64] {
                let t = build_tree(params(16, k, nodes), 3, &FakeScorer).unwrap();
                assert_valid(&t, nodes);
            }
        }
    }

    /// The 16-node reallocation setting: same verify budget as today's chain,
    /// spent on a tree instead. It must still fill the budget.
    #[test]
    fn sixteen_node_reallocation_fills_the_budget_and_branches() {
        let t = build_tree(params(16, 4, 16), 3, &FakeScorer).unwrap();
        assert_valid(&t, 16);
        assert_eq!(t.len(), 16);
        // At least one node must have a sibling, or this is just a chain.
        let mut seen = std::collections::HashMap::new();
        for &p in &t.parent_indices {
            *seen.entry(p).or_insert(0usize) += 1;
        }
        assert!(
            seen.values().any(|&c| c > 1),
            "k=4 produced no branching: {:?}",
            t.parent_indices
        );
    }

    /// Children must be conditioned on their own parent, not on depth alone.
    /// Parent-independent stitching is the paper's negative control (fig. 2)
    /// and would show up here as two different parents yielding equal children.
    #[test]
    fn siblings_of_different_parents_are_scored_from_their_own_parent() {
        let rows = FakeScorer.top_k_children(0, &[1, 2], 3).unwrap();
        assert_eq!(rows.len(), 2);
        assert_ne!(
            rows[0].iter().map(|c| c.token).collect::<Vec<_>>(),
            rows[1].iter().map(|c| c.token).collect::<Vec<_>>(),
            "scorer ignored the parent token"
        );
    }

    #[test]
    fn the_pool_bound_matches_the_papers_k_plus_b_minus_one_k_squared() {
        assert_eq!(params(16, 4, 31).pool_bound(), 4 + 15 * 16);
        assert_eq!(params(16, 1, 16).pool_bound(), 1 + 15);
    }

    /// A probability-1 child ties its parent's joint score exactly. Without the
    /// shallower-depth tiebreak it could sort ahead of its own parent and
    /// produce `parent_index > child_index`, which validate_tree rejects.
    #[test]
    fn zero_logprob_children_cannot_outrank_their_own_parent() {
        struct Certain;
        impl ParentConditionedScorer for Certain {
            fn top_k_children(
                &self,
                depth: usize,
                parents: &[u32],
                k: usize,
            ) -> Result<Vec<Vec<ScoredChild>>> {
                Ok(parents
                    .iter()
                    .map(|p| {
                        (0..k)
                            .map(|r| ScoredChild {
                                token: p.wrapping_add(depth as u32).wrapping_add(r as u32),
                                logprob: 0.0,
                            })
                            .collect()
                    })
                    .collect())
            }
        }
        let t = build_tree(params(8, 3, 20), 1, &Certain).unwrap();
        assert_valid(&t, 20);
        assert!(!t.is_empty());
    }

    /// A scorer that emits a positive log-prob would break the monotonicity the
    /// prefix-closure argument rests on, so it is rejected rather than silently
    /// producing an invalid tree.
    #[test]
    fn a_positive_log_prob_is_rejected() {
        struct Bad;
        impl ParentConditionedScorer for Bad {
            fn top_k_children(
                &self,
                _d: usize,
                p: &[u32],
                _k: usize,
            ) -> Result<Vec<Vec<ScoredChild>>> {
                Ok(p.iter()
                    .map(|_| {
                        vec![ScoredChild {
                            token: 1,
                            logprob: 0.5,
                        }]
                    })
                    .collect())
            }
        }
        assert!(build_tree(params(4, 2, 8), 0, &Bad).is_err());
    }

    /// A scorer returning the wrong number of rows is a contract violation, not
    /// something to paper over with a shorter frontier.
    #[test]
    fn a_mismatched_scorer_row_count_is_an_error() {
        struct Short;
        impl ParentConditionedScorer for Short {
            fn top_k_children(
                &self,
                _d: usize,
                _p: &[u32],
                _k: usize,
            ) -> Result<Vec<Vec<ScoredChild>>> {
                Ok(vec![])
            }
        }
        assert!(build_tree(params(4, 2, 8), 0, &Short).is_err());
    }

    /// Losslessness is a property of VERIFICATION, not of this module: the tree
    /// only decides what is proposed. This test pins the part we own — that the
    /// payload is a well-formed ancestor-closed tree for every shape — because
    /// a malformed mask is the one way drafting could corrupt a commit.
    #[test]
    fn pctree_losslessness() {
        for k in 1..=6 {
            for block in [1usize, 4, 8, 15, 16] {
                let t = build_tree(params(block, k, 31), 42, &FakeScorer).unwrap();
                assert_valid(&t, 31);
                // Every non-root parent index must address an already-emitted
                // node, so the ancestor-only mask is well defined for each row.
                for (child, &parent) in t.parent_indices.iter().enumerate() {
                    if parent >= 0 {
                        assert!((parent as usize) < child);
                    }
                }
            }
        }
    }

    #[test]
    fn scorer_kind_prefers_markov_then_selector_then_none() {
        assert_eq!(scorer_kind(true, true), ScorerKind::Markov);
        assert_eq!(scorer_kind(true, false), ScorerKind::Markov);
        assert_eq!(scorer_kind(false, true), ScorerKind::Selector);
        assert_eq!(scorer_kind(false, false), ScorerKind::None);
    }

    /// The production drafter's capability, transcribed from its checkpoint:
    /// 69 tensors under fc/hidden_norm/layers/norm, no head of either kind.
    #[test]
    fn the_production_drafter_cannot_run_pctree() {
        assert_eq!(scorer_kind(false, false), ScorerKind::None);
        assert!(!pctree_available(ScorerKind::None));
        assert!(pctree_available(ScorerKind::Markov));
        assert!(pctree_available(ScorerKind::Selector));
    }

    #[test]
    fn a_decision_carries_the_scorer_kind_and_shape() {
        let p = PcTreeParams {
            block: 15,
            k: 4,
            nodes: 16,
        };
        let d = PcTreeDecision::Engaged(ScorerKind::Markov, p);
        match d {
            PcTreeDecision::Engaged(k, params) => {
                assert_eq!(k, ScorerKind::Markov);
                assert_eq!(params.nodes, 16);
            }
            other => panic!("unexpected decision {other:?}"),
        }
    }
}
