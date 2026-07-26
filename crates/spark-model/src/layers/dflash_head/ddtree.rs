// SPDX-License-Identifier: AGPL-3.0-only

// M1 milestone (Laguna tree-verify port): the full API surface is
// intentionally present so later milestones (M2 verify bridge, scheduler
// accept walk) can consume it without churning this module. Suppress
// dead-code warnings until those callsites land.
#![allow(dead_code)]

//! DDTree (Draft-Diffusion Tree) — tree builder + greedy walk.
//!
//! Trimmed port of atlas-src `layers/dflash_head/ddtree.rs` for the
//! pure-attention Laguna target. Kept: the tree builder, the TreePayload
//! bridge type, the greedy runtime samplers (flat-safe + full), and the
//! free-slots sibling-branch payload builder. Dropped: GDN/SSM state
//! machinery, DFS reorder, KV-compaction planning, caterpillar/portfolio
//! builders, and batch parent-id metadata (not needed without recurrent
//! state replay).
//!
//! Pure-CPU logic; no CUDA dependency. Built payloads are stashed on the
//! proposer state under `ATLAS_DFLASH_TREE=1` and are UNUSED by verify in
//! M1 — flat DFlash behavior is byte-identical when the gate is off.

use std::collections::{BinaryHeap, HashMap, HashSet};

/// A single draft candidate from one DFlash MASK position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftCandidate {
    pub token_id: u32,
    pub logprob: f32,
}

/// One node in the DDTree. Root is always index 0 with `parent_index = None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeNode {
    pub index: usize,
    pub parent_index: Option<usize>,
    pub token_id: u32,
    pub depth: usize,
    pub score: f32,
}

/// Immutable flattened DDTree. `nodes[0]` is the synthetic root; verifier
/// payload consumes `nodes[1..]`.
#[derive(Debug, Clone)]
pub struct DDTree {
    pub nodes: Vec<TreeNode>,
}

impl DDTree {
    pub fn non_root_nodes(&self) -> &[TreeNode] {
        &self.nodes[1..]
    }

    /// Parent indices for non-root verifier nodes. `-1` (encoded as `i32`)
    /// signals the root.
    pub fn parent_indices_for_verifier(&self) -> Vec<i32> {
        self.non_root_nodes()
            .iter()
            .map(|node| match node.parent_index {
                None | Some(0) => -1,
                Some(p) => (p - 1) as i32,
            })
            .collect()
    }

    pub fn token_ids_for_verifier(&self) -> Vec<u32> {
        self.non_root_nodes().iter().map(|n| n.token_id).collect()
    }

    /// Tokens along the unique path from the root to `node_index`, root excluded.
    pub fn path_token_ids(&self, node_index: usize) -> Vec<u32> {
        let mut path = Vec::new();
        let mut cursor = Some(node_index);
        while let Some(idx) = cursor {
            if idx == 0 {
                break;
            }
            let node = &self.nodes[idx];
            path.push(node.token_id);
            cursor = node.parent_index;
        }
        path.reverse();
        path
    }

    /// Ancestor set of `node_index` (inclusive of root).
    pub fn ancestor_indices(&self, node_index: usize, include_self: bool) -> HashSet<usize> {
        let mut ancestors = HashSet::new();
        let mut cursor = if include_self {
            Some(node_index)
        } else {
            self.nodes[node_index].parent_index
        };
        while let Some(idx) = cursor {
            ancestors.insert(idx);
            cursor = self.nodes[idx].parent_index;
        }
        ancestors
    }

    /// Ancestor-only visibility mask for the verifier's attention.
    /// Returned as a `Vec<Vec<bool>>` shape `[n_nodes, n_nodes]`.
    pub fn visibility_mask(&self) -> Vec<Vec<bool>> {
        let n = self.nodes.len();
        self.nodes
            .iter()
            .map(|node| {
                let visible = self.ancestor_indices(node.index, true);
                (0..n).map(|col| visible.contains(&col)).collect()
            })
            .collect()
    }

    /// Children of `parent_index` keyed by their token id (used by the
    /// greedy walk to look up "what comes next from here").
    ///
    /// Duplicate-token ties resolve to the LOWEST node index (`or_insert`
    /// keeps the first hit in ascending node order): the chain/spine child
    /// is laid before any sibling fork, and its subtree extends deepest, so
    /// a tied fork adds nothing. Without this, a degenerate fork whose token
    /// equals its spine sibling's (ATLAS_DFLASH_TREE_DEGEN=1) hijacks the
    /// walk into the short branch tail.
    pub fn child_by_token(&self, parent_index: usize) -> HashMap<u32, usize> {
        let mut children = HashMap::new();
        for node in self.non_root_nodes() {
            if node.parent_index == Some(parent_index) {
                children.entry(node.token_id).or_insert(node.index);
            }
        }
        children
    }
}

/// Result of [`greedy_tree_walk`]. Includes accepted prefix + bonus token.
#[derive(Debug, Clone, PartialEq)]
pub struct GreedyTreeWalk {
    pub accepted_node_indices: Vec<usize>,
    pub accepted_token_ids: Vec<u32>,
    pub bonus_token_id: u32,
    pub visited_node_indices: Vec<usize>,
}

impl GreedyTreeWalk {
    pub fn output_token_ids(&self) -> Vec<u32> {
        let mut out = self.accepted_token_ids.clone();
        out.push(self.bonus_token_id);
        out
    }
}

/// Walk the tree with greedy target logits. `next_token_for_path` is the
/// verifier oracle: given accepted path tokens up to the current node,
/// return the target model's greedy next token.
pub fn greedy_tree_walk(
    tree: &DDTree,
    mut next_token_for_path: impl FnMut(&[u32]) -> u32,
) -> GreedyTreeWalk {
    let mut cursor = 0usize;
    let mut accepted_nodes = Vec::new();
    let mut accepted_tokens = Vec::new();
    let mut visited_nodes = vec![0usize];

    loop {
        let path = tree.path_token_ids(cursor);
        let next_token = next_token_for_path(&path);
        let children = tree.child_by_token(cursor);
        match children.get(&next_token).copied() {
            None => {
                return GreedyTreeWalk {
                    accepted_node_indices: accepted_nodes,
                    accepted_token_ids: accepted_tokens,
                    bonus_token_id: next_token,
                    visited_node_indices: visited_nodes,
                };
            }
            Some(child_index) => {
                accepted_nodes.push(child_index);
                accepted_tokens.push(next_token);
                visited_nodes.push(child_index);
                cursor = child_index;
            }
        }
    }
}

#[derive(Debug)]
pub enum DDTreeBuildError {
    InvalidTopK,
    InvalidBudget,
    EmptyDepth(usize),
    EmptyCandidates,
}

impl std::fmt::Display for DDTreeBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTopK => write!(f, "top_k must be >= 1"),
            Self::InvalidBudget => write!(f, "budget must be >= 1"),
            Self::EmptyDepth(d) => write!(f, "depth {d} has no draft candidates"),
            Self::EmptyCandidates => {
                write!(f, "candidates_by_depth must contain at least one depth")
            }
        }
    }
}

impl std::error::Error for DDTreeBuildError {}

fn normalize_candidates(
    candidates_by_depth: &[Vec<DraftCandidate>],
    top_k: usize,
) -> Result<Vec<Vec<DraftCandidate>>, DDTreeBuildError> {
    if top_k < 1 {
        return Err(DDTreeBuildError::InvalidTopK);
    }
    let mut normalized = Vec::with_capacity(candidates_by_depth.len());
    for (depth_idx, raw) in candidates_by_depth.iter().enumerate() {
        if raw.is_empty() {
            return Err(DDTreeBuildError::EmptyDepth(depth_idx + 1));
        }
        let mut sorted: Vec<DraftCandidate> = raw.clone();
        // Match Python sort: by logprob DESC, ties broken by token_id ASC.
        sorted.sort_by(|a, b| {
            b.logprob
                .partial_cmp(&a.logprob)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.token_id.cmp(&b.token_id))
        });
        sorted.truncate(top_k);
        normalized.push(sorted);
    }
    if normalized.is_empty() {
        return Err(DDTreeBuildError::EmptyCandidates);
    }
    Ok(normalized)
}

// Heap entry — best score first via Reverse on a negated key.
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    neg_score: f32,
    order: u64,
    parent_index: usize,
    candidate_idx: usize, // index into the per-depth candidate list
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.neg_score == other.neg_score && self.order == other.order
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is max-heap; we negate score so smallest neg_score = best.
        // Invert ordering on neg_score so the heap pops the BEST score first.
        match other
            .neg_score
            .partial_cmp(&self.neg_score)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Equal => other.order.cmp(&self.order),
            o => o,
        }
    }
}

/// Build a best-first speculative tree from per-depth draft candidates.
///
/// `budget` counts non-root verifier nodes. Output shape matches the vLLM
/// reference layout: flattened tree, parent indices, token ids, path scores,
/// and an ancestor-only visibility mask (via `DDTree::visibility_mask`).
pub fn build_ddtree(
    candidates_by_depth: &[Vec<DraftCandidate>],
    budget: usize,
    top_k: usize,
    chain_seed: bool,
    min_root_branches: usize,
    root_token_id: u32,
) -> Result<DDTree, DDTreeBuildError> {
    if budget < 1 {
        return Err(DDTreeBuildError::InvalidBudget);
    }
    let candidates = normalize_candidates(candidates_by_depth, top_k)?;

    let mut nodes: Vec<TreeNode> = vec![TreeNode {
        index: 0,
        parent_index: None,
        token_id: root_token_id,
        depth: 0,
        score: 0.0,
    }];
    let mut child_edges: HashSet<(usize, usize, u32)> = HashSet::new();

    let add_child = |nodes: &mut Vec<TreeNode>,
                     child_edges: &mut HashSet<(usize, usize, u32)>,
                     parent_index: usize,
                     candidate: DraftCandidate|
     -> Option<TreeNode> {
        let parent = nodes[parent_index];
        let depth = parent.depth + 1;
        if depth > candidates.len() {
            return None;
        }
        let edge = (parent_index, depth, candidate.token_id);
        if !child_edges.insert(edge) {
            return None;
        }
        let node = TreeNode {
            index: nodes.len(),
            parent_index: Some(parent_index),
            token_id: candidate.token_id,
            depth,
            score: parent.score + candidate.logprob,
        };
        nodes.push(node);
        Some(node)
    };

    if chain_seed {
        let mut cursor = 0usize;
        while nodes.len() - 1 < budget && nodes[cursor].depth < candidates.len() {
            let depth = nodes[cursor].depth;
            let cand = candidates[depth][0];
            match add_child(&mut nodes, &mut child_edges, cursor, cand) {
                Some(node) => cursor = node.index,
                None => break,
            }
        }
    } else if min_root_branches > 0 {
        let take = min_root_branches.min(candidates[0].len());
        for i in 0..take {
            if nodes.len() > budget {
                break;
            }
            let cand = candidates[0][i];
            add_child(&mut nodes, &mut child_edges, 0, cand);
        }
    }

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut order_counter: u64 = 0;

    let push_children = |nodes: &Vec<TreeNode>,
                         child_edges: &HashSet<(usize, usize, u32)>,
                         heap: &mut BinaryHeap<HeapEntry>,
                         order_counter: &mut u64,
                         parent_index: usize| {
        let parent = nodes[parent_index];
        if parent.depth >= candidates.len() {
            return;
        }
        let depth = parent.depth + 1;
        for (ci, candidate) in candidates[parent.depth].iter().enumerate() {
            let edge = (parent_index, depth, candidate.token_id);
            if child_edges.contains(&edge) {
                continue;
            }
            let score = parent.score + candidate.logprob;
            heap.push(HeapEntry {
                neg_score: -score,
                order: *order_counter,
                parent_index,
                candidate_idx: ci,
            });
            *order_counter += 1;
        }
    };

    // Seed heap from every node already in the tree.
    let initial_indices: Vec<usize> = nodes.iter().map(|n| n.index).collect();
    for idx in initial_indices {
        push_children(&nodes, &child_edges, &mut heap, &mut order_counter, idx);
    }

    while nodes.len() - 1 < budget {
        let Some(entry) = heap.pop() else { break };
        let parent = nodes[entry.parent_index];
        let depth = parent.depth + 1;
        if depth > candidates.len() {
            continue;
        }
        let candidate = candidates[parent.depth][entry.candidate_idx];
        let edge = (entry.parent_index, depth, candidate.token_id);
        if child_edges.contains(&edge) {
            continue;
        }
        let Some(node) = add_child(&mut nodes, &mut child_edges, entry.parent_index, candidate)
        else {
            continue;
        };
        push_children(
            &nodes,
            &child_edges,
            &mut heap,
            &mut order_counter,
            node.index,
        );
    }

    Ok(DDTree { nodes })
}

// ============================================================================
// Tree payload — the propose→verify bridge type
// ============================================================================

/// Sentinel parent id meaning "root of the tree" (encodes -1 in the i32 tensor).
pub const ROOT_PARENT: i32 = -1;

/// One request's tree payload as carried through the verifier bridge (M2+).
/// `tree_token_ids[i]` is compact non-root node `i + 1`; `parent_indices[i]`
/// indexes into the SAME non-root list (or -1 for root).
#[derive(Debug, Clone, Default)]
pub struct TreePayload {
    pub tree_token_ids: Vec<u32>,
    pub parent_indices: Vec<i32>,
}

impl TreePayload {
    pub fn is_empty(&self) -> bool {
        self.tree_token_ids.is_empty()
    }
    pub fn len(&self) -> usize {
        self.tree_token_ids.len()
    }
}

/// Translate a tree payload into the full root+tree parent-id list.
///
/// Output `parents[0] = ROOT_PARENT` always (the synthetic root row).
/// For `i >= 1`, `parents[i]` is the parent of compact node `i`: either
/// `ROOT_PARENT` if the payload's `parent_indices[i-1] < 0` (attaches to
/// root) or `payload.parent_indices[i-1] + 1` (rebasing payload indexing
/// onto the root+tree layout). This is the compact→kernel-frame parents
/// conversion the M2 verify bridge consumes.
pub fn full_parent_ids_from_payload(payload: &TreePayload) -> Result<Vec<i32>, DDTreeBuildError> {
    if payload.tree_token_ids.len() != payload.parent_indices.len() {
        return Err(DDTreeBuildError::EmptyCandidates); // reuse error; payload mismatch
    }
    let mut parents = Vec::with_capacity(payload.parent_indices.len() + 1);
    parents.push(ROOT_PARENT);
    for &p in &payload.parent_indices {
        parents.push(if p < 0 { ROOT_PARENT } else { p + 1 });
    }
    Ok(parents)
}

// ============================================================================
// Runtime sampler — greedy walk over compact verifier argmax rows
// ============================================================================
//
// Given a payload (tree_token_ids + parent_indices) and the target model's
// per-row argmax over compact verifier logits, walk the tree greedily to find
// the accepted branch and bonus token. Then apply the *deployable-safe*
// contract: commit only the contiguous flat prefix (compact index `i+1` for
// step `i`) and turn the first non-flat branch token into the bonus.

/// One request's flattened DDTree verifier payload (matches `TreePayload`
/// but kept distinct for the sampler so it can carry session state later).
#[derive(Debug, Clone)]
pub struct DDTreeRequestRuntime {
    pub req_id: String,
    pub tree_token_ids: Vec<u32>,
    pub parent_indices: Vec<i32>,
}

impl DDTreeRequestRuntime {
    pub fn num_nodes(&self) -> usize {
        self.tree_token_ids.len()
    }

    /// For each compact node (0 = root, 1..N = tree nodes), a map of
    /// `token_id → child compact index`.
    ///
    /// Duplicate-token ties resolve to the LOWEST compact index (`or_insert`
    /// keeps the first hit in ascending payload order). The spine is laid
    /// first (compact slot == depth) and its subtree extends deepest, so on
    /// an exact-token tie the spine child must win — a tied sibling fork
    /// adds nothing. The previous `insert` let the LAST duplicate (a branch
    /// node) overwrite the spine child, hijacking `walk_one_tree` into the
    /// short branch tail: under ATLAS_DFLASH_TREE_DEGEN=1 (fork token ==
    /// spine token) the accept collapsed to the branch length (live: accept
    /// 0.74, dist capped at 2 = branch rows, vs flat baseline ≈3.09).
    pub fn child_maps(&self) -> Vec<std::collections::HashMap<u32, usize>> {
        let mut children: Vec<std::collections::HashMap<u32, usize>> = (0..=self.num_nodes())
            .map(|_| std::collections::HashMap::new())
            .collect();
        for (node_index, (&tok, &parent)) in self
            .tree_token_ids
            .iter()
            .zip(self.parent_indices.iter())
            .enumerate()
        {
            let parent_compact = if parent < 0 { 0 } else { (parent + 1) as usize };
            children[parent_compact]
                .entry(tok)
                .or_insert(node_index + 1);
        }
        children
    }
}

/// Walk one tree using `target_argmax_per_row` (length = 1 + num_nodes,
/// indexed by compact row). Returns `(accepted_tokens, accepted_compact,
/// bonus_token, bonus_parent_compact)`.
fn walk_one_tree(
    req: &DDTreeRequestRuntime,
    target_argmax: &[u32],
) -> (Vec<u32>, Vec<usize>, u32, usize) {
    let children = req.child_maps();
    let mut accepted_tokens = Vec::new();
    let mut accepted_compact = Vec::new();
    let mut cursor_compact: usize = 0;
    loop {
        let next_token = target_argmax[cursor_compact];
        match children[cursor_compact].get(&next_token).copied() {
            None => {
                return (
                    accepted_tokens,
                    accepted_compact,
                    next_token,
                    cursor_compact,
                );
            }
            Some(child_compact) => {
                let node_index = child_compact - 1;
                accepted_tokens.push(req.tree_token_ids[node_index]);
                accepted_compact.push(child_compact);
                cursor_compact = child_compact;
            }
        }
    }
}

/// vLLM contract adapter: commit only contiguous flat prefix + bonus.
///
/// If the accepted path diverges from the flat chain at index `i`, we emit
/// the first `i` accepted tokens + ONE bonus token at the divergence point
/// (vLLM's `accepted_count + 1` contract) and report only the safe
/// accepted_compact prefix.
pub fn adapt_to_flat_safe_contract(
    accepted_tokens: &[u32],
    accepted_compact: &[usize],
    bonus_token: u32,
    bonus_parent: usize,
) -> (Vec<u32>, Vec<usize>, usize) {
    let mut flat_prefix_len = 0;
    for (i, &compact) in accepted_compact.iter().enumerate() {
        if compact == i + 1 {
            flat_prefix_len += 1;
        } else {
            break;
        }
    }

    if flat_prefix_len == accepted_compact.len() {
        // Whole accepted path was already flat → emit all + bonus.
        let mut emitted = accepted_tokens.to_vec();
        emitted.push(bonus_token);
        return (emitted, accepted_compact.to_vec(), bonus_parent);
    }

    // Diverged at flat_prefix_len → keep prefix, replace divergence-point
    // accepted token with what was actually proposed there (still a valid
    // bonus because target's verified token at that row is what we'd
    // commit next anyway).
    let safe_compact = accepted_compact[..flat_prefix_len].to_vec();
    let safe_bonus_parent = safe_compact.last().copied().unwrap_or(0);
    let mut emitted = accepted_tokens[..flat_prefix_len].to_vec();
    emitted.push(accepted_tokens[flat_prefix_len]);
    (emitted, safe_compact, safe_bonus_parent)
}

/// Result of one DDTree greedy sample.
#[derive(Debug, Clone)]
pub struct DDTreeGreedySample {
    pub output_token_ids: Vec<u32>,
    pub accepted_compact_indices: Vec<usize>,
    pub bonus_parent_compact_index: usize,
}

/// Greedy DDTree walk for a single request. Caller pre-computes argmax
/// over each compact verifier row (length = 1 + num_nodes).
pub fn greedy_sample_ddtree(
    req: &DDTreeRequestRuntime,
    target_argmax: &[u32],
) -> Result<DDTreeGreedySample, DDTreeBuildError> {
    let expected = 1 + req.num_nodes();
    if target_argmax.len() != expected {
        return Err(DDTreeBuildError::EmptyCandidates);
    }
    let (acc_tok, acc_compact, bonus_tok, bonus_parent) = walk_one_tree(req, target_argmax);
    let (emitted, safe_acc, safe_bonus_parent) =
        adapt_to_flat_safe_contract(&acc_tok, &acc_compact, bonus_tok, bonus_parent);
    Ok(DDTreeGreedySample {
        output_token_ids: emitted,
        accepted_compact_indices: safe_acc,
        bonus_parent_compact_index: safe_bonus_parent,
    })
}

/// Tree-path (full-branch) greedy DDTree walk — commits the WHOLE accepted
/// path the target's greedy oracle takes through the tree, including a tail
/// reached through a sibling fork, NOT just the contiguous flat prefix.
///
/// Difference from [`greedy_sample_ddtree`]: that function runs
/// `adapt_to_flat_safe_contract`, which truncates the accept at the first
/// fork (compact index != position+1) and turns the fork token into a
/// single bonus — so a fork's tail is never committed (zero branching gain).
/// This function returns the raw `walk_one_tree` result:
///   - `output_token_ids` = every accepted path token + the bonus
///   - `accepted_compact_indices` = the full (possibly NON-contiguous, e.g.
///     `[1, 2, 3, 7]`) compact index path the greedy walk traversed
///   - `bonus_parent_compact_index` = the compact row the bonus is read from
///
/// LOSSLESS: still commits ONLY tokens where `draft == target_argmax` along
/// the path, and the bonus is always the target's greedy at the path tip —
/// identical token contract to the flat path, only the recognized accept
/// SET changes. On the pure-attention Laguna target the caller is
/// responsible for KV compaction of the sparse accepted slots before the
/// next decode (an M2+ concern; unused in M1).
pub fn greedy_sample_ddtree_full(
    req: &DDTreeRequestRuntime,
    target_argmax: &[u32],
) -> Result<DDTreeGreedySample, DDTreeBuildError> {
    let expected = 1 + req.num_nodes();
    if target_argmax.len() != expected {
        return Err(DDTreeBuildError::EmptyCandidates);
    }
    let (acc_tok, acc_compact, bonus_tok, bonus_parent) = walk_one_tree(req, target_argmax);
    let mut emitted = acc_tok;
    emitted.push(bonus_tok);
    Ok(DDTreeGreedySample {
        output_token_ids: emitted,
        accepted_compact_indices: acc_compact,
        bonus_parent_compact_index: bonus_parent,
    })
}

/// Kernel intermediate slot of the **last accepted state** for a greedy
/// walk's `accepted_compact_indices`: slot `k` holds the state AFTER
/// applying `tree_token_ids[k-1]`, so the answer is the LAST (max) compact
/// index, `0` when nothing was accepted. Kept as the minimal dependency of
/// the free-slots / full-sampler tests (the wider SSM plumbing around it
/// was dropped — Laguna is pure-attention).
#[inline]
pub fn last_accepted_inter_slot(accepted_compact_indices: &[usize]) -> usize {
    accepted_compact_indices.last().copied().unwrap_or(0)
}

/// One sibling branch request for [`build_free_slots_payload`]: a fork token
/// that attaches BELOW spine depth `cliff_depth` (i.e. shares the parent of the
/// spine node at that depth), optionally carrying a short re-rooted tail.
#[derive(Debug, Clone)]
pub struct FreeSlotBranch {
    /// 1-based spine depth the fork attaches BELOW. The fork node itself has
    /// tree depth `cliff_depth` (a sibling of `spine[cliff_depth-1]`).
    pub cliff_depth: usize,
    /// The alternative (e.g. drafter top-2) token at the cliff.
    pub fork_token: u32,
    /// Tokens continuing after the fork, re-rooted onto it (contiguous within
    /// the branch). May be empty (a bare 1-node fork leaf).
    pub tail: Vec<u32>,
}

/// Build a FREE-SLOTS branch-verify payload.
///
/// The insight: the K-token DFlash verify is weight-bandwidth-bound, so a
/// few extra verify rows are *free candidate tokens*. Rather than spend
/// them uniformly, this builder spends them on SIBLING BRANCHES placed at
/// the low-confidence draft positions where the linear chain statistically
/// dies (the "cliffs") — a single well-placed fork clears the cliff far
/// more often than deepening the chain.
///
/// ## Layout (compact indices, 1-based; slot 0 = bonus/root)
///
/// The full γ spine is laid FIRST at contiguous compact slots `1..=spine_len`
/// (slot == depth, byte-identical to the flat baseline for every spine node —
/// this is the losslessness anchor: when the target rides the top-1 chain the
/// committed run is exactly the flat path). Then each requested branch is
/// appended as its own contiguous run starting at the next free slot:
///   * the fork node attaches to the spine node at compact slot
///     `cliff_depth - 1` (its cliff parent), so it is a genuine SIBLING of
///     `spine[cliff_depth-1]`;
///   * the branch's tail nodes chain off the fork contiguously.
///
/// Branches are added in ASCENDING cliff depth (shallowest first — EAGLE-2:
/// shallow accept dominates) until `max_nodes` is exhausted. A branch is added
/// whole or with a tail shortened to the remaining slots; the fork node is
/// always laid, and the tail is truncated (never the fork) when the budget runs
/// out — so no partial branch ever references a slot beyond the budget.
///
/// ## Losslessness — REQUIRES per-row ancestor attention for deep commits
///
/// The greedy walker commits a node ONLY when its token equals the target's
/// argmax at its PARENT's row, and the bonus is always the target's argmax at
/// the path-tip row. So the committed stream equals the greedy oracle iff
/// every row the walk READS FROM is correctly conditioned (attends to exactly
/// its ancestors + itself). Under flat prefix metadata only the spine rows
/// are so conditioned; a mis-conditioned fork row COMMITS non-oracle tokens
/// whenever the fork itself is (correctly) accepted. The M2 verify bridge
/// must therefore either supply ancestor-exact attention for branch rows or
/// degrade to the flat-safe walker (spine prefix + fork bonus — lossless
/// because those decisions read only correctly-conditioned spine rows).
///
/// Returns a plain flat spine payload (parents `[-1,0,1,…]`) when no branch
/// fits or `branches` is empty — byte-identical to the drafter baseline.
pub fn build_free_slots_payload(
    spine: &[u32],
    branches: &[FreeSlotBranch],
    max_nodes: usize,
) -> TreePayload {
    let spine_len = spine.len().min(max_nodes);
    let mut tree_token_ids: Vec<u32> = Vec::with_capacity(max_nodes);
    let mut parent_indices: Vec<i32> = Vec::with_capacity(max_nodes);
    // Spine: contiguous, slot == depth. spine[0] → root(-1); spine[i]→spine[i-1].
    for i in 0..spine_len {
        tree_token_ids.push(spine[i]);
        parent_indices.push(i as i32 - 1);
    }
    // Branches in ascending cliff depth (shallowest first).
    let mut ordered: Vec<&FreeSlotBranch> = branches
        .iter()
        .filter(|b| b.cliff_depth >= 1 && b.cliff_depth <= spine_len)
        .collect();
    ordered.sort_by_key(|b| b.cliff_depth);

    for b in ordered {
        let remaining = max_nodes.saturating_sub(tree_token_ids.len());
        if remaining == 0 {
            break;
        }
        // The fork's cliff parent is the spine node at compact slot
        // `cliff_depth - 1`. Compact slot s → payload index s-1. For a fork at
        // depth 1 (cliff_depth == 1) the parent is the root (payload parent -1).
        let cliff_parent_payload_idx: i32 = b.cliff_depth as i32 - 2;
        // Fork node.
        tree_token_ids.push(b.fork_token);
        parent_indices.push(cliff_parent_payload_idx);
        let mut prev = tree_token_ids.len() as i32 - 1; // fork's payload idx
        // Tail (bounded by remaining budget after the fork node).
        let tail_budget = remaining.saturating_sub(1);
        for &t in b.tail.iter().take(tail_budget) {
            tree_token_ids.push(t);
            parent_indices.push(prev);
            prev = tree_token_ids.len() as i32 - 1;
        }
    }

    TreePayload {
        tree_token_ids,
        parent_indices,
    }
}

// ============================================================================
// M2 verify-execution plan — pure host logic, GPU-free
// ============================================================================
//
// Row layout for a tree verify (K_t = 1 + payload.len() rows):
//   row 0            = bonus (last_token, depth 0)
//   rows 1..=S       = spine drafts (depth = row index)
//   remaining rows   = branch runs (fork node at depth d_b + optional tail
//                      at d_b+1, ...), each branch contiguous.
// Row-major payload order guarantees ancestors precede descendants; the plan
// builder REJECTS (returns `None`) any payload violating that shape so the
// verify degrades to the flat path instead of mis-addressing KV.

/// One branch's execution plan: its contiguous verify-row run and the
/// absolute KV block range its rows must remap onto scratch blocks
/// (copy-on-write — sibling branches get DISTINCT scratch blocks).
#[derive(Debug, Clone, PartialEq)]
pub struct TreeBranchPlan {
    /// First verify ROW of this branch (row 0 = bonus; spine rows 1..=S).
    pub first_row: usize,
    /// Number of contiguous rows (fork + tail).
    pub num_rows: usize,
    /// Depth of the branch's deepest row (fork depth + tail length).
    pub max_depth: usize,
    /// Inclusive absolute logical-block range `[lo, hi]` the branch rows
    /// remap: `base/bs ..= (base+max_depth)/bs`. Conservative — covers every
    /// block any of this step's rows can touch, so the branch's reads of
    /// this step's bonus+spine K/V route through the (re-seeded) scratch.
    pub touched_lo: usize,
    pub touched_hi: usize,
}

/// Host-side plan for executing a [`TreePayload`] as tree-shaped verify rows.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeVerifyPlan {
    /// Per-row depth from the verify base position (row 0 = 0, spine row r =
    /// r, branch rows = their tree depth). Length = 1 + payload.len().
    pub row_depths: Vec<usize>,
    /// Per-row branch id (`None` = bonus/spine rows on canonical blocks).
    pub row_branch: Vec<Option<usize>>,
    /// Number of spine rows (excluding the bonus row 0).
    pub spine_len: usize,
    pub branches: Vec<TreeBranchPlan>,
}

impl TreeVerifyPlan {
    pub fn num_rows(&self) -> usize {
        self.row_depths.len()
    }
    pub fn max_depth(&self) -> usize {
        self.row_depths.iter().copied().max().unwrap_or(0)
    }
    /// Branch whose FIRST row is `row`, if any — the per-layer loop re-seeds
    /// that branch's scratch blocks immediately before this row.
    pub fn branch_starting_at_row(&self, row: usize) -> Option<usize> {
        self.branches.iter().position(|b| b.first_row == row)
    }
}

/// Build the M2 execution plan from a payload. `base` is the verify base
/// position (`seq.seq_len` — the bonus row's position); `block_size` is the
/// paged-KV block size. Returns `None` (→ flat fallback) when the payload is
/// not in row-major spine+contiguous-branch shape or is empty/malformed.
pub fn build_tree_verify_plan(
    payload: &TreePayload,
    base: usize,
    block_size: usize,
) -> Option<TreeVerifyPlan> {
    if block_size == 0 {
        return None;
    }
    let n = payload.tree_token_ids.len();
    if n == 0 || payload.parent_indices.len() != n {
        return None;
    }
    // Spine = longest contiguous flat prefix (payload idx i ← parent i-1).
    let mut spine_len = 0usize;
    while spine_len < n && payload.parent_indices[spine_len] == spine_len as i32 - 1 {
        spine_len += 1;
    }
    if spine_len == 0 {
        return None; // no flat anchor — not an M2 shape
    }
    let mut row_depths = Vec::with_capacity(n + 1);
    row_depths.push(0usize); // bonus row
    for d in 1..=spine_len {
        row_depths.push(d);
    }
    let mut row_branch: Vec<Option<usize>> = vec![None; spine_len + 1];
    let mut branches: Vec<TreeBranchPlan> = Vec::new();
    let mut i = spine_len;
    while i < n {
        // Branch head: attaches to the root (-1) or a spine node (< spine).
        let head_parent = payload.parent_indices[i];
        if head_parent < -1 || head_parent >= spine_len as i32 {
            return None; // out-of-order / non-contiguous branch → reject
        }
        // Payload idx p (spine) sits at depth p+1; its child at p+2. The
        // root (-1) yields depth 1.
        let fork_depth = (head_parent + 2) as usize;
        let bidx = branches.len();
        let first_row = i + 1;
        row_depths.push(fork_depth);
        row_branch.push(Some(bidx));
        let mut depth = fork_depth;
        let mut j = i + 1;
        // Tail: chains off the immediately preceding payload node. Since
        // j-1 >= spine_len here, `parent == j-1` is unambiguously a
        // continuation (a new head's parent is < spine_len).
        while j < n && payload.parent_indices[j] == (j - 1) as i32 {
            depth += 1;
            row_depths.push(depth);
            row_branch.push(Some(bidx));
            j += 1;
        }
        branches.push(TreeBranchPlan {
            first_row,
            num_rows: j - i,
            max_depth: depth,
            touched_lo: base / block_size,
            touched_hi: (base + depth) / block_size,
        });
        i = j;
    }
    Some(TreeVerifyPlan {
        row_depths,
        row_branch,
        spine_len,
        branches,
    })
}

/// Per-row attention metadata for a tree verify, in the flat verify's
/// meta_base layout units. Pure host — the caller uploads these.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeRowMetadata {
    /// `positions[t] = base + depth[t]` (RoPE position per row).
    pub positions: Vec<u32>,
    /// `seq_lens[t] = base + depth[t] + 1` (causal visibility horizon).
    pub seq_lens: Vec<i32>,
    /// Physical KV write slot per row (`phys_block * bs + offset`), with
    /// branch rows routed through their scratch blocks.
    pub slots: Vec<i64>,
    /// Per-row block table: canonical, with branch `b`'s touched blocks
    /// substituted by its scratch blocks (spine rows keep canonical).
    pub block_tables: Vec<Vec<u32>>,
}

/// Build per-row metadata. `canonical_table[abs_block] = physical block`
/// (HSS is gated off on the tree path, so absolute == table index).
/// `branch_scratch[b]` lists `(abs_block, scratch_physical)` for branch `b`.
pub fn build_tree_row_metadata(
    plan: &TreeVerifyPlan,
    base: usize,
    block_size: usize,
    canonical_table: &[u32],
    branch_scratch: &[Vec<(usize, u32)>],
) -> TreeRowMetadata {
    let rows = plan.num_rows();
    let mut positions = Vec::with_capacity(rows);
    let mut seq_lens = Vec::with_capacity(rows);
    let mut slots = Vec::with_capacity(rows);
    let mut block_tables = Vec::with_capacity(rows);
    for t in 0..rows {
        let pos = base + plan.row_depths[t];
        positions.push(pos as u32);
        seq_lens.push((pos + 1) as i32);
        let blk = pos / block_size;
        let off = pos % block_size;
        let mut table = canonical_table.to_vec();
        if let Some(b) = plan.row_branch[t] {
            let scratch: &[(usize, u32)] =
                branch_scratch.get(b).map(|v| v.as_slice()).unwrap_or(&[]);
            for &(abs_blk, s) in scratch {
                if abs_blk < table.len() {
                    table[abs_blk] = s;
                }
            }
        }
        let phys = table.get(blk).copied().unwrap_or(0);
        slots.push((phys as i64) * (block_size as i64) + off as i64);
        block_tables.push(table);
    }
    TreeRowMetadata {
        positions,
        seq_lens,
        slots,
        block_tables,
    }
}

// ============================================================================
// M5 — graphed tree verify host helpers (shape key + indirect-copy args)
// ============================================================================

/// Cap on `(canonical, scratch)` re-seed block pairs the graphed tree
/// verify supports. Sizes the indirect-copy kernel's baked grid AND the
/// per-slot persistent scratch pool. A plan needing more pairs falls back
/// to the eager tree path. Typical steps use 1-4 pairs (1-2 branches ×
/// 1-2 touched blocks); 16 is comfortable headroom within one small meta
/// buffer (1 + 2×16 u32 = 132 bytes).
pub const TREE_RESEED_MAX_PAIRS: usize = 16;

/// Build the device-upload words for the `kv_block_indirect_copy` kernel:
/// `[n_pairs, src0, dst0, src1, dst1, ...]`. Returns `None` when the pair
/// count exceeds `max_pairs` (graph grid capacity) — the caller must fall
/// back to the eager tree path.
pub fn build_reseed_meta(pairs: &[(u32, u32)], max_pairs: usize) -> Option<Vec<u32>> {
    if pairs.len() > max_pairs {
        return None;
    }
    let mut v = Vec::with_capacity(1 + 2 * pairs.len());
    v.push(pairs.len() as u32);
    for &(src, dst) in pairs {
        v.push(src);
        v.push(dst);
    }
    Some(v)
}

/// M5 graph shape key: encodes `spine_len` + the branch row layout
/// (per-branch fork depth + row count) so a captured tree graph is only
/// replayed for byte-identical row structure. Two plans with the same
/// `(K_t, shape_id)` have identical `spine_end`, identical per-row depth
/// layout, and identical branch runs — everything the captured launch
/// sequence bakes; per-step values (positions, block ids, reseed pairs)
/// ride in device buffers uploaded pre-replay.
///
/// Packing: `spine_len` in bits 0..6, then 12 bits per branch
/// (fork_depth in 6, num_rows in 6). Returns `None` when the plan does
/// not fit (>4 branches or any field ≥ 64) — caller falls back to eager.
pub fn tree_shape_id(plan: &TreeVerifyPlan) -> Option<u64> {
    if plan.spine_len == 0 || plan.spine_len >= 64 {
        return None;
    }
    let mut id = plan.spine_len as u64;
    let mut shift = 6u32;
    for b in &plan.branches {
        if shift + 12 > 64 {
            return None; // >4 branches — rare shape, eager fallback
        }
        if b.num_rows == 0 || b.max_depth + 1 < b.num_rows {
            return None;
        }
        let fork_depth = b.max_depth + 1 - b.num_rows;
        if fork_depth >= 64 || b.num_rows >= 64 {
            return None;
        }
        id |= ((fork_depth as u64) << shift) | ((b.num_rows as u64) << (shift + 6));
        shift += 12;
    }
    Some(id)
}

/// Pick the branch cliffs for [`build_free_slots_payload`] from per-row top-2
/// logit margins. Returns the compact spine depths (1-based) whose top1−top2
/// margin is below `margin_thresh`, ASCENDING (shallowest first — the chain
/// dies at the FIRST cliff it reaches, so branching a deeper one the walk never
/// reaches is wasted), capped at `max_branches`.
///
/// `margins[r]` is the top1−top2 margin at spine row `r` (0-based); a fork at
/// depth `r+1` attaches below the spine node at depth `r` (compact slot `r`),
/// mirroring the caterpillar/branch cliff convention in `propose.rs`. Rows 0 and
/// the last row are excluded (row 0 gates the whole chain via the bonus; the
/// last spine node has no room for a meaningful sibling continuation).
pub fn pick_free_slot_cliffs(
    margins: &[f32],
    margin_thresh: f32,
    max_branches: usize,
) -> Vec<usize> {
    let n = margins.len();
    if n < 2 || max_branches == 0 {
        return Vec::new();
    }
    let mut cliffs = Vec::new();
    for r in 1..n - 1 {
        if margins[r] < margin_thresh {
            cliffs.push(r + 1); // cliff_depth is 1-based (r is 0-based row)
            if cliffs.len() >= max_branches {
                break;
            }
        }
    }
    cliffs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(token_id: u32, logprob: f32) -> DraftCandidate {
        DraftCandidate { token_id, logprob }
    }

    fn demo_candidates() -> Vec<Vec<DraftCandidate>> {
        vec![
            vec![cand(101, -0.05), cand(102, -0.80), cand(103, -1.20)],
            vec![cand(201, -0.10), cand(202, -0.35), cand(203, -1.30)],
            vec![cand(301, -0.25), cand(302, -0.40), cand(303, -0.90)],
            vec![cand(401, -0.20), cand(402, -0.55), cand(403, -1.10)],
        ]
    }

    #[test]
    fn build_with_chain_seed_then_branches() {
        let tree = build_ddtree(&demo_candidates(), 8, 3, true, 0, u32::MAX).unwrap();
        // budget=8 → 9 nodes including root.
        assert_eq!(tree.nodes.len(), 9);
        assert_eq!(tree.nodes[0].depth, 0);
        // Chain seed: nodes 1-4 should be the top-1 chain (101, 201, 301, 401).
        let chain_tokens: Vec<u32> = (1..=4).map(|i| tree.nodes[i].token_id).collect();
        assert_eq!(chain_tokens, vec![101, 201, 301, 401]);
        // Depths along chain.
        for (i, depth) in (1..=4).zip(1..=4usize) {
            assert_eq!(tree.nodes[i].depth, depth);
        }
        // Remaining 4 nodes added by best-first heap.
        for node in &tree.nodes[5..] {
            assert!(node.depth >= 1 && node.depth <= 4);
        }
    }

    #[test]
    fn parent_indices_for_verifier_offsets_correctly() {
        let tree = build_ddtree(&demo_candidates(), 8, 3, true, 0, u32::MAX).unwrap();
        let parents = tree.parent_indices_for_verifier();
        assert_eq!(parents.len(), 8); // non-root only
        // First non-root node attaches directly to root → -1 marker.
        assert_eq!(parents[0], -1);
    }

    #[test]
    fn path_token_ids_walks_root_excluded() {
        let tree = build_ddtree(&demo_candidates(), 8, 3, true, 0, u32::MAX).unwrap();
        // node 4 = end of chain.
        let path = tree.path_token_ids(4);
        assert_eq!(path, vec![101, 201, 301, 401]);
        // root.
        assert_eq!(tree.path_token_ids(0), Vec::<u32>::new());
    }

    #[test]
    fn ancestor_indices_includes_self_and_root() {
        let tree = build_ddtree(&demo_candidates(), 8, 3, true, 0, u32::MAX).unwrap();
        let a = tree.ancestor_indices(4, true);
        // Chain 0->1->2->3->4.
        assert!(
            a.contains(&0) && a.contains(&1) && a.contains(&2) && a.contains(&3) && a.contains(&4)
        );
        assert_eq!(a.len(), 5);
        let b = tree.ancestor_indices(4, false);
        assert!(!b.contains(&4));
    }

    #[test]
    fn visibility_mask_is_ancestor_only() {
        let tree = build_ddtree(&demo_candidates(), 8, 3, true, 0, u32::MAX).unwrap();
        let mask = tree.visibility_mask();
        let n = tree.nodes.len();
        assert_eq!(mask.len(), n);
        // Root row only sees root.
        for col in 0..n {
            assert_eq!(mask[0][col], col == 0);
        }
        // Every diagonal entry true (self is its own ancestor).
        for i in 0..n {
            assert!(mask[i][i], "diag at {} should be true", i);
        }
    }

    #[test]
    fn budget_validation() {
        let err = build_ddtree(&demo_candidates(), 0, 3, true, 0, u32::MAX).unwrap_err();
        matches!(err, DDTreeBuildError::InvalidBudget);
    }

    #[test]
    fn top_k_validation() {
        let err = build_ddtree(&demo_candidates(), 8, 0, true, 0, u32::MAX).unwrap_err();
        matches!(err, DDTreeBuildError::InvalidTopK);
    }

    #[test]
    fn empty_depth_rejected() {
        let cs = vec![vec![cand(1, -0.1)], vec![]];
        let err = build_ddtree(&cs, 4, 3, true, 0, u32::MAX).unwrap_err();
        matches!(err, DDTreeBuildError::EmptyDepth(_));
    }

    #[test]
    fn min_root_branches_when_no_chain_seed() {
        let tree = build_ddtree(&demo_candidates(), 8, 3, false, 3, u32::MAX).unwrap();
        // First 3 non-root nodes should each attach to root.
        for i in 1..=3 {
            assert_eq!(tree.nodes[i].parent_index, Some(0));
            assert_eq!(tree.nodes[i].depth, 1);
        }
    }

    #[test]
    fn greedy_walk_accepts_chain_when_target_matches() {
        let tree = build_ddtree(&demo_candidates(), 8, 3, true, 0, u32::MAX).unwrap();
        // Oracle: always pick chain tokens 101, 201, 301, 401.
        let chain = [101u32, 201, 301, 401];
        let oracle = |path: &[u32]| -> u32 {
            let next_depth = path.len();
            if next_depth < chain.len() {
                chain[next_depth]
            } else {
                999 // bonus / unreachable
            }
        };
        let walk = greedy_tree_walk(&tree, oracle);
        assert_eq!(walk.accepted_token_ids, vec![101, 201, 301, 401]);
        assert_eq!(walk.bonus_token_id, 999);
        assert_eq!(walk.output_token_ids(), vec![101, 201, 301, 401, 999]);
    }

    #[test]
    fn greedy_walk_terminates_at_first_mismatch() {
        let tree = build_ddtree(&demo_candidates(), 8, 3, true, 0, u32::MAX).unwrap();
        let oracle = |path: &[u32]| -> u32 {
            match path.len() {
                0 => 101, // matches chain[0]
                1 => 999, // mismatch — stop
                _ => 0,
            }
        };
        let walk = greedy_tree_walk(&tree, oracle);
        assert_eq!(walk.accepted_token_ids, vec![101]);
        assert_eq!(walk.bonus_token_id, 999);
    }

    // ---- TreePayload bridge tests ----

    fn payload(tokens: &[u32], parents: &[i32]) -> TreePayload {
        TreePayload {
            tree_token_ids: tokens.to_vec(),
            parent_indices: parents.to_vec(),
        }
    }

    #[test]
    fn full_parent_ids_prepends_root() {
        let p = payload(&[11, 21, 22, 31], &[-1, 0, 0, 2]);
        let parents = full_parent_ids_from_payload(&p).unwrap();
        assert_eq!(parents, vec![ROOT_PARENT, ROOT_PARENT, 1, 1, 3]);
    }

    // ---- runtime sampler tests ----

    fn req(tree_tokens: &[u32], parents: &[i32]) -> DDTreeRequestRuntime {
        DDTreeRequestRuntime {
            req_id: "test".to_string(),
            tree_token_ids: tree_tokens.to_vec(),
            parent_indices: parents.to_vec(),
        }
    }

    #[test]
    fn sampler_accepts_full_flat_chain() {
        // 4-node flat chain: tokens [10, 20, 30, 40], parents [-1, 0, 1, 2].
        let r = req(&[10, 20, 30, 40], &[-1, 0, 1, 2]);
        // Argmax per row: root → 10, node1 → 20, node2 → 30, node3 → 40, node4 → 99 (bonus).
        let argmax = vec![10u32, 20, 30, 40, 99];
        let s = greedy_sample_ddtree(&r, &argmax).unwrap();
        assert_eq!(s.output_token_ids, vec![10, 20, 30, 40, 99]);
        assert_eq!(s.accepted_compact_indices, vec![1, 2, 3, 4]);
        assert_eq!(s.bonus_parent_compact_index, 4);
    }

    #[test]
    fn sampler_stops_at_root_mismatch() {
        let r = req(&[10, 20, 30], &[-1, 0, 1]);
        let argmax = vec![99u32, 0, 0, 0]; // root prediction misses 10
        let s = greedy_sample_ddtree(&r, &argmax).unwrap();
        assert_eq!(s.output_token_ids, vec![99]); // just the bonus
        assert_eq!(s.accepted_compact_indices, Vec::<usize>::new());
        assert_eq!(s.bonus_parent_compact_index, 0);
    }

    #[test]
    fn sampler_flat_safe_truncates_at_branch_divergence() {
        // Tree:
        //   compact 1: root child A (token 10)
        //   compact 2: root child B (token 11)  — sibling of A
        //   compact 3: child of A (token 20)
        // Flat chain indices: [1, 3]. accept[0]=1 (flat ok), accept[1]=3 (not 2, diverges).
        let r = req(&[10, 11, 20], &[-1, -1, 0]);
        // Argmax: root → 10 (accept node1), node1 → 20 (accept node3 via child of compact 1)
        // node 2 row (compact 2) irrelevant, node 3 row → 999 bonus
        let argmax = vec![10u32, 20, 0, 999];
        let s = greedy_sample_ddtree(&r, &argmax).unwrap();
        // Walk accepted: [10, 20] at compact [1, 3]. flat_prefix_len = 1 (only compact 1).
        // Safe emit: [10 (flat), 20 (now the bonus)].
        assert_eq!(s.output_token_ids, vec![10, 20]);
        assert_eq!(s.accepted_compact_indices, vec![1]);
        assert_eq!(s.bonus_parent_compact_index, 1);
    }

    #[test]
    fn full_sampler_commits_fork_tail() {
        // Single-cliff branch shape:
        //   compact 1: chain token 10 (root child)
        //   compact 2: chain token 20 (child of 1)
        //   compact 3: leaf token 99 (fork off compact 1 — parent = 1)
        //
        // Target greedy: root->10 (compact 1), compact-1 row -> 99 (fork to
        // the leaf at compact 3 instead of the chain child 20).
        let r = req(&[10, 20, 99], &[-1, 0, 0]);
        // argmax rows: [root, after-1, after-2, after-3].
        let argmax = vec![10u32, 99, 0, 777];

        // Flat-safe contract: truncates at the fork — accepts only [1], emits
        // [10, 99] (99 becomes a single bonus), tail token 99's own row is
        // NOT advanced into.
        let flat = greedy_sample_ddtree(&r, &argmax).unwrap();
        assert_eq!(flat.accepted_compact_indices, vec![1]);
        assert_eq!(flat.output_token_ids, vec![10, 99]);

        // Full tree-path commit: accepts the whole walk [1, 3], emits the
        // path tokens [10, 99] AND the fresh bonus 777 read from compact 3.
        let full = greedy_sample_ddtree_full(&r, &argmax).unwrap();
        assert_eq!(full.accepted_compact_indices, vec![1, 3]);
        assert_eq!(full.output_token_ids, vec![10, 99, 777]);
        // Last accepted inter slot is the genuine max compact index (3),
        // which the state commit reads — NOT len-1 = 1.
        assert_eq!(last_accepted_inter_slot(&full.accepted_compact_indices), 3);
    }

    #[test]
    fn full_sampler_matches_flat_when_path_is_contiguous() {
        // When the greedy walk stays on the flat chain, the full sampler and
        // the flat-safe sampler must agree byte-for-byte (no-regress on the
        // common case).
        let r = req(&[10, 20, 30, 40], &[-1, 0, 1, 2]);
        let argmax = vec![10u32, 20, 30, 40, 99];
        let flat = greedy_sample_ddtree(&r, &argmax).unwrap();
        let full = greedy_sample_ddtree_full(&r, &argmax).unwrap();
        assert_eq!(flat.output_token_ids, full.output_token_ids);
        assert_eq!(flat.accepted_compact_indices, full.accepted_compact_indices);
    }

    #[test]
    fn full_sampler_prefers_spine_on_duplicate_token_tie() {
        // DEGEN shape (ATLAS_DFLASH_TREE_DEGEN=1): the fork token EQUALS the
        // spine sibling's token, so the cliff parent has TWO children with
        // the same token id. The walk must resolve the tie to the SPINE
        // child (smallest compact index) — entering the short branch tail
        // instead caps the accept at the branch length (observed live as
        // DEGEN accept 0.74, dist capped at 2 = branch rows).
        //
        //   spine: 10 20 30 40  (compact 1..4)
        //   branch: fork tok 20 off root-child compact 1 (DUPLICATE of
        //   compact 2), tail tok 30 (duplicate of compact 3) → compact 5, 6.
        let r = req(&[10, 20, 30, 40, 20, 30], &[-1, 0, 1, 2, 0, 4]);
        // Target rides the full spine; row 4 (spine tip) yields bonus 777.
        let argmax = vec![10u32, 20, 30, 40, 777, 30, 999];
        let full = greedy_sample_ddtree_full(&r, &argmax).unwrap();
        assert_eq!(
            full.accepted_compact_indices,
            vec![1, 2, 3, 4],
            "duplicate-token tie must walk the full spine, not the branch"
        );
        assert_eq!(full.output_token_ids, vec![10, 20, 30, 40, 777]);
        assert_eq!(full.bonus_parent_compact_index, 4);
    }

    #[test]
    fn greedy_tree_walk_prefers_lowest_index_child_on_duplicate_token() {
        // Same tie-break contract for the DDTree-based walk: two root
        // children with the same token — the earlier (chain/spine) node wins.
        let mut tree = DDTree { nodes: Vec::new() };
        tree.nodes.push(TreeNode {
            index: 0,
            token_id: u32::MAX,
            parent_index: None,
            depth: 0,
            score: 0.0,
        });
        // Chain child (index 1) and duplicate sibling (index 2), then a
        // deeper chain node under index 1.
        tree.nodes.push(TreeNode {
            index: 1,
            token_id: 10,
            parent_index: Some(0),
            depth: 1,
            score: 0.0,
        });
        tree.nodes.push(TreeNode {
            index: 2,
            token_id: 10,
            parent_index: Some(0),
            depth: 1,
            score: 0.0,
        });
        tree.nodes.push(TreeNode {
            index: 3,
            token_id: 20,
            parent_index: Some(1),
            depth: 2,
            score: 0.0,
        });
        let oracle = |path: &[u32]| -> u32 {
            match path.len() {
                0 => 10,
                1 => 20, // only node 1 has this child — reachable iff tie → node 1
                _ => 999,
            }
        };
        let walk = greedy_tree_walk(&tree, oracle);
        assert_eq!(walk.accepted_node_indices, vec![1, 3]);
        assert_eq!(walk.accepted_token_ids, vec![10, 20]);
    }

    #[test]
    fn sampler_wrong_argmax_length_errors() {
        let r = req(&[10], &[-1]);
        let argmax = vec![10u32]; // should be length 2 (1 + num_nodes)
        let err = greedy_sample_ddtree(&r, &argmax).unwrap_err();
        matches!(err, DDTreeBuildError::EmptyCandidates);
    }

    #[test]
    fn adapt_contract_passes_through_when_already_flat() {
        let emitted_tokens = vec![10u32, 20, 30];
        let compact = vec![1usize, 2, 3];
        let (out, safe_compact, bonus_parent) =
            adapt_to_flat_safe_contract(&emitted_tokens, &compact, 99, 3);
        assert_eq!(out, vec![10, 20, 30, 99]);
        assert_eq!(safe_compact, vec![1, 2, 3]);
        assert_eq!(bonus_parent, 3);
    }

    // ── FREE SLOTS branch placement ───────────────────────────────────────

    #[test]
    fn free_slots_no_branches_is_flat_spine() {
        // No branches → plain flat chain, byte-identical to the drafter path.
        let p = build_free_slots_payload(&[10, 20, 30, 40], &[], 8);
        assert_eq!(p.tree_token_ids, vec![10, 20, 30, 40]);
        assert_eq!(p.parent_indices, vec![-1, 0, 1, 2]);
    }

    #[test]
    fn free_slots_single_shallow_fork_layout() {
        // Spine [10,20,30,40]; one fork at cliff_depth 2 (below spine[1]=20),
        // token 99, tail [31,41]. Spine → slots 1..4 (parents -1,0,1,2). Fork at
        // slot 5 attaches to spine slot 1 (cliff_depth-1=1 → payload idx 0).
        // Tail 31→slot6 (parent=fork idx4), 41→slot7 (parent idx5).
        let b = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 99,
            tail: vec![31, 41],
        };
        let p = build_free_slots_payload(&[10, 20, 30, 40], &[b], 16);
        assert_eq!(p.tree_token_ids, vec![10, 20, 30, 40, 99, 31, 41]);
        //                                spine ------------  fork tail--
        assert_eq!(
            p.parent_indices,
            vec![-1, 0, 1, 2, /*fork→spine[0]*/ 0, 4, 5]
        );
    }

    #[test]
    fn free_slots_depth1_fork_attaches_to_root() {
        // cliff_depth 1 → the fork is a sibling of spine[0], attaching to the
        // root (payload parent -1). This is the portfolio-style root fork.
        let b = FreeSlotBranch {
            cliff_depth: 1,
            fork_token: 77,
            tail: vec![88],
        };
        let p = build_free_slots_payload(&[10, 20], &[b], 16);
        assert_eq!(p.tree_token_ids, vec![10, 20, 77, 88]);
        assert_eq!(p.parent_indices, vec![-1, 0, /*fork→root*/ -1, 2]);
    }

    #[test]
    fn free_slots_branches_ordered_shallowest_first() {
        // Two branches given deepest-first; builder must lay the shallower one
        // (cliff_depth 2) before the deeper (cliff_depth 3).
        let deep = FreeSlotBranch {
            cliff_depth: 3,
            fork_token: 300,
            tail: vec![],
        };
        let shallow = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 200,
            tail: vec![],
        };
        let p = build_free_slots_payload(&[1, 2, 3, 4], &[deep, shallow], 16);
        // Spine slots 1..4, then fork@2 (tok 200, parent spine slot1→idx0),
        // then fork@3 (tok 300, parent spine slot2→idx1).
        assert_eq!(p.tree_token_ids, vec![1, 2, 3, 4, 200, 300]);
        assert_eq!(p.parent_indices, vec![-1, 0, 1, 2, 0, 1]);
    }

    #[test]
    fn free_slots_respects_max_nodes_budget() {
        // max_nodes = 5: spine (4 nodes) + fork node (1) = 5, tail dropped.
        let b = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 99,
            tail: vec![31, 41, 51],
        };
        let p = build_free_slots_payload(&[10, 20, 30, 40], &[b], 5);
        assert_eq!(p.tree_token_ids, vec![10, 20, 30, 40, 99]);
        assert_eq!(p.parent_indices, vec![-1, 0, 1, 2, 0]);
        assert!(p.tree_token_ids.len() <= 5);
    }

    #[test]
    fn free_slots_no_room_for_any_branch_is_flat_spine() {
        // Spine already fills the budget → no branch slots → plain flat chain.
        let b = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 99,
            tail: vec![31],
        };
        let p = build_free_slots_payload(&[10, 20, 30, 40], &[b], 4);
        assert_eq!(p.tree_token_ids, vec![10, 20, 30, 40]);
        assert_eq!(p.parent_indices, vec![-1, 0, 1, 2]);
    }

    #[test]
    fn free_slots_out_of_range_cliff_ignored() {
        // cliff_depth 9 > spine_len 4 → filtered out; cliff_depth 0 invalid.
        let bad_deep = FreeSlotBranch {
            cliff_depth: 9,
            fork_token: 1,
            tail: vec![],
        };
        let bad_zero = FreeSlotBranch {
            cliff_depth: 0,
            fork_token: 2,
            tail: vec![],
        };
        let good = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 99,
            tail: vec![],
        };
        let p = build_free_slots_payload(&[10, 20, 30, 40], &[bad_deep, bad_zero, good], 16);
        assert_eq!(p.tree_token_ids, vec![10, 20, 30, 40, 99]);
        assert_eq!(p.parent_indices, vec![-1, 0, 1, 2, 0]);
    }

    #[test]
    fn free_slots_shallow_fork_commits_under_tree_semantics() {
        // Spine [10,20,30], fork at cliff_depth 2 (below spine[1]=20) carrying
        // token 99 and tail [98]. When the target diverges to the fork at the
        // cliff (row 1 argmax = 99), the fork + tail commit — the free-slot WIN.
        //   compact 1: 10 (parent root)
        //   compact 2: 20 (parent 1)
        //   compact 3: 30 (parent 2)
        //   compact 4: 99 fork (parent spine slot1 → payload idx 0 → compact 1)
        //   compact 5: 98 tail (parent 4)
        let b = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 99,
            tail: vec![98],
        };
        let p = build_free_slots_payload(&[10, 20, 30], &[b], 16);
        assert_eq!(p.tree_token_ids, vec![10, 20, 30, 99, 98]);
        assert_eq!(p.parent_indices, vec![-1, 0, 1, 0, 3]);
        let r = req(&p.tree_token_ids, &p.parent_indices);
        // rows: [root, c1, c2, c3, c4, c5]
        // root→10 (accept c1); c1(10)→99 (target picks the FORK, not spine 20);
        // c4(99)→98 (accept the tail); c5(98)→55 bonus.
        let argmax = vec![10u32, 99, 0, 0, 98, 55];
        let s = greedy_sample_ddtree_full(&r, &argmax).unwrap();
        assert_eq!(s.output_token_ids, vec![10, 99, 98, 55]);
        assert_eq!(s.accepted_compact_indices, vec![1, 4, 5]);
        // Last accepted kernel state slot = max compact index.
        assert_eq!(last_accepted_inter_slot(&s.accepted_compact_indices), 5);
    }

    #[test]
    fn free_slots_spine_ride_is_byte_identical_to_flat() {
        // When the target rides the top-1 spine, the committed run is exactly
        // the flat chain — the losslessness anchor. Fork present but never hit.
        let b = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 99,
            tail: vec![98],
        };
        let p = build_free_slots_payload(&[10, 20, 30], &[b], 16);
        let r = req(&p.tree_token_ids, &p.parent_indices);
        // root→10, c1→20 (spine, not fork), c2→30, c3→77 bonus.
        let argmax = vec![10u32, 20, 30, 77, 0, 0];
        let flat = build_free_slots_payload(&[10, 20, 30], &[], 16);
        let rf = req(&flat.tree_token_ids, &flat.parent_indices);
        let argmax_flat = vec![10u32, 20, 30, 77];
        let s = greedy_sample_ddtree_full(&r, &argmax).unwrap();
        let sf = greedy_sample_ddtree_full(&rf, &argmax_flat).unwrap();
        assert_eq!(s.output_token_ids, sf.output_token_ids);
        assert_eq!(s.output_token_ids, vec![10, 20, 30, 77]);
        assert_eq!(s.accepted_compact_indices, vec![1, 2, 3]);
    }

    #[test]
    fn pick_free_slot_cliffs_first_low_margin_ascending() {
        // margins: row0 huge (excluded), row1 low, row2 high, row3 low, last excluded.
        let margins = vec![10.0f32, 0.5, 5.0, 0.4, 6.0, 0.2 /*last, excluded*/];
        let cliffs = pick_free_slot_cliffs(&margins, 1.0, 8);
        // rows 1 and 3 below threshold → cliff depths 2 and 4 (r+1), ascending.
        assert_eq!(cliffs, vec![2, 4]);
    }

    #[test]
    fn pick_free_slot_cliffs_caps_at_max_branches() {
        let margins = vec![10.0f32, 0.1, 0.1, 0.1, 0.1, 9.0];
        let cliffs = pick_free_slot_cliffs(&margins, 1.0, 2);
        assert_eq!(cliffs.len(), 2);
        assert_eq!(cliffs, vec![2, 3]); // first two low-margin rows, ascending
    }

    #[test]
    fn pick_free_slot_cliffs_none_when_all_confident() {
        let margins = vec![10.0f32, 9.0, 8.0, 7.0];
        assert!(pick_free_slot_cliffs(&margins, 1.0, 8).is_empty());
    }

    // ── M2 verify-plan + metadata builder ─────────────────────────────────

    const BS: usize = 16; // block size for the metadata tests
    const GAMMA: usize = 6;

    /// γ=6 spine + one fork@cliff_depth 2 with a 1-token tail — the shape
    /// propose.rs builds under ATLAS_DFLASH_TREE=1.
    fn m2_payload() -> TreePayload {
        let spine = [10u32, 20, 30, 40, 50, 60];
        let b = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 99,
            tail: vec![31],
        };
        build_free_slots_payload(&spine, &[b], 20)
    }

    #[test]
    fn plan_shapes_spine_and_branch_rows() {
        let p = m2_payload();
        assert_eq!(p.tree_token_ids, vec![10, 20, 30, 40, 50, 60, 99, 31]);
        assert_eq!(p.parent_indices, vec![-1, 0, 1, 2, 3, 4, 0, 6]);
        let plan = build_tree_verify_plan(&p, 32, BS).unwrap();
        assert_eq!(plan.num_rows(), 9);
        assert_eq!(plan.spine_len, 6);
        // Depths: bonus 0, spine 1..=6, fork depth 2 (sibling of spine[1]),
        // tail depth 3.
        assert_eq!(plan.row_depths, vec![0, 1, 2, 3, 4, 5, 6, 2, 3]);
        assert_eq!(plan.row_branch[..7], vec![None; 7][..]);
        assert_eq!(plan.row_branch[7], Some(0));
        assert_eq!(plan.row_branch[8], Some(0));
        assert_eq!(plan.branches.len(), 1);
        let b = &plan.branches[0];
        assert_eq!((b.first_row, b.num_rows, b.max_depth), (7, 2, 3));
        assert_eq!(plan.branch_starting_at_row(7), Some(0));
        assert_eq!(plan.branch_starting_at_row(8), None);
        assert_eq!(plan.max_depth(), 6);
    }

    #[test]
    fn plan_rejects_malformed_payloads() {
        // Ancestor after descendant / branch attached to a branch that is
        // not its immediate predecessor → reject (flat fallback).
        let bad = payload(&[1, 2, 3, 4, 5], &[-1, 0, 0, 3, 2]);
        assert!(build_tree_verify_plan(&bad, 0, BS).is_none());
        // Parent index >= spine for a branch head.
        let bad2 = payload(&[1, 2, 3], &[-1, 0, 5]);
        assert!(build_tree_verify_plan(&bad2, 0, BS).is_none());
        // Empty payload / zero block size.
        assert!(build_tree_verify_plan(&payload(&[], &[]), 0, BS).is_none());
        assert!(build_tree_verify_plan(&m2_payload(), 0, 0).is_none());
        // No flat anchor (first node not root-attached-flat).
        let bad3 = payload(&[1, 2], &[1, -1]);
        assert!(build_tree_verify_plan(&bad3, 0, BS).is_none());
    }

    #[test]
    fn plan_flat_payload_has_no_branches() {
        let p = build_free_slots_payload(&[10, 20, 30], &[], 8);
        let plan = build_tree_verify_plan(&p, 100, BS).unwrap();
        assert_eq!(plan.spine_len, 3);
        assert!(plan.branches.is_empty());
        assert_eq!(plan.row_depths, vec![0, 1, 2, 3]);
    }

    /// M2b split-range invariant: the batched tree forward cache-writes rows
    /// `[0, spine_len+1)` (canonical) then rows `[spine_len+1, K_t)`
    /// (scratch) — valid ONLY if every accepted plan puts all bonus+spine
    /// rows strictly before all branch rows. Pin that shape, including for
    /// multi-branch payloads.
    #[test]
    fn plan_branch_rows_follow_spine_rows_contiguously() {
        let spine = [10u32, 20, 30, 40, 50, 60];
        let b1 = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 99,
            tail: vec![31, 32],
        };
        let b2 = FreeSlotBranch {
            cliff_depth: 4,
            fork_token: 88,
            tail: vec![51],
        };
        let p = build_free_slots_payload(&spine, &[b1, b2], 20);
        let plan = build_tree_verify_plan(&p, 32, BS).unwrap();
        let spine_end = plan.spine_len + 1;
        // Rows [0, spine_end): bonus + spine, never branch-mapped.
        for t in 0..spine_end {
            assert_eq!(plan.row_branch[t], None, "row {t} in spine range");
        }
        // Rows [spine_end, K_t): all branch-mapped, branches contiguous.
        for t in spine_end..plan.num_rows() {
            assert!(plan.row_branch[t].is_some(), "row {t} in branch range");
        }
        for b in &plan.branches {
            assert!(b.first_row >= spine_end);
            assert!(b.first_row + b.num_rows <= plan.num_rows());
        }
        assert_eq!(
            plan.branches.iter().map(|b| b.num_rows).sum::<usize>(),
            plan.num_rows() - spine_end,
        );
    }

    /// Shared checker: positions/seq_lens are base+depth (+1); spine rows
    /// keep the canonical table; branch rows substitute exactly the touched
    /// range with scratch; every row's write slot routes through its own
    /// table.
    fn check_metadata(base: usize) {
        let p = m2_payload();
        let plan = build_tree_verify_plan(&p, base, BS).unwrap();
        // Canonical physical blocks 100.. for as many blocks as the deepest
        // position needs.
        let n_blocks = (base + plan.max_depth()) / BS + 1;
        let canonical: Vec<u32> = (0..n_blocks as u32).map(|i| 100 + i).collect();
        // One branch; distinct scratch ids 200.. for its touched range.
        let b = &plan.branches[0];
        assert_eq!(b.touched_lo, base / BS);
        assert_eq!(b.touched_hi, (base + b.max_depth) / BS);
        let scratch: Vec<(usize, u32)> = (b.touched_lo..=b.touched_hi)
            .enumerate()
            .map(|(i, ab)| (ab, 200 + i as u32))
            .collect();
        let md = build_tree_row_metadata(&plan, base, BS, &canonical, &[scratch.clone()]);
        assert_eq!(md.positions.len(), plan.num_rows());
        for t in 0..plan.num_rows() {
            let pos = base + plan.row_depths[t];
            assert_eq!(md.positions[t], pos as u32, "row {t} position");
            assert_eq!(md.seq_lens[t], (pos + 1) as i32, "row {t} seq_len");
            let blk = pos / BS;
            let off = pos % BS;
            let expected_table: Vec<u32> = if plan.row_branch[t].is_some() {
                let mut tab = canonical.clone();
                for &(ab, s) in &scratch {
                    if ab < tab.len() {
                        tab[ab] = s;
                    }
                }
                tab
            } else {
                canonical.clone()
            };
            assert_eq!(md.block_tables[t], expected_table, "row {t} table");
            let phys = expected_table[blk] as i64;
            assert_eq!(md.slots[t], phys * BS as i64 + off as i64, "row {t} slot");
            // Branch rows must never write into a canonical block within
            // the touched range; spine rows must never write scratch.
            if plan.row_branch[t].is_some() {
                assert!(
                    md.slots[t] / (BS as i64) >= 200,
                    "row {t} (branch) writes canonical block"
                );
            } else {
                assert!(
                    md.slots[t] / (BS as i64) < 200,
                    "row {t} (spine) writes scratch block"
                );
            }
        }
    }

    #[test]
    fn metadata_at_block_start() {
        // base % bs == 0 — whole verify inside fresh block territory.
        check_metadata(2 * BS); // base = 32
    }

    #[test]
    fn metadata_at_block_end() {
        // base % bs == bs-1 — row 0 is the block's last slot; every deeper
        // row crosses into the next block(s).
        check_metadata(3 * BS - 1); // base = 47
    }

    #[test]
    fn metadata_straddling_gamma_tail() {
        // base % bs == bs - γ — the spine itself crosses the boundary.
        check_metadata(2 * BS - GAMMA + BS); // base = 42, 42 % 16 = 10 = bs-γ
    }

    /// M4 shape: TWO branches, each with a TAIL-2 re-rooted continuation,
    /// through BOTH builders — the exact payload propose.rs emits under
    /// ATLAS_DFLASH_TREE_BRANCHES=2 ATLAS_DFLASH_TREE_TAIL=2.
    #[test]
    fn free_slots_two_branches_tail2_through_both_builders() {
        // Spine [10,20,30,40,50] (γ=5). Branch A: cliff 2 (sibling of 20),
        // fork 99, tail = re-rooted top-1 drafts [30, 40]. Branch B: cliff 4
        // (sibling of 40), fork 88, tail [50, 55].
        let spine = [10u32, 20, 30, 40, 50];
        let a = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 99,
            tail: vec![30, 40],
        };
        let b = FreeSlotBranch {
            cliff_depth: 4,
            fork_token: 88,
            tail: vec![50, 55],
        };
        // Budget = spine(5) + 2×(1 fork + 2 tail) = 11 nodes → K_t 12 ≤ 20.
        let p = build_free_slots_payload(&spine, &[b.clone(), a.clone()], 11);
        // Shallowest-first: A (cliff 2) laid before B (cliff 4) despite the
        // reversed input order. Fork A parent = spine idx 0; fork B = idx 2.
        assert_eq!(
            p.tree_token_ids,
            vec![10, 20, 30, 40, 50, 99, 30, 40, 88, 50, 55]
        );
        assert_eq!(p.parent_indices, vec![-1, 0, 1, 2, 3, 0, 5, 6, 2, 8, 9]);

        // Plan builder accepts the shape: spine 5, two 3-row branches.
        let plan = build_tree_verify_plan(&p, 32, BS).unwrap();
        assert_eq!(plan.num_rows(), 12);
        assert_eq!(plan.spine_len, 5);
        // Depths: bonus 0, spine 1..=5, A fork 2 + tail 3,4; B fork 4 + tail 5,6.
        assert_eq!(plan.row_depths, vec![0, 1, 2, 3, 4, 5, 2, 3, 4, 4, 5, 6]);
        assert_eq!(plan.branches.len(), 2);
        assert_eq!(
            (
                plan.branches[0].first_row,
                plan.branches[0].num_rows,
                plan.branches[0].max_depth
            ),
            (6, 3, 4)
        );
        assert_eq!(
            (
                plan.branches[1].first_row,
                plan.branches[1].num_rows,
                plan.branches[1].max_depth
            ),
            (9, 3, 6)
        );
        assert_eq!(plan.row_branch[6..9], [Some(0); 3][..]);
        assert_eq!(plan.row_branch[9..12], [Some(1); 3][..]);

        // K_t arena squeeze (budget 9 nodes): A laid whole (fork+tail-2),
        // B's fork laid, tail truncated to fit — no partial-branch overflow.
        let squeezed = build_free_slots_payload(&spine, &[a, b], 9);
        assert_eq!(
            squeezed.tree_token_ids,
            vec![10, 20, 30, 40, 50, 99, 30, 40, 88]
        );
        assert_eq!(squeezed.parent_indices, vec![-1, 0, 1, 2, 3, 0, 5, 6, 2]);
        let plan2 = build_tree_verify_plan(&squeezed, 32, BS).unwrap();
        assert_eq!(plan2.branches.len(), 2);
        assert_eq!(plan2.branches[1].num_rows, 1); // bare fork leaf
    }

    #[test]
    fn metadata_sibling_branches_get_distinct_tables() {
        // Two branches (cliff depths 1 and 2) — each row's table must use
        // ITS OWN branch's scratch, never the sibling's.
        let spine = [10u32, 20, 30, 40];
        let b1 = FreeSlotBranch {
            cliff_depth: 1,
            fork_token: 77,
            tail: vec![],
        };
        let b2 = FreeSlotBranch {
            cliff_depth: 2,
            fork_token: 88,
            tail: vec![],
        };
        let p = build_free_slots_payload(&spine, &[b1, b2], 20);
        let base = 0usize;
        let plan = build_tree_verify_plan(&p, base, BS).unwrap();
        assert_eq!(plan.branches.len(), 2);
        let canonical = vec![100u32];
        let scratch = vec![vec![(0usize, 200u32)], vec![(0usize, 300u32)]];
        let md = build_tree_row_metadata(&plan, base, BS, &canonical, &scratch);
        let r1 = plan.branches[0].first_row;
        let r2 = plan.branches[1].first_row;
        assert_eq!(md.block_tables[r1], vec![200]);
        assert_eq!(md.block_tables[r2], vec![300]);
        // Spine rows stay canonical.
        assert_eq!(md.block_tables[1], vec![100]);
    }
}

#[cfg(test)]
mod m5_graph_tests {
    use super::*;

    fn payload(tokens: &[u32], parents: &[i32]) -> TreePayload {
        TreePayload {
            tree_token_ids: tokens.to_vec(),
            parent_indices: parents.to_vec(),
        }
    }

    // ---- build_reseed_meta ----

    #[test]
    fn reseed_meta_layout_is_count_then_pairs() {
        let meta = build_reseed_meta(&[(7, 100), (8, 101)], TREE_RESEED_MAX_PAIRS).unwrap();
        assert_eq!(meta, vec![2, 7, 100, 8, 101]);
    }

    #[test]
    fn reseed_meta_empty_pairs_yields_zero_count() {
        let meta = build_reseed_meta(&[], TREE_RESEED_MAX_PAIRS).unwrap();
        assert_eq!(meta, vec![0]);
    }

    #[test]
    fn reseed_meta_rejects_overflow() {
        let pairs: Vec<(u32, u32)> = (0..5).map(|i| (i, i + 100)).collect();
        assert!(build_reseed_meta(&pairs, 4).is_none());
        assert!(build_reseed_meta(&pairs, 5).is_some());
    }

    // ---- tree_shape_id ----

    #[test]
    fn shape_id_distinguishes_spine_len() {
        // Same K_t = 7, different spine/branch split.
        let a = build_tree_verify_plan(
            &payload(&[10, 20, 30, 40, 99, 31], &[-1, 0, 1, 2, 0, 4]),
            0,
            16,
        )
        .unwrap();
        let b = build_tree_verify_plan(
            &payload(&[10, 20, 30, 99, 31, 32], &[-1, 0, 1, 0, 3, 4]),
            0,
            16,
        )
        .unwrap();
        assert_eq!(a.spine_len, 4);
        assert_eq!(b.spine_len, 3);
        assert_ne!(tree_shape_id(&a).unwrap(), tree_shape_id(&b).unwrap());
    }

    #[test]
    fn shape_id_distinguishes_fork_depth() {
        // Identical spine + branch length; fork attaches at depth 2 vs 3.
        let a = build_tree_verify_plan(
            &payload(&[10, 20, 30, 40, 99, 31], &[-1, 0, 1, 2, 0, 4]),
            0,
            16,
        )
        .unwrap();
        let b = build_tree_verify_plan(
            &payload(&[10, 20, 30, 40, 99, 31], &[-1, 0, 1, 2, 1, 4]),
            0,
            16,
        )
        .unwrap();
        assert_ne!(tree_shape_id(&a).unwrap(), tree_shape_id(&b).unwrap());
    }

    #[test]
    fn shape_id_stable_across_base_positions() {
        // The SAME payload at different base positions (different touched
        // block ranges) must map to the SAME shape id — block ids are
        // per-replay device data, not graph structure.
        let p = payload(&[10, 20, 30, 40, 99, 31], &[-1, 0, 1, 2, 0, 4]);
        let a = build_tree_verify_plan(&p, 0, 16).unwrap();
        let b = build_tree_verify_plan(&p, 1000, 16).unwrap();
        assert_eq!(tree_shape_id(&a).unwrap(), tree_shape_id(&b).unwrap());
    }

    #[test]
    fn shape_id_two_branch_layout_roundtrips_uniquely() {
        // Two branches vs one branch with the same total rows.
        let two = build_tree_verify_plan(
            &payload(
                &[10, 20, 30, 40, 99, 55, 88, 66],
                &[-1, 0, 1, 2, 0, 4, 1, 6],
            ),
            0,
            16,
        )
        .unwrap();
        let one = build_tree_verify_plan(
            &payload(
                &[10, 20, 30, 40, 99, 55, 66, 77],
                &[-1, 0, 1, 2, 0, 4, 5, 6],
            ),
            0,
            16,
        )
        .unwrap();
        assert_eq!(two.num_rows(), one.num_rows());
        assert_ne!(tree_shape_id(&two).unwrap(), tree_shape_id(&one).unwrap());
    }

    #[test]
    fn shape_id_rejects_more_than_four_branches() {
        // Spine of 1 + five bare forks off the root — 5 branches exceed the
        // 4-branch packing budget (6 + 5*12 = 66 > 64 bits).
        let p = payload(&[10, 91, 92, 93, 94, 95], &[-1, -1, -1, -1, -1, -1]);
        let plan = build_tree_verify_plan(&p, 0, 16).unwrap();
        assert_eq!(plan.branches.len(), 5);
        assert!(tree_shape_id(&plan).is_none());
    }
}
