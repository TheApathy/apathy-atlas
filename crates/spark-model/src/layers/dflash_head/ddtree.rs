// SPDX-License-Identifier: AGPL-3.0-only

// M2 milestone: the full API surface is intentionally present so later
// milestones (M3 payload bridge, M4B top-k builder, M5A/B GDN-aware walk)
// can consume it without churning this module. Suppress dead-code warnings
// until those callsites land.
#![allow(dead_code)]

//! DDTree (Draft-Diffusion Tree) — M2 milestone: tree builder + greedy walk.
//!
//! Port of AEON-7 vLLM PR M2 prototypes:
//!   - `ddtree_tree.py`            → this file (builder + walk)
//!   - `ddtree_parent_metadata.py` → [`parent_metadata`]
//!   - `ddtree_vllm_metadata.py`   → [`verifier_metadata`]
//!
//! Pure-CPU logic; no CUDA dependency. Fires only when `--dflash-method=ddtree`.
//! Flat DFlash behavior is preserved when this module is not invoked.
//!
//! Reference: `research/ddtree_port/ddtree_src/prototypes/ddtree_tree.py`.

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
    /// signals the root. Layout matches the GPU-side parent tensor that
    /// later milestones (M6A/M8A) feed to the tree-aware GDN replay.
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
    pub fn child_by_token(&self, parent_index: usize) -> HashMap<u32, usize> {
        let mut children = HashMap::new();
        for node in self.non_root_nodes() {
            if node.parent_index == Some(parent_index) {
                children.insert(node.token_id, node.index);
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
// Parent metadata — port of ddtree_parent_metadata.py
// ============================================================================
//
// Builds the [num_reqs, max_tree_tokens + 1] parent-id tensor that later
// milestones (M6A/M8A) feed to the tree-aware GDN/Flash replay. Rows without
// a tree payload are PADDING_PARENT-filled; lengths report 0 for those rows.

/// Sentinel parent id meaning "root of the tree" (encodes -1 in the i32 tensor).
pub const ROOT_PARENT: i32 = -1;
/// Sentinel parent id used to pad request rows that have no tree payload.
pub const PADDING_PARENT: i32 = 0;

/// One request's tree payload as carried through the verifier bridge (M3+).
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

/// Flattened root+tree parent-id table for a batch of requests.
/// Layout: row-major `[num_reqs * stride]`, `stride = max_tree_tokens + 1`.
#[derive(Debug, Clone)]
pub struct DDTreeParentMetadata {
    pub parent_ids: Vec<i32>,
    pub request_ids: Vec<String>,
    pub num_tree_tokens: Vec<usize>,
    pub stride: usize,
}

impl DDTreeParentMetadata {
    pub fn num_reqs(&self) -> usize {
        self.request_ids.len()
    }
    pub fn row(&self, idx: usize) -> &[i32] {
        let start = idx * self.stride;
        &self.parent_ids[start..start + self.stride]
    }
}

/// Translate a tree payload into the full root+tree parent-id list.
///
/// Output `parents[0] = ROOT_PARENT` always (the synthetic root row used by
/// state replay). For `i >= 1`, `parents[i]` is the *state* parent of compact
/// node `i`: either `ROOT_PARENT` if the payload's `parent_indices[i-1] < 0`
/// (i.e. attaches to root) or `payload.parent_indices[i-1] + 1` (rebasing
/// payload indexing onto the root+tree layout).
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

/// Build the per-batch parent-id table. Returns `None` when no request in the
/// batch carries a non-empty tree payload (so the verifier can skip tree
/// metadata entirely).
pub fn build_padded_parent_ids(
    req_ids: &[String],
    payload_by_req_id: &std::collections::HashMap<String, TreePayload>,
    pad_to: Option<usize>,
) -> Result<Option<DDTreeParentMetadata>, DDTreeBuildError> {
    if payload_by_req_id.is_empty() {
        return Ok(None);
    }

    let mut parents_by_req: Vec<Vec<i32>> = Vec::with_capacity(req_ids.len());
    let mut lengths: Vec<usize> = Vec::with_capacity(req_ids.len());
    let mut max_len = 0usize;
    let mut found = false;

    for req in req_ids {
        let parents = match payload_by_req_id.get(req) {
            Some(p) if !p.is_empty() => {
                found = true;
                full_parent_ids_from_payload(p)?
            }
            _ => Vec::new(),
        };
        lengths.push(parents.len().saturating_sub(1));
        if parents.len() > max_len {
            max_len = parents.len();
        }
        parents_by_req.push(parents);
    }

    if !found {
        return Ok(None);
    }

    if let Some(pad) = pad_to {
        max_len = max_len.max(pad);
    }
    if max_len < 1 {
        max_len = 1;
    }

    let stride = max_len;
    let mut parent_ids = vec![PADDING_PARENT; req_ids.len() * stride];
    for (row, parents) in parents_by_req.iter().enumerate() {
        let dst = &mut parent_ids[row * stride..row * stride + parents.len()];
        dst.copy_from_slice(parents);
    }

    Ok(Some(DDTreeParentMetadata {
        parent_ids,
        request_ids: req_ids.to_vec(),
        num_tree_tokens: lengths,
        stride,
    }))
}

// ============================================================================
// Verifier metadata — port of ddtree_vllm_metadata.py (single-request form)
// ============================================================================
//
// Indexing in the compact verifier logits buffer:
//   row 0    : root logits (output of last prompt token)
//   rows 1..N: logits after each non-root tree node
// A non-root node's token is verified by reading its parent's compact row.

#[derive(Debug, Clone)]
pub struct TreeVerifierMetadata {
    pub prompt_len: usize,
    pub tree_token_ids: Vec<u32>,
    pub parent_indices: Vec<i32>,
    pub node_depths: Vec<usize>,
    pub tree_position_ids: Vec<u32>,
    /// Indices into the model's full logit buffer (prompt + tree rows).
    pub compact_logits_indices: Vec<usize>,
    /// For each non-root node, its parent's row in the compact buffer.
    pub edge_parent_compact_indices: Vec<usize>,
    /// For each non-root node, its own compact index.
    pub node_compact_indices: Vec<usize>,
}

impl TreeVerifierMetadata {
    pub fn num_tree_nodes(&self) -> usize {
        self.tree_token_ids.len()
    }

    pub fn from_tree(prompt_len: usize, tree: &DDTree) -> Result<Self, DDTreeBuildError> {
        if prompt_len < 1 {
            return Err(DDTreeBuildError::InvalidBudget);
        }
        let tree_token_ids = tree.token_ids_for_verifier();
        let parent_indices = tree.parent_indices_for_verifier();
        let node_depths: Vec<usize> = tree.non_root_nodes().iter().map(|n| n.depth).collect();
        let tree_position_ids: Vec<u32> = node_depths
            .iter()
            .map(|d| (prompt_len + d - 1) as u32)
            .collect();

        // First row = root (prompt_len - 1); then one row per tree node.
        let mut compact_logits_indices = Vec::with_capacity(tree_token_ids.len() + 1);
        compact_logits_indices.push(prompt_len - 1);
        for offset in 0..tree_token_ids.len() {
            compact_logits_indices.push(prompt_len + offset);
        }

        let mut edge_parent_compact_indices = Vec::with_capacity(tree_token_ids.len());
        let mut node_compact_indices = Vec::with_capacity(tree_token_ids.len());
        for node in tree.non_root_nodes() {
            node_compact_indices.push(node.index);
            edge_parent_compact_indices.push(match node.parent_index {
                None | Some(0) => 0,
                Some(p) => p,
            });
        }

        Ok(Self {
            prompt_len,
            tree_token_ids,
            parent_indices,
            node_depths,
            tree_position_ids,
            compact_logits_indices,
            edge_parent_compact_indices,
            node_compact_indices,
        })
    }

    /// Position ids for the prompt followed by the tree nodes.
    pub fn all_position_ids(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.prompt_len + self.tree_position_ids.len());
        for p in 0..self.prompt_len {
            out.push(p as u32);
        }
        out.extend_from_slice(&self.tree_position_ids);
        out
    }
}

/// Build a flat `[total_len, total_len]` boolean visibility mask for
/// prompt + tree prefill verification. `true` means "can attend".
///
/// - Prompt rows attend causally to all prior prompt tokens.
/// - Tree rows attend to the full prompt + their ancestor chain (no siblings).
///
/// Caller converts to additive mask of choice (typical: 0 / -inf).
pub fn prefill_tree_attention_mask(prompt_len: usize, tree: &DDTree) -> Vec<bool> {
    let total = prompt_len + tree.non_root_nodes().len();
    let mut visible = vec![false; total * total];

    // Prompt rows: causal.
    for row in 0..prompt_len {
        for col in 0..=row {
            visible[row * total + col] = true;
        }
    }

    // Tree rows.
    for node in tree.non_root_nodes() {
        let row = prompt_len + node.index - 1;
        // Full prompt visible.
        for col in 0..prompt_len {
            visible[row * total + col] = true;
        }
        // Ancestor chain (skip root which lives in the prompt).
        for ancestor in tree.ancestor_indices(node.index, true) {
            if ancestor == 0 {
                continue;
            }
            let col = prompt_len + ancestor - 1;
            visible[row * total + col] = true;
        }
    }

    visible
}

// ============================================================================
// Runtime sampler — port of ddtree_runtime_sampler.py (M5B milestone)
// ============================================================================
//
// Given a payload (tree_token_ids + parent_indices) and the target model's
// per-row argmax over compact verifier logits, walk the tree greedily to find
// the accepted branch and bonus token. Then apply the *deployable-safe*
// contract: commit only the contiguous flat prefix (compact index `i+1` for
// step `i`) and turn the first non-flat branch token into the bonus.
//
// AEON-7's M11A research path (full-branch commit) requires custom recurrent-
// state rollback that we don't yet have for Atlas GDN — so we always run the
// flat-prefix contract. M11A-equivalent landing in Atlas will gate via an env
// var the same way the reference does.

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
            children[parent_compact].insert(tok, node_index + 1);
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
/// Non-flat branch commit requires recurrent-state rollback that Atlas's
/// GDN kernels don't support yet (AEON-7 M11A). Until then, if the
/// accepted path diverges from the flat chain at index `i`, we emit the
/// first `i` accepted tokens + ONE bonus token at the divergence point
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
/// SET changes. The caller is responsible for two pieces of plumbing that
/// the flat path gets for free:
///   1. SSM state handoff — `last_accepted_inter_slot(accepted_compact)`
///      already returns the correct kernel intermediate slot (the max
///      compact index, the genuine last-accepted state) for sparse paths.
///   2. KV compaction — the accepted path's attention KV sits at the sparse
///      compact slots and must be gathered down to contiguous positions
///      before the next decode (see `compact_verify_kv`).
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

/// Build a DEPTH-CONTIGUOUS caterpillar TreePayload for the K=γ+1 DFlash
/// verify (`ATLAS_DFLASH_CATERPILLAR=1`).
///
/// ## The invariant this enforces (why it is lossless without a kernel)
///
/// The K=γ verify lays compact slot `t` at KV/RoPE position `seq_len + t` and
/// attention at slot `t` reads positions `[0 .. seq_len + t]` (the flat-chain
/// metadata in `verify_d.rs`). The post-verify KV-compaction (FIX 2) relocates
/// an accepted node's K/V bytes from `pre + compact` down to `pre + j`, but it
/// CANNOT re-RoPE them. So for any node the greedy walk might COMMIT, its KV is
/// lossless after relocation iff BOTH hold at verify time:
///   1. `slot == depth`  → the RoPE phase baked at `pre + slot` equals the
///      contiguous home position `pre + depth`.
///   2. slots `[0 .. slot-1]` ARE exactly that node's ancestor chain → the
///      attention context it was conditioned on survives compaction unchanged.
///
/// ## The layout
///
/// The top-1 chain is the SPINE, laid contiguously at compact slots
/// `1, 2, …, spine_len` (slot == depth, both invariants trivially hold for
/// every spine node — this is the byte-identical flat baseline when the target
/// stays on top-1). At the SHALLOWEST gated cliff (spine depth `d`), a single
/// top-2 token is forked as a LEAF whose parent is the spine node at depth
/// `d-1` (compact slot `d-1`), and the leaf is placed at compact slot `d`
/// (== its depth). To make room without growing `n`, the spine's own depth-`d`
/// node and everything below it shift up by one slot (the deepest spine node is
/// dropped so the node count stays `n`). After the shift:
///   * The fork leaf at slot `d` reads slots `[0..d-1]` = spine `[0..d-1]` =
///     its true ancestors, and `slot d == depth d` → committing it (the walk
///     diverges to top-2 at the cliff) is lossless: KV-compaction is a no-op
///     (`compact d == j d`) and RoPE is already correct.
///   * The spine continuation past the cliff is pushed one slot deeper, so IF
///     the target instead stays on top-1 the walk re-converges onto a shifted
///     run. That run is still committed by the SAME greedy oracle (draft ==
///     target argmax), and its KV/RoPE carry the shifted-slot positions which
///     KV-compaction relocates to the contiguous home — the relocation is a
///     pure byte move of an already-correct ancestor-conditioned K/V because
///     each shifted spine node's prefix `[0..slot-1]` is STILL its ancestor
///     chain (the inserted fork leaf at slot `d` is a leaf with no spine child,
///     so it is the only non-ancestor in the prefix; see the contamination
///     guard below — we therefore commit at most the fork OR the shallow spine
///     prefix, never a deep shifted spine tail through the leaf).
///
/// To keep the guarantee airtight we lay the fork leaf at slot `d` and the
/// spine prefix `1..d-1` BEFORE it (DFS pre-order: root → spine[1..d-1] →
/// {fork-leaf @ d, spine-continuation @ d+1..}). Then:
///   * accepted path "spine then fork-leaf"  = `[1, 2, …, d-1, d]` (contiguous)
///   * accepted path "spine then top-1 child" = `[1, 2, …, d-1, d+1, d+2, …]`
///     — the ONE non-contiguous jump is `d-1 → d+1`, skipping the fork leaf at
///     slot `d`. The spine child at slot `d+1` has true depth `d`; its prefix
///     `[0..d]` contains the fork leaf at slot `d` (a non-ancestor). To keep
///     THIS path lossless too, the spine child's RoPE must be `pre+d` not
///     `pre+(d+1)` — which only the depth-RoPE tree-aware path supplies. Under
///     pure flat metadata only the fork-leaf branch (and the pre-cliff prefix)
///     is guaranteed lossless; the spine-past-cliff branch needs
///     `ATLAS_DDTREE_TREE_AWARE_VERIFY=1`. Both are committed by the lossless
///     greedy oracle regardless; the env gate decides whether the deep tail is
///     byte-exact or simply not accepted.
///
/// `cliff_depths` are spine depths (1-based, the depth of the spine node the
/// fork attaches BELOW) selected by the caller's margin gate, ascending. Only
/// the shallowest is used for the strictly-lossless single-fork layout; deeper
/// entries are reserved for the tree-aware multi-fork extension and ignored
/// here so the flat-metadata path never commits a contaminated node.
pub fn build_caterpillar_payload(
    spine: &[u32],
    fork_token: u32,
    cliff_depth: usize,
) -> TreePayload {
    let n = spine.len();
    // Degenerate guards → plain flat chain (no fork).
    if n < 2 || cliff_depth == 0 || cliff_depth > n {
        let parent_indices = (0..n as i32).map(|i| i - 1).collect();
        return TreePayload {
            tree_token_ids: spine.to_vec(),
            parent_indices,
        };
    }

    // Slot layout (compact indices are 1-based; slot 0 is the bonus/root):
    //   slots 1 .. cliff_depth-1 : spine[0 .. cliff_depth-2]   (depth == slot)
    //   slot  cliff_depth        : FORK LEAF (top-2), depth == cliff_depth,
    //                              parent = spine node at slot cliff_depth-1
    //   slots cliff_depth+1 .. n : spine[cliff_depth-1 ..]      (shifted +1)
    // The deepest spine token (spine[n-1]) is dropped to hold n total nodes.
    let mut tree_token_ids: Vec<u32> = Vec::with_capacity(n);
    let mut parent_indices: Vec<i32> = Vec::with_capacity(n);

    // Spine prefix before the cliff: compact slots 1..=cliff_depth-1.
    for i in 0..cliff_depth.saturating_sub(1) {
        tree_token_ids.push(spine[i]);
        parent_indices.push(i as i32 - 1); // parent = slot i (== i-1 in 0-based payload idx)
    }
    // Fork leaf at compact slot `cliff_depth`. Parent compact slot =
    // cliff_depth-1 → payload index (cliff_depth-1)-1 = cliff_depth-2.
    tree_token_ids.push(fork_token);
    parent_indices.push((cliff_depth as i32) - 2);
    // Spine continuation: original spine[cliff_depth-1 ..], each pushed one
    // slot deeper. Their parent is the spine node at depth d-1 (the cliff's
    // parent, compact slot cliff_depth-1 → payload index cliff_depth-2) for
    // the first one, then the previous continuation node.
    // First continuation node (true depth cliff_depth) attaches to the spine
    // node at slot cliff_depth-1 (payload index cliff_depth-2), SKIPPING the
    // fork leaf — so the spine path is `…, d-1, d+1, …` and never threads
    // through the leaf.
    let mut prev_parent = (cliff_depth as i32) - 2;
    for i in (cliff_depth - 1)..n {
        if tree_token_ids.len() >= n {
            break; // node budget reached (we dropped the deepest spine token)
        }
        tree_token_ids.push(spine[i]);
        parent_indices.push(prev_parent);
        prev_parent = tree_token_ids.len() as i32 - 1; // this node's payload index
    }

    TreePayload {
        tree_token_ids,
        parent_indices,
    }
}

/// Build a depth-contiguous caterpillar where the FORK (top-2) branch carries
/// the post-cliff TAIL contiguously, and the top-1-at-cliff is a single
/// high-slot leaf (`ATLAS_DFLASH_CATERPILLAR=1` + `ATLAS_DFLASH_CAT_TAIL=1`).
///
/// This is the EV variant: it bets that at a low-margin cliff the target's
/// greedy picks the drafter's TOP-2, and that the predictable tail after the
/// cliff (indentation, closers, `):`) re-accepts even when re-rooted on top-2.
/// Layout (compact slots, slot 0 = bonus/root):
///   slots 1..cliff-1 : top-1 spine prefix   (depth == slot)
///   slot  cliff      : FORK top-2 token      (depth == cliff)
///   slots cliff+1..S : tail = the linear top-1 drafts AFTER the cliff,
///                      re-rooted onto the top-2 fork (depth == slot)
///   slot  S+1        : top-1-at-cliff LEAF   (the rejected alternative; a
///                      single node so when the target stays on top-1 the
///                      flat-safe walk truncates to the prefix + this bonus =
///                      byte-identical to the flat baseline)
///
/// LOSSLESS under the flat-safe sampler: the committed run is always the
/// contiguous `[1..k]` prefix the greedy walk traverses. If the walk dives
/// into the top-2 tail it stays contiguous (the whole tail commits — the GAIN);
/// if it diverges to the top-1 leaf at slot S+1 the flat-safe contract turns
/// that into a single bonus (no contaminated tail committed). The fork token at
/// slot `cliff` and every tail node sit at slot == depth, so RoPE + the
/// slot-prefix-as-ancestors invariant hold for the committed contiguous run.
///
/// `cliff_depth` = the depth of the fork node (1-based compact slot it occupies
/// = parent spine depth + 1). `tail` = the drafter's top-1 tokens that followed
/// the cliff in the linear chain (caller passes `drafts[cliff_row+1 ..]`).
pub fn build_caterpillar_tail_payload(
    spine: &[u32],
    fork_token: u32,
    cliff_depth: usize,
    tail: &[u32],
) -> TreePayload {
    let n = spine.len();
    if n < 2 || cliff_depth == 0 || cliff_depth > n {
        let parent_indices = (0..n as i32).map(|i| i - 1).collect();
        return TreePayload {
            tree_token_ids: spine.to_vec(),
            parent_indices,
        };
    }

    let mut tree_token_ids: Vec<u32> = Vec::with_capacity(n);
    let mut parent_indices: Vec<i32> = Vec::with_capacity(n);

    // Top-1 spine prefix: compact slots 1..=cliff_depth-1 (depth == slot).
    for i in 0..cliff_depth.saturating_sub(1) {
        tree_token_ids.push(spine[i]);
        parent_indices.push(i as i32 - 1);
    }
    // Fork node (top-2) at compact slot cliff_depth. Parent = slot
    // cliff_depth-1 → payload index cliff_depth-2.
    tree_token_ids.push(fork_token);
    parent_indices.push((cliff_depth as i32) - 2);
    // Tail re-rooted on the fork: each node is a child of the previous, laid
    // contiguously at slots cliff_depth+1.. Reserve ONE slot at the end for the
    // top-1 leaf, so the tail takes at most n - cliff_depth - 1 nodes.
    let mut prev = tree_token_ids.len() as i32 - 1; // fork node's payload idx
    let tail_budget = n.saturating_sub(cliff_depth + 1);
    for &t in tail.iter().take(tail_budget) {
        tree_token_ids.push(t);
        parent_indices.push(prev);
        prev = tree_token_ids.len() as i32 - 1;
    }
    // Top-1-at-cliff leaf at the final slot, sibling of the fork (same parent =
    // the spine node at slot cliff_depth-1, payload idx cliff_depth-2).
    if tree_token_ids.len() < n {
        tree_token_ids.push(spine[cliff_depth - 1]);
        parent_indices.push((cliff_depth as i32) - 2);
    }
    // Pad any remaining budget with a benign continuation off the leaf so the
    // node count stays exactly n (verify K shape unchanged). These padding
    // nodes are never on the committed contiguous run.
    while tree_token_ids.len() < n {
        let pad_idx = tree_token_ids.len() as i32 - 1;
        tree_token_ids.push(spine[(cliff_depth).min(n - 1)]);
        parent_indices.push(pad_idx);
    }

    TreePayload {
        tree_token_ids,
        parent_indices,
    }
}

/// Build a 2-ROOT PORTFOLIO forest payload (`ATLAS_DFLASH_PORTFOLIO=1`).
///
/// Verifies TWO independent flat chains in ONE verify pass by laying them as
/// two sibling branches under the shared root (compact slot 0 = the bonus /
/// `last_token`):
///
/// ```text
///                     root (slot 0 = last_token)
///                    /                          \
///          chain_a[0] (depth 1)          chain_b[0] (depth 1)
///              |                              |
///          chain_a[1]                     chain_b[1]
///              |                              |
///             ...                            ...
/// ```
///
/// `chain_a` is the DFlash neural drafter's γ-token chain; `chain_b` is the
/// SAM/retrieval chain. Both are internally FLAT (each node's parent is the
/// previous node in its own chain), so within either branch the greedy walk
/// commits a strictly-contiguous run — the ONLY non-flat structure is the
/// single fork at the root.
///
/// ## Why this is lossless
///
/// The greedy DDTree walker ([`greedy_sample_ddtree`]) at the root commits the
/// target's argmax `verified[0]` and descends into whichever branch's depth-1
/// token equals it. It then follows that branch as a plain contiguous chain,
/// accepting `chain[i]` only while `chain[i] == target_argmax` at each depth.
/// The other branch is NEVER committed. So the committed token stream is
/// exactly the target's greedy continuation regardless of which source fired —
/// byte-identical to running the SELECTED chain alone. The win is that BOTH
/// candidate continuations were verified in the same bandwidth-bound pass, so
/// whichever the target picks at the fork is already verified to full depth.
///
/// ## Layout (compact indices are 1-based; slot 0 is the root/bonus)
///
/// Chain A occupies compact slots `1 ..= a` (contiguous, depth == slot).
/// Chain B occupies compact slots `a+1 ..= a+b`; `chain_b[0]` attaches to the
/// ROOT (parent = -1), and `chain_b[i>0]` attaches to the previous B node.
/// So B's slots are NOT equal to its depths (B[0] is at slot a+1 but depth 1).
///
/// For attention/RoPE correctness on the B branch, the caller MUST enable a
/// depth-aware verify path (`ATLAS_DDTREE_DFS_REORDER=1` — DFS pre-order lays
/// each ancestor chain contiguously and is the validated-lossless mechanism —
/// or `ATLAS_DDTREE_TREE_AWARE_VERIFY=1`) AND `ATLAS_DDTREE_TREE_TOKENS_VERIFY=1`
/// so the verifier embeds the tree topology's tokens at each slot. Under pure
/// flat metadata only chain A (slot == depth) is guaranteed lossless; chain B's
/// deep tail would read sibling-A KV. The greedy oracle keeps output correct
/// regardless (a contaminated B row simply fails to match target argmax and is
/// rejected), so the flag combination trades acceptance depth, never
/// correctness.
///
/// ## Node budget
///
/// Total nodes `a + b` must be `<= max_nodes` (the K-1 verify capacity, from
/// `ATLAS_DDTREE_MAX_NODES`; tree-WY K_MAX is 32 so `max_nodes <= 31`). Chain A
/// is never truncated (it is the trained-drafter baseline — the flat-safe
/// fallback must stay byte-exact when B doesn't fire); chain B is truncated to
/// the remaining budget. Returns a plain flat-A payload when B has no room.
pub fn build_portfolio_payload(
    chain_a: &[u32],
    chain_b: &[u32],
    max_nodes: usize,
) -> TreePayload {
    let a = chain_a.len().min(max_nodes);
    // Chain A always laid first (contiguous, slot == depth).
    let mut tree_token_ids: Vec<u32> = Vec::with_capacity(max_nodes);
    let mut parent_indices: Vec<i32> = Vec::with_capacity(max_nodes);
    for i in 0..a {
        tree_token_ids.push(chain_a[i]);
        parent_indices.push(i as i32 - 1); // A[0] → root (-1); A[i] → A[i-1]
    }
    // Chain B fills the remaining budget as a sibling branch off the root.
    let b_budget = max_nodes.saturating_sub(a);
    let b = chain_b.len().min(b_budget);
    let a_end_payload_idx = tree_token_ids.len() as i32; // first B payload index
    for j in 0..b {
        tree_token_ids.push(chain_b[j]);
        if j == 0 {
            parent_indices.push(-1); // B[0] → root (sibling of A[0])
        } else {
            parent_indices.push(a_end_payload_idx + j as i32 - 1); // B[j] → B[j-1]
        }
    }
    TreePayload {
        tree_token_ids,
        parent_indices,
    }
}

/// Map a DDTree greedy walk's `accepted_compact_indices` to the kernel
/// intermediate slot index of the **last accepted state**.
///
/// Kernel slot layout (per `gdn_decode_tree_wy` / `gdn_decode_tree`):
/// `h_state_intermediates[t]` for `t = 0..T-1` where `T = γ+1` and slot `0`
/// holds the state AFTER the root token (i.e. the previously-committed
/// bonus / `last_token`). Slot `k` (for `k ∈ [1, T-1]`) holds the state
/// AFTER applying `tree_token_ids[k-1]` — i.e. compact index `k`.
///
/// So the kernel slot of the last accepted draft is exactly
/// `accepted_compact_indices.last()`. When no drafts were accepted (empty
/// vec) the only "accepted" position is the prefix bonus at slot `0`.
///
/// In chain mode (`accepted_compact_indices = [1, 2, ..., n]`) this returns
/// `n`, identical to the legacy `total_accepted - 1 = n` arithmetic. In
/// tree mode with non-contiguous indices (e.g. `[1, 4, 7]`) it returns the
/// correct sparse slot `7`, NOT `len - 1 = 2`.
#[inline]
pub fn last_accepted_inter_slot(accepted_compact_indices: &[usize]) -> usize {
    accepted_compact_indices.last().copied().unwrap_or(0)
}

/// Translate a compact-frame index into the KERNEL (DFS-reordered) frame.
///
/// ## Why this exists (root cause of the sparse fork-commit corruption)
///
/// `greedy_sample_ddtree{,_full}` returns `accepted_compact_indices` in the
/// ORIGINAL COMPACT frame: compact index `c` is the payload node whose token
/// is `tree_token_ids[c-1]` and whose verify argmax is row `c`. That is the
/// frame the caller reasons about (emit tokens, bonus row, SSM slot lookup —
/// all consume the UN-permuted `verified` vec, so compact indices are right
/// for those).
///
/// But two pieces of GPU-resident state are laid out in the KERNEL frame, not
/// the compact frame, when `ATLAS_DDTREE_DFS_REORDER=1` is active:
///
///   1. **Attention KV.** `verify_d.rs` writes KV for KERNEL slot `t` at KV
///      position `pre_verify_len + t`, and kernel slot `t` embeds the token
///      `tokens[dfs_perm[t]]` (compact index `dfs_perm[t]`). So compact index
///      `c`'s KV lives at position `pre + dfs_inv_perm[c]`, NOT `pre + c`.
///   2. **`dflash_hidden_save` capture rows.** `try_dflash_capture(layer, t)`
///      saves the hidden for KERNEL slot `t` at row `t`. Compact index `c`'s
///      captured hidden is therefore at row `dfs_inv_perm[c]`.
///
/// Before this helper, `compact_verify_kv_dispatch` and the ctx-hidden append
/// in `propose.rs` used the compact indices DIRECTLY as kernel-frame
/// positions/rows. On a genuine fork with DFS reorder on (the portfolio /
/// deep-sibling DDTree commit path — `permutation != identity`), that gathered
/// the WRONG KV bytes and appended the WRONG ctx hiddens → the committed KV /
/// drafter conditioning diverged from the emitted tokens → counting md5
/// corrupts and accept collapses. `commit_verify_state_async_with_slot`
/// ALREADY applied `dfs_inv_perm` to its single `last_inter_slot`, so it was
/// the odd one out that worked; these two paths were missing the same map.
///
/// `inv_perm[old_kernel_slot] = new_dfs_slot` is the SAME buffer stashed by
/// `verify_d.rs` (`ddtree_dfs_inv_perm`) and consumed by
/// `commit_verify_state_async_with_slot`. Empty `inv_perm` (chain mode, DFS
/// disabled, or an identity permutation that `verify_d.rs` collapses to empty)
/// ⇒ the map is the identity, so the flat/chain path is byte-for-byte
/// unchanged. An out-of-range compact index (defensive) also passes through
/// unmapped rather than panicking.
#[inline]
pub fn compact_to_kernel_slot(compact: usize, inv_perm: &[usize]) -> usize {
    if !inv_perm.is_empty() && compact < inv_perm.len() {
        inv_perm[compact]
    } else {
        compact
    }
}

/// Map a whole compact-frame accepted path into the kernel (DFS) frame.
///
/// Applies [`compact_to_kernel_slot`] element-wise. Used by
/// `compact_verify_kv` (to gather KV from the true kernel positions) and the
/// ctx-hidden append (to read the true capture rows). Identity when
/// `inv_perm` is empty (chain / DFS-off), preserving the flat path exactly.
pub fn map_compact_path_to_kernel(compact_path: &[usize], inv_perm: &[usize]) -> Vec<usize> {
    compact_path
        .iter()
        .map(|&c| compact_to_kernel_slot(c, inv_perm))
        .collect()
}

/// One relocation for post-verify KV compaction, in ABSOLUTE KV positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvMove {
    pub src_pos: usize,
    pub dst_pos: usize,
}

/// Plan the overlap-safe sequence of KV relocations that gathers a sparse
/// (fork) accepted path down to the contiguous run `pre+1 .. pre+N`.
///
/// ## Inputs
///   - `accepted_compact`: the greedy walk's path in COMPACT frame (walk
///     order, strictly increasing in compact index).
///   - `inv_perm`: `ddtree_dfs_inv_perm` — maps compact index → kernel slot.
///     Empty ⇒ kernel frame == compact frame (chain / DFS-off).
///   - `pre_verify_len`: sequence length before this verify advanced it.
///   - `k`: total verify width (= `tokens.len()`, the number of kernel slots).
///     Positions `pre+N+1 .. pre+k-1` are the UNCOMMITTED tail verify slots;
///     they are free scratch (the next verify overwrites them) and are used to
///     break clobber cycles.
///
/// ## Why ordering matters (root cause of the DFS-on fork corruption)
///
/// Step `j` (0-based) commits compact index `accepted_compact[j]`, whose KV
/// physically lives at KERNEL position `pre + inv_perm[compact]` (NOT
/// `pre + compact` — the pre-fix bug), and whose contiguous home is
/// `pre + j + 1`. On the flat/chain path `inv_perm` is empty and sources are
/// already `>= ` destinations, so plain ascending-dst order is safe (and no
/// move fires at all for a truly contiguous path). But once DFS reorder
/// permutes kernel slots, a source may equal a not-yet-consumed destination
/// (`dst_a == src_b`), so a naive in-place copy clobbers data a later move
/// still needs. We resolve this generally by first evacuating any source that
/// is also a destination into a free tail scratch position, then performing
/// the placements. The result is a byte-lossless permutation of committed KV.
///
/// Returns moves already filtered to `src != dst` (identity moves dropped) and
/// ordered so that executing them front-to-back never destroys unconsumed
/// data. Empty when the path is already contiguous in the kernel frame.
pub fn plan_kv_compaction_moves(
    accepted_compact: &[usize],
    inv_perm: &[usize],
    pre_verify_len: usize,
    k: usize,
) -> Vec<KvMove> {
    let n = accepted_compact.len();
    // Desired (src_pos, dst_pos) in absolute KV positions.
    let mut want: Vec<(usize, usize)> = Vec::with_capacity(n);
    for (j, &compact) in accepted_compact.iter().enumerate() {
        let kernel_slot = compact_to_kernel_slot(compact, inv_perm);
        let src = pre_verify_len + kernel_slot;
        let dst = pre_verify_len + j + 1;
        if src != dst {
            want.push((src, dst));
        }
    }
    if want.is_empty() {
        return Vec::new();
    }

    // Fast path: if no destination is also a source, ascending-dst is safe
    // (every write lands on a position no remaining move reads from).
    let dst_set: std::collections::HashSet<usize> = want.iter().map(|&(_, d)| d).collect();
    let src_set: std::collections::HashSet<usize> = want.iter().map(|&(s, _)| s).collect();
    let has_conflict = want.iter().any(|&(_, d)| src_set.contains(&d));
    if !has_conflict {
        let mut moves: Vec<KvMove> = want
            .into_iter()
            .map(|(src_pos, dst_pos)| KvMove { src_pos, dst_pos })
            .collect();
        moves.sort_by_key(|m| m.dst_pos);
        return moves;
    }

    // General path: evacuate every conflicting source (a source that is also
    // some move's destination) into a free tail scratch position first, then
    // run all placements. Free scratch = uncommitted verify slots
    // `pre + n + 1 .. pre + k - 1` (past the accepted run; overwritten next
    // verify). We only need as many scratch slots as there are conflicts.
    let mut moves: Vec<KvMove> = Vec::with_capacity(want.len() + n);
    // Free scratch = tail verify slots that are NOT themselves a source or a
    // destination of any wanted move (so evacuating into them can never
    // clobber live data). Sources may be kernel slots > n under DFS reorder,
    // so filter against both sets explicitly.
    let mut scratch_slots: Vec<usize> = ((pre_verify_len + n + 1)..(pre_verify_len + k))
        .filter(|p| !src_set.contains(p) && !dst_set.contains(p))
        .collect();
    let mut scratch_iter = scratch_slots.drain(..);
    // Map: original src_pos -> the scratch pos we evacuated it to.
    let mut relocated: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for &(src, _) in &want {
        // Evacuate each conflicting source (a source that is also some move's
        // destination) exactly once, into a free tail scratch slot. If scratch
        // is exhausted (cannot happen when callers pass k = full verify width,
        // since conflicts <= n-1 <= tail slots) the placement below still runs
        // from the original source — bounded, but callers guarantee enough
        // scratch so the relocation stays strictly lossless.
        if dst_set.contains(&src)
            && !relocated.contains_key(&src)
            && let Some(scratch) = scratch_iter.next()
        {
            moves.push(KvMove {
                src_pos: src,
                dst_pos: scratch,
            });
            relocated.insert(src, scratch);
        }
    }
    // Placements, ascending by destination. Read from the evacuated scratch
    // when the source was relocated, else from the original source.
    let mut placements: Vec<(usize, usize)> = want
        .iter()
        .map(|&(src, dst)| (*relocated.get(&src).unwrap_or(&src), dst))
        .collect();
    placements.sort_by_key(|&(_, dst)| dst);
    for (src_pos, dst_pos) in placements {
        moves.push(KvMove { src_pos, dst_pos });
    }
    moves
}

// ============================================================================
// DFS reorder — kernel-frame permutation for tree-aware attention (option C)
// ============================================================================
//
// The paged-decode attention kernel iterates `[0..seq_lens[t])` sequentially
// and reads slot indirection only via `block_table + pos % bs` — it has NO
// per-row KV mask. So for a non-flat tree, a depth-d query at compact slot k
// reads slots `[0..k]` which are SIBLINGS/COUSINS, not ancestors.
//
// Option C: permute the kernel-frame token order so each ancestor chain is
// contiguous in slot order (DFS pre-order). Then attention naturally reads
// the right neighborhood without any kernel surgery.
//
// `kernel_parents` is the K-element vector stashed by `set_ddtree_parent_ids`:
//   index 0 = bonus (parent = -1)
//   index i for i >= 1 = draft i-1 (parent = -1 or k where 0 <= k < i)
//
// Returns `(perm, inv_perm, depths_kernel)` where:
//   - perm[new] = old   (new = DFS slot index; old = kernel slot index)
//   - inv_perm[old] = new
//   - depths_kernel[i] = tree depth of kernel slot i (bonus at slot 0 = 0)

/// DFS pre-order traversal of the kernel-frame tree.
///
/// Returns `(permutation, inverse_permutation, depths_per_kernel_slot)`.
/// `permutation[i] = j` means "DFS slot i contains kernel slot j".
/// `inverse_permutation[j] = i` means "kernel slot j is at DFS slot i".
///
/// Children of each parent are visited in ascending kernel-slot order
/// (stable wrt original tree-build order). The bonus (kernel slot 0) is
/// always visited first.
pub fn dfs_reorder(kernel_parents: &[i32]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let k = kernel_parents.len();
    if k == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Depths in kernel frame. Slot 0 (bonus) is depth 0.
    // For i >= 1: parent < 0 means "root child of bonus" (depth 1).
    // Otherwise parent kernel slot p means depth = 1 + depth[p].
    let mut depths = vec![0usize; k];
    for i in 1..k {
        let p = kernel_parents[i];
        if p < 0 {
            depths[i] = 1;
        } else {
            let pi = p as usize;
            depths[i] = if pi >= i {
                // Malformed; fall back to chain depth so we don't panic.
                i
            } else {
                depths[pi].saturating_add(1)
            };
        }
    }

    // Build children adjacency. Treat parent == -1 as child of slot 0 (bonus)
    // for traversal purposes — slot 0 is always the root.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); k];
    for i in 1..k {
        let p = kernel_parents[i];
        let parent_slot = if p < 0 { 0 } else { (p as usize).min(k - 1) };
        children[parent_slot].push(i);
    }
    // Children in ascending slot order so DFS is deterministic.
    for c in &mut children {
        c.sort_unstable();
    }

    // Iterative DFS from slot 0. Push children in reverse so smallest is
    // popped first.
    let mut perm: Vec<usize> = Vec::with_capacity(k);
    let mut stack: Vec<usize> = Vec::with_capacity(k);
    stack.push(0);
    while let Some(node) = stack.pop() {
        perm.push(node);
        for &c in children[node].iter().rev() {
            stack.push(c);
        }
    }
    // Defensive: if some slot wasn't reachable (shouldn't happen for a valid
    // tree), append in order so perm.len() == k.
    if perm.len() != k {
        let mut visited = vec![false; k];
        for &p in &perm {
            visited[p] = true;
        }
        for i in 0..k {
            if !visited[i] {
                perm.push(i);
            }
        }
    }

    let mut inv = vec![0usize; k];
    for (new, &old) in perm.iter().enumerate() {
        inv[old] = new;
    }
    (perm, inv, depths)
}

/// Permute a kernel-frame parent_ids vector into DFS slot order.
///
/// New parent_ids in DFS frame:
///   new_parents[i] = -1 if perm[i] == 0 (the bonus / root)
///                  = inv_perm[old_parents[perm[i]] mapped to kernel slot]  otherwise
///
/// When old_parents[perm[i]] == -1 (root child), the kernel-frame "parent
/// kernel slot" is 0 (the bonus) → new_parents[i] = inv_perm[0] = 0
/// (since DFS always places bonus at slot 0).
pub fn permute_parent_ids(kernel_parents: &[i32], perm: &[usize], inv_perm: &[usize]) -> Vec<i32> {
    let k = kernel_parents.len();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let old_slot = perm[i];
        if old_slot == 0 {
            // Bonus slot — keep -1 marker (loads from pre-tree state).
            out.push(-1i32);
            continue;
        }
        let old_parent = kernel_parents[old_slot];
        let parent_kernel_slot: usize = if old_parent < 0 {
            // Root-child — kernel "parent slot" is 0 (bonus).
            0
        } else {
            (old_parent as usize).min(k - 1)
        };
        let new_parent_dfs = inv_perm[parent_kernel_slot] as i32;
        out.push(new_parent_dfs);
    }
    out
}

#[cfg(test)]
mod dfs_reorder_tests {
    use super::*;

    #[test]
    fn dfs_reorder_flat_chain_is_identity() {
        // kernel_parents for a 5-token flat chain: [-1, 0, 1, 2, 3]
        let kp = vec![-1i32, 0, 1, 2, 3];
        let (perm, inv, depths) = dfs_reorder(&kp);
        assert_eq!(perm, vec![0, 1, 2, 3, 4]);
        assert_eq!(inv, vec![0, 1, 2, 3, 4]);
        assert_eq!(depths, vec![0, 1, 2, 3, 4]);
        let np = permute_parent_ids(&kp, &perm, &inv);
        assert_eq!(np, kp);
    }

    #[test]
    fn dfs_reorder_root_children_then_chain() {
        // example from task: parent_indices payload (compact frame) =
        //   [-1, -1, -1, -1, -1, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        // → kernel_parents (compact + 1, bonus first) =
        //   [-1,  0,  0,  0,  0,  0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        // i.e. 5 root-child siblings (kernel slots 1..6, parent=0), then a
        // 10-deep chain off compact 0 (kernel slot 6, parent=1; slot 7
        // parent=6 etc.).
        let kp = vec![
            -1i32, 0, 0, 0, 0, 0, // bonus + 5 root children
            1, 6, 7, 8, 9, 10, 11, 12, 13, 14, // 10-deep chain off slot 1
        ];
        let (perm, _inv, depths) = dfs_reorder(&kp);
        assert_eq!(perm.len(), kp.len());
        assert_eq!(perm[0], 0, "bonus must be at DFS slot 0");
        // Slot 1 is the first child of bonus = kernel slot 1 (smallest).
        assert_eq!(perm[1], 1);
        // The chain off slot 1 is visited next: kernel slots 6, 7, 8, ..., 15.
        for (i, expected) in (2..=11).zip(6..=15) {
            assert_eq!(
                perm[i], expected,
                "DFS slot {} should be kernel slot {}",
                i, expected
            );
        }
        // Then the remaining root children at slots 2, 3, 4, 5.
        assert_eq!(&perm[12..16], &[2, 3, 4, 5]);
        // Depths are computed in kernel frame.
        assert_eq!(depths[0], 0);
        for i in 1..=5 {
            assert_eq!(depths[i], 1, "kernel slot {} should be depth 1", i);
        }
        for (i, d) in (6..=15).zip(2..=11) {
            assert_eq!(depths[i], d, "kernel slot {} should be depth {}", i, d);
        }
    }

    #[test]
    fn permute_parent_ids_makes_ancestors_contiguous() {
        // Use the example above and verify the permuted parents form
        // valid "previous-only" references (each parent < self) AND that the
        // chain rooted at DFS slot 1 has consecutive parents.
        let kp = vec![-1i32, 0, 0, 0, 0, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let (perm, inv, _depths) = dfs_reorder(&kp);
        let np = permute_parent_ids(&kp, &perm, &inv);
        // All non-root parents must be < their index (causal).
        for i in 1..np.len() {
            assert!(
                np[i] >= 0 && (np[i] as usize) < i,
                "parent {} at DFS slot {} must be < self",
                np[i],
                i
            );
        }
        // First chain off the chosen root: DFS slots 1..=11.
        // np[1] = 0 (parent is bonus), np[2] = 1, np[3] = 2, ..., np[11] = 10.
        assert_eq!(np[1], 0);
        for i in 2..=11 {
            assert_eq!(
                np[i],
                (i - 1) as i32,
                "chain slot {} parent should be {}",
                i,
                i - 1
            );
        }
        // Sibling root children at DFS slots 12, 13, 14, 15 all have parent 0.
        for i in 12..=15 {
            assert_eq!(
                np[i], 0,
                "sibling root child at DFS slot {} parent should be 0 (bonus)",
                i
            );
        }
    }

    #[test]
    fn dfs_reorder_handles_two_root_branches() {
        // bonus + two siblings, each with one child.
        // kernel_parents = [-1, 0, 0, 1, 2] (slot 1 + 2 are root children,
        // slot 3 child of 1, slot 4 child of 2).
        let kp = vec![-1i32, 0, 0, 1, 2];
        let (perm, inv, depths) = dfs_reorder(&kp);
        // DFS: 0 → 1 → 3 → 2 → 4.
        assert_eq!(perm, vec![0, 1, 3, 2, 4]);
        assert_eq!(inv[0], 0);
        assert_eq!(inv[1], 1);
        assert_eq!(inv[3], 2);
        assert_eq!(inv[2], 3);
        assert_eq!(inv[4], 4);
        assert_eq!(depths, vec![0, 1, 1, 2, 2]);
        let np = permute_parent_ids(&kp, &perm, &inv);
        // DFS slot 0: bonus → -1
        // DFS slot 1: kernel 1 (root child) → parent kernel 0 (bonus) → DFS 0
        // DFS slot 2: kernel 3 (child of 1) → parent kernel 1 → DFS 1
        // DFS slot 3: kernel 2 (root child) → parent kernel 0 → DFS 0
        // DFS slot 4: kernel 4 (child of 2) → parent kernel 2 → DFS 3
        assert_eq!(np, vec![-1, 0, 1, 0, 3]);
    }
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

    // ---- parent_metadata tests ----

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

    #[test]
    fn build_padded_parent_ids_empty_returns_none() {
        let req_ids = vec!["a".to_string()];
        let payloads = std::collections::HashMap::new();
        let result = build_padded_parent_ids(&req_ids, &payloads, None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn build_padded_parent_ids_packs_rows() {
        let req_ids = vec!["a".to_string(), "b".to_string()];
        let mut payloads = std::collections::HashMap::new();
        payloads.insert("a".to_string(), payload(&[11, 21, 22, 31], &[-1, 0, 0, 2]));
        let meta = build_padded_parent_ids(&req_ids, &payloads, None)
            .unwrap()
            .expect("should produce metadata");
        // row 0: parents [-1, -1, 1, 1, 3] length 5
        // row 1: pad row, all PADDING_PARENT
        assert_eq!(meta.stride, 5);
        assert_eq!(meta.num_tree_tokens, vec![4, 0]);
        assert_eq!(meta.row(0), &[ROOT_PARENT, ROOT_PARENT, 1, 1, 3]);
        assert_eq!(meta.row(1), &[PADDING_PARENT; 5]);
    }

    // ---- TreeVerifierMetadata tests ----

    #[test]
    fn verifier_metadata_from_demo_tree() {
        let tree = build_ddtree(&demo_candidates(), 8, 3, true, 0, u32::MAX).unwrap();
        let meta = TreeVerifierMetadata::from_tree(/*prompt_len=*/ 10, &tree).unwrap();
        assert_eq!(meta.num_tree_nodes(), 8);
        // First compact row = prompt_len - 1 (root logits).
        assert_eq!(meta.compact_logits_indices[0], 9);
        // Following rows = 10, 11, ..., 17.
        for i in 0..8 {
            assert_eq!(meta.compact_logits_indices[i + 1], 10 + i);
        }
        // All tree positions are >= prompt_len.
        for &p in &meta.tree_position_ids {
            assert!(p >= 10);
        }
    }

    #[test]
    fn verifier_metadata_all_position_ids_packs_prompt_then_tree() {
        let tree = build_ddtree(&demo_candidates(), 4, 3, true, 0, u32::MAX).unwrap();
        let meta = TreeVerifierMetadata::from_tree(5, &tree).unwrap();
        let pos = meta.all_position_ids();
        // Prompt section [0..5].
        assert_eq!(&pos[..5], &[0u32, 1, 2, 3, 4]);
        // Tree section: depth 1 → pos 5, depth 2 → pos 6, etc.
        let chain_depths: Vec<u32> = meta.node_depths.iter().map(|d| 5 + *d as u32 - 1).collect();
        assert_eq!(&pos[5..], chain_depths.as_slice());
    }

    #[test]
    fn prefill_tree_mask_root_row_causal() {
        let tree = build_ddtree(&demo_candidates(), 4, 3, true, 0, u32::MAX).unwrap();
        let prompt_len = 3;
        let total = prompt_len + tree.non_root_nodes().len();
        let mask = prefill_tree_attention_mask(prompt_len, &tree);
        // Prompt row 0 attends only to col 0.
        assert!(mask[0]);
        for c in 1..total {
            assert!(!mask[c]);
        }
        // Prompt row 2 attends to cols 0..=2.
        for c in 0..=2 {
            assert!(mask[2 * total + c]);
        }
        // No prompt row attends to any tree column.
        for r in 0..prompt_len {
            for c in prompt_len..total {
                assert!(
                    !mask[r * total + c],
                    "prompt row {r} should not see tree col {c}"
                );
            }
        }
        // Every tree row sees the full prompt.
        for r in prompt_len..total {
            for c in 0..prompt_len {
                assert!(mask[r * total + c]);
            }
        }
    }

    #[test]
    fn prefill_tree_mask_no_sibling_visibility() {
        // 2 siblings at depth 1, each with no children.
        let cs = vec![vec![cand(10, -0.1), cand(11, -0.2)]];
        let tree = build_ddtree(&cs, 2, 2, false, 2, u32::MAX).unwrap();
        assert_eq!(tree.nodes.len(), 3); // root + 2 siblings
        let prompt_len = 2;
        let total = prompt_len + 2;
        let mask = prefill_tree_attention_mask(prompt_len, &tree);
        let sibling_a_row = prompt_len + 1 - 1; // node index 1 → compact row 2
        let sibling_b_row = prompt_len + 2 - 1; // node index 2 → compact row 3
        let sibling_a_col = prompt_len + 1 - 1;
        let sibling_b_col = prompt_len + 2 - 1;
        // Sibling A cannot see Sibling B.
        assert!(!mask[sibling_a_row * total + sibling_b_col]);
        // Sibling B cannot see Sibling A.
        assert!(!mask[sibling_b_row * total + sibling_a_col]);
        // Each sees itself.
        assert!(mask[sibling_a_row * total + sibling_a_col]);
        assert!(mask[sibling_b_row * total + sibling_b_col]);
    }

    // ---- M5B runtime sampler tests ----

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

    // ── PORTFOLIO 2-root forest (ATLAS_DFLASH_PORTFOLIO) ──────────────────

    #[test]
    fn portfolio_layout_two_sibling_chains() {
        // chain_a = [10,20,30], chain_b = [11,21]. max_nodes = 8.
        // A → slots 1,2,3 (parents -1,0,1); B → slots 4,5 (parents -1,3).
        let p = build_portfolio_payload(&[10, 20, 30], &[11, 21], 8);
        assert_eq!(p.tree_token_ids, vec![10, 20, 30, 11, 21]);
        // payload parent indices: A[0]→root(-1), A[1]→A[0](0), A[2]→A[1](1),
        // B[0]→root(-1), B[1]→B[0](payload idx 3).
        assert_eq!(p.parent_indices, vec![-1, 0, 1, -1, 3]);
    }

    #[test]
    fn portfolio_commits_chain_a_when_target_picks_a() {
        // Root argmax selects chain_a's first token (10). The whole A chain
        // then matches → commit A in full; chain B never touched. Lossless:
        // identical to verifying chain A alone.
        let p = build_portfolio_payload(&[10, 20, 30], &[11, 21], 8);
        let r = req(&p.tree_token_ids, &p.parent_indices);
        // rows: [root, A0, A1, A2, B0, B1]
        // root→10, A0(10)→20, A1(20)→30, A2(30)→99(bonus), B rows irrelevant.
        let argmax = vec![10u32, 20, 30, 99, 0, 0];
        let s = greedy_sample_ddtree(&r, &argmax).unwrap();
        assert_eq!(s.output_token_ids, vec![10, 20, 30, 99]);
        assert_eq!(s.accepted_compact_indices, vec![1, 2, 3]);
    }

    #[test]
    fn portfolio_commits_chain_b_when_target_picks_b() {
        // Root argmax selects chain_b's first token (11) — the drafter chain A
        // would have diverged at the root, but the retrieval chain B matches to
        // full depth in the SAME verify pass. This is the portfolio WIN.
        let p = build_portfolio_payload(&[10, 20, 30], &[11, 21], 8);
        let r = req(&p.tree_token_ids, &p.parent_indices);
        // B occupies compact slots 4 (tok 11) and 5 (tok 21).
        // root→11 (picks B0), B0(11)→21 (picks B1), B1(21)→77 (bonus).
        let argmax = vec![11u32, 0, 0, 0, 21, 77];
        let s = greedy_sample_ddtree(&r, &argmax).unwrap();
        // Flat-safe contract: the accepted compact path is [4, 5] which is
        // NON-contiguous from position 1, so adapt_to_flat_safe truncates to
        // the flat prefix (empty, since compact 4 != index 1) + the bonus =
        // the first B token as a single free bonus. This is the LOSSLESS
        // guarantee under flat metadata: output is still the target's greedy
        // token (11). Deeper B commit requires the depth-aware verify path.
        assert_eq!(s.output_token_ids, vec![11]);
    }

    #[test]
    fn portfolio_full_b_commit_under_tree_commit_semantics() {
        // greedy_sample_ddtree_full commits the WHOLE accepted path (the
        // depth-aware / tree-commit path the portfolio uses in production),
        // so when the target rides chain B the entire B run + bonus commits.
        let p = build_portfolio_payload(&[10, 20, 30], &[11, 21], 8);
        let r = req(&p.tree_token_ids, &p.parent_indices);
        let argmax = vec![11u32, 0, 0, 0, 21, 77];
        let s = greedy_sample_ddtree_full(&r, &argmax).unwrap();
        assert_eq!(s.output_token_ids, vec![11, 21, 77]);
        assert_eq!(s.accepted_compact_indices, vec![4, 5]);
        // last accepted inter slot = max compact index (kernel state slot).
        assert_eq!(last_accepted_inter_slot(&s.accepted_compact_indices), 5);
    }

    #[test]
    fn portfolio_a_never_truncated_b_gets_remaining_budget() {
        // max_nodes = 4: chain A (3 nodes) laid in full; chain B (3 tokens)
        // truncated to the remaining 1 slot.
        let p = build_portfolio_payload(&[1, 2, 3], &[7, 8, 9], 4);
        assert_eq!(p.tree_token_ids, vec![1, 2, 3, 7]);
        assert_eq!(p.parent_indices, vec![-1, 0, 1, -1]);
    }

    #[test]
    fn portfolio_no_room_for_b_is_plain_flat_a() {
        // A fills the entire budget → B gets zero slots → payload is a plain
        // flat A chain, byte-identical to the non-portfolio drafter path.
        let p = build_portfolio_payload(&[1, 2, 3, 4], &[7, 8], 4);
        assert_eq!(p.tree_token_ids, vec![1, 2, 3, 4]);
        assert_eq!(p.parent_indices, vec![-1, 0, 1, 2]);
    }

    #[test]
    fn portfolio_empty_b_is_flat_a() {
        let p = build_portfolio_payload(&[5, 6], &[], 8);
        assert_eq!(p.tree_token_ids, vec![5, 6]);
        assert_eq!(p.parent_indices, vec![-1, 0]);
    }

    #[test]
    fn portfolio_first_token_collision_stays_lossless() {
        // chain_a and chain_b share the same first token (10). The root has two
        // children carrying token 10 (compact 1 = A[0], compact 4 = B[0]);
        // child_maps keys by token so the LATER insert (B[0], compact 4) wins
        // the root's 10-entry. The greedy walk therefore descends B[0], then
        // rides whichever branch the target continues. Correctness is
        // unaffected — the committed tokens are still the target's argmax; the
        // only cost is that A's chain isn't taken when the first tokens collide
        // (no gain, no corruption).
        let p = build_portfolio_payload(&[10, 20], &[10, 99], 8);
        assert_eq!(p.tree_token_ids, vec![10, 20, 10, 99]);
        assert_eq!(p.parent_indices, vec![-1, 0, -1, 2]);
        let r = req(&p.tree_token_ids, &p.parent_indices);
        // 4 nodes → argmax length must be 1 + 4 = 5 (root row + one per node).
        // root→10 (picks B0 at compact 3, the winning insert), B0(10)→77 bonus.
        let argmax = vec![10u32, 0, 0, 77, 0];
        let full = greedy_sample_ddtree_full(&r, &argmax).unwrap();
        // Whichever child is followed, the committed token stream begins with
        // the target's argmax 10 and the bonus — always lossless.
        assert_eq!(full.output_token_ids.first().copied(), Some(10));
    }

    #[test]
    fn sampler_flat_safe_truncates_at_branch_divergence() {
        // Tree: root → 10 → 20a, 20b; (siblings at depth 2).
        //   compact 1: token 10, parent 0 (root)
        //   compact 2: token 20a, parent 1
        //   compact 3: token 20b, parent 1
        // Flat chain by index: [1, 2, 3]. accept[0]=1, accept[1]=2 (flat),
        // accept[2]=3 would BREAK flat (compact 3 != index+1=3 still flat!)
        // Use a different shape that actually diverges:
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

    // ── Depth-contiguous caterpillar builder ──

    /// The committed fork-leaf branch must be at compact slot == its depth
    /// AND its slot-prefix must be exactly its ancestor chain (the lossless
    /// invariant). We assert both from the payload structure.
    #[test]
    fn caterpillar_fork_leaf_is_at_depth_slot() {
        // Spine of 8 top-1 tokens [10..80], cliff at spine depth 3 (fork off
        // the depth-2 spine node), fork token 999.
        let spine = [10u32, 20, 30, 40, 50, 60, 70, 80];
        let p = build_caterpillar_payload(&spine, 999, 3);
        // n nodes preserved.
        assert_eq!(p.tree_token_ids.len(), spine.len());
        assert_eq!(p.parent_indices.len(), spine.len());
        // Layout: slots 1,2 = spine[0],spine[1]; slot 3 = fork(999) parent slot2;
        // slots 4.. = spine[2],spine[3],... (shifted, deepest dropped).
        assert_eq!(p.tree_token_ids[0], 10); // slot 1
        assert_eq!(p.tree_token_ids[1], 20); // slot 2
        assert_eq!(p.tree_token_ids[2], 999); // slot 3 = FORK LEAF (depth 3)
        assert_eq!(p.tree_token_ids[3], 30); // slot 4 = spine[2] (depth 3, shifted)
        // Fork leaf (payload idx 2, compact slot 3) parent = compact slot 2
        // (payload idx 1).
        assert_eq!(p.parent_indices[2], 1);
        // First spine continuation (payload idx 3, slot 4) parent = compact
        // slot 2 (payload idx 1) — SKIPS the fork leaf at slot 3.
        assert_eq!(p.parent_indices[3], 1);

        // Build the runtime and verify the lossless invariant for the
        // fork-leaf branch: walk root→spine[0]→spine[1]→fork(999).
        let r = req(&p.tree_token_ids, &p.parent_indices);
        // argmax: root→10, after-slot1→20, after-slot2→999 (diverge to fork),
        // after-fork→bonus 7.
        let mut argmax = vec![0u32; 1 + p.tree_token_ids.len()];
        argmax[0] = 10; // root predicts spine[0]
        argmax[1] = 20; // slot1 predicts spine[1]
        argmax[2] = 999; // slot2 predicts the FORK token (target picks top-2)
        argmax[3] = 7; // fork-leaf row → bonus
        let full = greedy_sample_ddtree_full(&r, &argmax).unwrap();
        // Accepted compact path is [1, 2, 3] — CONTIGUOUS (== depths 1,2,3).
        assert_eq!(full.accepted_compact_indices, vec![1, 2, 3]);
        assert_eq!(full.output_token_ids, vec![10, 20, 999, 7]);
        // last inter slot == 3 == depth of the committed fork leaf.
        assert_eq!(last_accepted_inter_slot(&full.accepted_compact_indices), 3);
    }

    #[test]
    fn caterpillar_spine_path_stays_flat_safe() {
        // When the target follows top-1 (no divergence at the cliff), the walk
        // accepts the contiguous spine prefix then jumps to the shifted
        // continuation. greedy_sample_ddtree (flat-safe) truncates at the jump
        // → identical token contract to the flat baseline (the fork leaf is
        // simply unused). This guards the no-regress case.
        let spine = [10u32, 20, 30, 40, 50, 60, 70, 80];
        let p = build_caterpillar_payload(&spine, 999, 3);
        let r = req(&p.tree_token_ids, &p.parent_indices);
        // Target stays on top-1: root→10, slot1→20, slot2→30 (the spine
        // child, NOT the fork 999).
        let mut argmax = vec![0u32; 1 + p.tree_token_ids.len()];
        argmax[0] = 10;
        argmax[1] = 20;
        argmax[2] = 30; // spine child (compact slot 4) — diverges from compact order
        argmax[3] = 40;
        argmax[4] = 50;
        for k in 5..argmax.len() {
            argmax[k] = 0; // mismatch → stop
        }
        let flat = greedy_sample_ddtree(&r, &argmax).unwrap();
        // Flat-safe truncates at the first non-contiguous accept (slot 4 != 3),
        // committing [10, 20] + bonus 30 — byte-identical to flat-baseline.
        assert_eq!(flat.accepted_compact_indices, vec![1, 2]);
        assert_eq!(flat.output_token_ids, vec![10, 20, 30]);
    }

    #[test]
    fn caterpillar_tail_commits_fork_branch_contiguously() {
        // Spine top-1 [10,20,30,40,50,60,70,80], cliff at depth 3 (fork off the
        // depth-2 node), fork token 999, tail [31,41,51] (the post-cliff top-1
        // drafts re-rooted onto the fork).
        let spine = [10u32, 20, 30, 40, 50, 60, 70, 80];
        let tail = [31u32, 41, 51, 61, 71];
        let p = build_caterpillar_tail_payload(&spine, 999, 3, &tail);
        assert_eq!(p.tree_token_ids.len(), spine.len());
        // slots: 1=10, 2=20, 3=999(fork), 4=31, 5=41, 6=51, 7=61, 8=top1-leaf(30)
        assert_eq!(p.tree_token_ids[0], 10);
        assert_eq!(p.tree_token_ids[1], 20);
        assert_eq!(p.tree_token_ids[2], 999); // fork at slot 3 == depth 3
        assert_eq!(p.tree_token_ids[3], 31); // tail re-rooted on fork
        // top-1-at-cliff leaf is the LAST node (sibling of the fork).
        assert_eq!(*p.tree_token_ids.last().unwrap(), 30);
        assert_eq!(*p.parent_indices.last().unwrap(), 1); // parent = slot 2

        let r = req(&p.tree_token_ids, &p.parent_indices);
        // Target picks top-2 (999) at the cliff, then the tail re-accepts:
        // root→10, slot1→20, slot2→999, slot3(fork)→31, slot4→41, slot5→bonus.
        let mut argmax = vec![0u32; 1 + p.tree_token_ids.len()];
        argmax[0] = 10;
        argmax[1] = 20;
        argmax[2] = 999;
        argmax[3] = 31;
        argmax[4] = 41;
        argmax[5] = 7; // bonus
        let s = greedy_sample_ddtree(&r, &argmax).unwrap();
        // Contiguous commit [1,2,3,4,5] — the whole top-2 tail, the GAIN.
        assert_eq!(s.accepted_compact_indices, vec![1, 2, 3, 4, 5]);
        assert_eq!(s.output_token_ids, vec![10, 20, 999, 31, 41, 7]);
    }

    #[test]
    fn caterpillar_tail_top1_path_is_flat_safe() {
        // Target stays on top-1 at the cliff: root→10, slot1→20, slot2→30
        // (the top-1 token, which is the LEAF at the high slot). Flat-safe
        // truncates → [10,20] + bonus 30 = byte-identical to flat baseline.
        let spine = [10u32, 20, 30, 40, 50, 60, 70, 80];
        let tail = [31u32, 41, 51];
        let p = build_caterpillar_tail_payload(&spine, 999, 3, &tail);
        let r = req(&p.tree_token_ids, &p.parent_indices);
        let mut argmax = vec![0u32; 1 + p.tree_token_ids.len()];
        argmax[0] = 10;
        argmax[1] = 20;
        argmax[2] = 30; // top-1 → the high-slot leaf (non-contiguous)
        for k in 3..argmax.len() {
            argmax[k] = 0;
        }
        let s = greedy_sample_ddtree(&r, &argmax).unwrap();
        assert_eq!(s.accepted_compact_indices, vec![1, 2]);
        assert_eq!(s.output_token_ids, vec![10, 20, 30]);
    }

    #[test]
    fn caterpillar_tail_all_parents_causal() {
        let spine: Vec<u32> = (1..=16u32).map(|i| i * 10).collect();
        let tail: Vec<u32> = (1..=16u32).map(|i| i * 100).collect();
        for cliff in 1..=16usize {
            let p = build_caterpillar_tail_payload(&spine, 7777, cliff, &tail);
            assert_eq!(p.tree_token_ids.len(), spine.len(), "cliff={cliff}");
            for i in 0..p.parent_indices.len() {
                let par = p.parent_indices[i];
                assert!(
                    par >= -1 && par < i as i32,
                    "cliff={cliff} idx={i} par={par}"
                );
            }
        }
    }

    #[test]
    fn caterpillar_degenerate_inputs_fall_back_to_flat_chain() {
        // cliff_depth 0 or n<2 → plain flat chain (no fork node).
        let spine = [10u32, 20, 30];
        let p0 = build_caterpillar_payload(&spine, 999, 0);
        assert_eq!(p0.tree_token_ids, vec![10, 20, 30]);
        assert_eq!(p0.parent_indices, vec![-1, 0, 1]);
        let one = build_caterpillar_payload(&[42u32], 999, 1);
        assert_eq!(one.tree_token_ids, vec![42]);
        assert_eq!(one.parent_indices, vec![-1]);
    }

    #[test]
    fn caterpillar_all_parents_are_causal_and_well_formed() {
        // Every parent index must be < its own payload index (causal) and the
        // node count is preserved across a range of cliff depths.
        let spine: Vec<u32> = (1..=16u32).map(|i| i * 10).collect();
        for cliff in 1..=16usize {
            let p = build_caterpillar_payload(&spine, 7777, cliff);
            assert_eq!(p.tree_token_ids.len(), spine.len(), "cliff={cliff}");
            assert_eq!(p.parent_indices.len(), spine.len(), "cliff={cliff}");
            for i in 0..p.parent_indices.len() {
                let par = p.parent_indices[i];
                assert!(
                    par >= -1 && par < i as i32,
                    "cliff={cliff} idx={i} par={par}"
                );
            }
        }
    }

    #[test]
    fn full_sampler_commits_fork_tail() {
        // Single-cliff branch shape (the ATLAS_DFLASH_BRANCH=1 topology):
        //   compact 1: chain token 10 (root child)
        //   compact 2: chain token 20 (child of 1)
        //   compact 3: leaf token 99 (fork off compact 1 — parent = 1)
        // Flat indices: [1, 2] are contiguous; compact 3 is a sibling fork
        // off compact 1, so it diverges from the flat chain.
        //
        // Target greedy: root->10 (compact 1), compact-1 row -> 99 (fork to
        // the leaf at compact 3 instead of the chain child 20).
        let r = req(&[10, 20, 99], &[-1, 0, 0]);
        // argmax rows: [root, after-1, after-2, after-3].
        // root -> 10 (accept compact 1); compact-1 row -> 99 (accept leaf
        // compact 3); compact-3 row -> 777 (bonus).
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
        // which the SSM commit reads — NOT len-1 = 1.
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

    #[test]
    fn build_padded_parent_ids_respects_pad_to() {
        let req_ids = vec!["a".to_string()];
        let mut payloads = std::collections::HashMap::new();
        payloads.insert("a".to_string(), payload(&[11], &[-1]));
        let meta = build_padded_parent_ids(&req_ids, &payloads, Some(7))
            .unwrap()
            .expect("metadata");
        assert_eq!(meta.stride, 7);
        // Row 0: [-1, -1, 0, 0, 0, 0, 0]
        assert_eq!(meta.row(0), &[ROOT_PARENT, ROOT_PARENT, 0, 0, 0, 0, 0]);
    }

    // ── last_accepted_inter_slot: kernel slot mapping for commit ──

    #[test]
    fn last_inter_slot_empty_returns_root_slot_zero() {
        // num_accepted=0 (only the prefix bonus was "accepted"): the kernel
        // slot holding the canonical post-step state is slot 0 (root).
        assert_eq!(last_accepted_inter_slot(&[]), 0);
    }

    #[test]
    fn last_inter_slot_chain_returns_length_n() {
        // Chain accept: indices [1,2,3,4]. Kernel slot of last accepted state
        // is compact_index = 4. Matches legacy `total_accepted - 1 = 4`.
        assert_eq!(last_accepted_inter_slot(&[1, 2, 3, 4]), 4);
    }

    #[test]
    fn last_inter_slot_sparse_tree_returns_largest_compact_index() {
        // Tree accept: walk crossed forks, accepted compact indices are
        // [1, 4, 7]. Legacy `len - 1 = 2` would read kernel slot 2 (WRONG).
        // Correct: read slot 7 = the actual last accepted state.
        assert_eq!(last_accepted_inter_slot(&[1, 4, 7]), 7);
    }

    #[test]
    fn last_inter_slot_single_accept() {
        // Only the first draft was accepted (chain-prefix safe path may
        // truncate to length 1). Kernel slot = compact_index = 1.
        assert_eq!(last_accepted_inter_slot(&[1]), 1);
    }

    #[test]
    fn last_inter_slot_matches_greedy_sample_chain_case() {
        // End-to-end: greedy_sample_ddtree on a flat chain should yield
        // [1,2,3,4]; the helper should agree with chain arithmetic.
        let r = req(&[10, 20, 30, 40], &[-1, 0, 1, 2]);
        let argmax = vec![10u32, 20, 30, 40, 99];
        let s = greedy_sample_ddtree(&r, &argmax).unwrap();
        assert_eq!(s.accepted_compact_indices, vec![1, 2, 3, 4]);
        let slot = last_accepted_inter_slot(&s.accepted_compact_indices);
        assert_eq!(slot, 4);
        // Chain arithmetic equivalence: num_accepted == 4 → legacy slot == 4.
        assert_eq!(slot, s.accepted_compact_indices.len());
    }

    #[test]
    fn last_inter_slot_matches_greedy_sample_branch_divergence_case() {
        // Branch-divergent tree truncates to flat prefix [1]; helper picks
        // slot 1 (NOT 0). Demonstrates that even after the flat-safe
        // adapter, the slot read still matches the compact index, not the
        // sequential count.
        let r = req(&[10, 11, 20], &[-1, -1, 0]);
        let argmax = vec![10u32, 20, 0, 999];
        let s = greedy_sample_ddtree(&r, &argmax).unwrap();
        assert_eq!(s.accepted_compact_indices, vec![1]);
        assert_eq!(last_accepted_inter_slot(&s.accepted_compact_indices), 1);
    }

    #[test]
    fn last_inter_slot_synthetic_non_contiguous_regression() {
        // REGRESSION (M4B v2 prep): simulate a future tree-aware adapter
        // that emits a *non-contiguous* accept path (e.g. once branch-mode
        // landing relaxes the flat-prefix-only contract). Helper MUST pick
        // the largest compact index, not the count.
        let non_contig = vec![0usize, 1, 4, 7];
        // (Slot 0 included here to mimic root+drafts; semantically slot 0
        // is the root, but the helper still picks the max == 7.)
        assert_eq!(last_accepted_inter_slot(&non_contig), 7);

        // And a typical sparse pattern without leading root entry.
        let sparse = vec![1usize, 4, 7];
        assert_eq!(last_accepted_inter_slot(&sparse), 7);
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

    // ══════════════════════════════════════════════════════════════════════
    // TASK #29 — sparse (non-contiguous) fork-commit frame-mapping regression
    //
    // These tests exercise the compact→kernel(DFS) frame translation that the
    // KV-compaction and ctx-hidden-append paths were MISSING, which caused the
    // counting-md5 corruption with ATLAS_DFLASH_TREE_COMMIT=1 on a forked
    // accept when ATLAS_DDTREE_DFS_REORDER=1 (the portfolio / deep-DDTree
    // production config). Everything here is pure CPU (no GPU / no serve).
    // ══════════════════════════════════════════════════════════════════════

    /// Empty inv_perm (chain mode / DFS disabled) ⇒ the map is the identity so
    /// the flat path is byte-for-byte unchanged.
    #[test]
    fn compact_to_kernel_identity_when_no_dfs() {
        assert_eq!(compact_to_kernel_slot(0, &[]), 0);
        assert_eq!(compact_to_kernel_slot(3, &[]), 3);
        assert_eq!(map_compact_path_to_kernel(&[1, 2, 3, 4], &[]), vec![1, 2, 3, 4]);
    }

    /// Out-of-range compact index passes through unmapped (defensive; never
    /// panics on a malformed payload).
    #[test]
    fn compact_to_kernel_out_of_range_passthrough() {
        let inv = vec![0usize, 1, 2];
        assert_eq!(compact_to_kernel_slot(9, &inv), 9);
    }

    /// DFS reorder for two root branches (from `dfs_reorder_handles_two_root_branches`):
    /// kernel_parents = [-1,0,0,1,2] → perm=[0,1,3,2,4], inv=[0,1,3,2,4].
    /// A compact index `c` (== old kernel slot) maps to DFS slot inv[c].
    #[test]
    fn compact_to_kernel_uses_dfs_inv_perm() {
        let kp = vec![-1i32, 0, 0, 1, 2];
        let (_perm, inv, _d) = dfs_reorder(&kp);
        assert_eq!(inv, vec![0, 1, 3, 2, 4]);
        // Portfolio-style B branch sits at compact slots {2,4}; under DFS
        // reorder those live at kernel slots inv[2]=3, inv[4]=4.
        assert_eq!(compact_to_kernel_slot(2, &inv), 3);
        assert_eq!(compact_to_kernel_slot(4, &inv), 4);
        assert_eq!(map_compact_path_to_kernel(&[2, 4], &inv), vec![3, 4]);
    }

    // ── KV compaction move planner ──

    /// A truly contiguous accepted path in the kernel frame needs no moves.
    #[test]
    fn kv_plan_contiguous_is_noop() {
        // chain accept [1,2,3,4], DFS off → sources == destinations.
        let moves = plan_kv_compaction_moves(&[1, 2, 3, 4], &[], 100, 17);
        assert!(moves.is_empty());
    }

    /// Fork-at-root, DFS OFF: accepted compact path [4,5] (portfolio B branch).
    /// Kernel frame == compact frame here. KV lives at pre+4, pre+5; must be
    /// gathered to the contiguous home pre+1, pre+2. No src is a dst here
    /// (srcs {4,5}, dsts {1,2}) → simple ascending-dst order, lossless.
    #[test]
    fn kv_plan_fork_at_root_no_dfs() {
        let moves = plan_kv_compaction_moves(&[4, 5], &[], 100, 17);
        assert_eq!(
            moves,
            vec![
                KvMove { src_pos: 104, dst_pos: 101 },
                KvMove { src_pos: 105, dst_pos: 102 },
            ]
        );
    }

    /// The concrete corruption case: fork-at-root WITH DFS reorder on. The
    /// portfolio B branch is at compact {4,5}, but under DFS reorder its KV is
    /// physically at kernel slots inv[4], inv[5]. The planner MUST read from the
    /// mapped kernel positions, not pre+4/pre+5 (the pre-fix bug read the wrong
    /// bytes → md5 corruption).
    #[test]
    fn kv_plan_fork_with_dfs_maps_source() {
        // Construct a payload where compact 4,5 permute to different kernel
        // slots. kernel_parents for portfolio [A0,A1,A2,B0,B1] (3+2), bonus
        // first: [-1, 0,1,2, 0,4]  (A chain off bonus; B0 off bonus; B1 off B0)
        let kp = vec![-1i32, 0, 1, 2, 0, 4];
        let (_perm, inv, _d) = dfs_reorder(&kp);
        // DFS visits: 0 → 1 → 2 → 3 (A chain) → 4 → 5 (B chain).
        // Here that is already identity-ish; force a non-identity by swapping
        // the child visitation via a shape the reorder actually permutes.
        // Use the two-root-branch shape instead for a genuine permutation:
        let kp2 = vec![-1i32, 0, 0, 1, 2]; // inv = [0,1,3,2,4]
        let (_p2, inv2, _d2) = dfs_reorder(&kp2);
        assert_eq!(inv2, vec![0, 1, 3, 2, 4]);
        let _ = inv; // (first shape kept for documentation)
        // Accept the branch at compact {2,4}. Kernel positions: inv2[2]=3,
        // inv2[4]=4 → sources pre+3, pre+4. Homes pre+1, pre+2.
        let moves = plan_kv_compaction_moves(&[2, 4], &inv2, 100, 5);
        // Sources {103,104}, dsts {101,102} — no overlap → ascending dst.
        assert_eq!(
            moves,
            vec![
                KvMove { src_pos: 103, dst_pos: 101 },
                KvMove { src_pos: 104, dst_pos: 102 },
            ]
        );
    }

    /// Overlap hazard: a source position is ALSO a destination of another move.
    /// The planner must evacuate the conflicting source to a free tail scratch
    /// slot BEFORE placing, so nothing is clobbered. Verify (a) every wanted
    /// destination ends up sourced from the correct original byte, and (b) no
    /// placement reads a position an earlier placement already overwrote.
    #[test]
    fn kv_plan_overlap_is_evacuated_safely() {
        // accepted compact [2, 1] (walk visited kernel slots that map so that
        // src of move-0 == dst of move-1). Use inv_perm to force it: compact 2
        // → kernel 1, compact 1 → kernel 2. pre=100, k=8.
        //   step0: compact 2 → kernel 1 → src 101, dst 101+? (j=0 → dst 101)
        // To actually create dst==src conflict, craft:
        //   accepted [c0=3, c1=1], inv maps 3→1, 1→2.
        //   step0 (j0): src=pre+1(=101), dst=pre+1(=101) → identity, dropped.
        // Simplest deterministic conflict: identity inv, accepted [2,1]:
        //   step0 j0: src pre+2 (102) dst pre+1 (101)
        //   step1 j1: src pre+1 (101) dst pre+2 (102)
        // 101 is both src(step1) and dst(step0) → conflict → must evacuate.
        let moves = plan_kv_compaction_moves(&[2, 1], &[], 100, 8);
        // Simulate the moves on a position→value map to prove losslessness.
        // Initial: pos p holds the byte tagged p.
        use std::collections::HashMap;
        let mut mem: HashMap<usize, usize> = (100..108).map(|p| (p, p)).collect();
        for m in &moves {
            let val = *mem.get(&m.src_pos).expect("src populated");
            mem.insert(m.dst_pos, val);
        }
        // Desired final: dst pre+1(101) holds compact-2's byte (originally at
        // kernel pos pre+2=102), dst pre+2(102) holds compact-1's byte (pos 101).
        assert_eq!(mem[&101], 102, "home[1] must carry the byte from pos 102");
        assert_eq!(mem[&102], 101, "home[2] must carry the byte from pos 101");
    }

    /// Larger scramble: a 3-cycle of positions under an inv_perm permutation,
    /// proven lossless by simulation.
    #[test]
    fn kv_plan_three_cycle_lossless() {
        // accepted compact [1,2,3]; inv_perm maps 1→3, 2→1, 3→2 (a 3-cycle over
        // kernel slots {1,2,3}). Homes are pre+1,pre+2,pre+3.
        let inv = vec![0usize, 3, 1, 2]; // index by compact idx
        let pre = 100;
        let k = 8;
        let moves = plan_kv_compaction_moves(&[1, 2, 3], &inv, pre, k);
        use std::collections::HashMap;
        let mut mem: HashMap<usize, usize> = (pre..pre + k).map(|p| (p, p)).collect();
        for m in &moves {
            let val = *mem.get(&m.src_pos).expect("src populated");
            mem.insert(m.dst_pos, val);
        }
        // home[j+1] must hold the byte from kernel slot inv[accepted[j]].
        assert_eq!(mem[&(pre + 1)], pre + 3, "home1 <- kernel3 (inv[1])");
        assert_eq!(mem[&(pre + 2)], pre + 1, "home2 <- kernel1 (inv[2])");
        assert_eq!(mem[&(pre + 3)], pre + 2, "home3 <- kernel2 (inv[3])");
    }

    /// End-to-end frame-consistency: given a portfolio payload where the target
    /// rides chain B, the FULL sampler's accepted compact indices, when mapped
    /// through inv_perm, address DISTINCT kernel positions (no aliasing) — the
    /// invariant the KV gather and ctx-hidden append rely on.
    #[test]
    fn portfolio_b_commit_maps_to_distinct_kernel_slots() {
        let p = build_portfolio_payload(&[10, 20, 30], &[11, 21], 8);
        // kernel_parents = full_parent_ids_from_payload (bonus-prefixed).
        let kernel_parents = full_parent_ids_from_payload(&p).unwrap();
        let (_perm, inv, _d) = dfs_reorder(&kernel_parents);
        let r = req(&p.tree_token_ids, &p.parent_indices);
        let argmax = vec![11u32, 0, 0, 0, 21, 77]; // target rides B
        let s = greedy_sample_ddtree_full(&r, &argmax).unwrap();
        assert_eq!(s.accepted_compact_indices, vec![4, 5]);
        let kernel = map_compact_path_to_kernel(&s.accepted_compact_indices, &inv);
        // Distinct kernel slots (no aliasing → a valid permutation to gather).
        let uniq: std::collections::HashSet<_> = kernel.iter().copied().collect();
        assert_eq!(uniq.len(), kernel.len(), "kernel slots must be distinct");
        // And the SSM commit slot (max compact) maps through inv identically to
        // commit_verify_state_async_with_slot's mapping.
        let last = last_accepted_inter_slot(&s.accepted_compact_indices);
        assert_eq!(compact_to_kernel_slot(last, &inv), *kernel.last().unwrap());
    }
}
