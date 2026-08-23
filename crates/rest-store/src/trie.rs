// SPDX-License-Identifier: AGPL-3.0-only

//! Frequency-weighted continuation trie.
//!
//! Given the corpus positions where the context suffix matched, the
//! continuations that followed each match are merged into a trie whose
//! edge weights are occurrence counts. The trie is then pruned to a node
//! budget by a best-first walk: repeatedly admit the highest-count node
//! whose parent is already admitted.
//!
//! Admitting parents before children is not a nicety — it is what makes
//! the emitted node list satisfy the `DDTreePayload` invariant that every
//! `parent_indices[i]` is either `-1` or strictly less than `i`
//! (`DflashDraftBudget::validate_tree`). See `PHASE2.md`.

use std::collections::BinaryHeap;

/// Tunables for [`build_draft_trie`].
#[derive(Debug, Clone, Copy)]
pub struct TrieParams {
    /// Maximum continuation length below the root.
    pub depth: usize,
    /// Maximum number of non-root nodes in the emitted tree.
    pub max_nodes: usize,
    /// Token id that terminates a continuation (document separator).
    /// Continuations never cross a document boundary.
    pub sep_token: u32,
}

impl Default for TrieParams {
    fn default() -> Self {
        Self {
            depth: crate::DEFAULT_DEPTH,
            max_nodes: crate::DEFAULT_MAX_NODES,
            sep_token: 0,
        }
    }
}

/// One admitted draft node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftNode {
    /// Proposed token id.
    pub token: u32,
    /// Index of this node's parent in [`DraftTree::nodes`], or `-1` when
    /// the parent is the (implicit) root — the sequence's last token.
    pub parent: i32,
    /// How many corpus occurrences followed this path.
    pub count: u32,
    /// Distance from the root; the root's children have depth 1.
    pub depth: u16,
}

/// A pruned, frequency-ordered continuation tree.
///
/// The implicit root is the sequence's current last token and is *not* a
/// member of `nodes`, matching the `DDTreePayload` convention where slot 0
/// is the bonus/root row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DraftTree {
    /// Nodes in admission order: every node's parent precedes it.
    pub nodes: Vec<DraftNode>,
    /// Length of the context suffix that produced this tree.
    pub match_len: usize,
    /// Number of corpus occurrences the tree was built from.
    pub occurrences: usize,
}

impl DraftTree {
    /// Whether the tree carries no drafts.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of draft nodes (excluding the implicit root).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Split into the `(tree_token_ids, parent_indices)` pair that
    /// `DDTreePayload` carries.
    pub fn to_payload_parts(&self) -> (Vec<u32>, Vec<i32>) {
        (
            self.nodes.iter().map(|n| n.token).collect(),
            self.nodes.iter().map(|n| n.parent).collect(),
        )
    }

    /// The highest-count root-to-leaf chain, as a flat draft sequence.
    ///
    /// This is what a flat (non-tree) verifier would consume, and the
    /// fallback whenever the tree path declines a payload.
    pub fn spine(&self) -> Vec<u32> {
        let mut chain = Vec::new();
        let mut parent: i32 = -1;
        loop {
            let best = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.parent == parent)
                .max_by_key(|(i, n)| (n.count, std::cmp::Reverse(*i)));
            match best {
                Some((idx, node)) => {
                    chain.push(node.token);
                    parent = idx as i32;
                }
                None => return chain,
            }
        }
    }

    /// Length of the longest root-to-node path that is a prefix of
    /// `actual` — exactly what a tree verifier would accept if the target
    /// went on to emit `actual`.
    pub fn longest_accepted_path(&self, actual: &[u32]) -> usize {
        // depth_ok[i] = the node is reachable by a path matching `actual`.
        let mut matched = vec![false; self.nodes.len()];
        let mut best = 0usize;
        for (i, node) in self.nodes.iter().enumerate() {
            let parent_ok = node.parent < 0 || matched[node.parent as usize];
            let d = node.depth as usize;
            // Parents precede children, so `matched[parent]` is final here.
            if parent_ok && d <= actual.len() && actual[d - 1] == node.token {
                matched[i] = true;
                best = best.max(d);
            }
        }
        best
    }
}

/// Internal mutable trie node used only during construction.
struct BuildNode {
    token: u32,
    parent: usize,
    count: u32,
    depth: usize,
    children: Vec<usize>,
    /// Insertion order, used purely to make pruning deterministic.
    order: u32,
}

/// Build a pruned continuation trie from corpus match positions.
///
/// `corpus` is the full token stream; each entry of `positions` is the
/// start of a matched suffix, so the continuation begins at
/// `position + match_len`. Returns `None` when no occurrence had a
/// continuation (every match sat at a document or corpus boundary).
pub fn build_draft_trie(
    corpus: &[u32],
    positions: &[u32],
    match_len: usize,
    params: TrieParams,
) -> Option<DraftTree> {
    if params.depth == 0 || params.max_nodes == 0 {
        return None;
    }

    // Node 0 is the implicit root; it is never emitted.
    let mut nodes = vec![BuildNode {
        token: 0,
        parent: 0,
        count: 0,
        depth: 0,
        children: Vec::new(),
        order: 0,
    }];
    let mut order: u32 = 0;

    for &pos in positions {
        let start = pos as usize + match_len;
        let end = (start + params.depth).min(corpus.len());
        if start >= end {
            continue;
        }
        let mut cur = 0usize;
        for &tok in &corpus[start..end] {
            if tok == params.sep_token {
                break;
            }
            let existing = nodes[cur]
                .children
                .iter()
                .copied()
                .find(|&c| nodes[c].token == tok);
            cur = match existing {
                Some(c) => c,
                None => {
                    order += 1;
                    let depth = nodes[cur].depth + 1;
                    nodes.push(BuildNode {
                        token: tok,
                        parent: cur,
                        count: 0,
                        depth,
                        children: Vec::new(),
                        order,
                    });
                    let new = nodes.len() - 1;
                    nodes[cur].children.push(new);
                    new
                }
            };
            nodes[cur].count += 1;
        }
    }

    if nodes.len() == 1 {
        return None;
    }

    // Best-first prune. The heap key ranks by count, then prefers shallow
    // nodes (a deep node is worth less than a shallow one at equal count
    // because fewer verify slots ride on it), then earliest insertion.
    #[derive(PartialEq, Eq)]
    struct Key {
        count: u32,
        neg_depth: std::cmp::Reverse<usize>,
        neg_order: std::cmp::Reverse<u32>,
        idx: usize,
    }
    impl Ord for Key {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            (self.count, self.neg_depth, self.neg_order).cmp(&(
                other.count,
                other.neg_depth,
                other.neg_order,
            ))
        }
    }
    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    let key_of = |nodes: &Vec<BuildNode>, idx: usize| Key {
        count: nodes[idx].count,
        neg_depth: std::cmp::Reverse(nodes[idx].depth),
        neg_order: std::cmp::Reverse(nodes[idx].order),
        idx,
    };

    let mut heap: BinaryHeap<Key> = BinaryHeap::new();
    for &c in &nodes[0].children {
        heap.push(key_of(&nodes, c));
    }

    // build index -> emitted index; -1 marks "root" for the root's children.
    let mut emitted = vec![-1i32; nodes.len()];
    let mut out: Vec<DraftNode> = Vec::with_capacity(params.max_nodes);

    while out.len() < params.max_nodes {
        let Some(key) = heap.pop() else { break };
        let idx = key.idx;
        let parent = nodes[idx].parent;
        // Root children keep parent = -1; others point at the emitted slot.
        let parent_slot = if parent == 0 { -1 } else { emitted[parent] };
        debug_assert!(
            parent == 0 || parent_slot >= 0,
            "best-first admission must emit a parent before its child"
        );
        emitted[idx] = out.len() as i32;
        out.push(DraftNode {
            token: nodes[idx].token,
            parent: parent_slot,
            count: nodes[idx].count,
            depth: nodes[idx].depth as u16,
        });
        for &c in &nodes[idx].children {
            heap.push(key_of(&nodes, c));
        }
    }

    if out.is_empty() {
        return None;
    }
    Some(DraftTree {
        nodes: out,
        match_len,
        occurrences: positions.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(depth: usize, max_nodes: usize) -> TrieParams {
        TrieParams {
            depth,
            max_nodes,
            sep_token: 0,
        }
    }

    /// Every emitted node must satisfy the DDTree payload invariant.
    fn assert_payload_invariants(tree: &DraftTree, budget: usize) {
        let (toks, parents) = tree.to_payload_parts();
        assert_eq!(toks.len(), parents.len());
        assert!(!toks.is_empty());
        assert!(toks.len() <= budget);
        for (child, &p) in parents.iter().enumerate() {
            assert!(
                p == -1 || (p >= 0 && (p as usize) < child),
                "node {child} has parent {p}; must be -1 or < {child}"
            );
        }
    }

    #[test]
    fn single_occurrence_produces_a_chain() {
        // corpus: [1,2,3, 7,8,9]; match [1,2,3] at 0, continuation 7,8,9
        let corpus = vec![1, 2, 3, 7, 8, 9];
        let tree = build_draft_trie(&corpus, &[0], 3, params(16, 16)).unwrap();
        assert_eq!(tree.spine(), vec![7, 8, 9]);
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.nodes[0].parent, -1);
        assert_eq!(tree.nodes[1].parent, 0);
        assert_eq!(tree.nodes[2].parent, 1);
        assert_payload_invariants(&tree, 16);
    }

    #[test]
    fn frequency_orders_the_spine() {
        // Suffix [5] continues with 9 twice and with 4 once.
        let corpus = vec![5, 9, 5, 9, 5, 4];
        let tree = build_draft_trie(&corpus, &[0, 2, 4], 1, params(1, 16)).unwrap();
        // Both branches admitted; the count-2 branch is first and is the spine.
        assert_eq!(tree.nodes[0].token, 9);
        assert_eq!(tree.nodes[0].count, 2);
        assert_eq!(tree.nodes[1].token, 4);
        assert_eq!(tree.spine(), vec![9]);
        assert_payload_invariants(&tree, 16);
    }

    #[test]
    fn branches_share_a_prefix() {
        // Two continuations agree on the first token, diverge on the second.
        let corpus = vec![1, 7, 8, 1, 7, 9];
        let tree = build_draft_trie(&corpus, &[0, 3], 1, params(4, 16)).unwrap();
        assert_eq!(tree.nodes[0].token, 7);
        assert_eq!(tree.nodes[0].count, 2);
        let children: Vec<u32> = tree
            .nodes
            .iter()
            .filter(|n| n.parent == 0)
            .map(|n| n.token)
            .collect();
        assert_eq!(children.len(), 2);
        assert!(children.contains(&8) && children.contains(&9));
        assert_payload_invariants(&tree, 16);
    }

    #[test]
    fn respects_node_cap_and_keeps_parents() {
        let corpus: Vec<u32> = (1..=40).collect();
        let tree = build_draft_trie(&corpus, &[0], 3, params(16, 4)).unwrap();
        assert_eq!(tree.len(), 4);
        assert_payload_invariants(&tree, 4);
    }

    #[test]
    fn respects_depth_cap() {
        let corpus: Vec<u32> = (1..=40).collect();
        let tree = build_draft_trie(&corpus, &[0], 3, params(5, 64)).unwrap();
        assert_eq!(tree.len(), 5);
        assert_eq!(tree.nodes.iter().map(|n| n.depth).max(), Some(5));
    }

    #[test]
    fn continuation_stops_at_separator() {
        let corpus = vec![1, 2, 3, 7, 0, 9];
        let tree = build_draft_trie(&corpus, &[0], 3, params(16, 16)).unwrap();
        assert_eq!(tree.spine(), vec![7]);
    }

    #[test]
    fn no_continuation_returns_none() {
        // Match sits at the very end of the corpus.
        let corpus = vec![1, 2, 3];
        assert!(build_draft_trie(&corpus, &[0], 3, params(16, 16)).is_none());
        // Continuation is a separator only.
        let corpus = vec![1, 2, 3, 0];
        assert!(build_draft_trie(&corpus, &[0], 3, params(16, 16)).is_none());
    }

    #[test]
    fn zero_budget_returns_none() {
        let corpus = vec![1, 2, 3, 7, 8];
        assert!(build_draft_trie(&corpus, &[0], 3, params(0, 16)).is_none());
        assert!(build_draft_trie(&corpus, &[0], 3, params(16, 0)).is_none());
    }

    #[test]
    fn longest_accepted_path_walks_the_right_branch() {
        // Spine 7,8,9 (count 2) plus a sibling 7,8,5 (count 1).
        let corpus = vec![1, 7, 8, 9, 1, 7, 8, 9, 1, 7, 8, 5];
        // Depth 3 so continuations stop before wrapping into the next repeat.
        let tree = build_draft_trie(&corpus, &[0, 4, 8], 1, params(3, 16)).unwrap();
        assert_eq!(tree.spine(), vec![7, 8, 9]);
        // The target emitted the LOW-frequency branch — tree verify still
        // accepts all 3, which a flat spine draft would not.
        assert_eq!(tree.longest_accepted_path(&[7, 8, 5, 1]), 3);
        assert_eq!(tree.longest_accepted_path(&[7, 8, 9, 1]), 3);
        assert_eq!(tree.longest_accepted_path(&[7, 4]), 1);
        assert_eq!(tree.longest_accepted_path(&[4]), 0);
        assert_eq!(tree.longest_accepted_path(&[]), 0);
    }

    #[test]
    fn longest_accepted_path_ignores_unreachable_matches() {
        // A node whose token matches at its depth but whose parent did not
        // match must not count.
        let corpus = vec![1, 7, 8, 1, 4, 8];
        let tree = build_draft_trie(&corpus, &[0, 3], 1, params(4, 16)).unwrap();
        // actual = [4, 8] -> the "8" under parent "7" is unreachable, but
        // the "8" under parent "4" is reachable, so 2.
        assert_eq!(tree.longest_accepted_path(&[4, 8]), 2);
        // actual = [9, 8] -> nothing matches at depth 1, so 0.
        assert_eq!(tree.longest_accepted_path(&[9, 8]), 0);
    }
}
