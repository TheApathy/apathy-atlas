// SPDX-License-Identifier: AGPL-3.0-only

//! Candidate and lattice types (guide §5.2–§5.3, Layers B/C).

use std::collections::HashSet;
use std::fmt;

/// Provenance of a proposal source. Variant order is the stable source
/// priority for tie-breaking; do not reorder without updating the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProposalSourceId {
    /// The existing flat neural block drafter (DFlash / DSpark top-1).
    NeuralFlat,
    /// DFlash2 top-k candidate walk.
    Dflash2TopK,
    /// Exact suffix / SAM retrieval.
    Sam,
    /// Oilbird / key-distance retrieval.
    KeyRetrieval,
    /// Target-authored echo tail.
    Echo,
    /// Discarded drafter-tail recycle.
    Recycle,
    /// Future AST/symbol retrieval.
    Ast,
    /// Placeholder for unmapped/unknown provenance.
    Unknown,
}

impl ProposalSourceId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NeuralFlat => "neural-flat",
            Self::Dflash2TopK => "dflash2-topk",
            Self::Sam => "sam",
            Self::KeyRetrieval => "key-retrieval",
            Self::Echo => "echo",
            Self::Recycle => "recycle",
            Self::Ast => "ast",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for ProposalSourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which ancestor a candidate is conditioned on. Parallel block-drafter rows
/// are conditioned on their trained top-1 prefix, not arbitrary parents; the
/// planner must not extend alternatives under parents they were not scored on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandidateParent {
    /// Depth 1: conditioned on the verified bonus/root token.
    Root,
    /// Conditioned on the top-1 spine token at the given depth (>= 1).
    Spine(u16),
    /// An explicitly parent-conditioned alternative token id.
    Alternative(u32),
}

/// Bit flags annotating a candidate's origin and treatment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateFlags(u16);

impl CandidateFlags {
    pub const NONE: Self = Self(0);
    pub const FROM_ECHO: Self = Self(1 << 0);
    pub const FROM_RETRIEVAL: Self = Self(1 << 1);
    pub const REPAIR_SCORED: Self = Self(1 << 2);
    pub const LEAF_ONLY: Self = Self(1 << 3);

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// A single proposal token with full provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateToken {
    pub token_id: u32,
    pub depth: u16,
    pub parent: CandidateParent,
    pub source: ProposalSourceId,
    pub local_score: f32,
    pub calibrated_p: f32,
    pub retrieval_distance: u16,
    pub flags: CandidateFlags,
}

impl CandidateToken {
    /// Lattice merge key: two candidates are the same node when they agree on
    /// (parent, depth, token id).
    pub fn dedup_key(&self) -> (CandidateParent, u16, u32) {
        (self.parent, self.depth, self.token_id)
    }
}

/// Evidence describing how a source produced its candidates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceEvidence {
    None,
    /// Exact literal suffix match of `match_len` tokens.
    ExactSuffix {
        match_len: usize,
    },
    /// Key-distance retrieval at distance `d`.
    KeyDistance {
        distance: u16,
    },
    /// Target-authored echo tail of `tail_len` tokens.
    EchoTail {
        tail_len: usize,
    },
    /// Neural block forward (no retrieval).
    NeuralBlock,
}

/// One source's collected candidates, indexed by depth (1-based).
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateSet {
    pub source: ProposalSourceId,
    pub candidates_by_depth: Vec<Vec<CandidateToken>>,
    pub source_cost_us: u32,
    pub evidence: SourceEvidence,
}

impl CandidateSet {
    pub fn empty(source: ProposalSourceId) -> Self {
        Self {
            source,
            candidates_by_depth: Vec::new(),
            source_cost_us: 0,
            evidence: SourceEvidence::None,
        }
    }

    /// Total candidate count across depths.
    pub fn len(&self) -> usize {
        self.candidates_by_depth.iter().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Merge equal `(parent, depth, token)` nodes, retaining first-seen order.
/// This is the bounded-DAG merge of guide §5.3: no duplicate nodes, stable
/// source priority (first-seen wins), lowest token id as the final tie-break
/// (a caller pre-sorts candidates by token id before merging).
pub fn merge_equal_candidates(
    left: &[CandidateToken],
    right: &[CandidateToken],
) -> Vec<CandidateToken> {
    let mut seen: HashSet<(CandidateParent, u16, u32)> = HashSet::new();
    let mut out = Vec::with_capacity(left.len() + right.len());
    for token in left.iter().chain(right) {
        if seen.insert(token.dedup_key()) {
            out.push(token.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(depth: u16, token_id: u32) -> CandidateToken {
        CandidateToken {
            token_id,
            depth,
            parent: if depth == 1 {
                CandidateParent::Root
            } else {
                CandidateParent::Spine(depth - 1)
            },
            source: ProposalSourceId::NeuralFlat,
            local_score: 0.0,
            calibrated_p: 0.0,
            retrieval_distance: 0,
            flags: CandidateFlags::NONE,
        }
    }

    #[test]
    fn merge_dedupes_equal_nodes_and_keeps_first_seen_order() {
        let a = vec![token(1, 42), token(2, 7)];
        let b = vec![token(2, 7), token(2, 9)];
        let merged = merge_equal_candidates(&a, &b);
        assert_eq!(merged.len(), 3);
        assert_eq!(
            merged
                .iter()
                .map(|t| (t.depth, t.token_id))
                .collect::<Vec<_>>(),
            vec![(1, 42), (2, 7), (2, 9)]
        );
    }

    #[test]
    fn dedup_key_ignores_source_and_score() {
        let mut t1 = token(2, 7);
        t1.source = ProposalSourceId::Echo;
        let t2 = token(2, 7);
        assert_eq!(t1.dedup_key(), t2.dedup_key());
    }

    #[test]
    fn flags_bit_ops_are_consistent() {
        let mut f = CandidateFlags::NONE;
        f.insert(CandidateFlags::FROM_ECHO);
        f.insert(CandidateFlags::REPAIR_SCORED);
        assert!(f.contains(CandidateFlags::FROM_ECHO));
        assert!(f.contains(CandidateFlags::REPAIR_SCORED));
        assert!(!f.contains(CandidateFlags::FROM_RETRIEVAL));
    }

    #[test]
    fn empty_candidate_set_is_empty() {
        let set = CandidateSet::empty(ProposalSourceId::Sam);
        assert!(set.is_empty());
        assert_eq!(set.source, ProposalSourceId::Sam);
    }
}
