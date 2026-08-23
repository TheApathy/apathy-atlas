// SPDX-License-Identifier: AGPL-3.0-only

//! Candidate-source trait and registry (guide §5.2, Layer B).

use anyhow::Result;

use super::candidate::{CandidateSet, ProposalSourceId, SourceEvidence};
use super::observation::ProposalObservation;

/// Execution context handed to a source during collection. Phase 1 carries
/// only the bounds the flat neural adapter needs; Phase 2 passes the real
/// `draft_budget::DflashDraftBudget` and the device/stream handles alongside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateContext {
    /// Physical verify-token capacity (root + drafts) this cycle.
    pub physical_verify_k: usize,
    /// Trained drafter width ceiling.
    pub trained_width: usize,
}

impl CandidateContext {
    pub fn new(physical_verify_k: usize, trained_width: usize) -> Self {
        Self {
            physical_verify_k,
            trained_width,
        }
    }
}

/// A proposal source: collects candidates without deciding the final route.
/// The planner, not the source, chooses width/topology.
pub trait CandidateSource {
    fn source_id(&self) -> ProposalSourceId;
    fn collect(
        &self,
        observation: &ProposalObservation,
        ctx: &mut CandidateContext,
    ) -> Result<CandidateSet>;
}

/// Deterministic, insertion-ordered source registry. Source order is the
/// priority for tie-breaking (guide §6: stable source priority, stable score
/// tie-breaking, lowest token id as the final tie-breaker).
#[derive(Default)]
pub struct SourceRegistry {
    sources: Vec<Box<dyn CandidateSource + Send + Sync>>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, source: Box<dyn CandidateSource + Send + Sync>) {
        self.sources.push(source);
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn ids(&self) -> Vec<ProposalSourceId> {
        self.sources.iter().map(|s| s.source_id()).collect()
    }
}

/// Adapter for the existing flat neural proposal. **Not wired in Phase 1** —
/// it exists to exercise the trait/registry contract and to mark the single
/// source the legacy `propose.rs` path will be wrapped as in Phase 2.
#[derive(Debug, Clone)]
pub struct NeuralFlat {
    trained_width: usize,
}

impl NeuralFlat {
    pub fn new(trained_width: usize) -> Self {
        Self { trained_width }
    }

    pub fn trained_width(&self) -> usize {
        self.trained_width
    }
}

impl CandidateSource for NeuralFlat {
    fn source_id(&self) -> ProposalSourceId {
        ProposalSourceId::NeuralFlat
    }

    fn collect(
        &self,
        _observation: &ProposalObservation,
        ctx: &mut CandidateContext,
    ) -> Result<CandidateSet> {
        // Phase 1: no real collection. Produce an empty candidate set whose
        // provenance is correct so the facade/registry plumbing is testable.
        debug_assert!(self.trained_width <= ctx.trained_width || ctx.trained_width == 0);
        Ok(CandidateSet {
            source: ProposalSourceId::NeuralFlat,
            candidates_by_depth: vec![Vec::new(); self.trained_width],
            source_cost_us: 0,
            evidence: SourceEvidence::NeuralBlock,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_preserves_insertion_order_of_ids() {
        let mut registry = SourceRegistry::new();
        registry.push(Box::new(NeuralFlat::new(12)));
        registry.push(Box::new(NeuralFlat::new(4)));
        assert_eq!(
            registry.ids(),
            vec![ProposalSourceId::NeuralFlat, ProposalSourceId::NeuralFlat]
        );
        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
    }

    #[test]
    fn neural_flat_collects_empty_set_with_correct_source() {
        let source = NeuralFlat::new(12);
        let mut ctx = CandidateContext::new(13, 12);
        let obs = ProposalObservation {
            last_token: 0,
            absolute_position: 0,
            committed_tokens: &[],
            recent_accepts: &[],
            previous_source: ProposalSourceId::Unknown,
            previous_width: 0,
            previous_cycle_us: 0,
            context_bucket: super::super::observation::ContextBucket::Unknown,
            grammar_active: false,
            thinking_active: false,
            tree_capable: false,
            physical_verify_k: 13,
        };
        let set = source.collect(&obs, &mut ctx).unwrap();
        assert_eq!(set.source, ProposalSourceId::NeuralFlat);
        assert!(set.is_empty());
        assert_eq!(set.candidates_by_depth.len(), 12);
    }
}
