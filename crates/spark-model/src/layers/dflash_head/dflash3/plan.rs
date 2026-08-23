// SPDX-License-Identifier: AGPL-3.0-only

//! Plan and outcome types (guide §5.5–§5.7, Layers E/F/G).

use super::candidate::ProposalSourceId;

/// Topology of a compiled proposal plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanMode {
    Off,
    Flat,
    Forest,
    Tree,
}

impl PlanMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Flat => "flat",
            Self::Forest => "forest",
            Self::Tree => "tree",
        }
    }
}

/// Why the planner chose this plan. Fail-closed reasons must always be
/// permitted to narrow the plan to a safe fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanReason {
    FixedDefault,
    NeuralFlatFallback,
    RetrievalHit,
    EchoHit,
    RouterChoice,
    FailClosed,
}

impl PlanReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FixedDefault => "fixed-default",
            Self::NeuralFlatFallback => "neural-flat-fallback",
            Self::RetrievalHit => "retrieval-hit",
            Self::EchoHit => "echo-hit",
            Self::RouterChoice => "router-choice",
            Self::FailClosed => "fail-closed",
        }
    }
}

/// The planner's chosen action: source mix, width, and topology, plus the
/// expected economics used to pick it. The objective is complete-cycle
/// useful-token utility, never acceptance percentage alone.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftPlan {
    pub mode: PlanMode,
    pub primary_source: ProposalSourceId,
    pub secondary_sources: Vec<ProposalSourceId>,
    pub flat_width: u16,
    pub node_budget: u16,
    pub branch_depths: Vec<u16>,
    pub confidence_floor: f32,
    pub expected_delivered: f32,
    pub expected_cycle_us: f32,
    pub utility_tps: f32,
    pub reason: PlanReason,
}

impl DraftPlan {
    /// The Phase-1 fixed plan: one flat neural chain, no tree, no branching.
    /// `physical_verify_k` is accepted now so later tiers can validate the
    /// plan against the immutable `draft_budget::DflashDraftBudget`.
    pub fn neural_flat(width: u16, physical_verify_k: usize) -> Self {
        let _ = physical_verify_k;
        Self {
            mode: PlanMode::Flat,
            primary_source: ProposalSourceId::NeuralFlat,
            secondary_sources: Vec::new(),
            flat_width: width,
            node_budget: width,
            branch_depths: Vec::new(),
            confidence_floor: 0.0,
            expected_delivered: 0.0,
            expected_cycle_us: 0.0,
            utility_tps: 0.0,
            reason: PlanReason::FixedDefault,
        }
    }
}

/// The scheduler's measured result of executing a plan (guide §5.7). Replaces
/// ambiguous attribution ("SAM was enabled") with exact source-path economics.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanOutcome {
    pub plan_id: u64,
    pub source_mask: u32,
    pub drafted: u16,
    pub accepted: u16,
    pub delivered: u16,
    pub first_miss: u16,
    pub propose_us: u32,
    pub verify_us: u32,
    pub commit_us: u32,
    pub complete_cycle_us: u32,
    pub accepted_source_path: Vec<ProposalSourceId>,
}

impl PlanOutcome {
    /// Complete-cycle useful-token utility: `delivered / complete_cycle_seconds`.
    /// `0.0` when the cycle time is unmeasured.
    pub fn utility_tps(&self) -> f32 {
        if self.complete_cycle_us == 0 {
            0.0
        } else {
            (f64::from(self.delivered) / f64::from(self.complete_cycle_us) * 1_000_000.0) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neural_flat_plan_is_flat_and_unchanged_in_phase_1() {
        let plan = DraftPlan::neural_flat(12, 13);
        assert_eq!(plan.mode, PlanMode::Flat);
        assert_eq!(plan.primary_source, ProposalSourceId::NeuralFlat);
        assert!(plan.secondary_sources.is_empty());
        assert!(plan.branch_depths.is_empty());
        assert_eq!(plan.reason, PlanReason::FixedDefault);
    }

    #[test]
    fn utility_tps_is_delivered_over_cycle_seconds() {
        let outcome = PlanOutcome {
            plan_id: 1,
            source_mask: 0,
            drafted: 12,
            accepted: 6,
            delivered: 7,
            first_miss: 6,
            propose_us: 26_000,
            verify_us: 117_000,
            commit_us: 3_000,
            complete_cycle_us: 146_000,
            accepted_source_path: vec![ProposalSourceId::NeuralFlat],
        };
        // 7 / 0.146 s ≈ 47.95 tok/s.
        let tps = outcome.utility_tps();
        assert!((tps - 47.95).abs() < 0.05, "got {tps}");
    }

    #[test]
    fn utility_is_zero_when_cycle_unmeasured() {
        let outcome = PlanOutcome {
            plan_id: 2,
            source_mask: 0,
            drafted: 0,
            accepted: 0,
            delivered: 0,
            first_miss: 0,
            propose_us: 0,
            verify_us: 0,
            commit_us: 0,
            complete_cycle_us: 0,
            accepted_source_path: Vec::new(),
        };
        assert_eq!(outcome.utility_tps(), 0.0);
    }
}
