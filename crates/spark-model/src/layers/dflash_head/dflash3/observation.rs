// SPDX-License-Identifier: AGPL-3.0-only

//! Immutable per-cycle observation snapshot (guide §5.1, Layer A).

use super::candidate::ProposalSourceId;

/// Coarse task-phase bucket used to partition planner state so a request in
/// one phase cannot poison another request's adaptive state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextBucket {
    FreshCode,
    EditHeavy,
    Prose,
    ToolJson,
    Unknown,
}

/// Immutable snapshot built once per speculative cycle, before any source is
/// evaluated, so one source cannot mutate state another source still needs.
/// Device pointers and stream ownership travel separately in an execution
/// context; this carries only the decision-relevant features.
#[derive(Debug, Clone)]
pub struct ProposalObservation<'a> {
    /// The verified bonus token that seeds the next block.
    pub last_token: u32,
    /// Absolute decoded position of `last_token`.
    pub absolute_position: usize,
    /// Committed token prefix (borrowed; never copied on the hot path).
    pub committed_tokens: &'a [u32],
    /// Recent per-cycle accept counts.
    pub recent_accepts: &'a [u8],
    /// The source the previous cycle actually used.
    pub previous_source: ProposalSourceId,
    pub previous_width: usize,
    pub previous_cycle_us: u64,
    pub context_bucket: ContextBucket,
    pub grammar_active: bool,
    pub thinking_active: bool,
    pub tree_capable: bool,
    pub physical_verify_k: usize,
}
