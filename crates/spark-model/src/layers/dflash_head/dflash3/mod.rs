// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash3 — adaptive proposal tournament (Phase 1 scaffold).
//!
//! Phase 1 is a **behavior-preserving** type/trait scaffold. Nothing here is
//! wired into `propose.rs`; the legacy flat path remains the only publisher
//! and the target verifier remains the only commit authority. See
//! the DFlash3 build guide (internal, not published) for the full
//! architecture (observation → sources → lattice → plan → compile → verify).
#![allow(dead_code)]

pub mod candidate;
pub mod config;
pub mod observation;
pub mod plan;
pub mod sources;

pub use candidate::{
    CandidateFlags, CandidateParent, CandidateSet, CandidateToken, ProposalSourceId,
    SourceEvidence, merge_equal_candidates,
};
pub use config::{Dflash3Config, Dflash3Mode};
pub use observation::{ContextBucket, ProposalObservation};
pub use plan::{DraftPlan, PlanMode, PlanOutcome, PlanReason};
pub use sources::{CandidateContext, CandidateSource, NeuralFlat, SourceRegistry};
