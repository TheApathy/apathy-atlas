// SPDX-License-Identifier: AGPL-3.0-only

//! Schedule-walking executor for the GLM-5.3 target.
//!
//! `forward_one.rs` builds and validates a 234-event schedule but states in its
//! own doc that it "does not launch a kernel or mutate sequence state, and
//! therefore is not an `impl Model` execution path". That missing component is
//! what this file supplies: the walk that turns a validated plan into work.
//!
//! ## Why this is a dispatcher and not a forward pass
//!
//! Everything the walk calls already exists — the GGUF loader, the KDA
//! convolution, the DSA latent commit, the DSA visibility materialisation, the
//! T1 state transaction, the arena. What did not exist was the ordering seam
//! that binds them to the schedule. The schedule is the authority on order
//! (including the capture layers [5,14,24,33,42] and the KDA/DSA and dense/MoE
//! interleave); this file must never re-derive that order, only consume it.
//!
//! ## Fail-closed, per seam
//!
//! Each event kind resolves to a `Seam`. A seam is either `Wired` — a concrete
//! op is bound and the capability it proves is named — or `Unbound`, carrying
//! the exact reason. Executing an `Unbound` seam is an error, never a silent
//! skip, because a skipped layer produces fluent, plausible, wrong output: the
//! failure mode that cost this project several days on the Flash-Next batched
//! prefill (F42/F56/F57/F59).
//!
//! `Glm53Executor::readiness()` reports the wired/unbound split without running
//! anything, so admission can be decided from evidence rather than assertion.
//! This is deliberately the same discipline as `Glm53KernelAdmission`: a caller
//! cannot claim readiness with a boolean.

use anyhow::{Result, bail};

use crate::layers::{
    Glm53TargetAttentionKind, Glm53TargetEvent, Glm53TargetFfnKind, Glm53TargetSchedule,
};

use super::kernels::Glm53RuntimeCapability;

/// Which op services an event.
///
/// `Identified` means the concrete op that must run is known and exists in this
/// tree. It does NOT mean the executor calls it — see `execute`. Conflating the
/// two would let a census masquerade as a forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Seam {
    /// The op that must run is known and present in-tree. Names the capability
    /// this seam would exercise, so a real run becomes evidence for flipping
    /// exactly that admission entry.
    Identified {
        op: &'static str,
        proves: Option<Glm53RuntimeCapability>,
    },
    /// No implementation exists yet. `needs` states what must be built.
    Unbound { needs: &'static str },
}

impl Seam {
    pub fn is_identified(&self) -> bool {
        matches!(self, Seam::Identified { .. })
    }
}

/// Static map from a schedule event to its execution seam.
///
/// Deliberately total over `Glm53TargetEvent`: adding a variant to the schedule
/// without deciding its seam is a compile error, not a runtime surprise.
pub fn seam_for(event: &Glm53TargetEvent) -> Seam {
    use Glm53RuntimeCapability as Cap;
    match event {
        Glm53TargetEvent::ExpandMhc => Seam::Identified {
            op: "Glm53HyperKernels::expand",
            proves: None,
        },
        Glm53TargetEvent::PreAttention { .. } => Seam::Identified {
            op: "Glm53HyperKernels::pre",
            proves: None,
        },
        Glm53TargetEvent::Attention {
            kind: Glm53TargetAttentionKind::Kda,
            ..
        } => Seam::Identified {
            op: "Glm53KdaConvKernel::launch_stage + launch_commit",
            proves: Some(Cap::KdaThreeStreamF32Convolution),
        },
        Glm53TargetEvent::Attention {
            kind: Glm53TargetAttentionKind::Dsa,
            ..
        } => Seam::Identified {
            op: "Glm53DsaLatentCommitKernel::launch + Glm53DsaCurrentVisibility::launch",
            proves: Some(Cap::DsaLatentCommit),
        },
        Glm53TargetEvent::PostAttention { .. } => Seam::Identified {
            op: "Glm53HyperKernels::post",
            proves: None,
        },
        Glm53TargetEvent::Ffn {
            kind: Glm53TargetFfnKind::Dense,
            ..
        } => Seam::Identified {
            op: "Glm53ActivationKernels swiglu (Glm53SwigluPlan)",
            proves: None,
        },
        Glm53TargetEvent::Ffn {
            kind: Glm53TargetFfnKind::Moe,
            ..
        } => Seam::Identified {
            op: "Glm53RouterKernels::launch + Glm53ActivationKernels expert reduce",
            proves: None,
        },
        Glm53TargetEvent::PostFfn { .. } => Seam::Identified {
            op: "Glm53HyperKernels::post (hyper_comb combine)",
            proves: None,
        },
        Glm53TargetEvent::CaptureWidenedMhc { .. } => Seam::Identified {
            op: "Glm53DsaT1Transaction capture into the arena capture slot",
            proves: Some(Cap::CompleteT1Scratch),
        },
        Glm53TargetEvent::OrderedMean => Seam::Identified {
            op: "Glm53HyperKernels::mean",
            proves: None,
        },
        Glm53TargetEvent::FinalNormF32 => Seam::Identified {
            op: "Glm53HyperKernels::norm",
            proves: None,
        },
        // The generic model path already owns this (`impl_a3::lm_head`); GLM
        // supplies `output_norm.weight` via its GGUF catalog. Wired to the
        // shared head rather than a GLM-specific one, because nothing about
        // GLM's output projection differs from the common case.
        Glm53TargetEvent::LmHeadF32 => Seam::Identified {
            op: "model::impl_a3::lm_head (shared FP32 head)",
            proves: None,
        },
    }
}

/// Readiness census over a whole schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glm53ExecutorReadiness {
    pub total_events: usize,
    pub identified_events: usize,
    pub unbound_events: usize,
    /// Distinct reasons, each with the number of events blocked on it.
    pub blockers: Vec<(&'static str, usize)>,
    /// Capabilities that a full successful run would exercise.
    pub proves: Vec<Glm53RuntimeCapability>,
}

impl Glm53ExecutorReadiness {
    pub fn is_complete(&self) -> bool {
        self.unbound_events == 0
    }
}

/// Walks a validated schedule.
///
/// Construction re-validates the schedule: a walk over an unvalidated plan is
/// how a capture-order or census drift would reach the GPU unnoticed.
pub struct Glm53Executor {
    schedule: Glm53TargetSchedule,
}

impl Glm53Executor {
    pub fn new(schedule: Glm53TargetSchedule) -> Result<Self> {
        schedule.validate()?;
        Ok(Self { schedule })
    }

    pub fn schedule(&self) -> &Glm53TargetSchedule {
        &self.schedule
    }

    /// Census without executing anything.
    pub fn readiness(&self) -> Glm53ExecutorReadiness {
        let mut identified = 0usize;
        let mut blockers: Vec<(&'static str, usize)> = Vec::new();
        let mut proves: Vec<Glm53RuntimeCapability> = Vec::new();
        for event in self.schedule.events() {
            match seam_for(event) {
                Seam::Identified { proves: p, .. } => {
                    identified += 1;
                    if let Some(cap) = p
                        && !proves.contains(&cap)
                    {
                        proves.push(cap);
                    }
                }
                Seam::Unbound { needs } => match blockers.iter_mut().find(|(n, _)| *n == needs) {
                    Some((_, count)) => *count += 1,
                    None => blockers.push((needs, 1)),
                },
            }
        }
        let total = self.schedule.events().len();
        Glm53ExecutorReadiness {
            total_events: total,
            identified_events: identified,
            unbound_events: total - identified,
            blockers,
            proves,
        }
    }

    /// Execute the schedule.
    ///
    /// **Not implemented, and deliberately loud about it.** Every one of the 234
    /// events now resolves to a concrete in-tree op (`readiness()` proves that),
    /// but identifying an op is not calling it: dispatch still needs each op's
    /// buffers resolved from the workspace regions, its weights bound from the
    /// GGUF catalog, and the T1 state transaction driven across the walk.
    ///
    /// This returns `Err` rather than `Ok(())` on purpose. A walk that checked
    /// its census and returned success would be indistinguishable from a real
    /// forward pass to every caller and test — the precise shape of the
    /// Flash-Next batched-prefill failure, where a path that "ran" produced
    /// fluent, wrong output for days before anyone noticed.
    pub fn execute(&self) -> Result<()> {
        let readiness = self.readiness();
        if !readiness.is_complete() {
            let detail = readiness
                .blockers
                .iter()
                .map(|(needs, count)| format!("  {count:>3} event(s): {needs}"))
                .collect::<Vec<_>>()
                .join("\n");
            bail!(
                "GLM executor: {} of {} events have no implementation yet.\n{}",
                readiness.unbound_events,
                readiness.total_events,
                detail
            );
        }
        bail!(
            "GLM executor: all {} events resolve to in-tree ops, but dispatch is \
             not implemented. Remaining work is per-op buffer/weight binding: \
             resolve workspace regions (hidden_a/hidden_b/collapsed/widened_hc/\
             hyper_post/hyper_comb) to device pointers, bind weights from the \
             GGUF catalog, and drive the T1 state transaction across the walk. \
             Refusing rather than returning Ok, so this cannot be mistaken for a \
             forward pass.",
            readiness.total_events
        );
    }
}

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
