// SPDX-License-Identifier: AGPL-3.0-only

//! Admission gate for GLM-5.3 greedy DFlash2 speculation.
//!
//! Speculation is admitted on evidence that it is bit-identical to the target
//! walk, not on a reference comparison of its own: a speculative step that
//! reproduces the admitted target path exactly inherits that path's
//! correctness. The evidence is measured (`bench/glm53-spec/make_evidence.py`
//! over the recorded windows), written to `bench/glm53-spec/evidence.json`,
//! and compiled in here; a unit test requires the two to agree field for field,
//! so the record cannot drift from the receipts or be edited in one place only.
//! `admit()` never reads files at runtime.

use anyhow::{Result, ensure};

/// Minimum coverage an evidence record must show.
const MIN_PROMPTS: usize = 5;
const MIN_IDENTITY_TRIALS: u32 = 25;
const MIN_PREFIX_POSITIONS: u32 = 100;
const MIN_FULL_POSITIONS: u32 = 50;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Glm53SpeculativeEvidence {
    pub schema: &'static str,
    pub scope: &'static str,
    pub identity_binary_sha256: &'static str,
    pub state_binary_sha256: &'static str,
    pub identity_prompts: &'static [&'static str],
    pub identity_trials_equal: u32,
    pub identity_trials_differ: u32,
    pub perturbed_prompt_control_differs: bool,
    pub state_prefix_positions_equal: u32,
    pub state_full_positions_equal: u32,
    pub state_gate_passed: bool,
    pub state_layers: &'static str,
    pub control_legacy_restage_fails_state: bool,
    pub control_legacy_restage_fails_identity: bool,
    pub control_extra_row_fails_state: bool,
    pub control_extra_row_fails_identity: bool,
    pub kernel_harness_bit_identical: bool,
    pub chunked_prefill_covered: bool,
}

/// Measured 2026-09-23/24 (windows w4 and w5, perf/glm53-spec-meas binaries).
/// Regenerate on the binary being admitted before relying on it.
pub(crate) const GLM53_SPECULATIVE_EVIDENCE: Glm53SpeculativeEvidence = Glm53SpeculativeEvidence {
    schema: "atlas.glm53.speculative_admission.v1",
    scope: "greedy",
    identity_binary_sha256: "2f429c39934ca98a2ce57558d531561e0e3b361accf9d769aebe527a32bdf8ae",
    state_binary_sha256: "93d924fb7229e428af5c9d6c417c84babe9c0e1e02bd42d4c4b3cb5e34531f1b",
    identity_prompts: &["code2", "long", "prose", "short", "think"],
    identity_trials_equal: 35,
    identity_trials_differ: 0,
    perturbed_prompt_control_differs: true,
    state_prefix_positions_equal: 439,
    state_full_positions_equal: 350,
    state_gate_passed: true,
    state_layers: "kda 0/17/33 conv+recurrent; dsa 0..10 latent/pools/tail + pool-ahead validity",
    control_legacy_restage_fails_state: true,
    control_legacy_restage_fails_identity: true,
    control_extra_row_fails_state: true,
    control_extra_row_fails_identity: true,
    kernel_harness_bit_identical: true,
    chunked_prefill_covered: false,
};

/// The speculation gate controls (`bench/glm53-spec`) that deliberately break
/// the accept path. Admission refuses while either is set.
pub(crate) const SPECULATIVE_CONTROL_ENVS: [&str; 2] = [
    "ATLAS_GLM53_PREFIX_COMMIT_CONTROL_LEGACY_RESTAGE",
    "ATLAS_GLM53_SPEC_CONTROL_COMMIT_EXTRA_ROW",
];

pub(crate) fn speculative_control_active() -> bool {
    SPECULATIVE_CONTROL_ENVS
        .iter()
        .any(|name| std::env::var(name).is_ok_and(|value| value == "1"))
}

impl Glm53SpeculativeEvidence {
    /// Admit greedy speculation only when every identity and state check held,
    /// with enough coverage, and every negative control failed as it must.
    /// Chunked prefill is recorded, not required: the EXL3 target refuses
    /// chunked prompts itself, so no path can exercise it yet.
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "atlas.glm53.speculative_admission.v1" && self.scope == "greedy",
            "GLM speculative evidence has an unknown schema or scope"
        );
        ensure!(
            self.identity_prompts.len() >= MIN_PROMPTS
                && self.identity_trials_equal >= MIN_IDENTITY_TRIALS
                && self.identity_trials_differ == 0,
            "GLM speculative evidence: output identity is missing or failed"
        );
        ensure!(
            self.state_gate_passed
                && self.state_prefix_positions_equal >= MIN_PREFIX_POSITIONS
                && self.state_full_positions_equal >= MIN_FULL_POSITIONS,
            "GLM speculative evidence: per-layer state identity is missing or failed"
        );
        ensure!(
            self.perturbed_prompt_control_differs
                && self.control_legacy_restage_fails_state
                && self.control_legacy_restage_fails_identity
                && self.control_extra_row_fails_state
                && self.control_extra_row_fails_identity,
            "GLM speculative evidence: a negative control did not fail, so its gate cannot"
        );
        ensure!(
            self.kernel_harness_bit_identical,
            "GLM speculative evidence: the row-batched kernel is not bit-identical"
        );
        ensure!(
            self.identity_binary_sha256.len() == 64 && self.state_binary_sha256.len() == 64,
            "GLM speculative evidence must name the binaries it was measured on"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "speculative_admission_tests.rs"]
mod tests;
