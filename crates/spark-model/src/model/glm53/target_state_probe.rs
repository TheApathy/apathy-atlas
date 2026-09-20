// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit, exclusively borrowed diagnostic view. Not a serving readback path.
use super::*;
use crate::layers::ops::glm53_exact_wide_prefill_active;
use crate::model::glm53::state_probe_frame::{StateProbeFrame, StateProbeStamp};
use crate::model::glm53::state_read_plan::StateReadPlan;

#[path = "target_state_regions.rs"]
mod regions;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub enum StateProbeRegion {
    Kda {
        ordinal: usize,
    },
    Conv {
        ordinal: usize,
    },
    DsaLatent {
        ordinal: usize,
    },
    DsaPoolKeys {
        ordinal: usize,
    },
    DsaPoolValidity {
        ordinal: usize,
    },
    DsaTailKeys {
        ordinal: usize,
    },
    DsaTailGates {
        ordinal: usize,
    },
    DsaTailValidity {
        ordinal: usize,
    },
    /// Raw latest-pass capture slot; caller must select only known fresh rows.
    Capture {
        tap: u32,
        row: u32,
    },
    ProjectedContext,
    /// Raw latest-pass logit row; continuation decode makes row0 authoritative.
    Logits {
        row: u32,
    },
}

#[derive(Debug, serde::Serialize)]
pub struct StateProbeDescriptor {
    pub region: StateProbeRegion,
    pub bytes: usize,
}

pub struct Glm53StateProbe<'a> {
    model: &'a mut Glm53Exl3Model,
    stamp: StateProbeStamp,
    stream: u64,
}

impl Glm53Exl3Model {
    pub fn state_probe(&mut self, stream: u64) -> Result<Glm53StateProbe<'_>> {
        let stamp = StateProbeStamp::new(self.state_probe_frame(stream)?)?;
        Ok(Glm53StateProbe {
            model: self,
            stamp,
            stream,
        })
    }

    fn state_probe_frame(&self, stream: u64) -> Result<StateProbeFrame> {
        ensure!(
            stream == self.gpu.default_stream(),
            "state probe requires the model stream"
        );
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("state probe model owner poisoned"))?;
        let runtime = self
            .dflash2
            .lock()
            .map_err(|_| anyhow::anyhow!("state probe drafter owner poisoned"))?;
        let runtime = runtime
            .as_ref()
            .context("state probe requires the installed drafter")?;
        Ok(StateProbeFrame {
            generation: state.generation,
            nonce: state.nonce,
            position: state.position,
            context: runtime.context_tokens(),
            capacity: self.capacity,
            context_capacity: runtime.context_capacity(),
            stream,
            model_stream: self.gpu.default_stream(),
            live: self.live_sequence.load(Ordering::Acquire),
            poisoned: state.poisoned_stream.is_some(),
            capturing: self.gpu.stream_is_capturing(stream),
            prefill: glm53_exact_wide_prefill_active() || glm53_layer_major_prefill_active(),
        })
    }
}

impl Glm53StateProbe<'_> {
    pub fn stamp(&self) -> &StateProbeStamp {
        &self.stamp
    }

    /// Complete meaningful persistent regions, including zero-length pool prefixes.
    /// Captures/logits are explicitly selected separately, never guessed as fresh.
    pub fn regions(&self) -> Result<Vec<StateProbeDescriptor>> {
        self.stamp
            .check(self.model.state_probe_frame(self.stream)?)?;
        let mut keys = Vec::new();
        for ordinal in 0..self.model.kda_states.len() {
            keys.push(StateProbeRegion::Kda { ordinal });
        }
        for ordinal in 0..self.model.kda_conv.len() {
            keys.push(StateProbeRegion::Conv { ordinal });
        }
        for ordinal in 0..self.model.dsa_cache.len() {
            keys.extend([
                StateProbeRegion::DsaLatent { ordinal },
                StateProbeRegion::DsaPoolKeys { ordinal },
                StateProbeRegion::DsaPoolValidity { ordinal },
                StateProbeRegion::DsaTailKeys { ordinal },
                StateProbeRegion::DsaTailGates { ordinal },
                StateProbeRegion::DsaTailValidity { ordinal },
            ]);
        }
        keys.push(StateProbeRegion::ProjectedContext);
        keys.into_iter()
            .map(|region| {
                Ok(StateProbeDescriptor {
                    region,
                    bytes: regions::resolve(self.model, region, self.stamp.position())?
                        .view
                        .bytes,
                })
            })
            .collect()
    }

    pub fn region_bytes(&self, region: StateProbeRegion) -> Result<usize> {
        self.stamp
            .check(self.model.state_probe_frame(self.stream)?)?;
        Ok(regions::resolve(self.model, region, self.stamp.position())?
            .view
            .bytes)
    }

    pub fn read(
        &mut self,
        region: StateProbeRegion,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<()> {
        self.stamp
            .check(self.model.state_probe_frame(self.stream)?)?;
        let resolved = regions::resolve(self.model, region, self.stamp.position())?;
        let plan = StateReadPlan::new(
            resolved.parent.ptr.0,
            resolved.parent.bytes,
            resolved.view.ptr.0,
            resolved.view.bytes,
            offset,
            destination.len(),
        )?;
        self.model
            .copy_state_probe_region(&plan, destination, self.stream)
    }
}
