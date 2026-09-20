// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off synchronized stage timing for the physical GLM vision tower.

use std::ffi::OsStr;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use spark_runtime::gpu::GpuBackend;

#[derive(Clone, Copy)]
pub(super) enum Glm53VisionStage {
    Patch,
    Qkv,
    Attention,
    AttentionOutput,
    MlpInput,
    MlpOutput,
    Merger,
}

impl Glm53VisionStage {
    const COUNT: usize = 7;
}

pub(super) struct Glm53VisionTiming {
    enabled: bool,
    last: Instant,
    totals: [Duration; Glm53VisionStage::COUNT],
}

impl Glm53VisionTiming {
    pub(super) fn begin(gpu: &dyn GpuBackend, stream: u64) -> Result<Self> {
        let enabled = parse_enabled(std::env::var_os("ATLAS_GLM53_VISION_TIMING").as_deref())?;
        if enabled {
            gpu.synchronize(stream)?;
        }
        Ok(Self {
            enabled,
            last: Instant::now(),
            totals: [Duration::ZERO; Glm53VisionStage::COUNT],
        })
    }

    pub(super) fn mark(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
        stage: Glm53VisionStage,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        gpu.synchronize(stream)?;
        let now = Instant::now();
        self.totals[stage as usize] += now.duration_since(self.last);
        self.last = now;
        Ok(())
    }

    pub(super) fn report(&self, rows: u32, merged_rows: u32) {
        if !self.enabled {
            return;
        }
        let ms = |stage: Glm53VisionStage| self.totals[stage as usize].as_secs_f64() * 1_000.0;
        tracing::info!(
            target: "glm53_vision_timing",
            rows,
            merged_rows,
            patch_ms = ms(Glm53VisionStage::Patch),
            qkv_ms = ms(Glm53VisionStage::Qkv),
            attention_ms = ms(Glm53VisionStage::Attention),
            attention_output_ms = ms(Glm53VisionStage::AttentionOutput),
            mlp_input_ms = ms(Glm53VisionStage::MlpInput),
            mlp_output_ms = ms(Glm53VisionStage::MlpOutput),
            merger_ms = ms(Glm53VisionStage::Merger),
            "GLM vision synchronized stage timing"
        );
    }
}

fn parse_enabled(value: Option<&OsStr>) -> Result<bool> {
    match value.and_then(OsStr::to_str) {
        None => Ok(false),
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("ATLAS_GLM53_VISION_TIMING must be exactly 0 or 1"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm_vision_timing_contract_is_strict_and_default_off() {
        assert!(!parse_enabled(None).unwrap());
        assert!(!parse_enabled(Some(OsStr::new("0"))).unwrap());
        assert!(parse_enabled(Some(OsStr::new("1"))).unwrap());
        assert!(parse_enabled(Some(OsStr::new(""))).is_err());
        assert!(parse_enabled(Some(OsStr::new("yes"))).is_err());
    }
}
