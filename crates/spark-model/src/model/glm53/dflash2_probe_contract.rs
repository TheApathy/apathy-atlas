// SPDX-License-Identifier: AGPL-3.0-only
//! Derived extents for an explicit diagnostic, not another model validator.
use anyhow::{Context, Result, ensure};
use std::ops::Range;

pub const MAX_PROBE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dflash2ProbeMode {
    FullRecompute,
    CachedPrefix,
    StableFullProjection,
    StableCachedProjection,
    StableGemvFullProjection,
    StableGemvCachedProjection,
}
impl Dflash2ProbeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullRecompute => "full-recompute",
            Self::CachedPrefix => "cached-prefix",
            Self::StableFullProjection => "stable-full-projection",
            Self::StableCachedProjection => "stable-cached-projection",
            Self::StableGemvFullProjection => "stable-gemv-full-projection",
            Self::StableGemvCachedProjection => "stable-gemv-cached-projection",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeStage {
    ProjectedTargetBefore,
    KeyCache(usize),
    ValueCache(usize),
    Attention(usize),
    SelectedHidden,
    HeadLogits,
    DraftIds,
    ProjectedTargetAfter,
}
impl ProbeStage {
    pub fn name(self) -> String {
        match self {
            Self::ProjectedTargetBefore => "projected-target-before".into(),
            Self::KeyCache(layer) => format!("layer{layer}-key-cache"),
            Self::ValueCache(layer) => format!("layer{layer}-value-cache"),
            Self::Attention(layer) => format!("layer{layer}-attention"),
            Self::SelectedHidden => "selected-hidden".into(),
            Self::HeadLogits => "head-logits".into(),
            Self::DraftIds => "draft-ids".into(),
            Self::ProjectedTargetAfter => "projected-target-after".into(),
        }
    }
}

#[derive(Clone)]
pub struct ProbeLayout {
    stages: Vec<ProbeStage>,
    layers: usize,
    kv: usize,
    attention: usize,
    hidden: usize,
    logits: usize,
    ids: usize,
    total: usize,
    vocab: u32,
    projected: Option<(u32, usize)>,
}
pub struct KvRegions {
    pub committed: Range<usize>,
    pub noise: Range<usize>,
    pub unused: Range<usize>,
}

impl ProbeLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layers: usize,
        kv: usize,
        attention: usize,
        hidden: usize,
        logits: usize,
        ids: usize,
        vocab: u32,
    ) -> Result<Self> {
        ensure!(layers > 0 && vocab > 0, "empty diagnostic geometry");
        ensure!(
            [kv, attention, hidden, logits, ids]
                .iter()
                .all(|v| *v > 0 && *v % 2 == 0),
            "invalid BF16 diagnostic extent"
        );
        ensure!(ids % 4 == 0, "draft ID extent is not U32");
        let per_layer = kv
            .checked_mul(2)
            .and_then(|v| v.checked_add(attention))
            .context("layer extent overflow")?;
        let total = per_layer
            .checked_mul(layers)
            .and_then(|v| v.checked_add(hidden))
            .and_then(|v| v.checked_add(logits))
            .and_then(|v| v.checked_add(ids))
            .context("capture extent overflow")?;
        ensure!(total <= MAX_PROBE_BYTES, "diagnostic capture exceeds64MiB");
        let count = layers
            .checked_mul(3)
            .and_then(|v| v.checked_add(3))
            .context("stage count overflow")?;
        let mut stages = Vec::new();
        stages.try_reserve_exact(count)?;
        for layer in 0..layers {
            stages.extend([
                ProbeStage::KeyCache(layer),
                ProbeStage::ValueCache(layer),
                ProbeStage::Attention(layer),
            ]);
        }
        stages.extend([
            ProbeStage::SelectedHidden,
            ProbeStage::HeadLogits,
            ProbeStage::DraftIds,
        ]);
        Ok(Self {
            stages,
            layers,
            kv,
            attention,
            hidden,
            logits,
            ids,
            total,
            vocab,
            projected: None,
        })
    }
    /// Explicit diagnostic extension, bounded together with all original frames.
    /// `hidden` comes from the admitted runtime; this is not a model validator.
    pub fn with_projected_target(mut self, context: u32, hidden: usize) -> Result<Self> {
        ensure!(
            self.projected.is_none(),
            "projected input capture already selected"
        );
        ensure!(context > 0 && hidden > 0, "empty projected input geometry");
        let bytes = usize::try_from(context)?
            .checked_mul(hidden)
            .and_then(|v| v.checked_mul(2))
            .context("projected input extent overflow")?;
        let total = bytes
            .checked_mul(2)
            .and_then(|v| self.total.checked_add(v))
            .context("extended diagnostic capture overflow")?;
        ensure!(
            total <= MAX_PROBE_BYTES,
            "extended diagnostic capture exceeds64MiB"
        );
        self.stages.try_reserve_exact(2)?;
        self.stages.insert(0, ProbeStage::ProjectedTargetBefore);
        self.stages.push(ProbeStage::ProjectedTargetAfter);
        self.projected = Some((context, bytes));
        self.total = total;
        Ok(self)
    }
    pub fn projected_context(&self) -> Option<u32> {
        self.projected.map(|(context, _)| context)
    }
    pub fn stages(&self) -> &[ProbeStage] {
        &self.stages
    }
    pub fn total_bytes(&self) -> usize {
        self.total
    }
    pub fn vocab(&self) -> u32 {
        self.vocab
    }
    pub fn max_stage_bytes(&self) -> usize {
        [self.kv, self.attention, self.hidden, self.logits, self.ids]
            .into_iter()
            .chain(self.projected.map(|(_, bytes)| bytes))
            .max()
            .unwrap()
    }
    pub fn bytes(&self, stage: ProbeStage) -> Result<usize> {
        match stage {
            ProbeStage::ProjectedTargetBefore | ProbeStage::ProjectedTargetAfter => self
                .projected
                .map(|(_, bytes)| bytes)
                .context("projected input capture not selected"),
            ProbeStage::KeyCache(layer) | ProbeStage::ValueCache(layer) => {
                ensure!(
                    layer < self.layers,
                    "cache layer outside diagnostic geometry"
                );
                Ok(self.kv)
            }
            ProbeStage::Attention(layer) => {
                ensure!(
                    layer < self.layers,
                    "attention layer outside diagnostic geometry"
                );
                Ok(self.attention)
            }
            ProbeStage::SelectedHidden => Ok(self.hidden),
            ProbeStage::HeadLogits => Ok(self.logits),
            ProbeStage::DraftIds => Ok(self.ids),
        }
    }
    pub fn kv_regions(&self, context: u32, noise: u32, row_bytes: usize) -> Result<KvRegions> {
        ensure!(
            context > 0 && noise > 0 && row_bytes > 0 && row_bytes % 2 == 0,
            "invalid NHD diagnostic region"
        );
        let committed = usize::try_from(context)?
            .checked_mul(row_bytes)
            .context("committed region overflow")?;
        let end = usize::try_from(noise)?
            .checked_mul(row_bytes)
            .and_then(|n| committed.checked_add(n))
            .context("noise region overflow")?;
        ensure!(
            end <= self.kv && self.kv % row_bytes == 0,
            "NHD region exceeds pool"
        );
        Ok(KvRegions {
            committed: 0..committed,
            noise: committed..end,
            unused: end..self.kv,
        })
    }
}
