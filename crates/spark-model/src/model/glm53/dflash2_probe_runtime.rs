// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit diagnostic only: never selected by HTTP or an environment bypass.
use super::kv_prefix_runtime::{Attempt, DrainIo};
use super::*;
use crate::model::glm53::{ProbeCapture, ProbeIo};

/// The caller owns this observer and every pending host transfer until a
/// successful drain. A callback error/panic may already have submitted I/O.
pub trait Glm53Dflash2ProbeObserver {
    fn wants_projected_target(&self) -> bool {
        false
    }
    fn admit(&self, layout: &ProbeLayout, context: u32, stream: u64) -> Result<()>;
    fn observe(
        &mut self,
        stage: ProbeStage,
        source: GgmlIqBuffer,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()>;
}

struct DeviceProbeIo<'a>(&'a dyn GpuBackend);
impl ProbeIo for DeviceProbeIo<'_> {
    fn copy(&mut self, source: u64, dst: &mut [u8], stream: u64) -> Result<()> {
        self.0.copy_d2h_on_stream(DevicePtr(source), dst, stream)
    }
    fn drain(&mut self, stream: u64) -> Result<()> {
        self.0.synchronize(stream)
    }
}
impl Glm53Dflash2ProbeObserver for ProbeCapture {
    fn wants_projected_target(&self) -> bool {
        ProbeCapture::wants_projected_target(self)
    }
    fn admit(&self, layout: &ProbeLayout, context: u32, stream: u64) -> Result<()> {
        ProbeCapture::admit(self, layout, context, stream)
    }
    fn observe(
        &mut self,
        stage: ProbeStage,
        source: GgmlIqBuffer,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ProbeCapture::observe(
            self,
            stage,
            source.ptr.0,
            source.bytes,
            stream,
            &mut DeviceProbeIo(gpu),
        )
    }
}

impl Glm53Dflash2Runtime {
    pub fn probe_layout(&self) -> Result<ProbeLayout> {
        let attention = attention_plan(MAX_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS)?;
        ProbeLayout::new(
            self.weights.layers.len(),
            attention.cache_pool_bytes,
            attention.q_bytes,
            PREDICTED_TOKENS as usize * HIDDEN as usize * 2,
            PREDICTED_TOKENS as usize * 154_880 * 2,
            PREDICTED_TOKENS as usize * 4,
            154_880,
        )
    }

    /// Caller must retain model, both independent runtimes, and observer owners
    /// across every error and panic. No result here is a performance receipt.
    pub fn propose_diagnostic(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
        mode: Dflash2ProbeMode,
        observer: &mut dyn Glm53Dflash2ProbeObserver,
    ) -> Result<[u32; 7]> {
        ensure!(
            !target.gpu().stream_is_capturing(stream),
            "proposal diagnostic rejects graph capture"
        );
        target.ensure_dflash2_proposal_ready()?;
        ensure!(
            anchor < 154_880
                && self.context_tokens > 0
                && self.context_tokens <= MAX_CONTEXT_TOKENS
                && target.position() == self.context_tokens,
            "diagnostic anchor/committed context identity mismatch"
        );
        let projection = CommittedProjection::for_probe(target.gpu(), mode)?;
        self.preflight_gemv_projection(projection)?;
        match mode {
            Dflash2ProbeMode::CachedPrefix
            | Dflash2ProbeMode::StableCachedProjection
            | Dflash2ProbeMode::StableGemvCachedProjection => {
                self.ensure_projection_ready(projection.family())?;
            }
            Dflash2ProbeMode::FullRecompute
            | Dflash2ProbeMode::StableFullProjection
            | Dflash2ProbeMode::StableGemvFullProjection => {
                self.ensure_capture_kv_ready()?;
            }
        }
        let projected = observer.wants_projected_target();
        let layout = self.probe_layout()?;
        let layout = if projected {
            layout.with_projected_target(self.context_tokens, HIDDEN as usize)?
        } else {
            layout
        };
        observer.admit(&layout, self.context_tokens, stream)?;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<[u32; 7]> {
                if projected {
                    self.observe_projected(
                        ProbeStage::ProjectedTargetBefore,
                        &layout,
                        observer,
                        target.gpu(),
                        stream,
                    )?;
                }
                let proposal = match mode {
                    Dflash2ProbeMode::FullRecompute => {
                        self.propose_reference(target, anchor, stream, observer)
                    }
                    Dflash2ProbeMode::CachedPrefix => {
                        self.propose_cached_observed(target, anchor, stream, Some(&mut *observer))
                    }
                    Dflash2ProbeMode::StableFullProjection
                    | Dflash2ProbeMode::StableGemvFullProjection => self
                        .propose_reference_with_projection(
                            target, anchor, stream, observer, projection,
                        ),
                    Dflash2ProbeMode::StableCachedProjection
                    | Dflash2ProbeMode::StableGemvCachedProjection => self
                        .propose_cached_observed_with_projection(
                            target,
                            anchor,
                            stream,
                            Some(&mut *observer),
                            projection,
                        ),
                }?;
                if projected {
                    self.observe_projected(
                        ProbeStage::ProjectedTargetAfter,
                        &layout,
                        observer,
                        target.gpu(),
                        stream,
                    )?;
                }
                Ok(proposal)
            }));
        match result {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => {
                // An after callback can fail after the mode has completed.
                // The caller still owns its capture until a successful drain.
                target.poison_verify(stream);
                Err(error
                    .context("proposal diagnostic failed; target poisoned and owners retained"))
            }
            Err(_) => {
                target.poison_verify(stream);
                anyhow::bail!(
                    "proposal diagnostic panicked; runtime and observer owners must drain"
                )
            }
        }
    }

    fn observe_projected(
        &self,
        stage: ProbeStage,
        layout: &ProbeLayout,
        observer: &mut dyn Glm53Dflash2ProbeObserver,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let source = self.region(self.plan.projected_target);
        let bytes = layout.bytes(stage)?;
        ensure!(
            bytes <= source.bytes,
            "projected capture exceeds runtime source"
        );
        observer.observe(stage, exact(source.ptr, bytes), gpu, stream)
    }

    fn propose_reference(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
        observer: &mut dyn Glm53Dflash2ProbeObserver,
    ) -> Result<[u32; 7]> {
        self.propose_reference_with_projection(
            target,
            anchor,
            stream,
            observer,
            CommittedProjection::Original,
        )
    }

    fn propose_reference_with_projection(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
        observer: &mut dyn Glm53Dflash2ProbeObserver,
        projection: CommittedProjection,
    ) -> Result<[u32; 7]> {
        ensure!(anchor < 154_880, "reference anchor outside vocabulary");
        let gpu = target.gpu();
        self.reset_kv_prefix(gpu)?;
        let mut state = self
            .kv_prefix
            .lock()
            .map_err(|_| anyhow::anyhow!("reference owner poisoned"))?;
        state.select_projection(projection.family())?;
        state
            .prefix
            .begin(self.context_tokens, target.position(), stream, false)?;
        let mut attempt = Attempt {
            target,
            state: &mut state,
            stream,
            complete: false,
        };
        let proposal: Result<[u32; 7]> = (|| {
            let host = attempt
                .state
                .host
                .as_mut()
                .context("missing reference host staging")?;
            host.anchor = anchor.to_le_bytes();
            gpu.copy_h2d_async(&host.anchor, self.anchor, stream)?;
            gpu.synchronize(stream)?;
            // CRITICAL: the old full-context projections and attention branch,
            // not the cached adapter with a synthetic empty retained prefix.
            let (path, status) = match projection {
                CommittedProjection::Original => {
                    self.enqueue_proposal(target, anchor, stream, None, Some(observer))?
                }
                CommittedProjection::StableTc(_) | CommittedProjection::StableGemv(_) => self
                    .enqueue_proposal_with_projection(
                        target,
                        anchor,
                        stream,
                        None,
                        Some(observer),
                        projection,
                    )?,
            };
            let result = attempt.read_proposal(path, status)?;
            // Reference has no layer receipts and NEVER publishes KV cursors.
            attempt.state.reset(&mut DrainIo(gpu))?;
            Ok(result)
        })();
        match proposal {
            Ok(result) => {
                attempt.complete = true;
                Ok(result)
            }
            Err(error) => {
                let drain = attempt.state.prefix.abort(&mut DrainIo(gpu));
                Err(match drain {
                    Ok(()) => error.context("reference proposal failed; sequence poisoned"),
                    Err(fence) => error.context(format!(
                        "reference owner retained after failed drain: {fence:#}"
                    )),
                })
            }
        }
    }

    pub(super) fn observe_layer(
        &self,
        observer: &mut dyn Glm53Dflash2ProbeObserver,
        layer: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let plan = attention_plan(self.context_tokens, self.context_tokens)?;
        let (key, value) = self.kv[layer];
        // The callback follows actual attention in the shared proposal loop.
        // The outer caller's observer retains all fallible host transfers.
        observer.observe(
            ProbeStage::KeyCache(layer),
            exact(key, plan.cache_pool_bytes),
            gpu,
            stream,
        )?;
        observer.observe(
            ProbeStage::ValueCache(layer),
            exact(value, plan.cache_pool_bytes),
            gpu,
            stream,
        )?;
        observer.observe(
            ProbeStage::Attention(layer),
            self.region(self.plan.attention),
            gpu,
            stream,
        )
    }
}
