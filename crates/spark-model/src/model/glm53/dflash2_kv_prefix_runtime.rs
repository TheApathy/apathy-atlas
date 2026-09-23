// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit eager-only reuse of the drafter's five existing committed KV pairs.

use super::kv_prefix::{KvPrefixIo, KvTail, parse_kv_prefix_flag};
use super::*;

pub(super) fn kv_prefix_enabled() -> Result<bool> {
    match std::env::var("ATLAS_GLM53_DFLASH2_KV_PREFIX") {
        Ok(value) => parse_kv_prefix_flag(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_kv_prefix_flag(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("ATLAS_GLM53_DFLASH2_KV_PREFIX must be valid UTF-8")
        }
    }
}

pub(super) struct DrainIo<'a>(pub(super) &'a dyn GpuBackend);
impl KvPrefixIo for DrainIo<'_> {
    fn enqueue_layer(&mut self, _: usize, _: KvTail, _: u64) -> Result<()> {
        anyhow::bail!("drain-only KV binding cannot submit a layer")
    }
    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.0.synchronize(stream)
    }
}

pub(super) struct Attempt<'a> {
    pub(super) target: &'a Glm53Exl3Model,
    pub(super) state: &'a mut KvPrefixState,
    pub(super) stream: u64,
    pub(super) complete: bool,
}

impl Attempt<'_> {
    pub(super) fn read_proposal(
        &mut self,
        path: GgmlIqBuffer,
        status: GgmlIqBuffer,
    ) -> Result<[u32; 7]> {
        ensure!(
            path.bytes == 28 && status.bytes == 4,
            "cached selector extent changed"
        );
        let host = self
            .state
            .host
            .as_mut()
            .context("missing owned proposal staging")?;
        let gpu = self.target.gpu();
        // A backend copy may enqueue and then fail its fence. These buffers
        // remain heap-owned by the runtime until abort/reset observes completion.
        gpu.copy_d2h_on_stream(path.ptr, &mut host.path, self.stream)?;
        gpu.copy_d2h_on_stream(status.ptr, &mut host.status, self.stream)?;
        ensure!(
            u32::from_le_bytes(host.status) == 0,
            "GLM DFlash2 selector rejected device output"
        );
        let mut result = [0u32; 7];
        for (index, chunk) in host.path.chunks_exact(4).enumerate() {
            result[index] = u32::from_le_bytes(chunk.try_into().unwrap());
            ensure!(
                result[index] < 154_880,
                "GLM DFlash2 proposed out-of-vocabulary token"
            );
        }
        Ok(result)
    }
}

impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        if !self.complete {
            // No foreign GPU callback from Drop: a caught enqueue/read/fence
            // panic retains pending state and every host byte for explicit reset.
            self.target.poison_verify(self.stream);
        }
    }
}

impl Glm53Dflash2Runtime {
    pub(super) fn propose_with_kv_prefix(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
    ) -> Result<[u32; 7]> {
        self.propose_cached_observed(target, anchor, stream, None)
    }

    pub(super) fn propose_cached_observed(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
        observer: Option<&mut dyn Glm53Dflash2ProbeObserver>,
    ) -> Result<[u32; 7]> {
        self.propose_cached_observed_with_projection(
            target,
            anchor,
            stream,
            observer,
            CommittedProjection::Original,
        )
    }

    pub(super) fn propose_cached_observed_with_projection(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
        observer: Option<&mut dyn Glm53Dflash2ProbeObserver>,
        projection: CommittedProjection,
    ) -> Result<[u32; 7]> {
        target.ensure_dflash2_proposal_ready()?;
        let gpu = target.gpu();
        ensure!(anchor < 154_880, "GLM DFlash2 anchor outside vocabulary");
        let mut state = self.kv_prefix.lock().map_err(|_| {
            anyhow::anyhow!("GLM cached proposal panicked; explicit reset required")
        })?;
        state.select_projection(projection.family())?;
        state.prefix.begin(
            self.context_tokens,
            target.position(),
            stream,
            gpu.stream_is_capturing(stream),
        )?;
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
                .context("missing owned proposal staging")?;
            host.anchor = anchor.to_le_bytes();
            gpu.copy_h2d_group_on_stream(&[spark_runtime::gpu::HostToDeviceCopy::new(&host.anchor, self.anchor)], stream)?;
            gpu.synchronize(stream)?;
            let (path, status) = self.enqueue_proposal_with_projection(
                target,
                anchor,
                stream,
                Some(&mut attempt.state.prefix),
                observer,
                projection,
            )?;
            let result = attempt.read_proposal(path, status)?;
            attempt.state.prefix.finish(&mut DrainIo(gpu))?;
            Ok(result)
        })();
        match proposal {
            Ok(result) => {
                attempt.complete = true;
                Ok(result)
            }
            Err(error) => {
                // Preserve the original failure even when abort completes.
                // Failed abort retains the original stream/host/cache owner.
                let drain = attempt.state.prefix.abort(&mut DrainIo(gpu));
                Err(match drain {
                    Ok(()) => error.context("cached proposal failed; sequence poisoned"),
                    Err(drain_error) => error.context(format!(
                        "cached proposal failed; pending owner retained after drain failure: {drain_error:#}"
                    )),
                })
            }
        }
    }

    pub(super) fn enqueue_cached_attention(
        &self,
        prefix: &mut KvPrefix,
        layer: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
        projection: CommittedProjection,
    ) -> Result<()> {
        ensure!(
            prefix.pending_stream() == Some(stream),
            "cached proposal stream changed"
        );
        prefix.enqueue_layer(
            layer,
            &mut ProjectionIo {
                runtime: self,
                gpu,
                projection,
            },
        )
    }

    pub(super) fn ensure_uncached_kv_ready(&self) -> Result<()> {
        let state = self.kv_prefix.lock().map_err(|_| {
            anyhow::anyhow!("GLM cached proposal panicked; explicit reset required")
        })?;
        ensure!(
            !state.prefix.pending() && !state.prefix.poisoned(),
            "pending/poisoned KV owner requires reset"
        );
        for layer in 0..self.weights.layers.len() {
            ensure!(
                state.prefix.completed_rows(layer)? == 0,
                "changing KV-prefix mode requires request reset"
            );
        }
        Ok(())
    }

    pub(super) fn ensure_capture_kv_ready(&self) -> Result<()> {
        let state = self.kv_prefix.lock().map_err(|_| {
            anyhow::anyhow!("GLM cached proposal panicked; explicit reset required")
        })?;
        ensure!(
            !state.prefix.pending() && !state.prefix.poisoned(),
            "capture cannot reuse pending/poisoned drafter storage"
        );
        Ok(())
    }

    pub(super) fn ensure_projection_ready(
        &self,
        family: super::projection_contract::ProjectionFamily,
    ) -> Result<()> {
        let state = self.kv_prefix.lock().map_err(|_| {
            anyhow::anyhow!("GLM projection owner poisoned; explicit reset required")
        })?;
        // Admission only: release the lock before callbacks. Submission repeats
        // this same check and selects the family under its own owner lock.
        state.admit_projection(family)
    }

    pub(super) fn reset_kv_prefix(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut state = self.kv_prefix.lock().unwrap_or_else(|e| e.into_inner());
        state.reset(&mut DrainIo(gpu))?;
        self.kv_prefix.clear_poison();
        Ok(())
    }

    pub(in crate::model::glm53) fn drain_kv_prefix(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut state = self.kv_prefix.lock().unwrap_or_else(|e| e.into_inner());
        if state.prefix.pending() {
            state.prefix.abort(&mut DrainIo(gpu))?;
        }
        Ok(())
    }
}

struct ProjectionIo<'a> {
    runtime: &'a Glm53Dflash2Runtime,
    gpu: &'a dyn GpuBackend,
    projection: CommittedProjection,
}

impl KvPrefixIo for ProjectionIo<'_> {
    fn enqueue_layer(&mut self, index: usize, tail: KvTail, stream: u64) -> Result<()> {
        let runtime = self.runtime;
        let layer = runtime
            .weights
            .layers
            .get(index)
            .context("cached layer outside weights")?;
        ensure!(
            index < runtime.kv.len(),
            "cached layer outside owned KV pairs"
        );
        let plan = Glm53Dflash2AttentionPlan::new(
            1,
            QUERY_TOKENS,
            tail.new_rows(),
            tail.retained_rows(),
            tail.committed_end(),
            PHYSICAL_CACHE_BLOCKS,
            32,
            8,
            128,
            1.0e-5,
            10_000.0,
            2048,
            16,
        )?;
        ensure!(
            tail.committed_end() == runtime.context_tokens,
            "cached context cursor changed"
        );
        let projected = runtime.region(runtime.plan.projected_target);
        let source_bytes = usize::try_from(tail.source_row())?
            .checked_mul(HIDDEN as usize * 2)
            .context("cached source offset overflow")?;
        let input_bytes = usize::try_from(tail.new_rows())?
            .checked_mul(HIDDEN as usize * 2)
            .context("cached source extent overflow")?;
        let source = subspan(projected, source_bytes, input_bytes)?;
        let key = runtime.region(runtime.plan.key);
        let value = runtime.region(runtime.plan.value);
        let target_k = subspan(key, 0, plan.target_kv_bytes)?;
        let target_v = subspan(value, 0, plan.target_kv_bytes)?;
        let noise_k = subspan(key, plan.target_kv_bytes, plan.noise_kv_bytes)?;
        let noise_v = subspan(value, plan.target_kv_bytes, plan.noise_kv_bytes)?;
        let noise = runtime.region(runtime.plan.stream_b);
        // Only committed K/V may select the diagnostic TC projection. Noise,
        // normalization, RoPE and attention keep their original operators.
        self.projection.project_committed(
            self.gpu,
            source,
            &layer.k_proj,
            target_k,
            tail.new_rows(),
            stream,
        )?;
        self.projection.project_committed(
            self.gpu,
            source,
            &layer.v_proj,
            target_v,
            tail.new_rows(),
            stream,
        )?;
        dense(
            noise.ptr,
            layer.k_proj.weight,
            noise_k.ptr,
            QUERY_TOKENS,
            KV_WIDTH,
            HIDDEN,
            stream,
        )?;
        dense(
            noise.ptr,
            layer.v_proj.weight,
            noise_v.ptr,
            QUERY_TOKENS,
            KV_WIDTH,
            HIDDEN,
            stream,
        )?;
        let (k_cache, v_cache) = runtime.kv[index];
        runtime.attention.execute(
            self.gpu,
            plan,
            Glm53Dflash2AttentionBuffers {
                q_noise_bf16: runtime.region(runtime.plan.query),
                target_tail_k_bf16: target_k,
                target_tail_v_bf16: target_v,
                noise_k_bf16: noise_k,
                noise_v_bf16: noise_v,
                output_bf16: runtime.region(runtime.plan.attention),
                q_norm_weight_bf16: exact(layer.q_norm.weight, plan.norm_weight_bytes),
                k_norm_weight_bf16: exact(layer.k_norm.weight, plan.norm_weight_bytes),
                target_slots_i64: exact(runtime.target_slots, plan.target_slots_bytes),
                noise_slots_i64: exact(runtime.noise_slots, plan.noise_slots_bytes),
                block_tables_u32: exact(runtime.block_table, plan.block_tables_bytes),
                k_cache_bf16: exact(k_cache, plan.cache_pool_bytes),
                v_cache_bf16: exact(v_cache, plan.cache_pool_bytes),
            },
            stream,
        )
    }

    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.gpu.synchronize(stream)
    }
}

fn subspan(buffer: GgmlIqBuffer, offset: usize, bytes: usize) -> Result<GgmlIqBuffer> {
    ensure!(
        buffer.ptr != DevicePtr::NULL && bytes > 0,
        "empty cached projection span"
    );
    ensure!(
        offset % 2 == 0 && bytes % 2 == 0,
        "unaligned BF16 cached projection span"
    );
    ensure!(
        offset
            .checked_add(bytes)
            .is_some_and(|end| end <= buffer.bytes),
        "cached projection exceeds arena region"
    );
    let address = buffer
        .ptr
        .0
        .checked_add(u64::try_from(offset)?)
        .context("cached pointer overflow")?;
    address
        .checked_add(u64::try_from(bytes)?)
        .context("cached pointer end overflow")?;
    Ok(exact(DevicePtr(address), bytes))
}
