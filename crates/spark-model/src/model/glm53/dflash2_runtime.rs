// SPDX-License-Identifier: AGPL-3.0-only

//! Effectful single-sequence GLM-5.3 DFlash2 proposal runtime.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};

use crate::layers::ops::{
    self, GgmlIqBuffer, Glm53Dflash2AttentionBuffers, Glm53Dflash2AttentionKernels,
    Glm53Dflash2AttentionPlan, Glm53Dflash2ConvBuffers, Glm53Dflash2ConvKernel,
    Glm53Dflash2ConvPhase, Glm53Dflash2SelectorBuffers, Glm53Dflash2SelectorKernel,
    Glm53Dflash2SelectorPlan, Glm53Dflash2TopkBuffers, Glm53Dflash2TopkKernel,
    Glm53Dflash2TopkPlan, Glm53Exl3Buffer,
};
use crate::layers::{Glm53Dflash2RuntimeGeometry, Glm53Dflash2ScratchPlan};
use crate::weight_loader::{
    Glm53Dflash2Weights, Glm53Exl3Bf16LinearBuffers, Glm53Exl3Linear, PreparedGlm53Exl3Bf16Linear,
    load_glm53_dflash2_weights, parse_glm53_dflash2_config,
};

use super::Glm53Exl3Model;

#[path = "dflash2_committed_projection.rs"]
mod committed_projection;
#[path = "dflash2_gemv_plan.rs"]
mod gemv_plan;
#[path = "dflash2_gemv_projection.rs"]
mod gemv_projection;
#[path = "dflash2_kv_prefix.rs"]
mod kv_prefix;
#[path = "dflash2_kv_prefix_owner.rs"]
mod kv_prefix_owner;
#[path = "dflash2_kv_prefix_runtime.rs"]
mod kv_prefix_runtime;
#[path = "dflash2_ordered_capture.rs"]
mod ordered_capture;
#[path = "dflash2_prefill_capture.rs"]
mod prefill_capture;
#[path = "dflash2_probe_runtime.rs"]
mod probe;
#[path = "dflash2_projection_contract.rs"]
mod projection_contract;
#[path = "dflash2_proposal.rs"]
mod proposal;
#[path = "dflash2_serving.rs"]
mod serving;
#[path = "dflash2_serving_contract.rs"]
mod serving_contract;
#[path = "dflash2_state_probe.rs"]
mod state_probe;
use super::dflash2_probe_contract::{Dflash2ProbeMode, ProbeLayout, ProbeStage};
use committed_projection::CommittedProjection;
pub use probe::Glm53Dflash2ProbeObserver;
use serving_contract::ServingProjection;

use kv_prefix::KvPrefix;
use kv_prefix_owner::KvPrefixState;
pub(in crate::model::glm53) use kv_prefix_owner::retain_until_drained;
use kv_prefix_runtime::kv_prefix_enabled;

const HIDDEN: u32 = 4096;
const INTERMEDIATE: u32 = 12288;
pub(super) const QUERY_TOKENS: u32 = 8;
const PREDICTED_TOKENS: u32 = 7;
const KV_WIDTH: u32 = 1024;
const MAX_CONTEXT_TOKENS: u32 = 2047;
const PHYSICAL_CACHE_BLOCKS: u32 = 129;
const RESERVE_BYTES: usize = 8 * 1024 * 1024 * 1024;

#[must_use = "GLM DFlash2 runtime owns device allocations and requires explicit free"]
pub struct Glm53Dflash2Runtime {
    store: WeightStore,
    weights: Glm53Dflash2Weights,
    plan: Glm53Dflash2ScratchPlan,
    arena: DevicePtr,
    kv: Vec<(DevicePtr, DevicePtr)>,
    target_slots: DevicePtr,
    noise_slots: DevicePtr,
    block_table: DevicePtr,
    anchor: DevicePtr,
    capture: prefill_capture::CaptureTileResources,
    head_input_f16: DevicePtr,
    head_output_f16: DevicePtr,
    head_hadamard_f16: DevicePtr,
    head_locks: DevicePtr,
    head: PreparedGlm53Exl3Bf16Linear,
    attention: Glm53Dflash2AttentionKernels,
    conv: Glm53Dflash2ConvKernel,
    topk: Glm53Dflash2TopkKernel,
    selector: Glm53Dflash2SelectorKernel,
    rms_norm: KernelHandle,
    residual_add: KernelHandle,
    silu_mul: KernelHandle,
    serving_projection: CommittedProjection,
    context_tokens: u32,
    kv_prefix: Mutex<KvPrefixState>,
}

impl Glm53Dflash2Runtime {
    pub(super) fn preflight_target_rows(&self, position: u32, rows: u32) -> Result<()> {
        self.ensure_capture_kv_ready()?;
        super::dsa_verify_plan::validate_capture_advance(
            position,
            self.context_tokens,
            rows,
            MAX_CONTEXT_TOKENS,
        )
    }

    pub fn load(gpu: &dyn GpuBackend, root: &Path, lm_head: &Glm53Exl3Linear) -> Result<Self> {
        let capture_mode = prefill_capture::capture_tile_mode()?;
        let choice = ServingProjection::parse(
            std::env::var_os("ATLAS_GLM53_DFLASH2_COMMITTED_PROJECTION").as_deref(),
            kv_prefix_enabled()?,
        )?;
        let serving_projection = CommittedProjection::for_serving(gpu, choice)?;
        let config_json = std::fs::read_to_string(root.join("config.json"))
            .with_context(|| format!("read GLM DFlash2 config from {}", root.display()))?;
        let config = parse_glm53_dflash2_config(&config_json)?;
        let store = SafetensorsLoader::new().load(root, gpu, RESERVE_BYTES)?;
        let weights = load_glm53_dflash2_weights(&store, &config)?;
        let kv_prefix = KvPrefix::new(weights.layers.len(), MAX_CONTEXT_TOKENS)?;
        let plan = Glm53Dflash2ScratchPlan::new(Glm53Dflash2RuntimeGeometry::exact(
            1,
            MAX_CONTEXT_TOKENS,
            QUERY_TOKENS,
        ))?;
        let arena = gpu.alloc(plan.arena_bytes)?;
        gpu.memset(arena, 0, plan.arena_bytes)?;
        let attention_plan = attention_plan(MAX_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS)?;
        let mut kv = Vec::with_capacity(5);
        for _ in 0..5 {
            let k = gpu.alloc(attention_plan.cache_pool_bytes)?;
            let v = gpu.alloc(attention_plan.cache_pool_bytes)?;
            gpu.memset(k, 0, attention_plan.cache_pool_bytes)?;
            gpu.memset(v, 0, attention_plan.cache_pool_bytes)?;
            kv.push((k, v));
        }
        let target_slots = gpu.alloc(attention_plan.target_slots_bytes)?;
        let noise_slots = gpu.alloc(attention_plan.noise_slots_bytes)?;
        let block_table = gpu.alloc(attention_plan.block_tables_bytes)?;
        let block_table_bytes = (0..PHYSICAL_CACHE_BLOCKS)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        gpu.copy_h2d(&block_table_bytes, block_table)?;
        let anchor = gpu.alloc(4)?;
        let capture = prefill_capture::CaptureTileResources::load(gpu, capture_mode)?;
        let head_input_bytes = PREDICTED_TOKENS as usize * HIDDEN as usize * 2;
        let head_output_bytes = PREDICTED_TOKENS as usize * 154_880 * 2;
        let head_input_f16 = gpu.alloc(head_input_bytes)?;
        let head_output_f16 = gpu.alloc(head_output_bytes)?;
        let head_hadamard_f16 = gpu.alloc(head_input_bytes)?;
        let head_locks = gpu.alloc(4 * 1024 * 1024)?;
        gpu.memset(head_locks, 0, 4 * 1024 * 1024)?;
        Ok(Self {
            store,
            weights,
            plan,
            arena,
            kv,
            target_slots,
            noise_slots,
            block_table,
            anchor,
            capture,
            head_input_f16,
            head_output_f16,
            head_hadamard_f16,
            head_locks,
            head: lm_head.prepare_bf16(gpu, PREDICTED_TOKENS)?,
            attention: Glm53Dflash2AttentionKernels::load(gpu)?,
            conv: Glm53Dflash2ConvKernel::load(gpu)?,
            topk: Glm53Dflash2TopkKernel::load(gpu)?,
            selector: Glm53Dflash2SelectorKernel::load(gpu)?,
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            silu_mul: gpu.kernel("residual_add", "silu_mul_separate")?,
            serving_projection,
            context_tokens: 0,
            kv_prefix: Mutex::new(KvPrefixState::new(kv_prefix)),
        })
    }

    /// Retain one target token's five contracted capture taps as DFlash2 context.
    pub fn observe_target(&mut self, target: &Glm53Exl3Model, stream: u64) -> Result<()> {
        ensure!(
            target.position() == self.context_tokens + 1,
            "GLM DFlash2 target captures must be observed once, in position order"
        );
        self.observe_target_rows_inner(target, 0, 1, stream)
    }

    /// Retain every committed row from the most recent wide verifier pass.
    pub fn observe_target_rows(
        &mut self,
        target: &Glm53Exl3Model,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (2..=QUERY_TOKENS).contains(&rows) && target.position() == self.context_tokens + rows,
            "GLM DFlash2 wide captures must be observed once, in position order"
        );
        self.observe_target_rows_inner(target, 0, rows, stream)
    }

    fn observe_target_rows_inner(
        &mut self,
        target: &Glm53Exl3Model,
        first_capture_row: u32,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        self.ensure_capture_kv_ready()?;
        self.project_capture_rows(target, first_capture_row, rows, self.context_tokens, stream)?;
        self.context_tokens += rows;
        Ok(())
    }

    fn project_capture_rows(
        &self,
        target: &Glm53Exl3Model,
        first_capture_row: u32,
        rows: u32,
        destination_position: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            rows != 0
                && rows <= QUERY_TOKENS
                && first_capture_row
                    .checked_add(rows)
                    .is_some_and(|end| end <= QUERY_TOKENS)
                && destination_position
                    .checked_add(rows)
                    .is_some_and(|end| end <= MAX_CONTEXT_TOKENS),
            "GLM DFlash2 capture/destination rows exceed runtime bounds"
        );
        target.copy_dflash_capture_rows(first_capture_row, rows, self.capture.input, stream)?;
        let projected = self.region(self.plan.projected_target);
        let row_bytes = HIDDEN as usize * 2;
        let destination =
            DevicePtr(projected.ptr.0 + u64::from(destination_position) * row_bytes as u64);
        dense(
            self.capture.input,
            self.weights.fc.weight,
            destination,
            rows,
            HIDDEN,
            5 * HIDDEN,
            stream,
        )?;
        ops::rms_norm(
            target.gpu(),
            self.rms_norm,
            destination,
            &self.weights.hidden_norm,
            destination,
            rows,
            HIDDEN,
            1.0e-5,
            stream,
        )?;
        Ok(())
    }

    pub fn context_tokens(&self) -> u32 {
        self.context_tokens
    }

    pub(super) fn context_capacity(&self) -> u32 {
        MAX_CONTEXT_TOKENS
    }

    pub fn reset_context(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.reset_kv_prefix(gpu)?;
        self.context_tokens = 0;
        gpu.memset(
            self.region(self.plan.projected_target).ptr,
            0,
            self.plan.projected_target.bytes,
        )
    }

    /// Propose seven coherent draft tokens from one already-completed target token.
    pub fn propose(&self, target: &Glm53Exl3Model, anchor: u32, stream: u64) -> Result<[u32; 7]> {
        let prefix_enabled = kv_prefix_enabled()?;
        // Original keeps its existing capture-query boundary in the cached
        // body. The new family must reject capture before any proposal effects.
        let capturing = matches!(self.serving_projection, CommittedProjection::StableGemv(_))
            && target.gpu().stream_is_capturing(stream);
        let choice = self.admit_serving(prefix_enabled, capturing)?;
        if choice == ServingProjection::StableGemv {
            return self.propose_serving_gemv(target, anchor, stream);
        }
        if prefix_enabled {
            // Catch after the inner attempt/mutex have unwound, while the
            // target still owns its enclosing runtime lock. This preserves
            // reset admission; only the inner pending owner remains poisoned.
            return match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.propose_with_kv_prefix(target, anchor, stream)
            })) {
                Ok(result) => result,
                Err(_) => {
                    target.poison_verify(stream);
                    anyhow::bail!("GLM cached proposal panicked; pending owner retained for reset")
                }
            };
        }
        self.ensure_uncached_kv_ready()?;
        self.stage_anchor(target.gpu(), anchor, stream)?;
        let (path, status) = self.enqueue_proposal(target, anchor, stream, None, None)?;
        self.read_proposal(target.gpu(), path, status, stream)
    }

    /// Capture the fixed-context proposal body and time one replay. As with the
    /// target graph probe, context-length and embedding source addresses are
    /// frozen in the executable; this measures the graph ceiling but is not a
    /// reusable production graph yet.
    pub fn propose_graph_probe(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
    ) -> Result<([u32; 7], f64)> {
        let prefix_enabled = kv_prefix_enabled()?;
        self.admit_serving(prefix_enabled, false)?
            .admit_graph(prefix_enabled)?;
        self.ensure_uncached_kv_ready()?;
        ensure!(
            stream != 0,
            "GLM DFlash2 graph probe needs a non-legacy stream"
        );
        self.stage_anchor(target.gpu(), anchor, stream)?;
        let gpu = target.gpu();
        gpu.begin_capture(stream)?;
        let body_result = self.enqueue_proposal(target, anchor, stream, None, None);
        let (path, status) = match body_result {
            Ok(outputs) => outputs,
            Err(body_error) => {
                let cleanup = gpu
                    .end_capture(stream)
                    .and_then(|graph| gpu.destroy_graph(graph));
                return match cleanup {
                    Ok(()) => Err(body_error),
                    Err(cleanup_error) => Err(body_error.context(format!(
                        "DFlash2 CUDA graph capture cleanup also failed: {cleanup_error:#}"
                    ))),
                };
            }
        };
        let graph = gpu.end_capture(stream)?;
        ensure!(
            graph.0 != 0,
            "GLM DFlash2 graph capture returned a null handle"
        );
        let started = std::time::Instant::now();
        let replay_result = gpu
            .launch_graph(graph, stream)
            .and_then(|()| self.read_proposal(gpu, path, status, stream));
        let replay_seconds = started.elapsed().as_secs_f64();
        let destroy_result = gpu.destroy_graph(graph);
        let result = replay_result.context("GLM DFlash2 graph replay")?;
        destroy_result.context("GLM DFlash2 graph destroy")?;
        Ok((result, replay_seconds))
    }

    fn stage_anchor(&self, gpu: &dyn GpuBackend, anchor: u32, stream: u64) -> Result<()> {
        let bytes = anchor.to_le_bytes();
        gpu.copy_h2d_group_on_stream(&[spark_runtime::gpu::HostToDeviceCopy::new(&bytes, self.anchor)], stream)?;
        gpu.synchronize(stream)
    }

    fn read_proposal(
        &self,
        gpu: &dyn GpuBackend,
        path: GgmlIqBuffer,
        selector_status: GgmlIqBuffer,
        stream: u64,
    ) -> Result<[u32; 7]> {
        let mut bytes = [0u8; 28];
        let mut status = [0u8; 4];
        gpu.copy_d2h_on_stream(path.ptr, &mut bytes, stream)?;
        gpu.copy_d2h_on_stream(selector_status.ptr, &mut status, stream)?;
        ensure!(
            u32::from_le_bytes(status) == 0,
            "GLM DFlash2 selector rejected device output"
        );
        let mut result = [0u32; 7];
        for (index, chunk) in bytes.chunks_exact(4).enumerate() {
            result[index] = u32::from_le_bytes(chunk.try_into().unwrap());
            ensure!(
                result[index] < 154_880,
                "GLM DFlash2 proposed out-of-vocabulary token"
            );
        }
        Ok(result)
    }

    /// The caller must retain the enclosing backend owner until this succeeds.
    /// The target preflights this drain before destructuring any model fields.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        let this = retain_until_drained(self, |runtime| runtime.drain_kv_prefix(gpu))?;
        let mut weight_ptrs = BTreeSet::new();
        for name in this.store.names() {
            weight_ptrs.insert(this.store.get(name)?.ptr.0);
        }
        for ptr in [
            this.head_locks,
            this.head_hadamard_f16,
            this.head_output_f16,
            this.head_input_f16,
            this.capture.input,
            this.anchor,
            this.block_table,
            this.noise_slots,
            this.target_slots,
        ] {
            gpu.free(ptr)?;
        }
        for (k, v) in this.kv.into_iter().rev() {
            gpu.free(v)?;
            gpu.free(k)?;
        }
        gpu.free(this.arena)?;
        drop(this.weights);
        drop(this.store);
        for ptr in weight_ptrs.into_iter().rev() {
            gpu.free(DevicePtr(ptr))?;
        }
        Ok(())
    }

    fn region(&self, region: crate::layers::Glm53Dflash2ScratchRegion) -> GgmlIqBuffer {
        exact(DevicePtr(self.arena.0 + region.offset as u64), region.bytes)
    }

    fn prefix(
        &self,
        region: crate::layers::Glm53Dflash2ScratchRegion,
        bytes: usize,
    ) -> GgmlIqBuffer {
        debug_assert!(bytes <= region.bytes);
        exact(DevicePtr(self.arena.0 + region.offset as u64), bytes)
    }
}

fn attention_plan(
    context_tokens: u32,
    absolute_context_end: u32,
) -> Result<Glm53Dflash2AttentionPlan> {
    Glm53Dflash2AttentionPlan::new(
        1,
        8,
        context_tokens,
        0,
        absolute_context_end,
        PHYSICAL_CACHE_BLOCKS,
        32,
        8,
        128,
        1.0e-5,
        10_000.0,
        2048,
        16,
    )
}

fn dense(
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    rows: u32,
    columns: u32,
    inner: u32,
    stream: u64,
) -> Result<()> {
    spark_runtime::cublaslt::bf16_gemm_act_weight_t(
        input.0, weight.0, output.0, rows, columns, inner, stream,
    )
}

fn exact(ptr: DevicePtr, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer { ptr, bytes }
}

fn exl(buffer: GgmlIqBuffer) -> Glm53Exl3Buffer {
    Glm53Exl3Buffer {
        ptr: buffer.ptr,
        bytes: buffer.bytes,
    }
}
