// SPDX-License-Identifier: AGPL-3.0-only

//! Owning single-sequence GLM-5.3 EXL3 target used to qualify DFlash2.

use std::path::Path;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::ops::Glm53VisionPlan;
use crate::layers::ops::{
    GLM53_EXL3_MAX_WIDE_ROWS, GgmlIqBuffer, Glm53Exl3Buffer, Glm53Exl3MoePointerTables,
    Glm53Exl3RoutePolicy, glm53_layer_major_prefill_active,
};
use crate::layers::{Glm53TargetGeometry, Glm53TargetSchedule, forward_glm53_exl3_image};
use crate::weight_loader::{
    Glm53Exl3DeviceStore, Glm53Exl3FfnWeights, Glm53Exl3Linear, Glm53Exl3NativeDtype,
    Glm53Exl3NativeStore, Glm53Exl3TargetWeights, Glm53Exl3VisionCatalog,
};

use super::arena::Glm53ArenaPlan;
use super::capture_slots::Glm53CaptureSlots;
use super::dflash2_runtime::Glm53Dflash2Runtime;
use super::dispatch::{Glm53AttentionBinding, Glm53Dispatcher, Glm53LayerHyperOperands};
use super::dsa_attention::{Glm53DsaCacheSlots, Glm53DsaLayerGeometry};
use super::kda_attention::Glm53KdaConvSlots;
use super::kda_state_binding::Glm53KdaScratchState;
use super::owned_verify_readback::OwnedReadback;
use super::partial_replay::PartialReplaySetting;
use super::phase_timing::{HostClock, Phase, PhaseRecorder, TimingContext, TimingSetting};
use super::prefill_capture_owner::CaptureBankOwner;
use super::prefill_owner_exl3::PreparedOwner;
use super::target_model::Glm53Model;
use super::verify_policy_binding::VerifyBindingOwner;
use super::walk_scratch::Glm53WalkScratch;
use super::workspace_binding::Glm53BoundWorkspace;

const VOCAB: u32 = 154_880;
const HIDDEN: u32 = 4_096;
const DFLASH2_MAX_ROWS: usize = 8;
const DFLASH2_ARGMAX_INDEX_BYTES: usize = DFLASH2_MAX_ROWS * size_of::<u32>();
const DFLASH2_ARGMAX_VALUE_BYTES: usize = DFLASH2_MAX_ROWS * size_of::<f32>();
const DFLASH2_ARGMAX_BYTES: usize = DFLASH2_ARGMAX_INDEX_BYTES + DFLASH2_ARGMAX_VALUE_BYTES;

#[path = "dsa_verify_execution.rs"]
mod dsa_verify_execution;
#[path = "target_dflash2_probe.rs"]
mod target_dflash2_probe;
#[path = "target_partial_replay.rs"]
mod target_partial_replay;
#[path = "target_policy_binding.rs"]
mod target_policy_binding;
#[path = "target_policy_readback.rs"]
mod target_policy_readback;
#[path = "target_prefill_capture_exl3.rs"]
mod target_prefill_capture_exl3;
#[path = "target_prefill_exl3.rs"]
mod target_prefill_exl3;
#[path = "target_staged_exl3.rs"]
mod target_staged_exl3;
#[path = "target_state_probe.rs"]
mod target_state_probe;
pub use target_state_probe::{Glm53StateProbe, StateProbeDescriptor, StateProbeRegion};
#[path = "target_vision_exl3.rs"]
mod target_vision_exl3;

fn dflash2_device_argmax_enabled_from(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(value) => bail!(
            "ATLAS_GLM53_DFLASH2_DEVICE_ARGMAX must be absent or exactly 0 or 1; got {value:?}"
        ),
    }
}

fn dflash2_device_argmax_enabled() -> Result<bool> {
    match std::env::var("ATLAS_GLM53_DFLASH2_DEVICE_ARGMAX") {
        Ok(value) => dflash2_device_argmax_enabled_from(Some(&value)),
        Err(std::env::VarError::NotPresent) => dflash2_device_argmax_enabled_from(None),
        Err(error @ std::env::VarError::NotUnicode(_)) => {
            bail!("ATLAS_GLM53_DFLASH2_DEVICE_ARGMAX must be valid UTF-8: {error}")
        }
    }
}

struct WalkState {
    position: u32,
    nonce: u64,
    generation: u64,
    poisoned_stream: Option<u64>,
}

#[derive(Clone, Copy)]
enum WalkInput {
    Token(u32),
    ExternalEmbedding(DevicePtr),
}

fn external_embedding_rows(embeddings: Glm53Exl3Buffer) -> Result<usize> {
    let row_bytes = HIDDEN as usize * 2;
    ensure!(
        embeddings.ptr != DevicePtr::NULL
            && embeddings.bytes != 0
            && embeddings.bytes % row_bytes == 0,
        "GLM EXL3 external embeddings must be nonempty BF16 H4096 rows"
    );
    Ok(embeddings.bytes / row_bytes)
}

#[must_use = "GLM EXL3 model owns device stores and requires explicit free"]
pub struct Glm53Exl3Model {
    gpu: Box<dyn GpuBackend>,
    weights: Glm53Exl3TargetWeights,
    native_store: Glm53Exl3NativeStore,
    device_store: Glm53Exl3DeviceStore,
    arena: DevicePtr,
    wide_workspace_allocation: DevicePtr,
    wide_workspace_bytes: usize,
    scratch_allocation: DevicePtr,
    scratch_bytes: usize,
    moe_tables_allocation: DevicePtr,
    moe_tables: Vec<Option<Glm53Exl3MoePointerTables>>,
    scratch: Glm53WalkScratch,
    captures: Glm53CaptureSlots,
    hyper: Vec<Glm53LayerHyperOperands>,
    kda_states: Vec<Glm53KdaScratchState>,
    kda_conv: Vec<Glm53KdaConvSlots>,
    dsa_cache: Vec<Glm53DsaCacheSlots>,
    dsa_verify_backup: Mutex<DevicePtr>,
    schedule: Glm53TargetSchedule,
    logits: DevicePtr,
    dflash2_argmax_allocation: DevicePtr,
    dflash2_argmax_kernel: KernelHandle,
    dflash2_device_argmax: bool,
    phase_timing: TimingSetting,
    partial_replay: PartialReplaySetting,
    ffn_graphs: super::ffn_graph_runtime::FfnGraphs,
    capacity: u32,
    plan: Glm53ArenaPlan,
    state: Mutex<WalkState>,
    live_sequence: AtomicBool,
    verify_binding_owner: VerifyBindingOwner,
    verify_readback: Mutex<OwnedReadback>,
    vision_catalog: Option<Glm53Exl3VisionCatalog>,
    prepared_vision: Mutex<PreparedOwner>,
    prefill_capture_bank: Mutex<CaptureBankOwner>,
    dflash2: Mutex<Option<Glm53Dflash2Runtime>>,
}

impl Glm53Exl3Model {
    pub fn gpu(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    pub fn copy_dflash_capture_row(
        &self,
        row: u32,
        destination: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.copy_dflash_capture_rows(row, 1, destination, stream)
    }

    pub fn copy_dflash_capture_rows(
        &self,
        first_row: u32,
        rows: u32,
        destination: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            destination != DevicePtr::NULL && rows != 0 && first_row + rows <= 8,
            "GLM DFlash2 capture row batch is invalid"
        );
        for output_row in 0..rows {
            for slot in 0..5u32 {
                let source = self.captures.slot_row(slot, first_row + output_row)?;
                let output_slot = output_row * 5 + slot;
                self.gpu.copy_d2d_async(
                    source.ptr,
                    DevicePtr(destination.0 + u64::from(output_slot) * source.bytes as u64),
                    source.bytes,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    pub fn dflash_lm_head(&self) -> &Glm53Exl3Linear {
        &self.weights.lm_head
    }

    pub fn position(&self) -> u32 {
        self.state.lock().unwrap().position
    }

    pub fn embed_dflash_tokens(
        &self,
        tokens: &[u32],
        destination: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        let row_bytes = HIDDEN as usize * 2;
        ensure!(
            !tokens.is_empty() && destination.bytes == tokens.len() * row_bytes,
            "GLM DFlash2 embedding batch extent drift"
        );
        for (row, &token) in tokens.iter().enumerate() {
            self.embed(
                token,
                GgmlIqBuffer {
                    ptr: DevicePtr(destination.ptr.0 + (row * row_bytes) as u64),
                    bytes: row_bytes,
                },
                stream,
            )?;
        }
        Ok(())
    }

    pub fn new(
        gpu: Box<dyn GpuBackend>,
        device_store: Glm53Exl3DeviceStore,
        native_store: Glm53Exl3NativeStore,
        weights: Glm53Exl3TargetWeights,
        positions: u32,
    ) -> Result<Self> {
        Self::new_inner(gpu, device_store, native_store, weights, None, positions)
    }

    pub fn new_multimodal(
        gpu: Box<dyn GpuBackend>,
        device_store: Glm53Exl3DeviceStore,
        native_store: Glm53Exl3NativeStore,
        weights: Glm53Exl3TargetWeights,
        vision_catalog: Glm53Exl3VisionCatalog,
        positions: u32,
    ) -> Result<Self> {
        Self::new_inner(
            gpu,
            device_store,
            native_store,
            weights,
            Some(vision_catalog),
            positions,
        )
    }

    fn new_inner(
        gpu: Box<dyn GpuBackend>,
        device_store: Glm53Exl3DeviceStore,
        native_store: Glm53Exl3NativeStore,
        weights: Glm53Exl3TargetWeights,
        vision_catalog: Option<Glm53Exl3VisionCatalog>,
        positions: u32,
    ) -> Result<Self> {
        Glm53Model::admit()?;
        let cublaslt_prewarm = super::cublaslt_prewarm::Setting::from_env()?;
        let ffn_graphs = super::ffn_graph_runtime::FfnGraphs::from_env()?;
        let route_policy = Glm53Exl3RoutePolicy::parse(
            std::env::var_os("ATLAS_GLM53_EXL3_ROUTE_PRIVATE").as_deref(),
            std::env::var_os("ATLAS_GLM53_EXL3_ROUTE_PRIVATE_PREFILL").as_deref(),
        )?;
        route_policy.validate_moe_mode(std::env::var_os("ATLAS_GLM53_EXL3_MOE").as_deref())?;
        let dflash2_device_argmax = dflash2_device_argmax_enabled()?;
        let phase_timing =
            TimingSetting::parse(std::env::var_os("ATLAS_GLM53_PHASE_TIMING").as_deref())?;
        let partial_replay = PartialReplaySetting::parse(
            std::env::var_os("ATLAS_GLM53_PARTIAL_WIDE_REPLAY").as_deref(),
        )?;
        ensure!(
            weights.embedding.dtype() == Glm53Exl3NativeDtype::Bf16
                && weights.embedding.shape() == [VOCAB as u64, HIDDEN as u64],
            "GLM EXL3 embedding dtype/shape drift"
        );
        ensure!(
            weights.output_norm.dtype() == Glm53Exl3NativeDtype::F32
                && weights.output_norm.shape() == [HIDDEN as u64],
            "GLM EXL3 output norm dtype/shape drift"
        );
        let (moe_tables_allocation, moe_tables) = build_moe_pointer_tables(gpu.as_ref(), &weights)?;
        let plan = Glm53ArenaPlan::for_context(positions)?;
        let arena = gpu
            .alloc(usize::try_from(plan.known_bytes)?)
            .context("GLM EXL3 arena allocation")?;
        gpu.memset(arena, 0, usize::try_from(plan.known_bytes)?)?;
        let scratch_extent = Glm53WalkScratch::required_bytes_with_route_policy(route_policy);
        let scratch_bytes = usize::try_from(scratch_extent)?
            .checked_add(256)
            .context("GLM EXL3 aligned scratch allocation overflow")?;
        let scratch_allocation = gpu.alloc(scratch_bytes)?;
        gpu.memset(scratch_allocation, 0, scratch_bytes)?;
        let scratch_base = DevicePtr(
            scratch_allocation
                .0
                .checked_add(255)
                .context("GLM EXL3 scratch alignment overflow")?
                & !255,
        );
        let scratch =
            Glm53WalkScratch::bind_with_route_policy(scratch_base, scratch_extent, route_policy)?;
        cublaslt_prewarm.initialize(
            gpu.as_ref(),
            scratch_base,
            usize::try_from(scratch_extent)?,
        )?;
        let captures = Glm53CaptureSlots::bind(
            DevicePtr(arena.0 + plan.dflash_captures.offset_bytes),
            plan.dflash_captures.allocation_bytes,
        )?;
        let hyper = weights
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                Glm53LayerHyperOperands::resolve_exl3(index, &layer.hyper, &layer.norms)
            })
            .collect::<Result<Vec<_>>>()?;
        let wide_schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(u32::try_from(
            GLM53_EXL3_MAX_WIDE_ROWS,
        )?))?;
        let wide_workspace_bytes = usize::try_from(wide_schedule.workspace.arena_bytes)? + 256;
        let wide_workspace_allocation = gpu.alloc(wide_workspace_bytes)?;
        gpu.memset(wide_workspace_allocation, 0, wide_workspace_bytes)?;
        let logits = gpu.alloc(GLM53_EXL3_MAX_WIDE_ROWS * VOCAB as usize * 2)?;
        let dflash2_argmax_allocation = gpu.alloc(DFLASH2_ARGMAX_BYTES)?;
        gpu.memset(dflash2_argmax_allocation, 0, DFLASH2_ARGMAX_BYTES)?;
        let dflash2_argmax_kernel = gpu.kernel("glm53_kda", "atlas_glm53_argmax_bf16_rows")?;
        let (kda_states, kda_conv, dsa_cache) = Glm53Model::bind_state(&plan, arena)?;
        Ok(Self {
            gpu,
            weights,
            native_store,
            device_store,
            arena,
            wide_workspace_allocation,
            wide_workspace_bytes,
            scratch_allocation,
            scratch_bytes,
            moe_tables_allocation,
            moe_tables,
            scratch,
            captures,
            hyper,
            kda_states,
            kda_conv,
            dsa_cache,
            dsa_verify_backup: Mutex::new(DevicePtr::NULL),
            schedule: Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1))?,
            logits,
            dflash2_argmax_allocation,
            dflash2_argmax_kernel,
            dflash2_device_argmax,
            phase_timing,
            partial_replay,
            ffn_graphs,
            capacity: positions,
            plan,
            state: Mutex::new(WalkState {
                position: 0,
                nonce: 1,
                generation: 1,
                poisoned_stream: None,
            }),
            live_sequence: AtomicBool::new(false),
            verify_binding_owner: VerifyBindingOwner::new(),
            verify_readback: Mutex::new(OwnedReadback::new(
                GLM53_EXL3_MAX_WIDE_ROWS as usize * VOCAB as usize * 2,
            )?),
            vision_catalog,
            prepared_vision: Mutex::new(PreparedOwner::new()),
            prefill_capture_bank: Mutex::new(CaptureBankOwner::new()),
            dflash2: Mutex::new(None),
        })
    }

    pub fn install_dflash2(&self, root: &Path) -> Result<()> {
        let mut guard = self.dflash2.lock().unwrap();
        ensure!(guard.is_none(), "GLM DFlash2 runtime is already installed");
        *guard = Some(Glm53Dflash2Runtime::load(
            self.gpu.as_ref(),
            root,
            &self.weights.lm_head,
        )?);
        Ok(())
    }

    pub(super) fn has_dflash2(&self) -> bool {
        self.dflash2.lock().unwrap().is_some()
    }

    pub(in crate::model::glm53) fn ensure_dflash2_proposal_ready(&self) -> Result<()> {
        self.ensure_verify_healthy()
    }

    pub(super) fn propose_dflash2(
        &self,
        anchor: u32,
        requested: usize,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.ensure_verify_healthy()?;
        ensure!(
            (1..=7).contains(&requested),
            "GLM DFlash2 supports 1..=7 drafts"
        );
        let identity = self.phase_timing.enabled().then(|| {
            let state = self.state.lock().unwrap();
            (state.generation, state.position)
        });
        let guard = self.dflash2.lock().unwrap();
        let runtime = guard
            .as_ref()
            .context("GLM DFlash2 runtime is not installed")?;
        let clock = HostClock::default();
        let timing = PhaseRecorder::new(self.phase_timing, &clock);
        let result = timing.measure(Phase::Proposal, || {
            Ok(runtime.propose(self, anchor, stream)?[..requested].to_vec())
        });
        if let Some((generation, start)) = identity {
            timing.emit(TimingContext {
                kind: "proposal",
                generation,
                start: start as usize,
                rows: DFLASH2_MAX_ROWS,
                stream,
                outcome: if result.is_ok() { "proposed" } else { "error" },
                accepted: None,
                readback_bytes: 0,
            });
        }
        result
    }

    pub(super) fn claim_sequence(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let generation = state
            .generation
            .checked_add(1)
            .context("GLM request generation exhausted")?;
        ensure!(
            self.live_sequence
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "GLM EXL3 serves one sequence; a second concurrent request was refused"
        );
        state.generation = generation;
        Ok(())
    }

    pub(super) fn release_sequence(&self) {
        self.live_sequence.store(false, Ordering::Release);
    }

    pub(super) fn vocab(&self) -> usize {
        VOCAB as usize
    }

    pub(super) fn logits_ptr(&self) -> DevicePtr {
        self.logits
    }

    pub(super) fn bind_thread(&self) -> Result<()> {
        self.gpu.bind_to_thread()
    }

    pub(super) fn copy_logits(&self, logits: DevicePtr, destination: &mut [u8]) -> Result<()> {
        // This also supports scalar BF16 probes and every requested verify row.
        self.copy_policy_logits(
            logits,
            destination.len(),
            destination,
            self.gpu.default_stream(),
        )
    }

    fn workspace(&self) -> Result<Glm53BoundWorkspace> {
        Glm53BoundWorkspace::bind(
            &self.plan.workspace,
            DevicePtr(self.arena.0 + self.plan.transient.offset_bytes),
            self.plan.workspace.arena_bytes,
        )
    }

    fn embed(&self, token: u32, destination: GgmlIqBuffer, stream: u64) -> Result<()> {
        ensure!(
            token < VOCAB,
            "GLM EXL3 token {token} is outside vocabulary"
        );
        let row_bytes = HIDDEN as usize * 2;
        ensure!(
            destination.bytes == row_bytes,
            "GLM EXL3 embedding destination drift"
        );
        let offset = u64::from(token) * row_bytes as u64;
        self.gpu.copy_d2d_async(
            DevicePtr(self.weights.embedding.ptr().0 + offset),
            destination.ptr,
            row_bytes,
            stream,
        )
    }

    fn commit_accepted(&self, stream: u64) -> Result<()> {
        for state in &self.kda_states {
            let staged = state.buffer();
            self.gpu
                .copy_d2d_async(staged.ptr, state.persistent().ptr, staged.bytes, stream)?;
        }
        for conv in &self.kda_conv {
            self.gpu.copy_d2d_async(
                conv.staged_state_f32.ptr,
                conv.persistent_state_f32.ptr,
                conv.staged_state_f32.bytes,
                stream,
            )?;
        }
        Ok(())
    }

    fn walk_input(&self, input: WalkInput, stream: u64) -> Result<DevicePtr> {
        self.ensure_verify_healthy()?;
        let (position, nonce) = {
            let mut state = self.state.lock().unwrap();
            let position = state.position;
            state.nonce = state.nonce.wrapping_add(1).max(1);
            (position, state.nonce)
        };
        ensure!(
            position < self.capacity,
            "GLM EXL3 sequence reached capacity"
        );
        let workspace = self.workspace()?;
        match input {
            WalkInput::Token(token) => self.embed(token, workspace.collapsed, stream)?,
            WalkInput::ExternalEmbedding(source) => {
                ensure!(
                    source != DevicePtr::NULL,
                    "GLM EXL3 external embedding is null"
                );
                self.gpu.copy_d2d_async(
                    source,
                    workspace.collapsed.ptr,
                    HIDDEN as usize * 2,
                    stream,
                )?;
            }
        }
        let dispatcher = Glm53Dispatcher::new_exl3(
            self.gpu.as_ref(),
            1,
            workspace,
            self.hyper.clone(),
            GgmlIqBuffer {
                ptr: self.weights.output_norm.ptr(),
                bytes: self.weights.output_norm.bytes(),
            },
            self.scratch,
            self.captures,
            &self.weights.layers,
            &self.moe_tables,
            Glm53AttentionBinding {
                kda_states: self.kda_states.clone(),
                kda_conv: self.kda_conv.clone(),
                dsa_cache: self.dsa_cache.clone(),
                geometry: Glm53DsaLayerGeometry {
                    position,
                    capacity: self.capacity,
                    nonce,
                },
            },
            &self.weights.lm_head,
            GgmlIqBuffer {
                ptr: self.logits,
                bytes: VOCAB as usize * 2,
            },
        )?;
        for event in self.schedule.events() {
            dispatcher
                .dispatch(self.gpu.as_ref(), event, stream)
                .with_context(|| format!("GLM EXL3 walk failed at {event:?}"))?;
        }
        self.commit_accepted(stream)?;
        self.gpu.synchronize(stream)?;
        self.state.lock().unwrap().position = position + 1;
        if let Some(runtime) = self.dflash2.lock().unwrap().as_mut() {
            runtime.observe_target(self, stream)?;
        }
        Ok(self.logits)
    }

    pub(super) fn walk(&self, token: u32, stream: u64) -> Result<DevicePtr> {
        self.walk_input(WalkInput::Token(token), stream)
    }

    pub(super) fn observe_committed_wide_rows(&self, rows: u32, stream: u64) -> Result<()> {
        if let Some(runtime) = self.dflash2.lock().unwrap().as_mut() {
            runtime.observe_target_rows(self, rows, stream)?;
        }
        Ok(())
    }

    pub fn decode_token(&self, token: u32, stream: u64) -> Result<DevicePtr> {
        self.walk(token, stream)
    }

    /// Consume caller-owned `[rows,4096]` BF16 embeddings through the same
    /// sequential target state transition used by token embeddings. This is
    /// the exact language-side splice seam for GLM vision rows.
    pub fn prefill_embeddings(
        &self,
        embeddings: Glm53Exl3Buffer,
        stream: u64,
    ) -> Result<DevicePtr> {
        let row_bytes = HIDDEN as usize * 2;
        let rows = external_embedding_rows(embeddings)?;
        ensure!(
            self.position()
                .checked_add(u32::try_from(rows)?)
                .is_some_and(|end| end <= self.capacity),
            "GLM EXL3 external embeddings exceed sequence capacity"
        );
        let mut logits = self.logits;
        for row in 0..rows {
            logits = self.walk_input(
                WalkInput::ExternalEmbedding(embeddings.ptr.offset(row * row_bytes)),
                stream,
            )?;
        }
        Ok(logits)
    }

    fn verify_tokens_staged(&self, tokens: &[u32], stream: u64) -> Result<(DevicePtr, u32, u32)> {
        self.verify_inputs_staged(tokens.len(), stream, |destination, _position| {
            self.embed_dflash_tokens(tokens, destination, stream)
        })
    }

    /// Execute one layer-major causal verifier chunk and commit its full state.
    pub fn verify_tokens_full(&self, tokens: &[u32], stream: u64) -> Result<DevicePtr> {
        let (logits, position, rows) = self.verify_tokens_staged(tokens, stream)?;
        self.commit_accepted(stream)?;
        self.gpu.synchronize(stream)?;
        self.state.lock().unwrap().position = position + rows;
        Ok(logits)
    }

    /// Capture one fixed-position K8 verifier body, then time exactly one graph
    /// replay. This is deliberately a qualification probe rather than a graph
    /// cache: kernel scalar arguments still contain `position` and `nonce`, so
    /// reusing the executable at another position would be incorrect.
    pub fn verify_tokens_full_graph_probe(
        &self,
        tokens: &[u32],
        stream: u64,
    ) -> Result<(DevicePtr, f64)> {
        self.ensure_verify_healthy()?;
        ensure!(
            stream != 0,
            "GLM EXL3 graph probe needs a non-legacy stream"
        );
        ensure!(
            std::env::var_os("ATLAS_GLM53_WIDE_TIMING").is_none(),
            "GLM EXL3 event timing cannot synchronize during graph capture"
        );
        ensure!(
            (2..=8).contains(&tokens.len()),
            "GLM EXL3 graph verifier needs 2..=8 rows"
        );
        let rows = u32::try_from(tokens.len())?;
        let (position, nonce) = {
            let mut state = self.state.lock().unwrap();
            let position = state.position;
            ensure!(
                position
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.capacity),
                "GLM EXL3 graph verifier chunk exceeds sequence capacity"
            );
            state.nonce = state.nonce.wrapping_add(1).max(1);
            (position, state.nonce)
        };
        let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(rows))?;
        let base = DevicePtr((self.wide_workspace_allocation.0 + 255) & !255);
        let workspace =
            Glm53BoundWorkspace::bind(&schedule.workspace, base, schedule.workspace.arena_bytes)?;
        let dispatcher = Glm53Dispatcher::new_exl3(
            self.gpu.as_ref(),
            rows,
            workspace,
            self.hyper.clone(),
            GgmlIqBuffer {
                ptr: self.weights.output_norm.ptr(),
                bytes: self.weights.output_norm.bytes(),
            },
            self.scratch,
            self.captures,
            &self.weights.layers,
            &self.moe_tables,
            Glm53AttentionBinding {
                kda_states: self.kda_states.clone(),
                kda_conv: self.kda_conv.clone(),
                dsa_cache: self.dsa_cache.clone(),
                geometry: Glm53DsaLayerGeometry {
                    position,
                    capacity: self.capacity,
                    nonce,
                },
            },
            &self.weights.lm_head,
            GgmlIqBuffer {
                ptr: self.logits,
                bytes: rows as usize * VOCAB as usize * 2,
            },
        )?;

        self.gpu.begin_capture(stream)?;
        let body_result = (|| -> Result<()> {
            self.embed_dflash_tokens(tokens, workspace.collapsed, stream)?;
            for event in schedule.events() {
                dispatcher
                    .dispatch(self.gpu.as_ref(), event, stream)
                    .with_context(|| format!("GLM EXL3 graph capture failed at {event:?}"))?;
            }
            self.commit_accepted(stream)
        })();
        if let Err(body_error) = body_result {
            let cleanup = self
                .gpu
                .end_capture(stream)
                .and_then(|graph| self.gpu.destroy_graph(graph));
            return match cleanup {
                Ok(()) => Err(body_error),
                Err(cleanup_error) => Err(body_error.context(format!(
                    "CUDA graph capture cleanup also failed: {cleanup_error:#}"
                ))),
            };
        }
        let graph = self.gpu.end_capture(stream)?;
        ensure!(
            graph.0 != 0,
            "GLM EXL3 graph capture returned a null handle"
        );
        let started = std::time::Instant::now();
        let replay_result = self
            .gpu
            .launch_graph(graph, stream)
            .and_then(|()| self.gpu.synchronize(stream));
        let replay_seconds = started.elapsed().as_secs_f64();
        let destroy_result = self.gpu.destroy_graph(graph);
        replay_result.context("GLM EXL3 graph replay")?;
        destroy_result.context("GLM EXL3 graph destroy")?;
        self.state.lock().unwrap().position = position + rows;
        Ok((self.logits, replay_seconds))
    }

    pub(super) fn verify_dflash2_scheduler(&self, tokens: &[u32], stream: u64) -> Result<Vec<u32>> {
        self.verify_dflash2_transaction(tokens, stream)
    }

    pub fn prefill_tokens(&self, tokens: &[u32], stream: u64) -> Result<DevicePtr> {
        ensure!(!tokens.is_empty(), "GLM EXL3 prefill needs a token");
        let mut logits = self.logits;
        for &token in tokens {
            logits = self.walk(token, stream)?;
        }
        Ok(logits)
    }

    pub fn argmax_host(&self, logits: DevicePtr) -> Result<u32> {
        let mut bytes = vec![0u8; VOCAB as usize * 2];
        self.copy_logits(logits, &mut bytes)?;
        let mut best = (0u32, f32::NEG_INFINITY);
        for (index, pair) in bytes.chunks_exact(2).enumerate() {
            let value = half::bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32();
            if value > best.1 {
                best = (index as u32, value);
            }
        }
        ensure!(best.1.is_finite(), "GLM EXL3 logits are non-finite");
        Ok(best.0)
    }

    pub fn argmax_rows_host(&self, logits: DevicePtr, rows: usize) -> Result<Vec<u32>> {
        ensure!(
            (1..=8).contains(&rows),
            "GLM EXL3 argmax rows must be 1..=8"
        );
        let row_bytes = VOCAB as usize * 2;
        let mut bytes = vec![0u8; rows * row_bytes];
        self.copy_logits(logits, &mut bytes)?;
        bytes
            .chunks_exact(row_bytes)
            .map(|row| {
                let mut best = (0u32, f32::NEG_INFINITY);
                for (index, pair) in row.chunks_exact(2).enumerate() {
                    let value =
                        half::bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32();
                    if value > best.1 {
                        best = (index as u32, value);
                    }
                }
                ensure!(best.1.is_finite(), "GLM EXL3 logits are non-finite");
                Ok(best.0)
            })
            .collect()
    }

    fn argmax_rows_device(&self, logits: DevicePtr, rows: usize, stream: u64) -> Result<Vec<u32>> {
        ensure!(
            (1..=DFLASH2_MAX_ROWS).contains(&rows),
            "GLM EXL3 device argmax rows must be 1..={DFLASH2_MAX_ROWS}"
        );
        let values = self
            .dflash2_argmax_allocation
            .offset(DFLASH2_ARGMAX_INDEX_BYTES);
        KernelLaunch::new(self.gpu.as_ref(), self.dflash2_argmax_kernel)
            .grid([u32::try_from(rows)?, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(self.dflash2_argmax_allocation)
            .arg_ptr(values)
            .arg_u32(u32::try_from(rows)?)
            .arg_u32(VOCAB)
            .launch(stream)?;
        let mut receipt = [0u8; DFLASH2_ARGMAX_BYTES];
        self.gpu
            .copy_d2h_on_stream(self.dflash2_argmax_allocation, &mut receipt, stream)?;
        let mut indices = Vec::with_capacity(rows);
        for row in 0..rows {
            let index_offset = row * size_of::<u32>();
            let value_offset = DFLASH2_ARGMAX_INDEX_BYTES + row * size_of::<f32>();
            let index = u32::from_le_bytes(
                receipt[index_offset..index_offset + size_of::<u32>()]
                    .try_into()
                    .unwrap(),
            );
            let value = f32::from_le_bytes(
                receipt[value_offset..value_offset + size_of::<f32>()]
                    .try_into()
                    .unwrap(),
            );
            ensure!(
                index < VOCAB && value.is_finite(),
                "GLM EXL3 device argmax row {row} returned invalid index/value"
            );
            indices.push(index);
        }
        Ok(indices)
    }

    pub fn free(self) -> Result<()> {
        // Keep the complete model/backend lease through a failed or panicking
        // drafter fence. No field may be destructured or released before this.
        let this = super::dflash2_runtime::retain_until_drained(self, |model| {
            let guard = model.dflash2.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(runtime) = guard.as_ref() {
                runtime.drain_kv_prefix(model.gpu.as_ref())?;
            }
            Ok(())
        })?;
        let this = super::dflash2_runtime::retain_until_drained(this, |model| {
            model.ffn_graphs.drain(model.gpu.as_ref())
        })?;
        if let Err(error) = this.clear_prepared_vision() {
            // This consuming API cannot return its owner. Retain all fields on
            // failed draining; no GPU/backend/store teardown may run underneath
            // in-flight work. Resources remain unreclaimed until process teardown.
            std::mem::forget(this);
            return Err(error.context(
                "GLM shutdown refused: device/model resources remain unreclaimed until process teardown"
            ));
        }
        if let Err(error) = this.release_prefill_capture_bank() {
            std::mem::forget(this);
            return Err(error.context(
                "GLM shutdown refused: capture bank/model resources remain unreclaimed until process teardown"
            ));
        }
        let Self {
            gpu,
            weights,
            native_store,
            device_store,
            arena,
            wide_workspace_allocation,
            scratch_allocation,
            moe_tables_allocation,
            logits,
            dflash2_argmax_allocation,
            dsa_verify_backup,
            dflash2,
            ..
        } = this;
        let backup = dsa_verify_backup
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if backup != DevicePtr::NULL {
            gpu.free(backup)?;
        }
        if let Some(runtime) = dflash2
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            runtime.free(gpu.as_ref())?;
        }
        gpu.free(logits)?;
        gpu.free(dflash2_argmax_allocation)?;
        gpu.free(wide_workspace_allocation)?;
        gpu.free(moe_tables_allocation)?;
        gpu.free(scratch_allocation)?;
        gpu.free(arena)?;
        drop(weights);
        native_store.free(gpu.as_ref())?;
        device_store.free(gpu.as_ref()).map_err(|error| {
            anyhow::anyhow!("GLM EXL3 source store free failed: {:#}", error.failure())
        })
    }
}

#[cfg(test)]
mod vision_splice_tests {
    use super::*;

    const KDA_CUDA: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm5.3-flash/iq3/glm53_kda.cu"
    ));

    #[test]
    fn dflash2_device_argmax_selector_is_exact_and_default_on() {
        assert!(dflash2_device_argmax_enabled_from(None).unwrap());
        assert!(dflash2_device_argmax_enabled_from(Some("1")).unwrap());
        assert!(!dflash2_device_argmax_enabled_from(Some("0")).unwrap());
        assert!(dflash2_device_argmax_enabled_from(Some("")).is_err());
        assert!(dflash2_device_argmax_enabled_from(Some("true")).is_err());
    }

    #[test]
    fn dflash2_rows_argmax_cuda_preserves_first_index_and_reports_finite_max() {
        let signature = "atlas_glm53_argmax_bf16_rows(\n        const __nv_bfloat16 * __restrict__ logits,\n        unsigned int * __restrict__ out_indices,\n        float * __restrict__ out_values,\n        unsigned int rows, unsigned int columns)";
        assert!(KDA_CUDA.contains(signature));
        assert!(KDA_CUDA.contains("candidate == best && candidate_index < best_index"));
        assert!(KDA_CUDA.contains("out_indices[row] = shared_indices[0];"));
        assert!(KDA_CUDA.contains("out_values[row] = shared_values[0];"));
        let host = include_str!("target_model_exl3.rs");
        let device = host
            .split("fn argmax_rows_device")
            .nth(1)
            .expect("device argmax body");
        assert!(device.contains("copy_d2h_on_stream"));
    }

    #[test]
    fn external_embedding_extent_is_exact_bf16_h4096() {
        let row_bytes = HIDDEN as usize * 2;
        assert_eq!(
            external_embedding_rows(Glm53Exl3Buffer {
                ptr: DevicePtr(1),
                bytes: 16 * row_bytes,
            })
            .unwrap(),
            16
        );
        assert!(
            external_embedding_rows(Glm53Exl3Buffer {
                ptr: DevicePtr::NULL,
                bytes: row_bytes,
            })
            .is_err()
        );
        assert!(
            external_embedding_rows(Glm53Exl3Buffer {
                ptr: DevicePtr(1),
                bytes: row_bytes - 2,
            })
            .is_err()
        );
    }
}

fn build_moe_pointer_tables(
    gpu: &dyn GpuBackend,
    weights: &Glm53Exl3TargetWeights,
) -> Result<(DevicePtr, Vec<Option<Glm53Exl3MoePointerTables>>)> {
    const EXPERTS: usize = 288;
    const TABLE_BYTES: usize = EXPERTS * size_of::<u64>();
    const TABLES_PER_LAYER: usize = 9;

    let moe_layers = weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.ffn, Glm53Exl3FfnWeights::Moe(_)))
        .count();
    let total = moe_layers
        .checked_mul(TABLES_PER_LAYER * TABLE_BYTES)
        .context("GLM EXL3 MoE pointer table extent overflow")?;
    let mut host = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(weights.layers.len());
    for layer in &weights.layers {
        let Glm53Exl3FfnWeights::Moe(moe) = &layer.ffn else {
            offsets.push(None);
            continue;
        };
        ensure!(
            moe.experts.len() == EXPERTS,
            "GLM EXL3 MoE expert census drift"
        );
        let base = host.len();
        for select in [
            |expert: &crate::weight_loader::Glm53Exl3ExpertWeights| expert.gate.trellis_buffer(),
            |expert: &crate::weight_loader::Glm53Exl3ExpertWeights| expert.gate.scale_in_buffer(),
            |expert: &crate::weight_loader::Glm53Exl3ExpertWeights| expert.gate.scale_out_buffer(),
            |expert: &crate::weight_loader::Glm53Exl3ExpertWeights| expert.up.trellis_buffer(),
            |expert: &crate::weight_loader::Glm53Exl3ExpertWeights| expert.up.scale_in_buffer(),
            |expert: &crate::weight_loader::Glm53Exl3ExpertWeights| expert.up.scale_out_buffer(),
            |expert: &crate::weight_loader::Glm53Exl3ExpertWeights| expert.down.trellis_buffer(),
            |expert: &crate::weight_loader::Glm53Exl3ExpertWeights| expert.down.scale_in_buffer(),
            |expert: &crate::weight_loader::Glm53Exl3ExpertWeights| expert.down.scale_out_buffer(),
        ] {
            for expert in &moe.experts {
                ensure!(
                    expert.gate.bits() == 2 && expert.up.bits() == 2 && expert.down.bits() == 2,
                    "GLM EXL3 fused MoE requires uniform K2 experts"
                );
                host.extend_from_slice(&select(expert).ptr.0.to_le_bytes());
            }
        }
        offsets.push(Some(base));
    }
    ensure!(
        host.len() == total,
        "GLM EXL3 MoE pointer table packing drift"
    );
    let allocation = gpu.alloc(total)?;
    if let Err(error) = gpu.copy_h2d(&host, allocation) {
        let _ = gpu.free(allocation);
        return Err(error).context("GLM EXL3 MoE pointer table upload");
    }
    let table = |base: usize, index: usize| Glm53Exl3Buffer {
        ptr: DevicePtr(allocation.0 + (base + index * TABLE_BYTES) as u64),
        bytes: TABLE_BYTES,
    };
    let tables = offsets
        .into_iter()
        .map(|offset| {
            offset.map(|base| Glm53Exl3MoePointerTables {
                gate_trellis: table(base, 0),
                gate_suh: table(base, 1),
                gate_svh: table(base, 2),
                up_trellis: table(base, 3),
                up_suh: table(base, 4),
                up_svh: table(base, 5),
                down_trellis: table(base, 6),
                down_suh: table(base, 7),
                down_svh: table(base, 8),
            })
        })
        .collect();
    Ok((allocation, tables))
}
