// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek-V4.1 **routed MoE forward**: `y [T, hidden] bf16 -> routed sum [T, hidden] bf16`.
//!
//! Router (fp32 GEMM) -> `sqrt(softplus)` -> residency mask + per-row text/vision bias ->
//! top-6 -> expert-major permutation -> per expert: CB3 reconstruct + three cuBLASLt GEMMs ->
//! the engine's rounding points (`dsv41_moe_combine.cu`) -> fp32 sum over the six picks.
//!
//! This is the code `examples/cb3_moe_oracle_microtest.rs` validates, not a copy of it:
//! against the engine's `moe_routed` tap it measured rel_l2 1.1-2.8e-4 with 99.1% of outputs
//! bf16-bit-identical (runA/runC_2048/runE, layers 0 and 2), with a reversed-weights and a
//! wrong-expert control at 0.55 and 1.05.
//!
//! ## What it is NOT
//! - **Fast.** Routing runs on the HOST (one D2H of `[T, 384]` fp32 scores and a sync per
//!   layer), and each expert reconstructs ~70.8 MB of bf16 scratch. It is the correctness
//!   baseline a fused path must be measured against. Nothing here has been timed.
//! - **Graph-capturable**, for the same reason.
//!
//! ## Image rows
//! Rows at image-sentinel / image-pad positions route with `gate.bias_vl`. The forward
//! trait carries no token ids, so the driver MUST call [`Cb3RoutedMoe::set_pass_tokens`]
//! before each pass. Forgetting is an ERROR, not a text-only fallback: routing image rows
//! with the text bias changed 97 picks on runE_image and raises nothing.

use anyhow::{Context, Result, ensure};
use std::sync::{Arc, Mutex};

use spark_runtime::cublaslt::{GemmDtype, gemm_act_weight_t_typed};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::cb3_arena::Cb3ExpertArena;
use super::fwd::V41RoutedMoe;
use super::moe_decode::{self, MAX_DECODE_T, MoeDecode};
use super::moe::{
    COMBINE_MODULE, Cb3Matrix, FUSED_DOWN_FN, FUSED_GATE_UP_FN, FUSED_GEMM_MODULE, FUSED_TILE_M,
    FUSED_TILE_N, Cb3Permutation, Cb3Reconstruct, MOE_PERMUTE_MODULE,
    PERMUTE_KERNEL, ROUTE_TOPK_FN, SWIGLU_WEIGHTED_FN, UNPERMUTE_SUM_FN, expert_matrices, gemm_weight_t_f32out,
    group_by_expert,
};
use super::ops::{Dsv41Kernels, Ops};
use super::routing::{Routing, image_rows, score_of, select_experts_multimodal};

/// `num_experts_per_tok`.
pub const TOP_K: usize = 6;
/// Router id space.
pub const ROUTER_EXPERTS: usize = 384;
const BLOCK: u32 = 256;

/// One layer's router, in the dtypes the reference computes in.
///
/// `gate.weight` is bf16 on disk but the reference runs `y.float() @ gate_w.float()` — an
/// fp32 GEMM, because routing turns on ulps. It is widened ONCE at load. The two biases are
/// kept on the host: top-k runs there.
pub struct RouterF32 {
    pub gate_w: DevicePtr,
    /// The ORIGINAL bf16 `gate.weight` (store-owned, never freed here), when loaded from the
    /// store. The decode router reads this instead of the widened copy: bf16 -> fp32 is exact,
    /// so the logits are bit-identical at half the bytes (moe_decode.rs).
    pub gate_w_bf16: Option<DevicePtr>,
    pub bias: Vec<f32>,
    pub bias_vl: Vec<f32>,
}

impl RouterF32 {
    /// From the weight store (`layers.N.ffn.gate.{weight,bias,bias_vl}`).
    pub fn load(
        store: &WeightStore,
        layer: usize,
        hidden: usize,
        gpu: &dyn GpuBackend,
        kernels: &Dsv41Kernels,
        stream: u64,
    ) -> Result<Self> {
        let p = format!("layers.{layer}.ffn.gate");
        let w = store.get(&format!("{p}.weight"))?;
        ensure!(w.dtype == WeightDtype::BF16, "{p}.weight: expected BF16, got {:?}", w.dtype);
        ensure!(w.shape == [ROUTER_EXPERTS, hidden], "{p}.weight: shape {:?}", w.shape);
        let n = ROUTER_EXPERTS * hidden;
        let gate_w = gpu.alloc(n * 4)?;
        KernelLaunch::new(gpu, kernels.bf16_to_f32)
            .block([BLOCK, 1, 1])
            .grid([(n as u32).div_ceil(BLOCK), 1, 1])
            .arg_ptr(w.ptr)
            .arg_ptr(gate_w)
            .arg_u64(n as u64)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let host_f32 = |name: &str| -> Result<Vec<f32>> {
            let t = store.get(name)?;
            ensure!(t.dtype == WeightDtype::FP32, "{name}: expected F32 (a bf16 cast rounds the \
                ~9.8 bias to 0.0625 steps, coarser than the spread that decides routing)");
            ensure!(t.shape == [ROUTER_EXPERTS], "{name}: shape {:?}", t.shape);
            let mut bytes = vec![0u8; ROUTER_EXPERTS * 4];
            gpu.copy_d2h(t.ptr, &mut bytes)?;
            Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        Ok(Self {
            gate_w,
            gate_w_bf16: Some(w.ptr),
            bias: host_f32(&format!("{p}.bias"))?,
            bias_vl: host_f32(&format!("{p}.bias_vl"))?,
        })
    }

    /// From host copies (bf16 bits for the weight, as on disk). For microtests that read
    /// the shards directly.
    pub fn from_host(
        gate_w_bf16: &[u16],
        bias: Vec<f32>,
        bias_vl: Vec<f32>,
        hidden: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(gate_w_bf16.len() == ROUTER_EXPERTS * hidden, "gate weight extent");
        ensure!(bias.len() == ROUTER_EXPERTS && bias_vl.len() == ROUTER_EXPERTS, "bias extent");
        let wide: Vec<u8> = gate_w_bf16
            .iter()
            .flat_map(|b| f32::from_bits((*b as u32) << 16).to_le_bytes())
            .collect();
        let gate_w = gpu.alloc(wide.len())?;
        gpu.copy_h2d(&wide, gate_w)?;
        Ok(Self { gate_w, gate_w_bf16: None, bias, bias_vl })
    }
}

/// Device buffers for one pass of up to `max_t` tokens. Sized once; reused per layer.
struct Scratch {
    max_t: usize,
    y_f32: DevicePtr,
    logits: DevicePtr,
    perm: DevicePtr,
    gate: DevicePtr,
    up: DevicePtr,
    h: DevicePtr,
    down: DevicePtr,
    w1: DevicePtr,
    w3: DevicePtr,
    w2: DevicePtr,
    sorted: DevicePtr,
    tok2perm: DevicePtr,
    row_w: DevicePtr,
    image: DevicePtr,
    route_idx: DevicePtr,
    route_w: DevicePtr,
}

/// Per-pass knobs that exist for NEGATIVE CONTROLS only. Each produces finite, correctly
/// shaped, wrong output; a gate that passes one of them is not a gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MoeControl {
    #[default]
    None,
    /// Reverse each token's route weights across its picks.
    ReverseWeights,
    /// Reconstruct every group from the NEXT resident slot: a real expert, the wrong one.
    NextSlot,
}

/// How the expert GEMMs get their weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExpertKernel {
    /// Decode CB3 in shared memory inside a grouped tensor-core GEMM (`cb3_moe_gemm.cu`).
    /// Validated against the engine (moe_routed rel_l2 1.4-2.8e-4, 98.8-99.0% bf16-bit-
    /// identical, full router, runA/runC_2048/runE_image L0+L2, controls at 0.55/1.06) and
    /// 2.5x (T=2048) to 7x (T<=128) faster than `Reconstruct`.
    #[default]
    Fused,
    /// Reconstruct each expert to 70.8 MB of bf16 scratch, then cuBLASLt. The baseline the
    /// fused path is gated and timed against.
    Reconstruct,
}

/// Which parts of the per-expert loop run. **TIMING ONLY** — every value but `All` produces
/// wrong output, on purpose, so the cost of each part can be read by subtraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExpertWork {
    #[default]
    All,
    /// CB3 reconstruct only; no GEMMs, no SwiGLU.
    ReconstructOnly,
    /// GEMMs + SwiGLU against whatever the scratch weights hold; no reconstruct.
    GemmOnly,
    /// Neither: what remains is host planning, uploads, permute and the final sum.
    Skip,
}

pub struct Cb3RoutedMoe<'a> {
    gpu: &'a dyn GpuBackend,
    kernels: &'a Dsv41Kernels,
    arena: Arc<Cb3ExpertArena>,
    routers: Vec<(usize, RouterF32)>,
    reconstruct: Cb3Reconstruct,
    matrices: [Cb3Matrix; 3],
    k_permute: KernelHandle,
    k_swiglu: KernelHandle,
    k_sum: KernelHandle,
    hidden: usize,
    inter: usize,
    swiglu_limit: f32,
    route_scale: f32,
    scratch: Scratch,
    pass_tokens: Mutex<Option<Vec<i64>>>,
    control: Mutex<MoeControl>,
    work: Mutex<ExpertWork>,
    /// The decode-size path (t <= 8): GPU routing + CB3 GEMV. On unless `ATLAS_DSV41_MOE_DECODE=0`.
    decode: Option<MoeDecode>,
    kernel: Mutex<ExpertKernel>,
    /// Per layer: (layer, bias, bias_vl, resident u8 mask), all [384] on the device.
    device_routers: Vec<(usize, DevicePtr, DevicePtr, DevicePtr)>,
    k_route: KernelHandle,
    k_fused_gate_up: KernelHandle,
    k_fused_down: KernelHandle,
    tiles: DevicePtr,
    max_tiles: usize,
}

impl<'a> Cb3RoutedMoe<'a> {
    /// `routers` must cover every layer the arena holds, keyed by REAL layer index.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &'a dyn GpuBackend,
        kernels: &'a Dsv41Kernels,
        config: &atlas_core::config::ModelConfig,
        arena: Arc<Cb3ExpertArena>,
        routers: Vec<(usize, RouterF32)>,
        swiglu_limit: f32,
        route_scale: f32,
        max_t: usize,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let inter = config.moe_intermediate_size;
        ensure!(swiglu_limit > 0.0, "swiglu_limit must be positive, got {swiglu_limit}");
        ensure!(max_t > 0, "max_t must be positive");
        for (layer, _) in &routers {
            arena.layer(*layer).with_context(|| format!("router for layer {layer} has no resident experts"))?;
        }
        let decode = if moe_decode::enabled() { Some(MoeDecode::new(gpu, &arena, &routers)?) } else { None };
        ensure!(
            inter % FUSED_TILE_N == 0 && hidden % FUSED_TILE_N == 0 && inter % 32 == 0 && hidden % 32 == 0,
            "fused CB3 GEMM needs N % {FUSED_TILE_N} == 0 and K % 32 == 0 (hidden {hidden}, inter {inter})"
        );
        let e = max_t * TOP_K;
        // Every group contributes ceil(rows / BM) tiles: at most e / BM + one partial per expert.
        let max_tiles = e / FUSED_TILE_M + ROUTER_EXPERTS;
        let a = |bytes: usize| gpu.alloc(bytes);
        let scratch = Scratch {
            max_t,
            y_f32: a(max_t * hidden * 4)?,
            logits: a(max_t * ROUTER_EXPERTS * 4)?,
            perm: a(e * hidden * 2)?,
            gate: a(e * inter * 4)?,
            up: a(e * inter * 4)?,
            h: a(e * inter * 2)?,
            down: a(e * hidden * 4)?,
            w1: a(inter * hidden * 2)?,
            w3: a(inter * hidden * 2)?,
            w2: a(hidden * inter * 2)?,
            sorted: a(e * 4)?,
            tok2perm: a(e * 4)?,
            row_w: a(e * 4)?,
            image: a(max_t)?,
            route_idx: a(e * 4)?,
            route_w: a(e * 4)?,
        };
        // Per layer, on the device: text bias, vision bias, residency mask (u8).
        let mut device_routers = Vec::with_capacity(routers.len());
        for (layer, router) in &routers {
            let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let bias = gpu.alloc(ROUTER_EXPERTS * 4)?;
            gpu.copy_h2d(&f32_bytes(&router.bias), bias)?;
            let bias_vl = gpu.alloc(ROUTER_EXPERTS * 4)?;
            gpu.copy_h2d(&f32_bytes(&router.bias_vl), bias_vl)?;
            let mask: Vec<u8> = arena.routing_mask(*layer)?.iter().map(|m| u8::from(*m)).collect();
            ensure!(mask.len() == ROUTER_EXPERTS, "residency mask is {} wide", mask.len());
            let resident = gpu.alloc(ROUTER_EXPERTS)?;
            gpu.copy_h2d(&mask, resident)?;
            device_routers.push((*layer, bias, bias_vl, resident));
        }
        Ok(Self {
            gpu,
            kernels,
            arena,
            routers,
            reconstruct: Cb3Reconstruct::new(gpu)?,
            matrices: expert_matrices(config)?,
            k_permute: gpu.kernel(MOE_PERMUTE_MODULE, PERMUTE_KERNEL)?,
            k_swiglu: gpu.kernel(COMBINE_MODULE, SWIGLU_WEIGHTED_FN)?,
            k_sum: gpu.kernel(COMBINE_MODULE, UNPERMUTE_SUM_FN)?,
            hidden,
            inter,
            swiglu_limit,
            route_scale,
            scratch,
            pass_tokens: Mutex::new(None),
            control: Mutex::new(MoeControl::None),
            work: Mutex::new(ExpertWork::All),
            decode,
            kernel: Mutex::new(ExpertKernel::default()),
            device_routers,
            k_route: gpu.kernel(COMBINE_MODULE, ROUTE_TOPK_FN)?,
            k_fused_gate_up: gpu.kernel(FUSED_GEMM_MODULE, FUSED_GATE_UP_FN)?,
            k_fused_down: gpu.kernel(FUSED_GEMM_MODULE, FUSED_DOWN_FN)?,
            tiles: gpu.alloc(max_tiles * 16)?,
            max_tiles,
        })
    }

    /// Token ids of the NEXT pass, so image rows route with `gate.bias_vl`. Required.
    pub fn set_pass_tokens(&self, token_ids: &[i64]) {
        *self.pass_tokens.lock().expect("pass_tokens lock") = Some(token_ids.to_vec());
    }

    /// Whether a `t`-row pass takes the decode path.
    pub fn uses_decode_path(&self, t: usize) -> bool {
        self.decode.is_some() && t <= MAX_DECODE_T
    }

    /// The decode path's last device routing (synchronous; gates only).
    pub fn decode_routing(&self, t: usize, stream: u64) -> Result<Routing> {
        self.decode.as_ref().context("decode path is off")?.read_routing(self.gpu, t, stream)
    }

    /// Fails if any decode routing kernel picked a non-resident expert (synchronous).
    pub fn check_decode_error(&self, stream: u64) -> Result<()> {
        match &self.decode {
            Some(d) => d.check_error(self.gpu, stream),
            None => Ok(()),
        }
    }

    /// Decode path, expert half, from a routing decision already on the device, with the
    /// negative controls applied through a host round trip (controls only; never timed).
    fn decode_experts(&self, d: &MoeDecode, layer: usize, y: DevicePtr, out: DevicePtr, t: usize, host: Option<&Routing>, stream: u64) -> Result<()> {
        let control = *self.control.lock().expect("control lock");
        if host.is_some() || control != MoeControl::None {
            let mut routing = match host {
                Some(r) => r.clone(),
                None => d.read_routing(self.gpu, t, stream)?,
            };
            if control == MoeControl::ReverseWeights {
                for token in routing.weights.chunks_exact_mut(TOP_K) {
                    token.reverse();
                }
            }
            let keep = self.arena.packed_keep();
            let next = control == MoeControl::NextSlot;
            self.gpu.synchronize(stream)?;
            d.upload_routing(self.gpu, &routing, t, |id| {
                let s = self.arena.slot_of(layer, id)?;
                Ok(if next && s >= 0 { (s + 1) % keep as i32 } else { s })
            })?;
        }
        d.experts(self.gpu, self.arena.layer(layer)?, self.arena.packed_keep(), &self.matrices, y, out, t, self.swiglu_limit, stream)
    }

    /// Negative controls only.
    pub fn set_control(&self, control: MoeControl) {
        *self.control.lock().expect("control lock") = control;
    }

    /// Select the expert GEMM path (A/B and gating against the reconstruct baseline).
    pub fn set_expert_kernel(&self, kernel: ExpertKernel) {
        *self.kernel.lock().expect("kernel lock") = kernel;
    }

    /// TIMING ONLY; see [`ExpertWork`].
    pub fn set_expert_work(&self, work: ExpertWork) {
        *self.work.lock().expect("work lock") = work;
    }

    fn router(&self, layer: usize) -> Result<&RouterF32> {
        self.routers
            .iter()
            .find(|(index, _)| *index == layer)
            .map(|(_, router)| router)
            .with_context(|| format!("no router loaded for layer {layer}"))
    }

    /// Router scores `sqrt(softplus(y.float() @ gate_w^T))`, `[t, 384]` fp32, on the host.
    pub fn scores(&self, layer: usize, y: DevicePtr, t: usize, stream: u64) -> Result<Vec<f32>> {
        self.router_logits(layer, y, t, stream)?;
        self.gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; t * ROUTER_EXPERTS * 4];
        self.gpu.copy_d2h(self.scratch.logits, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| score_of(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect())
    }

    /// `y.float() @ gate_w^T` into `scratch.logits`, `[t, 384]` fp32. Enqueued, not synced.
    fn router_logits(&self, layer: usize, y: DevicePtr, t: usize, stream: u64) -> Result<()> {
        ensure!(t <= self.scratch.max_t, "pass of {t} tokens exceeds scratch for {}", self.scratch.max_t);
        let router = self.router(layer)?;
        let n = t * self.hidden;
        KernelLaunch::new(self.gpu, self.kernels.bf16_to_f32)
            .block([BLOCK, 1, 1])
            .grid([(n as u32).div_ceil(BLOCK), 1, 1])
            .arg_ptr(y)
            .arg_ptr(self.scratch.y_f32)
            .arg_u64(n as u64)
            .launch(stream)?;
        gemm_act_weight_t_typed(
            self.scratch.y_f32.0,
            self.hidden as u32,
            router.gate_w.0,
            self.scratch.logits.0,
            ROUTER_EXPERTS as u32,
            t as u32,
            ROUTER_EXPERTS as u32,
            self.hidden as u32,
            GemmDtype::F32,
            GemmDtype::F32,
            stream,
        )
    }

    /// The router entirely on the device (`dsv41_route_topk`): only `[t, 6]` indices and
    /// weights come back to the host, for the permutation plan. What `forward` runs.
    pub fn route_device(&self, layer: usize, y: DevicePtr, t: usize, stream: u64) -> Result<Routing> {
        let (bias, bias_vl, resident) = self
            .device_routers
            .iter()
            .find(|(index, ..)| *index == layer)
            .map(|(_, b, v, r)| (*b, *v, *r))
            .with_context(|| format!("no device router for layer {layer}"))?;
        let image: Vec<u8> = {
            let tokens = self.pass_tokens.lock().expect("pass_tokens lock");
            let ids = tokens.as_ref().context(
                "DeepSeek-V4.1 routed MoE: set_pass_tokens was not called for this pass. Image \
                 rows route with gate.bias_vl, and without the token ids they would silently \
                 take the text bias.",
            )?;
            ensure!(ids.len() == t, "pass tokens are {} ids for a {t}-token pass", ids.len());
            image_rows(ids).into_iter().map(u8::from).collect()
        };
        self.router_logits(layer, y, t, stream)?;
        self.gpu.synchronize(stream)?;
        self.gpu.copy_h2d(&image, self.scratch.image)?;
        KernelLaunch::new(self.gpu, self.k_route)
            .block([128, 1, 1])
            .grid([t as u32, 1, 1])
            .arg_ptr(self.scratch.logits)
            .arg_ptr(bias)
            .arg_ptr(bias_vl)
            .arg_ptr(self.scratch.image)
            .arg_ptr(resident)
            .arg_ptr(self.scratch.route_idx)
            .arg_ptr(self.scratch.route_w)
            .arg_u32(TOP_K as u32)
            .arg_f32(self.route_scale)
            .launch(stream)?;
        self.gpu.synchronize(stream)?;
        let mut idx = vec![0u8; t * TOP_K * 4];
        let mut w = vec![0u8; t * TOP_K * 4];
        self.gpu.copy_d2h(self.scratch.route_idx, &mut idx)?;
        self.gpu.copy_d2h(self.scratch.route_w, &mut w)?;
        Ok(Routing {
            indices: idx.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
            weights: w.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            k: TOP_K,
        })
    }

    /// Masked, per-row-biased top-6 over `scores`.
    pub fn route(&self, layer: usize, scores: &[f32], t: usize) -> Result<Routing> {
        let router = self.router(layer)?;
        let tokens = self.pass_tokens.lock().expect("pass_tokens lock");
        let ids = tokens.as_ref().context(
            "DeepSeek-V4.1 routed MoE: set_pass_tokens was not called for this pass. Image rows \
             route with gate.bias_vl, and without the token ids they would silently take the \
             text bias.",
        )?;
        ensure!(ids.len() == t, "pass tokens are {} ids for a {t}-token pass", ids.len());
        let image = image_rows(ids);
        select_experts_multimodal(
            scores,
            &router.bias,
            &router.bias_vl,
            &image,
            self.arena.routing_mask(layer)?,
            t,
            ROUTER_EXPERTS,
            TOP_K,
            self.route_scale,
        )
    }

    /// The expert half: given a routing decision, `y -> out`. Exposed so a microtest can feed
    /// the ENGINE's routing and isolate the GEMM path from the router.
    pub fn forward_routed(
        &self,
        layer: usize,
        y: DevicePtr,
        out: DevicePtr,
        t: usize,
        routing: &Routing,
        stream: u64,
    ) -> Result<()> {
        ensure!(t <= self.scratch.max_t, "pass of {t} tokens exceeds scratch for {}", self.scratch.max_t);
        ensure!(routing.k == TOP_K, "routing k {} is not {TOP_K}", routing.k);
        if let Some(d) = self.decode.as_ref().filter(|_| t <= MAX_DECODE_T) {
            return self.decode_experts(d, layer, y, out, t, Some(routing), stream);
        }
        let control = *self.control.lock().expect("control lock");
        let (hidden, inter, s) = (self.hidden, self.inter, &self.scratch);

        let mut weights = routing.weights.clone();
        if control == MoeControl::ReverseWeights {
            for token in weights.chunks_exact_mut(TOP_K) {
                token.reverse();
            }
        }
        let mut groups = group_by_expert(&routing.indices, &weights, t, TOP_K, |id| {
            self.arena.slot_of(layer, id)
        })?;
        if control == MoeControl::NextSlot {
            let keep = self.arena.packed_keep();
            for group in &mut groups {
                group.slot = (group.slot + 1) % keep;
            }
        }
        let plan = Cb3Permutation::build(&groups, t, TOP_K)?;
        let expanded = plan.total_expanded;
        // The route weight multiplies h, per PERMUTED row, as the engine's up kernel does.
        let mut row_weight = vec![0.0f32; expanded];
        for (flat, row) in plan.token_to_perm.iter().enumerate() {
            row_weight[*row as usize] = plan.weights[flat];
        }

        // The index buffers may still be read by the previous layer's kernels.
        self.gpu.synchronize(stream)?;
        self.gpu.copy_h2d(as_bytes_i32(&plan.sorted_token_ids), s.sorted)?;
        self.gpu.copy_h2d(as_bytes_i32(&plan.token_to_perm), s.tok2perm)?;
        self.gpu.copy_h2d(as_bytes_f32(&row_weight), s.row_w)?;

        let launch = |kernel: KernelHandle| KernelLaunch::new(self.gpu, kernel).block([BLOCK, 1, 1]);
        launch(self.k_permute)
            .grid([expanded as u32, 1, 1])
            .arg_ptr(y)
            .arg_ptr(s.perm)
            .arg_ptr(s.sorted)
            .arg_u32(hidden as u32)
            .arg_u32(expanded as u32)
            .launch(stream)?;

        let residency = self.arena.layer(layer)?;
        let keep = self.arena.packed_keep();
        if *self.kernel.lock().expect("kernel lock") == ExpertKernel::Fused {
            self.fused_experts(residency, &groups, &plan.group_rows, stream)?;
            return launch(self.k_sum)
                .grid([t as u32, 1, 1])
                .arg_ptr(s.down)
                .arg_ptr(out)
                .arg_ptr(s.tok2perm)
                .arg_u32(hidden as u32)
                .arg_u32(t as u32)
                .arg_u32(TOP_K as u32)
                .launch(stream);
        }
        let work = *self.work.lock().expect("work lock");
        let (do_reconstruct, do_gemm) = match work {
            ExpertWork::All => (true, true),
            ExpertWork::ReconstructOnly => (true, false),
            ExpertWork::GemmOnly => (false, true),
            ExpertWork::Skip => (false, false),
        };
        for (group, (begin, end)) in groups.iter().zip(&plan.group_rows) {
            let rows = end - begin;
            if rows == 0 {
                continue;
            }
            if do_reconstruct {
                for (matrix, dst) in self.matrices.iter().zip([s.w1, s.w3, s.w2]) {
                    self.reconstruct.run(residency, *matrix, group.slot, keep, dst, self.gpu, stream)?;
                }
            }
            if !do_gemm {
                continue;
            }
            let at = |base: DevicePtr, width: usize, elem: usize| DevicePtr(base.0 + (begin * width * elem) as u64);
            let (act, gate, up) = (at(s.perm, hidden, 2), at(s.gate, inter, 4), at(s.up, inter, 4));
            let (h, w, down) = (at(s.h, inter, 2), at(s.row_w, 1, 4), at(s.down, hidden, 4));

            gemm_weight_t_f32out(act, s.w1, gate, rows, inter, hidden, stream)?;
            gemm_weight_t_f32out(act, s.w3, up, rows, inter, hidden, stream)?;
            let total = (rows * inter) as u32;
            launch(self.k_swiglu)
                .grid([total.div_ceil(BLOCK), 1, 1])
                .arg_ptr(gate)
                .arg_ptr(up)
                .arg_ptr(w)
                .arg_ptr(h)
                .arg_u32(rows as u32)
                .arg_u32(inter as u32)
                .arg_f32(self.swiglu_limit)
                .launch(stream)?;
            gemm_weight_t_f32out(h, s.w2, down, rows, hidden, inter, stream)?;
        }

        launch(self.k_sum)
            .grid([t as u32, 1, 1])
            .arg_ptr(s.down)
            .arg_ptr(out)
            .arg_ptr(s.tok2perm)
            .arg_u32(hidden as u32)
            .arg_u32(t as u32)
            .arg_u32(TOP_K as u32)
            .launch(stream)
    }

    /// The expert GEMMs with CB3 decoded in shared memory: `perm -> h -> down`.
    fn fused_experts(
        &self,
        residency: &super::cb3_arena::Cb3LayerResidency,
        groups: &[super::moe::ExpertGroup],
        group_rows: &[(usize, usize)],
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let mut tiles: Vec<i32> = Vec::with_capacity(4 * self.max_tiles);
        for (group, (begin, end)) in groups.iter().zip(group_rows) {
            let mut row = *begin;
            while row < *end {
                let rows = (end - row).min(FUSED_TILE_M);
                tiles.extend_from_slice(&[row as i32, rows as i32, group.slot as i32, 0]);
                row += rows;
            }
        }
        let n_tiles = tiles.len() / 4;
        if n_tiles == 0 {
            return Ok(());
        }
        ensure!(n_tiles <= self.max_tiles, "{n_tiles} tiles exceed the {} allocated", self.max_tiles);
        self.gpu.copy_h2d(as_bytes_i32(&tiles), self.tiles)?;

        let [gate, up, down] = &self.matrices;
        let base = |t| residency.plane_base(t);
        let (w1_lo, lo_s) = base(gate.lo);
        let (w1_hi, hi_s) = base(gate.hi);
        let (w1_cb, cb_s) = base(gate.cb);
        let (w1_sc, sc_s) = base(gate.scale);
        let (w3_lo, lo3) = base(up.lo);
        let (w3_hi, hi3) = base(up.hi);
        let (w3_cb, cb3) = base(up.cb);
        let (w3_sc, sc3) = base(up.scale);
        ensure!((lo3, hi3, cb3, sc3) == (lo_s, hi_s, cb_s, sc_s), "w1 and w3 plane strides differ");
        let launch = |kernel: KernelHandle| KernelLaunch::new(self.gpu, kernel).block([256, 1, 1]);
        launch(self.k_fused_gate_up)
            .grid([(gate.rows / FUSED_TILE_N) as u32, n_tiles as u32, 1])
            .arg_ptr(s.perm)
            .arg_ptr(self.tiles)
            .arg_ptr(w1_lo).arg_ptr(w1_hi).arg_ptr(w1_cb).arg_ptr(w1_sc)
            .arg_ptr(w3_lo).arg_ptr(w3_hi).arg_ptr(w3_cb).arg_ptr(w3_sc)
            .arg_u64(lo_s).arg_u64(hi_s).arg_u64(cb_s).arg_u64(sc_s)
            .arg_ptr(s.row_w)
            .arg_ptr(s.h)
            .arg_i32(gate.rows as i32)
            .arg_i32(gate.cols as i32)
            .arg_f32(self.swiglu_limit)
            .launch(stream)?;

        let (w2_lo, lo2) = base(down.lo);
        let (w2_hi, hi2) = base(down.hi);
        let (w2_cb, cb2) = base(down.cb);
        let (w2_sc, sc2) = base(down.scale);
        launch(self.k_fused_down)
            .grid([(down.rows / FUSED_TILE_N) as u32, n_tiles as u32, 1])
            .arg_ptr(s.h)
            .arg_ptr(self.tiles)
            .arg_ptr(w2_lo).arg_ptr(w2_hi).arg_ptr(w2_cb).arg_ptr(w2_sc)
            .arg_u64(lo2).arg_u64(hi2).arg_u64(cb2).arg_u64(sc2)
            .arg_ptr(s.down)
            .arg_i32(down.rows as i32)
            .arg_i32(down.cols as i32)
            .launch(stream)
    }

    /// Release the scratch and the widened router weights.
    pub fn free(self) -> Result<()> {
        let s = &self.scratch;
        for ptr in [
            s.y_f32, s.logits, s.perm, s.gate, s.up, s.h, s.down, s.w1, s.w3, s.w2, s.sorted,
            s.tok2perm, s.row_w, self.tiles, s.image, s.route_idx, s.route_w,
        ] {
            self.gpu.free(ptr)?;
        }
        for (_, bias, bias_vl, resident) in &self.device_routers {
            for ptr in [*bias, *bias_vl, *resident] {
                self.gpu.free(ptr)?;
            }
        }
        for (_, router) in &self.routers {
            self.gpu.free(router.gate_w)?;
        }
        if let Some(d) = self.decode {
            d.free(self.gpu)?;
        }
        Ok(())
    }
}

impl V41RoutedMoe for Cb3RoutedMoe<'_> {
    fn begin_pass(&self, token_ids: &[u32]) -> Result<()> {
        let ids: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();
        self.set_pass_tokens(&ids);
        if let Some(d) = self.decode.as_ref().filter(|_| token_ids.len() <= MAX_DECODE_T) {
            d.set_ids(self.gpu, token_ids)?;
        }
        Ok(())
    }

    fn forward(&self, ops: &Ops, layer: usize, y: DevicePtr, out: DevicePtr, t: usize) -> Result<()> {
        let stream = ops.stream;
        if let Some(d) = self.decode.as_ref().filter(|_| t <= MAX_DECODE_T) {
            let router = self.router(layer)?;
            d.route(self.gpu, layer, router, y, t, self.route_scale, stream)?;
            return self.decode_experts(d, layer, y, out, t, None, stream);
        }
        let routing = self.route_device(layer, y, t, stream)?;
        self.forward_routed(layer, y, out, t, &routing, stream)
    }
}

fn as_bytes_i32(v: &[i32]) -> &[u8] {
    // SAFETY: i32 has no padding and every bit pattern is valid; read-only, lifetime tied to v.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_bytes_f32(v: &[f32]) -> &[u8] {
    // SAFETY: as above, for f32.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
