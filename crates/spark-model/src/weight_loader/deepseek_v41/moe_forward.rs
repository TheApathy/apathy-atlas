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
use super::moe::{
    COMBINE_MODULE, Cb3Matrix, Cb3Permutation, Cb3Reconstruct, MOE_PERMUTE_MODULE,
    PERMUTE_KERNEL, SWIGLU_WEIGHTED_FN, UNPERMUTE_SUM_FN, expert_matrices, gemm_weight_t_f32out,
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
        Ok(Self { gate_w, bias, bias_vl })
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
        let e = max_t * TOP_K;
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
        };
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
        })
    }

    /// Token ids of the NEXT pass, so image rows route with `gate.bias_vl`. Required.
    pub fn set_pass_tokens(&self, token_ids: &[i64]) {
        *self.pass_tokens.lock().expect("pass_tokens lock") = Some(token_ids.to_vec());
    }

    /// Negative controls only.
    pub fn set_control(&self, control: MoeControl) {
        *self.control.lock().expect("control lock") = control;
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
        )?;
        self.gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; t * ROUTER_EXPERTS * 4];
        self.gpu.copy_d2h(self.scratch.logits, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| score_of(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect())
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
        for (group, (begin, end)) in groups.iter().zip(&plan.group_rows) {
            let rows = end - begin;
            if rows == 0 {
                continue;
            }
            for (matrix, dst) in self.matrices.iter().zip([s.w1, s.w3, s.w2]) {
                self.reconstruct.run(residency, *matrix, group.slot, keep, dst, self.gpu, stream)?;
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

    /// Release the scratch and the widened router weights.
    pub fn free(self) -> Result<()> {
        let s = &self.scratch;
        for ptr in [
            s.y_f32, s.logits, s.perm, s.gate, s.up, s.h, s.down, s.w1, s.w3, s.w2, s.sorted,
            s.tok2perm, s.row_w,
        ] {
            self.gpu.free(ptr)?;
        }
        for (_, router) in &self.routers {
            self.gpu.free(router.gate_w)?;
        }
        Ok(())
    }
}

impl V41RoutedMoe for Cb3RoutedMoe<'_> {
    fn begin_pass(&self, token_ids: &[u32]) -> Result<()> {
        let ids: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();
        self.set_pass_tokens(&ids);
        Ok(())
    }

    fn forward(&self, ops: &Ops, layer: usize, y: DevicePtr, out: DevicePtr, t: usize) -> Result<()> {
        let stream = ops.stream;
        let scores = self.scores(layer, y, t, stream)?;
        let routing = self.route(layer, &scores, t)?;
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
