// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek-V4.1 block forward — the part no lane owns: the hyper-connection (mHC)
//! residual stream, the norms, the engram injection, the shared expert, and the combine.
//!
//! Mirrors `engine/model.py::Model.block` line for line. The one thing to read twice is the
//! **shifted `pre`**: the attention sub-layer collapses the stream with the `pre_mix` that
//! the PREVIOUS block's FFN side produced, and the FFN sub-layer collapses with THIS block's
//! attention-side `pre`. `hc_mixes` is computed on the stream as it stands before each
//! sub-layer, but its `pre` is consumed one sub-layer later:
//!
//! ```text
//! attn_pre, attn_post, attn_comb = hc_mixes(h, hc_attn)
//! h = hc_post(attn(rmsnorm(hc_pre(h, pre_mix))),  h, attn_post, attn_comb)
//! ffn_pre,  ffn_post,  ffn_comb  = hc_mixes(h, hc_ffn)
//! h = hc_post(moe(rmsnorm(hc_pre(h, attn_pre))),  h, ffn_post,  ffn_comb)
//! pre_mix = ffn_pre
//! ```
//!
//! Using each sub-layer's own `pre` — the V4-0731 wiring in `qwen3_attention` — runs, and
//! produces a different model with no error. So the V4 hc kernels are NOT reused here.

use anyhow::{Context, Result, ensure};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::ops::{Fp8Linear, HcParams, Ops, bf16_tensor, prof};

/// Model constants the block needs. Read from `ModelConfig` by the caller; kept as a plain
/// struct so the forward cannot silently read a field that means something else for V4.
#[derive(Clone, Copy, Debug)]
pub struct V41Dims {
    pub hidden: usize,
    pub hc: usize,
    pub moe_inter: usize,
    pub norm_eps: f32,
    pub hc_eps: f32,
    pub sinkhorn_iters: u32,
    pub swiglu_limit: f32,
}

impl V41Dims {
    pub fn from_config(c: &atlas_core::config::ModelConfig) -> Result<Self> {
        let d = Self {
            hidden: c.hidden_size,
            hc: 4,
            moe_inter: c.moe_intermediate_size,
            norm_eps: c.rms_norm_eps as f32,
            hc_eps: 1e-6,
            sinkhorn_iters: 20,
            swiglu_limit: 10.0,
        };
        ensure!(d.hidden == 5120 && d.moe_inter == 2304, "unexpected V4.1 dims {d:?}");
        Ok(d)
    }
}

/// The FP8 always-on shared expert (`ffn.shared_experts.w{1,2,3}`, block-32 UE8M0).
pub struct SharedExpert {
    pub w1: Fp8Linear,
    pub w2: Fp8Linear,
    pub w3: Fp8Linear,
}

impl SharedExpert {
    pub fn load(store: &WeightStore, layer: usize, dims: &V41Dims) -> Result<Self> {
        let p = format!("layers.{layer}.ffn.shared_experts");
        Ok(Self {
            w1: Fp8Linear::load(store, &format!("{p}.w1"), dims.moe_inter, dims.hidden)?,
            w2: Fp8Linear::load(store, &format!("{p}.w2"), dims.hidden, dims.moe_inter)?,
            w3: Fp8Linear::load(store, &format!("{p}.w3"), dims.moe_inter, dims.hidden)?,
        })
    }

    /// `v41_ref.expert_ffn(y, w1, w2, w3, limit)`: out = w2(bf16(silu(clamp(w1 y)) * clamp(w3 y))).
    pub fn forward(&self, ops: &Ops, y: DevicePtr, out: DevicePtr, t: usize, s: &PassScratch, dims: &V41Dims) -> Result<()> {
        let n = t * dims.moe_inter;
        ops.linear_fp8_tiled(y, &self.w1, s.wscratch, s.gate, t)?;
        ops.linear_fp8_tiled(y, &self.w3, s.wscratch, s.up, t)?;
        ops.swiglu(s.gate, s.up, s.act, n, dims.swiglu_limit)?;
        ops.linear_fp8_tiled(s.act, &self.w2, s.wscratch, out, t)
    }
}

/// The engram projection of layers 1 and 14 (`layers.N.engram.{wkv, q_weight, k_weight}`).
/// The ~95 GB table itself is gathered by the engram lane; this is what consumes the rows.
pub struct EngramProj {
    pub wkv: Fp8Linear,
    /// `q_weight.float() * k_weight.float()`, fp32 `[hc, hidden]`, formed once at load.
    pub weight: DevicePtr,
}

pub const ENGRAM_ROW_WIDTH: usize = 24 * 256;

impl EngramProj {
    pub fn load(store: &WeightStore, layer: usize, dims: &V41Dims, ops: &Ops) -> Result<Self> {
        let p = format!("layers.{layer}.engram");
        let wkv = Fp8Linear::load(store, &format!("{p}.wkv"), (dims.hc + 1) * dims.hidden, ENGRAM_ROW_WIDTH)?;
        let n = dims.hc * dims.hidden;
        let q = bf16_tensor(store, &format!("{p}.q_weight"), &[dims.hc, dims.hidden])?;
        let k = bf16_tensor(store, &format!("{p}.k_weight"), &[dims.hc, dims.hidden])?;
        let weight = ops.gpu.alloc(n * 4)?;
        ops.mul_bf16_to_f32(q, k, weight, n)?;
        ops.gpu.synchronize(ops.stream)?;
        Ok(Self { wkv, weight })
    }

    /// `v41_ref.engram_forward(h, rows)` with the dead-head mask applied first, in place on `h`.
    /// `rows` is f32 `[t, 24, 256]` (PRE-mask, as the gather seam returns it); `dead` is u8
    /// `[t, 24]` or NULL for "no dead heads".
    pub fn forward(&self, ops: &Ops, h: DevicePtr, rows: DevicePtr, dead: DevicePtr, t: usize, s: &PassScratch, dims: &V41Dims) -> Result<()> {
        ops.engram_rows_bf16(rows, dead, s.engram_rows_bf16, t * ENGRAM_ROW_WIDTH)?;
        ops.linear_fp8_tiled(s.engram_rows_bf16, &self.wkv, s.wscratch, s.engram_kv, t)?;
        ops.engram_gate(h, s.engram_kv, self.weight, t, dims.hc, dims.hidden, dims.norm_eps)
    }
}

/// Per-layer weights of the glue this module owns.
pub struct V41BlockWeights {
    pub layer: usize,
    pub attn_norm: DevicePtr,
    pub ffn_norm: DevicePtr,
    pub hc_attn: HcParams,
    pub hc_ffn: HcParams,
    pub shared: SharedExpert,
    pub engram: Option<EngramProj>,
}

impl V41BlockWeights {
    pub fn load(store: &WeightStore, layer: usize, dims: &V41Dims, ops: &Ops) -> Result<Self> {
        let lp = format!("layers.{layer}");
        Ok(Self {
            layer,
            attn_norm: bf16_tensor(store, &format!("{lp}.attn_norm.weight"), &[dims.hidden])?,
            ffn_norm: bf16_tensor(store, &format!("{lp}.ffn_norm.weight"), &[dims.hidden])?,
            hc_attn: HcParams::load(store, &format!("{lp}.hc_attn"), dims.hc, dims.hidden)?,
            hc_ffn: HcParams::load(store, &format!("{lp}.hc_ffn"), dims.hc, dims.hidden)?,
            shared: SharedExpert::load(store, layer, dims)?,
            engram: if super::seams::ENGRAM_LAYERS.contains(&layer) {
                Some(EngramProj::load(store, layer, dims, ops)?)
            } else {
                None
            },
        })
    }
}

/// Scratch for one pass of up to `max_t` tokens. Allocated once, reused by every layer.
pub struct PassScratch {
    pub max_t: usize,
    /// The mHC stream `[T, hc, hidden]` bf16 — the model's residual. Persistent across layers.
    pub h: DevicePtr,
    /// `[T, hc]` f32 — the `pre` the next attention sub-layer collapses with.
    pub pre_mix: DevicePtr,
    pub attn_pre: DevicePtr,
    pub attn_post: DevicePtr,
    pub attn_comb: DevicePtr,
    pub ffn_pre: DevicePtr,
    pub ffn_post: DevicePtr,
    pub ffn_comb: DevicePtr,
    /// `[T, hidden]` bf16: the collapsed + normed sub-layer input.
    pub x: DevicePtr,
    /// `[T, hidden]` bf16: a sub-layer's output.
    pub y: DevicePtr,
    /// `[T, hidden]` bf16: the routed and shared expert outputs.
    pub routed: DevicePtr,
    pub shared: DevicePtr,
    /// `[T, moe_inter]` bf16.
    pub gate: DevicePtr,
    pub up: DevicePtr,
    pub act: DevicePtr,
    /// `[T, 24, 256]` f32 rows from the gather, and their bf16 `[T, 6144]` form.
    pub engram_rows: DevicePtr,
    pub engram_rows_bf16: DevicePtr,
    /// `[T, (hc+1) * hidden]` bf16.
    pub engram_kv: DevicePtr,
    /// `[T, 24]` u8 dead-head mask.
    pub engram_dead: DevicePtr,
    /// Transient bf16 copy of one FP8 weight; sized for the largest (engram wkv).
    pub wscratch: DevicePtr,
    pub wscratch_bytes: usize,
    allocations: Vec<DevicePtr>,
}

impl PassScratch {
    /// The device buffers this scratch allocated (for an owner that frees them on drop).
    pub fn allocations(&self) -> &[DevicePtr] {
        &self.allocations
    }

    pub fn new(gpu: &dyn GpuBackend, dims: &V41Dims, max_t: usize, largest_fp8_weight: usize) -> Result<Self> {
        let mut allocations = Vec::new();
        let mut a = |bytes: usize| -> Result<DevicePtr> {
            let p = gpu.alloc(bytes.max(256))?;
            allocations.push(p);
            Ok(p)
        };
        let (d, hc) = (dims.hidden, dims.hc);
        // Every activation buffer holds whole MM_TILE tiles (see ops::tiled_rows).
        let t = super::ops::tiled_rows(max_t);
        let s = Self {
            max_t,
            h: a(t * hc * d * 2)?,
            pre_mix: a(t * hc * 4)?,
            attn_pre: a(t * hc * 4)?,
            attn_post: a(t * hc * 4)?,
            attn_comb: a(t * hc * hc * 4)?,
            ffn_pre: a(t * hc * 4)?,
            ffn_post: a(t * hc * 4)?,
            ffn_comb: a(t * hc * hc * 4)?,
            x: a(t * d * 2)?,
            y: a(t * d * 2)?,
            routed: a(t * d * 2)?,
            shared: a(t * d * 2)?,
            gate: a(t * dims.moe_inter * 2)?,
            up: a(t * dims.moe_inter * 2)?,
            act: a(t * dims.moe_inter * 2)?,
            engram_rows: a(t * ENGRAM_ROW_WIDTH * 4)?,
            engram_rows_bf16: a(t * ENGRAM_ROW_WIDTH * 2)?,
            engram_kv: a(t * (hc + 1) * d * 2)?,
            engram_dead: a(t * 24)?,
            wscratch: a(largest_fp8_weight * 2)?,
            wscratch_bytes: largest_fp8_weight * 2,
            allocations: Vec::new(),
        };
        Ok(Self { allocations, ..s })
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in self.allocations {
            gpu.free(p)?;
        }
        Ok(())
    }
}

/// One sub-layer's worth of attention: post-`attn_norm` x `[T, hidden]` -> `[T, hidden]`.
/// Everything from `wq_a` to `wo_b`, including the ring/compressor/indexer state updates.
pub trait V41AttentionBlock {
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, ops: &Ops, layer: usize, x: DevicePtr, out: DevicePtr, t: usize, start: usize) -> Result<()>;
}

/// The routed experts of one layer: post-`ffn_norm` y `[T, hidden]` -> routed sum `[T, hidden]`
/// bf16 (router, residency mask, top-k, CB3 experts, weighted combine). Shared expert NOT
/// included — that is [`SharedExpert`], here.
pub trait V41RoutedMoe {
    /// The token ids of the rows the NEXT pass carries (chunk ids, the replay tail's ids, or
    /// the decode token), for routing that depends on them (image rows use `gate.bias_vl`).
    fn begin_pass(&self, _token_ids: &[u32]) -> Result<()> {
        Ok(())
    }
    fn forward(&self, ops: &Ops, layer: usize, y: DevicePtr, out: DevicePtr, t: usize) -> Result<()>;
}

/// A deliberately WRONG wiring, for negative controls only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockControl {
    /// The reference wiring.
    None,
    /// Each sub-layer collapses with its OWN `pre` (the V4-0731 wiring). Runs, finite, and
    /// wrong: a gate on `h` must reject it.
    OwnPre,
}

/// `Model.block`: one layer, in place on `s.h` / `s.pre_mix`.
#[allow(clippy::too_many_arguments)]
pub fn block(
    ops: &Ops,
    w: &V41BlockWeights,
    dims: &V41Dims,
    s: &PassScratch,
    t: usize,
    start: usize,
    attn: &dyn V41AttentionBlock,
    moe: &dyn V41RoutedMoe,
    tap: &Tap,
    control: BlockControl,
) -> Result<()> {
    ensure!(t <= s.max_t, "pass of {t} tokens exceeds scratch sized for {}", s.max_t);
    let (d, hc, l) = (dims.hidden, dims.hc, w.layer);
    let (eps, it, hce) = (dims.norm_eps, dims.sinkhorn_iters, dims.hc_eps);

    // ---- attention sub-layer
    let attn_side_pre = match control {
        BlockControl::None => s.pre_mix,
        BlockControl::OwnPre => s.attn_pre,
    };
    prof(ops, "mhc+norm", || {
        ops.hc_mixes(s.h, &w.hc_attn, s.attn_pre, s.attn_post, s.attn_comb, t, d, it, eps, hce)?;
        ops.hc_pre(s.h, attn_side_pre, s.x, t, d)?;
        ops.rmsnorm(s.x, w.attn_norm, s.x, t, d, eps)
    })?;
    tap.bf16(ops, "attn_x", l, s.x, &[t, d])?;
    prof(ops, "attention", || attn.forward(ops, l, s.x, s.y, t, start))?;
    tap.bf16(ops, "attn_out", l, s.y, &[t, d])?;

    // ---- FFN sub-layer (collapses with THIS block's attention-side pre)
    let ffn_side_pre = match control {
        BlockControl::None => s.attn_pre,
        BlockControl::OwnPre => s.ffn_pre,
    };
    prof(ops, "mhc+norm", || {
        ops.hc_post(s.y, s.h, s.attn_post, s.attn_comb, s.h, t, d)?;
        ops.hc_mixes(s.h, &w.hc_ffn, s.ffn_pre, s.ffn_post, s.ffn_comb, t, d, it, eps, hce)?;
        ops.hc_pre(s.h, ffn_side_pre, s.x, t, d)?;
        ops.rmsnorm(s.x, w.ffn_norm, s.x, t, d, eps)
    })?;
    tap.bf16(ops, "moe_in", l, s.x, &[t, d])?;
    prof(ops, "moe.routed", || moe.forward(ops, l, s.x, s.routed, t))?;
    prof(ops, "moe.shared", || w.shared.forward(ops, s.x, s.shared, t, s, dims))?;
    tap.bf16(ops, "moe_routed", l, s.routed, &[t, d])?;
    tap.bf16(ops, "moe_shared", l, s.shared, &[t, d])?;
    prof(ops, "mhc+norm", || {
        ops.add_bf16(s.routed, s.shared, s.y, t * d)?;
        ops.hc_post(s.y, s.h, s.ffn_post, s.ffn_comb, s.h, t, d)?;
        ops.gpu.copy_d2d_async(s.ffn_pre, s.pre_mix, t * hc * 4, ops.stream)
    })?;

    tap.bf16(ops, "h", l, s.h, &[t, hc, d])?;
    tap.f32(ops, "pre_mix", l, s.pre_mix, &[t, hc])?;
    Ok(())
}

/// Optional per-layer dumps in the oracle's file convention (`L02.attn_out.000.bin`, raw
/// little-endian, bf16 as its u16 bits). Off unless a directory is given; when on, every
/// tap SYNCHRONIZES the stream, so never time a tapped run.
pub struct Tap {
    dir: Option<PathBuf>,
    counts: Mutex<HashMap<String, usize>>,
    /// Layers to dump (empty = all).
    layers: Vec<usize>,
}

impl Tap {
    pub fn off() -> Self {
        Self { dir: None, counts: Mutex::new(HashMap::new()), layers: Vec::new() }
    }

    pub fn to_dir(dir: PathBuf, layers: Vec<usize>) -> Result<Self> {
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        Ok(Self { dir: Some(dir), counts: Mutex::new(HashMap::new()), layers })
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    fn write(&self, ops: &Ops, name: &str, layer: usize, ptr: DevicePtr, bytes: usize) -> Result<()> {
        let Some(dir) = &self.dir else { return Ok(()) };
        if !self.layers.is_empty() && !self.layers.contains(&layer) {
            return Ok(());
        }
        // ATLAS_DSV41_TAP_LAYERS=0,1,40 restricts the dump to those layers.
        if let Ok(only) = std::env::var("ATLAS_DSV41_TAP_LAYERS")
            && !only.split(',').any(|l| l.parse::<usize>().ok() == Some(layer))
        {
            return Ok(());
        }
        // ATLAS_DSV41_TAP_NAMES=h,logits_last restricts the dump to those tap names.
        if let Ok(only) = std::env::var("ATLAS_DSV41_TAP_NAMES")
            && !only.split(',').any(|n| n == name)
        {
            return Ok(());
        }
        let key = format!("L{layer:02}.{name}");
        let occ = {
            let mut c = self.counts.lock().expect("tap counts poisoned");
            let e = c.entry(key.clone()).or_insert(0);
            let o = *e;
            *e += 1;
            o
        };
        ops.gpu.synchronize(ops.stream)?;
        let mut host = vec![0u8; bytes];
        ops.gpu.copy_d2h(ptr, &mut host)?;
        let path = dir.join(format!("{key}.{occ:03}.bin"));
        std::fs::write(&path, host).with_context(|| format!("write {}", path.display()))
    }

    pub fn bf16(&self, ops: &Ops, name: &str, layer: usize, ptr: DevicePtr, shape: &[usize]) -> Result<()> {
        self.write(ops, name, layer, ptr, shape.iter().product::<usize>() * 2)
    }

    pub fn f32(&self, ops: &Ops, name: &str, layer: usize, ptr: DevicePtr, shape: &[usize]) -> Result<()> {
        self.write(ops, name, layer, ptr, shape.iter().product::<usize>() * 4)
    }

    pub fn bytes(&self, ops: &Ops, name: &str, layer: usize, ptr: DevicePtr, bytes: usize) -> Result<()> {
        self.write(ops, name, layer, ptr, bytes)
    }
}

/// `Model.forward`'s tail for the LAST row of the pass: `x = hc_pre(h, pre_mix)`,
/// `rmsnorm(x, norm)`, `logits = head(x)` — bf16 GEMM with fp32 accumulate and bf16 logits,
/// exactly `R.head_logits` for a bf16 head. `logits` must hold `[MM_TILE, vocab]` bf16 (the
/// GEMM runs as one 16-row tile like `R.mm`); row 0 is the result.
///
/// Rows other than the last are never needed at prefill; computing them would be a
/// `[T, 129280]` GEMM for nothing.
#[allow(clippy::too_many_arguments)]
pub fn final_logits_last_row(
    ops: &Ops,
    dims: &V41Dims,
    s: &PassScratch,
    t: usize,
    norm: DevicePtr,
    head: DevicePtr,
    vocab: usize,
    logits: DevicePtr,
) -> Result<()> {
    ensure!(t >= 1 && t <= s.max_t, "final_logits: bad t {t}");
    let (d, hc) = (dims.hidden, dims.hc);
    let h_last = s.h.offset((t - 1) * hc * d * 2);
    let pre_last = s.pre_mix.offset((t - 1) * hc * 4);
    ops.hc_pre(h_last, pre_last, s.x, 1, d)?;
    ops.rmsnorm(s.x, norm, s.x, 1, d, dims.norm_eps)?;
    ops.linear_bf16_tiled(s.x, d, head, logits, vocab, 1, vocab, d)
}
