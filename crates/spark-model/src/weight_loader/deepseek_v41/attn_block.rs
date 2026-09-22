// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek-V4.1 attention sub-layer AROUND the sparse core: everything in
//! `engine/model.py::Model.attention` except the compressor, the indexer and the one-pass
//! sparse attention, which are the attention lane's (`seams::Dsv41Attention`).
//!
//! ```text
//! qr  = rmsnorm(wq_a x, q_norm)                    [T, 1280]
//! q   = rope(wq_b qr)                              [T, 64, 512]   (tail 64 dims)
//! kv  = rope(rmsnorm(wkv x, kv_norm))              [T, 512]       -> ring[pos % RING]
//! o   = CORE(q, ring, wpos, ...)                   [T, 64, 512]   (attention lane)
//! o   = inverse_rope(o)
//! out = wo_b(grouped wo_a(o))                      [T, 5120]
//! ```
//!
//! RoPE table per layer: `freqs_c` (YaRN) when `compress_ratio != 0`, else `freqs_w` —
//! i.e. layers 0 and 1 use the plain table. The query, the window kv AND the inverse
//! rotation of the output all use the layer's table at the ABSOLUTE position.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::fwd::Tap;
use super::ops::{Fp8Linear, MM_TILE, Ops, RopeTable, bf16_tensor, bytemuck_i32, f32_tensor, prof};

/// Window ring slots (`engine/model.py` RING). Must exceed window + longest chunk.
pub const RING: usize = 4096;
pub const WINDOW: usize = 128;
pub const N_HEADS: usize = 64;
pub const HEAD_DIM: usize = 512;
pub const ROPE_DIM: usize = 64;
pub const Q_LORA: usize = 1280;
pub const O_GROUPS: usize = 8;
pub const O_LORA: usize = 1024;

/// `compress_ratios` for the 40 main layers.
pub fn compress_ratio(layer: usize) -> usize {
    match layer {
        0 | 1 => 0,
        2..=19 => 2,
        _ => 1,
    }
}

pub struct V41AttnWeights {
    pub layer: usize,
    pub wq_a: Fp8Linear,
    pub q_norm: DevicePtr,
    pub wq_b: Fp8Linear,
    pub wkv: Fp8Linear,
    pub kv_norm: DevicePtr,
    /// `[O_GROUPS * O_LORA, HEAD_DIM * N_HEADS / O_GROUPS]` = [8192, 4096]; group g is rows
    /// g*1024 .. (g+1)*1024.
    pub wo_a: Fp8Linear,
    pub wo_b: Fp8Linear,
    pub attn_sink: DevicePtr,
}

impl V41AttnWeights {
    pub fn load(store: &WeightStore, layer: usize, hidden: usize) -> Result<Self> {
        let p = format!("layers.{layer}.attn");
        let grp_k = HEAD_DIM * N_HEADS / O_GROUPS;
        Ok(Self {
            layer,
            wq_a: Fp8Linear::load(store, &format!("{p}.wq_a"), Q_LORA, hidden)?,
            q_norm: bf16_tensor(store, &format!("{p}.q_norm.weight"), &[Q_LORA])?,
            wq_b: Fp8Linear::load(store, &format!("{p}.wq_b"), N_HEADS * HEAD_DIM, Q_LORA)?,
            wkv: Fp8Linear::load(store, &format!("{p}.wkv"), HEAD_DIM, hidden)?,
            kv_norm: bf16_tensor(store, &format!("{p}.kv_norm.weight"), &[HEAD_DIM])?,
            wo_a: Fp8Linear::load(store, &format!("{p}.wo_a"), O_GROUPS * O_LORA, grp_k)?,
            wo_b: Fp8Linear::load(store, &format!("{p}.wo_b"), hidden, O_GROUPS * O_LORA)?,
            attn_sink: f32_tensor(store, &format!("{p}.attn_sink"), &[N_HEADS])?,
        })
    }

    pub fn largest_weight(&self) -> usize {
        [self.wq_a, self.wq_b, self.wkv, self.wo_a, self.wo_b].iter().map(|w| w.n * w.k).max().unwrap_or(0)
    }
}

/// Scratch for the attention sub-layer, sized for `max_t` tokens.
pub struct AttnScratch {
    pub max_t: usize,
    pub qr: DevicePtr,
    pub q: DevicePtr,
    pub kv: DevicePtr,
    pub o: DevicePtr,
    pub o2: DevicePtr,
    /// `[T]` i32 absolute positions.
    pub pos: DevicePtr,
    /// `[T, WINDOW]` i32 window positions, -1 = none.
    pub wpos: DevicePtr,
    allocations: Vec<DevicePtr>,
}

impl AttnScratch {
    /// The device buffers this scratch allocated (for an owner that frees them on drop).
    pub fn allocations(&self) -> &[DevicePtr] {
        &self.allocations
    }

    pub fn new(gpu: &dyn GpuBackend, max_t: usize) -> Result<Self> {
        let mut allocations = Vec::new();
        let mut a = |bytes: usize| -> Result<DevicePtr> {
            let p = gpu.alloc(bytes.max(256))?;
            allocations.push(p);
            Ok(p)
        };
        let t = super::ops::tiled_rows(max_t);
        let s = Self {
            max_t,
            qr: a(t * Q_LORA * 2)?,
            q: a(t * N_HEADS * HEAD_DIM * 2)?,
            kv: a(t * HEAD_DIM * 2)?,
            o: a(t * N_HEADS * HEAD_DIM * 2)?,
            o2: a(t * O_GROUPS * O_LORA * 2)?,
            pos: a(t * 4)?,
            wpos: a(t * WINDOW * 4)?,
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

/// What the attention lane's side receives for one layer and one pass.
pub struct CoreArgs {
    pub layer: usize,
    /// Post-`attn_norm` input `[T, hidden]` bf16 (the compressor and indexer read it).
    pub x: DevicePtr,
    /// `[T, 1280]` bf16, rmsnorm'd q_lora.
    pub qr: DevicePtr,
    /// `[T, 64, 512]` bf16, RoPE'd.
    pub q: DevicePtr,
    /// This layer's window ring `[RING, 512]` bf16, ALREADY holding this chunk's kv.
    pub ring: DevicePtr,
    /// `[T, 128]` i32 absolute window positions, -1 = none.
    pub wpos: DevicePtr,
    /// First position the ring genuinely holds for this pass (0, or P-128 in the replay).
    pub win_lo: usize,
    pub sink: DevicePtr,
    pub t: usize,
    pub start: usize,
    /// `[T, 64, 512]` bf16 out, BEFORE the inverse RoPE.
    pub out: DevicePtr,
}

/// The attention lane's side of one layer: compress (kv-source) -> index (index-source)
/// -> sparse attention (every layer). Implemented over `seams::Dsv41Attention` in serving;
/// fed from a capture in the driver.
pub trait AttnCore {
    fn run(&self, ops: &Ops, a: &CoreArgs) -> Result<()>;
}

/// `[T, 128]` window positions: `pos - 127 .. pos`, -1 below 0 (`Model._window_positions`).
/// `win_lo` is applied by the kernel as a mask, not here — this matches the reference, which
/// taps `wpos` before masking.
pub fn window_positions(start: usize, t: usize) -> Vec<i32> {
    let mut v = Vec::with_capacity(t * WINDOW);
    for i in 0..t {
        let p = (start + i) as i64;
        for j in 0..WINDOW {
            let q = p - (WINDOW as i64 - 1) + j as i64;
            v.push(if q >= 0 { q as i32 } else { -1 });
        }
    }
    v
}

/// One attention sub-layer. `ring` is this sequence's ring for `w.layer`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    ops: &Ops,
    w: &V41AttnWeights,
    s: &AttnScratch,
    wscratch: DevicePtr,
    rope: &RopeTable,
    ring: DevicePtr,
    x: DevicePtr,
    out: DevicePtr,
    t: usize,
    start: usize,
    win_lo: usize,
    norm_eps: f32,
    core: &dyn AttnCore,
    tap: &Tap,
) -> Result<()> {
    ensure!(t <= s.max_t && t <= RING - WINDOW, "attention: chunk {t} too large");
    ensure!(start + t <= rope.positions, "attention: position {} past the RoPE table", start + t);
    let l = w.layer;
    let gpu = ops.gpu;
    let pos: Vec<i32> = (start..start + t).map(|p| p as i32).collect();
    gpu.copy_h2d_async(bytemuck_i32(&pos), s.pos, ops.stream)?;
    let wpos = window_positions(start, t);
    gpu.copy_h2d_async(bytemuck_i32(&wpos), s.wpos, ops.stream)?;

    ops.linear_fp8_tiled(x, &w.wq_a, wscratch, s.qr, t)?;
    ops.rmsnorm(s.qr, w.q_norm, s.qr, t, Q_LORA, norm_eps)?;
    tap.bf16(ops, "qr", l, s.qr, &[t, Q_LORA])?;
    ops.linear_fp8_tiled(s.qr, &w.wq_b, wscratch, s.q, t)?;
    ops.rope_tail(s.q, s.pos, rope, t, N_HEADS, HEAD_DIM, false)?;
    tap.bf16(ops, "q", l, s.q, &[t, N_HEADS, HEAD_DIM])?;

    ops.linear_fp8_tiled(x, &w.wkv, wscratch, s.kv, t)?;
    ops.rmsnorm(s.kv, w.kv_norm, s.kv, t, HEAD_DIM, norm_eps)?;
    ops.rope_tail(s.kv, s.pos, rope, t, 1, HEAD_DIM, false)?;
    tap.bf16(ops, "kv_new", l, s.kv, &[t, HEAD_DIM])?;
    // ring[pos % RING] = kv — at most two contiguous runs.
    let row = HEAD_DIM * 2;
    let first = start % RING;
    let n1 = t.min(RING - first);
    gpu.copy_d2d_async(s.kv, ring.offset(first * row), n1 * row, ops.stream)?;
    if n1 < t {
        gpu.copy_d2d_async(s.kv.offset(n1 * row), ring, (t - n1) * row, ops.stream)?;
    }

    prof(ops, "attention/core", || {
        core.run(
            ops,
            &CoreArgs { layer: l, x, qr: s.qr, q: s.q, ring, wpos: s.wpos, win_lo, sink: w.attn_sink, t, start, out: s.o },
        )
    })?;
    tap.bf16(ops, "attn_o_pre_inverse_rope", l, s.o, &[t, N_HEADS, HEAD_DIM])?;
    ops.rope_tail(s.o, s.pos, rope, t, N_HEADS, HEAD_DIM, true)?;
    tap.bf16(ops, "attn_o_post_inverse_rope", l, s.o, &[t, N_HEADS, HEAD_DIM])?;

    // grouped wo_a: o [T, 8, 4096] -> o2 [T, 8, 1024], group g uses wo_a rows g*1024..
    let grp_k = HEAD_DIM * N_HEADS / O_GROUPS;
    if t == 1 && ops.k.fp8_gemv_m1.is_some() {
        // One grouped fp8 GEMV: output row n reads activation group n / 1024.
        ops.fp8_gemv_m1(s.o, &w.wo_a, s.o2, O_LORA, grp_k)?;
        return ops.linear_fp8_tiled(s.o2, &w.wo_b, wscratch, out, t);
    }
    if t <= super::ops::GEMV_MAX_M && ops.k.fp8_gemv_m8.is_some() {
        ops.fp8_gemv_rows(s.o, N_HEADS * HEAD_DIM, &w.wo_a, s.o2, O_GROUPS * O_LORA, t, O_LORA, grp_k)?;
        return ops.linear_fp8_tiled(s.o2, &w.wo_b, wscratch, out, t);
    }
    // `v41_ref.wo_a_proj` runs the grouped fp8 kernel (row-invariant, not row-tiled), so the
    // same row policy as `linear_fp8_tiled`: one GEMM per group at M > 16, one tile at M <= 16.
    prof(ops, "dense/dequant", || ops.dequant(&w.wo_a, wscratch))?;
    let grouped = |x: DevicePtr, lda: usize, wt: DevicePtr, o: DevicePtr, ldc: usize, m: usize, n: usize, k: usize| {
        if m > MM_TILE && !super::ops::fp8_force_rowtile() {
            ops.linear_bf16_policy(x, lda, wt, o, ldc, m, n, k)
        } else { ops.linear_bf16_tiled(x, lda, wt, o, ldc, m, n, k) }
    };
    for g in 0..O_GROUPS {
        grouped(
            s.o.offset(g * grp_k * 2),
            N_HEADS * HEAD_DIM,
            wscratch.offset(g * O_LORA * grp_k * 2),
            s.o2.offset(g * O_LORA * 2),
            O_GROUPS * O_LORA,
            t,
            O_LORA,
            grp_k,
        )?;
    }
    ops.linear_fp8_tiled(s.o2, &w.wo_b, wscratch, out, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Window positions: the newest slot is the query itself; below 0 is -1.
    #[test]
    fn window_positions_match_the_reference_shape() {
        let w = window_positions(0, 2);
        assert_eq!(w.len(), 2 * WINDOW);
        assert_eq!(w[WINDOW - 1], 0, "token 0 sees itself in the last slot");
        assert!(w[..WINDOW - 1].iter().all(|&p| p == -1));
        let w = window_positions(1000, 1);
        assert_eq!(w[0], 1000 - 127);
        assert_eq!(w[WINDOW - 1], 1000);
    }

    #[test]
    fn ratios_follow_the_config() {
        assert_eq!(compress_ratio(0), 0);
        assert_eq!(compress_ratio(1), 0);
        assert_eq!(compress_ratio(2), 2);
        assert_eq!(compress_ratio(19), 2);
        assert_eq!(compress_ratio(20), 1);
        assert_eq!(compress_ratio(39), 1);
    }
}
