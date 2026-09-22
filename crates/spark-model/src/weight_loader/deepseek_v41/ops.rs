// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 forward primitives: typed launches of `cb3/dsv41_fwd.cu`, the FP8 dense
//! linear, and the two RoPE tables.
//!
//! Numerics follow the Python reference with `DSV41_DENSE_FP4=off` — the CHECKPOINT's
//! numerics. An FP8 block-32 weight is dequantized to bf16 into a transient scratch and fed
//! to a bf16 cuBLASLt GEMM with fp32 accumulate: exactly `F.linear(x.bf16(), w.dequant())`,
//! the path `v41_ref.dense` takes at M > 16. (The production Python engine re-quantizes the
//! attention projections to FP4 at load; that is a lossy speed choice of that engine and is
//! NOT reproduced here — see DSV41_PORT/integrate.)

use anyhow::{Context, Result, ensure};
use spark_runtime::cublaslt::{GemmDtype, gemm_act_weight_t_typed_ex};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{WeightDtype, WeightStore};

/// Kernel module compiled from `kernels/gb10/deepseek-v4.1/cb3/dsv41_fwd.cu`.
pub const FWD_MODULE: &str = "dsv41_fwd";
const BLOCK: u32 = 256;
/// Row tile of every activation GEMM (`engine/model.py` MM_TILE / `v41_ref.mm`): the reference
/// issues each GEMM on fixed 16-row tiles so a row's result does not depend on how many rows
/// share the call. Without it chunking changes the model (dsv41-attention measured L20's
/// replay-tail top-k moving on 19/128 rows under a re-chunking).
pub const MM_TILE: usize = 16;
/// Every DeepSeek-V4.1 GEMM forbids split-K (see `gemm_act_weight_t_typed_ex`): the
/// reference's torch GEMMs do not split K at these shapes, and split-K changes top-k.
const NO_SPLIT_K: bool = true;

/// Rows a buffer must hold for `t` logical rows to go through the tiled GEMMs: every tile is
/// issued at exactly MM_TILE rows, so the last tile reads and writes up to MM_TILE-1 slack rows.
pub fn tiled_rows(t: usize) -> usize {
    t.div_ceil(MM_TILE) * MM_TILE
}

/// Every kernel handle of `dsv41_fwd`, resolved once.
#[derive(Clone, Copy, Debug)]
pub struct Dsv41Kernels {
    pub dequant: KernelHandle,
    pub bf16_to_f32: KernelHandle,
    pub f32_to_bf16: KernelHandle,
    pub add_bf16: KernelHandle,
    pub embed: KernelHandle,
    pub hc_expand: KernelHandle,
    pub rmsnorm: KernelHandle,
    pub hc_mixes: KernelHandle,
    pub hc_pre: KernelHandle,
    pub hc_post: KernelHandle,
    pub swiglu: KernelHandle,
    pub rope_tail: KernelHandle,
    pub engram_rows_bf16: KernelHandle,
    pub engram_gate: KernelHandle,
    pub mul_bf16_to_f32: KernelHandle,
}

impl Dsv41Kernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: &str| {
            gpu.kernel(FWD_MODULE, name).with_context(|| {
                format!(
                    "{FWD_MODULE}::{name} is not in the compiled PTX. It is built only for the \
                     (gb10, deepseek-v4.1, cb3) target; a NEW .cu file may also need \
                     `touch crates/atlas-kernels/build.rs` before cargo notices it."
                )
            })
        };
        Ok(Self {
            dequant: k("dsv41_dequant_fp8_ue8m0")?,
            bf16_to_f32: k("dsv41_bf16_to_f32")?,
            f32_to_bf16: k("dsv41_f32_to_bf16")?,
            add_bf16: k("dsv41_add_bf16")?,
            embed: k("dsv41_embed")?,
            hc_expand: k("dsv41_hc_expand")?,
            rmsnorm: k("dsv41_rmsnorm")?,
            hc_mixes: k("dsv41_hc_mixes")?,
            hc_pre: k("dsv41_hc_pre")?,
            hc_post: k("dsv41_hc_post")?,
            swiglu: k("dsv41_swiglu")?,
            rope_tail: k("dsv41_rope_tail")?,
            engram_rows_bf16: k("dsv41_engram_rows_bf16")?,
            engram_gate: k("dsv41_engram_gate")?,
            mul_bf16_to_f32: k("dsv41_mul_bf16_to_f32")?,
        })
    }
}

fn grid_1d(n: usize) -> Result<u32> {
    u32::try_from(n.div_ceil(BLOCK as usize)).context("1-D grid overflow")
}

/// Launchers. Every extent is explicit; nothing is inferred from a pointer.
pub struct Ops<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub k: &'a Dsv41Kernels,
    pub stream: u64,
}

impl Ops<'_> {
    fn l(&self, kernel: KernelHandle) -> KernelLaunch<'_> {
        KernelLaunch::new(self.gpu, kernel).block([BLOCK, 1, 1])
    }

    pub fn bf16_to_f32(&self, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        self.l(self.k.bf16_to_f32).grid([grid_1d(n)?, 1, 1]).arg_ptr(x).arg_ptr(y).arg_u64(n as u64).launch(self.stream)
    }

    pub fn f32_to_bf16(&self, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        self.l(self.k.f32_to_bf16).grid([grid_1d(n)?, 1, 1]).arg_ptr(x).arg_ptr(y).arg_u64(n as u64).launch(self.stream)
    }

    pub fn add_bf16(&self, a: DevicePtr, b: DevicePtr, out: DevicePtr, n: usize) -> Result<()> {
        self.l(self.k.add_bf16).grid([grid_1d(n)?, 1, 1]).arg_ptr(a).arg_ptr(b).arg_ptr(out).arg_u64(n as u64).launch(self.stream)
    }

    pub fn embed(&self, table: DevicePtr, ids: DevicePtr, out: DevicePtr, t: usize, d: usize) -> Result<()> {
        self.l(self.k.embed).grid([t as u32, 1, 1]).arg_ptr(table).arg_ptr(ids).arg_ptr(out).arg_u32(d as u32).launch(self.stream)
    }

    pub fn hc_expand(&self, x: DevicePtr, h: DevicePtr, pre_mix: DevicePtr, t: usize, d: usize) -> Result<()> {
        self.l(self.k.hc_expand).grid([t as u32, 1, 1]).arg_ptr(x).arg_ptr(h).arg_ptr(pre_mix).arg_u32(d as u32).launch(self.stream)
    }

    /// `rows` rows of width `n`, bf16 in and out (may alias).
    pub fn rmsnorm(&self, x: DevicePtr, w: DevicePtr, y: DevicePtr, rows: usize, n: usize, eps: f32) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        self.l(self.k.rmsnorm).grid([rows as u32, 1, 1]).arg_ptr(x).arg_ptr(w).arg_ptr(y).arg_u32(n as u32).arg_f32(eps).launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_mixes(&self, h: DevicePtr, hc: &HcParams, pre: DevicePtr, post: DevicePtr, comb: DevicePtr, t: usize, d: usize, iters: u32, eps: f32, hc_eps: f32) -> Result<()> {
        self.l(self.k.hc_mixes)
            .grid([t as u32, 1, 1])
            .arg_ptr(h).arg_ptr(hc.func).arg_ptr(hc.scale).arg_ptr(hc.base)
            .arg_ptr(pre).arg_ptr(post).arg_ptr(comb)
            .arg_u32(d as u32).arg_u32(iters).arg_f32(eps).arg_f32(hc_eps)
            .launch(self.stream)
    }

    pub fn hc_pre(&self, h: DevicePtr, pre: DevicePtr, y: DevicePtr, t: usize, d: usize) -> Result<()> {
        self.l(self.k.hc_pre).grid([t as u32, 1, 1]).arg_ptr(h).arg_ptr(pre).arg_ptr(y).arg_u32(d as u32).launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_post(&self, y: DevicePtr, res: DevicePtr, post: DevicePtr, comb: DevicePtr, out: DevicePtr, t: usize, d: usize) -> Result<()> {
        self.l(self.k.hc_post).grid([t as u32, 1, 1]).arg_ptr(y).arg_ptr(res).arg_ptr(post).arg_ptr(comb).arg_ptr(out).arg_u32(d as u32).launch(self.stream)
    }

    pub fn swiglu(&self, gate: DevicePtr, up: DevicePtr, out: DevicePtr, n: usize, limit: f32) -> Result<()> {
        self.l(self.k.swiglu).grid([grid_1d(n)?, 1, 1]).arg_ptr(gate).arg_ptr(up).arg_ptr(out).arg_u64(n as u64).arg_f32(limit).launch(self.stream)
    }

    /// Rotate the last `2 * table.half` dims of each `[heads, head_dim]` row of `x` in place.
    /// `pos` is a device `[rows]` i32 array of absolute positions into `table`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_tail(&self, x: DevicePtr, pos: DevicePtr, table: &RopeTable, rows: usize, heads: usize, head_dim: usize, inverse: bool) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        KernelLaunch::new(self.gpu, self.k.rope_tail)
            .grid([rows as u32, heads as u32, 1])
            .block([table.half as u32, 1, 1])
            .arg_ptr(x).arg_ptr(pos).arg_ptr(table.cos).arg_ptr(table.sin)
            .arg_u32(heads as u32).arg_u32(head_dim as u32).arg_u32(table.half as u32)
            .arg_i32(inverse as i32)
            .launch(self.stream)
    }

    pub fn engram_rows_bf16(&self, rows: DevicePtr, dead: DevicePtr, out: DevicePtr, n: usize) -> Result<()> {
        self.l(self.k.engram_rows_bf16).grid([grid_1d(n)?, 1, 1]).arg_ptr(rows).arg_ptr(dead).arg_ptr(out).arg_u64(n as u64).launch(self.stream)
    }

    pub fn engram_gate(&self, h: DevicePtr, kv: DevicePtr, weight: DevicePtr, t: usize, hc: usize, d: usize, eps: f32) -> Result<()> {
        self.l(self.k.engram_gate).grid([t as u32, hc as u32, 1]).arg_ptr(h).arg_ptr(kv).arg_ptr(weight).arg_u32(d as u32).arg_f32(eps).launch(self.stream)
    }

    pub fn mul_bf16_to_f32(&self, a: DevicePtr, b: DevicePtr, out: DevicePtr, n: usize) -> Result<()> {
        self.l(self.k.mul_bf16_to_f32).grid([grid_1d(n)?, 1, 1]).arg_ptr(a).arg_ptr(b).arg_ptr(out).arg_u64(n as u64).launch(self.stream)
    }

    /// `out[m, n] = x[m, k] @ w[n, k]^T`, bf16, fp32 accumulate.
    pub fn linear_bf16(&self, x: DevicePtr, w: DevicePtr, out: DevicePtr, m: usize, n: usize, k: usize) -> Result<()> {
        gemm_act_weight_t_typed_ex(x.0, k as u32, w.0, out.0, n as u32, m as u32, n as u32, k as u32, GemmDtype::Bf16, GemmDtype::Bf16, NO_SPLIT_K, self.stream)
    }

    /// Strided bf16 GEMM, for the grouped `wo_a` (column slices of wider row-major matrices).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16_strided(&self, x: DevicePtr, lda: usize, w: DevicePtr, out: DevicePtr, ldc: usize, m: usize, n: usize, k: usize) -> Result<()> {
        gemm_act_weight_t_typed_ex(x.0, lda as u32, w.0, out.0, ldc as u32, m as u32, n as u32, k as u32, GemmDtype::Bf16, GemmDtype::Bf16, NO_SPLIT_K, self.stream)
    }

    /// TRUE fp32 GEMM (no TF32): `out[m, n] = x[m, k] @ w[n, k]^T`.
    pub fn linear_f32(&self, x: DevicePtr, w: DevicePtr, out: DevicePtr, m: usize, n: usize, k: usize) -> Result<()> {
        gemm_act_weight_t_typed_ex(x.0, k as u32, w.0, out.0, n as u32, m as u32, n as u32, k as u32, GemmDtype::F32, GemmDtype::F32, NO_SPLIT_K, self.stream)
    }

    /// [`Self::linear_bf16_strided`] in fixed [`MM_TILE`]-row tiles. `x` and `out` must hold
    /// [`tiled_rows`]`(m)` rows (the last tile touches slack rows; their contents are garbage
    /// and nothing may read them).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16_tiled(&self, x: DevicePtr, lda: usize, w: DevicePtr, out: DevicePtr, ldc: usize, m: usize, n: usize, k: usize) -> Result<()> {
        for r in (0..m).step_by(MM_TILE) {
            gemm_act_weight_t_typed_ex(
                x.offset(r * lda * 2).0, lda as u32, w.0, out.offset(r * ldc * 2).0, ldc as u32,
                MM_TILE as u32, n as u32, k as u32, GemmDtype::Bf16, GemmDtype::Bf16, NO_SPLIT_K, self.stream)?;
        }
        Ok(())
    }

    /// TRUE fp32 GEMM in fixed [`MM_TILE`]-row tiles; same slack-row contract.
    pub fn linear_f32_tiled(&self, x: DevicePtr, w: DevicePtr, out: DevicePtr, m: usize, n: usize, k: usize) -> Result<()> {
        for r in (0..m).step_by(MM_TILE) {
            gemm_act_weight_t_typed_ex(
                x.offset(r * k * 4).0, k as u32, w.0, out.offset(r * n * 4).0, n as u32,
                MM_TILE as u32, n as u32, k as u32, GemmDtype::F32, GemmDtype::F32, NO_SPLIT_K, self.stream)?;
        }
        Ok(())
    }

    /// [`Self::linear_fp8`] with the GEMM tiled; same slack-row contract.
    pub fn linear_fp8_tiled(&self, x: DevicePtr, w: &Fp8Linear, scratch: DevicePtr, out: DevicePtr, m: usize) -> Result<()> {
        self.dequant(w, scratch)?;
        self.linear_bf16_tiled(x, w.k, scratch, out, w.n, m, w.n, w.k)
    }

    /// Dequantize `w` to bf16 into `scratch` (at least `w.n * w.k * 2` bytes).
    pub fn dequant(&self, w: &Fp8Linear, scratch: DevicePtr) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.dequant)
            .grid([div_ceil(w.k as u32, BLOCK), w.n as u32, 1])
            .block([BLOCK, 1, 1])
            .arg_ptr(w.weight).arg_ptr(w.scale).arg_ptr(scratch)
            .arg_u32(w.n as u32).arg_u32(w.k as u32)
            .launch(self.stream)
    }

    /// `out[m, w.n] = x[m, w.k] @ dequant(w)^T` through a transient bf16 copy of `w`.
    pub fn linear_fp8(&self, x: DevicePtr, w: &Fp8Linear, scratch: DevicePtr, out: DevicePtr, m: usize) -> Result<()> {
        self.dequant(w, scratch)?;
        self.linear_bf16(x, scratch, out, m, w.n, w.k)
    }
}

/// `hc_{attn,ffn}_{fn,scale,base}`: fp32 [24, HC*D], [3], [24].
#[derive(Clone, Copy, Debug)]
pub struct HcParams {
    pub func: DevicePtr,
    pub scale: DevicePtr,
    pub base: DevicePtr,
}

impl HcParams {
    /// `prefix` is `layers.N.hc_attn` or `layers.N.hc_ffn`.
    pub fn load(store: &WeightStore, prefix: &str, hc: usize, d: usize) -> Result<Self> {
        let nmix = (2 + hc) * hc;
        Ok(Self {
            func: f32_tensor(store, &format!("{prefix}_fn"), &[nmix, hc * d])?,
            scale: f32_tensor(store, &format!("{prefix}_scale"), &[3])?,
            base: f32_tensor(store, &format!("{prefix}_base"), &[nmix])?,
        })
    }
}

/// A tensor that MUST be fp32 in the store with this exact shape — no silent bf16 cast.
///
/// This exists because `dense_auto` converts an fp32 tensor to bf16. For the router bias
/// (mean ~9.8, std 0.036) that rounds to steps of 0.0625 — larger than the spread that
/// decides routing. fp32 tensors are consumed as fp32 here, always.
pub fn f32_tensor(store: &WeightStore, name: &str, shape: &[usize]) -> Result<DevicePtr> {
    let t = store.get(name)?;
    ensure!(t.dtype == WeightDtype::FP32, "{name}: expected F32, got {:?}", t.dtype);
    ensure!(t.shape == shape, "{name}: expected shape {shape:?}, got {:?}", t.shape);
    Ok(t.ptr)
}

/// A bf16 tensor with an exact shape.
pub fn bf16_tensor(store: &WeightStore, name: &str, shape: &[usize]) -> Result<DevicePtr> {
    let t = store.get(name)?;
    ensure!(t.dtype == WeightDtype::BF16, "{name}: expected BF16, got {:?}", t.dtype);
    ensure!(t.shape == shape, "{name}: expected shape {shape:?}, got {:?}", t.shape);
    Ok(t.ptr)
}

/// An FP8 e4m3 weight `[n, k]` with UE8M0 block-32 scales `[ceil(n/32), ceil(k/32)]`.
#[derive(Clone, Copy, Debug)]
pub struct Fp8Linear {
    pub weight: DevicePtr,
    pub scale: DevicePtr,
    pub n: usize,
    pub k: usize,
}

impl Fp8Linear {
    /// `prefix` is e.g. `layers.2.attn.wq_a` (reads `.weight` and `.scale`).
    pub fn load(store: &WeightStore, prefix: &str, n: usize, k: usize) -> Result<Self> {
        let w = store.get(&format!("{prefix}.weight"))?;
        let s = store.get(&format!("{prefix}.scale"))?;
        ensure!(w.dtype == WeightDtype::FP8E4M3, "{prefix}.weight: expected F8_E4M3, got {:?}", w.dtype);
        ensure!(w.shape == [n, k], "{prefix}.weight: expected [{n}, {k}], got {:?}", w.shape);
        ensure!(s.dtype == WeightDtype::FP8E8M0, "{prefix}.scale: expected F8_E8M0, got {:?}", s.dtype);
        let want = [n.div_ceil(32), k.div_ceil(32)];
        ensure!(
            s.shape == want,
            "{prefix}.scale: expected block-32 scales {want:?}, got {:?} (a different block size \
             would dequantize every weight with the wrong scale)",
            s.shape
        );
        Ok(Self { weight: w.ptr, scale: s.ptr, n, k })
    }

    pub fn bf16_bytes(&self) -> usize {
        self.n * self.k * 2
    }
}

/// One RoPE table: `cos`/`sin` fp32 `[positions, half]` on the device.
#[derive(Clone, Copy, Debug)]
pub struct RopeTable {
    pub cos: DevicePtr,
    pub sin: DevicePtr,
    pub half: usize,
    pub positions: usize,
}

/// Parameters of `v41_ref.precompute_freqs_cis`.
#[derive(Clone, Copy, Debug)]
pub struct RopeSpec {
    pub dim: usize,
    pub original_seq_len: usize,
    pub base: f64,
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

impl RopeSpec {
    /// Port of `v41_ref.precompute_freqs_cis` (YaRN when `original_seq_len > 0`), returning
    /// the per-pair inverse frequencies in fp32 exactly as torch forms them.
    pub fn inv_freqs(&self) -> Vec<f32> {
        let dim = self.dim;
        // freqs = 1.0 / (base ** (arange(0, dim, 2, float32) / dim))  -- fp32 torch ops
        let mut f: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| {
                let e = (i as f32) / (dim as f32);
                1.0f32 / (self.base as f32).powf(e)
            })
            .collect();
        if self.original_seq_len > 0 {
            let corrected = |rot: f64| {
                dim as f64 * (self.original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * self.base.ln())
            };
            let low = corrected(self.beta_fast).floor().max(0.0);
            let high = corrected(self.beta_slow).ceil().min(dim as f64 - 1.0);
            let span = (high - low).max(1e-3) as f32;
            for (j, v) in f.iter_mut().enumerate() {
                let ramp = ((j as f32 - low as f32) / span).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *v = *v / self.factor as f32 * (1.0 - smooth) + *v * smooth;
            }
        }
        f
    }

    /// Upload `[positions, dim/2]` cos and sin tables.
    pub fn upload(&self, gpu: &dyn GpuBackend, positions: usize) -> Result<RopeTable> {
        let inv = self.inv_freqs();
        let half = inv.len();
        let mut cos = Vec::with_capacity(positions * half);
        let mut sin = Vec::with_capacity(positions * half);
        for p in 0..positions {
            for &w in &inv {
                // torch.outer(arange(seqlen), freqs): the int position is promoted to fp32
                // and multiplied in fp32; polar() then takes cos/sin of that fp32 angle.
                let a = (p as f32) * w;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        let bytes = positions * half * 4;
        let c = gpu.alloc(bytes)?;
        let s = gpu.alloc(bytes)?;
        gpu.copy_h2d(bytemuck_f32(&cos), c)?;
        gpu.copy_h2d(bytemuck_f32(&sin), s)?;
        Ok(RopeTable { cos: c, sin: s, half, positions })
    }
}

pub fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any byte pattern of the resulting slice is valid u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn bytemuck_i32(v: &[i32]) -> &[u8] {
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn bytemuck_u32(v: &[u32]) -> &[u8] {
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two tables must differ, and only YaRN may bend the low frequencies.
    ///
    /// The control is built in: the non-YaRN spec must leave `inv_freqs` at the plain
    /// `base^(-i/dim)` curve, and the YaRN spec must NOT. A port that ignored
    /// `original_seq_len` would pass the first assertion and fail the second.
    #[test]
    fn yarn_bends_only_the_compressed_table() {
        let w = RopeSpec { dim: 64, original_seq_len: 0, base: 10000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
        let c = RopeSpec { original_seq_len: 65536, base: 160000.0, ..w };
        let fw = w.inv_freqs();
        for (i, v) in fw.iter().enumerate() {
            let plain = 1.0f32 / 10000f32.powf((2 * i) as f32 / 64.0);
            assert_eq!(*v, plain, "freqs_w pair {i} must be plain RoPE");
        }
        let fc = c.inv_freqs();
        let plain_c: Vec<f32> = (0..32).map(|i| 1.0f32 / 160000f32.powf((2 * i) as f32 / 64.0)).collect();
        assert_eq!(fc[0], plain_c[0], "the highest frequency is left alone by YaRN");
        assert!(
            (fc[31] - plain_c[31] / 16.0).abs() <= plain_c[31] * 1e-6,
            "the lowest frequency is divided by the YaRN factor: {} vs {}",
            fc[31],
            plain_c[31] / 16.0
        );
        assert_ne!(fc, plain_c, "freqs_c must NOT be plain RoPE");
    }
}
