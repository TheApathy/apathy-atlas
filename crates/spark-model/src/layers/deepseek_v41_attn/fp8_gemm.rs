// SPDX-License-Identifier: AGPL-3.0-only

//! Typed launch of the fused FP8-weight GEMM (`kernels/gb10/deepseek-v4.1/cb3/dsv41_fp8_gemm.cu`,
//! `dsv41_fp8_gemm_nt_v7_{m256,n256}`): `out[m, n] = x[m, k] @ dequant(w[n, k])^T` without the
//! bf16 copy of the weight that `Ops::linear_fp8` materialises first.
//!
//! v7 gate (`kernels/gb10/deepseek-v4.1/fp8gemm3/gate.log`, same fixtures, M = 512 AND 2048):
//! byte-identical to dequant + cuBLASLt, rows at M = 1/4/16/20 identical with nothing stored past
//! M; at M = 2048 1.15-1.44x the dequant + GEMM path on wq_b / wo_b / w1 / w2 / wq_a.
//!
//! Gate (`kernels/gb10/deepseek-v4.1/fp8gemm/fp8_gemm_gate.log`, real layer-2 weights, M=512):
//! BYTE-IDENTICAL to dequant + cuBLASLt on all seven dense shapes, chunk-invariant (rows at M=20 ==
//! rows at M=512), control (scales +1) fails. Speed is shape-dependent, so [`fused_wins`] encodes
//! where it measured faster; elsewhere keep the current path.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::weight_loader::deepseek_v41::ops::Fp8Linear;

pub const FP8_GEMM_MODULE: &str = "dsv41_fp8_gemm";
const BN: usize = 128;
const BK: usize = 32;
/// Dynamic shared memory of the two v7 configs (`Cfg6<256,128,32,3>` / `Cfg6<128,256,32,3>`:
/// 3 x (A + raw FP8) stages + 2 bf16 B tiles).
const SMEM_M256: u32 = 94_208;
const SMEM_N256: u32 = 96_256;

/// Which weight shapes take the fused kernel: decided by the WEIGHT alone (N, K), never by M, so a
/// given linear runs the SAME kernel for a 512-row chunk and a 20-row tail. Chunk invariance then
/// holds by construction, instead of resting on fused == pinned cuBLASLt bytewise (lead ruling).
/// v7, measured at M = 512 / 2048 vs dequant + GEMM: wq_b 2.29x / 1.15x, wo_b 2.18x / 1.24x,
/// w1 1.98x / 1.44x, w2 2.01x / 1.27x, wq_a 1.00x / 1.30x; losses excluded: wkv (N = 512)
/// 0.43x / 0.92x, a wo_a group (N = 1024) 0.66x / 0.86x.
pub fn fused_wins(n: usize, k: usize) -> bool {
    k >= 1280 && n >= 1280 && n % BN == 0 && k % BK == 0
}

/// The 128 x 256 CTA (N-wide) measured faster on wq_b (N = 32768) and wo_b (K = 8192); the
/// 256 x 128 one on w1 / w2 / wq_a. Weight-keyed, like [`fused_wins`].
fn wide_n(n: usize, k: usize) -> bool {
    n % 256 == 0 && (n >= 16384 || k >= 8192)
}

#[derive(Clone, Copy, Debug)]
pub struct Fp8Gemm {
    m256: KernelHandle,
    n256: KernelHandle,
}

impl Fp8Gemm {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: &str| gpu.kernel(FP8_GEMM_MODULE, name).with_context(|| format!("{FP8_GEMM_MODULE}::{name} is not in the PTX"));
        Ok(Self { m256: k("dsv41_fp8_gemm_nt_v7_m256")?, n256: k("dsv41_fp8_gemm_nt_v7_n256")? })
    }

    /// `out` (row stride `ldc`) = `x` (row stride `lda`) @ dequant(`w`)^T, bf16, fp32 accumulate.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(&self, gpu: &dyn GpuBackend, x: DevicePtr, lda: usize, w: &Fp8Linear, out: DevicePtr, ldc: usize, m: usize, stream: u64) -> Result<()> {
        ensure!(w.n % BN == 0 && w.k % BK == 0, "fused fp8 GEMM needs N % {BN} == 0 and K % {BK} == 0 (got {}x{})", w.n, w.k);
        ensure!(lda >= w.k && ldc >= w.n, "lda {lda} / ldc {ldc} too small for {}x{}", w.n, w.k);
        if m == 0 {
            return Ok(());
        }
        // 1D L2-grouped grid: the kernel maps blockIdx.x to (m-tile, n-tile) itself.
        let (kernel, bm, bn, smem) = if wide_n(w.n, w.k) { (self.n256, 128, 256, SMEM_N256) } else { (self.m256, 256, 128, SMEM_M256) };
        KernelLaunch::new(gpu, kernel)
            .grid([((w.n / bn) * m.div_ceil(bm)) as u32, 1, 1])
            .block([256, 1, 1])
            .shared_mem(smem)
            .arg_ptr(x)
            .arg_i32(lda as i32)
            .arg_ptr(w.weight)
            .arg_ptr(w.scale)
            .arg_i32(w.k.div_ceil(32) as i32)
            .arg_ptr(out)
            .arg_i32(ldc as i32)
            .arg_i32(m as i32)
            .arg_i32(w.n as i32)
            .arg_i32(w.k as i32)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dispatch rule reproduces the v7 gate's measured wins and losses.
    #[test]
    fn fused_wins_where_it_measured_faster() {
        for (n, k) in [(32768, 1280), (5120, 8192), (5120, 2304), (2304, 5120), (1280, 5120)] {
            assert!(fused_wins(n, k), "{n}x{k} measured faster fused");
        }
        assert!(wide_n(32768, 1280) && wide_n(5120, 8192), "wq_b / wo_b take the 128 x 256 CTA");
        assert!(!wide_n(5120, 2304) && !wide_n(2304, 5120) && !wide_n(1280, 5120), "w2 / w1 / wq_a take 256 x 128");
        for (n, k) in [(512, 5120), (1024, 4096)] {
            assert!(!fused_wins(n, k), "{n}x{k} measured slower fused");
        }
    }
}
