// SPDX-License-Identifier: AGPL-3.0-only

//! Typed launch of the fused FP8-weight GEMM (`kernels/gb10/deepseek-v4.1/cb3/dsv41_fp8_gemm.cu`,
//! `dsv41_fp8_gemm_nt_v2`): `out[m, n] = x[m, k] @ dequant(w[n, k])^T` without the bf16 copy of
//! the weight that `Ops::linear_fp8` materialises first.
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
const BM: usize = 128;
const BN: usize = 128;
const BK: usize = 32;

/// Which weight shapes take the fused kernel: decided by the WEIGHT alone (N, K), never by M, so a
/// given linear runs the SAME kernel for a 512-row chunk and a 20-row tail. Chunk invariance then
/// holds by construction, instead of resting on fused == pinned cuBLASLt bytewise (lead ruling).
/// Measured at M = 512 (gate log): wins wo_b 1.50x, w2 1.51x, w1/w3 1.14x, wq_a 1.15x (all
/// K >= 2048, N >= 1280); losses excluded: wq_b (K = 1280) 0.82x, wkv (N = 512) 0.49x, a wo_a group
/// (N = 1024) 0.86x. N >= 1280 is the old ">= 40 CTAs at M = 512" rule with M fixed at 512.
pub fn fused_wins(n: usize, k: usize) -> bool {
    k >= 2048 && n >= 1280 && n % BN == 0 && k % BK == 0
}

#[derive(Clone, Copy, Debug)]
pub struct Fp8Gemm {
    kernel: KernelHandle,
}

impl Fp8Gemm {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let kernel = gpu
            .kernel(FP8_GEMM_MODULE, "dsv41_fp8_gemm_nt_v2")
            .with_context(|| format!("{FP8_GEMM_MODULE}::dsv41_fp8_gemm_nt_v2 is not in the PTX"))?;
        Ok(Self { kernel })
    }

    /// `out` (row stride `ldc`) = `x` (row stride `lda`) @ dequant(`w`)^T, bf16, fp32 accumulate.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(&self, gpu: &dyn GpuBackend, x: DevicePtr, lda: usize, w: &Fp8Linear, out: DevicePtr, ldc: usize, m: usize, stream: u64) -> Result<()> {
        ensure!(w.n % BN == 0 && w.k % BK == 0, "fused fp8 GEMM needs N % {BN} == 0 and K % {BK} == 0 (got {}x{})", w.n, w.k);
        ensure!(lda >= w.k && ldc >= w.n, "lda {lda} / ldc {ldc} too small for {}x{}", w.n, w.k);
        if m == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.kernel)
            .grid([(w.n / BN) as u32, m.div_ceil(BM) as u32, 1])
            .block([256, 1, 1])
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

    /// The dispatch rule reproduces the gate's measured wins and losses at M=512.
    #[test]
    fn fused_wins_where_it_measured_faster() {
        for (n, k) in [(5120, 8192), (5120, 2304), (2304, 5120), (1280, 5120)] {
            assert!(fused_wins(n, k), "{n}x{k} measured faster fused");
        }
        for (n, k) in [(32768, 1280), (512, 5120), (1024, 4096)] {
            assert!(!fused_wins(n, k), "{n}x{k} measured slower fused");
        }
    }
}
