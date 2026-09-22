// SPDX-License-Identifier: AGPL-3.0-only

//! Dtype- and stride-parameterised `out[M,N] = act[M,K] @ weight[N,K]ᵀ`.
//!
//! The BF16 entry points in the parent module fix the operand type. DeepSeek-V4.1 also needs
//! TRUE fp32 GEMMs — the router (`y.float() @ gate_w`), the ratio-2 compressor and the
//! mHC mixes all run in fp32 in the reference, and routing turns on ulps — so this adds one
//! path where the operand and output types are arguments. `CUBLAS_COMPUTE_32F` with
//! `CUDA_R_32F` operands is IEEE fp32 (no TF32: that is `CUBLAS_COMPUTE_32F_FAST_TF32`).

use anyhow::{Result, bail};
use std::ffi::c_void;

use super::*;

/// `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK` (u32).
const PREF_REDUCTION_SCHEME_MASK: u32 = 3;

/// Element type of a GEMM operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmDtype {
    Bf16,
    F32,
}

impl GemmDtype {
    fn cuda(self) -> i32 {
        match self {
            GemmDtype::Bf16 => CUDA_R_16BF,
            GemmDtype::F32 => CUDA_R_32F,
        }
    }
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ` with explicit row strides (`lda` for `act`,
/// `ldc` for `out`, in elements; the weight is packed `[N,K]`), fp32 accumulate.
///
/// `in_dtype` applies to both `act` and `weight`; `out_dtype` to `out`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_act_weight_t_typed(
    act: u64,
    lda: u32,
    weight: u64,
    out: u64,
    ldc: u32,
    m: u32,
    n: u32,
    k: u32,
    in_dtype: GemmDtype,
    out_dtype: GemmDtype,
    stream: u64,
) -> Result<()> {
    if lda < k || ldc < n {
        bail!("cuBLASLt typed: lda ({lda}) < k ({k}) or ldc ({ldc}) < n ({n})");
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        chk(
            cublasLtMatmulDescSetAttribute(desc, DESC_TRANSA, &ta as *const i32 as *const c_void, 4),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(desc, DESC_TRANSB, &tb as *const i32 as *const c_void, 4),
            "TRANSB",
        )?;
        let (ti, to) = (in_dtype.cuda(), out_dtype.cuda());
        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(cublasLtMatrixLayoutCreate(&mut la, ti, k as u64, n as u64, k as i64), "LayoutA")?;
        chk(cublasLtMatrixLayoutCreate(&mut lb, ti, k as u64, m as u64, lda as i64), "LayoutB")?;
        chk(cublasLtMatrixLayoutCreate(&mut ld_, to, n as u64, m as u64, ldc as i64), "LayoutD")?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        // NO split-K. The reference's M=16 torch GEMMs do not split K, and even an fp32
        // split-K reduction reorders the sum enough to move bf16 outputs: measured by
        // dsv41-attention on L20's compressor (16x512x5120), the default heuristic took ckv
        // from 100.00% to 70.7% bit-exact vs runF and changed top-k on 296/512 rows; mask 0
        // restored 100.00% and 0/1024 rows differing across two chunkings.
        let no_split_k: u32 = 0;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_REDUCTION_SCHEME_MASK,
                &no_split_k as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
            ),
            "PrefReductionScheme",
        )?;
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        let heur = cublasLtMatmulAlgoGetHeuristic(
            ctx.handle,
            desc,
            la,
            lb,
            ld_,
            ld_,
            pref,
            1,
            result.as_mut_ptr() as *mut c_void,
            &mut returned,
        );
        let status = if heur != 0 || returned < 1 {
            None
        } else {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            Some(cublasLtMatmul(
                ctx.handle,
                desc,
                &alpha as *const f32 as *const c_void,
                weight as *const c_void,
                la,
                act as *const c_void,
                lb,
                &beta as *const f32 as *const c_void,
                out as *const c_void,
                ld_,
                out as *mut c_void,
                ld_,
                result.as_ptr() as *const c_void,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            ))
        };
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        match status {
            None => bail!(
                "cuBLASLt typed: no algorithm for {m}x{n}x{k} {in_dtype:?}->{out_dtype:?} \
                 (heuristic status {heur}, returned {returned})"
            ),
            Some(s) => chk(s, "MatmulTyped"),
        }
    }
}
