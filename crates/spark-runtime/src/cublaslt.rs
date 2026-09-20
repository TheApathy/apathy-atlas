// SPDX-License-Identifier: AGPL-3.0-only
//! Minimal cuBLASLt FFI for the high-efficiency GEMM path (`ATLAS_CUBLAS_GEMM`).
//!
//! The hand-written mma.sync projection/MoE GEMMs reach only ~30% of the cuBLAS
//! ceiling on GB10 (measured: 32 vs 85 TFLOPS bf16, 152 fp8, on the SSM-qkvz
//! shape 3537×12288×2048). This routes those GEMMs through cuBLASLt instead.
//! BF16 only for now — correctness-clean (no scale-format issues); native fp8
//! block-scaled is the follow-up once the end-to-end win is proven.

use anyhow::{Result, bail};
use std::ffi::c_void;
use std::sync::OnceLock;

// Native FP8 (E4M3) GEMM paths live in the `fp8` sibling (≤500 LoC split);
// re-exported so `spark_runtime::cublaslt::fp8_gemm_*` paths are unchanged.
mod fp8;
pub use fp8::{fp8_gemm_act_weight_t_blkscaled, fp8_gemm_act_weight_t_rowwise};
mod diagnostic;
mod diagnostic_contract;
pub use diagnostic::Bf16GemmReceipt;
use diagnostic_contract::HeuristicResult;
pub use diagnostic_contract::ReductionPolicy;

#[allow(non_camel_case_types)]
type cublasLtHandle_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulDesc_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatrixLayout_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulPreference_t = *mut c_void;

const CUDA_R_16BF: i32 = 14;
const CUDA_R_16F: i32 = 2;
const CUDA_R_32F: i32 = 0;
const CUDA_R_8F_E4M3: i32 = 28;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const DESC_TRANSA: u32 = 3;
const DESC_TRANSB: u32 = 4;
const DESC_A_SCALE_POINTER: u32 = 17;
const DESC_B_SCALE_POINTER: u32 = 18;
const DESC_A_SCALE_MODE: u32 = 31;
const DESC_B_SCALE_MODE: u32 = 32;
const SCALE_MODE_OUTER_VEC_32F: i32 = 3;
const SCALE_MODE_VEC128_32F: i32 = 4;
const SCALE_MODE_BLK128X128_32F: i32 = 5;
const PREF_MAX_WORKSPACE_BYTES: u32 = 1;
const LAYOUT_BATCH_COUNT: u32 = 5;
const LAYOUT_STRIDED_BATCH_OFFSET: u32 = 6;

/// Strided batch description for `bf16_gemm_batched`: element strides.
#[derive(Clone, Copy, Debug)]
pub struct StridedBatch {
    pub count: i32,
    pub stride_act: i64,
    pub stride_weight: i64,
    pub stride_out: i64,
    /// Row stride (elements) of `act` / `out`; 0 = dense (K / N).
    pub ld_act: i64,
    pub ld_out: i64,
}

unsafe extern "C" {
    fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> i32;
    fn cublasLtMatmulDescCreate(
        desc: *mut cublasLtMatmulDesc_t,
        compute_type: i32,
        scale_type: i32,
    ) -> i32;
    fn cublasLtMatmulDescSetAttribute(
        desc: cublasLtMatmulDesc_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> i32;
    fn cublasLtMatrixLayoutCreate(
        layout: *mut cublasLtMatrixLayout_t,
        dtype: i32,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> i32;
    fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> i32;
    fn cublasLtMatrixLayoutSetAttribute(
        layout: cublasLtMatrixLayout_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> i32;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: cublasLtMatmulPreference_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoGetHeuristic(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        pref: cublasLtMatmulPreference_t,
        requested: i32,
        results: *mut c_void,
        returned: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmul(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        layout_a: cublasLtMatrixLayout_t,
        b: *const c_void,
        layout_b: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        layout_c: cublasLtMatrixLayout_t,
        d: *mut c_void,
        layout_d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
}

struct Ctx {
    handle: cublasLtHandle_t,
    workspace: u64,
    ws_size: usize,
}
// cuBLASLt handle + device workspace are process-global; matmul is invoked
// serially from the single-threaded scheduler forward.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    let mut handle: cublasLtHandle_t = std::ptr::null_mut();
    let st = unsafe { cublasLtCreate(&mut handle) };
    if st != 0 {
        bail!("cublasLtCreate failed: {st}");
    }
    let ws_size = 64 * 1024 * 1024;
    let mut ws: u64 = 0;
    let st = unsafe { cuMemAlloc_v2(&mut ws, ws_size) };
    if st != 0 {
        bail!("cuMemAlloc cuBLASLt workspace failed: {st}");
    }
    let _ = CTX.set(Ctx {
        handle,
        workspace: ws,
        ws_size,
    });
    Ok(CTX.get().unwrap())
}

fn chk(status: i32, what: &str) -> Result<()> {
    if status != 0 {
        bail!("cuBLASLt {what} failed: status {status}");
    }
    Ok(())
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all BF16 — the standard
/// projection GEMM (activation × transposed weight). Maps to cuBLASLt's
/// column-major convention as `D[N,M] = opT(weightᶜ[K,N]) · opN(actᶜ[K,M])`.
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    bf16_gemm(act, weight, out, m, n, k, stream, true)
}

/// Row-major `out[M,N] = act[M,K] @ weight[K,N]`, all BF16.
pub fn bf16_gemm_act_weight(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    bf16_gemm(act, weight, out, m, n, k, stream, false)
}

#[allow(clippy::too_many_arguments)]
fn bf16_gemm(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    weight_is_nk: bool,
) -> Result<()> {
    bf16_gemm_impl(act, weight, out, m, n, k, stream, weight_is_nk, None, CUDA_R_16BF).map(|_| ())
}

/// Strided-batched row-major BF16 GEMM: per batch `out = act @ weight[K,N]` (or
/// `@ weight[N,K]ᵀ` when `weight_is_nk`). Used by the GLM DSA absorption banks
/// (64 heads).
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_batched(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    weight_is_nk: bool,
    batch: StridedBatch,
    stream: u64,
) -> Result<()> {
    gemm_impl_batched(act, weight, out, m, n, k, stream, weight_is_nk, None, CUDA_R_16BF, Some(batch))
        .map(|_| ())
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all F32 (full fp32, no TF32).
/// Used by the GLM prompt-scope router logits and mHC mixing projections.
pub fn f32_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    bf16_gemm_impl(act, weight, out, m, n, k, stream, true, None, CUDA_R_32F).map(|_| ())
}

/// Row-major `out[M,N] = act[M,K] @ weight[K,N]`, all F16 (fp32 accumulate).
/// Used by the GLM EXL3 reconstructed-weight prefill projections.
pub fn f16_gemm_act_weight(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    bf16_gemm_impl(act, weight, out, m, n, k, stream, false, None, CUDA_R_16F).map(|_| ())
}

/// Same projection core with explicit experimental reduction selection and a
/// receipt for its actual heuristic. Caller owns buffers and completion, and
/// must serialize use of the process-global workspace just as in production.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_diagnostic(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    policy: ReductionPolicy,
) -> Result<Bf16GemmReceipt> {
    anyhow::ensure!(
        act != 0 && weight != 0 && out != 0 && m > 0 && n > 0 && k > 0,
        "invalid diagnostic GEMM pointers/dimensions"
    );
    bf16_gemm_impl(act, weight, out, m, n, k, stream, true, Some(policy), CUDA_R_16BF)?
        .ok_or_else(|| anyhow::anyhow!("missing explicit GEMM receipt"))
}

#[allow(clippy::too_many_arguments)]
fn bf16_gemm_impl(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    weight_is_nk: bool,
    policy: Option<ReductionPolicy>,
    dtype: i32,
) -> Result<Option<Bf16GemmReceipt>> {
    gemm_impl_batched(act, weight, out, m, n, k, stream, weight_is_nk, policy, dtype, None)
}

#[allow(clippy::too_many_arguments)]
fn gemm_impl_batched(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    weight_is_nk: bool,
    policy: Option<ReductionPolicy>,
    dtype: i32,
    batch: Option<StridedBatch>,
) -> Result<Option<Bf16GemmReceipt>> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = if weight_is_nk {
            CUBLAS_OP_T
        } else {
            CUBLAS_OP_N
        };
        let tb = CUBLAS_OP_N;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        // A = weight stored row-major [N,K] == col-major [K,N], ld=K, opT → [N,K]
        // B = act    stored row-major [M,K] == col-major [K,M], ld=K, opN → [K,M]
        // D = out    row-major [M,N]        == col-major [N,M], ld=N
        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let (weight_rows, weight_cols, weight_ld) = if weight_is_nk {
            (k as u64, n as u64, k as i64)
        } else {
            (n as u64, k as u64, n as i64)
        };
        chk(
            cublasLtMatrixLayoutCreate(&mut la, dtype, weight_rows, weight_cols, weight_ld),
            "LayoutA",
        )?;
        let ld_act = match batch {
            Some(b) if b.ld_act > 0 => b.ld_act,
            _ => k as i64,
        };
        let ld_out = match batch {
            Some(b) if b.ld_out > 0 => b.ld_out,
            _ => n as i64,
        };
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, dtype, k as u64, m as u64, ld_act),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, dtype, n as u64, m as u64, ld_out),
            "LayoutD",
        )?;
        if let Some(batch) = batch {
            for (layout, stride) in [
                (la, batch.stride_weight),
                (lb, batch.stride_act),
                (ld_, batch.stride_out),
            ] {
                chk(
                    cublasLtMatrixLayoutSetAttribute(
                        layout,
                        LAYOUT_BATCH_COUNT,
                        &batch.count as *const i32 as *const c_void,
                        4,
                    ),
                    "LayoutBatchCount",
                )?;
                chk(
                    cublasLtMatrixLayoutSetAttribute(
                        layout,
                        LAYOUT_STRIDED_BATCH_OFFSET,
                        &stride as *const i64 as *const c_void,
                        8,
                    ),
                    "LayoutBatchStride",
                )?;
            }
        }
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
        // Always retain the original first-heuristic control. Only the explicit
        // diagnostic API supplies a policy or asks for algorithm attributes.
        let operation = (|| -> Result<Option<Bf16GemmReceipt>> {
            if let Some(policy) = policy {
                diagnostic::configure(pref, policy)?;
            }
            let mut result = HeuristicResult::default();
            let mut returned: i32 = 0;
            chk(
                cublasLtMatmulAlgoGetHeuristic(
                    ctx.handle,
                    desc,
                    la,
                    lb,
                    ld_,
                    ld_,
                    pref,
                    1,
                    (&mut result as *mut HeuristicResult).cast(),
                    &mut returned,
                ),
                "AlgoGetHeuristic",
            )?;
            result
                .admit(returned, ctx.ws_size)
                .map_err(anyhow::Error::msg)?;
            let receipt = policy
                .map(|policy| diagnostic::receipt(&result, policy, [m, n, k], ctx.ws_size))
                .transpose()?;
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let status = cublasLtMatmul(
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
                result.algo.as_ptr().cast(),
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            );
            chk(status, "Matmul")?;
            Ok(receipt)
        })();
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        operation
    }
}
