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
use std::sync::{Mutex, OnceLock};

// Native FP8 (E4M3) GEMM paths live in the `fp8` sibling (≤500 LoC split);
// re-exported so `spark_runtime::cublaslt::fp8_gemm_*` paths are unchanged.
mod fp8;
pub use fp8::{fp8_gemm_act_weight_t_blkscaled, fp8_gemm_act_weight_t_rowwise};
mod bf16_plan_cache;
#[cfg(test)]
mod bf16_plan_cache_tests;
mod classic_bf16;
#[cfg(test)]
mod classic_bf16_tests;
mod diagnostic;
mod diagnostic_contract;
pub use diagnostic::Bf16GemmReceipt;
pub use diagnostic_contract::ReductionPolicy;
mod serial_rows_contract;
mod serial_rows_driver;
mod serial_rows_ffi;
mod serial_rows_cache_ffi;
#[cfg(test)]
mod serial_rows_cache_tests;
mod strided_rows_contract;
pub use serial_rows_contract::{ByteSpan, Orientation, SerialRowsRequest};
pub use serial_rows_ffi::bf16_gemm_batched_rows;
pub use serial_rows_ffi::bf16_gemm_serial_rows;
pub use serial_rows_ffi::bf16_gemm_serial_rows_diagnostic;

#[allow(non_camel_case_types)]
type cublasLtHandle_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulDesc_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatrixLayout_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulPreference_t = *mut c_void;

const CUDA_R_16BF: i32 = 14;
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
    plans: Mutex<bf16_plan_cache::PlanCache>,
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
        plans: Mutex::new(bf16_plan_cache::PlanCache::default()),
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
    bf16_gemm_impl(act, weight, out, m, n, k, stream, weight_is_nk, None).map(|_| ())
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
    bf16_gemm_impl(act, weight, out, m, n, k, stream, true, Some(policy))?
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
) -> Result<Option<Bf16GemmReceipt>> {
    let key = bf16_plan_cache::GemmKey::new(m, n, k, weight_is_nk).map_err(anyhow::Error::msg)?;
    let context = ctx()?;
    if policy.is_none() && classic_bf16::enabled()? {
        classic_bf16::execute(context, key, act, weight, out, stream)?;
        return Ok(None);
    }
    match bf16_plan_cache::route(policy, bf16_plan_cache::enabled()?) {
        bf16_plan_cache::PlanRoute::Cached => {
            bf16_plan_cache::execute_cached(context, key, act, weight, out, stream)?;
            Ok(None)
        }
        bf16_plan_cache::PlanRoute::Ephemeral => {
            // Diagnostics intentionally remain ephemeral: their preference
            // changes must never poison the production plan cache.
            bf16_plan_cache::execute_ephemeral(context, key, act, weight, out, stream, policy)
        }
    }
}
