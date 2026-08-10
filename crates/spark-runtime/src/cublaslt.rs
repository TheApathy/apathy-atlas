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
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuMemsetD8_v2(dptr: u64, uc: u8, n: usize) -> i32;
    fn cuStreamCreate(stream: *mut u64, flags: u32) -> i32;
    fn cuStreamDestroy_v2(stream: u64) -> i32;
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// True ABI layout of cublasLtMatmulHeuristicResult_t (96 bytes). The
/// single-result path above uses a padded 128-byte scratch, but an ARRAY of
/// results must use the exact stride or every entry past [0] is garbage.
#[repr(C)]
#[derive(Clone, Copy)]
struct HeurResult {
    algo: [u8; 64],
    workspace_size: usize,
    state: i32,
    waves_count: f32,
    reserved: [i32; 4],
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

/// Immortal per-shape matmul plan: desc + layouts + the algo that won the
/// first-use autotune. cublasLt objects are immutable after setup, so sharing
/// one plan across calls (and inside CUDA graph capture) is safe.
struct TunedPlan {
    desc: usize,
    la: usize,
    lb: usize,
    ld: usize,
    algo: [u8; 64],
}
unsafe impl Send for TunedPlan {}
unsafe impl Sync for TunedPlan {}

/// Plan-cache key: (m, n, k, lda, ldc) — lda/ldc are the row strides of the
/// row-major activation/output (elements); packed calls use (k, n).
type PlanKey = (u32, u32, u32, u32, u32);
type PlanMap = std::sync::Mutex<std::collections::HashMap<PlanKey, &'static TunedPlan>>;

static PLANS: OnceLock<PlanMap> = OnceLock::new();

/// Build desc+layouts for `out[M,N]=act[M,K]@W[N,K]ᵀ` (same mapping as
/// `bf16_gemm_act_weight_t`), autotune over the heuristic's top-16 algos on a
/// PRIVATE stream with zeroed dummy operands (runs once per shape; safe while
/// another stream is mid graph-capture), and cache the winner. The naive
/// heuristic[0] pick left the K=8 verify o_proj at 113 GB/s (~24 CTAs on 48
/// SMs, no split-K) — tuning recovers the split-K/tile choice per shape.
///
/// `lda`/`ldc` are the ROW strides (in elements) of the row-major activation
/// and output: `lda == k` / `ldc == n` is the packed case; larger values read
/// A as a column slice of a wider matrix and write C as a column slice of a
/// wider matrix (cublasLtMatrixLayout carries ld natively — same GEMM, same
/// math, only WHERE operands are read/written changes). The weight is always
/// packed `[N,K]`.
fn tuned_plan(m: u32, n: u32, k: u32, lda: u32, ldc: u32) -> Result<&'static TunedPlan> {
    let plans = PLANS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(p) = plans.lock().unwrap().get(&(m, n, k, lda, ldc)) {
        return Ok(p);
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
        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_16BF, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_16BF, k as u64, m as u64, lda as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, ldc as i64),
            "LayoutD",
        )?;
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
        let mut results = [HeurResult {
            algo: [0; 64],
            workspace_size: 0,
            state: 0,
            waves_count: 0.0,
            reserved: [0; 4],
        }; 16];
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
                16,
                results.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        cublasLtMatmulPreferenceDestroy(pref);
        if returned < 1 {
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }

        // Dummy operands (zeroed) + private stream: tune without touching the
        // caller's stream (which may be mid graph-capture). Extents cover the
        // FULL row strides so strided plans tune against realistic addresses.
        let (mut dw, mut da, mut dd): (u64, u64, u64) = (0, 0, 0);
        chk(
            cuMemAlloc_v2(&mut dw, n as usize * k as usize * 2),
            "tuneAllocW",
        )?;
        chk(
            cuMemAlloc_v2(&mut da, m as usize * lda as usize * 2),
            "tuneAllocA",
        )?;
        chk(
            cuMemAlloc_v2(&mut dd, m as usize * ldc as usize * 2),
            "tuneAllocD",
        )?;
        cuMemsetD8_v2(dw, 0, n as usize * k as usize * 2);
        cuMemsetD8_v2(da, 0, m as usize * lda as usize * 2);
        let mut ts: u64 = 0;
        chk(cuStreamCreate(&mut ts, 1), "tuneStream")?;
        let (mut e0, mut e1): (u64, u64) = (0, 0);
        cuEventCreate(&mut e0, 0);
        cuEventCreate(&mut e1, 0);
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let mut best = 0usize;
        let mut best_ms = f32::INFINITY;
        let iters = 10;
        for (i, r) in results.iter().enumerate().take(returned as usize) {
            if r.state != 0 {
                continue;
            }
            let run = |st: u64| {
                cublasLtMatmul(
                    ctx.handle,
                    desc,
                    &alpha as *const f32 as *const c_void,
                    dw as *const c_void,
                    la,
                    da as *const c_void,
                    lb,
                    &beta as *const f32 as *const c_void,
                    dd as *const c_void,
                    ld_,
                    dd as *mut c_void,
                    ld_,
                    r.algo.as_ptr() as *const c_void,
                    ctx.workspace as *mut c_void,
                    ctx.ws_size,
                    st as *mut c_void,
                )
            };
            if run(ts) != 0 {
                continue; // algo rejected at runtime — skip
            }
            cuEventRecord(e0, ts);
            for _ in 0..iters {
                let _ = run(ts);
            }
            cuEventRecord(e1, ts);
            if cuEventSynchronize(e1) != 0 {
                continue;
            }
            let mut ms = 0f32;
            cuEventElapsedTime(&mut ms, e0, e1);
            let per = ms / iters as f32;
            if per < best_ms {
                best_ms = per;
                best = i;
            }
        }
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
        cuStreamDestroy_v2(ts);
        cuMemFree_v2(dw);
        cuMemFree_v2(da);
        cuMemFree_v2(dd);

        let bytes = n as u64 * k as u64 * 2;
        tracing::info!(
            "cuBLASLt tune {m}x{n}x{k} (lda={lda} ldc={ldc}): algo[{best}] of {returned} \
             @ {best_ms:.3}ms ({:.0} GB/s weight-read)",
            bytes as f64 / (best_ms as f64 / 1e3) / 1e9,
        );
        let plan: &'static TunedPlan = Box::leak(Box::new(TunedPlan {
            desc: desc as usize,
            la: la as usize,
            lb: lb as usize,
            ld: ld_ as usize,
            algo: results[best].algo,
        }));
        plans.lock().unwrap().insert((m, n, k, lda, ldc), plan);
        Ok(plan)
    }
}

/// Tuned variant of [`bf16_gemm_act_weight_t`]: per-shape cached plan
/// (autotuned on first use), zero per-call descriptor/heuristic overhead.
pub fn bf16_gemm_act_weight_t_tuned(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let plan = tuned_plan(m, n, k, k, n)?;
    let ctx = ctx()?;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        chk(
            cublasLtMatmul(
                ctx.handle,
                plan.desc as cublasLtMatmulDesc_t,
                &alpha as *const f32 as *const c_void,
                weight as *const c_void,
                plan.la as cublasLtMatrixLayout_t,
                act as *const c_void,
                plan.lb as cublasLtMatrixLayout_t,
                &beta as *const f32 as *const c_void,
                out as *const c_void,
                plan.ld as cublasLtMatrixLayout_t,
                out as *mut c_void,
                plan.ld as cublasLtMatrixLayout_t,
                plan.algo.as_ptr() as *const c_void,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            ),
            "MatmulTuned",
        )
    }
}

/// Strided variant of [`bf16_gemm_act_weight_t_tuned`] for block-diagonal /
/// grouped projections (V4 grouped wo_a prefill): the activation is a COLUMN
/// SLICE `[M,K]` of a wider row-major matrix (row stride `lda` elements) and
/// the output is a column slice `[M,N]` of a wider row-major matrix (row
/// stride `ldc` elements). The weight stays packed `[N,K]`.
/// cublasLtMatrixLayout carries ld natively, so this is the SAME GEMM/math as
/// the packed call — only the operand addressing changes. Plans are cached per
/// (m, n, k, lda, ldc) and autotuned on first use.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_strided(
    act: u64,
    lda: u32,
    weight: u64,
    out: u64,
    ldc: u32,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if lda < k || ldc < n {
        bail!("cuBLASLt strided: lda ({lda}) < k ({k}) or ldc ({ldc}) < n ({n})");
    }
    let plan = tuned_plan(m, n, k, lda, ldc)?;
    let ctx = ctx()?;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        chk(
            cublasLtMatmul(
                ctx.handle,
                plan.desc as cublasLtMatmulDesc_t,
                &alpha as *const f32 as *const c_void,
                weight as *const c_void,
                plan.la as cublasLtMatrixLayout_t,
                act as *const c_void,
                plan.lb as cublasLtMatrixLayout_t,
                &beta as *const f32 as *const c_void,
                out as *const c_void,
                plan.ld as cublasLtMatrixLayout_t,
                out as *mut c_void,
                plan.ld as cublasLtMatrixLayout_t,
                plan.algo.as_ptr() as *const c_void,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            ),
            "MatmulStrided",
        )
    }
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
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_16BF, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_16BF, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
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
        // cublasLtMatmulHeuristicResult_t = { algo[64B], workspaceSize, state,
        // wavesCount, reserved[4] } ≈ 96B; algo at offset 0. 128B for margin.
        let mut result = [0u8; 128];
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
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }
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
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}
