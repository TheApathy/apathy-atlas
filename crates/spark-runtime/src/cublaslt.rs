// SPDX-License-Identifier: AGPL-3.0-only
//! Minimal cuBLASLt FFI for the high-efficiency GEMM path (`ATLAS_CUBLAS_GEMM`).
//!
//! Ported from the Laguna decode campaign (atlas-laguna 6ac39db3 + 02834be4).
//! The hand-written BF16 GEMM kernels run well off the cuBLAS ceiling on GB10
//! (Laguna measured the naive scalar path at up to 6.5x off the bandwidth
//! floor; tuned cuBLASLt streams at 267-290 GB/s ~= the 273 GB/s LPDDR5x
//! wall). In this tree the eligible sites are the BF16 DFlash drafter
//! projections (`dense_gemm_routed`); the target model is NVFP4 w4a16 and
//! keeps its own kernels. BF16 only — correctness-clean, no scale formats.

use anyhow::{Result, bail};
use std::ffi::c_void;
use std::sync::OnceLock;

// DeepSeek-V4.1 needs dtype-parameterised (bf16/fp32) GEMMs on top of this module's bf16-only
// entry points -- see typed.rs's own doc. Ported from dsv41/integration; every symbol it
// references via `use super::*` (the cublasLt type aliases, CUDA_R_* constants) already exists
// below unchanged.
mod typed;
pub use typed::{
    GemmDtype, gemm_act_weight_t_typed, gemm_act_weight_t_typed_ex, gemm_act_weight_t_typed_pinned,
    register_stream_workspace,
};

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
/// `library_types.h`: real as an `nv_fp8_e4m3`.
const CUDA_R_8F_E4M3: i32 = 28;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const DESC_TRANSA: u32 = 3;
const DESC_TRANSB: u32 = 4;
const PREF_MAX_WORKSPACE_BYTES: u32 = 1;
const LAYOUT_BATCH_COUNT: u32 = 5;
const LAYOUT_STRIDED_BATCH_OFFSET: u32 = 6;

// GLM-5.3 reduction-policy diagnostics (explicit-policy API only).
mod diagnostic;
mod diagnostic_contract;
pub use diagnostic::Bf16GemmReceipt;
use diagnostic_contract::HeuristicResult;
pub use diagnostic_contract::ReductionPolicy;

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
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuMemsetD8_v2(dptr: u64, uc: u8, n: usize) -> i32;
    fn cuStreamCreate(stream: *mut u64, flags: u32) -> i32;
    fn cuStreamDestroy_v2(stream: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
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

type PlanShape = (u32, u32, u32, i32, u32);
type PlanCache = std::collections::HashMap<PlanShape, &'static TunedPlan>;

static PLANS: OnceLock<std::sync::Mutex<PlanCache>> = OnceLock::new();

/// Build desc+layouts for `out[M,N]=act[M,K]@W[N,K]ᵀ` (same mapping as
/// `bf16_gemm_act_weight_t`), autotune over the heuristic's top-16 algos on a
/// PRIVATE stream with zeroed dummy operands (runs once per shape; safe while
/// another stream is mid graph-capture), and cache the winner. The naive
/// heuristic[0] pick left the K=8 verify o_proj at 113 GB/s (~24 CTAs on 48
/// SMs, no split-K) — tuning recovers the split-K/tile choice per shape.
fn tuned_plan(m: u32, n: u32, k: u32, ldc: u32) -> Result<&'static TunedPlan> {
    tuned_plan_dt(m, n, k, CUDA_R_16BF, 2, ldc)
}

/// As [`tuned_plan`], but with the A/B operand element type chosen by the
/// caller (`at`, `ab` = its size in bytes). D stays BF16. The plan cache is
/// keyed by operand type AND the output row stride as well as shape, so FP8
/// and BF16 plans — and plans for different `ldc` — never collide.
fn tuned_plan_dt(
    m: u32,
    n: u32,
    k: u32,
    at: i32,
    ab: usize,
    ldc: u32,
) -> Result<&'static TunedPlan> {
    let plans = PLANS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(p) = plans.lock().unwrap().get(&(m, n, k, at, ldc)) {
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
            cublasLtMatrixLayoutCreate(&mut la, at, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, at, k as u64, m as u64, k as i64),
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
        // caller's stream (which may be mid graph-capture).
        let (mut dw, mut da, mut dd): (u64, u64, u64) = (0, 0, 0);
        chk(
            cuMemAlloc_v2(&mut dw, n as usize * k as usize * ab),
            "tuneAllocW",
        )?;
        chk(
            cuMemAlloc_v2(&mut da, m as usize * k as usize * ab),
            "tuneAllocA",
        )?;
        chk(
            cuMemAlloc_v2(&mut dd, m as usize * ldc as usize * 2),
            "tuneAllocD",
        )?;
        cuMemsetD8_v2(dw, 0, n as usize * k as usize * ab);
        cuMemsetD8_v2(da, 0, m as usize * k as usize * ab);
        let mut ts: u64 = 0;
        chk(cuStreamCreate(&mut ts, 1), "tuneStream")?;
        let (mut e0, mut e1): (u64, u64) = (0, 0);
        cuEventCreate(&mut e0, 0);
        cuEventCreate(&mut e1, 0);
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // Candidate timings, indexed 1:1 with `results`. NaN = candidate was
        // rejected (bad state, runtime refusal, or event failure) and is not
        // eligible. Selection happens AFTER the loop so the winner is a pure
        // function of the ordering, not of measurement sequence.
        let mut times = vec![f32::NAN; returned as usize];
        let iters = 10;
        // Bursts are min-reduced: cross-process spread for a FIXED algo was
        // measured at 2-23% on GB10 (one transient per ~9 runs dominates the
        // tail), and min-of-k is the robust estimator that filters it.
        let bursts = 3;
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
            // Warmup burst, untimed: first-touch of this algo's workspace and
            // any lazy kernel load must not land inside a measured window.
            for _ in 0..iters {
                let _ = run(ts);
            }
            if cuStreamSynchronize(ts) != 0 {
                continue;
            }
            let mut per = f32::INFINITY;
            let mut ok = true;
            for _ in 0..bursts {
                cuEventRecord(e0, ts);
                for _ in 0..iters {
                    let _ = run(ts);
                }
                cuEventRecord(e1, ts);
                if cuEventSynchronize(e1) != 0 {
                    ok = false;
                    break;
                }
                let mut ms = 0f32;
                cuEventElapsedTime(&mut ms, e0, e1);
                per = per.min(ms / iters as f32);
            }
            if ok && per.is_finite() {
                times[i] = per;
            }
        }
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
        cuStreamDestroy_v2(ts);
        cuMemFree_v2(dw);
        cuMemFree_v2(da);
        cuMemFree_v2(dd);

        // Deterministic selection. A wall-clock race between candidates that
        // are numerically DIFFERENT but equally fast is a per-process coin
        // flip: whichever algo wins picks a different split-K count, hence a
        // different fp32 accumulation order, hence different bf16 roundings.
        // On Qwen3.8-Flash-Next the 2048x4x10240 hyperconnection inject offers
        // 8 split-K variants that all measure 0.148ms, and the flip propagated
        // a 1-ULP perturbation from layer 0 into an argmax flip at the head --
        // the same binary and config produced different completions at T=0.
        //
        // So: take the fastest time, then keep the LOWEST-INDEXED candidate
        // within TIE_EPS of it. cuBLASLt returns candidates in its own
        // deterministic preference order, so ties resolve to the heuristic's
        // own top pick and the outcome is a function of the ORDERING. A rule
        // that instead threaded a running incumbent through the loop would
        // still let measured duration decide, only at a smaller margin.
        //
        // TIE_EPS must exceed the cross-process noise floor for a FIXED algo,
        // which was measured at 2-23% on GB10 before the warmup/min-of-bursts
        // hardening above. Verify with two processes and diff the tune lines
        // across EVERY shape after changing it.
        // 0.20 is calibrated, not chosen: the worst WITHIN-process ratio of
        // candidate[0] to the fastest candidate observed across a two-process
        // receipt was 1.099 (2048x10240x320, where the two candidates' order
        // reversed between processes), so this is ~2x headroom on measured
        // data. Its cost is bounded by the same number: at most ~10% on that
        // one shape, 0-3.4% on the other eight, against a prefill where that
        // GEMM is <0.1% of the time. Re-derive it from a receipt if the
        // candidate set changes.
        const TIE_EPS: f32 = 0.20;
        let t_min = times
            .iter()
            .copied()
            .filter(|t| t.is_finite())
            .fold(f32::INFINITY, f32::min);
        if !t_min.is_finite() {
            bail!("cuBLASLt: every algorithm for {m}x{n}x{k} failed to run");
        }
        let band = t_min * (1.0 + TIE_EPS);
        let best = times
            .iter()
            .position(|t| t.is_finite() && *t <= band)
            .expect("t_min is finite, so at least one candidate is within the band");
        let best_ms = times[best];
        let in_band = times
            .iter()
            .filter(|t| t.is_finite() && **t <= band)
            .count();
        let bytes = n as u64 * k as u64 * ab as u64;
        tracing::info!(
            "cuBLASLt tune {m}x{n}x{k} dtype={at}: algo[{best}] of {returned} @ {best_ms:.3}ms \
             ({:.0} GB/s weight-read) fastest={t_min:.3}ms tie_band={in_band} \
             eps={TIE_EPS} times={times:.4?}",
            bytes as f64 / (best_ms as f64 / 1e3) / 1e9,
        );
        let plan: &'static TunedPlan = Box::leak(Box::new(TunedPlan {
            desc: desc as usize,
            la: la as usize,
            lb: lb as usize,
            ld: ld_ as usize,
            algo: results[best].algo,
        }));
        plans.lock().unwrap().insert((m, n, k, at, ldc), plan);
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
    bf16_gemm_act_weight_t_tuned_ex(act, weight, out, m, n, k, n, 1.0, stream)
}

/// Tuned variant with explicit output stride and alpha (plans cached per (m,n,k,ldc)).
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_tuned_ex(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    alpha: f32,
    stream: u64,
) -> Result<()> {
    if ldc < n {
        bail!("cuBLASLt: ldc {ldc} < n {n}");
    }
    let plan = tuned_plan(m, n, k, ldc)?;
    let ctx = ctx()?;
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
    bf16_gemm_act_weight_t_ldc(act, weight, out, m, n, k, n, 1.0, stream)
}

/// [`bf16_gemm_act_weight_t`] with an explicit output row stride `ldc >= n`
/// (elements), so a projection can land inside a wider interleaved row.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_ldc(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    alpha: f32,
    stream: u64,
) -> Result<()> {
    if ldc < n {
        bail!("cuBLASLt: ldc {ldc} < n {n}");
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
        let alpha: f32 = alpha;
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

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ` with **FP8 e4m3 operands and
/// a BF16 result**, using the same per-shape immortal plan cache as
/// [`bf16_gemm_act_weight_t_tuned`], so there is no per-call descriptor or
/// heuristic work. The first call at a given shape autotunes over the
/// heuristic's top-16 algorithms on a private stream; every later call is a
/// single `cublasLtMatmul`.
///
/// The hand-written prefill kernels round both operands to e4m3 before
/// `mma.sync...f32.e4m3.e4m3.f32` (the W4A16 path dequantises NVFP4 and
/// re-rounds via `cvt.rn.satfinite.e4m3x2.f32`), so passing those same e4m3
/// bytes here computes identical products and differs only in accumulation
/// order. Not bit-exact by construction; gated and gate-tested.
///
/// `k` and `n` must be multiples of 16 (cuBLASLt FP8 constraint).
pub fn fp8_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if k % 16 != 0 || n % 16 != 0 {
        bail!("cuBLASLt FP8 requires k and n multiples of 16, got k={k} n={n}");
    }
    // FP8 path writes a packed [m, n] output, so the row stride is n.
    let plan = tuned_plan_dt(m, n, k, CUDA_R_8F_E4M3, 1, n)?;
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
            "MatmulFp8Tuned",
        )
    }
}

// ── GLM-5.3 GEMM entry points ────────────────────────────────────────────────
// Ported from perf/glm-5.3-flash-prefill-20260920. They use their own
// first-heuristic core (`gemm_impl_batched`) so the GLM path keeps its exact
// algorithm choice; `bf16_gemm_act_weight_t` above is shared and unchanged
// (it too takes the first heuristic, so GLM's calls to it are numerically the
// same as on the GLM branch).

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
