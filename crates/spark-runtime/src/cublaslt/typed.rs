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

/// Dedicated workspaces for side streams (same size as the global one, so the heuristic and
/// the pinned algorithms are unchanged): two GEMMs running concurrently on different streams
/// must not share the process-global workspace.
fn side_workspaces() -> &'static std::sync::Mutex<std::collections::HashMap<u64, u64>> {
    static W: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u64, u64>>> = std::sync::OnceLock::new();
    W.get_or_init(Default::default)
}

fn stream_workspace(stream: u64) -> Option<u64> {
    side_workspaces().lock().ok()?.get(&stream).copied()
}

/// Give `stream` its own cuBLASLt workspace for the typed GEMMs (idempotent). Process-lifetime,
/// like the global workspace.
pub fn register_stream_workspace(stream: u64) -> Result<()> {
    let size = ctx()?.ws_size;
    let mut map = side_workspaces().lock().map_err(|_| anyhow::anyhow!("cuBLASLt side workspaces poisoned"))?;
    if map.contains_key(&stream) {
        return Ok(());
    }
    let mut ws: u64 = 0;
    let st = unsafe { cuMemAlloc_v2(&mut ws, size) };
    if st != 0 {
        bail!("cuMemAlloc side-stream cuBLASLt workspace failed: {st}");
    }
    map.insert(stream, ws);
    Ok(())
}

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
/// `in_dtype` applies to both `act` and `weight`; `out_dtype` to `out`. cuBLASLt's default
/// heuristic (split-K allowed). See [`gemm_act_weight_t_typed_ex`] for the no-split-K form.
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
    gemm_act_weight_t_typed_ex(act, lda, weight, out, ldc, m, n, k, in_dtype, out_dtype, false, stream)
}

/// [`gemm_act_weight_t_typed`] with a PER-CALL reduction policy. `no_split_k = true` restricts
/// the heuristic to algorithms that do not split K
/// (`CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 0`), so a row's sum is accumulated in one
/// order whatever M is.
///
/// Why per call and not global: this file is shared across models, and the only caller that
/// needs it measured it. dsv41-attention, on DeepSeek-V4.1's L20 compressor (16x512x5120,
/// runF_faithful): default heuristic (split-K chosen) ckv 70.7% bf16-exact, top-k 296/512 rows
/// wrong; split-K with fp32 reduction (mask 2) 97.75% / 53/512; mask 0 100.00% / 1/512.
/// (The integrate/all-models branch has a separate deterministic-SELECTION fix for the tuned
/// bf16 path in cublaslt.rs — tie-breaking between measured candidates. It does not restrict
/// the reduction scheme and does not cover this untuned typed path.)
#[allow(clippy::too_many_arguments)]
pub fn gemm_act_weight_t_typed_ex(
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
    no_split_k: bool,
    stream: u64,
) -> Result<()> {
    typed_impl(act, lda, weight, out, ldc, m, n, k, in_dtype, out_dtype, no_split_k, None, stream)
}

/// [`gemm_act_weight_t_typed_ex`] with ONE algorithm per shape, independent of M: the heuristic
/// runs once per (n, k, lda, ldc, dtypes, no_split_k) at M = `ref_m`, and that algorithm is then
/// issued at the TRUE M of every call. With a fixed tile configuration and no split-K, an output
/// element's K-loop does not depend on how many rows share the call, so results become
/// M-invariant without padding M (the property DeepSeek-V4.1's chunk invariance needs).
/// Measured, not assumed: see bench/dsv41/compare_splits.py.
#[allow(clippy::too_many_arguments)]
pub fn gemm_act_weight_t_typed_pinned(
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
    no_split_k: bool,
    ref_m: u32,
    stream: u64,
) -> Result<()> {
    typed_impl(act, lda, weight, out, ldc, m, n, k, in_dtype, out_dtype, no_split_k, Some(ref_m), stream)
}

type PinKey = (u32, u32, u32, u32, i32, i32, bool, u32);

fn pinned_algos() -> &'static std::sync::Mutex<std::collections::HashMap<PinKey, [u8; 128]>> {
    static P: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<PinKey, [u8; 128]>>> =
        std::sync::OnceLock::new();
    P.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

#[allow(clippy::too_many_arguments)]
fn typed_impl(
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
    no_split_k: bool,
    pin_ref_m: Option<u32>,
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
        if no_split_k {
            let mask: u32 = 0;
            chk(
                cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    PREF_REDUCTION_SCHEME_MASK,
                    &mask as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                ),
                "PrefReductionScheme",
            )?;
        }
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        let pin_key: Option<PinKey> =
            pin_ref_m.map(|r| (n, k, lda, ldc, ti, to, no_split_k, r));
        let cached = pin_key.and_then(|key| pinned_algos().lock().ok()?.get(&key).copied());
        let heur = if let Some(c) = cached {
            result = c;
            returned = 1;
            0
        } else if let Some(r) = pin_ref_m {
            // Choose the algorithm at the REFERENCE M, then reuse it at every M.
            let mut lb_r: cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut ld_r: cublasLtMatrixLayout_t = std::ptr::null_mut();
            chk(cublasLtMatrixLayoutCreate(&mut lb_r, ti, k as u64, r as u64, lda as i64), "LayoutBref")?;
            chk(cublasLtMatrixLayoutCreate(&mut ld_r, to, n as u64, r as u64, ldc as i64), "LayoutDref")?;
            let h = cublasLtMatmulAlgoGetHeuristic(
                ctx.handle, desc, la, lb_r, ld_r, ld_r, pref, 1,
                result.as_mut_ptr() as *mut c_void, &mut returned,
            );
            cublasLtMatrixLayoutDestroy(lb_r);
            cublasLtMatrixLayoutDestroy(ld_r);
            if h == 0 && returned >= 1 {
                if let (Some(key), Ok(mut t)) = (pin_key, pinned_algos().lock()) {
                    t.insert(key, result);
                }
                // ATLAS_LOG_PINNED_ALGO=1: one line per pinned shape, so two processes can be
                // diffed. The heuristic can pick differently when the workspace, stream or
                // allocation alignment differ; this makes that visible instead of inferred.
                if std::env::var("ATLAS_LOG_PINNED_ALGO").as_deref() == Ok("1") {
                    let hex: String = result[..64].iter().map(|b| format!("{b:02x}")).collect();
                    eprintln!(
                        "PINNED_ALGO n={n} k={k} lda={lda} ldc={ldc} in={ti} out={to} no_split_k={no_split_k} ref_m={r} ws={} stream={stream:#x} algo={hex}",
                        ctx.ws_size
                    );
                }
            }
            h
        } else {
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
            )
        };
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
                stream_workspace(stream).unwrap_or(ctx.workspace) as *mut c_void,
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

/// `cublasLtMatmulHeuristicResult_t` is 96 bytes: the 64-byte algo, workspace size, state,
/// waves count and 4 reserved ints.
const HEURISTIC_RESULT_BYTES: usize = 96;

unsafe extern "C" {
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const c_void,
        attr: u32,
        buf: *mut c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// Autotune support for [`gemm_act_weight_t_typed_pinned`]: up to `max` heuristic candidates
/// (no split-K) for a shape at M = `ref_m`, best-ranked first, each as the blob the pinned path
/// stores. Index 0 is what the pinned path picks on its own.
#[allow(clippy::too_many_arguments)]
pub fn pinned_candidates(
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    in_dtype: GemmDtype,
    out_dtype: GemmDtype,
    ref_m: u32,
    max: usize,
) -> Result<Vec<[u8; 128]>> {
    let ctx = ctx()?;
    let (ti, to) = (in_dtype.cuda(), out_dtype.cuda());
    let mut results = vec![0u8; max * HEURISTIC_RESULT_BYTES];
    let mut returned: i32 = 0;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "DescCreate")?;
        let (ta, tb) = (CUBLAS_OP_T, CUBLAS_OP_N);
        chk(cublasLtMatmulDescSetAttribute(desc, DESC_TRANSA, &ta as *const i32 as *const c_void, 4), "TRANSA")?;
        chk(cublasLtMatmulDescSetAttribute(desc, DESC_TRANSB, &tb as *const i32 as *const c_void, 4), "TRANSB")?;
        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(cublasLtMatrixLayoutCreate(&mut la, ti, k as u64, n as u64, k as i64), "LayoutA")?;
        chk(cublasLtMatrixLayoutCreate(&mut lb, ti, k as u64, ref_m as u64, lda as i64), "LayoutB")?;
        chk(cublasLtMatrixLayoutCreate(&mut ld_, to, n as u64, ref_m as u64, ldc as i64), "LayoutD")?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE_BYTES, &ws_size as *const usize as *const c_void, std::mem::size_of::<usize>()),
            "PrefWorkspace",
        )?;
        let mask: u32 = 0;
        chk(
            cublasLtMatmulPreferenceSetAttribute(pref, PREF_REDUCTION_SCHEME_MASK, &mask as *const u32 as *const c_void, std::mem::size_of::<u32>()),
            "PrefReductionScheme",
        )?;
        let h = cublasLtMatmulAlgoGetHeuristic(
            ctx.handle, desc, la, lb, ld_, ld_, pref, max as i32, results.as_mut_ptr() as *mut c_void, &mut returned,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(h, "AlgoGetHeuristic")?;
    }
    Ok(results
        .chunks_exact(HEURISTIC_RESULT_BYTES)
        .take(returned.max(0) as usize)
        .map(|r| {
            let mut blob = [0u8; 128];
            blob[..HEURISTIC_RESULT_BYTES].copy_from_slice(r);
            blob
        })
        .collect())
}

/// Replace the algorithm [`gemm_act_weight_t_typed_pinned`] uses for a shape (autotune A/B).
#[allow(clippy::too_many_arguments)]
pub fn set_pinned_algo(
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    in_dtype: GemmDtype,
    out_dtype: GemmDtype,
    no_split_k: bool,
    ref_m: u32,
    algo: [u8; 128],
) -> Result<()> {
    let key = (n, k, lda, ldc, in_dtype.cuda(), out_dtype.cuda(), no_split_k, ref_m);
    pinned_algos().lock().map_err(|_| anyhow::anyhow!("pinned algos poisoned"))?.insert(key, algo);
    Ok(())
}

/// The algorithm's config (id, tile, stages, split-K, reduction, swizzle, custom option, inner
/// shape, cluster shape) as `name=value` pairs, for logs and pin tables.
pub fn describe_algo(algo: &[u8; 128]) -> String {
    const ATTRS: [(&str, u32); 9] = [
        ("id", 0), ("tile", 1), ("splitk", 2), ("red", 3), ("swz", 4), ("custom", 5), ("stages", 6), ("inner", 7), ("cluster", 8),
    ];
    let ws = u64::from_le_bytes(algo[64..72].try_into().expect("8 bytes"));
    let mut s = String::new();
    for (name, attr) in ATTRS {
        // Size query first (null buffer): the call rejects a size that is not the attribute's.
        let mut size = 0usize;
        let a = algo.as_ptr() as *const c_void;
        if unsafe { cublasLtMatmulAlgoConfigGetAttribute(a, attr, std::ptr::null_mut(), 0, &mut size) } != 0 || size == 0 || size > 8 {
            continue;
        }
        let mut v: u64 = 0;
        let mut written = 0usize;
        let st = unsafe { cublasLtMatmulAlgoConfigGetAttribute(a, attr, &mut v as *mut u64 as *mut c_void, size, &mut written) };
        if st == 0 {
            s.push_str(&format!("{name}={v} "));
        }
    }
    s.push_str(&format!("ws={ws}"));
    s
}
