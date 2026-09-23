// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit diagnostic receipts. No environment switch or production promotion.
use super::diagnostic_contract::{HeuristicResult, ReductionPolicy};
use super::{chk, cublasLtMatmulPreference_t, cublasLtMatmulPreferenceSetAttribute};
use anyhow::{Result, ensure};
use std::ffi::c_void;

unsafe extern "C" {
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const c_void,
        attr: u32,
        value: *mut c_void,
        bytes: usize,
        written: *mut usize,
    ) -> i32;
    fn cublasLtGetVersion() -> usize;
    fn cublasLtGetCudartVersion() -> usize;
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Bf16GemmReceipt {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub reduction_mask: Option<u32>,
    pub algorithm_id: i32,
    pub tile_id: u32,
    pub split_k: i32,
    pub reduction_scheme: u32,
    pub workspace_bytes: usize,
    pub workspace_available: usize,
    pub waves: f32,
    pub cublaslt_version: usize,
    pub cudart_version: usize,
}

pub(super) fn configure(pref: cublasLtMatmulPreference_t, policy: ReductionPolicy) -> Result<()> {
    if let Some(mask) = policy.preference_mask() {
        chk(
            unsafe {
                cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    3,
                    (&mask as *const u32).cast(),
                    size_of::<u32>(),
                )
            },
            "PrefReductionScheme",
        )?;
    }
    Ok(())
}

fn attribute(result: &HeuristicResult, attr: u32) -> Result<u32> {
    let mut value = 0u32;
    let mut size_written = 0usize;
    chk(
        unsafe {
            cublasLtMatmulAlgoConfigGetAttribute(
                result.algo.as_ptr().cast(),
                attr,
                (&mut value as *mut u32).cast(),
                size_of::<u32>(),
                &mut size_written,
            )
        },
        "AlgoConfigGetAttribute",
    )?;
    ensure!(
        size_written == size_of::<u32>(),
        "algorithm attribute size drift"
    );
    Ok(value)
}

pub(super) fn receipt(
    result: &HeuristicResult,
    policy: ReductionPolicy,
    dims: [u32; 3],
    workspace_available: usize,
) -> Result<Bf16GemmReceipt> {
    let algorithm_id = attribute(result, 0)? as i32;
    let tile_id = attribute(result, 1)?;
    let split_k = attribute(result, 2)? as i32;
    let reduction_scheme = attribute(result, 3)?;
    policy
        .validate_selected(split_k, reduction_scheme)
        .map_err(anyhow::Error::msg)?;
    let cublaslt_version = unsafe { cublasLtGetVersion() };
    let cudart_version = unsafe { cublasLtGetCudartVersion() };
    ensure!(
        cublaslt_version > 0 && cudart_version > 0,
        "missing cuBLASLt library identity"
    );
    let [m, n, k] = dims;
    Ok(Bf16GemmReceipt {
        m,
        n,
        k,
        reduction_mask: policy.preference_mask(),
        algorithm_id,
        tile_id,
        split_k,
        reduction_scheme,
        workspace_bytes: result.workspace_bytes,
        workspace_available,
        waves: result.waves,
        cublaslt_version,
        cudart_version,
    })
}
