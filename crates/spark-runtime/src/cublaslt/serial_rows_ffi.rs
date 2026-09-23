// SPDX-License-Identifier: AGPL-3.0-only

//! Additive cuBLASLt adapter for the checked native M1 row-span driver.
//! The caller owns tensor lifetimes, completion, and serialization of the
//! existing process-global workspace. GLM admits this path only when eager.

use anyhow::{Result, anyhow, bail};
use std::ffi::c_void;
use std::mem::size_of;

use super::diagnostic_contract::{HeuristicResult, ReductionPolicy};
use super::serial_rows_contract::{
    ByteSpan, MatrixLayout, RowCall, SerialRowsPlan, SerialRowsRequest,
};
use super::serial_rows_driver::{
    DescriptorSet, ResourceKind, SerialRowsIo, run_batched_rows, run_serial_rows,
};
#[path = "serial_rows_batch_ffi.rs"]
mod batched;
use super::{Bf16GemmReceipt, diagnostic};
use super::{
    Ctx, chk, ctx, cublasLtMatmul, cublasLtMatmulAlgoGetHeuristic, cublasLtMatmulDescCreate,
    cublasLtMatmulDescDestroy, cublasLtMatmulDescSetAttribute, cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceSetAttribute,
    cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy,
};

/// Preserve the original M1 calls and per-row heuristic selection while reusing
/// one scoped descriptor set. This does not admit graph capture or add fences.
pub fn bf16_gemm_serial_rows(request: SerialRowsRequest) -> Result<()> {
    serial_rows_impl(request, false, false).map(|_| ())
}

/// Receipts describe each admitted result actually submitted to matmul. This
/// explicit diagnostic adds attribute queries only, not heuristics or fences.
pub fn bf16_gemm_serial_rows_diagnostic(
    request: SerialRowsRequest,
) -> Result<Vec<Bf16GemmReceipt>> {
    serial_rows_impl(request, true, false)
}

/// Explicit strided-M1 opt-in. Unsupported admission is an error, never fallback.
pub fn bf16_gemm_batched_rows(request: SerialRowsRequest) -> Result<()> {
    if super::serial_rows_cache_ffi::enabled()? {
        super::serial_rows_cache_ffi::execute(request)
    } else if super::classic_bf16::enabled()? {
        super::classic_bf16::execute_batched(request)
    } else {
        serial_rows_impl(request, false, true).map(|_| ())
    }
}

fn serial_rows_impl(
    request: SerialRowsRequest,
    diagnostic_enabled: bool,
    batched: bool,
) -> Result<Vec<Bf16GemmReceipt>> {
    // Malformed operands must not initialize even the old cold CUDA workspace.
    let plan = SerialRowsPlan::new(request).map_err(anyhow::Error::msg)?;
    let receipts = diagnostic_enabled.then(|| Vec::with_capacity(request.rows as usize));
    let context = ctx()?;
    let workspace = ByteSpan {
        address: context.workspace,
        bytes: context.ws_size,
    };
    plan.bind_workspace(workspace).map_err(anyhow::Error::msg)?;
    let mut io = NativeRowsIo {
        context,
        receipts,
        dims: [1, request.n, request.k],
    };
    // Raw descriptor handles are not Send/Sync. Preserve the complete primary
    // and cleanup report as an owned error string after the driver closes.
    if batched {
        run_batched_rows(&mut io, request, workspace).map_err(|error| anyhow!("{error}"))?;
    } else {
        run_serial_rows(&mut io, request, workspace).map_err(|error| anyhow!("{error}"))?;
    }
    Ok(io.receipts.unwrap_or_default())
}

pub(super) struct NativeRowsIo<'a> {
    pub(super) context: &'a Ctx,
    pub(super) receipts: Option<Vec<Bf16GemmReceipt>>,
    pub(super) dims: [u32; 3],
}

pub(super) fn destroy_raw(kind: ResourceKind, handle: *mut c_void) -> i32 {
    // Handles originate only from the corresponding cuBLASLt create call.
    unsafe {
        match kind {
            ResourceKind::Descriptor => cublasLtMatmulDescDestroy(handle),
            ResourceKind::Layout => cublasLtMatrixLayoutDestroy(handle),
            ResourceKind::Preference => cublasLtMatmulPreferenceDestroy(handle),
        }
    }
}

fn created(
    kind: ResourceKind,
    handle: *mut c_void,
    status: i32,
    operation: &'static str,
) -> Result<*mut c_void> {
    if status == 0 && !handle.is_null() {
        return Ok(handle);
    }
    // An unsuccessful foreign create may still return an owned out-handle.
    // The driver has not received it, so this adapter performs its cleanup.
    // Do this before constructing an allocated error message.
    let cleanup_status = if handle.is_null() {
        None
    } else {
        Some(destroy_raw(kind, handle))
    };
    bail!(
        "cuBLASLt {operation} failed: status={status}, kind={kind:?}, \
         returned_handle={handle:p}, cleanup_status={cleanup_status:?}"
    )
}

impl SerialRowsIo for NativeRowsIo<'_> {
    type Handle = *mut c_void;
    type Error = anyhow::Error;

    fn prepare_strided_m1(
        &mut self,
        set: DescriptorSet<Self::Handle>,
        request: SerialRowsRequest,
        call: RowCall,
        result: &HeuristicResult,
    ) -> Result<bool> {
        batched::prepare(self, set, request, call, result)
    }

    fn create_matmul(&mut self, compute: i32, scale: i32) -> Result<Self::Handle> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { cublasLtMatmulDescCreate(&mut handle, compute, scale) };
        created(
            ResourceKind::Descriptor,
            handle,
            status,
            "SerialRowsDescCreate",
        )
    }

    fn set_matmul_i32(&mut self, desc: Self::Handle, attr: u32, value: i32) -> Result<()> {
        chk(
            unsafe {
                cublasLtMatmulDescSetAttribute(
                    desc,
                    attr,
                    (&value as *const i32).cast(),
                    size_of::<i32>(),
                )
            },
            "SerialRowsDescSetAttribute",
        )
    }

    fn create_layout(&mut self, layout: MatrixLayout) -> Result<Self::Handle> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            cublasLtMatrixLayoutCreate(
                &mut handle,
                layout.dtype,
                layout.rows,
                layout.cols,
                layout.ld,
            )
        };
        created(
            ResourceKind::Layout,
            handle,
            status,
            "SerialRowsLayoutCreate",
        )
    }

    fn create_preference(&mut self) -> Result<Self::Handle> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { cublasLtMatmulPreferenceCreate(&mut handle) };
        created(
            ResourceKind::Preference,
            handle,
            status,
            "SerialRowsPrefCreate",
        )
    }

    fn set_preference_usize(&mut self, pref: Self::Handle, attr: u32, value: usize) -> Result<()> {
        chk(
            unsafe {
                cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    attr,
                    (&value as *const usize).cast(),
                    size_of::<usize>(),
                )
            },
            "SerialRowsPrefSetAttribute",
        )
    }

    fn heuristic(
        &mut self,
        set: DescriptorSet<Self::Handle>,
        requested: i32,
    ) -> Result<(HeuristicResult, i32)> {
        let mut result = HeuristicResult::default();
        let mut returned = 0;
        chk(
            unsafe {
                cublasLtMatmulAlgoGetHeuristic(
                    self.context.handle,
                    set.desc,
                    set.a,
                    set.b,
                    set.c,
                    set.d,
                    set.pref,
                    requested,
                    (&mut result as *mut HeuristicResult).cast(),
                    &mut returned,
                )
            },
            "SerialRowsAlgoGetHeuristic",
        )?;
        // The shared production driver calls the original result.admit before
        // passing this exact result back to matmul for the corresponding row.
        Ok((result, returned))
    }

    fn matmul(
        &mut self,
        set: DescriptorSet<Self::Handle>,
        call: RowCall,
        result: &HeuristicResult,
    ) -> Result<()> {
        let receipt = if self.receipts.is_some() {
            Some(diagnostic::receipt(
                result,
                ReductionPolicy::Baseline,
                self.dims,
                call.workspace_bytes,
            )?)
        } else {
            None
        };
        let alpha = f32::from_bits(call.alpha_bits);
        let beta = f32::from_bits(call.beta_bits);
        chk(
            unsafe {
                cublasLtMatmul(
                    self.context.handle,
                    set.desc,
                    (&alpha as *const f32).cast(),
                    call.weight as *const c_void,
                    set.a,
                    call.act as *const c_void,
                    set.b,
                    (&beta as *const f32).cast(),
                    call.out as *const c_void,
                    set.c,
                    call.out as *mut c_void,
                    set.d,
                    result.algo.as_ptr().cast(),
                    call.workspace as *mut c_void,
                    call.workspace_bytes,
                    call.stream as *mut c_void,
                )
            },
            "SerialRowsMatmul",
        )?;
        if let (Some(receipts), Some(receipt)) = (&mut self.receipts, receipt) {
            receipts.push(receipt);
        }
        Ok(())
    }

    fn destroy(&mut self, kind: ResourceKind, handle: Self::Handle) -> Result<()> {
        chk(destroy_raw(kind, handle), "SerialRowsResourceDestroy")
    }
}
