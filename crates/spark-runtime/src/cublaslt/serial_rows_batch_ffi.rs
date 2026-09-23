// SPDX-License-Identifier: AGPL-3.0-only
//! Checked strided-M1 preparation inside the existing scoped resource owner.
use super::super::strided_rows_contract::admit_strided_m1;
use super::{DescriptorSet, HeuristicResult, NativeRowsIo, RowCall, SerialRowsRequest, chk};
use anyhow::{Result, ensure};
use std::ffi::c_void;
use std::mem::size_of;

unsafe extern "C" {
    fn cublasLtMatmulAlgoCapGetAttribute(
        algo: *const c_void,
        attr: u32,
        value: *mut c_void,
        bytes: usize,
        written: *mut usize,
    ) -> i32;
    fn cublasLtMatrixLayoutSetAttribute(
        layout: *mut c_void,
        attr: u32,
        value: *const c_void,
        bytes: usize,
    ) -> i32;
    fn cublasLtMatmulAlgoCheck(
        handle: *mut c_void,
        desc: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
        d: *mut c_void,
        algo: *const c_void,
        result: *mut HeuristicResult,
    ) -> i32;
}

fn capability(result: &HeuristicResult, attr: u32) -> Result<u32> {
    let mut value = 0u32;
    let mut written = 0usize;
    chk(
        unsafe {
            cublasLtMatmulAlgoCapGetAttribute(
                result.algo.as_ptr().cast(),
                attr,
                (&mut value as *mut u32).cast(),
                size_of::<u32>(),
                &mut written,
            )
        },
        "StridedM1AlgoCapability",
    )?;
    ensure!(
        written == size_of::<u32>(),
        "strided M1 capability ABI size drift"
    );
    Ok(value)
}

pub(super) fn prepare(
    io: &NativeRowsIo<'_>,
    set: DescriptorSet<*mut c_void>,
    request: SerialRowsRequest,
    call: RowCall,
    result: &HeuristicResult,
) -> Result<bool> {
    // Installed CUDA13: strided support=3, minimum A/B/C/D alignment=16..19.
    let caps = [
        capability(result, 3)?,
        capability(result, 16)?,
        capability(result, 17)?,
        capability(result, 18)?,
        capability(result, 19)?,
    ];
    let strides = admit_strided_m1(request, call, caps).map_err(anyhow::Error::msg)?;
    ensure!(
        set.c == set.d,
        "strided M1 requires the original aliased C/D layout"
    );
    let rows = i32::try_from(request.rows)?;
    for (layout, stride) in [set.a, set.b, set.d].into_iter().zip(strides) {
        // batch_count=5 (i32), strided_batch_offset=6 (i64 elements).
        chk(
            unsafe {
                cublasLtMatrixLayoutSetAttribute(
                    layout,
                    5,
                    (&rows as *const i32).cast(),
                    size_of::<i32>(),
                )
            },
            "StridedM1BatchCount",
        )?;
        chk(
            unsafe {
                cublasLtMatrixLayoutSetAttribute(
                    layout,
                    6,
                    (&stride as *const i64).cast(),
                    size_of::<i64>(),
                )
            },
            "StridedM1ElementStride",
        )?;
    }
    let mut checked = HeuristicResult::default();
    chk(
        unsafe {
            cublasLtMatmulAlgoCheck(
                io.context.handle,
                set.desc,
                set.a,
                set.b,
                set.c,
                set.d,
                result.algo.as_ptr().cast(),
                &mut checked,
            )
        },
        "StridedM1AlgoCheck",
    )?;
    checked
        .admit(1, call.workspace_bytes)
        .map_err(anyhow::Error::msg)?;
    // AlgoCheck does not update result.algo. The caller submits all original
    // eight words, not a reconstructed ID or an alternate heuristic.
    Ok(true)
}
