// SPDX-License-Identifier: AGPL-3.0-only
//! Default-off classic cuBLAS BF16 adapter with no per-call resource creation.

use anyhow::{Result, bail, ensure};
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use super::bf16_plan_cache::GemmKey;
use super::serial_rows_contract::{ByteSpan, Orientation, SerialRowsPlan, SerialRowsRequest};
use super::{CUBLAS_COMPUTE_32F, CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16BF, Ctx, ctx};

type Handle = *mut c_void;
const SELECTOR: &str = "ATLAS_CUBLAS_BF16_GEMM";
const CUBLAS_GEMM_DEFAULT_TENSOR_OP: i32 = 99;

unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut Handle) -> i32;
    fn cublasSetStream_v2(handle: Handle, stream: *mut c_void) -> i32;
    fn cublasSetWorkspace_v2(handle: Handle, workspace: *mut c_void, bytes: usize) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasGemmEx(
        handle: Handle,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: i32,
        lda: i32,
        b: *const c_void,
        b_type: i32,
        ldb: i32,
        beta: *const c_void,
        c: *mut c_void,
        c_type: i32,
        ldc: i32,
        compute_type: i32,
        algorithm: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasGemmStridedBatchedEx(
        handle: Handle,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: i32,
        lda: i32,
        stride_a: i64,
        b: *const c_void,
        b_type: i32,
        ldb: i32,
        stride_b: i64,
        beta: *const c_void,
        c: *mut c_void,
        c_type: i32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
        compute_type: i32,
        algorithm: i32,
    ) -> i32;
}

struct ClassicHandle(Handle);
unsafe impl Send for ClassicHandle {}

static HANDLE: OnceLock<Result<Mutex<ClassicHandle>, String>> = OnceLock::new();
static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();

pub(super) fn parse_setting(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_CUBLAS_BF16_GEMM must be exactly 0 or 1"),
    }
}

pub(super) fn enabled() -> Result<bool> {
    match ENABLED.get_or_init(|| match std::env::var(SELECTOR) {
        Ok(value) => parse_setting(Some(&value)).map_err(str::to_owned),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{SELECTOR} must contain valid Unicode and be exactly 0 or 1"
        )),
    }) {
        Ok(value) => Ok(*value),
        Err(message) => bail!(message.clone()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GemmPlan {
    pub(super) trans_a: i32,
    pub(super) m: i32,
    pub(super) n: i32,
    pub(super) k: i32,
    pub(super) lda: i32,
    pub(super) ldb: i32,
    pub(super) ldc: i32,
}

impl GemmPlan {
    pub(super) fn new(key: GemmKey) -> Result<Self, &'static str> {
        let m = i32::try_from(key.m).map_err(|_| "classic cuBLAS M exceeds i32")?;
        let n = i32::try_from(key.n).map_err(|_| "classic cuBLAS N exceeds i32")?;
        let k = i32::try_from(key.k).map_err(|_| "classic cuBLAS K exceeds i32")?;
        Ok(Self {
            trans_a: if key.weight_is_nk {
                CUBLAS_OP_T
            } else {
                CUBLAS_OP_N
            },
            m: n,
            n: m,
            k,
            lda: if key.weight_is_nk { k } else { n },
            ldb: k,
            ldc: n,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BatchedPlan {
    pub(super) gemm: GemmPlan,
    pub(super) stride_weight: i64,
    pub(super) stride_act: i64,
    pub(super) stride_out: i64,
    pub(super) batch_count: i32,
}

impl BatchedPlan {
    pub(super) fn new(request: SerialRowsRequest) -> Result<Self, &'static str> {
        SerialRowsPlan::new(request)?;
        let weight_is_nk = request.orientation == Orientation::Nk;
        Ok(Self {
            gemm: GemmPlan::new(GemmKey::new(1, request.n, request.k, weight_is_nk)?)?,
            stride_weight: 0,
            stride_act: i64::from(request.k),
            stride_out: i64::from(request.n),
            batch_count: i32::try_from(request.rows)
                .map_err(|_| "classic cuBLAS batch count exceeds i32")?,
        })
    }
}

fn handle() -> Result<&'static Mutex<ClassicHandle>> {
    match HANDLE.get_or_init(|| {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut handle) };
        if status != 0 || handle.is_null() {
            Err(format!(
                "cublasCreate_v2 failed: status {status}, null={}",
                handle.is_null()
            ))
        } else {
            Ok(Mutex::new(ClassicHandle(handle)))
        }
    }) {
        Ok(handle) => Ok(handle),
        Err(message) => bail!(message.clone()),
    }
}

fn check(status: i32, operation: &str) -> Result<()> {
    ensure!(
        status == 0,
        "classic cuBLAS {operation} failed: status {status}"
    );
    Ok(())
}

fn bind_handle(
    context: &Ctx,
    stream: u64,
) -> Result<std::sync::MutexGuard<'static, ClassicHandle>> {
    let guard = handle()?
        .lock()
        .map_err(|_| anyhow::anyhow!("classic cuBLAS handle lock poisoned"))?;
    check(
        unsafe { cublasSetStream_v2(guard.0, stream as *mut c_void) },
        "SetStream",
    )?;
    // cublasSetStream resets the handle workspace.
    check(
        unsafe {
            cublasSetWorkspace_v2(guard.0, context.workspace as *mut c_void, context.ws_size)
        },
        "SetWorkspace",
    )?;
    Ok(guard)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute(
    context: &Ctx,
    key: GemmKey,
    act: u64,
    weight: u64,
    out: u64,
    stream: u64,
) -> Result<()> {
    ensure!(
        act != 0 && weight != 0 && out != 0,
        "classic cuBLAS received a null operand"
    );
    let plan = GemmPlan::new(key).map_err(anyhow::Error::msg)?;
    let guard = bind_handle(context, stream)?;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    check(
        unsafe {
            cublasGemmEx(
                guard.0,
                plan.trans_a,
                CUBLAS_OP_N,
                plan.m,
                plan.n,
                plan.k,
                (&alpha as *const f32).cast(),
                weight as *const c_void,
                CUDA_R_16BF,
                plan.lda,
                act as *const c_void,
                CUDA_R_16BF,
                plan.ldb,
                (&beta as *const f32).cast(),
                out as *mut c_void,
                CUDA_R_16BF,
                plan.ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        },
        "GemmEx",
    )
}

pub(super) fn execute_batched(request: SerialRowsRequest) -> Result<()> {
    let plan = BatchedPlan::new(request).map_err(anyhow::Error::msg)?;
    // Reject malformed spans before initializing the cold CUDA context.
    let context = ctx()?;
    SerialRowsPlan::new(request)
        .map_err(anyhow::Error::msg)?
        .bind_workspace(ByteSpan {
            address: context.workspace,
            bytes: context.ws_size,
        })
        .map_err(anyhow::Error::msg)?;
    let guard = bind_handle(context, request.stream)?;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    check(
        unsafe {
            cublasGemmStridedBatchedEx(
                guard.0,
                plan.gemm.trans_a,
                CUBLAS_OP_N,
                plan.gemm.m,
                plan.gemm.n,
                plan.gemm.k,
                (&alpha as *const f32).cast(),
                request.weight.address as *const c_void,
                CUDA_R_16BF,
                plan.gemm.lda,
                plan.stride_weight,
                request.act.address as *const c_void,
                CUDA_R_16BF,
                plan.gemm.ldb,
                plan.stride_act,
                (&beta as *const f32).cast(),
                request.out.address as *mut c_void,
                CUDA_R_16BF,
                plan.gemm.ldc,
                plan.stride_out,
                plan.batch_count,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        },
        "GemmStridedBatchedEx",
    )
}
