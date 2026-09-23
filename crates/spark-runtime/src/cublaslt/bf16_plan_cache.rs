// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded ownership for immutable BF16 cuBLASLt plans.

use anyhow::{Result, bail};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use super::diagnostic_contract::{HeuristicResult, ReductionPolicy};
use super::{
    Bf16GemmReceipt, CUBLAS_COMPUTE_32F, CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16BF, CUDA_R_32F, Ctx,
    DESC_TRANSA, DESC_TRANSB, PREF_MAX_WORKSPACE_BYTES, chk, cublasLtMatmul,
    cublasLtMatmulAlgoGetHeuristic, cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy,
    cublasLtMatmulDescSetAttribute, cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceSetAttribute,
    cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy, diagnostic,
};

pub(super) const MAX_PLANS: usize = 64;
const SELECTOR: &str = "ATLAS_CUBLASLT_BF16_PLAN_CACHE";
static ENGAGEMENT_REPORTED: AtomicBool = AtomicBool::new(false);
static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct GemmKey {
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) weight_is_nk: bool,
}

impl GemmKey {
    pub(super) fn new(m: u32, n: u32, k: u32, weight_is_nk: bool) -> Result<Self, &'static str> {
        if m == 0 || n == 0 || k == 0 {
            return Err("BF16 plan dimensions must be non-zero");
        }
        Ok(Self {
            m,
            n,
            k,
            weight_is_nk,
        })
    }
}

pub(super) fn parse_setting(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_CUBLASLT_BF16_PLAN_CACHE must be exactly 0 or 1"),
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
        Ok(enabled) => Ok(*enabled),
        Err(message) => bail!(message.clone()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KeySlot {
    Existing,
    Vacant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PlanRoute {
    Ephemeral,
    Cached,
}

pub(super) fn route(policy: Option<ReductionPolicy>, cache_enabled: bool) -> PlanRoute {
    if policy.is_none() && cache_enabled {
        PlanRoute::Cached
    } else {
        PlanRoute::Ephemeral
    }
}

#[derive(Default)]
pub(super) struct BoundedKeys(Vec<GemmKey>);

impl BoundedKeys {
    pub(super) fn locate(&self, key: GemmKey) -> Result<KeySlot, &'static str> {
        if self.0.contains(&key) {
            return Ok(KeySlot::Existing);
        }
        if self.0.len() == MAX_PLANS {
            return Err("BF16 cuBLASLt plan cache reached its fixed 64-plan bound");
        }
        Ok(KeySlot::Vacant)
    }

    pub(super) fn record(&mut self, key: GemmKey) -> Result<(), &'static str> {
        if self.locate(key)? == KeySlot::Existing {
            return Err("BF16 cuBLASLt plan key was recorded twice");
        }
        self.0.push(key);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Default)]
pub(super) struct PlanCache {
    keys: BoundedKeys,
    algorithms: BTreeMap<GemmKey, HeuristicResult>,
}

struct Plan {
    desc: *mut c_void,
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    heuristic: HeuristicResult,
}

impl Plan {
    fn empty() -> Self {
        Self {
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            d: std::ptr::null_mut(),
            heuristic: HeuristicResult::default(),
        }
    }

    fn create(
        context: &Ctx,
        key: GemmKey,
        policy: Option<ReductionPolicy>,
        selected: Option<HeuristicResult>,
    ) -> Result<Self> {
        let mut plan = Self::empty();
        let mut pref = Preference::create()?;
        plan.desc = created(
            Resource::Descriptor,
            unsafe { cublasLtMatmulDescCreate(&mut plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) },
            plan.desc,
            "DescCreate",
        )?;
        let trans_a = if key.weight_is_nk {
            CUBLAS_OP_T
        } else {
            CUBLAS_OP_N
        };
        set_desc(plan.desc, DESC_TRANSA, trans_a, "TRANSA")?;
        set_desc(plan.desc, DESC_TRANSB, CUBLAS_OP_N, "TRANSB")?;

        let (weight_rows, weight_cols, weight_ld) = if key.weight_is_nk {
            (key.k as u64, key.n as u64, key.k as i64)
        } else {
            (key.n as u64, key.k as u64, key.n as i64)
        };
        plan.a = create_layout(weight_rows, weight_cols, weight_ld, "LayoutA")?;
        plan.b = create_layout(key.k as u64, key.m as u64, key.k as i64, "LayoutB")?;
        plan.d = create_layout(key.n as u64, key.m as u64, key.n as i64, "LayoutD")?;
        set_workspace(pref.0, context.ws_size)?;
        if let Some(selected) = selected {
            anyhow::ensure!(
                policy.is_none(),
                "diagnostic policy cannot reuse an algorithm"
            );
            plan.heuristic = selected;
            plan.heuristic
                .admit(1, context.ws_size)
                .map_err(anyhow::Error::msg)?;
        } else {
            if let Some(policy) = policy {
                diagnostic::configure(pref.0, policy)?;
            }
            let mut returned = 0;
            chk(
                unsafe {
                    cublasLtMatmulAlgoGetHeuristic(
                        context.handle,
                        plan.desc,
                        plan.a,
                        plan.b,
                        plan.d,
                        plan.d,
                        pref.0,
                        1,
                        (&mut plan.heuristic as *mut HeuristicResult).cast(),
                        &mut returned,
                    )
                },
                "AlgoGetHeuristic",
            )?;
            plan.heuristic
                .admit(returned, context.ws_size)
                .map_err(anyhow::Error::msg)?;
        }
        pref.close()?;
        Ok(plan)
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul(&self, context: &Ctx, act: u64, weight: u64, out: u64, stream: u64) -> Result<()> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        chk(
            unsafe {
                cublasLtMatmul(
                    context.handle,
                    self.desc,
                    (&alpha as *const f32).cast(),
                    weight as *const c_void,
                    self.a,
                    act as *const c_void,
                    self.b,
                    (&beta as *const f32).cast(),
                    out as *const c_void,
                    self.d,
                    out as *mut c_void,
                    self.d,
                    self.heuristic.algo.as_ptr().cast(),
                    context.workspace as *mut c_void,
                    context.ws_size,
                    stream as *mut c_void,
                )
            },
            "Matmul",
        )
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            if !self.d.is_null() {
                cublasLtMatrixLayoutDestroy(self.d);
            }
            if !self.b.is_null() {
                cublasLtMatrixLayoutDestroy(self.b);
            }
            if !self.a.is_null() {
                cublasLtMatrixLayoutDestroy(self.a);
            }
            if !self.desc.is_null() {
                cublasLtMatmulDescDestroy(self.desc);
            }
        }
    }
}

struct Preference(*mut c_void);

impl Preference {
    fn create() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { cublasLtMatmulPreferenceCreate(&mut handle) };
        Ok(Self(created(
            Resource::Preference,
            status,
            handle,
            "PrefCreate",
        )?))
    }

    fn close(&mut self) -> Result<()> {
        if self.0.is_null() {
            return Ok(());
        }
        let handle = std::mem::replace(&mut self.0, std::ptr::null_mut());
        chk(
            unsafe { cublasLtMatmulPreferenceDestroy(handle) },
            "PrefDestroy",
        )
    }
}

impl Drop for Preference {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { cublasLtMatmulPreferenceDestroy(self.0) };
        }
    }
}

#[derive(Clone, Copy)]
enum Resource {
    Descriptor,
    Layout,
    Preference,
}

fn created(
    resource: Resource,
    status: i32,
    handle: *mut c_void,
    name: &str,
) -> Result<*mut c_void> {
    if status == 0 && !handle.is_null() {
        return Ok(handle);
    }
    let cleanup = if handle.is_null() {
        None
    } else {
        Some(unsafe {
            match resource {
                Resource::Descriptor => cublasLtMatmulDescDestroy(handle),
                Resource::Layout => cublasLtMatrixLayoutDestroy(handle),
                Resource::Preference => cublasLtMatmulPreferenceDestroy(handle),
            }
        })
    };
    bail!("cuBLASLt {name} failed: status {status}, cleanup={cleanup:?}")
}

fn set_desc(desc: *mut c_void, attr: u32, value: i32, name: &str) -> Result<()> {
    chk(
        unsafe {
            cublasLtMatmulDescSetAttribute(
                desc,
                attr,
                (&value as *const i32).cast(),
                size_of::<i32>(),
            )
        },
        name,
    )
}

fn create_layout(rows: u64, cols: u64, ld: i64, name: &str) -> Result<*mut c_void> {
    let mut handle = std::ptr::null_mut();
    let status = unsafe { cublasLtMatrixLayoutCreate(&mut handle, CUDA_R_16BF, rows, cols, ld) };
    created(Resource::Layout, status, handle, name)
}

fn set_workspace(pref: *mut c_void, bytes: usize) -> Result<()> {
    chk(
        unsafe {
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                (&bytes as *const usize).cast(),
                size_of::<usize>(),
            )
        },
        "PrefWorkspace",
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_ephemeral(
    context: &Ctx,
    key: GemmKey,
    act: u64,
    weight: u64,
    out: u64,
    stream: u64,
    policy: Option<ReductionPolicy>,
) -> Result<Option<Bf16GemmReceipt>> {
    let plan = Plan::create(context, key, policy, None)?;
    let receipt = policy
        .map(|policy| {
            diagnostic::receipt(
                &plan.heuristic,
                policy,
                [key.m, key.n, key.k],
                context.ws_size,
            )
        })
        .transpose()?;
    plan.matmul(context, act, weight, out, stream)?;
    Ok(receipt)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_cached(
    context: &Ctx,
    key: GemmKey,
    act: u64,
    weight: u64,
    out: u64,
    stream: u64,
) -> Result<()> {
    let mut cache = context
        .plans
        .lock()
        .map_err(|_| anyhow::anyhow!("BF16 cuBLASLt plan cache lock poisoned"))?;
    let slot = cache.keys.locate(key).map_err(anyhow::Error::msg)?;
    let selected = cache.algorithms.get(&key).copied();
    if slot == KeySlot::Existing && selected.is_none() {
        bail!("BF16 cuBLASLt plan index/map divergence");
    }
    let plan = Plan::create(context, key, None, selected)?;
    plan.matmul(context, act, weight, out, stream)?;
    if slot == KeySlot::Vacant {
        cache.keys.record(key).map_err(anyhow::Error::msg)?;
        cache.algorithms.insert(key, plan.heuristic);
    } else if !ENGAGEMENT_REPORTED.swap(true, Ordering::Relaxed) {
        tracing::info!(
            algorithms = cache.algorithms.len(),
            max_plans = MAX_PLANS,
            "CUBLASLT_BF16_PLAN_CACHE_ENGAGED"
        );
    }
    Ok(())
}
