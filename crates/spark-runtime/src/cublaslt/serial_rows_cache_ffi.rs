// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded, opt-in reuse of native strided-row cuBLASLt resources.
//!
//! Algorithms are deliberately not cached. Every call restores the scalar-M1
//! layouts, obtains a fresh heuristic result, applies the checked strided-M1
//! attributes, and runs AlgoCheck before submission.

use anyhow::{Result, anyhow, bail};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use super::serial_rows_contract::{ByteSpan, Orientation, SerialRowsPlan, SerialRowsRequest};
use super::serial_rows_driver::{DescriptorSet, ResourceKind, SerialRowsIo};
use super::serial_rows_ffi::{NativeRowsIo, destroy_raw};
use super::{chk, ctx};

const SELECTOR: &str = "ATLAS_CUBLASLT_SERIAL_RESOURCE_CACHE";
pub(super) const MAX_RESOURCE_SETS: usize = 64;
static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();
static CACHE: OnceLock<Mutex<ResourceCache>> = OnceLock::new();
static ENGAGEMENT_REPORTED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn cublasLtMatrixLayoutSetAttribute(
        layout: *mut c_void,
        attr: u32,
        value: *const c_void,
        bytes: usize,
    ) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ResourceKey {
    rows: u32,
    n: u32,
    k: u32,
    weight_is_nk: bool,
}

impl ResourceKey {
    pub(super) fn new(request: SerialRowsRequest) -> Result<Self, &'static str> {
        SerialRowsPlan::new(request)?;
        Ok(Self {
            rows: request.rows,
            n: request.n,
            k: request.k,
            weight_is_nk: request.orientation == Orientation::Nk,
        })
    }
}

pub(super) fn parse_setting(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_CUBLASLT_SERIAL_RESOURCE_CACHE must be exactly 0 or 1"),
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
pub(super) enum CacheSlot {
    Existing,
    Vacant,
}

#[derive(Default)]
pub(super) struct BoundedKeys(Vec<ResourceKey>);

impl BoundedKeys {
    pub(super) fn locate(&self, key: ResourceKey) -> Result<CacheSlot, &'static str> {
        if self.0.contains(&key) {
            return Ok(CacheSlot::Existing);
        }
        if self.0.len() == MAX_RESOURCE_SETS {
            return Err("native-row resource cache reached its fixed 64-entry bound");
        }
        Ok(CacheSlot::Vacant)
    }

    pub(super) fn record(&mut self, key: ResourceKey) -> Result<(), &'static str> {
        if self.locate(key)? == CacheSlot::Existing {
            return Err("native-row resource key was recorded twice");
        }
        self.0.push(key);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
}

struct CachedSet {
    desc: *mut c_void,
    a: *mut c_void,
    b: *mut c_void,
    d: *mut c_void,
    pref: *mut c_void,
}

// Every handle is created, used, and destroyed only while the process-global
// ResourceCache mutex is held. Ctx separately documents serialized execution.
unsafe impl Send for CachedSet {}

impl CachedSet {
    fn empty() -> Self {
        Self {
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            d: std::ptr::null_mut(),
            pref: std::ptr::null_mut(),
        }
    }

    fn create(io: &mut NativeRowsIo<'_>, plan: SerialRowsPlan) -> Result<Self> {
        let mut cached = Self::empty();
        cached.desc = io.create_matmul(68, 0)?;
        io.set_matmul_i32(cached.desc, 3, plan.trans_a())?;
        io.set_matmul_i32(cached.desc, 4, 0)?;

        let [layout_a, layout_b, layout_d] = plan.layouts();
        cached.a = io.create_layout(layout_a)?;
        cached.b = io.create_layout(layout_b)?;
        cached.d = io.create_layout(layout_d)?;
        cached.pref = io.create_preference()?;
        io.set_preference_usize(cached.pref, 1, super::serial_rows_contract::WORKSPACE_BYTES)?;
        Ok(cached)
    }

    fn descriptor_set(&self) -> DescriptorSet<*mut c_void> {
        DescriptorSet {
            desc: self.desc,
            a: self.a,
            b: self.b,
            c: self.d,
            d: self.d,
            pref: self.pref,
        }
    }

    fn restore_scalar_layouts(&self) -> Result<()> {
        let batch_count = 1i32;
        let stride = 0i64;
        for layout in [self.a, self.b, self.d] {
            chk(
                unsafe {
                    cublasLtMatrixLayoutSetAttribute(
                        layout,
                        5,
                        (&batch_count as *const i32).cast(),
                        size_of::<i32>(),
                    )
                },
                "SerialResourceCacheResetBatchCount",
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
                "SerialResourceCacheResetElementStride",
            )?;
        }
        Ok(())
    }
}

impl Drop for CachedSet {
    fn drop(&mut self) {
        for (kind, handle) in [
            (ResourceKind::Preference, self.pref),
            (ResourceKind::Layout, self.d),
            (ResourceKind::Layout, self.b),
            (ResourceKind::Layout, self.a),
            (ResourceKind::Descriptor, self.desc),
        ] {
            if !handle.is_null() {
                let _ = destroy_raw(kind, handle);
            }
        }
    }
}

#[derive(Default)]
struct ResourceCache {
    keys: BoundedKeys,
    sets: BTreeMap<ResourceKey, CachedSet>,
}

pub(super) fn execute(request: SerialRowsRequest) -> Result<()> {
    // Preserve the cold-context admission boundary.
    let plan = SerialRowsPlan::new(request).map_err(anyhow::Error::msg)?;
    let key = ResourceKey::new(request).map_err(anyhow::Error::msg)?;
    let context = ctx()?;
    let workspace = ByteSpan {
        address: context.workspace,
        bytes: context.ws_size,
    };
    let bound = plan.bind_workspace(workspace).map_err(anyhow::Error::msg)?;
    let call = bound.row(0).map_err(anyhow::Error::msg)?;
    let mut io = NativeRowsIo {
        context,
        receipts: None,
        dims: [1, request.n, request.k],
    };

    let cache = CACHE.get_or_init(|| Mutex::new(ResourceCache::default()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow!("native-row resource cache mutex is poisoned"))?;
    let existing = cache.keys.locate(key).map_err(anyhow::Error::msg)? == CacheSlot::Existing;
    if !existing {
        let set = CachedSet::create(&mut io, plan)?;
        cache.keys.record(key).map_err(anyhow::Error::msg)?;
        if cache.sets.insert(key, set).is_some() {
            bail!("native-row resource cache replaced an existing owner");
        }
    }
    let entry_count = cache.sets.len();
    let set = cache
        .sets
        .get(&key)
        .ok_or_else(|| anyhow!("native-row resource cache index and owner map diverged"))?;
    set.restore_scalar_layouts()?;
    let descriptors = set.descriptor_set();
    let (result, returned) = io.heuristic(descriptors, 1)?;
    result
        .admit(returned, call.workspace_bytes)
        .map_err(anyhow::Error::msg)?;
    if !io.prepare_strided_m1(descriptors, request, call, &result)? {
        bail!("cached native-row strided M1 was not admitted");
    }
    io.matmul(descriptors, call, &result)?;

    if existing && !ENGAGEMENT_REPORTED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "CUBLASLT_SERIAL_RESOURCE_CACHE_ENGAGED entries={entry_count} max={MAX_RESOURCE_SETS}"
        );
    }
    Ok(())
}
