// SPDX-License-Identifier: AGPL-3.0-only

//! Raw device ownership for the admitted GLM-5.3 EXL3 checkpoint.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{Context, anyhow};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::glm53_exl3::{Glm53Exl3Dtype, Glm53Exl3Files};

const COPY_WINDOW_BYTES: usize = 64 * 1024 * 1024;
const DEVICE_TENSOR_ALIGNMENT: usize = 256;

#[derive(Debug, PartialEq, Eq)]
pub struct Glm53Exl3DeviceTensor {
    pub ptr: DevicePtr,
    pub shape: Vec<u64>,
    pub dtype: Glm53Exl3Dtype,
    pub byte_len: usize,
    pub shard: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct DeviceSlab {
    ptr: DevicePtr,
    bytes: usize,
    payload_bytes: usize,
}

#[derive(Debug)]
struct PackedTensorRoute {
    name: String,
    device_offset: usize,
    byte_len: usize,
}

#[derive(Debug)]
struct PackedShardLayout {
    allocation_bytes: usize,
    payload_bytes: usize,
    routes: Vec<PackedTensorRoute>,
}

#[must_use = "EXL3 device allocations must be explicitly freed or handed to an owner"]
#[derive(Debug)]
pub struct Glm53Exl3DeviceStore {
    tensors: BTreeMap<String, Glm53Exl3DeviceTensor>,
    slabs: BTreeMap<usize, DeviceSlab>,
    allocated_bytes: usize,
    payload_bytes: usize,
}

#[must_use = "failed EXL3 frees retain allocations and must be retried"]
#[derive(Debug)]
pub struct Glm53Exl3DeviceStoreFreeError {
    store: Glm53Exl3DeviceStore,
    first_error: anyhow::Error,
    failed_slabs: usize,
}

#[must_use = "failed EXL3 loads may retain allocations and must be inspected"]
pub struct Glm53Exl3DeviceLoadError {
    primary: anyhow::Error,
    cleanup: Option<Glm53Exl3DeviceStoreFreeError>,
}

impl Glm53Exl3DeviceStore {
    pub fn get(&self, name: &str) -> Option<&Glm53Exl3DeviceTensor> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn slab_count(&self) -> usize {
        self.slabs.len()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    pub fn free(
        mut self,
        gpu: &dyn GpuBackend,
    ) -> std::result::Result<(), Glm53Exl3DeviceStoreFreeError> {
        let shard_indices = self.slabs.keys().copied().collect::<Vec<_>>();
        let mut first_error = None;
        let mut failed_slabs = 0usize;
        for shard in shard_indices {
            let slab = self
                .slabs
                .get(&shard)
                .expect("collected EXL3 slab must remain owned");
            match gpu.free(slab.ptr) {
                Ok(()) => {
                    let released = self
                        .slabs
                        .remove(&shard)
                        .expect("freed EXL3 slab must remain owned");
                    self.allocated_bytes = self
                        .allocated_bytes
                        .checked_sub(released.bytes)
                        .expect("EXL3 allocation accounting must remain valid");
                    self.payload_bytes = self
                        .payload_bytes
                        .checked_sub(released.payload_bytes)
                        .expect("EXL3 payload accounting must remain valid");
                    self.tensors.retain(|_, tensor| tensor.shard != shard);
                }
                Err(error) => {
                    failed_slabs += 1;
                    first_error.get_or_insert(error);
                }
            }
        }
        match first_error {
            None => Ok(()),
            Some(first_error) => Err(Glm53Exl3DeviceStoreFreeError {
                store: self,
                first_error,
                failed_slabs,
            }),
        }
    }
}

impl Glm53Exl3DeviceStoreFreeError {
    pub fn failure(&self) -> &anyhow::Error {
        &self.first_error
    }

    pub fn failed_slab_count(&self) -> usize {
        self.failed_slabs
    }

    pub fn store(&self) -> &Glm53Exl3DeviceStore {
        &self.store
    }

    pub fn retry(
        self,
        gpu: &dyn GpuBackend,
    ) -> std::result::Result<(), Glm53Exl3DeviceStoreFreeError> {
        self.store.free(gpu)
    }
}

impl Glm53Exl3DeviceLoadError {
    fn new(primary: anyhow::Error) -> Self {
        Self {
            primary,
            cleanup: None,
        }
    }

    pub fn failure(&self) -> &anyhow::Error {
        &self.primary
    }

    pub fn retained_store(&self) -> Option<&Glm53Exl3DeviceStore> {
        self.cleanup.as_ref().map(|failure| failure.store())
    }

    pub fn retry_cleanup(
        self,
        gpu: &dyn GpuBackend,
    ) -> std::result::Result<anyhow::Error, Glm53Exl3DeviceLoadError> {
        let Some(cleanup) = self.cleanup else {
            return Ok(self.primary);
        };
        match cleanup.retry(gpu) {
            Ok(()) => Ok(self.primary),
            Err(cleanup) => Err(Self {
                primary: self.primary,
                cleanup: Some(cleanup),
            }),
        }
    }
}

impl From<anyhow::Error> for Glm53Exl3DeviceLoadError {
    fn from(primary: anyhow::Error) -> Self {
        Self::new(primary)
    }
}

impl fmt::Debug for Glm53Exl3DeviceLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53Exl3DeviceLoadError")
            .field(
                "retained_bytes",
                &self.retained_store().map(|s| s.allocated_bytes()),
            )
            .field("primary", &self.primary)
            .finish()
    }
}

impl fmt::Display for Glm53Exl3DeviceLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "GLM EXL3 device load failed: {:#}", self.primary)
    }
}

fn rollback(
    primary: anyhow::Error,
    store: Glm53Exl3DeviceStore,
    gpu: &dyn GpuBackend,
) -> Glm53Exl3DeviceLoadError {
    let mut error = Glm53Exl3DeviceLoadError::new(primary);
    if !store.slabs.is_empty() {
        error.cleanup = store.free(gpu).err();
    }
    error
}

/// Repack each shard into one slab with every tensor start device-aligned.
pub fn load_glm53_exl3_store(
    files: &Glm53Exl3Files,
    gpu: &dyn GpuBackend,
    reserve_bytes: usize,
) -> std::result::Result<Glm53Exl3DeviceStore, Glm53Exl3DeviceLoadError> {
    let expected_payload = usize::try_from(files.admission.data_bytes)
        .context("GLM EXL3 admitted payload is too large to address")?;
    let layouts = build_packed_layout(files)?;
    let expected_allocation = layouts.iter().try_fold(0usize, |sum, layout| {
        sum.checked_add(layout.allocation_bytes)
            .context("GLM EXL3 packed allocation count overflow")
    })?;
    let required = expected_allocation
        .checked_add(reserve_bytes)
        .context("GLM EXL3 packed allocation plus reserve overflows address space")?;
    let free = gpu.free_memory()?;
    if required > free {
        return Err(Glm53Exl3DeviceLoadError::new(anyhow!(
            "GLM EXL3 preflight requires {expected_allocation} packed bytes for {expected_payload} payload bytes plus {reserve_bytes} reserve bytes, but only {free} bytes are free"
        )));
    }

    if layouts
        .iter()
        .try_fold(0usize, |sum, layout| sum.checked_add(layout.payload_bytes))
        != Some(expected_payload)
    {
        return Err(Glm53Exl3DeviceLoadError::new(anyhow!(
            "GLM EXL3 shard payloads do not match admitted byte census"
        )));
    }

    let mut store = Glm53Exl3DeviceStore {
        tensors: BTreeMap::new(),
        slabs: BTreeMap::new(),
        allocated_bytes: 0,
        payload_bytes: 0,
    };
    for (shard_index, (shard, layout)) in files.shards.iter().zip(&layouts).enumerate() {
        let ptr = match gpu.alloc(layout.allocation_bytes) {
            Ok(ptr) => ptr,
            Err(error) => return Err(rollback(error, store, gpu)),
        };
        if ptr.0 % u64::try_from(DEVICE_TENSOR_ALIGNMENT).expect("alignment fits u64") != 0 {
            return Err(rollback(
                anyhow!("GLM EXL3 device allocation is not 256-byte aligned"),
                store,
                gpu,
            ));
        }
        store.slabs.insert(
            shard_index,
            DeviceSlab {
                ptr,
                bytes: layout.allocation_bytes,
                payload_bytes: layout.payload_bytes,
            },
        );
        store.allocated_bytes = match store.allocated_bytes.checked_add(layout.allocation_bytes) {
            Some(bytes) => bytes,
            None => {
                return Err(rollback(
                    anyhow!("EXL3 allocation count overflow"),
                    store,
                    gpu,
                ));
            }
        };
        store.payload_bytes = match store.payload_bytes.checked_add(layout.payload_bytes) {
            Some(bytes) => bytes,
            None => return Err(rollback(anyhow!("EXL3 payload count overflow"), store, gpu)),
        };

        let path = files.root.join(&shard.file_name);
        if let Err(error) = copy_packed_shard(&path, shard.data_start, layout, ptr, gpu) {
            return Err(rollback(error, store, gpu));
        }
    }

    if let Err(error) = install_views(&mut store, files, &layouts) {
        return Err(rollback(error, store, gpu));
    }
    if store.allocated_bytes != expected_allocation
        || store.payload_bytes != expected_payload
        || store.tensors.len() != files.admission.tensor_count
    {
        return Err(rollback(
            anyhow!("GLM EXL3 loaded store does not match admitted census"),
            store,
            gpu,
        ));
    }
    Ok(store)
}

fn build_packed_layout(files: &Glm53Exl3Files) -> anyhow::Result<Vec<PackedShardLayout>> {
    let mut by_shard = (0..files.shards.len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    for (name, info) in &files.tensors {
        by_shard
            .get_mut(info.shard)
            .context("GLM EXL3 tensor route has no admitted shard")?
            .push((name, info));
    }

    let mut layouts = Vec::with_capacity(files.shards.len());
    for (shard_index, mut tensors) in by_shard.into_iter().enumerate() {
        tensors.sort_by_key(|(_, info)| info.data_offset);
        let payload_bytes = usize::try_from(files.shards[shard_index].data_bytes)
            .context("GLM EXL3 shard payload is too large to address")?;
        let mut source_cursor = 0usize;
        let mut device_cursor = 0usize;
        let mut routes = Vec::with_capacity(tensors.len());
        for (name, info) in tensors {
            let source_offset = usize::try_from(info.data_offset)
                .context("GLM EXL3 tensor offset is too large to address")?;
            let byte_len = usize::try_from(info.byte_len)
                .context("GLM EXL3 tensor is too large to address")?;
            if source_offset != source_cursor {
                return Err(anyhow!(
                    "GLM EXL3 tensor payload has a gap or overlap in its admitted shard"
                ));
            }
            let source_end = source_offset
                .checked_add(byte_len)
                .context("GLM EXL3 tensor source extent overflow")?;
            if source_end > payload_bytes {
                return Err(anyhow!("GLM EXL3 tensor extent escapes its admitted shard"));
            }
            let device_offset = align_up(device_cursor, DEVICE_TENSOR_ALIGNMENT)?;
            device_cursor = device_offset
                .checked_add(byte_len)
                .context("GLM EXL3 packed tensor extent overflow")?;
            source_cursor = source_end;
            routes.push(PackedTensorRoute {
                name: name.clone(),
                device_offset,
                byte_len,
            });
        }
        if source_cursor != payload_bytes {
            return Err(anyhow!(
                "GLM EXL3 tensor extents do not cover the admitted shard payload"
            ));
        }
        layouts.push(PackedShardLayout {
            allocation_bytes: device_cursor,
            payload_bytes,
            routes,
        });
    }
    Ok(layouts)
}

fn align_up(value: usize, alignment: usize) -> anyhow::Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|padded| padded & !(alignment - 1))
        .context("GLM EXL3 tensor alignment overflow")
}

fn install_views(
    store: &mut Glm53Exl3DeviceStore,
    files: &Glm53Exl3Files,
    layouts: &[PackedShardLayout],
) -> anyhow::Result<()> {
    for (shard_index, layout) in layouts.iter().enumerate() {
        let slab = store
            .slabs
            .get(&shard_index)
            .context("GLM EXL3 tensor route has no device slab")?;
        for route in &layout.routes {
            let info = files
                .tensors
                .get(&route.name)
                .context("GLM EXL3 packed route lost its admitted tensor")?;
            let end = route
                .device_offset
                .checked_add(route.byte_len)
                .context("GLM EXL3 tensor device extent overflow")?;
            if end > slab.bytes {
                return Err(anyhow!("GLM EXL3 tensor view escapes its packed shard"));
            }
            let address = slab
                .ptr
                .0
                .checked_add(u64::try_from(route.device_offset)?)
                .context("GLM EXL3 device address overflow")?;
            if address % u64::try_from(DEVICE_TENSOR_ALIGNMENT)? != 0 {
                return Err(anyhow!("GLM EXL3 packed tensor address is misaligned"));
            }
            store.tensors.insert(
                route.name.clone(),
                Glm53Exl3DeviceTensor {
                    ptr: DevicePtr(address),
                    shape: info.shape.clone(),
                    dtype: info.dtype,
                    byte_len: route.byte_len,
                    shard: info.shard,
                },
            );
        }
    }
    Ok(())
}

fn copy_packed_shard(
    path: &std::path::Path,
    data_start: u64,
    layout: &PackedShardLayout,
    base: DevicePtr,
    gpu: &dyn GpuBackend,
) -> anyhow::Result<()> {
    copy_packed_shard_window(path, data_start, layout, base, gpu, COPY_WINDOW_BYTES)
}

fn copy_packed_shard_window(
    path: &std::path::Path,
    data_start: u64,
    layout: &PackedShardLayout,
    base: DevicePtr,
    gpu: &dyn GpuBackend,
    window_bytes: usize,
) -> anyhow::Result<()> {
    if window_bytes == 0 {
        return Err(anyhow!("GLM EXL3 packed-copy window must be nonzero"));
    }
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    file.seek(SeekFrom::Start(data_start))?;
    let scratch_bytes = layout.allocation_bytes.min(window_bytes).max(1);
    let mut scratch = vec![0u8; scratch_bytes];
    let mut packed_cursor = 0usize;
    let mut route_index = 0usize;
    let mut route_copied = 0usize;
    let mut payload_copied = 0usize;
    while packed_cursor < layout.allocation_bytes {
        let chunk = (layout.allocation_bytes - packed_cursor).min(scratch.len());
        scratch[..chunk].fill(0);
        let packed_end = packed_cursor
            .checked_add(chunk)
            .context("GLM EXL3 packed-copy window overflow")?;
        while let Some(route) = layout.routes.get(route_index) {
            if route_copied == route.byte_len {
                route_index += 1;
                route_copied = 0;
                continue;
            }
            let route_cursor = route
                .device_offset
                .checked_add(route_copied)
                .context("GLM EXL3 packed route cursor overflow")?;
            if route_cursor >= packed_end {
                break;
            }
            if route_cursor < packed_cursor {
                return Err(anyhow!("GLM EXL3 packed route order drift"));
            }
            let copy_bytes = (route.byte_len - route_copied).min(packed_end - route_cursor);
            let scratch_offset = route_cursor - packed_cursor;
            file.read_exact(&mut scratch[scratch_offset..scratch_offset + copy_bytes])?;
            route_copied += copy_bytes;
            payload_copied = payload_copied
                .checked_add(copy_bytes)
                .context("GLM EXL3 packed payload counter overflow")?;
        }
        let address = base
            .0
            .checked_add(u64::try_from(packed_cursor)?)
            .context("GLM EXL3 shard device address overflow")?;
        gpu.copy_h2d(&scratch[..chunk], DevicePtr(address))?;
        packed_cursor = packed_end;
    }
    while layout
        .routes
        .get(route_index)
        .is_some_and(|route| route_copied == route.byte_len)
    {
        route_index += 1;
        route_copied = 0;
    }
    if route_index != layout.routes.len()
        || route_copied != 0
        || payload_copied != layout.payload_bytes
    {
        return Err(anyhow!(
            "GLM EXL3 packed copy did not consume its exact layout"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_exl3_device_tests.rs"]
mod tests;
