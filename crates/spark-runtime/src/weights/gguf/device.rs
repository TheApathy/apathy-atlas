// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, anyhow};
use std::{collections::BTreeMap, fmt};

use crate::gpu::{DevicePtr, GpuBackend};

use super::{GgmlType, Glm53GgufFiles, Glm53Iq3Files, Glm53QuantProfile};

const COPY_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// One admitted GGUF tensor copied byte-for-byte to device memory.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "device allocation must be explicitly freed or moved into a store"]
pub struct GgufDeviceTensor {
    pub ptr: DevicePtr,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Logical payload: the exact GGUF byte length, and the extent every
    /// consumer's shape check compares against.
    pub byte_len: usize,
    /// What was actually handed to `gpu.alloc`, which is `byte_len` plus the
    /// MMQ tail slack from [`mmq_tensor_alloc_bytes`]. Recorded rather than
    /// re-derived: a derived value could not catch a loader that forgot the
    /// slack, which is the whole point of tracking it. Residency accounting
    /// sums THIS, never `byte_len` -- an allocation that grows without
    /// preflight seeing it is how a GB10 host gets killed.
    ///
    /// SETTING THIS IN A TEST FIXTURE: derive it, never write a literal --
    ///
    ///     alloc_bytes: mmq_tensor_alloc_bytes(kind, dims, byte_len)?
    ///
    /// A fixture that hardcodes `byte_len` (or a number that happens to match
    /// today) silently stops modelling the allocation the loader really makes,
    /// and `require_mmq_tail_slack` in spark-model then either fires on a
    /// correct loader or, worse, passes on a broken one. Deriving it also means
    /// the fixture keeps telling the truth if the tensor's dtype or K changes.
    /// Adding this field to a fixture is per-site work: the surrounding helpers
    /// name their locals differently (`kind` vs `ggml_type`, `dims` vs
    /// `dimensions`, and payloads called `per`, `plan.weight_bytes` or
    /// `(elements * 4) as usize`), so a find/replace across call sites does not
    /// compile at any of them.
    pub alloc_bytes: usize,
}

/// Bytes to allocate for one GGUF tensor: its payload plus the MMQ tail slack
/// its shape requires.
///
/// The slack rule is [`GgmlType::mmq_weight_tail_slack_bytes`]; K is the first
/// GGUF dimension. Only rank-2 tensors can reach the MMQ weight path, so
/// nothing else is widened.
pub fn mmq_tensor_alloc_bytes(
    ggml_type: GgmlType,
    dimensions: &[u64],
    byte_len: usize,
) -> anyhow::Result<usize> {
    let slack = match dimensions {
        // K is the first GGUF dimension for both a [K, N] matrix and a packed
        // [K, N, E] expert bank, whose LAST expert overreads past the pack.
        [k, _] | [k, _, _] => ggml_type.mmq_weight_tail_slack_bytes(*k)?,
        _ => 0,
    };
    byte_len
        .checked_add(usize::try_from(slack)?)
        .context("GGUF tensor allocation size overflow")
}

/// Complete admitted GGUF directory resident as raw device tensors.
#[must_use = "device allocations must be explicitly freed or handed to another owner"]
pub struct GgufDeviceStore {
    tensors: BTreeMap<String, GgufDeviceTensor>,
    total_bytes: usize,
    allocated_bytes: usize,
}

/// A store teardown that retains every allocation whose free failed.
///
/// This type deliberately does not implement [`std::error::Error`]: converting
/// it into a generic error would discard the only handle that can retry the
/// failed frees.
#[must_use = "failed device frees retain allocations and must be retried or recovered"]
pub struct GgufDeviceStoreFreeError {
    store: GgufDeviceStore,
    first_error: anyhow::Error,
    failed_tensors: usize,
}

/// A device-load failure that retains every allocation whose rollback failed.
///
/// This type deliberately does not implement [`std::error::Error`]. A generic
/// error conversion could discard the only owners able to retry cleanup.
#[must_use = "failed device loads may retain allocations and must be inspected"]
pub struct GgufDeviceLoadError {
    primary: anyhow::Error,
    cleanup_failures: Vec<GgufDeviceStoreFreeError>,
}

impl GgufDeviceLoadError {
    fn new(primary: anyhow::Error) -> Self {
        Self {
            primary,
            cleanup_failures: Vec::new(),
        }
    }

    fn push_cleanup(&mut self, failure: GgufDeviceStoreFreeError) {
        self.cleanup_failures.push(failure);
    }

    pub fn failure(&self) -> &anyhow::Error {
        &self.primary
    }

    pub fn cleanup_failure_count(&self) -> usize {
        self.cleanup_failures.len()
    }

    pub fn cleanup_failure(&self, index: usize) -> Option<&GgufDeviceStoreFreeError> {
        self.cleanup_failures.get(index)
    }

    pub fn retained_tensor_count(&self) -> usize {
        self.cleanup_failures
            .iter()
            .map(GgufDeviceStoreFreeError::failed_tensor_count)
            .sum()
    }

    pub fn retained_bytes(&self) -> usize {
        self.cleanup_failures
            .iter()
            .map(|failure| failure.store().total_bytes())
            .sum()
    }

    /// Retry every retained cleanup owner. Successful frees are forgotten;
    /// only owners that still fail are returned.
    pub fn retry_cleanup(
        mut self,
        gpu: &dyn GpuBackend,
    ) -> std::result::Result<anyhow::Error, GgufDeviceLoadError> {
        let mut remaining = Vec::new();
        for cleanup in std::mem::take(&mut self.cleanup_failures) {
            if let Err(failure) = cleanup.retry(gpu) {
                remaining.push(failure);
            }
        }
        if remaining.is_empty() {
            Ok(self.primary)
        } else {
            self.cleanup_failures = remaining;
            Err(self)
        }
    }
}

impl From<anyhow::Error> for GgufDeviceLoadError {
    fn from(primary: anyhow::Error) -> Self {
        Self::new(primary)
    }
}

impl fmt::Debug for GgufDeviceLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GgufDeviceLoadError")
            .field("cleanup_failures", &self.cleanup_failure_count())
            .field("retained_tensors", &self.retained_tensor_count())
            .field("retained_bytes", &self.retained_bytes())
            .field("primary", &self.primary)
            .finish()
    }
}

impl fmt::Display for GgufDeviceLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GGUF device load failed ({:#}); {} cleanup owners retain {} tensors ({} bytes)",
            self.primary,
            self.cleanup_failure_count(),
            self.retained_tensor_count(),
            self.retained_bytes()
        )
    }
}

impl GgufDeviceStoreFreeError {
    pub fn failure(&self) -> &anyhow::Error {
        &self.first_error
    }

    pub fn failed_tensor_count(&self) -> usize {
        self.failed_tensors
    }

    pub fn store(&self) -> &GgufDeviceStore {
        &self.store
    }

    pub fn into_store(self) -> GgufDeviceStore {
        self.store
    }

    pub fn retry(self, gpu: &dyn GpuBackend) -> std::result::Result<(), GgufDeviceStoreFreeError> {
        self.store.free(gpu)
    }
}

impl fmt::Debug for GgufDeviceStoreFreeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GgufDeviceStoreFreeError")
            .field("failed_tensors", &self.failed_tensors)
            .field("remaining_bytes", &self.store.total_bytes)
            .field("first_error", &self.first_error)
            .finish()
    }
}

impl fmt::Display for GgufDeviceStoreFreeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "failed to free {} GGUF device tensors ({} bytes retained); first error: {:#}",
            self.failed_tensors, self.store.total_bytes, self.first_error
        )
    }
}

impl GgufDeviceStore {
    fn one(name: String, tensor: GgufDeviceTensor) -> Self {
        let total_bytes = tensor.byte_len;
        let allocated_bytes = tensor.alloc_bytes;
        Self {
            tensors: BTreeMap::from([(name, tensor)]),
            total_bytes,
            allocated_bytes,
        }
    }

    pub fn get(&self, name: &str) -> Option<&GgufDeviceTensor> {
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

    /// Sum of the tensors' LOGICAL payloads. This is the figure the admitted
    /// checkpoint pins and `load_glm53_store` compares for equality, so it must
    /// never absorb the MMQ tail slack -- use [`Self::allocated_bytes`] for
    /// anything that asks how much device memory is actually held.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Sum of what was actually handed to `gpu.alloc`: payload plus MMQ tail
    /// slack. This, not [`Self::total_bytes`], is the residency figure.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Attempt every free. Successful allocations are removed; failed ones
    /// remain owned by the returned error and can be retried.
    pub fn free(
        mut self,
        gpu: &dyn GpuBackend,
    ) -> std::result::Result<(), GgufDeviceStoreFreeError> {
        let names = self.tensors.keys().cloned().collect::<Vec<_>>();
        let mut first_error = None;
        let mut failed_tensors = 0usize;
        for name in names {
            let tensor = self
                .tensors
                .get(&name)
                .expect("name was collected from this device store");
            match gpu.free(tensor.ptr) {
                Ok(()) => {
                    let released = self
                        .tensors
                        .remove(&name)
                        .expect("successfully freed tensor must remain owned");
                    self.total_bytes = self
                        .total_bytes
                        .checked_sub(released.byte_len)
                        .expect("device-store byte accounting must remain valid");
                    self.allocated_bytes = self
                        .allocated_bytes
                        .checked_sub(released.alloc_bytes)
                        .expect("device-store allocation accounting must remain valid");
                }
                Err(error) => {
                    failed_tensors += 1;
                    first_error.get_or_insert(error);
                }
            }
        }
        match first_error {
            None => Ok(()),
            Some(first_error) => Err(GgufDeviceStoreFreeError {
                store: self,
                first_error,
                failed_tensors,
            }),
        }
    }
}

/// Allocate and copy a raw quantized tensor from the retained admitted file handle.
pub fn load_glm53_tensor(
    files: &mut Glm53GgufFiles,
    name: &str,
    gpu: &dyn GpuBackend,
) -> std::result::Result<GgufDeviceTensor, GgufDeviceLoadError> {
    let info = files
        .tensor(name)
        .context("requested tensor is absent from admitted GLM-5.3 checkpoint")?
        .info
        .clone();
    let byte_len = usize::try_from(info.byte_len).context("GGUF tensor is too large to address")?;
    if byte_len == 0 {
        return Err(GgufDeviceLoadError::new(anyhow!(
            "admitted GGUF tensor has an empty payload"
        )));
    }
    // The MMQ tile loop reads a whole MMQ_ITER_K of K past a short row, so the
    // allocation -- not just the copy -- has to cover it. The copy below still
    // writes exactly `byte_len`; the slack is never read for its contents, only
    // multiplied by zero-padded activations.
    let alloc_bytes = mmq_tensor_alloc_bytes(info.ggml_type, &info.dimensions, byte_len)
        .map_err(GgufDeviceLoadError::new)?;
    let ptr = gpu.alloc(alloc_bytes).map_err(GgufDeviceLoadError::new)?;
    let mut scratch = vec![0u8; byte_len.min(COPY_CHUNK_BYTES)];
    let copied = files.stream_tensor(name, &mut scratch, |offset, bytes| {
        let offset = usize::try_from(offset).context("GGUF device offset overflow")?;
        let address = ptr
            .0
            .checked_add(u64::try_from(offset)?)
            .context("GGUF device address overflow")?;
        gpu.copy_h2d(bytes, DevicePtr(address))
    });
    if let Err(copy_error) = copied {
        let tensor = GgufDeviceTensor {
            ptr,
            dimensions: info.dimensions,
            ggml_type: info.ggml_type,
            byte_len,
            alloc_bytes,
        };
        return Err(attach_cleanup(
            GgufDeviceLoadError::new(copy_error),
            GgufDeviceStore::one(name.to_owned(), tensor),
            gpu,
        ));
    }
    Ok(GgufDeviceTensor {
        ptr,
        dimensions: info.dimensions,
        ggml_type: info.ggml_type,
        byte_len,
        alloc_bytes,
    })
}

/// Load every tensor after an exact raw-byte preflight, rolling back on any error.
pub fn load_glm53_store(
    files: &mut Glm53GgufFiles,
    gpu: &dyn GpuBackend,
    reserve_bytes: usize,
) -> std::result::Result<GgufDeviceStore, GgufDeviceLoadError> {
    let expected = usize::try_from(files.summary().tensor_bytes)
        .context("admitted GGUF payload is too large to address")?;
    // Preflight must ask for what will actually be ALLOCATED, not what the
    // checkpoint's payload weighs: every MMQ weight whose K is short of
    // MMQ_ITER_K is allocated with tail slack. Small on GLM-5.3, but an
    // allocation that grows without preflight seeing it is exactly how this
    // host gets killed.
    let slack: usize = files
        .tensor_names()
        .map(str::to_owned)
        .collect::<Vec<_>>()
        .iter()
        .map(|name| {
            let info = &files
                .tensor(name)
                .context("admitted tensor vanished between summary and preflight")?
                .info;
            let byte_len =
                usize::try_from(info.byte_len).context("GGUF tensor is too large to address")?;
            Ok(mmq_tensor_alloc_bytes(info.ggml_type, &info.dimensions, byte_len)? - byte_len)
        })
        .sum::<anyhow::Result<usize>>()?;
    let required = expected
        .checked_add(slack)
        .and_then(|payload| payload.checked_add(reserve_bytes))
        .context("GGUF payload plus reserve overflows address space")?;
    let free = gpu.free_memory()?;
    if required > free {
        return Err(GgufDeviceLoadError::new(anyhow!(
            "GGUF device preflight requires {expected} payload bytes plus {reserve_bytes} reserve bytes, but only {free} bytes are free"
        )));
    }

    let names: Vec<String> = files.tensor_names().map(str::to_owned).collect();
    let mut tensors = BTreeMap::new();
    let mut total_bytes = 0usize;
    for name in names {
        let tensor = match load_glm53_tensor(files, &name, gpu) {
            Ok(tensor) => tensor,
            Err(error) => return Err(rollback_error(error, tensors, total_bytes, gpu)),
        };
        let tensor_bytes = tensor.byte_len;
        let Some(next_total) = total_bytes.checked_add(tensor_bytes) else {
            let error = attach_cleanup(
                GgufDeviceLoadError::new(anyhow!("GGUF device-store byte count overflow")),
                GgufDeviceStore::one(name, tensor),
                gpu,
            );
            return Err(rollback_error(error, tensors, total_bytes, gpu));
        };
        if next_total > expected {
            let error = attach_cleanup(
                GgufDeviceLoadError::new(anyhow!(
                    "GGUF device-store byte count exceeds admitted payload"
                )),
                GgufDeviceStore::one(name, tensor),
                gpu,
            );
            return Err(rollback_error(error, tensors, total_bytes, gpu));
        }
        tensors.insert(name, tensor);
        total_bytes = next_total;
    }
    if total_bytes != expected {
        return Err(rollback_error(
            GgufDeviceLoadError::new(anyhow!(
                "GGUF device-store byte mismatch: loaded {total_bytes}, admitted {expected}"
            )),
            tensors,
            total_bytes,
            gpu,
        ));
    }
    Ok(GgufDeviceStore {
        allocated_bytes: tensors.values().map(|tensor| tensor.alloc_bytes).sum(),
        tensors,
        total_bytes,
    })
}

/// Compatibility wrapper that cannot accidentally consume a Q2 admission.
pub fn load_glm53_iq3_tensor(
    files: &mut Glm53Iq3Files,
    name: &str,
    gpu: &dyn GpuBackend,
) -> std::result::Result<GgufDeviceTensor, GgufDeviceLoadError> {
    if files.profile() != Glm53QuantProfile::UdIq3Xxs {
        return Err(GgufDeviceLoadError::new(anyhow!(
            "IQ3 tensor loader received another GLM-5.3 quant profile"
        )));
    }
    load_glm53_tensor(files, name, gpu)
}

/// Compatibility wrapper that cannot accidentally consume a Q2 admission.
pub fn load_glm53_iq3_store(
    files: &mut Glm53Iq3Files,
    gpu: &dyn GpuBackend,
    reserve_bytes: usize,
) -> std::result::Result<GgufDeviceStore, GgufDeviceLoadError> {
    if files.profile() != Glm53QuantProfile::UdIq3Xxs {
        return Err(GgufDeviceLoadError::new(anyhow!(
            "IQ3 store loader received another GLM-5.3 quant profile"
        )));
    }
    load_glm53_store(files, gpu, reserve_bytes)
}

fn rollback_error(
    error: GgufDeviceLoadError,
    tensors: BTreeMap<String, GgufDeviceTensor>,
    total_bytes: usize,
    gpu: &dyn GpuBackend,
) -> GgufDeviceLoadError {
    attach_cleanup(
        error,
        GgufDeviceStore {
            allocated_bytes: tensors.values().map(|tensor| tensor.alloc_bytes).sum(),
            tensors,
            total_bytes,
        },
        gpu,
    )
}

fn attach_cleanup(
    mut error: GgufDeviceLoadError,
    store: GgufDeviceStore,
    gpu: &dyn GpuBackend,
) -> GgufDeviceLoadError {
    if store.is_empty() {
        return error;
    }
    if let Err(cleanup) = store.free(gpu) {
        error.push_cleanup(cleanup);
    }
    error
}
