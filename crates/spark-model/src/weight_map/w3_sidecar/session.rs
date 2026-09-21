// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::QuantizedWeight;
use super::W3FfnLayerWeights;
use super::admission::W3SidecarRequest;
use super::digest::sha256;
use super::manifest::{LayerPlan, ProjectionPlan, validate_manifest};

#[cfg(target_os = "linux")]
const MAX_SIDECAR_BYTES: u64 = 16 << 30;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct W3SidecarReceipt {
    pub path: PathBuf,
    pub sha256: [u8; 32],
    pub size: u64,
    pub device: u64,
    pub inode: u64,
    pub requested_layers: Vec<usize>,
    pub validated_layers: Vec<usize>,
    pub uploaded_layers: Vec<usize>,
    pub installed_layers: Vec<usize>,
    pub requested_count: usize,
    pub validated_count: usize,
    pub uploaded_count: usize,
    pub installed_count: usize,
}

/// Transactional loader for an admitted W3 sidecar.
///
/// `prepare` retains the exact authenticated bytes and validates every
/// requested tensor before `upload_layer` can allocate device memory. A caller
/// must publish returned weights into its unpublished model object, call
/// `mark_installed`, and require `finish` before publishing that model.
pub struct W3SidecarSession {
    request: W3SidecarRequest,
    _held_file: File,
    artifact: Vec<u8>,
    identity: FileIdentity,
    plans: BTreeMap<usize, LayerPlan>,
    uploaded: BTreeSet<usize>,
    installed: BTreeSet<usize>,
    awaiting_install: Option<usize>,
    poisoned: Option<String>,
}

impl W3SidecarSession {
    pub fn prepare(
        request: W3SidecarRequest,
        layer_prefixes: &[String],
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let (held_file, artifact, identity) = read_held_artifact(&request)?;
        ensure!(
            sha256(&artifact) == request.sha256(),
            "W3 sidecar SHA256 differs from the admitted digest"
        );
        let plans = validate_manifest(&artifact, &request, layer_prefixes, hidden, intermediate)?;
        ensure!(
            plans.keys().copied().collect::<BTreeSet<_>>() == *request.layers(),
            "W3 sidecar validation did not cover the full requested layer set"
        );
        Ok(Self {
            request,
            _held_file: held_file,
            artifact,
            identity,
            plans,
            uploaded: BTreeSet::new(),
            installed: BTreeSet::new(),
            awaiting_install: None,
            poisoned: None,
        })
    }

    /// Upload one prevalidated layer. Unrequested model layers return `None`;
    /// every requested layer must be uploaded exactly once and then marked
    /// installed before another requested upload begins.
    pub fn upload_layer(
        &mut self,
        layer: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<W3FfnLayerWeights>> {
        self.upload_layer_with(layer, |artifact, plan| {
            upload_layer_plan(gpu, artifact, plan)
        })
    }

    pub fn mark_installed(&mut self, layer: usize) -> Result<()> {
        self.require_healthy()?;
        if !self.request.layers().contains(&layer) {
            return self.poison(format!("cannot install unrequested W3 layer {layer}"));
        }
        if self.awaiting_install != Some(layer) {
            return self.poison(format!(
                "W3 layer {layer} install is out of order; awaiting {:?}",
                self.awaiting_install
            ));
        }
        if !self.uploaded.contains(&layer) || !self.installed.insert(layer) {
            return self.poison(format!("W3 layer {layer} install census is inconsistent"));
        }
        self.awaiting_install = None;
        Ok(())
    }

    pub fn finish(self) -> Result<W3SidecarReceipt> {
        if let Some(reason) = self.poisoned {
            bail!("W3 sidecar session is poisoned: {reason}");
        }
        ensure!(
            self.awaiting_install.is_none(),
            "W3 sidecar finish called before the last uploaded layer was installed"
        );
        let requested = self.request.layers().clone();
        let validated = self.plans.keys().copied().collect::<BTreeSet<_>>();
        ensure!(
            validated == requested,
            "W3 validated layer census is incomplete"
        );
        ensure!(
            self.uploaded == requested,
            "W3 uploaded layer census is incomplete"
        );
        ensure!(
            self.installed == requested,
            "W3 installed layer census is incomplete"
        );
        let requested_count = requested.len();
        let validated_count = validated.len();
        let uploaded_count = self.uploaded.len();
        let installed_count = self.installed.len();
        Ok(W3SidecarReceipt {
            path: self.request.path().to_path_buf(),
            sha256: self.request.sha256(),
            size: self.request.size(),
            device: self.identity.device,
            inode: self.identity.inode,
            requested_layers: requested.iter().copied().collect(),
            validated_layers: validated.iter().copied().collect(),
            uploaded_layers: self.uploaded.iter().copied().collect(),
            installed_layers: self.installed.iter().copied().collect(),
            requested_count,
            validated_count,
            uploaded_count,
            installed_count,
        })
    }

    pub(super) fn upload_layer_with<T>(
        &mut self,
        layer: usize,
        upload: impl FnOnce(&[u8], &LayerPlan) -> Result<T>,
    ) -> Result<Option<T>> {
        self.require_healthy()?;
        if !self.request.layers().contains(&layer) {
            return Ok(None);
        }
        if let Some(pending) = self.awaiting_install {
            return self.poison(format!(
                "W3 layer {pending} must be installed before uploading layer {layer}"
            ));
        }
        if self.uploaded.contains(&layer) {
            return self.poison(format!("W3 layer {layer} was uploaded more than once"));
        }
        let plan = self
            .plans
            .get(&layer)
            .expect("requested W3 layer was validated during prepare")
            .clone();
        let uploaded = match upload(&self.artifact, &plan) {
            Ok(uploaded) => uploaded,
            Err(error) => {
                self.poisoned = Some(format!("W3 layer {layer} upload failed: {error:#}"));
                return Err(error).context(format!("upload W3 layer {layer}"));
            }
        };
        ensure!(self.uploaded.insert(layer));
        self.awaiting_install = Some(layer);
        Ok(Some(uploaded))
    }

    fn require_healthy(&self) -> Result<()> {
        if let Some(reason) = &self.poisoned {
            bail!("W3 sidecar session is poisoned: {reason}");
        }
        Ok(())
    }

    fn poison<T>(&mut self, reason: String) -> Result<T> {
        self.poisoned = Some(reason.clone());
        bail!("W3 sidecar session poisoned: {reason}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[cfg(target_os = "linux")]
fn read_held_artifact(request: &W3SidecarRequest) -> Result<(File, Vec<u8>, FileIdentity)> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    const O_NOFOLLOW: i32 = 0o00400000;

    ensure!(
        request.size() <= MAX_SIDECAR_BYTES,
        "W3 sidecar exceeds the {MAX_SIDECAR_BYTES}-byte safety limit"
    );
    let canonical_path = std::fs::canonicalize(request.path())
        .with_context(|| format!("canonicalize W3 sidecar {}", request.path().display()))?;
    ensure!(
        canonical_path == request.path(),
        "W3 sidecar path must already be canonical"
    );
    let pre_metadata = std::fs::symlink_metadata(request.path())
        .with_context(|| format!("stat W3 sidecar {}", request.path().display()))?;
    ensure!(
        pre_metadata.file_type().is_file(),
        "W3 sidecar must be a regular file"
    );
    ensure!(
        pre_metadata.nlink() == 1,
        "W3 sidecar must not have hard links"
    );
    let pre_identity = file_identity(&pre_metadata);
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(request.path())
        .with_context(|| format!("open held W3 sidecar {}", request.path().display()))?;
    let opened_metadata = file.metadata().context("stat held W3 sidecar descriptor")?;
    ensure!(
        opened_metadata.file_type().is_file(),
        "held W3 sidecar is not regular"
    );
    let opened_identity = file_identity(&opened_metadata);
    ensure!(
        pre_identity == opened_identity,
        "W3 sidecar changed between path stat and held open"
    );
    ensure!(
        opened_identity.size == request.size(),
        "W3 sidecar size differs from the admitted size"
    );
    let expected = usize::try_from(request.size()).context("W3 sidecar size does not fit host")?;
    let read_limit = expected
        .checked_add(1)
        .context("W3 sidecar read limit overflow")?;
    let mut artifact = Vec::new();
    artifact
        .try_reserve_exact(read_limit)
        .context("reserve authenticated W3 sidecar bytes")?;
    (&mut file)
        .take(read_limit as u64)
        .read_to_end(&mut artifact)
        .context("read held W3 sidecar bytes")?;
    ensure!(
        artifact.len() == expected,
        "W3 sidecar changed size while reading"
    );

    let post_descriptor = file
        .metadata()
        .context("restat held W3 sidecar descriptor")?;
    let post_path = std::fs::symlink_metadata(request.path())
        .with_context(|| format!("restat W3 sidecar {}", request.path().display()))?;
    ensure!(
        post_path.file_type().is_file(),
        "W3 sidecar path ceased to be regular"
    );
    ensure!(
        post_path.nlink() == 1,
        "W3 sidecar acquired a hard link while reading"
    );
    ensure!(
        file_identity(&post_descriptor) == opened_identity
            && file_identity(&post_path) == opened_identity,
        "W3 sidecar identity or metadata changed during authenticated read"
    );
    Ok((file, artifact, opened_identity))
}

#[cfg(target_os = "linux")]
fn file_identity(metadata: &std::fs::Metadata) -> FileIdentity {
    use std::os::unix::fs::MetadataExt;

    FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        mode: metadata.mode(),
        links: metadata.nlink(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

#[cfg(not(target_os = "linux"))]
fn read_held_artifact(_request: &W3SidecarRequest) -> Result<(File, Vec<u8>, FileIdentity)> {
    bail!("fail-closed W3 sidecar sessions currently require Linux O_NOFOLLOW")
}

struct ProjectionHost<'a> {
    packed: &'a [u8],
    scales: &'a [u8],
    packed_t: Vec<u8>,
    scales_t: Vec<u8>,
    scale2: f32,
}

fn prepare_projection<'a>(artifact: &'a [u8], plan: &ProjectionPlan) -> Result<ProjectionHost<'a>> {
    let packed = &artifact[plan.packed.clone()];
    let scales = &artifact[plan.scales.clone()];
    let packed_columns = plan
        .k
        .checked_div(8)
        .and_then(|octets| octets.checked_mul(3))
        .context("W3 transpose packed-column count overflow")?;
    let packed_t = transpose_checked(packed, plan.n, packed_columns)?;
    let scales_t = transpose_checked(scales, plan.n, plan.k / 16)?;
    Ok(ProjectionHost {
        packed,
        scales,
        packed_t,
        scales_t,
        scale2: plan.scale2,
    })
}

fn transpose_checked(source: &[u8], rows: usize, columns: usize) -> Result<Vec<u8>> {
    ensure!(
        rows.checked_mul(columns) == Some(source.len()),
        "W3 transpose source extent mismatch"
    );
    let padded_rows = rows
        .checked_add(63)
        .context("W3 transpose row padding overflow")?
        / 64
        * 64;
    let output_len = columns
        .checked_mul(padded_rows)
        .context("W3 transpose output extent overflow")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .context("reserve W3 transposed host bytes")?;
    output.resize(output_len, 0);
    for row in 0..rows {
        for column in 0..columns {
            output[column * padded_rows + row] = source[row * columns + column];
        }
    }
    Ok(output)
}

fn upload_layer_plan(
    gpu: &dyn GpuBackend,
    artifact: &[u8],
    plan: &LayerPlan,
) -> Result<W3FfnLayerWeights> {
    upload_layer_plan_with(&GpuUploadTransaction { gpu }, artifact, plan)
}

/// Narrow transaction boundary used by the production uploader and hostile
/// CPU tests. Keeping allocation, copy, and rollback behind one interface
/// makes the partial-allocation contract testable without emulating unrelated
/// GPU operations.
pub(super) trait UploadTransaction {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;
    fn copy_h2d(&self, bytes: &[u8], destination: DevicePtr) -> Result<()>;
    fn free(&self, pointer: DevicePtr) -> Result<()>;
}

struct GpuUploadTransaction<'a> {
    gpu: &'a dyn GpuBackend,
}

impl UploadTransaction for GpuUploadTransaction<'_> {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.gpu.alloc(bytes)
    }

    fn copy_h2d(&self, bytes: &[u8], destination: DevicePtr) -> Result<()> {
        self.gpu.copy_h2d(bytes, destination)
    }

    fn free(&self, pointer: DevicePtr) -> Result<()> {
        self.gpu.free(pointer)
    }
}

pub(super) fn upload_layer_plan_with(
    transaction: &dyn UploadTransaction,
    artifact: &[u8],
    plan: &LayerPlan,
) -> Result<W3FfnLayerWeights> {
    let gate = prepare_projection(artifact, &plan.gate)?;
    let up = prepare_projection(artifact, &plan.up)?;
    let down = prepare_projection(artifact, &plan.down)?;
    let mut allocations = Vec::with_capacity(12);
    let result = (|| {
        let (gate, gate_t) = upload_projection(transaction, &gate, &mut allocations)?;
        let (up, up_t) = upload_projection(transaction, &up, &mut allocations)?;
        let (down, down_t) = upload_projection(transaction, &down, &mut allocations)?;
        Ok(W3FfnLayerWeights {
            gate,
            up,
            down,
            gate_t,
            up_t,
            down_t,
        })
    })();
    if result.is_err() {
        for pointer in allocations.into_iter().rev() {
            if let Err(error) = transaction.free(pointer) {
                tracing::error!(?pointer, %error, "failed to release partial W3 upload");
            }
        }
    }
    result
}

fn upload_projection(
    transaction: &dyn UploadTransaction,
    host: &ProjectionHost<'_>,
    allocations: &mut Vec<DevicePtr>,
) -> Result<(QuantizedWeight, QuantizedWeight)> {
    let packed = upload_bytes(transaction, host.packed, allocations)?;
    let scales = upload_bytes(transaction, host.scales, allocations)?;
    let packed_t = upload_bytes(transaction, &host.packed_t, allocations)?;
    let scales_t = upload_bytes(transaction, &host.scales_t, allocations)?;
    Ok((
        QuantizedWeight {
            weight: packed,
            weight_scale: scales,
            weight_scale_2: host.scale2,
            input_scale: DevicePtr::NULL,
        },
        QuantizedWeight {
            weight: packed_t,
            weight_scale: scales_t,
            weight_scale_2: host.scale2,
            input_scale: DevicePtr::NULL,
        },
    ))
}

fn upload_bytes(
    transaction: &dyn UploadTransaction,
    bytes: &[u8],
    allocations: &mut Vec<DevicePtr>,
) -> Result<DevicePtr> {
    let pointer = transaction.alloc(bytes.len())?;
    allocations.push(pointer);
    transaction.copy_h2d(bytes, pointer)?;
    Ok(pointer)
}
