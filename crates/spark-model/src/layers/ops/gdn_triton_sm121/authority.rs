// SPDX-License-Identifier: AGPL-3.0-only

use super::digest::hex_sha256;
use super::manifest::{
    CACHE_ROOT, EXECUTOR_SOURCE_BUNDLE_SHA256, EXECUTOR_SOURCES, GATE_SOURCE_BUNDLE_SHA256,
    GATE_SOURCES, KERNELS, MANIFEST_BYTES, MANIFEST_FILE, MANIFEST_SHA256, NATIVE_ROOT,
};
use anyhow::{Context, Result, ensure};
use std::fs::{self, File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

const MAX_FILE_BYTES: u64 = 1 << 20;

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileIdentity {
    dev: u64,
    ino: u64,
    size: u64,
    mode: u32,
    mtime_nsec: i64,
    ctime_nsec: i64,
}

impl FileIdentity {
    fn from(metadata: &Metadata) -> Self {
        Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            size: metadata.size(),
            mode: metadata.mode(),
            mtime_nsec: metadata.mtime_nsec(),
            ctime_nsec: metadata.ctime_nsec(),
        }
    }
}

#[derive(Debug)]
pub(super) struct HeldFile {
    path: PathBuf,
    pub(super) bytes: Box<[u8]>,
    pub(super) sha256: String,
    identity: FileIdentity,
}

fn stable_read(
    path: &Path,
    expected_sha256: Option<&str>,
    expected_bytes: Option<u64>,
) -> Result<HeldFile> {
    ensure!(path.is_absolute(), "artifact/source path must be absolute");
    let path_before = fs::symlink_metadata(path).context("path metadata")?;
    ensure!(
        path_before.is_file(),
        "artifact/source must be a regular file"
    );
    ensure!(
        fs::canonicalize(path)? == path,
        "artifact/source path is not canonical"
    );
    ensure!(
        path_before.len() <= MAX_FILE_BYTES,
        "artifact/source exceeds size cap"
    );
    if let Some(expected) = expected_bytes {
        ensure!(path_before.len() == expected, "artifact/source size drift");
    }
    let mut file = File::open(path).context("stable file open")?;
    let descriptor_before = file.metadata()?;
    ensure!(descriptor_before.is_file(), "opened object is not regular");
    ensure!(
        FileIdentity::from(&path_before) == FileIdentity::from(&descriptor_before),
        "path changed during open"
    );
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::with_capacity(usize::try_from(path_before.len())?);
    file.by_ref()
        .take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_FILE_BYTES,
        "artifact/source grew while reading"
    );
    let descriptor_after = file.metadata()?;
    let path_after = fs::symlink_metadata(path)?;
    let identity = FileIdentity::from(&path_before);
    ensure!(
        [
            FileIdentity::from(&descriptor_before),
            FileIdentity::from(&descriptor_after),
            FileIdentity::from(&path_after),
        ]
        .iter()
        .all(|value| value == &identity),
        "artifact/source identity changed while reading"
    );
    ensure!(
        bytes.len() as u64 == identity.size,
        "artifact/source short read"
    );
    let sha256 = hex_sha256(&bytes);
    if let Some(expected) = expected_sha256 {
        ensure!(sha256 == expected, "artifact/source SHA-256 drift");
    }
    Ok(HeldFile {
        path: path.to_owned(),
        bytes: bytes.into_boxed_slice(),
        sha256,
        identity,
    })
}

impl HeldFile {
    pub(super) fn recheck(&self) -> Result<()> {
        let current = stable_read(&self.path, Some(&self.sha256), Some(self.identity.size))?;
        ensure!(
            current.identity == self.identity,
            "held source/artifact replaced"
        );
        ensure!(hex_sha256(&self.bytes) == self.sha256, "held bytes changed");
        Ok(())
    }
}

fn source_bundle(root: &Path, names: &[&str], expected: &str) -> Result<Vec<HeldFile>> {
    let mut records = Vec::with_capacity(names.len());
    for name in names {
        records.push((*name, stable_read(&root.join(name), None, None)?));
    }
    records.sort_unstable_by_key(|record| record.0);
    let mut canonical = String::from("{");
    for (index, (name, held)) in records.iter().enumerate() {
        if index != 0 {
            canonical.push(',');
        }
        canonical.push_str(&format!(
            "\"{name}\":[\"{}\",{}]",
            held.sha256, held.identity.size
        ));
    }
    canonical.push('}');
    ensure!(
        hex_sha256(canonical.as_bytes()) == expected,
        "source bundle drift"
    );
    Ok(records.into_iter().map(|record| record.1).collect())
}

pub(super) fn preflight_authority() -> Result<Vec<HeldFile>> {
    let native_root = Path::new(NATIVE_ROOT);
    ensure!(
        fs::canonicalize(native_root)? == native_root,
        "native root drift"
    );
    let mut authority = vec![stable_read(
        &native_root.join(MANIFEST_FILE),
        Some(MANIFEST_SHA256),
        Some(MANIFEST_BYTES),
    )?];
    authority.extend(source_bundle(
        native_root,
        GATE_SOURCES,
        GATE_SOURCE_BUNDLE_SHA256,
    )?);
    authority.extend(source_bundle(
        native_root,
        EXECUTOR_SOURCES,
        EXECUTOR_SOURCE_BUNDLE_SHA256,
    )?);
    Ok(authority)
}

pub(super) fn preflight_cubins() -> Result<Vec<HeldFile>> {
    KERNELS
        .iter()
        .map(|spec| {
            let path = Path::new(CACHE_ROOT)
                .join(spec.cache_dir)
                .join(format!("{}.cubin", spec.function));
            stable_read(&path, Some(spec.cubin_sha256), Some(spec.cubin_bytes))
        })
        .collect()
}
