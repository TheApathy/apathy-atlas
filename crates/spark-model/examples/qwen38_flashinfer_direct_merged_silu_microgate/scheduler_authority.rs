// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::FromRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use super::authority::HeldFile;
use super::provenance::{check_hash, sha256_bytes};
use super::scheduler_build::{
    BuildAuthority, manifest_field_count, open_build_authority, validate_build_manifest,
};
use super::scheduler_trust::{FileIdentity, trusted_root_file};
use super::{BinaryIdentity, BundleIdentity};

pub(super) const RELEASE_SCHEDULER_MANIFEST_PATH: &str = "UNRELEASED-SCHEDULER-MANIFEST-PATH";
const SCHEMA: &str = "qwen38-direct-merged-silu-scheduler-v1";
const TICKET_BYTES: usize = 32;

pub(super) struct SchedulerSession {
    manifest: HeldFile,
    receipt: File,
    receipt_path: PathBuf,
    receipt_identity: FileIdentity,
    session_id: String,
    build: BuildAuthority,
}

impl SchedulerSession {
    pub(super) fn claim(
        binary: &BinaryIdentity,
        bundle: &BundleIdentity,
        sources: &[(&str, &str)],
    ) -> Result<Self> {
        let manifest_path = Path::new(RELEASE_SCHEDULER_MANIFEST_PATH);
        ensure!(
            manifest_path.is_absolute()
                && !RELEASE_SCHEDULER_MANIFEST_PATH.starts_with("UNRELEASED-"),
            "scheduler manifest path is unreleased"
        );
        trusted_root_file(manifest_path, 0o444, "scheduler manifest")?;
        let manifest = HeldFile::open(manifest_path, None)?;
        let fields = parse_manifest(&manifest.read_text()?)?;
        bind_manifest(&fields, binary, bundle, sources)?;
        let build = open_build_authority(&fields, binary, bundle, sources)?;

        let session_id = field(&fields, "session_id")?.to_owned();
        ensure!(
            session_id.len() >= 32
                && session_id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "scheduler session id is not canonical"
        );
        let ticket_fd = decimal::<i32>(&fields, "ticket_fd")?;
        let receipt_fd = decimal::<i32>(&fields, "receipt_fd")?;
        ensure!(
            ticket_fd >= 3 && receipt_fd >= 3 && ticket_fd != receipt_fd,
            "scheduler descriptors are invalid"
        );

        // SAFETY: the root-owned scheduler manifest uniquely binds two distinct
        // inherited descriptors. Taking ownership makes every failure fail-sticky.
        let mut ticket = unsafe { File::from_raw_fd(ticket_fd) };
        // SAFETY: same argument as above; this descriptor is the sole receipt writer.
        let receipt = unsafe { File::from_raw_fd(receipt_fd) };
        let ticket_meta = ticket.metadata()?;
        ensure!(
            ticket_meta.mode() & 0o170_000 == 0o010_000
                && ticket_meta.dev() == decimal(&fields, "ticket_dev")?
                && ticket_meta.ino() == decimal(&fields, "ticket_ino")?,
            "scheduler ticket descriptor identity changed"
        );
        let mut ticket_bytes = [0_u8; TICKET_BYTES];
        ticket.read_exact(&mut ticket_bytes)?;
        let mut extra = [0_u8; 1];
        ensure!(
            ticket.read(&mut extra)? == 0,
            "scheduler ticket has trailing bytes"
        );
        check_hash(
            "scheduler ticket",
            &sha256_bytes(&ticket_bytes)?,
            field(&fields, "ticket_sha256")?,
        )?;

        let receipt_path = PathBuf::from(field(&fields, "receipt_path")?);
        let path_meta = trusted_root_file(&receipt_path, 0o400, "scheduler receipt")?;
        let receipt_meta = receipt.metadata()?;
        ensure!(
            receipt_meta.is_file()
                && receipt_meta.uid() == 0
                && receipt_meta.mode() & 0o7777 == 0o400
                && receipt_meta.nlink() == 1
                && receipt_meta.len() == 0
                && receipt_meta.dev() == decimal(&fields, "receipt_dev")?
                && receipt_meta.ino() == decimal(&fields, "receipt_ino")?
                && receipt_meta.dev() == path_meta.dev()
                && receipt_meta.ino() == path_meta.ino(),
            "scheduler receipt descriptor identity changed"
        );
        manifest.verify_unchanged()?;
        Ok(Self {
            manifest,
            receipt,
            receipt_path,
            receipt_identity: FileIdentity::capture(&receipt_meta),
            session_id,
            build,
        })
    }

    pub(super) fn session_id(&self) -> &str {
        &self.session_id
    }

    pub(super) fn evidence(&self) -> &serde_json::Value {
        self.build.evidence()
    }

    pub(super) fn publish(mut self, bytes: &[u8]) -> Result<PathBuf> {
        ensure!(!bytes.is_empty(), "empty receipt publication");
        self.manifest.verify_unchanged()?;
        self.build.verify_unchanged()?;
        self.verify_receipt(0)?;
        self.receipt.seek(SeekFrom::Start(0))?;
        self.receipt.write_all(bytes)?;
        self.receipt.sync_all()?;
        self.verify_receipt(u64::try_from(bytes.len())?)?;
        self.build.verify_unchanged()?;
        self.manifest.verify_unchanged()?;
        Ok(self.receipt_path)
    }

    fn verify_receipt(&self, expected_len: u64) -> Result<()> {
        let path_meta = trusted_root_file(&self.receipt_path, 0o400, "scheduler receipt")?;
        let descriptor_meta = self.receipt.metadata()?;
        ensure!(
            self.receipt_identity.matches(&path_meta)
                && self.receipt_identity.matches(&descriptor_meta)
                && descriptor_meta.nlink() == 1
                && descriptor_meta.len() == expected_len,
            "scheduler receipt path, identity, or length changed"
        );
        Ok(())
    }
}

pub(super) fn parse_manifest(text: &str) -> Result<BTreeMap<&str, &str>> {
    ensure!(text.is_ascii() && text.ends_with('\n') && !text.contains('\r'));
    let mut fields = BTreeMap::new();
    for line in text.lines() {
        let (key, value) = line
            .split_once('=')
            .context("malformed scheduler manifest line")?;
        ensure!(
            !key.is_empty()
                && !value.is_empty()
                && key.bytes().all(|byte| !byte.is_ascii_whitespace())
                && value.bytes().all(|byte| !byte.is_ascii_whitespace()),
            "noncanonical scheduler manifest field"
        );
        ensure!(
            fields.insert(key, value).is_none(),
            "duplicate scheduler manifest field"
        );
    }
    Ok(fields)
}

pub(super) fn bind_manifest(
    fields: &BTreeMap<&str, &str>,
    binary: &BinaryIdentity,
    bundle: &BundleIdentity,
    sources: &[(&str, &str)],
) -> Result<()> {
    let fixed = [
        ("schema", SCHEMA.to_owned()),
        ("binary_sha256", binary.sha256.clone()),
        ("bundle_sha256", bundle.sha256.clone()),
        ("bundle_target", bundle.target.clone()),
        ("bundle_module_count", bundle.module_count.to_string()),
    ];
    for (key, expected) in fixed {
        ensure!(
            field(fields, key)? == expected,
            "scheduler manifest binding changed"
        );
    }
    for (path, sha256) in sources {
        ensure!(
            field(fields, &format!("source:{path}"))? == *sha256,
            "scheduler source binding changed"
        );
    }
    for (name, sha256) in &bundle.direct_modules {
        ensure!(
            field(fields, &format!("ptx:{name}"))? == sha256,
            "scheduler PTX binding changed"
        );
    }
    validate_build_manifest(fields)?;
    ensure!(
        fields.len() == manifest_field_count(sources.len(), bundle.direct_modules.len()),
        "scheduler manifest field census changed"
    );
    Ok(())
}

pub(super) fn field<'a>(fields: &'a BTreeMap<&str, &str>, key: &str) -> Result<&'a str> {
    fields
        .get(key)
        .copied()
        .with_context(|| format!("scheduler manifest omitted {key}"))
}

fn decimal<T: std::str::FromStr>(fields: &BTreeMap<&str, &str>, key: &str) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    field(fields, key)?.parse().map_err(Into::into)
}
