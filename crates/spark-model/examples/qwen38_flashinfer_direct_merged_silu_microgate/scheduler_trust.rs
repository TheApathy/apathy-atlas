// SPDX-License-Identifier: AGPL-3.0-only

use std::fs::Metadata;
use std::collections::BTreeMap;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy)]
pub(super) struct FileIdentity {
    dev: u64,
    ino: u64,
    mode: u32,
    uid: u32,
}

impl FileIdentity {
    pub(super) fn capture(metadata: &Metadata) -> Self {
        Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            mode: metadata.mode(),
            uid: metadata.uid(),
        }
    }

    pub(super) fn matches(self, metadata: &Metadata) -> bool {
        (self.dev, self.ino, self.mode, self.uid)
            == (
                metadata.dev(),
                metadata.ino(),
                metadata.mode(),
                metadata.uid(),
            )
    }
}

pub(super) fn trusted_root_file(path: &Path, permissions: u32, label: &str) -> Result<Metadata> {
    ensure!(path.is_absolute(), "{label} path is not absolute");
    let parent = path.parent().context("trusted path has no parent")?;
    let mut current = PathBuf::from("/");
    for component in parent.components() {
        match component {
            Component::RootDir => continue,
            Component::Normal(part) => current.push(part),
            _ => anyhow::bail!("{label} path is not canonical"),
        }
        let metadata = std::fs::symlink_metadata(&current)?;
        ensure!(
            metadata.is_dir() && metadata.uid() == 0 && metadata.mode() & 0o022 == 0,
            "{label} parent is not root-owned and non-writable"
        );
    }
    let metadata = std::fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file()
            && metadata.uid() == 0
            && metadata.mode() & 0o7777 == permissions
            && metadata.nlink() == 1,
        "{label} is not a root-owned immutable regular file"
    );
    Ok(metadata)
}

pub(super) fn validate_build_recipe(argv_hex: &str, env_hex: &str, cargo: &str) -> Result<()> {
    let argv = decode_records(argv_hex, "build argv")?;
    ensure!(
        argv.first().is_some_and(|value| value == cargo)
            && argv[1..]
                == [
                    "build",
                    "--release",
                    "--locked",
                    "-p",
                    "spark-model",
                    "--example",
                    "qwen38_flashinfer_direct_merged_silu_microgate",
                ],
        "build argv is not the exact released example command"
    );
    let mut environment = BTreeMap::new();
    // BOUND FIRST so the owned `String`s outlive the map. `decode_records`
    // returns `Vec<String>`; iterating it by value drops each record at the end
    // of its iteration while `environment` still holds `&str` slices into it.
    let env_records = decode_records(env_hex, "build environment")?;
    for record in &env_records {
        let (key, value) = record
            .split_once('=')
            .context("build environment record omitted equals")?;
        ensure!(
            !key.is_empty() && environment.insert(key, value).is_none(),
            "duplicate or empty build environment key"
        );
    }
    ensure!(
        environment.keys().copied().collect::<Vec<_>>()
            == [
                "CARGO_HOME",
                "CUDA_HOME",
                "CUDA_VISIBLE_DEVICES",
                "LC_ALL",
                "PATH",
                "RUSTUP_HOME",
                "SOURCE_DATE_EPOCH",
            ]
            && environment["CUDA_HOME"] == "/usr/local/cuda-13.0"
            && environment["CUDA_VISIBLE_DEVICES"].is_empty()
            && environment["LC_ALL"] == "C"
            && environment["SOURCE_DATE_EPOCH"] == "0"
            && Path::new(environment["CARGO_HOME"]).is_absolute()
            && Path::new(environment["RUSTUP_HOME"]).is_absolute()
            && environment["PATH"].split(':').all(|path| Path::new(path).is_absolute()),
        "build environment is not exact, sanitized, and GPU-hidden"
    );
    Ok(())
}

fn decode_records(value: &str, label: &str) -> Result<Vec<String>> {
    let body = value.strip_prefix("hex:").context("encoded field omitted hex prefix")?;
    ensure!(
        !body.is_empty()
            && body.len() <= 16_384
            && body.len().is_multiple_of(2)
            && body.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} is not canonical lowercase hex"
    );
    let bytes = body
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    let records = bytes
        .split(|byte| *byte == 0)
        .map(|record| String::from_utf8(record.to_vec()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(records.iter().all(|record| !record.is_empty()), "{label} contains an empty record");
    Ok(records)
}
