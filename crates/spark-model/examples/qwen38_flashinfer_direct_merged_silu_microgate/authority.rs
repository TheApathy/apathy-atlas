// SPDX-License-Identifier: AGPL-3.0-only

use std::fs::File;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};

use super::BinaryIdentity;
use super::provenance::{check_hash, source_specs};

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileStamp {
    dev: u64,
    ino: u64,
    mode: u32,
    len: u64,
    sha256: String,
}

pub(super) struct HeldFile {
    file: File,
    path: PathBuf,
    stamp: FileStamp,
}

impl HeldFile {
    pub(super) fn open(path: &Path, expected: Option<&str>) -> Result<Self> {
        let path = std::fs::canonicalize(path)?;
        let file = File::open(&path)?;
        let stamp = stamp(&file)?;
        let path_meta = std::fs::metadata(&path)?;
        ensure!(
            same_object(&stamp, &path_meta),
            "path changed while opening"
        );
        if let Some(expected) = expected {
            check_hash(&path.display().to_string(), &stamp.sha256, expected)?;
        }
        Ok(Self { file, path, stamp })
    }

    pub(super) fn verify_unchanged(&self) -> Result<()> {
        let now = stamp(&self.file)?;
        let path_meta = std::fs::metadata(&self.path)?;
        ensure!(now == self.stamp, "held file bytes or metadata changed");
        ensure!(
            same_object(&self.stamp, &path_meta),
            "held path was replaced"
        );
        Ok(())
    }

    pub(super) fn read_text(&self) -> Result<String> {
        let mut file = self.file.try_clone()?;
        let mut text = String::new();
        file.read_to_string(&mut text)?;
        Ok(text)
    }
}

fn same_object(stamp: &FileStamp, metadata: &std::fs::Metadata) -> bool {
    metadata.is_file()
        && (
            metadata.dev(),
            metadata.ino(),
            metadata.mode(),
            metadata.len(),
        ) == (stamp.dev, stamp.ino, stamp.mode, stamp.len)
}

fn stamp(file: &File) -> Result<FileStamp> {
    let metadata = file.metadata()?;
    ensure!(metadata.is_file(), "identity target is not a regular file");
    let fd_path = format!("/proc/{}/fd/{}", std::process::id(), file.as_raw_fd());
    let output = Command::new("/usr/bin/sha256sum")
        .env_clear()
        .arg(fd_path)
        .output()?;
    ensure!(output.status.success(), "descriptor sha256sum failed");
    let sha256 = std::str::from_utf8(&output.stdout)?
        .split_whitespace()
        .next()
        .context("descriptor sha256sum omitted digest")?
        .to_owned();
    Ok(FileStamp {
        dev: metadata.dev(),
        ino: metadata.ino(),
        mode: metadata.mode(),
        len: metadata.len(),
        sha256,
    })
}

pub(super) struct QualificationAuthority {
    binary: HeldFile,
    sources: Vec<(&'static str, &'static str, HeldFile)>,
}

impl QualificationAuthority {
    pub(super) fn seal() -> Result<Self> {
        let binary = HeldFile::open(&std::env::current_exe()?, None)?;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let sources = source_specs()
            .into_iter()
            .map(|(path, expected)| {
                Ok((
                    path,
                    expected,
                    HeldFile::open(&root.join(path), Some(expected))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { binary, sources })
    }

    pub(super) fn binary(&self) -> BinaryIdentity {
        BinaryIdentity {
            path: self.binary.path.to_string_lossy().into_owned(),
            sha256: self.binary.stamp.sha256.clone(),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
        }
    }

    pub(super) fn sources(&self) -> Vec<(&'static str, &'static str)> {
        self.sources
            .iter()
            .map(|(path, sha, _)| (*path, *sha))
            .collect()
    }

    pub(super) fn verify_unchanged(&self) -> Result<()> {
        self.binary.verify_unchanged()?;
        for (_, _, source) in &self.sources {
            source.verify_unchanged()?;
        }
        Ok(())
    }
}
