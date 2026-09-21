// SPDX-License-Identifier: AGPL-3.0-only

//! Filesystem primitives shared only by the two bounded Vision captures.

use anyhow::{Context, Result, ensure};
#[cfg(unix)]
use std::os::{
    fd::AsRawFd,
    unix::fs::{DirBuilderExt, OpenOptionsExt},
};
use std::{
    ffi::OsStr,
    fs::{File, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
};

pub(crate) fn validate_path(path: &OsStr) -> Result<&Path> {
    let text = path.to_str().context("Vision L0 dump path is not UTF-8")?;
    let path = Path::new(text);
    ensure!(
        !text.is_empty() && path.is_absolute() && path.file_name().is_some(),
        "Vision L0 dump requires an absolute NEW directory"
    );
    ensure!(
        !text.split('/').any(|part| part == "." || part == ".."),
        "Vision L0 dump rejects dot path components"
    );
    ensure!(
        path.components()
            .all(|part| matches!(part, Component::RootDir | Component::Normal(_))),
        "unsupported dump path component"
    );
    Ok(path)
}

// Linux directory handles pin every parent while walking; O_NOFOLLOW rejects
// symlinks, and writes stay beneath the retained descriptor even after rename.
#[cfg(unix)]
fn fd_path(dir: &File) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{}", dir.as_raw_fd()))
}

#[cfg(unix)]
pub(crate) fn create_root(path: &OsStr) -> Result<File> {
    ensure!(
        cfg!(all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )),
        "Vision L0 dump requires Linux ARM64/x86_64"
    );
    let path = validate_path(path)?;
    // Linux ARM64 asm/fcntl.h differs from asm-generic/fcntl.h here.
    const DIRECTORY_NOFOLLOW: i32 = if cfg!(target_arch = "aarch64") {
        0o40000 | 0o100000
    } else {
        0o200000 | 0o400000
    };
    let open_dir = |path: &Path| {
        OpenOptions::new()
            .read(true)
            .custom_flags(DIRECTORY_NOFOLLOW)
            .open(path)
    };
    let mut parent = open_dir(Path::new("/"))?;
    let names: Vec<_> = path
        .components()
        .filter_map(|part| {
            if let Component::Normal(name) = part {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    for name in &names[..names.len() - 1] {
        parent = open_dir(&fd_path(&parent).join(name))?;
    }
    let child = fd_path(&parent).join(names.last().unwrap());
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&child)
        .context("Vision L0 dump directory must not exist")?;
    Ok(open_dir(&child)?)
}

#[cfg(not(unix))]
pub(crate) fn create_root(_: &OsStr) -> Result<File> {
    anyhow::bail!("Vision L0 dump is Linux-only")
}
#[cfg(not(unix))]
fn fd_path(_: &File) -> PathBuf {
    unreachable!("create_root rejects non-Linux capture")
}

pub(crate) fn open_new(root: &File, name: &str) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options
        .open(fd_path(root).join(name))
        .with_context(|| format!("creating new capture payload {name}"))
}

pub(crate) fn write_new(root: &File, name: &str, bytes: &[u8]) -> Result<()> {
    let mut file = open_new(root, name)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
