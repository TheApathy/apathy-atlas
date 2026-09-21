// SPDX-License-Identifier: AGPL-3.0-only

//! Where Atlas keeps recipes, artifacts and run history: `~/.atlas`.
//!
//! PORTED, NOT DEPENDED ON. Upstream resolves this inside its `avarok-plugin`
//! crate (`ArtifactStore::discover` -> `AvarokHome::resolve`), and the TUI's
//! Library tab is the only thing in the ported set that needs it. Taking a
//! 260-file plugin crate to resolve one directory would have dragged the whole
//! benchmark framework in behind it, so the ~30 lines that matter live here.
//!
//! THE NAME PRECEDENCE IS DELIBERATELY THE INVERSE OF UPSTREAM'S. Upstream
//! treats `~/.atlas` as the LEGACY name and `~/.avarok` as current. This is the
//! Atlas engine, so `~/.atlas` is the current name — and `~/.avarok` is
//! honoured only when `~/.atlas` does not exist, so anyone who has run an
//! upstream build keeps their artifacts instead of silently starting empty.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// Where the root came from — worth keeping, because two commands disagreeing
/// about the root is otherwise invisible to an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomeSource {
    /// `ATLAS_HOME` was set.
    Env,
    /// `$HOME/.atlas`.
    HomeDefault,
    /// `$HOME/.avarok`, because `~/.atlas` does not exist yet.
    UpstreamHomeDefault,
}

#[derive(Clone, Debug)]
pub struct AtlasHome {
    pub root: PathBuf,
    pub source: HomeSource,
}

impl AtlasHome {
    pub fn resolve() -> Result<Self> {
        resolve_from(std::env::var_os("ATLAS_HOME"), std::env::var_os("HOME"))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// One line naming the root and where it came from.
    pub fn describe(&self) -> String {
        let src = match self.source {
            HomeSource::Env => "ATLAS_HOME",
            HomeSource::HomeDefault => "$HOME/.atlas",
            HomeSource::UpstreamHomeDefault => "$HOME/.avarok (upstream layout)",
        };
        format!("{} ({src})", self.root.display())
    }
}

/// [`AtlasHome::resolve`] over explicit inputs, so the rules can be tested.
///
/// Pure over the ENVIRONMENT on purpose: `set_var` is unsafe and
/// process-global, so a test that mutated `HOME` could race another test's read
/// and produce exactly the intermittent failure this is meant to avoid.
pub(crate) fn resolve_from(
    atlas_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Result<AtlasHome> {
    if let Some(explicit) = atlas_home {
        let root = PathBuf::from(explicit);
        if root.as_os_str().is_empty() {
            bail!("ATLAS_HOME is set but empty");
        }
        return Ok(AtlasHome {
            root,
            source: HomeSource::Env,
        });
    }
    let home = home
        .map(PathBuf::from)
        .filter(|h| !h.as_os_str().is_empty())
        .context("neither ATLAS_HOME nor HOME is set — cannot place ~/.atlas")?;
    let root = home.join(".atlas");
    // Only while our own name is absent: once `~/.atlas` exists it always wins,
    // so a box that has both never silently reads the upstream one.
    if !root.exists() && home.join(".avarok").is_dir() {
        return Ok(AtlasHome {
            root: home.join(".avarok"),
            source: HomeSource::UpstreamHomeDefault,
        });
    }
    Ok(AtlasHome {
        root,
        source: HomeSource::HomeDefault,
    })
}

/// Can this process actually use the home? `None` means yes.
///
/// PROBES BY WRITING, not by reading a mode bit — upstream's insight and worth
/// carrying: ownership, ACLs, a read-only mount and a full disk all present
/// differently in the metadata and identically in the thing that matters,
/// which is whether the next write lands.
pub fn check_usable(root: &Path) -> Option<String> {
    if root.exists() && !root.is_dir() {
        return Some(format!("{} exists and is not a directory", root.display()));
    }
    if !root.exists() {
        if let Err(e) = std::fs::create_dir_all(root) {
            return Some(format!("could not create {}: {e}", root.display()));
        }
    }
    let probe = root.join(".atlas-write-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            None
        }
        Err(e) => Some(format!("{} is not writable: {e}", root.display())),
    }
}

#[cfg(test)]
#[path = "atlas_home_tests.rs"]
mod tests;
