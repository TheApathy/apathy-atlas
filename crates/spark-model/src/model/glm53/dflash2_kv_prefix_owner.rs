// SPDX-License-Identifier: AGPL-3.0-only
//! Stable host staging and the consuming shutdown quarantine boundary.

use super::kv_prefix::{KvPrefix, KvPrefixIo};
use super::projection_contract::{ProjectionBinding, ProjectionFamily};
use anyhow::{Context, Result};

#[derive(Default)]
pub(super) struct ProposalHost {
    pub(super) anchor: [u8; 4],
    pub(super) path: [u8; 28],
    pub(super) status: [u8; 4],
}

pub(super) struct KvPrefixState {
    pub(super) prefix: KvPrefix,
    pub(super) host: Option<Box<ProposalHost>>,
    projection: ProjectionBinding,
}

impl KvPrefixState {
    pub(super) fn new(prefix: KvPrefix) -> Self {
        Self {
            prefix,
            host: Some(Box::default()),
            projection: ProjectionBinding::new(),
        }
    }
    pub(super) fn select_projection(&mut self, family: ProjectionFamily) -> Result<()> {
        self.projection.select(family, &self.prefix)
    }
    pub(super) fn admit_projection(&self, family: ProjectionFamily) -> Result<()> {
        self.projection.admit(family, &self.prefix)
    }
    pub(super) fn reset(&mut self, io: &mut dyn KvPrefixIo) -> Result<()> {
        self.projection.reset(&mut self.prefix, io)
    }
}

impl Drop for KvPrefixState {
    fn drop(&mut self) {
        // Explicit model shutdown drains before dropping any backend/device
        // owner. This final host guard also prevents async host UAF if a caller
        // abandons the runtime without following that consuming API.
        if self.prefix.pending() {
            if let Some(host) = self.host.take() {
                std::mem::forget(host);
                tracing::error!(
                    "GLM DFlash2 pending host staging quarantined until process teardown"
                );
            }
        }
    }
}

/// Return ownership only after an observed drain. A consuming error cannot
/// return its owner under the existing API, so retain *all* of it, including
/// the outer model's backend lease. Never unwind through its field destructors
/// after a panicking completion callback. This is explicit quarantine, not a
/// successful release, and provides no retry handle under that legacy API.
pub fn retain_until_drained<T>(owner: T, drain: impl FnOnce(&T) -> Result<()>) -> Result<T> {
    let completion = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drain(&owner)));
    let failure = match completion {
        Ok(Ok(())) => return Ok(owner),
        Ok(Err(error)) => error,
        Err(_) => anyhow::anyhow!("GLM DFlash2 shutdown drain panicked"),
    };
    std::mem::forget(owner);
    Err(failure).context(
        "GLM shutdown refused: enclosing owner quarantined until process teardown; backend must remain alive",
    )
}
