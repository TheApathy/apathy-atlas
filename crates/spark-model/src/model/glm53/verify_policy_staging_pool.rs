// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded reuse for policy logits that must outlive the model readback lock.

use anyhow::{Context, Result, bail, ensure};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

const SELECTOR: &str = "ATLAS_GLM53_POLICY_STAGING_POOL";
pub(super) const MAX_BYTES: usize = 8 * 154_880 * 2;
static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();
static POOL: Mutex<PoolState> = Mutex::new(PoolState::new());
static ENGAGEMENT_REPORTED: AtomicBool = AtomicBool::new(false);

pub(super) fn parse_setting(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_GLM53_POLICY_STAGING_POOL must be exactly 0 or 1"),
    }
}

fn enabled() -> Result<bool> {
    match ENABLED.get_or_init(|| match std::env::var(SELECTOR) {
        Ok(value) => parse_setting(Some(&value)).map_err(str::to_owned),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{SELECTOR} must contain valid Unicode and be exactly 0 or 1"
        )),
    }) {
        Ok(enabled) => Ok(*enabled),
        Err(message) => bail!(message.clone()),
    }
}

#[derive(Default)]
pub(super) struct PoolState {
    slot: Option<Vec<u8>>,
}

impl PoolState {
    pub(super) const fn new() -> Self {
        Self { slot: None }
    }

    pub(super) fn take(&mut self) -> (Vec<u8>, bool) {
        match self.slot.take() {
            Some(bytes) => (bytes, true),
            None => (Vec::new(), false),
        }
    }

    pub(super) fn put(&mut self, mut bytes: Vec<u8>) -> bool {
        bytes.clear();
        if bytes.capacity() > MAX_BYTES || self.slot.is_some() {
            return false;
        }
        self.slot = Some(bytes);
        true
    }

    #[cfg(test)]
    pub(super) fn capacity(&self) -> Option<usize> {
        self.slot.as_ref().map(Vec::capacity)
    }
}

pub(super) struct StagingBytes {
    bytes: Vec<u8>,
    reusable: bool,
}

impl StagingBytes {
    pub(super) fn acquire(len: usize) -> Result<Self> {
        ensure!(
            len > 0 && len <= MAX_BYTES,
            "GLM policy staging buffer exceeds its fixed extent"
        );
        let reusable = enabled()?;
        let (mut bytes, hit) = if reusable {
            POOL.lock()
                .map_err(|_| anyhow::anyhow!("GLM policy staging pool mutex is poisoned"))?
                .take()
        } else {
            (Vec::new(), false)
        };
        if bytes.capacity() < len {
            bytes
                .try_reserve_exact(len)
                .context("GLM policy logits allocation failed")?;
        }
        bytes.resize(len, 0);
        if hit && !ENGAGEMENT_REPORTED.swap(true, Ordering::Relaxed) {
            eprintln!("GLM_POLICY_STAGING_POOL_ENGAGED bytes={len} max={MAX_BYTES}");
        }
        Ok(Self { bytes, reusable })
    }

    pub(super) fn len(&self) -> usize {
        self.bytes.len()
    }

    pub(super) fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

impl Drop for StagingBytes {
    fn drop(&mut self) {
        if !self.reusable {
            return;
        }
        let bytes = std::mem::take(&mut self.bytes);
        if let Ok(mut pool) = POOL.lock() {
            pool.put(bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_BYTES, PoolState, parse_setting};

    #[test]
    fn selector_is_strict_and_defaults_off() {
        assert!(!parse_setting(None).unwrap());
        assert!(!parse_setting(Some("0")).unwrap());
        assert!(parse_setting(Some("1")).unwrap());
        for invalid in ["", "true", "2", " 1"] {
            assert!(parse_setting(Some(invalid)).is_err());
        }
    }

    #[test]
    fn one_bounded_slot_is_reused_and_never_replaced() {
        let mut pool = PoolState::new();
        let (mut first, hit) = pool.take();
        assert!(!hit);
        first.try_reserve_exact(1024).unwrap();
        first.resize(1024, 7);
        assert!(pool.put(first));
        assert_eq!(pool.capacity(), Some(1024));

        let (reused, hit) = pool.take();
        assert!(hit);
        assert!(reused.is_empty());
        assert_eq!(reused.capacity(), 1024);
        assert!(pool.put(reused));
        assert!(!pool.put(Vec::with_capacity(16)));

        let too_large = Vec::with_capacity(MAX_BYTES + 1);
        let _ = pool.take();
        assert!(!pool.put(too_large));
        assert_eq!(pool.capacity(), None);
    }
}
