// SPDX-License-Identifier: AGPL-3.0-only

use std::fmt;

use anyhow::{Result, anyhow};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Sole move-only authority for allocations made by one raw-gate case.
pub(super) struct AllocationOwner {
    live: Vec<DevicePtr>,
}

impl AllocationOwner {
    pub(super) fn new() -> Self {
        Self { live: Vec::new() }
    }

    pub(super) fn allocate(&mut self, gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
        self.live.try_reserve(1)?;
        let ptr = gpu.alloc(bytes)?;
        self.live.push(ptr);
        Ok(ptr)
    }

    pub(super) fn release_all(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.release_with(|ptr| gpu.free(ptr))
    }

    pub(super) fn release_with(
        &mut self,
        mut free: impl FnMut(DevicePtr) -> Result<()>,
    ) -> Result<()> {
        let mut cursor = self.live.len();
        let mut first_error = None;
        let mut failures = 0usize;
        while cursor != 0 {
            cursor -= 1;
            let ptr = self.live[cursor];
            match free(ptr) {
                Ok(()) => {
                    self.live.remove(cursor);
                }
                Err(error) => {
                    failures += 1;
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        if let Some(error) = first_error {
            Err(error.context(format!(
                "failed to free {failures} allocation(s); every failed owner retained"
            )))
        } else {
            Ok(())
        }
    }

    pub(super) fn finish<T>(
        mut self,
        gpu: &dyn GpuBackend,
        outcome: Result<T>,
    ) -> std::result::Result<T, RetainedGpuFailure> {
        let cleanup = self.release_all(gpu);
        match (outcome, cleanup) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(primary), Ok(())) => Err(RetainedGpuFailure::new(primary, None)),
            (Ok(_), Err(cleanup)) => Err(RetainedGpuFailure::new(cleanup, Some(self))),
            (Err(primary), Err(cleanup)) => Err(RetainedGpuFailure::new(
                primary.context(format!("cleanup also failed: {cleanup:#}")),
                Some(self),
            )),
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(ptrs: impl IntoIterator<Item = DevicePtr>) -> Self {
        Self {
            live: ptrs.into_iter().collect(),
        }
    }

    #[cfg(test)]
    pub(super) fn live_for_test(&self) -> Vec<u64> {
        self.live.iter().map(|ptr| ptr.0).collect()
    }

    #[cfg(test)]
    pub(super) fn finish_for_test<T>(self, outcome: Result<T>) -> RetainedGpuFailure {
        let primary = outcome.err().expect("test outcome must fail");
        RetainedGpuFailure::new(primary, Some(self))
    }
}

/// Not an `Error`: generic anyhow conversion must not erase retained ownership.
pub(super) struct RetainedGpuFailure {
    primary: anyhow::Error,
    owner: Option<AllocationOwner>,
}

impl RetainedGpuFailure {
    fn new(primary: anyhow::Error, owner: Option<AllocationOwner>) -> Self {
        Self { primary, owner }
    }

    pub(super) fn retry_cleanup(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.retry_with(|ptr| gpu.free(ptr))
    }

    pub(super) fn retry_with(&mut self, free: impl FnMut(DevicePtr) -> Result<()>) -> Result<()> {
        let owner = self
            .owner
            .as_mut()
            .ok_or_else(|| anyhow!("no live allocations"))?;
        let result = owner.release_with(free);
        if owner.live.is_empty() {
            self.owner = None;
        }
        result
    }

    pub(super) fn retains_cleanup_authority(&self) -> bool {
        self.owner.is_some()
    }

    pub(super) fn live_allocations(&self) -> usize {
        self.owner.as_ref().map_or(0, |owner| owner.live.len())
    }
}

impl fmt::Debug for RetainedGpuFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedGpuFailure")
            .field("primary", &format_args!("{:#}", self.primary))
            .field("live_allocations", &self.live_allocations())
            .finish()
    }
}
