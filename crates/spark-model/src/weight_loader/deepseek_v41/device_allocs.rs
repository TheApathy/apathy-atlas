// SPDX-License-Identifier: AGPL-3.0-only

//! Owned device allocations that are FREED when their owner drops.
//!
//! The CB3 arena is 71.7 GB at the served keep and the routed MoE's scratch is ~0.6 GB. They
//! used to have no `Drop`, and the MoE borrowed the backend (`Cb3RoutedMoe<'a>`), so the served
//! model leaked the backend to `'static` and dropping DeepSeek freed none of it — a TUI model
//! swap could never get the memory back. Owning an `Arc` to the backend lets `Drop` free.
//!
//! Every allocation is recorded the moment it is made, so a constructor that fails half-way
//! frees what it already took instead of leaking it: the guard is dropped with the error.

use std::sync::Arc;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// A backend handle that can outlive the constructor that received it.
pub type SharedGpu = Arc<dyn GpuBackend>;

/// Device allocations freed on drop through an owned backend handle.
///
/// `owner: None` is the explicit NON-owning mode for callers that only have a `&dyn
/// GpuBackend` (the legacy `load_layers` path): allocations are recorded but never freed,
/// exactly the old behaviour, and the choice is visible at the call site.
pub struct DeviceAllocs {
    owner: Option<SharedGpu>,
    ptrs: Vec<DevicePtr>,
}

impl DeviceAllocs {
    pub fn owned(gpu: SharedGpu) -> Self {
        Self { owner: Some(gpu), ptrs: Vec::new() }
    }

    /// Record but never free. Only for callers without an owned backend handle.
    pub fn unowned() -> Self {
        Self { owner: None, ptrs: Vec::new() }
    }

    /// Allocate through `gpu` and record the pointer for freeing.
    pub fn alloc(&mut self, gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
        let ptr = gpu.alloc(bytes)?;
        self.ptrs.push(ptr);
        Ok(ptr)
    }

    /// Record a pointer allocated elsewhere, so it is freed with the rest.
    pub fn adopt(&mut self, ptr: DevicePtr) {
        self.ptrs.push(ptr);
    }

    pub fn len(&self) -> usize {
        self.ptrs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ptrs.is_empty()
    }

    /// The owning backend handle, if any (for owners of non-allocation resources, e.g. graphs).
    pub fn owner(&self) -> Option<SharedGpu> {
        self.owner.clone()
    }

    pub fn is_owned(&self) -> bool {
        self.owner.is_some()
    }
}

impl Drop for DeviceAllocs {
    fn drop(&mut self) {
        let Some(gpu) = &self.owner else {
            return;
        };
        // Drop can run on a thread other than the one that allocated.
        if let Err(e) = gpu.bind_to_thread() {
            tracing::warn!("DeviceAllocs: bind_to_thread failed before freeing {} allocations: {e}", self.ptrs.len());
        }
        for ptr in self.ptrs.drain(..).rev() {
            if let Err(e) = gpu.free(ptr) {
                tracing::warn!("DeviceAllocs: freeing {:#x} failed: {e}", ptr.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// Owned allocations are freed on drop. NEGATIVE CONTROL: the unowned mode (the old
    /// behaviour) leaves them allocated — so the count below can fail, and does for the
    /// mode that leaks.
    #[test]
    fn owned_allocations_are_freed_on_drop_and_unowned_ones_are_not() {
        let mock = Arc::new(MockGpuBackend::new());
        let shared: SharedGpu = mock.clone();

        let mut owned = DeviceAllocs::owned(shared.clone());
        for bytes in [64, 128, 256] {
            owned.alloc(shared.as_ref(), bytes).unwrap();
        }
        let adopted = shared.alloc(32).unwrap();
        owned.adopt(adopted);
        assert_eq!(mock.alloc_count(), 4);
        drop(owned);
        assert_eq!(mock.alloc_count(), 0, "an owned guard must free everything it recorded");

        let mut leaky = DeviceAllocs::unowned();
        leaky.alloc(shared.as_ref(), 64).unwrap();
        drop(leaky);
        assert_eq!(mock.alloc_count(), 1, "the unowned mode must NOT free (and so leaks)");
    }

    /// A constructor that fails half-way frees what it already took: the guard is dropped
    /// with the error.
    #[test]
    fn a_failed_constructor_frees_its_partial_allocations() {
        let mock = Arc::new(MockGpuBackend::new());
        let shared: SharedGpu = mock.clone();
        let build = || -> Result<DeviceAllocs> {
            let mut allocs = DeviceAllocs::owned(shared.clone());
            allocs.alloc(shared.as_ref(), 1024)?;
            allocs.alloc(shared.as_ref(), 1024)?;
            anyhow::bail!("simulated failure after two allocations");
        };
        assert!(build().is_err());
        assert_eq!(mock.alloc_count(), 0);
    }

    /// What the served model needs to hold the MoE in a `Box<dyn V41RoutedMoe + Send + Sync>`.
    #[test]
    fn the_routed_moe_is_static_send_and_sync() {
        fn assert_static_send_sync<T: Send + Sync + 'static>() {}
        assert_static_send_sync::<super::super::moe_forward::Cb3RoutedMoe>();
        assert_static_send_sync::<super::super::cb3_arena::Cb3ExpertArena>();
    }
}
