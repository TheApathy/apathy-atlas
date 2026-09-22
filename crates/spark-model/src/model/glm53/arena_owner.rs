// SPDX-License-Identifier: AGPL-3.0-only

//! Dormant live-allocation owner for the exact GLM-5.3 B1/full-1M arena.

use std::fmt;
use std::marker::PhantomData;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{
    Glm53DsaCache, Glm53DsaDeviceIdentity, Glm53DsaStorage as RuntimeDsaStorage,
};

use super::arena::{GLM53_KNOWN_ARENA_BYTES, Glm53ArenaPlan};

const ALIGNMENT_BYTES: u64 = 256;
const B1_BF16_DSA_BYTES: usize = 12_552_258_304;

/// Owns the sole allocation and cache identity backing one B1 target arena.
#[must_use = "the GLM target arena requires explicit consuming teardown"]
pub(super) struct Glm53ArenaOwner {
    plan: Glm53ArenaPlan,
    allocation: DevicePtr,
    allocation_bytes: usize,
    dsa_cache: Glm53DsaCache,
}

/// A lifetime-bound subview. Its raw pointer stays private to the GLM driver.
#[must_use = "a GLM arena view is valid only while its owner is borrowed"]
pub(super) struct Glm53ArenaView<'owner> {
    pointer: DevicePtr,
    payload_bytes: u64,
    allocation_bytes: u64,
    _owner: PhantomData<&'owner DevicePtr>,
}

/// The eight persistent context regions plus four outer arena regions.
#[must_use = "GLM arena views borrow their allocation owner"]
pub(super) struct Glm53ArenaViews<'owner> {
    pub(super) kda_recurrent_f32: Glm53ArenaView<'owner>,
    pub(super) kda_conv_f32: Glm53ArenaView<'owner>,
    pub(super) dsa_latent: Glm53ArenaView<'owner>,
    pub(super) dsa_pooled_index: Glm53ArenaView<'owner>,
    pub(super) dsa_pool_validity: Glm53ArenaView<'owner>,
    pub(super) dsa_tail_keys: Glm53ArenaView<'owner>,
    pub(super) dsa_tail_gates: Glm53ArenaView<'owner>,
    pub(super) dsa_tail_validity: Glm53ArenaView<'owner>,
    pub(super) mhc_expanded_f32: Glm53ArenaView<'owner>,
    pub(super) t1_transaction: Glm53ArenaView<'owner>,
    pub(super) transient: Glm53ArenaView<'owner>,
    pub(super) dflash_captures: Glm53ArenaView<'owner>,
}

/// Disjoint borrow of immutable device views and the mutable logical cache.
#[must_use = "the split borrow carries the arena allocation lifetime"]
pub(super) struct Glm53ArenaSplit<'owner> {
    pub(super) views: Glm53ArenaViews<'owner>,
    pub(super) dsa_cache: &'owner mut Glm53DsaCache,
}

/// Allocation failed, or live publication failed and cleanup may still own it.
#[must_use = "arena allocation failure may retain a device allocation"]
pub(super) struct Glm53ArenaAllocationError {
    primary: anyhow::Error,
    cleanup_failure: Option<Glm53ArenaFreeError>,
}

/// Device teardown failed; this value retains the complete arena owner.
#[must_use = "arena free failure retains the allocation and must be retried"]
pub(super) struct Glm53ArenaFreeError {
    failure: anyhow::Error,
    owner: Glm53ArenaOwner,
}

impl Glm53ArenaOwner {
    pub(super) fn allocate(
        gpu: &dyn GpuBackend,
        plan: Glm53ArenaPlan,
    ) -> std::result::Result<Self, Glm53ArenaAllocationError> {
        let allocation_bytes = validate_plan(plan)
            .and_then(|()| {
                usize::try_from(plan.known_bytes)
                    .context("GLM target arena byte count does not fit usize")
            })
            .map_err(Glm53ArenaAllocationError::without_owner)?;
        let dsa_cache = Glm53DsaCache::new(1, RuntimeDsaStorage::Bf16)
            .and_then(|cache| {
                ensure!(
                    cache.layout().batch == 1
                        && cache.layout().storage == RuntimeDsaStorage::Bf16
                        && cache.layout().total_bytes == B1_BF16_DSA_BYTES,
                    "GLM target B1 BF16 DSA cache layout drift"
                );
                Ok(cache)
            })
            .map_err(Glm53ArenaAllocationError::without_owner)?;
        let allocation = gpu
            .alloc(allocation_bytes)
            .context("GLM target arena allocation failed")
            .map_err(Glm53ArenaAllocationError::without_owner)?;
        let owner = Self {
            plan,
            allocation,
            allocation_bytes,
            dsa_cache,
        };
        if let Err(primary) = owner.validate_live_range() {
            return Err(Glm53ArenaAllocationError::after_allocation(
                primary, owner, gpu,
            ));
        }
        if let Err(primary) = gpu
            .memset(owner.allocation, 0, owner.allocation_bytes)
            .context("GLM target arena synchronous zero failed")
        {
            return Err(Glm53ArenaAllocationError::after_allocation(
                primary, owner, gpu,
            ));
        }
        Ok(owner)
    }

    pub(super) fn plan(&self) -> Glm53ArenaPlan {
        self.plan
    }

    pub(super) fn device_identity(&self) -> Glm53DsaDeviceIdentity {
        self.dsa_cache.device_identity()
    }

    pub(super) fn split(&mut self) -> Result<Glm53ArenaSplit<'_>> {
        self.validate_live_range()?;
        let views = Glm53ArenaViews::new(&self.allocation, self.plan)?;
        Ok(Glm53ArenaSplit {
            views,
            dsa_cache: &mut self.dsa_cache,
        })
    }

    pub(super) fn free(self, gpu: &dyn GpuBackend) -> std::result::Result<(), Glm53ArenaFreeError> {
        match gpu.free(self.allocation) {
            Ok(()) => Ok(()),
            Err(failure) => Err(Glm53ArenaFreeError {
                failure: failure.context("GLM target arena free failed"),
                owner: self,
            }),
        }
    }

    fn validate_live_range(&self) -> Result<()> {
        ensure!(
            self.plan.known_bytes == GLM53_KNOWN_ARENA_BYTES
                && u64::try_from(self.allocation_bytes)? == GLM53_KNOWN_ARENA_BYTES,
            "GLM target arena allocation extent drift"
        );
        ensure!(
            !self.allocation.is_null() && self.allocation.0 % ALIGNMENT_BYTES == 0,
            "GLM target arena base must be non-null and 256-byte aligned"
        );
        self.allocation
            .0
            .checked_add(self.plan.known_bytes)
            .context("GLM target arena device address overflow")?;
        Ok(())
    }
}

impl<'owner> Glm53ArenaViews<'owner> {
    fn new(allocation: &'owner DevicePtr, plan: Glm53ArenaPlan) -> Result<Self> {
        let context = plan.context;
        let context_view = |region: crate::weight_loader::Glm53ContextRegion| {
            Glm53ArenaView::new(
                allocation,
                plan.known_bytes,
                region.offset_bytes,
                region.payload_bytes,
                region.allocation_bytes,
            )
        };
        let arena_view = |region: super::arena::Glm53ArenaRegion| {
            Glm53ArenaView::new(
                allocation,
                plan.known_bytes,
                region.offset_bytes,
                region.allocation_bytes,
                region.allocation_bytes,
            )
        };
        Ok(Self {
            kda_recurrent_f32: context_view(context.kda_recurrent_f32)?,
            kda_conv_f32: context_view(context.kda_conv_f32)?,
            dsa_latent: context_view(context.dsa_latent)?,
            dsa_pooled_index: context_view(context.dsa_pooled_index)?,
            dsa_pool_validity: context_view(context.dsa_pool_validity)?,
            dsa_tail_keys: context_view(context.dsa_tail_keys)?,
            dsa_tail_gates: context_view(context.dsa_tail_gates)?,
            dsa_tail_validity: context_view(context.dsa_tail_validity)?,
            mhc_expanded_f32: arena_view(plan.mhc_expanded_f32)?,
            t1_transaction: arena_view(plan.t1_transaction)?,
            transient: arena_view(plan.transient)?,
            dflash_captures: arena_view(plan.dflash_captures)?,
        })
    }

    pub(super) fn regions(&self) -> [&Glm53ArenaView<'owner>; 12] {
        [
            &self.kda_recurrent_f32,
            &self.kda_conv_f32,
            &self.dsa_latent,
            &self.dsa_pooled_index,
            &self.dsa_pool_validity,
            &self.dsa_tail_keys,
            &self.dsa_tail_gates,
            &self.dsa_tail_validity,
            &self.mhc_expanded_f32,
            &self.t1_transaction,
            &self.transient,
            &self.dflash_captures,
        ]
    }
}

impl<'owner> Glm53ArenaView<'owner> {
    fn new(
        allocation: &'owner DevicePtr,
        arena_bytes: u64,
        offset_bytes: u64,
        payload_bytes: u64,
        allocation_bytes: u64,
    ) -> Result<Self> {
        ensure!(
            offset_bytes % ALIGNMENT_BYTES == 0
                && allocation_bytes % ALIGNMENT_BYTES == 0
                && payload_bytes <= allocation_bytes,
            "GLM target arena subview geometry drift"
        );
        let offset_end = offset_bytes
            .checked_add(allocation_bytes)
            .context("GLM target arena subview extent overflow")?;
        ensure!(
            offset_end <= arena_bytes,
            "GLM target arena subview is out of range"
        );
        let address = allocation
            .0
            .checked_add(offset_bytes)
            .context("GLM target arena subview address overflow")?;
        let address_end = address
            .checked_add(allocation_bytes)
            .context("GLM target arena subview address extent overflow")?;
        let arena_end = allocation
            .0
            .checked_add(arena_bytes)
            .context("GLM target arena address overflow")?;
        ensure!(
            address_end <= arena_end,
            "GLM target arena subview address escapes owner"
        );
        Ok(Self {
            pointer: DevicePtr(address),
            payload_bytes,
            allocation_bytes,
            _owner: PhantomData,
        })
    }

    pub(super) fn device_ptr(&self) -> DevicePtr {
        self.pointer
    }

    pub(super) fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub(super) fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }
}

impl Glm53ArenaAllocationError {
    fn without_owner(primary: anyhow::Error) -> Self {
        Self {
            primary,
            cleanup_failure: None,
        }
    }

    fn after_allocation(
        primary: anyhow::Error,
        owner: Glm53ArenaOwner,
        gpu: &dyn GpuBackend,
    ) -> Self {
        let cleanup_failure = owner.free(gpu).err();
        Self {
            primary,
            cleanup_failure,
        }
    }

    pub(super) fn primary(&self) -> &anyhow::Error {
        &self.primary
    }

    pub(super) fn cleanup_failure(&self) -> Option<&Glm53ArenaFreeError> {
        self.cleanup_failure.as_ref()
    }

    pub(super) fn retry_cleanup(
        self,
        gpu: &dyn GpuBackend,
    ) -> std::result::Result<anyhow::Error, Self> {
        let Self {
            primary,
            cleanup_failure,
        } = self;
        let Some(cleanup_failure) = cleanup_failure else {
            return Ok(primary);
        };
        match cleanup_failure.retry(gpu) {
            Ok(()) => Ok(primary),
            Err(cleanup_failure) => Err(Self {
                primary,
                cleanup_failure: Some(cleanup_failure),
            }),
        }
    }
}

impl Glm53ArenaFreeError {
    pub(super) fn failure(&self) -> &anyhow::Error {
        &self.failure
    }

    pub(super) fn owner(&self) -> &Glm53ArenaOwner {
        &self.owner
    }

    pub(super) fn retry(
        self,
        gpu: &dyn GpuBackend,
    ) -> std::result::Result<(), Glm53ArenaFreeError> {
        self.owner.free(gpu)
    }
}

impl fmt::Debug for Glm53ArenaOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53ArenaOwner")
            .field("known_bytes", &self.plan.known_bytes)
            .field("context_positions", &self.plan.context.positions)
            .finish()
    }
}

impl fmt::Debug for Glm53ArenaView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53ArenaView")
            .field("payload_bytes", &self.payload_bytes)
            .field("allocation_bytes", &self.allocation_bytes)
            .finish()
    }
}

impl fmt::Debug for Glm53ArenaAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53ArenaAllocationError")
            .field("primary", &self.primary)
            .field("cleanup_failure", &self.cleanup_failure)
            .finish()
    }
}

impl fmt::Display for Glm53ArenaAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GLM target arena allocation failed: {:#}",
            self.primary
        )?;
        if let Some(cleanup) = &self.cleanup_failure {
            write!(formatter, "; cleanup also failed: {cleanup}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Glm53ArenaFreeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53ArenaFreeError")
            .field("failure", &self.failure)
            .field("owner", &self.owner)
            .finish()
    }
}

impl fmt::Display for Glm53ArenaFreeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GLM target arena free failed: {:#}",
            self.failure
        )
    }
}

fn validate_plan(plan: Glm53ArenaPlan) -> Result<()> {
    plan.validate()?;
    ensure!(
        plan == Glm53ArenaPlan::exact_full_1m_b1()?
            && plan.known_bytes == GLM53_KNOWN_ARENA_BYTES
            && plan.context.batch == 1
            && plan.context.positions == 1_048_576
            && plan.context.storage == crate::weight_loader::Glm53DsaStorage::BF16,
        "GLM target arena owner requires the exact B1/full-1M/BF16 plan"
    );
    Ok(())
}

#[cfg(test)]
#[path = "arena_owner_tests.rs"]
mod tests;
