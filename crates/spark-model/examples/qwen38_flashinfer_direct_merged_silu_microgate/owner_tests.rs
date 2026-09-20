// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::owner::AllocationOwner;

#[test]
fn aggregate_cleanup_attempts_all_and_retains_only_failures() {
    let mut owner = AllocationOwner::for_test([DevicePtr(11), DevicePtr(22), DevicePtr(33)]);
    let mut attempted = Vec::new();
    assert!(
        owner
            .release_with(|ptr| {
                attempted.push(ptr.0);
                if ptr.0 != 22 {
                    Ok(())
                } else {
                    bail!("injected free")
                }
            })
            .is_err()
    );
    assert_eq!(attempted, [33, 22, 11]);
    assert_eq!(owner.live_for_test(), vec![22]);
    assert!(owner.release_with(|_| Ok::<_, anyhow::Error>(())).is_ok());
    assert!(owner.live_for_test().is_empty());
}

#[test]
fn copy_failure_keeps_registered_allocation_for_retry() -> Result<()> {
    let owner = AllocationOwner::for_test([DevicePtr(44)]);
    let primary = anyhow::anyhow!("injected copy failure");
    let mut failure = owner.finish_for_test::<()>(Err(primary));
    assert!(failure.retains_cleanup_authority());
    let mut attempts = 0;
    failure.retry_with(|ptr| {
        attempts += 1;
        assert_eq!(ptr.0, 44);
        Ok(())
    })?;
    assert_eq!(attempts, 1);
    assert!(!failure.retains_cleanup_authority());
    Ok(())
}

#[test]
fn persistent_free_failure_never_discards_owner() {
    let owner = AllocationOwner::for_test([DevicePtr(55), DevicePtr(66)]);
    let mut failure = owner.finish_for_test::<()>(Err(anyhow::anyhow!("launch")));
    for _ in 0..3 {
        assert!(failure.retry_with(|_| bail!("still live")).is_err());
        assert_eq!(failure.live_allocations(), 2);
    }
}
