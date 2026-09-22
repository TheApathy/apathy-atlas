// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Mutex;

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;

const BASE: u64 = 0x1000_0000_0000;
const OWNER_SOURCE_SHA256: &str =
    "c23fb52b921b68399e2c0f95d1e049fc46aabca2eab96b8753126378ea16df22";

#[path = "t1_state_transaction_sha256.rs"]
mod source_sha256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Call {
    Alloc(usize),
    Memset(DevicePtr, u8, usize),
    Free(DevicePtr),
}

struct MetadataGpu {
    pointer: DevicePtr,
    fail_alloc: bool,
    fail_memset: bool,
    free_failures: Mutex<usize>,
    calls: Mutex<Vec<Call>>,
}

impl MetadataGpu {
    fn new(pointer: u64) -> Self {
        Self {
            pointer: DevicePtr(pointer),
            fail_alloc: false,
            fail_memset: false,
            free_failures: Mutex::new(0),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl GpuBackend for MetadataGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.calls.lock().unwrap().push(Call::Alloc(bytes));
        if self.fail_alloc {
            bail!("injected metadata allocation failure")
        }
        Ok(self.pointer)
    }

    fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
        bail!("managed allocation is forbidden")
    }

    fn free(&self, pointer: DevicePtr) -> Result<()> {
        self.calls.lock().unwrap().push(Call::Free(pointer));
        let mut failures = self.free_failures.lock().unwrap();
        if *failures != 0 {
            *failures -= 1;
            bail!("injected metadata free failure")
        }
        Ok(())
    }

    fn copy_h2d(&self, _source: &[u8], _destination: DevicePtr) -> Result<()> {
        bail!("copy is unused")
    }

    fn copy_d2h(&self, _source: DevicePtr, _destination: &mut [u8]) -> Result<()> {
        bail!("copy is unused")
    }

    fn copy_d2d(&self, _source: DevicePtr, _destination: DevicePtr, _bytes: usize) -> Result<()> {
        bail!("copy is unused")
    }

    fn launch(
        &self,
        _function: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _parameters: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        bail!("launch is unused")
    }

    fn synchronize(&self, _stream: u64) -> Result<()> {
        bail!("synchronize is unused")
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn kernel(&self, _module: &str, _function: &str) -> Result<KernelHandle> {
        bail!("kernel lookup is unused")
    }

    fn memset(&self, pointer: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.calls
            .lock()
            .unwrap()
            .push(Call::Memset(pointer, value, bytes));
        if self.fail_memset {
            bail!("injected metadata memset failure")
        }
        Ok(())
    }

    fn memset_async(
        &self,
        _pointer: DevicePtr,
        _value: u8,
        _bytes: usize,
        _stream: u64,
    ) -> Result<()> {
        bail!("asynchronous zero is forbidden")
    }

    fn total_memory(&self) -> Result<usize> {
        Ok(usize::MAX)
    }

    fn free_memory(&self) -> Result<usize> {
        Ok(usize::MAX)
    }
}

fn exact() -> Glm53ArenaPlan {
    Glm53ArenaPlan::exact_full_1m_b1().unwrap()
}

#[test]
fn exact_owner_allocates_then_synchronously_zeros_and_exposes_twelve_views() {
    let gpu = MetadataGpu::new(BASE);
    let mut owner = Glm53ArenaOwner::allocate(&gpu, exact()).unwrap();
    assert_eq!(owner.plan(), exact());
    assert_eq!(
        gpu.calls(),
        [
            Call::Alloc(13_163_242_496),
            Call::Memset(DevicePtr(BASE), 0, 13_163_242_496),
        ]
    );
    assert_eq!(
        format!("{owner:?}"),
        "Glm53ArenaOwner { known_bytes: 13163242496, context_positions: 1048576 }"
    );

    let split = owner.split().unwrap();
    assert_eq!(split.dsa_cache.layout().batch, 1);
    assert_eq!(split.dsa_cache.layout().storage, RuntimeDsaStorage::Bf16);
    let regions = split.views.regions();
    assert_eq!(regions.len(), 12);
    let expected = [
        (0, 285_212_672, 285_212_672),
        (285_212_672, 26_738_688, 26_738_688),
        (311_951_360, 11_811_160_064, 11_811_160_064),
        (12_123_111_424, 738_197_504, 738_197_504),
        (12_861_308_928, 2_883_584, 2_883_584),
        (12_864_192_512, 8_448, 8_448),
        (12_864_200_960, 8_448, 8_448),
        (12_864_209_408, 33, 256),
        (12_864_209_664, 141_557_760, 141_557_760),
        (13_005_767_424, 156_008_192, 156_008_192),
        (13_161_775_616, 1_139_200, 1_139_200),
        (13_162_914_816, 327_680, 327_680),
    ];
    for (view, (offset, payload, allocation)) in regions.into_iter().zip(expected) {
        assert_eq!(view.device_ptr(), DevicePtr(BASE + offset));
        assert_eq!(view.payload_bytes(), payload);
        assert_eq!(view.allocation_bytes(), allocation);
        assert!(!format!("{view:?}").contains(&format!("{:x}", BASE + offset)));
    }
    for pair in split.views.regions().windows(2) {
        assert_eq!(
            pair[0].device_ptr().0 + pair[0].allocation_bytes(),
            pair[1].device_ptr().0
        );
    }
    assert_eq!(
        split.views.dflash_captures.device_ptr().0 + split.views.dflash_captures.allocation_bytes(),
        BASE + GLM53_KNOWN_ARENA_BYTES
    );
    owner.free(&gpu).unwrap();
    assert_eq!(gpu.calls().last(), Some(&Call::Free(DevicePtr(BASE))));
}

#[test]
fn preflight_and_every_post_alloc_validation_fail_before_live_publication() {
    let mut forged = exact();
    forged.known_bytes -= 256;
    let gpu = MetadataGpu::new(BASE);
    let error = Glm53ArenaOwner::allocate(&gpu, forged).unwrap_err();
    assert!(error.cleanup_failure().is_none());
    assert!(gpu.calls().is_empty());

    let mut alloc_failure = MetadataGpu::new(BASE);
    alloc_failure.fail_alloc = true;
    let error = Glm53ArenaOwner::allocate(&alloc_failure, exact()).unwrap_err();
    assert!(error.primary().to_string().contains("allocation"));
    assert!(error.cleanup_failure().is_none());
    assert_eq!(alloc_failure.calls(), [Call::Alloc(13_163_242_496)]);

    for pointer in [0, BASE + 1, u64::MAX - GLM53_KNOWN_ARENA_BYTES + 1] {
        let gpu = MetadataGpu::new(pointer);
        let error = Glm53ArenaOwner::allocate(&gpu, exact()).unwrap_err();
        assert!(error.cleanup_failure().is_none());
        assert_eq!(
            gpu.calls(),
            [Call::Alloc(13_163_242_496), Call::Free(DevicePtr(pointer))]
        );
    }
}

#[test]
fn zero_and_free_failures_retain_exact_owner_until_consuming_retry_succeeds() {
    let mut gpu = MetadataGpu::new(BASE);
    gpu.fail_memset = true;
    *gpu.free_failures.lock().unwrap() = 2;
    let error = Glm53ArenaOwner::allocate(&gpu, exact()).unwrap_err();
    assert!(error.primary().to_string().contains("synchronous zero"));
    let retained = error.cleanup_failure().unwrap();
    assert_eq!(retained.owner().plan().known_bytes, GLM53_KNOWN_ARENA_BYTES);
    assert!(retained.failure().to_string().contains("free"));
    let error = error.retry_cleanup(&gpu).unwrap_err();
    assert!(error.cleanup_failure().is_some());
    let primary = error.retry_cleanup(&gpu).unwrap();
    assert!(primary.to_string().contains("synchronous zero"));
    assert_eq!(
        gpu.calls(),
        [
            Call::Alloc(13_163_242_496),
            Call::Memset(DevicePtr(BASE), 0, 13_163_242_496),
            Call::Free(DevicePtr(BASE)),
            Call::Free(DevicePtr(BASE)),
            Call::Free(DevicePtr(BASE)),
        ]
    );

    let gpu = MetadataGpu::new(BASE);
    *gpu.free_failures.lock().unwrap() = 1;
    let owner = Glm53ArenaOwner::allocate(&gpu, exact()).unwrap();
    let failure = owner.free(&gpu).unwrap_err();
    failure.retry(&gpu).unwrap();
    assert_eq!(
        gpu.calls()
            .iter()
            .filter(|call| matches!(call, Call::Free(_)))
            .count(),
        2
    );
}

#[test]
fn reused_address_has_distinct_cache_identity_and_poison_does_not_block_teardown() {
    let gpu = MetadataGpu::new(BASE);
    let first = Glm53ArenaOwner::allocate(&gpu, exact()).unwrap();
    let first_identity = first.device_identity();
    first.free(&gpu).unwrap();

    let mut second = Glm53ArenaOwner::allocate(&gpu, exact()).unwrap();
    let second_identity = second.device_identity();
    assert!(first_identity != second_identity);
    assert_eq!(second.device_identity(), second_identity);
    {
        let split = second.split().unwrap();
        assert_eq!(split.dsa_cache.device_identity(), second_identity);
        assert_eq!(split.dsa_cache.device_identity(), second_identity);
        let handle = split.dsa_cache.claim_sequence().unwrap();
        split.dsa_cache.poison_sequence(handle).unwrap();
        assert!(split.dsa_cache.sequence_is_poisoned(handle).unwrap());
    }
    second.free(&gpu).unwrap();
}

#[test]
fn normal_and_pretty_debug_are_canonical_and_omit_two_pointer_sentinels() {
    for pointer in [BASE, 0x2000_0000_0000] {
        let assert_redacted = |output: &str| {
            assert!(!output.contains(&pointer.to_string()));
            assert!(!output.contains(&format!("{pointer:x}")));
            assert!(!output.contains(&format!("{pointer:X}")));
        };
        let gpu = MetadataGpu::new(pointer);
        let mut owner = Glm53ArenaOwner::allocate(&gpu, exact()).unwrap();
        let owner_normal = format!("{owner:?}");
        let owner_pretty = format!("{owner:#?}");
        assert_eq!(
            owner_normal,
            "Glm53ArenaOwner { known_bytes: 13163242496, context_positions: 1048576 }"
        );
        assert_eq!(
            owner_pretty,
            "Glm53ArenaOwner {\n    known_bytes: 13163242496,\n    context_positions: 1048576,\n}"
        );
        let (view_normal, view_pretty) = {
            let split = owner.split().unwrap();
            let view = &split.views.kda_recurrent_f32;
            (format!("{view:?}"), format!("{view:#?}"))
        };
        assert_eq!(
            view_normal,
            "Glm53ArenaView { payload_bytes: 285212672, allocation_bytes: 285212672 }"
        );
        assert_eq!(
            view_pretty,
            "Glm53ArenaView {\n    payload_bytes: 285212672,\n    allocation_bytes: 285212672,\n}"
        );
        for output in [owner_normal, owner_pretty, view_normal, view_pretty] {
            assert_redacted(&output);
        }
        owner.free(&gpu).unwrap();

        let free_gpu = MetadataGpu::new(pointer);
        *free_gpu.free_failures.lock().unwrap() = 1;
        let owner = Glm53ArenaOwner::allocate(&free_gpu, exact()).unwrap();
        let free_error = owner.free(&free_gpu).unwrap_err();
        assert_redacted(&format!("{free_error:?}"));
        assert_redacted(&format!("{free_error:#?}"));
        free_error.retry(&free_gpu).unwrap();

        let mut allocation_gpu = MetadataGpu::new(pointer);
        allocation_gpu.fail_memset = true;
        *allocation_gpu.free_failures.lock().unwrap() = 1;
        let allocation_error = Glm53ArenaOwner::allocate(&allocation_gpu, exact()).unwrap_err();
        assert_redacted(&format!("{allocation_error:?}"));
        assert_redacted(&format!("{allocation_error:#?}"));
        allocation_error.retry_cleanup(&allocation_gpu).unwrap();
    }
}

fn owner_source_is_exact(source: &str) -> bool {
    source_sha256::hex(source_sha256::digest(source.as_bytes())) == OWNER_SOURCE_SHA256
}

fn structural_source_contract(source: &str) -> bool {
    let needles = [
        "gpu\n            .alloc(allocation_bytes)",
        ".memset(owner.allocation, 0, owner.allocation_bytes)",
        ".checked_add(self.plan.known_bytes)",
        ".checked_add(offset_bytes)",
        ".checked_add(allocation_bytes)",
        "let cleanup_failure = owner.free(gpu).err();",
        "match cleanup_failure.retry(gpu)",
        "dsa_cache: &'owner mut Glm53DsaCache",
        "PhantomData<&'owner DevicePtr>",
        "usize::try_from(plan.known_bytes)",
    ];
    needles.iter().all(|needle| source.contains(needle))
        && source.matches(".checked_add(allocation_bytes)").count() == 2
        && source.matches("pub(super) struct Glm53ArenaOwner").count() == 1
        && !source.contains("impl Drop for")
        && !source.contains("impl Clone for Glm53ArenaOwner")
        && !source.contains("impl Copy for Glm53ArenaOwner")
        && !source.contains("#[derive(Clone")
        && !source.contains("#[derive(Copy")
        && !source.contains("impl std::error::Error for Glm53Arena")
        && !source.contains("gpu.alloc_managed(")
        && !source.contains("gpu.memset_async(")
        && !source.contains("DevicePtr::offset")
        && !source.contains("pub fn device_ptr")
        && !source.contains(".field(\"allocation\"")
        && !source.contains(".field(\"pointer\"")
}

fn source_contract(source: &str) -> bool {
    owner_source_is_exact(source) && structural_source_contract(source)
}

#[test]
fn source_sha256_vectors_cover_both_padding_sides_and_exact_owner() {
    for (input, expected) in [
        (
            &b""[..],
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            &b"abc"[..],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            &[b'a'; 55][..],
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
        ),
        (
            &[b'a'; 56][..],
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
        ),
    ] {
        assert_eq!(source_sha256::hex(source_sha256::digest(input)), expected);
    }
    assert!(source_sha256::matches(include_str!(
        "t1_state_transaction.rs"
    )));
    assert!(owner_source_is_exact(include_str!("arena_owner.rs")));
}

#[test]
fn exact_source_authority_rejects_identity_zero_bypass_and_decimal_leak_hostiles() {
    let source = include_str!("arena_owner.rs");
    assert!(source_contract(source));
    for (from, to) in [
        (
            "self.dsa_cache.device_identity()",
            "Glm53DsaCache::new(1, RuntimeDsaStorage::Bf16)\n            .unwrap()\n            .device_identity()",
        ),
        (
            "        if let Err(primary) = gpu\n            .memset(owner.allocation, 0, owner.allocation_bytes)",
            "        #[cfg(not(test))]\n        if owner.plan.context.positions == 1_048_576 {\n            return Ok(owner);\n        }\n        if let Err(primary) = gpu\n            .memset(owner.allocation, 0, owner.allocation_bytes)",
        ),
        (
            ".field(\"allocation_bytes\", &self.allocation_bytes)\n            .finish()",
            ".field(\"allocation_bytes\", &self.allocation_bytes)\n            .field(\"base\", &self.pointer.0)\n            .finish()",
        ),
    ] {
        let mutant = source.replacen(from, to, 1);
        assert_ne!(mutant, source, "missing mutation needle: {from}");
        assert!(
            structural_source_contract(&mutant),
            "hostile must exercise exact-source authority: {from}"
        );
        assert!(!source_contract(&mutant), "mutation survived: {from}");
    }
}

/// Registration landed on 2026-09-01, deliberately replacing the former
/// `standalone_registration_remains_an_explicit_deny_warnings_boundary`.
///
/// That test asserted `arena_owner` stayed *unregistered*, and closed with the
/// instruction that registration "must deliberately replace this test and
/// either land with all production consumers or add a narrowly documented
/// dormant allowance." It landed with a real consumer:
/// `kda_recurrent_commit` takes `Glm53ArenaView` to move staged recurrent state
/// into persistent state, which is the whole point of the module.
///
/// The safety property the old test actually protected was never "stay
/// unregistered" — it was "registration must not be smuggled in behind a
/// dead-code allowance that hides an unused, unreviewed effect surface under
/// `#![deny(warnings)]`". That property is what this test now enforces.
#[test]
fn registration_landed_with_real_consumers_and_no_dead_code_escape() {
    let crate_root = include_str!("../../lib.rs");
    let glm_module = include_str!("mod.rs");
    let owner_source = include_str!("arena_owner.rs");
    let production = glm_module.split("#[cfg(test)]").next().unwrap();

    // The crate-wide warning denial is what makes an unused module a build
    // failure rather than a silent passenger. Without it, everything below is
    // decoration.
    assert!(crate_root.contains("#![deny(warnings)]"));

    // Registered in production, not behind a `cfg(test)` alias.
    assert!(
        production.contains("mod arena_owner;"),
        "arena_owner must be production-registered now that it has consumers"
    );

    // No dead-code escape hatch, in either the module tree or the owner itself.
    assert!(
        !production.contains("allow(dead_code)"),
        "registration must not be smuggled in behind a dead-code allowance"
    );
    assert!(!owner_source.contains("#![allow(dead_code)]"));
    assert!(!owner_source.contains("#[allow(dead_code)]"));

    // A real production consumer must exist. `kda_recurrent_commit` is the one
    // that justified registration; if it stops using the owner, the module is
    // dormant again and this test should fail rather than let it drift.
    let consumer = include_str!("kda_recurrent_commit.rs");
    assert!(
        consumer.contains("use super::arena_owner::"),
        "arena_owner has no production consumer; it is dormant again"
    );
    assert!(
        production.contains("mod kda_recurrent_commit;"),
        "the consumer itself must be production-registered, or the dependency \
         is only reachable from tests"
    );
}

#[test]
fn source_contract_rejects_owner_erasure_partial_zero_and_unchecked_ranges() {
    let source = include_str!("arena_owner.rs");
    assert!(source_contract(source));
    for (from, to) in [
        (
            ".alloc(allocation_bytes)",
            ".alloc_managed(allocation_bytes)",
        ),
        ("owner.allocation_bytes)", "owner.allocation_bytes / 2)"),
        ("owner.allocation, 0", "owner.allocation, 1"),
        (".checked_add(offset_bytes)", ".wrapping_add(offset_bytes)"),
        (
            "let cleanup_failure = owner.free(gpu).err();",
            "let cleanup_failure = None;",
        ),
        (
            "match cleanup_failure.retry(gpu)",
            "match { drop(cleanup_failure); Ok(()) }",
        ),
        (
            "pub(super) struct Glm53ArenaOwner",
            "#[derive(Clone, Copy)]\npub(super) struct Glm53ArenaOwner",
        ),
        (
            ".finish()\n    }\n}\n\nimpl fmt::Debug for Glm53ArenaView",
            ".field(\"allocation\", &self.allocation)\n            .finish()\n    }\n}\n\nimpl fmt::Debug for Glm53ArenaView",
        ),
    ] {
        let mutant = source.replacen(from, to, 1);
        assert_ne!(mutant, source, "missing mutation needle: {from}");
        assert!(!source_contract(&mutant), "mutation survived: {from}");
    }
}
