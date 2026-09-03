// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[path = "../weights/gguf_device_test_sha256.rs"]
mod source_sha256;

#[test]
fn exact_layout_is_single_latent_plus_pooled_index_and_validity() {
    let layout = Glm53DsaLayout::new(1, Glm53DsaStorage::Bf16).unwrap();
    assert_eq!(layout.latent.bytes, 11 * 1_048_576 * 512 * 2);
    assert_eq!(layout.pool_keys.bytes, 11 * 262_144 * 128 * 2);
    assert_eq!(layout.pool_validity.bytes, 11 * 262_144);
    assert_eq!(layout.total_bytes, 12_552_258_304);
    assert!(!GLM53_DSA_RUNTIME_KERNEL_IMPLEMENTED);
    for (left, right) in [
        (layout.latent, layout.pool_keys),
        (layout.pool_keys, layout.pool_validity),
        (layout.pool_validity, layout.tail_keys),
        (layout.tail_keys, layout.tail_gates),
        (layout.tail_gates, layout.tail_validity),
    ] {
        assert!(left.offset.checked_add(left.bytes).unwrap() <= right.offset);
    }
    let fp8 = Glm53DsaLayout::new(1, Glm53DsaStorage::Fp8).unwrap();
    assert_eq!(fp8.total_bytes, 6_277_571_328);
    assert!(fp8.total_bytes < layout.total_bytes);
}

#[test]
fn complete_pool_and_tail_validity_cross_exact_boundaries() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let handle = cache.claim_sequence().unwrap();
    let three = cache.begin_append(handle, 3).unwrap();
    assert_eq!(
        cache.commit_append(three, 3).unwrap().tail_valid_mask,
        0b111
    );
    let one = cache.begin_append(handle, 1).unwrap();
    assert_eq!(one.complete_pools_to_write, 1);
    let view = cache.commit_append(one, 1).unwrap();
    assert_eq!((view.complete_pools, view.tail_len), (1, 0));
    assert_eq!(view.valid_complete_pools, 1);
}

#[test]
fn partial_commit_and_rollback_hide_unaccepted_writes() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let handle = cache.claim_sequence().unwrap();
    let plan = cache.begin_append(handle, 8).unwrap();
    assert_eq!(cache.commit_append(plan, 3).unwrap().logical_len, 3);
    assert!(cache.commit_append(plan, 3).is_err());
    let rolled_back = cache.begin_append(handle, 4).unwrap();
    cache.rollback_append(rolled_back).unwrap();
    assert!(cache.commit_append(rolled_back, 4).is_err());
    assert_eq!(cache.sequence_view(handle).unwrap().logical_len, 3);
}

#[test]
fn final_legal_position_is_admitted_and_next_is_rejected() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let handle = cache.claim_sequence().unwrap();
    let full = cache.begin_append(handle, GLM53_DSA_MAX_POSITIONS).unwrap();
    cache.commit_append(full, GLM53_DSA_MAX_POSITIONS).unwrap();
    assert!(cache.begin_append(handle, 1).is_err());
}

#[test]
fn prefix_and_free_are_generation_bound_and_never_alias() {
    let mut cache = Glm53DsaCache::new(2, Glm53DsaStorage::Bf16).unwrap();
    let first = cache.claim_sequence().unwrap();
    let second = cache.claim_sequence().unwrap();
    let append = cache.begin_append(first, 9).unwrap();
    cache.commit_append(append, 9).unwrap();
    assert!(cache.snapshot_prefix(first, 5).is_err());
    let prefix = cache.snapshot_prefix(first, 4).unwrap();
    assert!(cache.restore_prefix(second, prefix).is_err());
    assert_eq!(cache.restore_prefix(first, prefix).unwrap().logical_len, 4);
    cache.free_sequence(first).unwrap();
    assert!(cache.sequence_view(first).is_err());
    let replacement = cache.claim_sequence().unwrap();
    assert_eq!(replacement.slot(), first.slot());
    assert_ne!(replacement, first);
}

#[test]
fn capacity_overflow_zero_append_and_parallel_transactions_fail_closed() {
    assert!(Glm53DsaCache::new(0, Glm53DsaStorage::Bf16).is_err());
    assert!(Glm53DsaCache::new(usize::MAX, Glm53DsaStorage::Fp8).is_err());
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let handle = cache.claim_sequence().unwrap();
    assert!(cache.begin_append(handle, 0).is_err());
    let active = cache.begin_append(handle, 1).unwrap();
    assert!(cache.begin_append(handle, 1).is_err());
    assert!(cache.free_sequence(handle).is_err());
    cache.rollback_append(active).unwrap();
}

#[test]
fn handles_bind_device_identity_before_slot_and_generation() {
    let mut left = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let mut right = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let left_handle = left.claim_sequence().unwrap();
    let right_handle = right.claim_sequence().unwrap();
    assert_eq!(left_handle.slot(), right_handle.slot());
    assert_eq!(left_handle.generation(), right_handle.generation());
    assert_eq!(left_handle.device_identity(), left.device_identity());
    assert_eq!(right_handle.device_identity(), right.device_identity());
    assert_ne!(left.device_identity(), right.device_identity());

    let prefix = left.snapshot_prefix(left_handle, 0).unwrap();
    let plan = left.begin_append(left_handle, 1).unwrap();
    assert!(right.sequence_view(left_handle).is_err());
    assert!(right.begin_append(left_handle, 1).is_err());
    assert!(right.commit_append(plan, 1).is_err());
    assert!(right.rollback_append(plan).is_err());
    assert!(right.snapshot_prefix(left_handle, 0).is_err());
    assert!(right.restore_prefix(left_handle, prefix).is_err());
    assert!(right.poison_sequence(left_handle).is_err());
    assert!(right.sequence_is_poisoned(left_handle).is_err());
    assert!(right.free_sequence(left_handle).is_err());
    assert_eq!(right.sequence_view(right_handle).unwrap().logical_len, 0);
    assert!(!right.sequence_is_poisoned(right_handle).unwrap());
    left.rollback_append(plan).unwrap();
}

#[test]
fn poison_is_sticky_idempotent_and_preserves_active_diagnostics() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let handle = cache.claim_sequence().unwrap();
    let seed = cache.begin_append(handle, 3).unwrap();
    cache.commit_append(seed, 3).unwrap();
    let prefix = cache.snapshot_prefix(handle, 3).unwrap();
    let active = cache.begin_append(handle, 1).unwrap();
    let before = *cache.owned_state(handle).unwrap();

    cache.poison_sequence(handle).unwrap();
    cache.poison_sequence(handle).unwrap();
    let after = *cache.owned_state(handle).unwrap();
    assert!(after.poisoned);
    assert_eq!(after.logical_len, before.logical_len);
    assert_eq!(after.epoch, before.epoch);
    assert_eq!(after.active, before.active);
    assert!(cache.sequence_is_poisoned(handle).unwrap());

    assert!(cache.sequence_view(handle).is_err());
    assert!(cache.begin_append(handle, 1).is_err());
    assert!(cache.commit_append(active, 1).is_err());
    assert!(cache.rollback_append(active).is_err());
    assert!(cache.snapshot_prefix(handle, 3).is_err());
    assert!(cache.restore_prefix(handle, prefix).is_err());
    assert!(cache.free_sequence(handle).is_err());
    assert!(cache.claim_sequence().is_err());
}

#[test]
fn stale_and_foreign_poison_cannot_quarantine_a_live_sequence() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let old = cache.claim_sequence().unwrap();
    cache.free_sequence(old).unwrap();
    let current = cache.claim_sequence().unwrap();
    assert_eq!(old.device_identity(), current.device_identity());
    assert_ne!(old.generation(), current.generation());
    assert!(cache.poison_sequence(old).is_err());
    assert!(!cache.sequence_is_poisoned(current).unwrap());

    let mut foreign = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let foreign_handle = foreign.claim_sequence().unwrap();
    assert!(cache.poison_sequence(foreign_handle).is_err());
    assert!(!cache.sequence_is_poisoned(current).unwrap());

    cache.poison_sequence(current).unwrap();
    assert!(cache.free_sequence(current).is_err());
    assert!(cache.claim_sequence().is_err());
}

#[test]
fn device_identity_counter_rejects_zero_and_exhaustion_without_wrap() {
    let (identity, next) = identity_step(1).unwrap();
    assert_eq!(identity.0.get(), 1);
    assert_eq!(next, 2);
    assert!(identity_step(0).is_err());
    assert!(identity_step(u64::MAX).is_err());
}

#[test]
fn generation_exhaustion_retains_the_owned_live_slot() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let claimed = cache.claim_sequence().unwrap();
    cache.slots[0].generation = u64::MAX;
    let terminal = Glm53DsaSequenceHandle {
        device_identity: claimed.device_identity(),
        slot: claimed.slot(),
        generation: u64::MAX,
    };
    assert!(cache.free_sequence(terminal).is_err());
    assert_eq!(cache.sequence_view(terminal).unwrap().logical_len, 0);
    assert!(!cache.sequence_is_poisoned(terminal).unwrap());
}

const SOURCE: &str = include_str!("glm53_dsa.rs");
const SOURCE_SHA256: &str = "23c65d811f2158041f87d59a9e7090af14d9fc406b244294c5465f7e5f68695a";

const ENTRY: &str = r#"    fn entry(&self, handle: Glm53DsaSequenceHandle) -> Result<&Slot> {
        if handle.device_identity != self.device_identity {
            bail!("foreign GLM DSA device identity");
        }
        let entry = self
            .slots
            .get(handle.slot)
            .context("GLM DSA sequence slot is out of range")?;
        if entry.generation != handle.generation {
            bail!("stale GLM DSA sequence handle");
        }
        Ok(entry)
    }"#;

const CONDITIONAL_ENTRY: &str = r#"    #[cfg(debug_assertions)]
    fn entry(&self, handle: Glm53DsaSequenceHandle) -> Result<&Slot> {
        if handle.device_identity != self.device_identity {
            bail!("foreign GLM DSA device identity");
        }
        let entry = self
            .slots
            .get(handle.slot)
            .context("GLM DSA sequence slot is out of range")?;
        if entry.generation != handle.generation {
            bail!("stale GLM DSA sequence handle");
        }
        Ok(entry)
    }

    #[cfg(not(debug_assertions))]
    fn entry(&self, handle: Glm53DsaSequenceHandle) -> Result<&Slot> {
        let entry = self
            .slots
            .get(handle.slot)
            .context("GLM DSA sequence slot is out of range")?;
        if entry.generation != handle.generation {
            bail!("stale GLM DSA sequence handle");
        }
        Ok(entry)
    }"#;

const ENTRY_MUT: &str = r#"    fn entry_mut(&mut self, handle: Glm53DsaSequenceHandle) -> Result<&mut Slot> {
        if handle.device_identity != self.device_identity {
            bail!("foreign GLM DSA device identity");
        }
        let entry = self
            .slots
            .get_mut(handle.slot)
            .context("GLM DSA sequence slot is out of range")?;
        if entry.generation != handle.generation {
            bail!("stale GLM DSA sequence handle");
        }
        Ok(entry)
    }"#;

const CONDITIONAL_ENTRY_MUT: &str = r#"    #[cfg(debug_assertions)]
    fn entry_mut(&mut self, handle: Glm53DsaSequenceHandle) -> Result<&mut Slot> {
        if handle.device_identity != self.device_identity {
            bail!("foreign GLM DSA device identity");
        }
        let entry = self
            .slots
            .get_mut(handle.slot)
            .context("GLM DSA sequence slot is out of range")?;
        if entry.generation != handle.generation {
            bail!("stale GLM DSA sequence handle");
        }
        Ok(entry)
    }

    #[cfg(not(debug_assertions))]
    fn entry_mut(&mut self, handle: Glm53DsaSequenceHandle) -> Result<&mut Slot> {
        let entry = self
            .slots
            .get_mut(handle.slot)
            .context("GLM DSA sequence slot is out of range")?;
        if entry.generation != handle.generation {
            bail!("stale GLM DSA sequence handle");
        }
        Ok(entry)
    }"#;

fn identity_precedes_lookup(source: &str, signature: &str, lookup: &str) -> bool {
    let Some(body) = source.split_once(signature).map(|(_, body)| body) else {
        return false;
    };
    let Some(body) = body.split_once("Ok(entry)").map(|(body, _)| body) else {
        return false;
    };
    let Some(identity) = body.find("if handle.device_identity != self.device_identity {") else {
        return false;
    };
    body.find(lookup).is_some_and(|slot| identity < slot)
}

fn ownership_contract(source: &str) -> bool {
    let identity_gate = "if handle.device_identity != self.device_identity {";
    source.matches(identity_gate).count() == 2
        && source.matches("if state.poisoned {").count() == 3
        && source
            .matches("self.owned_state_mut(handle)?.poisoned = true;")
            .count()
            == 1
        && source.contains(".fetch_update(Ordering::Relaxed, Ordering::Relaxed")
        && source.contains("pub struct Glm53DsaSequenceHandle {\n    device_identity:")
        && identity_precedes_lookup(source, "fn entry(&self", ".slots\n            .get(")
        && identity_precedes_lookup(
            source,
            "fn entry_mut(&mut self",
            ".slots\n            .get_mut(",
        )
        && !source.contains("wrapping_add")
        && !source.contains("poisoned = false")
        && !source.contains("pub fn unpoison")
        && !source.contains("pub struct Glm53DsaDeviceIdentity(pub")
        && !source.contains("pub device_identity:")
        && !source.contains("pub generation:")
}

#[test]
fn full_source_sha_rejects_debug_safe_release_gate_free_alternatives() {
    assert_eq!(
        source_sha256::hex(source_sha256::digest(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        source_sha256::hex(source_sha256::digest(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert!(source_sha256::matches(SOURCE, SOURCE_SHA256));

    let mutant =
        SOURCE
            .replacen(ENTRY, CONDITIONAL_ENTRY, 1)
            .replacen(ENTRY_MUT, CONDITIONAL_ENTRY_MUT, 1);
    assert_ne!(mutant, SOURCE);
    assert!(mutant.contains("#[cfg(debug_assertions)]"));
    assert!(mutant.contains("#[cfg(not(debug_assertions))]"));
    let mutant_production = mutant.split("#[cfg(test)]").next().unwrap();
    assert!(ownership_contract(mutant_production));
    assert!(!source_sha256::matches(&mutant, SOURCE_SHA256));
}

#[test]
fn source_contract_rejects_identity_poison_wrap_and_recovery_mutants() {
    let source = SOURCE.split("#[cfg(test)]").next().unwrap();
    assert!(ownership_contract(source));
    let identity_gate = "if handle.device_identity != self.device_identity {";
    assert!(!ownership_contract(&source.replacen(
        identity_gate,
        "if false {",
        1
    )));
    let reordered = source.replacen(
        "        if handle.device_identity != self.device_identity {\n            bail!(\"foreign GLM DSA device identity\");\n        }\n        let entry = self\n            .slots\n            .get(handle.slot)\n            .context(\"GLM DSA sequence slot is out of range\")?;",
        "        let entry = self\n            .slots\n            .get(handle.slot)\n            .context(\"GLM DSA sequence slot is out of range\")?;\n        if handle.device_identity != self.device_identity {\n            bail!(\"foreign GLM DSA device identity\");\n        }",
        1,
    );
    assert!(!ownership_contract(&reordered));
    assert!(!ownership_contract(&source.replacen(
        "if state.poisoned {",
        "if false {",
        1
    )));
    assert!(!ownership_contract(&source.replacen(
        "checked_add(1)",
        "wrapping_add(1)",
        1
    )));
    assert!(!ownership_contract(&source.replacen(
        "poisoned = true",
        "poisoned = false",
        1
    )));
    assert!(!ownership_contract(&format!(
        "{source}\npub fn unpoison() {{}}"
    )));
    assert!(!ownership_contract(&source.replacen(
        "pub struct Glm53DsaDeviceIdentity(NonZeroU64)",
        "pub struct Glm53DsaDeviceIdentity(pub NonZeroU64)",
        1,
    )));
    assert!(!ownership_contract(&source.replacen(
        "state.active = None;",
        "state.active = None; state.poisoned = false;",
        1,
    )));
}
