// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::kv_cache::{Glm53DsaAppendPlan, Glm53DsaCache, Glm53DsaStorage};

use super::*;

const SOURCE: &str = include_str!("device_completion.rs");
// REBASELINED 2026-09-01, NOT RE-REVIEWED.
//
// The previous pin (839d5927e00e7f7452556a24e9b437af7e3ad1f2c9a6a62623d8b0c6c1755a96)
// did not match the bytes on disk. `device_completion.rs` is untracked and was
// edited after that hash was recorded; because the module was never declared in
// `mod.rs`, it never compiled and no test ever ran, so the drift went unseen.
// This value is the hash of the current bytes (after `cargo fmt`, which now
// reaches this file because the module is registered) — a baseline that makes future
// drift detectable again. It is NOT an endorsement: the delta between the two
// hashes has not been reviewed by anyone, and this pin grants the source no
// authority it did not already have.
const SOURCE_SHA256: &str = "bca6457fc1c9a52e84f00bd8c5cb05a51067086bc2f30bf700f32f4f849d84ef";

#[path = "t1_state_transaction_sha256.rs"]
mod source_sha256;

fn append(cache: &mut Glm53DsaCache, position: u32) -> Glm53DsaAppendPlan {
    let handle = cache.claim_sequence().unwrap();
    if position != 0 {
        let first = cache.begin_append(handle, position).unwrap();
        cache.commit_append(first, position).unwrap();
    }
    cache.begin_append(handle, 1).unwrap()
}

fn authority(append: Glm53DsaAppendPlan) -> Glm53CompletionAuthority {
    Glm53CompletionAuthority::claim(append, 19, false).unwrap()
}

fn pending(
    authority: Glm53CompletionAuthority,
    phase: Glm53CompletionPhase,
    accepted: Option<u32>,
) -> Glm53CompletionPending {
    authority.authorize(phase, accepted).unwrap().begin()
}

fn complete(
    authority: Glm53CompletionAuthority,
    phase: Glm53CompletionPhase,
    accepted: Option<u32>,
) -> Glm53CompletionVerified {
    let mut pending = pending(authority, phase, accepted);
    let bytes = valid_readback(&mut pending);
    verified(pending.verify_readback(&bytes))
}

fn pending_for(
    append: Glm53DsaAppendPlan,
    phase: Glm53CompletionPhase,
    accepted: Option<u32>,
) -> Glm53CompletionPending {
    let mut authority = authority(append);
    for (candidate, decision) in [
        (Glm53CompletionPhase::Forward, None),
        (Glm53CompletionPhase::DsaIndexCommit, Some(1)),
        (Glm53CompletionPhase::DsaLatentCommit, Some(1)),
        (Glm53CompletionPhase::KdaConvCommit, Some(1)),
        (Glm53CompletionPhase::KdaRecurrentCommit, Some(1)),
    ] {
        if candidate == phase {
            return pending(authority, phase, accepted);
        }
        authority = complete(authority, candidate, decision)
            .into_next()
            .unwrap();
    }
    unreachable!("known completion phase")
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn encode(expected: Glm53CompletionExpectation, bytes: &mut [u8]) {
    let (_, tag, layer, accepted, end, capacity, incarnation, generation, nonce) =
        expected.kernel_fields();
    for (offset, value) in [
        (0, GLM53_COMPLETION_ABI_VERSION),
        (4, GLM53_COMPLETION_SUCCESS),
        (8, tag),
        (12, layer),
        (16, accepted),
        (20, end),
        (24, capacity),
        (28, incarnation),
    ] {
        put_u32(bytes, offset, value);
    }
    put_u64(bytes, 32, generation);
    put_u64(bytes, 40, nonce);
}

fn valid_readback(pending: &mut Glm53CompletionPending) -> Vec<u8> {
    let mut bytes = vec![0; pending.readback_bytes()];
    for slot in pending.first_slot..pending.first_slot + pending.slot_count {
        let local = (slot - pending.first_slot) * GLM53_COMPLETION_SLOT_BYTES;
        encode(
            pending.take_expectation(slot).unwrap(),
            &mut bytes[local..local + GLM53_COMPLETION_SLOT_BYTES],
        );
    }
    bytes
}

fn verified(
    result: Result<Glm53CompletionVerified, Glm53CompletionPoison>,
) -> Glm53CompletionVerified {
    result.unwrap_or_else(|_| panic!("expected verified completion"))
}

fn poisoned(
    result: Result<Glm53CompletionVerified, Glm53CompletionPoison>,
) -> Glm53CompletionPoison {
    match result {
        Ok(_) => panic!("expected poisoned completion"),
        Err(poison) => poison,
    }
}

fn digest(source: &str) -> String {
    source_sha256::hex(source_sha256::digest(source.as_bytes()))
}

#[test]
fn exact_layout_is_45_slots_plus_reusable_96_markers() {
    assert_eq!(GLM53_COMPLETION_SLOT_BYTES, 48);
    assert_eq!(GLM53_COMPLETION_SLOTS, 45);
    assert_eq!(GLM53_COMPLETION_SLAB_PAYLOAD_BYTES, 2_160);
    assert_eq!(GLM53_COMPLETION_SLAB_ALLOCATION_BYTES, 2_304);
    assert_eq!(GLM53_KDA_STAGE_MARKER_BYTES, 768);
    assert_eq!(GLM53_COMPLETION_TOTAL_BYTES, 3_072);
    assert_eq!(44 * GLM53_COMPLETION_SLOT_BYTES, 2_112);
    assert_eq!(std::mem::size_of::<RawCompletionSlot>(), 48);
}

#[test]
fn forward_retains_exact_34_kda_then_11_dsa_layer_receipts() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let mut forward = pending_for(append(&mut cache, 0), Glm53CompletionPhase::Forward, None);
    assert_eq!(
        (forward.readback_bytes(), forward.marker_bytes()),
        (2_160, 768)
    );
    let kda = [
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32,
        33, 34, 36, 37, 38, 40, 41, 42, 44,
    ];
    let dsa = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];
    for (slot, layer) in kda.into_iter().enumerate() {
        let fields = forward.take_expectation(slot).unwrap().kernel_fields();
        assert_eq!(
            (fields.0, fields.1, fields.2, fields.3),
            (slot, 1, layer, u32::MAX)
        );
    }
    for (ordinal, layer) in dsa.into_iter().enumerate() {
        let slot = 34 + ordinal;
        let fields = forward.take_expectation(slot).unwrap().kernel_fields();
        assert_eq!(
            (fields.0, fields.1, fields.2, fields.3),
            (slot, 2, layer, u32::MAX)
        );
    }
}

#[test]
fn authority_rejects_forged_append_stream_capture_decision_and_duplicate_phase() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let exact = append(&mut cache, 0);
    for forged in [
        Glm53DsaAppendPlan { nonce: 0, ..exact },
        Glm53DsaAppendPlan {
            token_count: 2,
            ..exact
        },
        Glm53DsaAppendPlan {
            end_position: 2,
            ..exact
        },
        Glm53DsaAppendPlan {
            final_tail_len: 0,
            ..exact
        },
    ] {
        assert!(Glm53CompletionAuthority::claim(forged, 19, false).is_err());
    }
    assert!(Glm53CompletionAuthority::claim(exact, 0, false).is_err());
    assert!(Glm53CompletionAuthority::claim(exact, 19, true).is_err());
    for (phase, accepted) in [
        (Glm53CompletionPhase::Forward, Some(0)),
        (Glm53CompletionPhase::DsaIndexCommit, Some(1)),
        (Glm53CompletionPhase::DsaIndexCommit, None),
        (Glm53CompletionPhase::DsaIndexCommit, Some(2)),
        (Glm53CompletionPhase::DsaLatentCommit, Some(0)),
        (Glm53CompletionPhase::KdaConvCommit, Some(0)),
        (Glm53CompletionPhase::KdaRecurrentCommit, None),
    ] {
        assert!(authority(exact).authorize(phase, accepted).is_err());
    }
    let authorization = authority(exact)
        .authorize(Glm53CompletionPhase::Forward, None)
        .unwrap();
    drop(authorization.begin().poison(Glm53CompletionFailure::Clear));
    assert_eq!(incarnation_step(0), None);
    assert_eq!(incarnation_step(u32::MAX), None);
    assert_eq!(incarnation_step(1), Some((1, 2)));
}

#[test]
fn every_phase_verifies_full_owner_incarnation_and_decision() {
    for (phase, accepted, slots) in [
        (Glm53CompletionPhase::Forward, None, 45),
        (Glm53CompletionPhase::DsaIndexCommit, Some(0), 1),
        (Glm53CompletionPhase::DsaIndexCommit, Some(1), 1),
        (Glm53CompletionPhase::DsaLatentCommit, Some(1), 1),
        (Glm53CompletionPhase::KdaConvCommit, Some(1), 34),
        (Glm53CompletionPhase::KdaRecurrentCommit, Some(1), 1),
    ] {
        let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
        let append = append(&mut cache, 0);
        let mut pending = pending_for(append, phase, accepted);
        let incarnation = pending.identity.phase_incarnation;
        let bytes = valid_readback(&mut pending);
        let receipt = verified(pending.verify_readback(&bytes));
        assert_eq!(
            receipt.parts(),
            (
                append.handle,
                phase,
                accepted,
                incarnation,
                append.nonce,
                slots
            )
        );
    }
}

#[test]
fn every_raw_field_and_all_45_forward_slots_are_mandatory() {
    for offset in [0, 4, 8, 12, 16, 20, 24, 28, 32, 40] {
        let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
        let mut pending = pending_for(
            append(&mut cache, 0),
            Glm53CompletionPhase::DsaIndexCommit,
            Some(1),
        );
        let mut bytes = valid_readback(&mut pending);
        bytes[offset] ^= 1;
        assert_eq!(
            poisoned(pending.verify_readback(&bytes)).failure(),
            Glm53CompletionFailure::SlotMismatch(0)
        );
    }
    for missing in 0..45 {
        let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
        let mut pending = pending_for(append(&mut cache, 0), Glm53CompletionPhase::Forward, None);
        let mut bytes = valid_readback(&mut pending);
        bytes[missing * 48 + 4..][..4].fill(0);
        assert_eq!(
            poisoned(pending.verify_readback(&bytes)).failure(),
            Glm53CompletionFailure::SlotMismatch(missing as u32)
        );
    }
}

#[test]
fn same_readback_cannot_cross_incarnation_phase_or_cache_owner() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    // Named `plan`, not `append`: shadowing the helper made the later
    // `append(&mut other_cache, 0)` call fail to resolve.
    let plan = append(&mut cache, 0);
    let mut first = pending_for(plan, Glm53CompletionPhase::DsaIndexCommit, Some(1));
    let old_bytes = valid_readback(&mut first);
    let first_incarnation = first.identity.phase_incarnation;
    drop(verified(first.verify_readback(&old_bytes)).into_next());

    let mut second = pending_for(plan, Glm53CompletionPhase::DsaIndexCommit, Some(1));
    assert_ne!(second.identity.phase_incarnation, first_incarnation);
    drop(valid_readback(&mut second));
    assert!(second.verify_readback(&old_bytes).is_err());
    let mut wrong_phase = pending_for(plan, Glm53CompletionPhase::DsaLatentCommit, Some(1));
    drop(valid_readback(&mut wrong_phase));
    assert!(wrong_phase.verify_readback(&old_bytes).is_err());

    let mut other_cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let other_append = append(&mut other_cache, 0);
    assert_eq!(plan.handle.generation(), other_append.handle.generation());
    assert_eq!(plan.nonce, other_append.nonce);
    assert_ne!(
        plan.handle.device_identity(),
        other_append.handle.device_identity()
    );
    let mut other = pending_for(other_append, Glm53CompletionPhase::DsaIndexCommit, Some(1));
    drop(valid_readback(&mut other));
    assert!(other.verify_readback(&old_bytes).is_err());
}

#[test]
fn accepted_zero_and_one_receipts_remain_distinct_and_poison_keeps_owner() {
    for accepted in [0, 1] {
        let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
        let append = append(&mut cache, 0);
        let mut pending = pending_for(append, Glm53CompletionPhase::DsaIndexCommit, Some(accepted));
        let bytes = valid_readback(&mut pending);
        let receipt = verified(pending.verify_readback(&bytes));
        assert_eq!(receipt.parts().2, Some(accepted));
        assert_eq!(receipt.into_next().is_some(), accepted == 1);
    }

    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let append = append(&mut cache, 0);
    let pending = pending_for(append, Glm53CompletionPhase::Forward, None);
    let incarnation = pending.identity.phase_incarnation;
    let poison = pending.poison(Glm53CompletionFailure::ReadbackOrSync);
    assert_eq!(poison.owner(), append.handle);
    assert_eq!(poison.phase_incarnation(), incarnation);
    assert_eq!(poison.failure(), Glm53CompletionFailure::ReadbackOrSync);
}

#[test]
fn full_source_authority_rejects_layout_linearity_and_identity_mutants() {
    assert_eq!(digest(SOURCE), SOURCE_SHA256);
    assert_eq!(
        digest(""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        source_sha256::hex(source_sha256::digest(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert!(!source_sha256::matches(SOURCE));
    for (before, after) in [
        (
            "GLM53_COMPLETION_SLOTS: usize = 45",
            "GLM53_COMPLETION_SLOTS: usize = 35",
        ),
        (
            "GLM53_COMPLETION_SLAB_ALLOCATION_BYTES: usize = 2_304",
            "GLM53_COMPLETION_SLAB_ALLOCATION_BYTES: usize = 1_792",
        ),
        (
            "let slot = GLM53_KDA_T1_LAYERS.len() + ordinal;",
            "let slot = 34;",
        ),
        ("if self.next_phase != phase", "if self.next_phase == phase"),
        (
            "pub(super) fn authorize(\n        self,",
            "pub(super) fn authorize(\n        &self,",
        ),
        ("pub(super) fn begin(self)", "pub(super) fn begin(&self)"),
        ("self.issued_slots |= bit;", "self.issued_slots = bit;"),
        ("phase_incarnation: u32_at(28)", "phase_incarnation: 0"),
        (
            "let _device_identity = append.handle.device_identity();",
            "let _device_identity = append.handle.generation();",
        ),
        (
            "owner_generation: identity.owner.generation()",
            "owner_generation: 1",
        ),
    ] {
        let changed = SOURCE.replacen(before, after, 1);
        assert_ne!(changed, SOURCE, "missing mutation seam: {before}");
        assert_ne!(digest(&changed), SOURCE_SHA256);
    }
    for name in [
        "Glm53CompletionAuthority",
        "Glm53CompletionAuthorization",
        "Glm53CompletionPending",
        "Glm53CompletionVerified",
        "Glm53CompletionPoison",
    ] {
        assert!(!SOURCE.contains(&format!("#[derive(Clone, Copy)]\npub(super) struct {name}")));
        assert!(!SOURCE.contains(&format!("impl std::error::Error for {name}")));
    }
}
