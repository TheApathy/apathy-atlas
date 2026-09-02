// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::kv_cache::{
    Glm53DsaAppendPlan, Glm53DsaCache, Glm53DsaSequenceHandle, Glm53DsaStorage,
};

use super::*;

const STREAM: u64 = 0x53;

fn append_at(start: u32) -> Glm53DsaAppendPlan {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let owner = cache.claim_sequence().unwrap();
    if start != 0 {
        let seed = cache.begin_append(owner, start).unwrap();
        cache.commit_append(seed, start).unwrap();
    }
    cache.begin_append(owner, 1).unwrap()
}

fn lease(append: Glm53DsaAppendPlan) -> Glm53T1ExclusiveStreamLease {
    Glm53T1ExclusiveStreamLease::new(append.handle, append.nonce, STREAM, false).unwrap()
}

fn begin() -> (Glm53DsaAppendPlan, Glm53TargetT1StateTransaction) {
    let append = append_at(0);
    let transaction = Glm53TargetT1StateTransaction::begin(
        append,
        lease(append),
        Glm53T1StateLayout::exact().unwrap(),
    )
    .unwrap();
    (append, transaction)
}

fn kda_ready(append: Glm53DsaAppendPlan, layer: u32) -> Glm53KdaT1LayerReady {
    Glm53KdaT1LayerReady {
        owner: append.handle,
        layer,
        transaction_nonce: append.nonce,
        exclusive_end: append.end_position,
        stream: STREAM,
        capture_observed: false,
        staged_h_ready: true,
        staged_conv_ready: true,
        persistent_state_untouched: true,
        device_status: 0,
    }
}

fn dsa_ready(append: Glm53DsaAppendPlan, layer: u32) -> Glm53TargetT1DsaLayerReady {
    Glm53TargetT1DsaLayerReady {
        owner: append.handle,
        layer,
        transaction_nonce: append.nonce,
        exclusive_end: append.end_position,
        stream: STREAM,
        capture_observed: false,
        latent_ready: true,
        index_ready: true,
        current_visibility_ready: true,
        device_status: 0,
    }
}

fn ready_all(transaction: &mut Glm53TargetT1StateTransaction, append: Glm53DsaAppendPlan) {
    for layer in 0..GLM53_TARGET_T1_LAYERS {
        if GLM53_KDA_T1_LAYERS.contains(&layer) {
            transaction
                .record_kda_layer(kda_ready(append, layer))
                .unwrap();
        } else {
            transaction
                .record_dsa_layer(dsa_ready(append, layer))
                .unwrap();
        }
    }
}

fn device(
    transition: Glm53TargetT1Transition,
    append: Glm53DsaAppendPlan,
    kind: Glm53TargetT1Effect,
) -> Glm53TargetT1DeviceEffect {
    let Glm53TargetT1Transition::Device(effect) = transition else {
        panic!("expected device effect")
    };
    assert_eq!(effect.kind, kind);
    assert_eq!(effect.owner, append.handle);
    assert_eq!(effect.transaction_nonce, append.nonce);
    assert_eq!(effect.exclusive_end, append.end_position);
    assert_eq!(effect.stream, STREAM);
    effect
}

#[test]
fn exact_layout_is_contiguous_aligned_and_preserves_dummy_slot() {
    let layout = Glm53T1StateLayout::exact().unwrap();
    assert_eq!(GLM53_KDA_T1_LAYERS.len(), 34);
    assert_eq!(GLM53_DSA_T1_LAYERS.len(), 11);
    assert_eq!(layout.kda_recurrent_f32, region(0, 142_606_336));
    assert_eq!(layout.kda_conv_f32, region(142_606_336, 13_369_344));
    assert_eq!(layout.dsa_latent_overlay, region(155_975_680, 11_264));
    assert_eq!(layout.dsa_pool_tail_overlay, region(155_986_944, 20_224));
    assert_eq!(layout.published_ends_u32, region(156_007_168, 180));
    assert_eq!(layout.published_nonces_u64, region(156_007_424, 360));
    assert_eq!(layout.logical_lengths_u32, region(156_007_936, 180));
    assert_eq!(layout.total_bytes, 156_008_192);
    assert_eq!(layout.kda_candidate_bytes(), 155_975_680);
    assert_eq!(GLM53_KDA_PERSISTENT_B1_WITH_DUMMY_BYTES, 311_951_360);
    layout.validate().unwrap();

    let mut forged = layout;
    forged.kda_conv_f32.offset_bytes -= 256;
    assert!(forged.validate().is_err());
    forged = layout;
    forged.total_bytes = u64::MAX;
    assert!(forged.validate().is_err());
}

#[test]
fn exact_layer_lists_and_final_legal_position_are_admitted() {
    assert_eq!(
        GLM53_KDA_T1_LAYERS,
        [
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30,
            32, 33, 34, 36, 37, 38, 40, 41, 42, 44,
        ]
    );
    assert_eq!(
        GLM53_DSA_T1_LAYERS,
        [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    );
    let append = append_at(GLM53_T1_MAX_POSITIONS - 1);
    let _transaction = Glm53TargetT1StateTransaction::begin(
        append,
        lease(append),
        Glm53T1StateLayout::exact().unwrap(),
    )
    .unwrap();
    assert_eq!(
        (append.start_position, append.end_position),
        (1_048_575, 1_048_576)
    );
}

#[test]
fn begin_rejects_forged_geometry_slot_stream_capture_and_nonce() {
    let layout = Glm53T1StateLayout::exact().unwrap();
    let append = append_at(0);
    let mut bad = append;
    bad.nonce = 0;
    assert!(Glm53T1ExclusiveStreamLease::new(bad.handle, 0, STREAM, false).is_err());
    assert!(Glm53T1ExclusiveStreamLease::new(append.handle, append.nonce, 0, false).is_err());
    assert!(Glm53T1ExclusiveStreamLease::new(append.handle, append.nonce, STREAM, true).is_err());

    bad = append;
    bad.end_position = 2;
    assert!(Glm53TargetT1StateTransaction::begin(bad, lease(append), layout).is_err());

    bad = append_at(GLM53_T1_MAX_POSITIONS - 1);
    bad.start_position = GLM53_T1_MAX_POSITIONS;
    assert!(Glm53TargetT1StateTransaction::begin(bad, lease(bad), layout).is_err());

    let mut cache = Glm53DsaCache::new(2, Glm53DsaStorage::Bf16).unwrap();
    let _slot_zero = cache.claim_sequence().unwrap();
    let slot_one = cache.claim_sequence().unwrap();
    let slot_one_append = cache.begin_append(slot_one, 1).unwrap();
    assert!(
        Glm53TargetT1StateTransaction::begin(
            slot_one_append,
            Glm53T1ExclusiveStreamLease::new(slot_one, slot_one_append.nonce, STREAM, false,)
                .unwrap(),
            layout,
        )
        .is_err()
    );
}

#[test]
fn all_45_layers_are_required_in_exact_interleaved_order() {
    let (append, mut transaction) = begin();
    assert!(transaction.record_kda_layer(kda_ready(append, 1)).is_err());
    assert!(transaction.is_poisoned());

    let (append, mut transaction) = begin();
    transaction.record_kda_layer(kda_ready(append, 0)).unwrap();
    assert!(transaction.record_kda_layer(kda_ready(append, 0)).is_err());

    let (append, mut transaction) = begin();
    transaction.record_kda_layer(kda_ready(append, 0)).unwrap();
    assert!(transaction.record_dsa_layer(dsa_ready(append, 3)).is_err());

    let (append, mut transaction) = begin();
    let mut wrong = kda_ready(append, 0);
    wrong.transaction_nonce += 1;
    assert!(transaction.record_kda_layer(wrong).is_err());

    let (append, mut transaction) = begin();
    let mut wrong = kda_ready(append, 0);
    wrong.stream += 1;
    assert!(transaction.record_kda_layer(wrong).is_err());

    let (append, mut transaction) = begin();
    let mut wrong = kda_ready(append, 0);
    wrong.capture_observed = true;
    assert!(transaction.record_kda_layer(wrong).is_err());

    let (append, mut transaction) = begin();
    ready_all(&mut transaction, append);
    assert_eq!(transaction.ready_counts(), (34, 11));
    assert_eq!(transaction.phase(), Glm53TargetT1Phase::AwaitingForwardSync);
    assert!(transaction.record_kda_layer(kda_ready(append, 44)).is_err());
}

#[test]
fn old_generation_and_cross_owner_receipts_are_rejected() {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let old_owner = cache.claim_sequence().unwrap();
    let old_append = cache.begin_append(old_owner, 1).unwrap();
    cache.rollback_append(old_append).unwrap();
    cache.free_sequence(old_owner).unwrap();
    let new_owner = cache.claim_sequence().unwrap();
    let append = cache.begin_append(new_owner, 1).unwrap();
    let mut transaction = Glm53TargetT1StateTransaction::begin(
        append,
        Glm53T1ExclusiveStreamLease::new(new_owner, append.nonce, STREAM, false).unwrap(),
        Glm53T1StateLayout::exact().unwrap(),
    )
    .unwrap();
    let mut stale = kda_ready(append, 0);
    stale.owner = old_owner;
    assert!(transaction.record_kda_layer(stale).is_err());
}

#[test]
fn accepted_zero_retires_without_a_persistent_write() {
    let (append, mut transaction) = begin();
    ready_all(&mut transaction, append);
    transaction
        .confirm_forward_sync(Glm53TargetT1DeviceOutcome::Success)
        .unwrap();
    let retire = device(
        transaction.decide(0).unwrap(),
        append,
        Glm53TargetT1Effect::RetireRejectedMarkers,
    );
    let sync = device(
        transaction
            .complete_effect(retire, Glm53TargetT1DeviceOutcome::Success)
            .unwrap(),
        append,
        Glm53TargetT1Effect::FinalStreamSync,
    );
    let publication = match transaction
        .complete_effect(sync, Glm53TargetT1DeviceOutcome::Success)
        .unwrap()
    {
        Glm53TargetT1Transition::PublishCpu(publication) => publication,
        other => panic!("unexpected transition: {other:?}"),
    };
    assert_eq!(publication.into_parts(), (append, 0));
    transaction
        .confirm_cpu_publication(Glm53TargetT1CpuOutcome::Success)
        .unwrap();
    assert_eq!(transaction.phase(), Glm53TargetT1Phase::Complete);
}

#[test]
fn accepted_one_has_one_exact_cross_cache_commit_order() {
    let (append, mut transaction) = begin();
    ready_all(&mut transaction, append);
    transaction
        .confirm_forward_sync(Glm53TargetT1DeviceOutcome::Success)
        .unwrap();
    let expected = [
        Glm53TargetT1Effect::CommitDsaIndexAllLayers,
        Glm53TargetT1Effect::CommitDsaLatentAllLayers,
        Glm53TargetT1Effect::CommitKdaAllLayers,
        Glm53TargetT1Effect::FinalStreamSync,
    ];
    let mut transition = transaction.decide(1).unwrap();
    for effect in expected {
        let authorization = device(transition, append, effect);
        transition = transaction
            .complete_effect(authorization, Glm53TargetT1DeviceOutcome::Success)
            .unwrap();
    }
    let publication = match transition {
        Glm53TargetT1Transition::PublishCpu(publication) => publication,
        other => panic!("unexpected transition: {other:?}"),
    };
    assert_eq!(publication.into_parts(), (append, 1));
    transaction
        .confirm_cpu_publication(Glm53TargetT1CpuOutcome::Success)
        .unwrap();
}

#[test]
fn premature_decisions_reordered_effects_and_failures_poison_owner() {
    let (append, mut transaction) = begin();
    assert!(transaction.decide(1).is_err());
    assert_eq!(transaction.poisoned_owner(), Some(append.handle));

    let (append, mut transaction) = begin();
    ready_all(&mut transaction, append);
    assert!(
        transaction
            .confirm_forward_sync(Glm53TargetT1DeviceOutcome::AsyncFailure)
            .is_err()
    );
    assert!(transaction.is_poisoned());

    for mutation in 0..4 {
        let (append, mut transaction) = begin();
        ready_all(&mut transaction, append);
        transaction
            .confirm_forward_sync(Glm53TargetT1DeviceOutcome::Success)
            .unwrap();
        let mut wrong = device(
            transaction.decide(1).unwrap(),
            append,
            Glm53TargetT1Effect::CommitDsaIndexAllLayers,
        );
        match mutation {
            0 => wrong.kind = Glm53TargetT1Effect::CommitDsaLatentAllLayers,
            1 => wrong.transaction_nonce += 1,
            2 => wrong.exclusive_end += 1,
            _ => wrong.stream += 1,
        }
        assert!(
            transaction
                .complete_effect(wrong, Glm53TargetT1DeviceOutcome::Success)
                .is_err()
        );
    }

    let (append, mut transaction) = begin();
    ready_all(&mut transaction, append);
    transaction
        .confirm_forward_sync(Glm53TargetT1DeviceOutcome::Success)
        .unwrap();
    let effect = device(
        transaction.decide(1).unwrap(),
        append,
        Glm53TargetT1Effect::CommitDsaIndexAllLayers,
    );
    assert!(
        transaction
            .complete_effect(effect, Glm53TargetT1DeviceOutcome::LaunchFailure)
            .is_err()
    );
    assert!(transaction.is_poisoned());
}

#[test]
fn incomplete_receipts_final_sync_and_cpu_rejection_poison() {
    let (append, mut transaction) = begin();
    let mut wrong = kda_ready(append, 0);
    wrong.persistent_state_untouched = false;
    assert!(transaction.record_kda_layer(wrong).is_err());

    let (append, mut transaction) = begin();
    for layer in 0..3 {
        transaction
            .record_kda_layer(kda_ready(append, layer))
            .unwrap();
    }
    let mut wrong = dsa_ready(append, 3);
    wrong.current_visibility_ready = false;
    assert!(transaction.record_dsa_layer(wrong).is_err());

    let (append, mut transaction) = begin();
    ready_all(&mut transaction, append);
    transaction
        .confirm_forward_sync(Glm53TargetT1DeviceOutcome::Success)
        .unwrap();
    let mut transition = transaction.decide(1).unwrap();
    for effect in [
        Glm53TargetT1Effect::CommitDsaIndexAllLayers,
        Glm53TargetT1Effect::CommitDsaLatentAllLayers,
        Glm53TargetT1Effect::CommitKdaAllLayers,
    ] {
        let authorization = device(transition, append, effect);
        transition = transaction
            .complete_effect(authorization, Glm53TargetT1DeviceOutcome::Success)
            .unwrap();
    }
    let sync = device(transition, append, Glm53TargetT1Effect::FinalStreamSync);
    assert!(
        transaction
            .complete_effect(sync, Glm53TargetT1DeviceOutcome::AsyncFailure)
            .is_err()
    );

    let (append, mut transaction) = begin();
    ready_all(&mut transaction, append);
    transaction
        .confirm_forward_sync(Glm53TargetT1DeviceOutcome::Success)
        .unwrap();
    let retire = device(
        transaction.decide(0).unwrap(),
        append,
        Glm53TargetT1Effect::RetireRejectedMarkers,
    );
    let sync = device(
        transaction
            .complete_effect(retire, Glm53TargetT1DeviceOutcome::Success)
            .unwrap(),
        append,
        Glm53TargetT1Effect::FinalStreamSync,
    );
    transaction
        .complete_effect(sync, Glm53TargetT1DeviceOutcome::Success)
        .unwrap();
    assert!(
        transaction
            .confirm_cpu_publication(Glm53TargetT1CpuOutcome::Rejected)
            .is_err()
    );
}

fn region(offset_bytes: u64, payload_bytes: u64) -> Glm53T1StateRegion {
    Glm53T1StateRegion {
        offset_bytes,
        payload_bytes,
        allocation_bytes: (payload_bytes + 255) & !255,
    }
}

fn source_contract(source: &str) -> bool {
    let needles = [
        "pub const GLM53_TARGET_T1_TRANSACTION_BYTES: u64 = 156_008_192;",
        "pub const GLM53_KDA_T1_LAYERS: [u32; 34]",
        "pub const GLM53_DSA_T1_LAYERS: [u32; 11]",
        "append.handle.slot() != 0",
        "lease.stream == 0 || lease.capture_observed",
        "place(&mut cursor, 142_606_336)?",
        "place(&mut cursor, 13_369_344)?",
        "|| !receipt.persistent_state_untouched",
        "&& stream == self.stream",
        "|| effect.owner != self.append.handle",
        "|| effect.transaction_nonce != self.append.nonce",
        "RetireRejectedMarkers => Glm53TargetT1Effect::FinalStreamSync",
        "CommitDsaIndexAllLayers => {\n                Glm53TargetT1Effect::CommitDsaLatentAllLayers",
        "CommitDsaLatentAllLayers => {\n                Glm53TargetT1Effect::CommitKdaAllLayers",
        "CommitKdaAllLayers => Glm53TargetT1Effect::FinalStreamSync",
        "FinalStreamSync => return self.finish_device_transaction()",
        "self.phase = Glm53TargetT1Phase::Poisoned;",
    ];
    needles.iter().all(|needle| source.contains(needle))
        && !source.contains("impl Model for")
        && !source.contains("GpuBackend")
        && !source.contains("KernelLaunch")
        && !source.contains("copy_d2d")
        && !source.contains(".launch(")
}

#[test]
fn source_contract_rejects_core_mutations_and_effectful_claims() {
    let source = include_str!("t1_state_transaction.rs");
    assert!(source_contract(source));
    for (from, to) in [
        ("[u32; 34]", "[u32; 33]"),
        ("append.handle.slot() != 0", "append.handle.slot() != 1"),
        ("lease.stream == 0", "lease.stream == u64::MAX"),
        ("|| lease.capture_observed", "&& lease.capture_observed"),
        ("&& stream == self.stream", "&& stream != self.stream"),
        (
            "|| effect.owner != self.append.handle",
            "|| effect.owner == self.append.handle",
        ),
        (
            "|| effect.transaction_nonce != self.append.nonce",
            "|| effect.transaction_nonce == self.append.nonce",
        ),
        (
            "|| !receipt.persistent_state_untouched",
            "|| receipt.persistent_state_untouched",
        ),
        ("142_606_336", "142_606_080"),
        (
            "Glm53TargetT1Effect::CommitDsaLatentAllLayers =>",
            "Glm53TargetT1Effect::CommitKdaAllLayers =>",
        ),
        (
            "self.phase = Glm53TargetT1Phase::Poisoned;",
            "self.phase = Glm53TargetT1Phase::Complete;",
        ),
    ] {
        let mutant = source.replacen(from, to, 1);
        assert_ne!(mutant, source, "missing mutation needle: {from}");
        assert!(!source_contract(&mutant), "mutation survived: {from}");
    }
}

#[path = "t1_state_transaction_boundary_tests.rs"]
mod boundary_tests;
