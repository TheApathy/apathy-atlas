// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::kv_cache::{Glm53DsaCache, Glm53DsaStorage};

use super::*;

fn begin_at(length: u32) -> (Glm53DsaCache, Glm53DsaAppendPlan, Glm53DsaT1Transaction) {
    let mut cache = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let handle = cache.claim_sequence().unwrap();
    if length != 0 {
        let prefix = cache.begin_append(handle, length).unwrap();
        cache.commit_append(prefix, length).unwrap();
    }
    let append = cache.begin_append(handle, 1).unwrap();
    let transaction = Glm53DsaT1Transaction::begin(append).unwrap();
    (cache, append, transaction)
}

fn ready(append: Glm53DsaAppendPlan, layer: u32) -> Glm53DsaT1LayerReady {
    Glm53DsaT1LayerReady {
        owner: append.handle,
        layer,
        transaction_nonce: append.nonce,
        exclusive_end: append.end_position,
        latent_ready: true,
        index_ready: true,
        current_visibility_ready: true,
        device_status: 0,
    }
}

fn stage_all(transaction: &mut Glm53DsaT1Transaction, append: Glm53DsaAppendPlan) {
    for layer in GLM53_DSA_T1_LAYERS {
        let permission = transaction
            .record_layer_ready(ready(append, layer))
            .unwrap();
        assert_eq!(permission.layer, layer);
        assert_eq!(permission.view.visible_len, append.end_position);
        assert_eq!(permission.view.query_position, append.start_position);
        assert!(!permission.view.persistent_tail_write_allowed);
    }
}

fn expect_device(transition: Glm53DsaT1Transition, expected: Glm53DsaT1Effect) {
    assert_eq!(transition, Glm53DsaT1Transition::Device(expected));
}

#[test]
fn kpool_current_visibility_is_exact_at_every_boundary() {
    assert_eq!(
        GLM53_DSA_T1_LAYERS,
        [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    );
    for (length, before, after, initial_tail, final_tail, future_pool, score_pools) in [
        (0, 0, 0, 0, 1, None, 1),
        (1, 0, 0, 1, 2, None, 1),
        (2, 0, 0, 2, 3, None, 1),
        (3, 0, 1, 3, 0, Some(0), 1),
        (1_048_575, 262_143, 262_144, 3, 0, Some(262_143), 262_144),
    ] {
        let (_, append, transaction) = begin_at(length);
        let view = transaction.planned_visibility();
        assert_eq!(view.committed_len, length);
        assert_eq!(view.visible_len, length + 1);
        assert_eq!(view.future_latent_row, length);
        assert_eq!(view.future_pool_row, future_pool);
        assert_eq!(view.committed_complete_pools, before);
        assert_eq!(view.visible_complete_pools, after);
        assert_eq!(view.initial_tail_len, initial_tail);
        assert_eq!(view.staged_tail_len, final_tail);
        assert_eq!(view.score_pool_count, score_pools);
        assert!(view.use_staged_tail);
        assert!(!view.persistent_tail_write_allowed);
        assert_eq!(append.start_position, length);
    }
}

#[test]
fn accepted_one_orders_index_latent_sync_then_cpu_publication() {
    let (mut cache, append, mut transaction) = begin_at(3);
    stage_all(&mut transaction, append);
    assert_eq!(transaction.ready_mask(), (1u16 << 11) - 1);
    assert_eq!(transaction.phase(), Glm53DsaT1Phase::AwaitingForwardSync);
    transaction
        .confirm_forward_sync(Glm53DsaT1DeviceOutcome::Success)
        .unwrap();
    expect_device(
        transaction.decide(1).unwrap(),
        Glm53DsaT1Effect::CommitIndexAllLayers,
    );
    expect_device(
        transaction
            .complete_effect(
                Glm53DsaT1Effect::CommitIndexAllLayers,
                Glm53DsaT1DeviceOutcome::Success,
            )
            .unwrap(),
        Glm53DsaT1Effect::CommitLatentAllLayers,
    );
    expect_device(
        transaction
            .complete_effect(
                Glm53DsaT1Effect::CommitLatentAllLayers,
                Glm53DsaT1DeviceOutcome::Success,
            )
            .unwrap(),
        Glm53DsaT1Effect::FinalStreamSync,
    );
    let publication = transaction
        .complete_effect(
            Glm53DsaT1Effect::FinalStreamSync,
            Glm53DsaT1DeviceOutcome::Success,
        )
        .unwrap();
    let Glm53DsaT1Transition::PublishCpu(publication) = publication else {
        panic!("final sync did not emit the CPU publication receipt");
    };
    assert_eq!(cache.sequence_view(append.handle).unwrap().logical_len, 3);
    let (owned_append, accepted) = publication.into_parts();
    assert_eq!(
        cache
            .commit_append(owned_append, accepted)
            .unwrap()
            .logical_len,
        4
    );
    transaction
        .confirm_cpu_publication(Glm53DsaT1CpuOutcome::Success)
        .unwrap();
    assert_eq!(transaction.phase(), Glm53DsaT1Phase::Complete);
}

#[test]
fn accepted_zero_retires_markers_and_publishes_unchanged_length_last() {
    let (mut cache, append, mut transaction) = begin_at(2);
    stage_all(&mut transaction, append);
    transaction
        .confirm_forward_sync(Glm53DsaT1DeviceOutcome::Success)
        .unwrap();
    expect_device(
        transaction.decide(0).unwrap(),
        Glm53DsaT1Effect::RetireRejectedMarkers,
    );
    expect_device(
        transaction
            .complete_effect(
                Glm53DsaT1Effect::RetireRejectedMarkers,
                Glm53DsaT1DeviceOutcome::Success,
            )
            .unwrap(),
        Glm53DsaT1Effect::FinalStreamSync,
    );
    let publication = transaction
        .complete_effect(
            Glm53DsaT1Effect::FinalStreamSync,
            Glm53DsaT1DeviceOutcome::Success,
        )
        .unwrap();
    let Glm53DsaT1Transition::PublishCpu(publication) = publication else {
        panic!("rejection did not emit the CPU publication receipt");
    };
    let (owned_append, accepted) = publication.into_parts();
    assert_eq!(accepted, 0);
    let view = cache.commit_append(owned_append, accepted).unwrap();
    transaction
        .confirm_cpu_publication(Glm53DsaT1CpuOutcome::Success)
        .unwrap();
    assert_eq!((view.logical_len, view.tail_len), (2, 2));
    assert_eq!(transaction.phase(), Glm53DsaT1Phase::Complete);
}

#[test]
fn stale_owner_nonce_status_and_order_irreversibly_poison() {
    let (_, append, mut wrong_order) = begin_at(0);
    assert!(wrong_order.record_layer_ready(ready(append, 7)).is_err());
    assert!(wrong_order.is_poisoned());
    assert!(wrong_order.record_layer_ready(ready(append, 3)).is_err());

    let (_, append, mut wrong_nonce) = begin_at(0);
    let mut receipt = ready(append, 3);
    receipt.transaction_nonce += 1;
    assert!(wrong_nonce.record_layer_ready(receipt).is_err());
    assert!(wrong_nonce.is_poisoned());

    let (_, append, mut wrong_end) = begin_at(0);
    let mut receipt = ready(append, 3);
    receipt.exclusive_end += 1;
    assert!(wrong_end.record_layer_ready(receipt).is_err());
    assert!(wrong_end.is_poisoned());

    let (_, append, mut failed_status) = begin_at(0);
    let mut receipt = ready(append, 3);
    receipt.device_status = 17;
    assert!(failed_status.record_layer_ready(receipt).is_err());
    assert!(failed_status.is_poisoned());

    for missing in 0..3 {
        let (_, append, mut incomplete) = begin_at(0);
        let mut receipt = ready(append, 3);
        match missing {
            0 => receipt.latent_ready = false,
            1 => receipt.index_ready = false,
            _ => receipt.current_visibility_ready = false,
        }
        assert!(incomplete.record_layer_ready(receipt).is_err());
        assert!(incomplete.is_poisoned());
    }

    let mut owners = Glm53DsaCache::new(1, Glm53DsaStorage::Bf16).unwrap();
    let stale_generation = owners.claim_sequence().unwrap();
    owners.free_sequence(stale_generation).unwrap();
    let current_generation = owners.claim_sequence().unwrap();
    let current_append = owners.begin_append(current_generation, 1).unwrap();
    let mut transaction = Glm53DsaT1Transaction::begin(current_append).unwrap();
    let mut stale = ready(current_append, 3);
    stale.owner = stale_generation;
    assert!(transaction.record_layer_ready(stale).is_err());
    assert!(transaction.is_poisoned());
    assert_eq!(transaction.poisoned_owner(), Some(current_generation));

    let (_, append, mut duplicate) = begin_at(0);
    duplicate.record_layer_ready(ready(append, 3)).unwrap();
    assert!(duplicate.record_layer_ready(ready(append, 3)).is_err());
    assert!(duplicate.is_poisoned());
}

#[test]
fn premature_decision_missing_layer_or_any_device_failure_poison() {
    let (_, _, mut premature) = begin_at(0);
    assert!(premature.decide(1).is_err());
    assert!(premature.is_poisoned());

    let (_, append, mut invalid_acceptance) = begin_at(0);
    stage_all(&mut invalid_acceptance, append);
    invalid_acceptance
        .confirm_forward_sync(Glm53DsaT1DeviceOutcome::Success)
        .unwrap();
    assert!(invalid_acceptance.decide(2).is_err());
    assert!(invalid_acceptance.is_poisoned());

    let (_, append, mut missing) = begin_at(0);
    missing.record_layer_ready(ready(append, 3)).unwrap();
    assert!(
        missing
            .confirm_forward_sync(Glm53DsaT1DeviceOutcome::Success)
            .is_err()
    );
    assert!(missing.is_poisoned());

    for failure in [
        Glm53DsaT1DeviceOutcome::LaunchFailure,
        Glm53DsaT1DeviceOutcome::AsyncFailure,
        Glm53DsaT1DeviceOutcome::StatusFailure(1),
    ] {
        let (_, append, mut transaction) = begin_at(0);
        stage_all(&mut transaction, append);
        assert!(transaction.confirm_forward_sync(failure).is_err());
        assert!(transaction.is_poisoned());
    }

    let (_, append, mut commit_failure) = begin_at(0);
    stage_all(&mut commit_failure, append);
    commit_failure
        .confirm_forward_sync(Glm53DsaT1DeviceOutcome::Success)
        .unwrap();
    commit_failure.decide(1).unwrap();
    assert!(
        commit_failure
            .complete_effect(
                Glm53DsaT1Effect::CommitIndexAllLayers,
                Glm53DsaT1DeviceOutcome::AsyncFailure,
            )
            .is_err()
    );
    assert!(commit_failure.is_poisoned());

    let (_, append, mut reordered) = begin_at(0);
    stage_all(&mut reordered, append);
    reordered
        .confirm_forward_sync(Glm53DsaT1DeviceOutcome::Success)
        .unwrap();
    reordered.decide(1).unwrap();
    assert!(
        reordered
            .complete_effect(
                Glm53DsaT1Effect::CommitLatentAllLayers,
                Glm53DsaT1DeviceOutcome::Success,
            )
            .is_err()
    );
    assert!(reordered.is_poisoned());

    let (cache, append, mut final_sync_failure) = begin_at(0);
    stage_all(&mut final_sync_failure, append);
    final_sync_failure
        .confirm_forward_sync(Glm53DsaT1DeviceOutcome::Success)
        .unwrap();
    final_sync_failure.decide(1).unwrap();
    final_sync_failure
        .complete_effect(
            Glm53DsaT1Effect::CommitIndexAllLayers,
            Glm53DsaT1DeviceOutcome::Success,
        )
        .unwrap();
    final_sync_failure
        .complete_effect(
            Glm53DsaT1Effect::CommitLatentAllLayers,
            Glm53DsaT1DeviceOutcome::Success,
        )
        .unwrap();
    assert!(
        final_sync_failure
            .complete_effect(
                Glm53DsaT1Effect::FinalStreamSync,
                Glm53DsaT1DeviceOutcome::AsyncFailure,
            )
            .is_err()
    );
    assert!(final_sync_failure.is_poisoned());
    assert_eq!(cache.sequence_view(append.handle).unwrap().logical_len, 0);
}

#[test]
fn stale_cpu_nonce_after_rollback_and_reuse_poison_before_completion() {
    let (mut cache, append, mut transaction) = begin_at(0);
    stage_all(&mut transaction, append);
    transaction
        .confirm_forward_sync(Glm53DsaT1DeviceOutcome::Success)
        .unwrap();

    cache.rollback_append(append).unwrap();
    let replacement = cache.begin_append(append.handle, 1).unwrap();
    assert_ne!(replacement.nonce, append.nonce);

    transaction.decide(1).unwrap();
    transaction
        .complete_effect(
            Glm53DsaT1Effect::CommitIndexAllLayers,
            Glm53DsaT1DeviceOutcome::Success,
        )
        .unwrap();
    transaction
        .complete_effect(
            Glm53DsaT1Effect::CommitLatentAllLayers,
            Glm53DsaT1DeviceOutcome::Success,
        )
        .unwrap();
    let publication = transaction
        .complete_effect(
            Glm53DsaT1Effect::FinalStreamSync,
            Glm53DsaT1DeviceOutcome::Success,
        )
        .unwrap();
    let Glm53DsaT1Transition::PublishCpu(publication) = publication else {
        panic!("final sync did not emit CPU publication");
    };
    let (stale_append, accepted) = publication.into_parts();
    assert!(cache.commit_append(stale_append, accepted).is_err());
    assert!(
        transaction
            .confirm_cpu_publication(Glm53DsaT1CpuOutcome::Rejected)
            .is_err()
    );
    assert!(transaction.is_poisoned());
    assert_eq!(transaction.poisoned_owner(), Some(append.handle));
    assert_eq!(cache.sequence_view(append.handle).unwrap().logical_len, 0);
    cache.rollback_append(replacement).unwrap();
}

#[test]
fn forged_append_geometry_and_unimplemented_effects_stay_closed() {
    let (_, append, _) = begin_at(3);
    let mut forged = append;
    forged.complete_pools_to_write = 0;
    assert!(Glm53DsaT1Transaction::begin(forged).is_err());
    forged = append;
    forged.token_count = 2;
    assert!(Glm53DsaT1Transaction::begin(forged).is_err());
    assert!(!GLM53_DSA_T1_EXECUTION_IMPLEMENTED);
    assert_eq!(
        GLM53_DSA_T1_MISSING_CAPABILITIES,
        [
            Glm53DsaT1MissingCapability::ExactNormAndIndexerProjection,
            Glm53DsaT1MissingCapability::CurrentTokenVisibilityMaterializer,
            Glm53DsaT1MissingCapability::AllLayerIndexCommit,
            Glm53DsaT1MissingCapability::DsaLayerComposer,
            Glm53DsaT1MissingCapability::CompleteScratchPlan,
            Glm53DsaT1MissingCapability::FalliblePoisonOwner,
        ]
    );

    let source = include_str!("glm53_dsa_t1_transaction.rs");
    for forbidden in [
        "GpuBackend",
        "KernelLaunch",
        "impl Model",
        ".commit_append(",
    ] {
        assert!(
            !source.contains(forbidden),
            "unexpected effectful source: {forbidden}"
        );
    }
}
