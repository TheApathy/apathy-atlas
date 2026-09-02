// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[path = "t1_state_transaction_sha256.rs"]
mod source_sha256;

fn execution_claim_stays_closed(source: &str) -> bool {
    source_sha256::matches(source)
        && source.contains("pub const GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED: bool = false;")
        && !source.contains("pub const GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED: bool = true;")
}

#[test]
fn execution_claim_is_false_and_true_mutation_rejects() {
    let source = include_str!("t1_state_transaction.rs");
    assert!(!GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED);
    assert!(execution_claim_stays_closed(source));
    let mutant = source.replacen(
        "GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED: bool = false",
        "GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED: bool = true",
        1,
    );
    assert_ne!(mutant, source);
    assert!(!execution_claim_stays_closed(&mutant));

    let conditional = source.replacen(
        "pub const GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED: bool = false;",
        "#[cfg(test)]\npub const GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED: bool = false;\n#[cfg(not(test))]\npub const GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED: bool = !false;",
        1,
    );
    assert_ne!(conditional, source);
    assert!(!execution_claim_stays_closed(&conditional));
}

#[test]
fn source_sha256_vectors_cover_the_padding_boundary() {
    assert_eq!(
        source_sha256::hex(source_sha256::digest(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        source_sha256::hex(source_sha256::digest(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        source_sha256::hex(source_sha256::digest(&[b'a'; 55])),
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
    );
    assert_eq!(
        source_sha256::hex(source_sha256::digest(&[b'a'; 56])),
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
    );
}

#[test]
fn unbranded_cpu_success_documents_the_activation_boundary() {
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
    let publication = transaction
        .complete_effect(sync, Glm53TargetT1DeviceOutcome::Success)
        .unwrap();
    drop(publication);

    transaction
        .confirm_cpu_publication(Glm53TargetT1CpuOutcome::Success)
        .unwrap();
    assert_eq!(transaction.phase(), Glm53TargetT1Phase::Complete);
    assert!(!GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED);
}
