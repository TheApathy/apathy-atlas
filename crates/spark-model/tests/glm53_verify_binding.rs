// SPDX-License-Identifier: AGPL-3.0-only

use spark_model::model::glm53::verify_policy_binding::{VerifyBindingOwner, VerifyFrame};
use spark_model::model::glm53::verify_policy_transaction;
#[path = "support/glm53_verify_policy_fixture.rs"]
mod fixture;

fn frame() -> VerifyFrame {
    VerifyFrame {
        generation: 3,
        nonce: 7,
        position: 4,
        capacity: 64,
        vocab: 3,
        stream: 19,
    }
}

#[test]
fn binding_owns_inputs_and_exact_host_prefix() {
    let owner = VerifyBindingOwner::new();
    let mut prefix = vec![0, 1, 2, 0];
    let mut inputs = vec![0, 1];
    let bound = owner.bind(frame(), &prefix, &inputs).unwrap();
    prefix.fill(2);
    inputs.fill(2);
    let request = owner.consume(bound, frame()).unwrap();
    assert_eq!(request.inputs(), [0, 1]);
    assert_eq!(request.start(), 4);
    assert_eq!(request.host_prefix(), Some(&[0, 1, 2, 0][..]));
}

#[test]
fn a_different_model_cannot_consume_the_binding() {
    let owner = VerifyBindingOwner::new();
    let other = VerifyBindingOwner::new();
    let bound = owner.bind(frame(), &[0; 4], &[0, 1]).unwrap();
    assert!(other.consume(bound, frame()).is_err());
}

#[test]
fn every_live_frame_component_is_revalidated() {
    let owner = VerifyBindingOwner::new();
    for axis in 0..7 {
        let bound = owner.bind(frame(), &[0; 4], &[0, 1]).unwrap();
        let mut now = frame();
        match axis {
            0 => now.generation += 1,
            1 => now.nonce += 1,
            2 => now.position += 1,
            3 => now.capacity += 1,
            4 => now.vocab += 1,
            5 => now.stream += 1,
            _ => now.generation -= 1,
        }
        assert!(owner.consume(bound, now).is_err(), "axis {axis}");
    }
}

#[test]
fn malformed_host_prefix_is_rejected_at_binding() {
    let owner = VerifyBindingOwner::new();
    assert!(owner.bind(frame(), &[0; 3], &[0, 1]).is_err());
    assert!(owner.bind(frame(), &[0, 1, 3, 0], &[0, 1]).is_err());
    assert!(owner.bind(frame(), &[0; 4], &[0]).is_err());
}

#[test]
fn committed_bound_receipt_rejects_same_length_different_prefix_without_mutation() {
    let owner = VerifyBindingOwner::new();
    let bound = owner.bind(frame(), &[0, 1, 2, 0], &[0, 1]).unwrap();
    let request = owner.consume(bound, frame()).unwrap();
    let (mut source, mut policy, mut target, _) =
        fixture::fixture(&[vec![0., 10., 9.], vec![0., 1., 12.]]);
    let receipt = fixture::committed(
        verify_policy_transaction::run_verify_policy_transaction(
            &request,
            &mut source,
            &mut policy,
            &mut target,
        )
        .unwrap(),
    );
    let (mut tokens, mut len, mut valid) = (vec![0; 4], 4, 4);
    let before = (tokens.clone(), len, valid);
    assert!(receipt.publish(&mut tokens, &mut len, &mut valid).is_err());
    assert_eq!((tokens, len, valid), before);
}

#[test]
fn committed_bound_receipt_publishes_exact_original_prefix() {
    let owner = VerifyBindingOwner::new();
    let bound = owner.bind(frame(), &[0, 1, 2, 0], &[0, 1]).unwrap();
    let request = owner.consume(bound, frame()).unwrap();
    let (mut source, mut policy, mut target, _) =
        fixture::fixture(&[vec![0., 10., 9.], vec![0., 1., 12.]]);
    let receipt = fixture::committed(
        verify_policy_transaction::run_verify_policy_transaction(
            &request,
            &mut source,
            &mut policy,
            &mut target,
        )
        .unwrap(),
    );
    let (mut tokens, mut len, mut valid) = (vec![0, 1, 2, 0], 4, 4);
    let published = receipt.publish(&mut tokens, &mut len, &mut valid).unwrap();
    assert_eq!(tokens, [0, 1, 2, 0, 0, 1]);
    assert_eq!((len, valid), (6, 6));
    assert_eq!(published.emitted_tokens(), [1, 2]);
}
