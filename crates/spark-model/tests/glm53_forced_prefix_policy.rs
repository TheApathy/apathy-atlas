// SPDX-License-Identifier: AGPL-3.0-only

//! Forced acceptance exercises commit paths, not natural model quality.
#[path = "../examples/glm53_partial_replay_parity/forced_policy.rs"]
mod forced_policy;
use forced_policy::ForcedPrefixPolicy;
use spark_model::model::glm53::verify_policy_transaction::{PolicyAdvance, VerifyPolicy};

#[test]
fn every_width_and_committed_prefix_selects_exactly_the_requested_rejection_row() {
    let logits = vec![0u8; 154_880 * 2];
    for width in 2..=8 {
        let inputs = (0..width).map(|i| 101 + i as u32).collect::<Vec<_>>();
        for rows in 1..=width {
            let mut policy = ForcedPrefixPolicy::new(inputs.clone(), rows).unwrap();
            policy.checkpoint().unwrap();
            for row in 0..rows {
                let picked = policy.pick(row, &logits).unwrap();
                assert!(picked < 154_880);
                if row + 1 < rows {
                    assert_eq!(picked, inputs[row + 1]);
                } else if rows < width {
                    assert_ne!(picked, inputs[row + 1]);
                }
                assert!(matches!(
                    policy.advance(picked).unwrap(),
                    PolicyAdvance::Continue
                ));
            }
            assert!(
                policy.pick(rows, &logits).is_err(),
                "no work after the forced commit extent"
            );
            policy.restore().unwrap();
            let picked = policy.pick(0, &logits).unwrap();
            if rows == 1 {
                assert_ne!(picked, inputs[1]);
            } else {
                assert_eq!(picked, inputs[1]);
            }
        }
    }
}

#[test]
fn invalid_geometry_and_protocol_order_do_not_advance_the_policy_cursor() {
    for width in [0, 1, 9] {
        assert!(ForcedPrefixPolicy::new(vec![1; width], 1).is_err());
    }
    for rows in [0, 9, usize::MAX] {
        assert!(ForcedPrefixPolicy::new(vec![1; 8], rows).is_err());
    }
    assert!(ForcedPrefixPolicy::new(vec![0, 154_880], 1).is_err());
    let logits = vec![0u8; 154_880 * 2];
    let mut policy = ForcedPrefixPolicy::new(vec![10, 11, 12], 2).unwrap();
    assert!(policy.pick(1, &logits).is_err());
    assert!(policy.pick(0, &logits[..10]).is_err());
    assert!(policy.advance(11).is_err());
    policy.checkpoint().unwrap();
    assert_eq!(policy.pick(0, &logits).unwrap(), 11);
    assert!(policy.advance(12).is_err());
    assert!(matches!(
        policy.advance(11).unwrap(),
        PolicyAdvance::Continue
    ));
    assert!(policy.pick(0, &logits).is_err());
    policy.restore().unwrap();
    assert_eq!(policy.pick(0, &logits).unwrap(), 11);
}

#[test]
fn rejection_wrap_is_in_vocabulary_and_distinct_from_the_last_token_id() {
    let logits = vec![0u8; 154_880 * 2];
    let mut policy = ForcedPrefixPolicy::new(vec![42, 154_879], 1).unwrap();
    policy.checkpoint().unwrap();
    assert_eq!(policy.pick(0, &logits).unwrap(), 0);
}

#[test]
fn published_emission_and_nonterminal_status_are_both_required_for_exactness() {
    for width in 2..=8 {
        let inputs = (0..width).map(|i| 40 + i as u32).collect::<Vec<_>>();
        for rows in 1..=width {
            let policy = ForcedPrefixPolicy::new(inputs.clone(), rows).unwrap();
            let mut expected = inputs[1..rows].to_vec();
            expected.push(if rows < width { inputs[rows] + 1 } else { 0 });
            assert!(policy.publication_matches(&expected, false));
            assert!(!policy.publication_matches(&expected, true));
            assert!(!policy.publication_matches(&expected[..expected.len() - 1], false));
            let mut changed = expected.clone();
            changed[0] = (changed[0] + 1) % 154_880;
            assert!(!policy.publication_matches(&changed, false));
            let mut extended = expected.clone();
            extended.push(0);
            assert!(!policy.publication_matches(&extended, false));
        }
    }
}
