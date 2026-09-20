// SPDX-License-Identifier: AGPL-3.0-only

//! Reuse the existing selector oracle; never change its supplied-mask semantics.

use super::reference;

#[test]
fn current_visible_masks_include_current_tokens_while_prior_masks_demonstrate_omission() {
    let scores = [3.0, 2.0, 1.0];
    let pools_valid = [true; 3];
    for position in 4usize..=9 {
        let prior: [bool; 3] = std::array::from_fn(|slot| slot < position % 4);
        let current: [bool; 3] = std::array::from_fn(|slot| slot < (position + 1) % 4);
        let actual = reference(
            &scores,
            &pools_valid,
            position + 1,
            &current,
            position,
            true,
        );
        let wanted: Vec<i32> = (0..=position).map(|index| index as i32).collect();
        assert_eq!(&actual[..wanted.len()], wanted.as_slice(), "q={position}");
        assert!(actual[wanted.len()..].iter().all(|&index| index == -1));
        let old = reference(&scores, &pools_valid, position + 1, &prior, position, true);
        if position % 4 == 3 {
            assert_eq!(old, actual, "pool-completing control q={position}");
        } else {
            assert!(
                !old.contains(&(position as i32)),
                "negative control q={position}"
            );
            assert_ne!(
                old, actual,
                "fixture must distinguish prior from current mask"
            );
        }
    }
}

#[test]
fn post_pool_mask_preserves_invalid_earlier_tail_entries_and_query_rejection() {
    let scores = [3.0, 2.0];
    let pools_valid = [true; 2];
    let prior = [true, false, false];
    let current = [true, false, true];
    let old = reference(&scores, &pools_valid, 7, &prior, 6, true);
    let actual = reference(&scores, &pools_valid, 7, &current, 6, true);
    assert_eq!(&old[..8], &[0, 1, 2, 3, 4, -1, -1, -1]);
    assert_eq!(&actual[..8], &[0, 1, 2, 3, 4, -1, 6, -1]);
    assert!(actual[8..].iter().all(|&index| index == -1));
    for (position, valid) in [(6, false), (7, true)] {
        assert!(
            reference(&scores, &pools_valid, 7, &current, position, valid)
                .iter()
                .all(|&index| index == -1)
        );
    }
    assert_eq!(prior, [true, false, false]);
    assert_eq!(current, [true, false, true]);
    assert_eq!(scores, [3.0, 2.0]);
}

#[test]
fn completing_pools_do_not_depend_on_a_raw_tail_mask() {
    for position in [3usize, 7, 11] {
        let output = reference(
            &[3.0, 2.0, 1.0],
            &[true; 3],
            position + 1,
            &[false; 3],
            position,
            true,
        );
        let wanted: Vec<i32> = (0..=position).map(|index| index as i32).collect();
        assert_eq!(&output[..wanted.len()], wanted.as_slice());
        assert!(output[wanted.len()..].iter().all(|&index| index == -1));
    }
}
