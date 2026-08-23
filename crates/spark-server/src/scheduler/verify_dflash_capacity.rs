// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed DFlash capacity checks before scheduler/model side effects.

use anyhow::Result;
use spark_model::layers::DDTreePayload;

fn validate_frame(
    physical_k: Option<usize>,
    requested_max: usize,
    drafts_len: usize,
    tree: Option<&DDTreePayload>,
) -> Result<usize> {
    let physical_k = physical_k
        .ok_or_else(|| anyhow::anyhow!("DFlash model omitted its physical verify capacity"))?;
    anyhow::ensure!(
        physical_k > 0,
        "DFlash model reported zero physical verify capacity"
    );
    anyhow::ensure!(drafts_len > 0, "DFlash verify frame has no draft rows");
    anyhow::ensure!(
        drafts_len <= requested_max,
        "DFlash proposer returned {drafts_len} drafts above caller maximum {requested_max}"
    );

    let mut non_root_rows = drafts_len;
    if let Some(tree) = tree {
        let tree_len = tree.tree_token_ids.len();
        anyhow::ensure!(
            tree_len > 0 && tree.parent_indices.len() == tree_len,
            "DFlash tree topology length mismatch: tokens={tree_len}, parents={}",
            tree.parent_indices.len()
        );
        // A tree payload is bounded by the PHYSICAL verify capacity, not by the
        // caller's flat draft count. `requested_max` is gamma; a DDTree run
        // deliberately emits more non-root rows than that (spine gamma plus
        // sibling branches) into buffers sized by ATLAS_DDTREE_MAX_NODES. The
        // `verify_k <= physical_k` check below is the real capacity bound and
        // is still enforced unconditionally.
        //
        // Left as-is by default so the flat path keeps its tightest bound;
        // ATLAS_DDTREE_UNCAP=1 selects the physical bound, and must be set
        // together with the same flag on the proposer side (draft_budget.rs) —
        // relaxing only one of the two produces a payload that is then dropped
        // before verify with `pending_drafts` cleared and nothing emitted.
        let tree_bound = if std::env::var("ATLAS_DDTREE_UNCAP").ok().as_deref() == Some("1") {
            physical_k.saturating_sub(1)
        } else {
            requested_max
        };
        anyhow::ensure!(
            tree_len <= tree_bound,
            "DFlash tree returned {tree_len} nodes above the {} maximum {tree_bound}",
            if tree_bound == requested_max {
                "caller"
            } else {
                "physical"
            }
        );
        for (child, &parent) in tree.parent_indices.iter().enumerate() {
            anyhow::ensure!(
                parent == -1 || (parent >= 0 && (parent as usize) < child),
                "DFlash tree parent {parent} is invalid for child {child}"
            );
        }
        non_root_rows = non_root_rows.max(tree_len);
    }

    let verify_k = non_root_rows
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("DFlash verify width overflow"))?;
    anyhow::ensure!(
        verify_k <= physical_k,
        "DFlash verify K={verify_k} exceeds physical model capacity K={physical_k}"
    );
    Ok(verify_k)
}

fn validate_batch<'a>(
    physical_k: Option<usize>,
    requested_max: usize,
    frames: impl IntoIterator<Item = (usize, Option<&'a DDTreePayload>)>,
) -> Result<usize> {
    let mut expected_k = None;
    let mut count = 0usize;
    for (drafts_len, tree) in frames {
        anyhow::ensure!(
            tree.is_none(),
            "batched DFlash verify is flat-only and cannot discard tree token order"
        );
        let verify_k = validate_frame(physical_k, requested_max, drafts_len, tree)?;
        if let Some(expected_k) = expected_k {
            anyhow::ensure!(
                verify_k == expected_k,
                "heterogeneous DFlash batch widths: K={verify_k} != K={expected_k}"
            );
        } else {
            expected_k = Some(verify_k);
        }
        count += 1;
    }
    anyhow::ensure!(count > 0, "DFlash verify batch is empty");
    Ok(expected_k.expect("nonempty batch has a width"))
}

pub(in crate::scheduler) fn preflight_frame_then<T>(
    is_dflash: bool,
    capacity: impl FnOnce() -> Option<usize>,
    requested_max: usize,
    drafts_len: usize,
    tree: Option<&DDTreePayload>,
    first_effect: impl FnOnce() -> T,
) -> Result<T> {
    if is_dflash {
        validate_frame(capacity(), requested_max, drafts_len, tree)?;
    }
    Ok(first_effect())
}

pub(in crate::scheduler) fn preflight_batch_then<'a, T>(
    is_dflash: bool,
    capacity: impl FnOnce() -> Option<usize>,
    requested_max: usize,
    frames: impl IntoIterator<Item = (usize, Option<&'a DDTreePayload>)>,
    first_effect: impl FnOnce() -> T,
) -> Result<T> {
    if is_dflash {
        validate_batch(capacity(), requested_max, frames)?;
    }
    Ok(first_effect())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    fn tree(tokens: usize, parents: Vec<i32>) -> DDTreePayload {
        DDTreePayload {
            tree_token_ids: vec![7; tokens],
            parent_indices: parents,
        }
    }

    #[test]
    fn absent_zero_requested_and_physical_overflow_fail_closed() {
        assert!(validate_frame(None, 7, 7, None).is_err());
        assert!(validate_frame(Some(0), 7, 7, None).is_err());
        assert!(validate_frame(Some(8), 0, 1, None).is_err());
        assert!(validate_frame(Some(8), 7, 8, None).is_err());
        assert!(validate_frame(Some(8), 9, 8, None).is_err());
        assert_eq!(validate_frame(Some(8), 7, 7, None).unwrap(), 8);
    }

    #[test]
    fn tree_shape_topology_and_independent_width_are_validated() {
        assert!(validate_frame(Some(8), 7, 2, Some(&tree(2, vec![-1]))).is_err());
        for parents in [vec![-2], vec![0], vec![-1, 1], vec![-1, 2]] {
            assert!(validate_frame(Some(8), 7, 2, Some(&tree(parents.len(), parents))).is_err());
        }
        let wide = tree(6, vec![-1, 0, 1, -1, 3, 4]);
        assert_eq!(validate_frame(Some(7), 6, 4, Some(&wide)).unwrap(), 7);
        assert!(validate_frame(Some(6), 6, 4, Some(&wide)).is_err());
    }

    #[test]
    fn batch_rejects_heterogeneous_width_and_every_tree() {
        assert!(validate_batch(Some(8), 7, [(7, None), (6, None)]).is_err());
        assert!(validate_batch(Some(8), 7, []).is_err());
        let payload = tree(2, vec![-1, 0]);
        assert!(validate_batch(Some(8), 7, [(7, None), (7, Some(&payload))]).is_err());
    }

    #[test]
    fn production_preflight_is_lazy_and_native_bypasses_capacity() {
        let capacity_calls = Cell::new(0);
        let effect_calls = Cell::new(0);
        let rejected = preflight_frame_then(
            true,
            || {
                capacity_calls.set(capacity_calls.get() + 1);
                Some(4)
            },
            7,
            7,
            None,
            || effect_calls.set(effect_calls.get() + 1),
        );
        assert!(rejected.is_err());
        assert_eq!((capacity_calls.get(), effect_calls.get()), (1, 0));

        preflight_frame_then(
            false,
            || {
                capacity_calls.set(capacity_calls.get() + 1);
                None
            },
            7,
            99,
            None,
            || effect_calls.set(effect_calls.get() + 1),
        )
        .unwrap();
        assert_eq!((capacity_calls.get(), effect_calls.get()), (1, 1));
    }

    #[test]
    fn batched_tree_rejection_prevents_first_effect() {
        let payload = tree(2, vec![-1, 0]);
        let effects = Cell::new(0);
        let result = preflight_batch_then(
            true,
            || Some(8),
            7,
            [(7, None), (7, Some(&payload))],
            || effects.set(effects.get() + 1),
        );
        assert!(result.is_err());
        assert_eq!(effects.get(), 0);
    }
}
