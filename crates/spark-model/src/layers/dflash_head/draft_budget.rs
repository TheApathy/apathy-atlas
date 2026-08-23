// SPDX-License-Identifier: AGPL-3.0-only

//! Single-source draft and verify-capacity limits for DFlash proposal paths.

use anyhow::Result;

use super::DflashProposerState;
use super::ddtree::TreePayload;

/// Immutable limits resolved once at the outer `DraftProposer::propose` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DflashDraftBudget {
    /// Public caller ceiling from `DraftProposer::propose`.
    pub requested: usize,
    /// Neural/flat forward width after trained/runtime and physical limits.
    pub flat: usize,
    /// Maximum non-root candidates exposed to verify.
    pub tree_nodes: usize,
    /// Physical verify width, including the bonus/root token.
    pub physical_verify_k: usize,
}

impl DflashDraftBudget {
    pub fn new(requested: usize, trained_width: usize, physical_verify_k: usize) -> Result<Self> {
        if physical_verify_k == 0 {
            anyhow::bail!("DFlash physical verify capacity must be at least one")
        }
        let physical_drafts = physical_verify_k - 1;
        // `tree_nodes` is the ceiling on NON-ROOT CANDIDATES exposed to verify.
        // Capping it by `requested` conflates two different quantities:
        // `requested` is how many FLAT drafts the caller wants (gamma), while
        // the tree budget is how many verify ROWS the physical buffers can hold
        // (ATLAS_DDTREE_MAX_NODES). Because `flat` is also bounded by
        // `requested`, the original `tree_nodes: requested.min(physical_drafts)`
        // makes tree_nodes == flat for every legal configuration, so
        // `remaining = tree_nodes - flat` is ALWAYS ZERO and the free-slot
        // branch builder exits before placing its first node.
        //
        // That is why DDTree measured as an exact no-op on Qwen3.8 (2026-08-19:
        // ddtree gamma=15 and flat gamma=15 both reported k=16 and accepted
        // 7.03 to three significant figures, with no error logged), and why no
        // env flag could switch it on — including ATLAS_DDTREE_MAX_NODES=32,
        // which raises `physical_drafts` but not the `requested` cap.
        //
        // Gated: unset/0 reproduces the historical zero-slot behaviour exactly.
        let uncap_tree = std::env::var("ATLAS_DDTREE_UNCAP").ok().as_deref() == Some("1");
        Ok(Self {
            requested,
            flat: requested.min(trained_width).min(physical_drafts),
            tree_nodes: if uncap_tree {
                physical_drafts
            } else {
                requested.min(physical_drafts)
            },
            physical_verify_k,
        })
    }

    pub fn validate_head(gamma: usize, physical_verify_k: usize) -> Result<()> {
        let required = gamma
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("DFlash gamma overflows verify width"))?;
        if required > physical_verify_k {
            anyhow::bail!(
                "DFlash head requires verify K={required}, but physical capacity is K={physical_verify_k}"
            )
        }
        Ok(())
    }

    pub fn validate_flat_width(self, width: usize) -> Result<()> {
        if width == 0 || width > self.flat {
            anyhow::bail!(
                "DFlash flat width {width} is outside resolved budget 1..={}",
                self.flat
            )
        }
        if width
            .checked_add(1)
            .is_none_or(|verify_k| verify_k > self.physical_verify_k)
        {
            anyhow::bail!("DFlash flat width {width} exceeds physical verify capacity")
        }
        Ok(())
    }

    pub fn validate_tree(self, payload: &TreePayload) -> Result<()> {
        let n = payload.tree_token_ids.len();
        if n == 0 || payload.parent_indices.len() != n {
            anyhow::bail!(
                "DFlash tree has mismatched/empty topology: tokens={n}, parents={}",
                payload.parent_indices.len()
            )
        }
        if n > self.tree_nodes {
            anyhow::bail!(
                "DFlash tree has {n} nodes, exceeding resolved budget {}",
                self.tree_nodes
            )
        }
        for (child, &parent) in payload.parent_indices.iter().enumerate() {
            if parent != -1 && (parent < 0 || parent as usize >= child) {
                anyhow::bail!(
                    "DFlash tree parent {parent} is invalid for child {child}; expected -1 or 0..{child}"
                )
            }
        }
        Ok(())
    }
}

pub(super) fn clear_proposal_outputs(state: &mut DflashProposerState) {
    state.last_num_drafted = 0;
    state.pending_tree_payload = None;
    state.async_placeholder = false;
}

fn validate_proposal_exposure(
    budget: DflashDraftBudget,
    drafts: &[u32],
    payload: Option<&TreePayload>,
) -> Result<()> {
    if drafts.len() > budget.tree_nodes {
        anyhow::bail!(
            "DFlash proposal has {} drafts, exceeding resolved budget {}",
            drafts.len(),
            budget.tree_nodes
        )
    }
    if let Some(payload) = payload {
        budget.validate_tree(payload)?;
    }
    Ok(())
}

/// Last boundary before a proposal is exposed to the scheduler.
pub(super) fn finalize_proposal(
    state: &mut DflashProposerState,
    budget: DflashDraftBudget,
    drafts: Vec<u32>,
) -> Vec<u32> {
    if let Err(error) =
        validate_proposal_exposure(budget, &drafts, state.pending_tree_payload.as_ref())
    {
        tracing::warn!("DFlash proposal rejected at outer boundary: {error:#}");
        clear_proposal_outputs(state);
        return Vec::new();
    }
    if drafts.is_empty() {
        state.pending_tree_payload = None;
    }
    state.last_num_drafted = drafts.len();
    drafts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_and_physical_limits_are_absolute() {
        let budget = DflashDraftBudget::new(3, 15, 32).unwrap();
        assert_eq!((budget.flat, budget.tree_nodes), (3, 3));
        let physical = DflashDraftBudget::new(99, 15, 8).unwrap();
        assert_eq!((physical.flat, physical.tree_nodes), (7, 7));
        let zero = DflashDraftBudget::new(0, 15, 16).unwrap();
        assert_eq!((zero.flat, zero.tree_nodes), (0, 0));
    }

    #[test]
    fn construction_rejects_head_wider_than_physical_verify() {
        DflashDraftBudget::validate_head(15, 16).unwrap();
        let error = DflashDraftBudget::validate_head(16, 16).unwrap_err();
        assert!(error.to_string().contains("requires verify K=17"));
    }

    #[test]
    fn topology_requires_backward_or_root_parents() {
        let budget = DflashDraftBudget::new(4, 4, 5).unwrap();
        budget
            .validate_tree(&TreePayload {
                tree_token_ids: vec![10, 11, 12, 13],
                parent_indices: vec![-1, 0, -1, 2],
            })
            .unwrap();
        for parents in [vec![-2], vec![0], vec![-1, 1], vec![-1, 2]] {
            assert!(
                budget
                    .validate_tree(&TreePayload {
                        tree_token_ids: vec![1; parents.len()],
                        parent_indices: parents,
                    })
                    .is_err()
            );
        }
    }

    #[test]
    fn topology_rejects_shape_and_budget_mismatch() {
        let budget = DflashDraftBudget::new(2, 2, 3).unwrap();
        assert!(budget.validate_tree(&TreePayload::default()).is_err());
        assert!(
            budget
                .validate_tree(&TreePayload {
                    tree_token_ids: vec![1, 2],
                    parent_indices: vec![-1],
                })
                .is_err()
        );
        assert!(
            budget
                .validate_tree(&TreePayload {
                    tree_token_ids: vec![1, 2, 3],
                    parent_indices: vec![-1, 0, 1],
                })
                .is_err()
        );
    }

    #[test]
    fn every_flat_route_obeys_the_same_outer_ceiling() {
        let budget = DflashDraftBudget::new(3, 3, 8).unwrap();
        for (route, width) in [
            ("echo", 3),
            ("pld", 3),
            ("retrieval", 3),
            ("recycle", 3),
            ("fallback", 0),
            ("async", 3),
            ("neural", 3),
        ] {
            assert!(
                validate_proposal_exposure(budget, &vec![1; width], None).is_ok(),
                "{route}"
            );
        }
        assert!(validate_proposal_exposure(budget, &[1, 2, 3, 4], None).is_err());
    }

    #[test]
    fn tree_and_spine_have_independent_absolute_ceilings() {
        let budget = DflashDraftBudget::new(6, 4, 7).unwrap();
        let wide_tree = TreePayload {
            tree_token_ids: vec![1, 2, 3, 4, 5, 6],
            parent_indices: vec![-1, 0, 1, -1, 3, 4],
        };
        validate_proposal_exposure(budget, &[10, 11, 12, 13], Some(&wide_tree)).unwrap();
        assert!(
            validate_proposal_exposure(budget, &[10, 11, 12, 13, 14, 15, 16], Some(&wide_tree))
                .is_err()
        );
        let oversized_tree = TreePayload {
            tree_token_ids: vec![1, 2, 3, 4, 5, 6, 7],
            parent_indices: vec![-1, 0, 1, 2, 3, 4, 5],
        };
        assert!(validate_proposal_exposure(budget, &[10; 4], Some(&oversized_tree)).is_err());
    }
}
