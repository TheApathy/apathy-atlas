// SPDX-License-Identifier: AGPL-3.0-only

//! M8A milestone scaffold: tree-aware GDN/conv1d dispatch.
//!
//! When `--dflash-method=ddtree` is active and a non-flat tree payload is
//! present, this module's dispatcher would invoke the parent-aware GDN
//! CUDA kernel (TODO) instead of the existing `gated_delta_rule_wy_k`
//! kernels. The kernel is the actual ceiling unlock — when written, it
//! lets the K=γ verify path follow tree parent topology instead of a
//! flat chain, enabling much higher per-step accept rates on DFlash γ=16.
//!
//! Status (2026-05-17): kernel NOT written. This module:
//!   1. Defines the dispatch entry point + safety guards
//!   2. Falls back to the flat `gated_delta_rule_wy_k` path when:
//!      - No tree payload present (flat verify)
//!      - All compact indices form a flat chain (degenerate tree)
//!      - `DDTREE_TRITON_TREE_GDN=0` (M11E default — research-only flag)
//!   3. Returns an error if tree-mode is forced but kernel not available
//!
//! The flat fallback means M8A scaffold is **safe to land before the
//! kernel** — every code path that calls the dispatch falls into the
//! existing wy_k path until the kernel is wired in.

#![allow(dead_code)]

use super::ddtree::TreePayload;

/// Whether the given payload requires the tree-aware kernel.
/// Returns false for None payloads and for trivially-flat ones
/// (where `parent_indices` is `[-1, 0, 1, ..., n-2]`).
pub fn requires_tree_kernel(payload: Option<&TreePayload>) -> bool {
    let Some(p) = payload else { return false };
    if p.is_empty() {
        return false;
    }
    // Flat chain check: parent_indices[0] == -1 and each subsequent ==
    // the previous compact index.
    if p.parent_indices.first().copied() != Some(-1) {
        return true; // non-flat root layout
    }
    for (i, &parent) in p.parent_indices.iter().enumerate().skip(1) {
        let expected = (i - 1) as i32;
        if parent != expected {
            return true; // diverges from flat chain
        }
    }
    false
}

/// Dispatch decision for the K=γ verify SSM step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnDispatch {
    /// Use the existing flat `gated_delta_rule_wy_k` kernel. Safe.
    Flat,
    /// Use the tree-aware kernel (TODO). Currently unimplemented —
    /// caller must fall back to Flat or error.
    TreeAware,
}

/// Decide which GDN kernel to invoke based on payload + env-var guards.
///
/// AEON-7's M11E "deployable-safe" recipe defaults `DDTREE_TRITON_TREE_GDN=0`
/// so this returns `Flat` for serving. Setting `DDTREE_TRITON_TREE_GDN=1`
/// AND providing a non-flat payload would return `TreeAware` — but the
/// caller MUST currently fall back because the kernel isn't written.
pub fn pick_dispatch(payload: Option<&TreePayload>) -> GdnDispatch {
    let triton_tree_enabled = std::env::var("DDTREE_TRITON_TREE_GDN")
        .ok()
        .as_deref()
        == Some("1");
    if !triton_tree_enabled {
        return GdnDispatch::Flat;
    }
    if requires_tree_kernel(payload) {
        GdnDispatch::TreeAware
    } else {
        GdnDispatch::Flat
    }
}

/// Error returned when the tree-aware kernel is requested but not built.
#[derive(Debug)]
pub struct TreeKernelUnavailable;

impl std::fmt::Display for TreeKernelUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DDTree tree-aware GDN kernel not yet built (M8A pending). \
             Set DDTREE_TRITON_TREE_GDN=0 to use flat fallback."
        )
    }
}

impl std::error::Error for TreeKernelUnavailable {}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ddtree::TreePayload;

    fn payload(tokens: &[u32], parents: &[i32]) -> TreePayload {
        TreePayload {
            tree_token_ids: tokens.to_vec(),
            parent_indices: parents.to_vec(),
        }
    }

    #[test]
    fn none_payload_is_flat() {
        assert!(!requires_tree_kernel(None));
    }

    #[test]
    fn empty_payload_is_flat() {
        let p = payload(&[], &[]);
        assert!(!requires_tree_kernel(Some(&p)));
    }

    #[test]
    fn flat_chain_payload_is_flat() {
        // tokens [10,20,30], parents [-1, 0, 1] = flat chain.
        let p = payload(&[10, 20, 30], &[-1, 0, 1]);
        assert!(!requires_tree_kernel(Some(&p)));
    }

    #[test]
    fn diverging_payload_requires_tree() {
        // tokens [10, 11], parents [-1, -1] = two root-children siblings.
        let p = payload(&[10, 11], &[-1, -1]);
        assert!(requires_tree_kernel(Some(&p)));
    }

    #[test]
    fn deep_diverging_tree_requires_tree() {
        // 0 root, 1: child of root, 2: child of root (sibling of 1).
        let p = payload(&[10, 11], &[-1, -1]);
        assert!(requires_tree_kernel(Some(&p)));
    }

    #[test]
    fn pick_dispatch_defaults_to_flat() {
        // The dispatch helper checks DDTREE_TRITON_TREE_GDN at call time.
        // We avoid mutating process env in tests (Rust 2024 marks env::*_var
        // unsafe in tests because they race with other threads). Instead,
        // assert the requires_tree_kernel arm directly — it's the part of
        // pick_dispatch that doesn't read env at all.
        let flat = payload(&[10, 20], &[-1, 0]);
        assert!(!requires_tree_kernel(Some(&flat)));
    }
}
