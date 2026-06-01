// SPDX-License-Identifier: AGPL-3.0-only

//! M5A milestone: tree-aware GDN/conv reference math — Atlas contract.
//!
//! The actual reference implementation lives in Python at
//! `research/ddtree_port/ddtree_src/prototypes/ddtree_gdn_reference.py`
//! and is the **bit-equality oracle** for the M8A CUDA kernel.
//!
//! This module documents the I/O contract so that:
//!   1. M8A's CUDA kernel knows the exact tensor shapes + semantics
//!   2. Atlas's integration tests can dump (q, k, v, gate, beta, parent_ids,
//!      initial_state) → run the Python reference → diff Atlas kernel output
//!
//! Rationale for not duplicating the math in Rust:
//!   - Reference does state-shape [HV=32, V=128, K=128] = 524 K float ops/token
//!   - All M8A consumers are CUDA kernels, not CPU code — the Python ref is
//!     the right place for a once-per-test oracle, not pure-Rust math
//!   - AEON-7's own validation pipeline uses Python ref + diff
//!
//! When M8A lands, write a Rust integration test that:
//!   1. Captures Atlas (q,k,v,gate,beta,parent_ids,initial_state) at one
//!      decode step (e.g. via env-var-gated dump)
//!   2. Invokes `python3 research/ddtree_port/ddtree_src/prototypes/ddtree_gdn_reference.py`
//!      with the dump as input
//!   3. Bit-diffs reference output against Atlas's M8A kernel output

#![allow(dead_code)]

/// Tree-aware GDN tensor I/O contract for one decode step.
///
/// All shapes follow vLLM's Triton GDN kernel convention so the Python
/// reference and Atlas CUDA kernel produce comparable outputs.
#[derive(Debug, Clone)]
pub struct TreeGdnInputs {
    /// `q`: [T, H, K_dim] — query per tree token.
    pub q: Vec<f32>,
    /// `k`: [T, H, K_dim] — key per tree token.
    pub k: Vec<f32>,
    /// `v`: [T, HV, V_dim] — value per tree token.
    pub v: Vec<f32>,
    /// `gate`: [T, HV] — per-token decay (raw, will be exp'd inside kernel).
    pub gate: Vec<f32>,
    /// `beta`: [T, HV] — per-token correction strength (sigmoid'd in kernel).
    pub beta: Vec<f32>,
    /// `parent_ids`: [T] — tree parent index per token. `-1` = root
    /// (read pre-tree `initial_state`); else `parent_ids[i] < i` and points
    /// at the row whose state should be loaded as this token's input state.
    pub parent_ids: Vec<i32>,
    /// `initial_state`: [HV, V_dim, K_dim] — pre-tree recurrent state.
    pub initial_state: Vec<f32>,

    pub num_tree_tokens: usize,
    pub num_heads: usize,
    pub num_v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
}

/// Tree-aware GDN expected output.
#[derive(Debug, Clone)]
pub struct TreeGdnOutputs {
    /// `output`: [T, HV, V_dim] — per-token GDN output.
    pub output: Vec<f32>,
    /// `final_states`: [T, HV, V_dim, K_dim] — state after each tree token
    /// (so descendants can reload their parent's state by index).
    pub final_states: Vec<f32>,
}

/// Tree-aware causal conv1d contract (depthwise, kernel size K).
///
/// Reference: `tree_conv1d_reference` in the Python prototype.
#[derive(Debug, Clone)]
pub struct TreeConv1dInputs {
    /// `x`: [T, D] — current tree-token projection.
    pub x: Vec<f32>,
    /// `weight`: [D, K] — depthwise conv kernel.
    pub weight: Vec<f32>,
    /// `bias`: Option<[D]>.
    pub bias: Option<Vec<f32>>,
    /// `parent_ids`: [T] — same as GDN contract.
    pub parent_ids: Vec<i32>,
    /// `initial_state`: [K-1, D] — pre-tree conv history, oldest → newest.
    pub initial_state: Vec<f32>,
    /// Whether to apply SiLU after conv+bias.
    pub apply_silu: bool,

    pub num_tree_tokens: usize,
    pub d: usize,
    pub kernel_size: usize,
}

/// Path-length information needed by both conv1d and GDN kernels.
///
/// For each tree token, we need to know its path-length-from-root so the
/// conv1d kernel knows how many parents to walk before falling off into
/// the pre-tree `initial_state` slots.
pub fn path_lengths(parent_ids: &[i32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(parent_ids.len());
    for (i, &p) in parent_ids.iter().enumerate() {
        if p < 0 {
            out.push(1);
        } else {
            let pi = p as usize;
            debug_assert!(pi < i, "parent_ids[{i}] = {pi} must be < {i}");
            out.push(out[pi] + 1);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_lengths_chain_is_monotonic() {
        // Linear chain: 0 (root) → 1 → 2 → 3.
        let parents = vec![-1, 0, 1, 2];
        let lens = path_lengths(&parents);
        assert_eq!(lens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn path_lengths_diverging_tree() {
        // Two siblings then one grandchild:
        // 0 → 1, 0 → 2, 1 → 3.
        let parents = vec![-1, -1, 0];
        let lens = path_lengths(&parents);
        assert_eq!(lens, vec![1, 1, 2]);
    }

    #[test]
    fn path_lengths_root_only() {
        let lens = path_lengths(&[-1]);
        assert_eq!(lens, vec![1]);
    }
}
