// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime binding for the REST retrieval draft store.
//!
//! The store itself (format, suffix array, trie) lives in the `rest-store`
//! crate so the offline builder can share it. This module owns the
//! *server-side* concerns: env configuration, the process-wide singleton,
//! and the engage gate that decides whether a lookup is worth a draft.
//!
//! Phase 2 wires this into the scheduler as *conditional pre-emption*: a
//! retrieved chain replaces the DFlash proposal for one step when — and
//! only when — it clears the gates in [`engage`]. It never replaces the
//! DFlash proposer, and it never touches verification, so output at
//! temperature 0 is bit-identical to a run with `ATLAS_REST_STORE` unset.
//! See `rest_store/PHASE2.md` for the design and the measurements behind
//! it, and [`engage`] for the gate.
//!
//! # Environment
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `ATLAS_REST_STORE` | unset | Path to the store; unset disables REST drafting entirely |
//! | `ATLAS_REST_MIN_MATCH` | 10 | Engage gate — shorter matches propose nothing |
//! | `ATLAS_REST_MAX_DRAFTER_ACCEPT` | 13 | Skip pre-emption when the drafter's last frame accepted this many or more |
//! | `ATLAS_REST_MAX_K` | 16 | Longest context suffix considered |
//! | `ATLAS_REST_MAX_OCC` | 64 | Cap on suffix-array occurrences scanned per lookup |
//! | `ATLAS_REST_DEPTH` | 16 | Continuation depth |
//! | `ATLAS_REST_MAX_NODES` | 16 | Node budget for the proposed tree |

use std::sync::OnceLock;

use anyhow::Result;
use rest_store::{DraftTree, RestStore};

pub mod engage;
pub mod self_context;
pub use engage::{MIN_PREEMPT_WIDTH, max_drafter_accept, preempt, record_accepted};

/// Scheduler-side engage gate, stricter than [`rest_store::DEFAULT_MIN_MATCH`].
///
/// The library default (8) is the point where a retrieved continuation
/// beats nothing. Pre-empting DFlash is a higher bar: the eval's
/// `min_match >= 10` rows are where wasted engagements fall far enough
/// for the chain to pay for the drafter step it displaces. `PHASE2.md` §4.
pub const SCHEDULER_MIN_MATCH: usize = 10;

/// Resolved REST drafting configuration.
#[derive(Debug, Clone, Copy)]
pub struct RestConfig {
    /// Minimum suffix-match length before a draft is proposed.
    pub min_match: usize,
    /// Maximum context suffix length searched.
    pub max_k: usize,
    /// Cap on occurrences sampled from the suffix-array range.
    pub max_occurrences: usize,
    /// Continuation depth.
    pub depth: usize,
    /// Node budget for the emitted tree.
    pub max_nodes: usize,
}

impl Default for RestConfig {
    fn default() -> Self {
        Self {
            min_match: SCHEDULER_MIN_MATCH,
            max_k: rest_store::DEFAULT_MAX_K,
            max_occurrences: rest_store::DEFAULT_MAX_OCCURRENCES,
            depth: rest_store::DEFAULT_DEPTH,
            max_nodes: rest_store::DEFAULT_MAX_NODES,
        }
    }
}

impl RestConfig {
    /// Read the configuration from the environment, falling back to
    /// [`RestConfig::default`] for anything unset or unparseable.
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            min_match: env_usize("ATLAS_REST_MIN_MATCH", d.min_match),
            max_k: env_usize("ATLAS_REST_MAX_K", d.max_k),
            max_occurrences: env_usize("ATLAS_REST_MAX_OCC", d.max_occurrences),
            depth: env_usize("ATLAS_REST_DEPTH", d.depth),
            max_nodes: env_usize("ATLAS_REST_MAX_NODES", d.max_nodes),
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(key, value = %v, default, "unparseable REST env value; using default");
                default
            }
        },
        Err(_) => default,
    }
}

/// Cached configuration. Read once — this sits on the decode hot path and
/// re-reading the environment per step would show up in step time.
pub fn config() -> &'static RestConfig {
    static CFG: OnceLock<RestConfig> = OnceLock::new();
    CFG.get_or_init(RestConfig::from_env)
}

/// The process-wide store, mapped on first use.
///
/// `None` means REST drafting is off, either because `ATLAS_REST_STORE` is
/// unset or because opening the store failed. A failed open is logged once
/// and then behaves exactly like "off": a bad store must never take the
/// server down, because the drafter is a pure optimization.
static STORE: OnceLock<Option<RestStore>> = OnceLock::new();

/// Initialize the store explicitly, validating it against the tokenizer
/// the server actually loaded.
///
/// Call this once at startup, after the tokenizer is available. Returns
/// `Ok(false)` when REST drafting is disabled by configuration.
pub fn init(tokenizer_json: &[u8]) -> Result<bool> {
    let fp = rest_store::tokenizer_fingerprint(tokenizer_json);
    let Some(path) = std::env::var_os("ATLAS_REST_STORE") else {
        let _ = STORE.set(None);
        return Ok(false);
    };
    match RestStore::open(&path, Some(fp)) {
        Ok(store) => {
            let cfg = config();
            tracing::info!(
                min_match = cfg.min_match,
                max_k = cfg.max_k,
                max_nodes = cfg.max_nodes,
                depth = cfg.depth,
                "REST draft store enabled"
            );
            let _ = STORE.set(Some(store));
            Ok(true)
        }
        Err(e) => {
            let _ = STORE.set(None);
            Err(e)
        }
    }
}

/// The mapped store, if REST drafting is enabled and initialized.
///
/// Falls back to an unvalidated open when [`init`] was never called, so
/// offline tools and tests can use the module without a tokenizer. The
/// server path always goes through [`init`].
pub fn store() -> Option<&'static RestStore> {
    STORE
        .get_or_init(|| {
            let path = std::env::var_os("ATLAS_REST_STORE")?;
            match RestStore::open(&path, None) {
                Ok(s) => Some(s),
                Err(e) => {
                    tracing::warn!(error = %e, "REST store open failed; retrieval drafting disabled");
                    None
                }
            }
        })
        .as_ref()
}

/// Whether REST drafting is available.
pub fn enabled() -> bool {
    store().is_some()
}

/// Propose a draft tree for `ctx`, or `None` when the store declines.
///
/// Declines when REST is disabled, when no suffix of the last `max_k`
/// context tokens occurs in the corpus, when the longest match is shorter
/// than the engage gate, or when no occurrence had a continuation.
///
/// The engage gate is the whole economics of this drafter: a match shorter
/// than `min_match` is common enough to be meaningless, and a draft built
/// from it burns verify slots that the neural drafter would have used
/// better.
pub fn propose(ctx: &[u32]) -> Option<DraftTree> {
    let cfg = config();
    let tree = store()?.propose(
        ctx,
        cfg.max_k,
        cfg.min_match,
        cfg.max_occurrences,
        cfg.depth,
        cfg.max_nodes,
    )?;
    debug_assert!(
        validate_tree_shape(&tree),
        "REST tree violates the DDTree parent ordering invariant"
    );
    Some(tree)
}

/// Check the invariant the verify path enforces: every parent index is
/// either `-1` (root) or strictly less than its child's index.
///
/// Mirrors `DflashDraftBudget::validate_tree`. Kept here so a malformed
/// tree is caught at the producer rather than rejected at the outer DFlash
/// boundary, where the only recovery is to drop the whole proposal.
pub fn validate_tree_shape(tree: &DraftTree) -> bool {
    !tree.is_empty()
        && tree
            .nodes
            .iter()
            .enumerate()
            .all(|(i, n)| n.parent == -1 || (n.parent >= 0 && (n.parent as usize) < i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rest_store::trie::{DraftNode, DraftTree};

    fn node(token: u32, parent: i32, depth: u16) -> DraftNode {
        DraftNode {
            token,
            parent,
            count: 1,
            depth,
        }
    }

    #[test]
    fn default_config_matches_the_documented_defaults() {
        let d = RestConfig::default();
        assert_eq!(d.min_match, SCHEDULER_MIN_MATCH);
        assert_eq!(d.max_k, 16);
        assert_eq!(d.max_nodes, 16);
        assert_eq!(d.depth, 16);
        assert_eq!(d.max_occurrences, 64);
    }

    #[test]
    fn tree_shape_validation_accepts_a_well_formed_tree() {
        let tree = DraftTree {
            nodes: vec![node(1, -1, 1), node(2, 0, 2), node(3, -1, 1), node(4, 1, 3)],
            match_len: 8,
            occurrences: 3,
        };
        assert!(validate_tree_shape(&tree));
    }

    #[test]
    fn tree_shape_validation_rejects_forward_and_self_parents() {
        let forward = DraftTree {
            nodes: vec![node(1, 1, 1), node(2, -1, 1)],
            match_len: 8,
            occurrences: 1,
        };
        assert!(
            !validate_tree_shape(&forward),
            "forward parent must be rejected"
        );

        let self_parent = DraftTree {
            nodes: vec![node(1, -1, 1), node(2, 1, 2)],
            match_len: 8,
            occurrences: 1,
        };
        assert!(
            !validate_tree_shape(&self_parent),
            "self parent must be rejected"
        );

        assert!(
            !validate_tree_shape(&DraftTree::default()),
            "empty tree is not proposable"
        );
    }

    #[test]
    fn disabled_without_a_store_path() {
        // `store()` is a process-wide OnceLock, so this only asserts the
        // no-path branch — which is the one that must never panic.
        if std::env::var_os("ATLAS_REST_STORE").is_none() {
            assert!(!enabled());
            assert!(propose(&[1, 2, 3, 4, 5, 6, 7, 8]).is_none());
        }
    }
}
