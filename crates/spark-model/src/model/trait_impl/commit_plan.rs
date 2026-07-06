// SPDX-License-Identifier: AGPL-3.0-only

//! Pure, CPU-testable planning logic for the K=γ verify → SSM-commit
//! contract (task #34: K≠17 losslessness).
//!
//! ## Why this module exists
//!
//! The K=γ DFlash verify involves THREE components that must agree on which
//! buffer holds the canonical post-accept SSM state:
//!
//!   1. The GDN kernel dispatch
//!      (`layers/qwen3_ssm/trait_decode_batched_conv_gdn.rs`) — decides which
//!      kernel family runs and therefore WHICH buffers get written:
//!        * `gated_delta_rule_wy17` (K=17 only): writes `h_state` (post-K) +
//!          intermediates `0..=K-2`.
//!        * chunked wy4/wy3/wy2 (K∈{5..16, 18..32}): writes `h_state` +
//!          intermediates `0..=K-2` (kernel slots + chunk-end D2D copies).
//!        * fused wy2/wy3/wy4 (K∈{2,3,4}): writes `h_state` + `0..=K-2`.
//!        * `gated_delta_rule_tree_wy` (tree payloads AND the graph-safe
//!          flat-chain injection): writes intermediates `0..=K-1` and leaves
//!          `h_state` UNTOUCHED (declared read-only,
//!          `kernels/gb10/common/gated_delta_rule_tree_wy.cu:50`).
//!   2. The verify entry (`trait_impl/verify_d.rs`) — under CUDA graphs, when
//!      no tree payload is stashed and `k == ddtree_parent_ids_capacity`, it
//!      SUBSTITUTES the persistent linear-chain parent_ids so the captured
//!      graph always references `gated_delta_rule_tree_wy` (pointer-stable).
//!   3. The commit (`trait_impl/async_chkpt.rs`) — picks the canonical-state
//!      source: live `h_state` (full-accept fast path), an intermediate slot,
//!      or the wy17-lazy replay kernel.
//!
//! ## The K=12 bug (task #34)
//!
//! With `--dflash-gamma 11` (K = capacity = 12, `serve-aeon-27b-dspark.sh`)
//! and `ATLAS_DISABLE_TREE_WY` unset, the flat-chain injection (2) reroutes
//! every graphed verify to `gated_delta_rule_tree_wy` — but it only set the
//! LOCAL `ForwardContext.ddtree_parent_ids_dev`, never the model-level stash
//! that the commit (3) reads as `was_tree_mode`. On a FULL accept
//! (`num_accepted == k`) the commit then took the fast path and committed the
//! live `h_state` — which the tree kernel never wrote — i.e. the STALE
//! pre-verify root state. Partial accepts read intermediates (which the tree
//! kernel does write) and were correct. Hence: low acceptance fine, high
//! acceptance (full accepts) corrupt. K=17 production is unaffected only
//! because `serve-aeon-27b-dflash.sh` exports `ATLAS_DISABLE_TREE_WY=1`,
//! which suppresses the injection entirely.
//!
//! A sibling hazard exists in the WY17 lazy-commit replay gate: the commit
//! previously decided to run `gated_delta_rule_wy17_replay` from env gates
//! alone (`ATLAS_WY17_LAZY>1` + `ATLAS_WY17_LAZY_COMMIT=1`), without checking
//! that the lazy wy17 kernel ACTUALLY ran this verify. At K≠17 (chunked
//! path) or with `ATLAS_WY17_SPLIT` active, the retention buffers the replay
//! reads are never populated → replay reconstructs garbage.
//!
//! The production call sites (`verify_d.rs`, `async_chkpt.rs`,
//! `trait_decode_batched_conv_gdn.rs`) route their decisions through the pure
//! functions below so the cross-component contract is unit-testable on CPU.

#![allow(dead_code)]

/// Which GDN kernel family the K-token verify dispatch runs.
///
/// Mirrors the dispatch precedence in
/// `layers/qwen3_ssm/trait_decode_batched_conv_gdn.rs::decode_batched_conv_gdn`
/// (K∈{2,3,4} fused branches → tree branch → K==17 wy17 branch → chunked).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GdnVerifyRoute {
    /// K∈{2,3,4}: fused wy2/wy3/wy4 single-chunk kernels.
    FusedSmallK,
    /// `gated_delta_rule_tree_wy` (or tree v1): tree payloads and the
    /// graph-safe flat-chain injection.
    TreeWy,
    /// `gated_delta_rule_wy17` (incl. lazy / vsplit variants). K==17 only.
    Wy17,
    /// Chunked wy4/wy3/wy2 composition, K∈{5..16, 18..32}.
    Chunked,
}

/// Route selection. Mirrors `decode_batched_conv_gdn` dispatch order exactly:
/// the fused K∈{2,3,4} branches run BEFORE the tree branch, which runs BEFORE
/// the `num_tokens == 17` wy17 branch; everything else is chunked.
pub(crate) fn gdn_route(
    k: usize,
    parent_ids_in_ctx: bool,
    tree_kernel_loaded: bool,
    force_wy17_env: bool,
    wy17_kernel_loaded: bool,
) -> GdnVerifyRoute {
    if (2..=4).contains(&k) {
        return GdnVerifyRoute::FusedSmallK;
    }
    if parent_ids_in_ctx && tree_kernel_loaded && !force_wy17_env {
        return GdnVerifyRoute::TreeWy;
    }
    if k == 17 && wy17_kernel_loaded {
        return GdnVerifyRoute::Wy17;
    }
    GdnVerifyRoute::Chunked
}

/// TRUE iff the route's kernel writes the live `h_state` to the post-K state.
///
/// `gated_delta_rule_tree_wy` / `gated_delta_rule_tree` treat `h_state` as
/// READ-ONLY (root state) and write per-token states ONLY into the
/// intermediate pool — there is no single "final" state for a tree.
pub(crate) fn route_writes_live_h_state(route: GdnVerifyRoute) -> bool {
    route != GdnVerifyRoute::TreeWy
}

/// Highest intermediate pool slot (inclusive) holding a REAL state after the
/// verify, per route. Non-tree routes leave the post-(K-1) state in `h_state`
/// only, so their highest written slot is K-2. The tree kernels write every
/// slot 0..=K-1.
pub(crate) fn route_max_written_inter_slot(route: GdnVerifyRoute, k: usize) -> Option<usize> {
    match route {
        GdnVerifyRoute::TreeWy => k.checked_sub(1),
        _ => k.checked_sub(2),
    }
}

/// The graph-safe flat-chain tree_wy injection predicate
/// (`verify_d.rs`, "CUDA-graph-safe path" block).
///
/// CONTRACT: when this returns true, the verify reroutes the SSM dispatch to
/// `gated_delta_rule_tree_wy`, which leaves `h_state` stale. The caller MUST
/// therefore also mark tree mode for the commit (see
/// [`commit_sees_tree_mode`]) — committing the live `h_state` after an
/// injected verify commits the PRE-VERIFY root state (task #34).
pub(crate) fn flat_tree_wy_injection_applies(
    disable_tree_wy_env: bool,
    use_graphs: bool,
    scheduler_stash_set: bool,
    parent_ids_capacity: usize,
    k: usize,
) -> bool {
    !disable_tree_wy_env
        && use_graphs
        && !scheduler_stash_set
        && parent_ids_capacity > 0
        && k == parent_ids_capacity
}

/// The commit-visible "tree mode" flag: TRUE iff the verify that just ran
/// left `h_state` untouched (tree_wy route), so the commit must source the
/// canonical state from the intermediate pool even on a full accept.
///
/// `scheduler_stash_set`: `model.ddtree_parent_ids_dev` was set by
/// `set_ddtree_parent_ids` (a real tree payload).
/// `injected_flat_tree_route`: the verify's own graph-safe injection fired
/// (see [`flat_tree_wy_injection_applies`]).
pub(crate) fn commit_sees_tree_mode(
    scheduler_stash_set: bool,
    injected_flat_tree_route: bool,
) -> bool {
    // FIX (task #34): the flat-chain injection ALSO runs the tree_wy kernel
    // (h_state left stale), so the commit must treat it as tree mode. The
    // pre-fix behavior consulted only `scheduler_stash_set`, so an injected
    // full accept took the LiveHState fast path and committed the pre-verify
    // root state.
    scheduler_stash_set || injected_flat_tree_route
}

/// WY17 lazy-commit replay gate (`async_chkpt.rs`).
///
/// The replay kernel reconstructs a SKIPPED intermediate slot from the
/// pre-verify root + the per-layer k/v/gate/beta retention buffers. Those
/// retention buffers are populated ONLY by the `gdn_decode_wy17_lazy` branch
/// of the dispatch (K==17, lazy gates on, vsplit off) — `wy17_lazy_armed`
/// carries that fact from the dispatch to the commit. Without it, a K≠17
/// (chunked) verify under global lazy env gates would replay from stale /
/// zero retention data.
pub(crate) fn wy17_replay_allowed(
    lazy_commit_env: bool,
    lazy_j: u32,
    tree_mode: bool,
    num_accepted: usize,
    k: usize,
    last_inter_slot: usize,
    wy17_lazy_armed: bool,
) -> bool {
    // FIX (task #34 sibling hazard): `wy17_lazy_armed` is required — env
    // gates alone previously decided, so a K≠17 (chunked / fused-small-K)
    // verify under global lazy env gates replayed from unpopulated retention.
    wy17_lazy_armed
        && lazy_commit_env
        && lazy_j > 1
        && !tree_mode
        && num_accepted != k
        && !crate::layers::wy17_is_checkpoint(last_inter_slot, lazy_j, k)
}

/// Where the commit sources the canonical post-accept H state from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HCommitSource {
    /// Full reject: canonical state untouched (rollback to checkpoint).
    Untouched,
    /// Full-accept fast path: live `h_state` already holds post-K state.
    LiveHState,
    /// D2D copy `h_intermediate[slot]` → `h_state`.
    InterSlot(usize),
    /// `gated_delta_rule_wy17_replay` from the pre-verify root, over the
    /// retained inputs, up to `slot` (inclusive).
    LazyReplayFromRoot(usize),
}

/// Commit H-source selection. Mirrors
/// `async_chkpt.rs::commit_verify_state_async_dispatch`.
pub(crate) fn plan_h_commit_source(
    num_accepted: usize,
    k: usize,
    tree_mode: bool,
    last_inter_slot: usize,
    replay_allowed: bool,
) -> HCommitSource {
    if num_accepted == 0 {
        return HCommitSource::Untouched;
    }
    if num_accepted == k && !tree_mode {
        return HCommitSource::LiveHState;
    }
    if replay_allowed {
        return HCommitSource::LazyReplayFromRoot(last_inter_slot);
    }
    HCommitSource::InterSlot(last_inter_slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validity oracle: does `plan` read a buffer that the route's kernel
    /// actually wrote with the state-after-token(`num_accepted`-1)?
    ///
    /// `lazy_armed_j`: Some(J) iff the wy17 LAZY kernel ran (K==17 only) —
    /// then non-checkpoint intermediate slots are STALE and only checkpoints
    /// or the replay kernel are valid partial-accept sources.
    fn commit_source_is_valid(
        plan: HCommitSource,
        route: GdnVerifyRoute,
        num_accepted: usize,
        k: usize,
        lazy_armed_j: Option<u32>,
    ) -> bool {
        match plan {
            HCommitSource::Untouched => num_accepted == 0,
            HCommitSource::LiveHState => {
                // Live h_state holds post-K state only if the kernel wrote it
                // AND every token was accepted.
                route_writes_live_h_state(route) && num_accepted == k
            }
            HCommitSource::InterSlot(s) => {
                let in_range = match route_max_written_inter_slot(route, k) {
                    Some(max) => s <= max,
                    None => false,
                };
                let is_last_accepted_state = s == num_accepted - 1;
                // Under an armed lazy wy17 verify, skipped slots are stale.
                let persisted = match lazy_armed_j {
                    Some(j) => crate::layers::wy17_is_checkpoint(s, j, k),
                    None => true,
                };
                in_range && is_last_accepted_state && persisted
            }
            HCommitSource::LazyReplayFromRoot(s) => {
                // Replay is only bit-exact when the retention buffers were
                // populated this step — i.e. the lazy wy17 kernel ran.
                lazy_armed_j.is_some()
                    && route == GdnVerifyRoute::Wy17
                    && s == num_accepted - 1
                    && s < k
            }
        }
    }

    /// Compose the three components for a FLAT-CHAIN (no scheduler tree
    /// payload) verify+commit exactly as the production wiring does, and
    /// return (route, plan).
    #[allow(clippy::too_many_arguments)]
    fn flat_chain_verify_commit(
        k: usize,
        capacity: usize,
        disable_tree_wy_env: bool,
        use_graphs: bool,
        num_accepted_total: usize,
        lazy_commit_env: bool,
        lazy_j: u32,
        wy17_lazy_armed: bool,
    ) -> (GdnVerifyRoute, HCommitSource) {
        // verify_d.rs: graph-safe injection decision.
        let injected = flat_tree_wy_injection_applies(
            disable_tree_wy_env,
            use_graphs,
            false, // no scheduler tree payload (flat chain)
            capacity,
            k,
        );
        // qwen3_ssm dispatch: which kernel family runs. Tree + wy17 kernels
        // are loaded on the production target; FORCE_WY17 unset.
        let route = gdn_route(k, injected, true, false, true);
        // async_chkpt.rs: commit decisions.
        let tree_mode = commit_sees_tree_mode(false, injected);
        let last_inter_slot = num_accepted_total.saturating_sub(1);
        let replay = wy17_replay_allowed(
            lazy_commit_env,
            lazy_j,
            tree_mode,
            num_accepted_total,
            k,
            last_inter_slot,
            wy17_lazy_armed,
        );
        let plan = plan_h_commit_source(num_accepted_total, k, tree_mode, last_inter_slot, replay);
        (route, plan)
    }

    // ── task #34: K=12 (γ=11, DSpark A/B config) ────────────────────────

    /// serve-aeon-27b-dspark.sh config: --dflash-gamma 11 → capacity=12,
    /// ATLAS_DISABLE_TREE_WY unset, CUDA graphs on (post-FP8-calibration).
    /// FULL accept (11/11 drafts + bonus): the commit must NOT read the live
    /// h_state, because the injected tree_wy route never writes it.
    #[test]
    fn k12_injected_flat_full_accept_commit_reads_kernel_written_state() {
        let k = 12;
        let (route, plan) = flat_chain_verify_commit(k, 12, false, true, k, false, 1, false);
        // The injection reroutes to tree_wy — that part is by design.
        assert_eq!(
            route,
            GdnVerifyRoute::TreeWy,
            "injection must fire at k==capacity"
        );
        // THE BUG: pre-fix, plan == LiveHState — the stale pre-verify root.
        assert!(
            commit_source_is_valid(plan, route, k, k, None),
            "K=12 full-accept commit reads a buffer the tree_wy kernel never \
             wrote (plan={plan:?}); this commits the STALE pre-verify h_state \
             → non-lossless at high acceptance (task #34)"
        );
        assert_eq!(
            plan,
            HCommitSource::InterSlot(k - 1),
            "full accept on the tree_wy route must commit inter[K-1]"
        );
    }

    /// Same failure shape at K=10 (γ=9) — the bug is any k == capacity with
    /// graphs on and ATLAS_DISABLE_TREE_WY unset, not K=12-specific.
    #[test]
    fn k10_injected_flat_full_accept_commit_reads_kernel_written_state() {
        let k = 10;
        let (route, plan) = flat_chain_verify_commit(k, 10, false, true, k, false, 1, false);
        assert_eq!(route, GdnVerifyRoute::TreeWy);
        assert!(
            commit_source_is_valid(plan, route, k, k, None),
            "K=10 full-accept commit invalid (plan={plan:?})"
        );
    }

    /// Partial accepts on the injected route were always correct (tree_wy
    /// writes every intermediate slot) — this is why LOW acceptance looked
    /// fine in the K=12 A/B while high acceptance corrupted.
    #[test]
    fn k12_injected_flat_partial_accepts_were_always_valid() {
        let k = 12;
        for total in 1..k {
            let (route, plan) =
                flat_chain_verify_commit(k, 12, false, true, total, false, 1, false);
            assert_eq!(route, GdnVerifyRoute::TreeWy);
            assert!(
                commit_source_is_valid(plan, route, total, k, None),
                "partial accept total={total} plan={plan:?}"
            );
        }
    }

    // ── K=17 production invariance (the md5 constitution) ──────────────

    /// Production env (serve-aeon-27b-dflash.sh): ATLAS_DISABLE_TREE_WY=1.
    /// The injection never fires, the wy17 route runs, and a full accept
    /// takes the LiveHState fast path — this must NOT change.
    #[test]
    fn k17_production_env_plans_are_unchanged() {
        let k = 17;
        // Full accept → fast path.
        let (route, plan) = flat_chain_verify_commit(k, 17, true, true, k, false, 1, false);
        assert_eq!(route, GdnVerifyRoute::Wy17);
        assert_eq!(plan, HCommitSource::LiveHState);
        // Every partial accept → inter[total-1].
        for total in 1..k {
            let (route, plan) = flat_chain_verify_commit(k, 17, true, true, total, false, 1, false);
            assert_eq!(route, GdnVerifyRoute::Wy17);
            assert_eq!(plan, HCommitSource::InterSlot(total - 1));
            assert!(commit_source_is_valid(plan, route, total, k, None));
        }
        // Eager mode (graphs suppressed): identical plans.
        let (route, plan) = flat_chain_verify_commit(k, 17, true, false, k, false, 1, false);
        assert_eq!(route, GdnVerifyRoute::Wy17);
        assert_eq!(plan, HCommitSource::LiveHState);
    }

    /// K=17 with ATLAS_WY17_LAZY=8 + ATLAS_WY17_LAZY_COMMIT=1 (staged win):
    /// the lazy wy17 kernel ran (armed), so a partial accept on a skipped
    /// slot must replay and a checkpoint slot must D2D — unchanged semantics.
    #[test]
    fn k17_lazy_armed_partial_accept_replays_skipped_slots() {
        let k = 17;
        let j = 8u32;
        for total in 1..k {
            let slot = total - 1;
            let (route, plan) = flat_chain_verify_commit(k, 17, true, true, total, true, j, true);
            assert_eq!(route, GdnVerifyRoute::Wy17);
            if crate::layers::wy17_is_checkpoint(slot, j, k) {
                assert_eq!(
                    plan,
                    HCommitSource::InterSlot(slot),
                    "checkpoint slot {slot}"
                );
            } else {
                assert_eq!(
                    plan,
                    HCommitSource::LazyReplayFromRoot(slot),
                    "skipped slot {slot} must replay"
                );
            }
            assert!(commit_source_is_valid(plan, route, total, k, Some(j)));
        }
        // Full accept under lazy: fast path, no replay.
        let (_, plan) = flat_chain_verify_commit(k, 17, true, true, k, true, j, true);
        assert_eq!(plan, HCommitSource::LiveHState);
    }

    // ── task #34 sibling hazard: lazy replay must not fire off-route ────

    /// K=12 chunked verify (ATLAS_DISABLE_TREE_WY=1) with the lazy env gates
    /// globally on: the chunked path wrote ALL intermediate slots and never
    /// populated the retention buffers — the commit must use the D2D path,
    /// NOT the replay kernel (which would read stale retention data).
    #[test]
    fn k12_chunked_with_lazy_env_must_not_replay() {
        let k = 12;
        let j = 8u32;
        for total in 1..k {
            let (route, plan) = flat_chain_verify_commit(
                k,
                17, // capacity 17 (γ=16 server) or DISABLE_TREE_WY=1 — chunked either way
                true, true, total, true, j, /* armed = */ false,
            );
            assert_eq!(route, GdnVerifyRoute::Chunked);
            assert!(
                commit_source_is_valid(plan, route, total, k, None),
                "K=12 chunked, lazy env on, total={total}: plan={plan:?} reads \
                 retention buffers that were never populated this step"
            );
            assert_eq!(plan, HCommitSource::InterSlot(total - 1));
        }
    }

    // ── FREE SLOTS K=32: capacity=32 flat-vs-tree commit-source selection ──
    //
    // The free-slots lever sets ATLAS_DDTREE_MAX_NODES=32 so `capacity == 32`.
    // Two distinct verifies then coexist on the SAME model and MUST pick the
    // right h_state source:
    //   (a) a normal FLAT verify at the drafter width (k=17) — the injection
    //       predicate `k == capacity` is 17 != 32 → does NOT fire → wy17/chunked
    //       writes h_state live → a full accept takes the LiveHState fast path.
    //   (b) a real DEEP-TREE verify (scheduler stashed a branch payload) at
    //       k up to 32 — the tree_wy kernel runs (h_state stale) → the commit
    //       MUST source from the intermediate pool even on a full accept.
    // This is the crux the task asked to trace: at capacity=32 the `k==capacity`
    // predicate does NOT mis-fire for the k=17 flat verify.

    /// (a) capacity=32, flat k=17 verify: injection does NOT fire, so full
    /// accept commits the live h_state (wy17 route). No regression vs the
    /// pre-free-slots γ=16/cap=17 world, because the predicate keys on equality
    /// with capacity, not on ">0".
    #[test]
    fn cap32_flat_k17_verify_injection_does_not_fire() {
        let k = 17;
        let cap = 32;
        // ATLAS_DISABLE_TREE_WY unset (false) — the strongest test: even with
        // injection ELIGIBLE by env, k != capacity keeps it off.
        let injected = flat_tree_wy_injection_applies(false, true, false, cap, k);
        assert!(
            !injected,
            "k=17 != capacity=32 must NOT inject the flat tree_wy route"
        );
        let (route, plan) = flat_chain_verify_commit(k, cap, false, true, k, false, 1, false);
        // wy17 route runs (k==17), h_state written live → LiveHState fast path.
        assert_eq!(route, GdnVerifyRoute::Wy17);
        assert_eq!(plan, HCommitSource::LiveHState);
        assert!(commit_source_is_valid(plan, route, k, k, None));
        // Every partial accept reads inter[total-1] (chain-contiguous).
        for total in 1..k {
            let (route, plan) = flat_chain_verify_commit(k, cap, false, true, total, false, 1, false);
            assert_eq!(route, GdnVerifyRoute::Wy17);
            assert_eq!(plan, HCommitSource::InterSlot(total - 1));
            assert!(commit_source_is_valid(plan, route, total, k, None));
        }
    }

    /// (b) capacity=32, a REAL scheduler tree payload at any k in [2,32]: the
    /// commit must see tree mode and source from the intermediate pool on BOTH
    /// full and partial accepts (the tree_wy kernel leaves h_state stale). This
    /// is the deep-sibling free-slots commit contract.
    #[test]
    fn cap32_deep_tree_verify_commits_from_inter_pool() {
        // Free-slots trees are ALWAYS wide (spine γ=16 + branches ⇒ k >= 18),
        // so k >= 5 always holds — the fused-small-K shortcut (k in 2..4) never
        // applies. Sweep k in [5, 32] where a stashed payload routes tree_wy.
        for k in 5..=32usize {
            let injected = false; // stash set, not the flat injection
            // A real tree payload → parent_ids_in_ctx = true → tree_wy branch
            // (checked before the k==17 wy17 branch). Mirror gdn_route order.
            let route = gdn_route(k, /*parent_ids_in_ctx=*/ true, true, false, true);
            assert_eq!(
                route,
                GdnVerifyRoute::TreeWy,
                "a stashed tree payload at k={k} must route tree_wy"
            );
            for total in 1..=k {
                let tree_mode = commit_sees_tree_mode(true, injected);
                assert!(tree_mode, "a real tree payload must set tree mode");
                let slot = total - 1;
                let replay = wy17_replay_allowed(false, 1, tree_mode, total, k, slot, false);
                let plan = plan_h_commit_source(total, k, tree_mode, slot, replay);
                // Tree mode: never LiveHState (even on full accept) — the
                // tree_wy kernel leaves h_state stale and writes every
                // intermediate slot 0..=k-1, so InterSlot(total-1) is valid for
                // full AND partial accepts.
                assert_eq!(
                    plan,
                    HCommitSource::InterSlot(slot),
                    "deep tree k={k} total={total} must commit inter[{slot}]"
                );
                assert!(
                    commit_source_is_valid(plan, route, total, k, None),
                    "deep tree commit invalid k={k} total={total} plan={plan:?}"
                );
            }
        }
    }

    // ── exhaustive flat-chain sweep: any K in [2, 32] must be safe ──────

    /// For every K∈[2,32], every capacity, graphs on/off, injection env
    /// on/off, lazy env on/off, and every accept count: the commit must read
    /// a buffer the route's kernel actually wrote.
    #[test]
    fn flat_chain_commit_source_always_valid_for_any_k() {
        for k in 2..=32usize {
            for capacity in [12usize, 17, 25, 32] {
                for disable_tree_wy in [false, true] {
                    for use_graphs in [false, true] {
                        for (lazy_env, lazy_j) in [(false, 1u32), (true, 8u32)] {
                            for total in 1..=k {
                                let injected = flat_tree_wy_injection_applies(
                                    disable_tree_wy,
                                    use_graphs,
                                    false,
                                    capacity,
                                    k,
                                );
                                let route = gdn_route(k, injected, true, false, true);
                                // The dispatch arms the lazy replay ONLY on the
                                // wy17 route (K==17, vsplit off) — mirror that.
                                let armed = lazy_env && lazy_j > 1 && route == GdnVerifyRoute::Wy17;
                                let tree_mode = commit_sees_tree_mode(false, injected);
                                let slot = total - 1;
                                let replay = wy17_replay_allowed(
                                    lazy_env, lazy_j, tree_mode, total, k, slot, armed,
                                );
                                let plan = plan_h_commit_source(total, k, tree_mode, slot, replay);
                                let lazy_armed_j = if armed { Some(lazy_j) } else { None };
                                assert!(
                                    commit_source_is_valid(plan, route, total, k, lazy_armed_j),
                                    "INVALID commit source: k={k} capacity={capacity} \
                                     disable_tree_wy={disable_tree_wy} graphs={use_graphs} \
                                     lazy=({lazy_env},{lazy_j}) total={total} \
                                     route={route:?} plan={plan:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
