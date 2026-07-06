# Predictive Turn Prefetch — Design Doc (Lever 3 of the decoding-efficiency wave)

Status: DESIGN ONLY (stub not shipped — implementation estimated > 1 day).
Author: decoding-efficiency wave agent, 2026-07-05.
Gate (proposed): `ATLAS_TURN_PREFETCH=1`, default OFF.

## Goal

During idle (no active or in-progress sequences), speculatively continue the
last conversation's *likely next turn* so that when the real next request
arrives it can be served from a warm, partially-decoded state — cutting TTFT
for multi-turn agentic sessions (Claude Code / OpenCode) where the next user
turn is highly correlated with the just-finished assistant turn.

## Why this is > 1 day (feasibility verdict)

The plumbing to do this *correctly* does not exist yet, and three of the
required pieces are non-trivial:

1. **No retained per-turn decode state.** `SessionSsmManager`
   (`crates/spark-server/src/session_manager.rs`) tracks only which SSM
   *snapshot slot* a session hash owns, plus a TTL. It does NOT retain:
   - the finished turn's full token stream,
   - the KV-cache block table for that turn,
   - a decode-ready `SequenceState`.
   All of that is torn down by `retire_finished_sequences` /
   `finish_sequence` (`scheduler/lifecycle.rs`) the moment a turn ends. A
   prefetch would need a new "warm hold" lifecycle state that keeps a
   finished sequence's KV + SSM resident (or cheaply restorable) instead of
   freeing it.

2. **No "likely next turn" signal.** We would have to synthesize a speculative
   next-user-prompt. Options, all research-grade:
   - a fixed continuation template ("continue" / "and then?") — weak,
   - a small predictor over the turn history — new model/infra,
   - replay of the model's own drafted follow-up — needs a second forward.
   None is a drop-in.

3. **Correctness + resource hazards.** Prefetch must be *provably discardable*:
   the speculative work must never contaminate the KV/prefix cache, must be
   abandoned the instant a real request lands (mid-flight cancellation), and
   must not hold blocks that a real request needs (it competes with
   `num_free_blocks()` and the swap-out victim logic). Getting the
   cancellation + block-accounting right is the bulk of the effort.

## Idle-detection hook points (identified)

The scheduler already has a clean, zero-CPU idle boundary:

- **Block point:** `scheduler/mod_helpers.rs:76`,
  `drain_pending_requests` → `cv.wait(&mut g)`. When `active`, `prefilling`,
  and the pending queue are all empty, the scheduler blocks here on the
  condvar with zero CPU until the receiver thread signals a new request.
- **Loop guard:** `scheduler/mod.rs:213`,
  `if new_reqs.is_empty() && active.is_empty() && prefilling.is_empty()` — the
  exact predicate that defines "idle".

### Minimal hook shape (proposed, not implemented)

Replace the unconditional `cv.wait` with a bounded wait when prefetch is on
and a warm session exists:

```rust
// mod_helpers::drain_pending_requests, prefetch-enabled branch (sketch):
if turn_prefetch_enabled() && has_warm_session() {
    // Wake after IDLE_PREFETCH_DELAY so a real request still preempts.
    if cv.wait_for(&mut g, IDLE_PREFETCH_DELAY).timed_out() && g.requests.is_empty() {
        return DrainOutcome::Idle; // caller enters a prefetch step
    }
}
cv.wait(&mut g); // unchanged fully-idle path
```

The `loop` in `scheduler::run` then gets a new arm, entered only on
`DrainOutcome::Idle`, that:
1. restores the warm session's SSM + KV into a scratch slot,
2. runs a bounded number of speculative decode steps into a *scratch* KV
   region that is never linked into the prefix cache,
3. checks `pending` after *every* step and abandons + frees the scratch slot
   the instant a real request appears,
4. on a real hit whose prompt matches the prefetched continuation, promotes
   the scratch state instead of prefilling from scratch.

## Required new plumbing (the > 1 day work)

- `WarmHold` lifecycle state in `scheduler/types.rs` + `lifecycle.rs`
  (retain-on-finish instead of free-on-finish, TTL-bounded).
- Scratch KV/SSM region distinct from the prefix cache, with hard eviction
  priority below every real sequence.
- `DrainOutcome` enum (currently the fn returns a `Vec`) to signal idle vs
  woken-with-work.
- Next-turn hypothesis source (template first; predictor later).
- Prefetch-hit matching in `phase_start_prefills` (compare incoming prompt
  suffix against the prefetched continuation; promote on match).
- Metrics: prefetch attempts / hits / wasted-token count, TTFT delta.

## Recommendation

Ship Levers 1 + 2 + 3a (hesitation penalty, soft exit bias, adaptive budget)
now — they are self-contained logit/budget shaping with no lifecycle changes.
Treat predictive turn prefetch as a separate, scoped follow-up: land the
`WarmHold` lifecycle + scratch-KV region first (independently useful for
speculative prefetch AND for cross-turn KV reuse), then layer the next-turn
hypothesis on top. Estimated 2–4 days for a correct, cancellation-safe v1.
