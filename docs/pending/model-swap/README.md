# Pending: in-process model swap

**This directory is NOT wired into the build.** `model_swap.rs` is not in
`main_modules/mod.rs`, and nothing calls `swap()`. It is designed, it compiled,
and its 8 tests passed — against the model-host indirection that landed
separately. **It has never been run against a GPU.**

It is committed here so 385 lines of designed work are not lost to a scratchpad
directory's lifetime. It is deliberately the only thing on this branch: no
source changes, so the branch cannot be mistaken for a landing.

## Files

| File | What it is |
|---|---|
| `model_swap.rs` | The swap. Drops in at `crates/spark-server/src/main_modules/model_swap.rs`. The recoverability table is the module doc. |
| `model_swap_tests.rs` | 8 tests, all passing at the time this was set aside. Beside it as `model_swap_tests.rs`. |
| `launch_wiring.rs.patch` | The 5 edits that re-attach it: `serve.rs`, `app_library.rs`, `lib_state.rs`, `mod.rs`, and dropping `#[allow(dead_code)]` from `Carried::from_previous`. |

## What this applies on top of

Commit `81bdbe130` **plus** the model-host indirection change on
`feat/tui-port`. The indirection is a hard prerequisite, not a convenience:
before it, the router was stated on an `Arc<AppState>` baked in at boot, so
`host.publish(new_state)` changed nothing a request could observe and every
handler kept serving the state captured at boot — whose scheduler the swap had
just joined and whose weights it had just freed. That was the **default**
outcome, not an edge case.

## Three things that are easy to undo by accident

1. **`release_state`'s empty-host branch must NOT fall back to
   `Carried::from_env()`.** It looks like a missing default. It is the silent
   store-loss the `Carried` type exists to prevent — every stored conversation
   and response dropped, every rate-limit bucket reset, no error.
2. **`carry_process_flags` omits upstream's `auto_swap` / `no_auto_swap`
   deliberately.** This tree has neither field; `ModelHost::auto_swap_enabled`
   returns a hardcoded `false`. Do not "fix" the omission — add them with the
   auto-swap port.
3. **Restoring the levers publish means restoring `load_model`'s
   `tui_handles_tx` parameter** and the `RunHandles` send beside the scheduler
   spawn. Both were reverted as out of scope for the indirection change. Note
   that `scheduler::run` takes no levers argument in this tree — the loop
   watchdog is gated by the global `scheduler::set_enable_loop_watchdog` — so a
   published lever is correct to READ but toggling it does not reach the
   scheduler.

## Before exposing this, decide the two NOT-RECOVERABLE rows

The full table is in `model_swap.rs`. Neither row can strand a GPU allocation —
in both the outgoing model has already been freed by its own teardown — but both
end with the server answering 503 until an operator intervenes:

- **New model fails to load AND the restore also fails.** The error names both.
- **The scheduler thread panics during drain.** The worse of the two, because
  unlike a failed load it produces no diagnosis of its own: a joined thread and
  no model looks like the server simply stopped.

## Two known characteristics, neither fixed

- `responses_stream.rs:386` captures an `Arc<AppState>` inside its spawned SSE
  task for the life of the stream. A generation still streaming after
  `DRAIN_GRACE` makes `release_state` refuse — fail-safe (model restored, still
  serving), but the operator sees "cannot swap" and must retry.
- **`DRAIN_GRACE` is 30s, and that is an untuned guess.** It was never measured
  against real generation lengths. Treat it as a value to establish, not one to
  inherit.
