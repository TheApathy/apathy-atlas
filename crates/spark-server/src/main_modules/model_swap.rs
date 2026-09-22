// SPDX-License-Identifier: AGPL-3.0-only

//! Replacing the running model without restarting the process.
//!
//! PENDING — not in the module tree. Drops in at
//! `crates/spark-server/src/main_modules/model_swap.rs` on top of the
//! model-host indirection, plus `pub(crate) mod model_swap;` in `mod.rs`.
//! It compiled and its 8 tests passed against that indirection; it has NEVER
//! been run against a GPU.
//!
//! The order below is the whole design, and every step exists because skipping
//! it breaks something specific:
//!
//! 1. **Clear the host.** New requests get 503 `model_not_loaded` immediately.
//!    Requests already running keep the `Arc` they took and finish against the
//!    model they started with — see `ModelHost::current`.
//! 2. **Drop the outgoing `AppState`.** That closes `request_tx`, which is the
//!    only way the scheduler learns to stop.
//! 3. **Join the scheduler.** It returns only once fully drained. This join is
//!    what proves nothing is still touching the weights — without it, teardown
//!    races live kernels.
//! 4. **Tear the model down.** Only safe once (3) has returned: on GB10 a free
//!    interleaved with other allocation traffic corrupts neighbouring
//!    allocations, and a drained, joined scheduler is the quiescent moment that
//!    makes it safe.
//! 5. **Load the new model**, carrying the process-scoped stores forward.
//! 6. **Publish.** Requests resume against the new model.
//!
//! ## Recoverability — the table that decides whether this is safe to expose
//!
//! | Failure point                                   | Outcome |
//! |-------------------------------------------------|---------|
//! | Bad argv / missing checkpoint / no kernels for   | **Nothing released.** Old model still serving. |
//! | the model / shutting down / multi-rank          | |
//! | A leaked `Arc<AppState>` outlives the drain      | **Refused, transactionally.** Model put back, still serving. |
//! | New model fails to load                          | **Old argv reloaded automatically** into memory its own teardown just freed. Returns `Err`. |
//! | New model fails AND restore fails                | **NOT RECOVERABLE.** No model loaded, 503s, error names both failures. |
//! | Scheduler thread panics during drain             | **NOT RECOVERABLE.** Returns before loading; no model. |
//!
//! The two NOT RECOVERABLE rows are the ones that decide exposure. Neither can
//! strand a GPU allocation — in both the outgoing model has already been freed
//! by its own teardown, so the process ends up with no model rather than with
//! leaked memory — but both end with the server answering 503 until an operator
//! intervenes. A scheduler panic during drain is the worse of the two, because
//! unlike a failed load it carries no diagnosis of its own.
//!
//! **The cost, and what is done about it.** The swap is committed: by step 4
//! the old model is gone, so a failure in step 5 cannot be undone by simply not
//! proceeding. Three things narrow that window, in order of how much they buy:
//!
//! 1. **Validate before step 1.** A bad flag combination, an absent checkpoint
//!    or a model this binary has no kernels for never reaches the drain, so the
//!    overwhelmingly common failure costs nothing at all.
//! 2. **Restore on failure.** The memory it needs was just freed by its own
//!    teardown, so the restore is loading a model that demonstrably fit moments
//!    ago — the case with the best odds of succeeding.
//! 3. **Report honestly when both fail.** The error names BOTH failures. A
//!    restore that fails silently is worse than no restore, because the
//!    operator then debugs the wrong model.
//!
//! ## Known operational characteristic
//!
//! `responses_stream.rs:386` captures an `Arc<AppState>` inside its spawned SSE
//! task for the life of the stream. A generation still streaming after
//! `DRAIN_GRACE` makes `release_state` refuse — fail-safe (model restored,
//! still serving), but the operator sees "cannot swap" and must retry.

use std::sync::Arc;

use anyhow::Result;

use super::model_host::ModelHost;
use super::serve_load::{Carried, load_model};
use crate::cli;

/// What a swap needs to know to undo itself.
#[derive(Debug)]
pub(crate) struct SwapOutcome {
    /// The argv of the model that was replaced, for a restore offer.
    pub previous: Option<cli::ServeArgs>,
}

/// Do not start a multi-minute load into a process that is on its way out.
///
/// The accept loop stops the moment shutdown is requested, so the model would
/// finish loading with nothing left to serve it — and the release it performs
/// first would take the OUTGOING model down with it, turning a clean drain into
/// an abrupt one.
///
/// Takes the answer rather than reading the global, so the rule is testable
/// without a test mutating a process-wide latch that has no reset and that
/// every other test calling `swap` would then trip over.
fn refuse_if_shutting_down(shutting_down: bool) -> Result<()> {
    anyhow::ensure!(
        !shutting_down,
        "shutdown is in progress — not starting a model load"
    );
    Ok(())
}

/// Refuse a model this binary has no kernels for, BEFORE anything is released.
///
/// The check itself already exists inside the load — but it runs at phase 3,
/// long after the outgoing model has been torn down. Discovering it there costs
/// a live server its model for a reason that was knowable from a JSON file
/// before anything was touched.
/// Returns the checkpoint's `model_type` so the caller can look up its launch profile
/// without reading config.json twice.
fn preflight_kernel_target(args: &cli::ServeArgs) -> Result<String> {
    let model_dir = super::serve_phases::resolve_model_dir(args)?;
    let (config, _) = super::serve_phases::load_model_config(&model_dir)?;
    if atlas_kernels::ptx_for_config(&config.model_type, config.hidden_size).is_none() {
        anyhow::bail!(
            "this build has no compiled kernels for model_type '{}' / hidden_size={} \
             (available: {:?}) — the running model is untouched",
            config.model_type,
            config.hidden_size,
            atlas_kernels::available_targets()
                .iter()
                .map(|t| &t.target.model)
                .collect::<Vec<_>>()
        );
    }
    Ok(config.model_type)
}

/// Copy the flags that describe the PROCESS, not the model, from the argv that
/// is running onto the argv that is about to.
///
/// A recipe describes a model: its checkpoint, quantization, context, batch
/// shape. It has no business deciding which socket the operator bound. The
/// socket is the starkest case — it is bound for the process lifetime and
/// cannot move, so a recipe's port is unserveable by construction.
fn carry_process_flags(next: &mut cli::ServeArgs, previous: &cli::ServeArgs) {
    next.bind = previous.bind.clone();
    next.port = previous.port;
    // Request dumping is an operator's observability choice, and no recipe sets
    // it. Without this, a swap replaces argv with the recipe's and the dump
    // silently stops — the file stays where it was, simply never written to
    // again, which is the worst way for a diagnostic to fail.
    next.dump = previous.dump.clone();
    // Auth is NOT carried here, because it is not carried through argv at all:
    // `swap` hands `load_model` the policy the host has held since boot, so a
    // recipe cannot drop `--require-auth` however its flags are written.
    //
    // NOTE for the upstream diff: upstream also carries `auto_swap` /
    // `no_auto_swap` here. This tree has neither field — `ModelHost::
    // auto_swap_enabled` returns a hardcoded false — so those two lines are
    // deliberately absent, not forgotten. Add them with the auto-swap port.
}

/// How long in-flight requests get to release the outgoing model before the
/// swap gives up and puts it back.
const DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

/// Block until `state` has no other owners, returning how many remain.
///
/// Split out from `release_state` so the waiting rule is testable without
/// standing up an `AppState`, which needs a loaded model.
fn wait_for_sole_owner<T>(state: &Arc<T>, grace: std::time::Duration) -> usize {
    let deadline = std::time::Instant::now() + grace;
    while Arc::strong_count(state) > 1 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Arc::strong_count(state) - 1
}

/// Take the outgoing model out of the host and drop it, carrying the
/// process-scoped stores forward.
///
/// The wait is the point. An `Arc<AppState>` that outlives this window keeps
/// `request_tx` open, so the scheduler never drains and the join below never
/// returns — the swap wedges the server with no model loaded and no way back.
/// Two kinds of holder reach this point and the wait separates them: an
/// in-flight request finishes on its own, while a structural leak never does
/// and is reported as a refusal, with the model put back and still serving.
///
/// This is the reason the router is stated on the `ModelHost` rather than on an
/// `Arc<AppState>`: a state bound into a router layer is held for the router's
/// lifetime, so it could never reach a strong count of 1 and this function
/// could only ever refuse.
fn release_state(host: &Arc<ModelHost>, grace: std::time::Duration) -> Result<Carried> {
    // Nothing loaded. Nothing to drain, nothing to lose. NOT `Carried::
    // from_env()` on this branch: rebuilding them here is the silent
    // store-loss this whole type exists to prevent.
    let Some(state) = host.take() else {
        return host
            .process()
            .ok_or_else(|| anyhow::anyhow!("process-scoped state was never installed"));
    };
    // Taken off the outgoing state in the same breath as it leaves the host, so
    // there is no window in which these could be rebuilt by mistake.
    let carried = Carried::from_previous(&state);
    let holders = wait_for_sole_owner(&state, grace);
    if holders > 0 {
        // Transactional: the host had it a moment ago and nothing has been
        // freed, so putting it back restores the exact state we started in.
        host.publish(state);
        anyhow::bail!(
            "cannot swap: {holders} reference(s) to the running model outlived \
             the {}s drain window, so it can never be released. The model is \
             still serving. This is a leaked `Arc<AppState>` — most likely one \
             bound into a router layer or a spawned task rather than resolved \
             per request.",
            grace.as_secs()
        );
    }
    drop(state);
    Ok(carried)
}

/// Release the serving model and JOIN its scheduler before the process exits.
///
/// Returning from `main` while the scheduler thread is still dropping the model
/// races process teardown: the CUDA driver deinitialises underneath the drop,
/// `TransformerModel::drop`'s stream sync fails with CUDA_ERROR_DEINITIALIZED
/// and aborts, and every clean Ctrl+C dumps core. Same two steps a swap takes
/// before loading (release the state that owns `request_tx`, then join), under
/// the same guard so a swap in flight is not torn in half. This carries
/// phaseA-a1's `serve_shutdown` intent onto the ModelHost design.
pub(crate) fn retire_for_shutdown(host: &Arc<ModelHost>) -> Result<()> {
    let _swapping = host.swap_guard();
    release_state(host, DRAIN_GRACE)?;
    match host.take_scheduler() {
        Some(handle) => super::serve_shutdown::join_scheduler(handle),
        None => Ok(()),
    }
}

/// Mark the router/listening phases done and announce ready.
///
/// The listener is bound at boot and a swap never touches it, so `load_model`
/// cannot emit these itself — but a dashboard that never sees them keeps a
/// half-finished checklist and a LOADING pill over a server that is serving.
/// Both the swap and the restore need it, which is why it is a function and not
/// two copies.
fn signal_listener_phases(host: &Arc<ModelHost>) {
    if let Some((bind, port)) = host.bound() {
        spark_runtime::progress::phase(10, "router");
        spark_runtime::progress::phase(11, "listening");
        spark_runtime::progress::ready(port);
        tracing::info!(
            "swap complete: serving {} on {bind}:{port}",
            host.live_model().as_deref().unwrap_or("<unknown>"),
        );
    }
}

/// Replace the running model with the one `next` describes.
///
/// Blocking — it loads a model. Call it off the runtime.
pub(crate) fn swap(host: &Arc<ModelHost>, next: cli::ServeArgs) -> Result<SwapOutcome> {
    // Serialised HERE, not at one call site. Two swaps at once both call
    // `ModelHost::take`; the second gets `None`, mistakes it for a modelless
    // boot, and loads a second model onto a GPU that is already loading one.
    // A dashboard user pressing `s` twice is enough to reach that.
    let _swapping = host.swap_guard();

    // Taken from the host, not a parameter: a caller that has to supply the
    // outgoing argv is a caller that can forget, and forgetting it silently
    // disables restore-on-failure.
    let previous_args = host.args();

    let mut next = next;
    if let Some(previous) = previous_args.as_ref() {
        if next.port != previous.port || next.bind != previous.bind {
            tracing::warn!(
                "this recipe asks to bind {}:{}, but the listener is on {}:{} for the \
                 process lifetime — serving the new model there instead",
                next.bind,
                next.port,
                previous.bind,
                previous.port
            );
        }
        carry_process_flags(&mut next, previous);
    }

    // Whoever waited on the guard may be asking for what the winner just
    // loaded. Re-checked AFTER the guard, not before: while a request waited,
    // the winner may have loaded exactly the model it wanted, and repeating a
    // multi-minute load is a second outage for nothing.
    //
    // Compared as a WHOLE argv rather than by model id, because switching
    // between two recipes for the SAME checkpoint is a real swap. And AFTER
    // carrying, not before: `previous_args` is the LIVE argv and already holds
    // the carried flags while `next` does not, so comparing them first would
    // make the two unequal whenever any process flag was set — which is exactly
    // when requests queue behind a swap.
    if previous_args.as_ref() == Some(&next) && host.current().is_some() {
        tracing::info!("swap: the requested model is already the one serving — nothing to do");
        return Ok(SwapOutcome {
            previous: previous_args,
        });
    }

    // Refuse before anything is torn down. A bad flag combination, a missing
    // checkpoint or an impossible VRAM budget must cost nothing — the window
    // where the server has no model is opened only for a config that has
    // already passed everything cheap.
    cli::validate_serve_args(&next).map_err(|e| anyhow::anyhow!("{e}"))?;

    refuse_if_shutting_down(crate::tui::shutdown::requested())?;

    // The load spawns Tokio tasks. Entering here rather than at each call site
    // means a caller that is already inside the runtime and one that is not
    // (the TUI's plain thread) both work, and neither has to know which it is.
    let runtime = host.runtime();
    let _entered = runtime.as_ref().map(|h| h.enter());

    // Multi-rank is out of scope and must fail loudly rather than half-swap:
    // the EP worker takes the model by `Option::take` and only returns when the
    // head exits, so there is no "load a different model" command to send it.
    //
    // UNTESTED ON HARDWARE, and unavoidably so — multi-node needs two boxes and
    // a real EP deployment. What is verified is that the refusal fires; what is
    // not is the behaviour of a head or worker that reaches this path in a
    // genuine world_size > 1 run.
    anyhow::ensure!(
        next.world_size <= 1 && next.rank == 0,
        "hot-swap is single-node only (world_size={}, rank={})",
        next.world_size,
        next.rank
    );

    // Cheapest checks first: this one reads the checkpoint's config.json, so it
    // runs after the ones that need nothing but the argv.
    let model_type = preflight_kernel_target(&next)?;

    // A model with a launch profile (memory, run-alone, env) — DeepSeek-V4.1 today. Refused
    // HERE, before anything is released, only when it could never fit on this machine at all;
    // the real admission is after the release below, on the memory that is actually free.
    let profile = super::model_profile::profile_for_model_type(&model_type);
    if let Some(p) = profile.as_ref().filter(|p| p.run_alone) {
        let total = super::model_profile::mem_total_bytes()?;
        anyhow::ensure!(
            total >= p.required_bytes(false),
            "{} needs {:.1} GB free but this machine has {:.1} GB in total — the running model \
             is untouched",
            p.recipe_id,
            p.required_bytes(false) as f64 / 1e9,
            total as f64 / 1e9
        );
    }

    // The policy the host has held since boot. NOT rebuilt from `next`: a
    // recipe's argv must not be able to drop `--require-auth` from a server
    // that was started with it.
    let auth = host.auth();

    // 1 + 2. Stop admitting work, and release the state that owns request_tx.
    let carried = release_state(host, DRAIN_GRACE)?;

    // 3. Wait for the scheduler to finish draining. JOINED, not detached: this
    // return is what proves the weights are no longer in use.
    if let Some(handle) = host.take_scheduler() {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("the scheduler thread panicked while draining"))?;
    }

    // 4 + 5. The model drops as the scheduler thread unwinds, which is where
    // its pools are freed; then the new one loads.
    let next_args = next.clone();
    // From the HOST: a swap must republish its handles or the dashboard keeps
    // sampling the scheduler it just joined.
    //
    // NOTE: `load_model` in the landed tree does NOT yet take a
    // `tui_handles_tx` — the levers publish was reverted as out of scope for
    // the indirection change. Restoring this line means restoring that
    // parameter and the `RunHandles` send beside the scheduler spawn.
    let tui_handles_tx = host.tui_handles();
    // Launch profile: its environment first (the loader reads it), then run-alone admission
    // on the memory that is free NOW that the outgoing model is gone. A refusal is treated
    // exactly like a failed load: the previous model is restored below.
    let admitted = match profile.as_ref() {
        None => Ok(()),
        Some(p) => {
            let set = super::model_profile::apply_env(p);
            if !set.is_empty() {
                tracing::info!("{}: launch environment {:?}", p.recipe_id, set);
            }
            let dspark_on = std::env::var("ATLAS_DSV41_DSPARK").is_ok_and(|v| v == "1");
            super::model_profile::mem_available_bytes()
                .and_then(|available| super::model_profile::admit(p, available, dspark_on))
        }
    };
    let load_result = match admitted {
        Ok(()) => load_model(next, tui_handles_tx.clone(), carried.clone(), auth.clone()),
        Err(e) => Err(e),
    };
    let load_err = match load_result {
        Ok(Some(prepared)) => {
            // 6.
            host.set_scheduler(prepared.scheduler);
            host.set_args(next_args);
            host.publish(prepared.state);
            signal_listener_phases(host);
            return Ok(SwapOutcome {
                previous: previous_args,
            });
        }
        Ok(None) => anyhow::anyhow!("hot-swap reached an EP-worker path on rank 0"),
        Err(e) => e,
    };

    // The new model did not load and the old one is already gone. Put the old
    // one back: its memory was freed by its own teardown moments ago, so this
    // is the load with the best chance of succeeding.
    let Some(previous) = previous_args else {
        return Err(load_err
            .context("the new model failed to load and there was no previous model to restore"));
    };
    tracing::warn!("load failed, restoring the previous model: {load_err:#}");
    match load_model(previous.clone(), tui_handles_tx, carried, auth) {
        Ok(Some(prepared)) => {
            host.set_scheduler(prepared.scheduler);
            host.set_args(previous);
            host.publish(prepared.state);
            // The restored model is serving, so the dashboard must say so.
            // Without this the checklist stays frozen part-way and the pill
            // reads LOADING for a server that is answering requests.
            signal_listener_phases(host);
            // Deliberately an Err: the requested swap did NOT happen, and
            // returning Ok would tell the caller it did.
            Err(load_err.context("the new model failed to load; the previous one was restored"))
        }
        // Both failed. Name both — an operator told only about the restore
        // failure debugs the wrong model.
        Ok(None) => Err(load_err.context("restore reached an EP-worker path")),
        Err(restore_err) => Err(load_err.context(format!(
            "the new model failed to load AND the previous one could not be \
             restored ({restore_err:#}) — no model is loaded"
        ))),
    }
}

#[cfg(test)]
#[path = "model_swap_tests.rs"]
mod tests;
