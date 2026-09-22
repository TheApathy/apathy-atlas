// SPDX-License-Identifier: AGPL-3.0-only

//! Process-scoped startup: the parts of bringing a server up that happen ONCE,
//! however many models the process goes on to serve.
//!
//! The model-dependent remainder — config, weights, KV cache, tokenizer,
//! scheduler, `AppState` — lives in `serve_load::load_model`, which this calls
//! and which a swap calls again. The line between them is not cosmetic: it is
//! the difference between "load another model" and "install a second set of
//! signal handlers, start a second dashboard thread, and spawn a second OOM
//! watchdog", which is what the fused version would have done.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::cli;
use crate::main_modules::model_host::ModelHost;
use crate::main_modules::serve_load::{self, Prepared};

/// Bring the engine up, then serve.
///
/// Startup — weight load, KV allocation, kernel audit, graph capture — is ~50s of
/// SYNCHRONOUS CPU/IO/CUDA work containing not one `await`. Running it in the body
/// of this future would mean whatever polls the future is blocked for that whole
/// time: an `async fn` that never yields is a blocking call wearing `async`, and
/// it is why `q`/Ctrl+C appeared dead during a model load — nothing, not even the
/// signal listener, could make progress until loading finished.
///
/// So startup runs on the blocking pool and is AWAITED here. That await is a real
/// yield point, which is what lets `main` race this future against a shutdown
/// channel, and it keeps the async workers free regardless of how the runtime is
/// sized.
pub(crate) async fn serve(
    args: cli::ServeArgs,
    tui_progress: Option<std::sync::mpsc::Receiver<crate::tui::capture_layer::ProgressEvent>>,
) -> Result<()> {
    // ONE host for the process lifetime, built HERE rather than inside startup
    // — and built before the load, so the dashboard can hold it while the first
    // model is still reading shards, and so a swap has somewhere to publish to.
    //
    // Constructed on the runtime deliberately: `ModelHost::empty` captures
    // `Handle::try_current()`, and a swap driven from the TUI's plain thread
    // has no runtime in scope. Without a handle captured here, the first
    // Library launch panics with "there is no reactor running" — after the
    // outgoing model has already been released.
    let host = Arc::new(ModelHost::empty());
    // Recorded BEFORE `args` moves into startup: the first swap restores to
    // this if its load fails, and without it the first swap is the one swap
    // with no safety net.
    host.set_args(args.clone());
    // Process-scoped, and in force from the moment the listener is up —
    // including while a swap has no model loaded. Whether a request is
    // authorised must not depend on whether a model happens to be resident,
    // and a recipe's argv must never be able to drop `--require-auth`.
    let auth = build_auth_config(&args)?;
    host.set_auth(auth.clone());
    // Likewise process-scoped: these three outlive every model. Built once,
    // here, so a swap can only ever carry them forward.
    let carried = serve_load::Carried::from_env()
        .map_err(|e| anyhow::anyhow!("process-scoped state: {e}"))?;
    host.set_process(carried.clone());

    let Some(prepared) =
        tokio::task::spawn_blocking({
            let host = host.clone();
            move || startup(args, tui_progress, carried, auth, host)
        })
        .await??
    else {
        return Ok(()); // EP worker: no router on this rank
    };

    // Into the HOST, not into a local. The router reads the current model
    // through the host on every request, which is what lets a swap change what
    // is served without rebuilding the router or moving the socket.
    host.publish(prepared.state);
    // The first load's scheduler belongs to the host too. Dropping a
    // `JoinHandle` detaches the thread, which is harmless for a process that
    // runs one model and exits — and makes a swap impossible, because the
    // outgoing scheduler can then never be joined, and without that join
    // teardown races live kernels.
    host.set_scheduler(prepared.scheduler);
    let serve_result =
        crate::main_modules::serve_router::build_and_serve(host.clone(), &prepared.bind, prepared.port)
            .await;
    // Join the scheduler (and so the model's drop) before `main` returns;
    // see `model_swap::retire_for_shutdown`.
    let scheduler_join = tokio::task::spawn_blocking(move || {
        crate::main_modules::model_swap::retire_for_shutdown(&host)
    })
    .await;
    crate::main_modules::serve_shutdown::resolve_serve_result(serve_result, scheduler_join)
}

/// The once-per-process half. Everything here is either global state or a
/// thread that outlives every model; nothing here is re-run by a swap.
fn startup(
    args: cli::ServeArgs,
    tui_progress: Option<std::sync::mpsc::Receiver<crate::tui::capture_layer::ProgressEvent>>,
    carried: serve_load::Carried,
    auth: Option<Arc<crate::auth::AuthConfig>>,
    host: Arc<ModelHost>,
) -> Result<Option<Prepared>> {
    tracing::info!("Atlas Spark starting...");
    tracing::info!("Licensed under AGPL-3.0-only — see /LICENSE in this container");
    spark_runtime::progress::phase(0, "banner");

    // BEFORE any allocation: the dashboard reports "atlas used" as
    // `baseline - free`, and a baseline taken after the weights are
    // resident collapses that difference to ~0. Idempotent, so an earlier
    // reader cannot spoil it.
    spark_runtime::gpu::capture_baseline();

    // Clean shutdown: SIGINT/SIGTERM now request a drain-and-exit instead of
    // killing the process mid-write. In TUI mode Ctrl+C additionally arrives
    // as a key event (raw mode) and calls the same request().
    crate::tui::shutdown::install_signal_listeners();

    // Start the dashboard thread as early as possible so the operator watches
    // the load, not a blank screen. Everything it reads is process-global
    // (log ring, progress channel, metrics, scheduler snapshot) plus this
    // args snapshot for the badge chips. Head node only.
    let mut tui_handles_tx: Option<std::sync::mpsc::Sender<crate::tui::RunHandles>> = None;
    if let Some(progress_rx) = tui_progress
        && args.rank == 0
    {
        // The host, not `None`: this is what lets the Library START a model.
        // The sender it returns goes straight back into the host, so every
        // later load — the first one below included — republishes its levers
        // to the pane that samples them.
        let tx = crate::tui::start(args.clone(), progress_rx, Some(host.clone()));
        host.set_tui_handles(tx.clone());
        tui_handles_tx = Some(tx);
    }

    // Runtime per-kernel profiling toggle. SIGUSR1 enables `ATLAS_FULL_PROFILE`
    // behavior on the live instance (disables CUDA graph capture + activates
    // kprof! per-kernel timing on the DFlash K-γ verify path), SIGUSR2
    // disables it. Lets the operator profile without restarting the server.
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut enable =
            signal(SignalKind::user_defined1()).context("install SIGUSR1 profile handler")?;
        let mut disable =
            signal(SignalKind::user_defined2()).context("install SIGUSR2 profile handler")?;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = enable.recv() => spark_model::full_profile::set_runtime_profile(true),
                    _ = disable.recv() => spark_model::full_profile::set_runtime_profile(false),
                    else => break,
                }
            }
        });
        tracing::info!("runtime profile toggle armed: SIGUSR1=enable SIGUSR2=disable");
    }

    // Everything above is process-scoped. Everything below is the model, and
    // this call is the one a swap will run again.
    serve_load::load_model(args, tui_handles_tx, carried, auth)
}

/// Resolve `--require-auth` / `--auth-tokens-file` / `--auth-token` into an
/// optional `AuthConfig`. Validates at startup so misconfigurations fail
/// loudly instead of letting an unauthenticated server run silently.
pub(super) fn build_auth_config(args: &cli::ServeArgs) -> Result<Option<Arc<crate::auth::AuthConfig>>> {
    if !args.require_auth {
        if args.auth_tokens_file.is_some() || args.auth_token.is_some() {
            tracing::warn!(
                "--auth-tokens-file / --auth-token supplied without --require-auth; \
                 tokens are loaded but the auth gate is OFF. Pass --require-auth to enforce."
            );
        }
        return Ok(None);
    }
    let cfg = match (&args.auth_tokens_file, &args.auth_token) {
        (Some(path), None) => crate::auth::AuthConfig::from_file(path)?,
        (None, Some(tok)) => {
            tracing::warn!(
                "--auth-token sets the bearer token via the command line; the value \
                 is visible to other local users via `ps`/`/proc/<pid>/cmdline`. \
                 Use --auth-tokens-file with permissions 0600 in production."
            );
            crate::auth::AuthConfig::from_inline(tok)?
        }
        (None, None) => {
            return Err(anyhow::anyhow!(
                "--require-auth was set but neither --auth-tokens-file nor \
                 --auth-token was supplied. Pick one (a tokens file is preferred)."
            ));
        }
        (Some(_), Some(_)) => unreachable!("clap conflicts_with should have rejected this"),
    };
    tracing::info!(
        "auth: require_auth=ON ({} bearer token{} loaded)",
        cfg.token_count(),
        if cfg.token_count() == 1 { "" } else { "s" },
    );
    Ok(Some(Arc::new(cfg)))
}
