// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]
// `#![allow(dead_code)]` is retained at the binary-crate level because
// spark-server ships ~20 module files that are operational scaffolding
// (LASER routing, lookback-lens reranking, retrieval-head probes,
// SymbolTrie token-constraint, etc.) — instantiated by an upcoming
// scheduler hook but not yet wired in. Auditing and per-item narrowing
// is desirable but mechanical; the lib-side audit was the high-value
// pass (it masked exactly ONE leaking duplicate, ChatTokenizer::
// TEMPLATE_OVERRIDE_DIR). File-level allows added to the 5 highest-
// density research-scaffolding modules below tighten the loop:
//   - lqer.rs, tool_rag.rs, moe_quality.rs, llmlingua.rs, lookback_lens.rs
// This crate-level allow shrinks but doesn't disappear pending a
// dedicated dead-code sweep.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! Atlas Spark — pure Rust LLM inference server.
//!
//! Startup sequence:
//! 1. Parse CLI args
//! 2. Load model config
//! 3. Initialize GPU backend (AtlasCudaBackend)
//! 4. Load model weights (SafetensorsLoader)
//! 5. Build model via factory
//! 6. Load tokenizer
//! 7. Spawn scheduler thread
//! 8. Start axum HTTP server

// Global allocator: mimalloc. Faster than glibc's ptmalloc on the
// Vec-heavy / String-heavy allocations in the per-token scheduler
// hot path (draft Vecs, D2H buffers, content sanitiser, accumulator
// strings). Picked over jemalloc for: smaller binary, better aarch64
// performance, and active maintenance.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod adaptive_sampler;
mod anthropic;
mod api;
mod auth;
mod citation;
mod cli;
mod conversation_store;
pub mod grammar;
mod halluc_probe;
mod hint_injector;
mod llmlingua;
mod lookback_lens;
mod loop_detector;
mod loop_simhash;
mod lqer;
mod main_modules;
pub mod metrics;
mod model_resolver;
mod moe_quality;
mod ngram;
mod observation_mask;
mod openai;
mod rate_limiter;
pub mod reasoning_parser;
mod refusal;
mod request_dumper;
mod response_store;
// REST retrieval draft store: conditional pre-emption of the DFlash
// proposal in `scheduler::mtp_step`. Inert unless ATLAS_REST_STORE is set.
mod rest_store;
mod retrieval_heads;
mod scheduler;
mod scheduling_policy;
mod session_manager;
mod symbol_trie;
mod task_pin;
mod tokenizer;
mod tool_arg_dedup;
pub mod tool_parser;
mod tool_rag;
mod tool_salvage;
mod tscg;
pub mod tui;

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, Command};
use crate::main_modules::serve;

pub(crate) use crate::main_modules::AppState;

/// Re-export for convenience in api.rs / anthropic.rs.
pub type ModelBehavior = atlas_kernels::ModelBehavior;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse BEFORE subscriber install so the TUI gate can see `--no-tui`.
    // clap emits no tracing events, so plain-mode output is unchanged.
    let cli = Cli::parse();
    let no_tui = match &cli.command {
        Command::Serve(args) => args.no_tui || args.rank > 0,
    };

    let plain_mode = tui::plain_mode(no_tui);
    tui::set_active(!plain_mode);
    let tui_channels = if plain_mode {
        // The pre-TUI init, byte-for-byte: this exact fmt layout is the
        // contract every benchmark driver and gate script greps.
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .init();
        None
    } else {
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        tui::init::install_tty_subscriber(progress_tx);
        Some(progress_rx)
    };

    // Race the server against shutdown. No spawn: `serve()` is a real future that
    // yields while its blocking startup runs on the blocking pool, so pinning it
    // here is enough for `select!` to poll the other branch. (It would NOT be
    // enough if startup still blocked inside the future — `select!` chooses at
    // await points, it does not preempt.)
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<&'static str>();
    tui::shutdown::arm_startup_escape(shutdown_tx);
    let result = match cli.command {
        Command::Serve(args) => {
            let serving = serve(args, tui_channels);
            tokio::pin!(serving);
            // Only a SEND means shutdown. The sender is parked for the life of the
            // process rather than dropped when startup ends, so the channel should
            // never close; this arm exists so that if one ever did, a closed
            // channel could not masquerade as a shutdown and kill a healthy server.
            let shutdown_signal = async {
                match shutdown_rx.await {
                    Ok(reason) => reason,
                    Err(_) => std::future::pending::<&'static str>().await,
                }
            };
            tokio::pin!(shutdown_signal);
            tokio::select! {
                res = &mut serving => res,
                reason = &mut shutdown_signal => {
                    // Cancelled before the server came up. Nothing is in flight
                    // and no client is connected, so there is nothing to drain —
                    // the startup task is abandoned where it stands.
                    tracing::info!(
                        "Shutdown requested ({reason}) during startup — exiting before the server came up"
                    );
                    // Cleanup that would otherwise run below, then exit without
                    // waiting on the runtime: a task parked inside a synchronous
                    // CUDA call cannot be aborted, and dropping the runtime would
                    // block on it — reintroducing the very wait this fixes.
                    tui::stop_and_join(std::time::Duration::from_secs(2));
                    tui::terminal_guard::restore();
                    tui::init::flush_tee();
                    std::process::exit(0);
                }
            }
        }
    };
    // If serve() returned while the TUI owned the screen (startup error, clean
    // shutdown), stop the dashboard thread and wait for its TerminalGuard to
    // drop BEFORE the error prints — main's exit never runs another thread's
    // Drop, and a bare restore() races the thread's raw-mode entry when
    // serve() fails within milliseconds. restore() stays as the backstop.
    tui::stop_and_join(std::time::Duration::from_secs(2));
    tui::terminal_guard::restore();
    tui::init::flush_tee();
    result
}
