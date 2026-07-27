// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler: batched concurrent decode on a single GPU thread.
//!
//! Architecture:
//! - Receiver thread: blocks on request channel, pushes to pending queue,
//!   signals condvar (instantaneous wake, zero polling).
//! - Scheduler thread: prefills new requests sequentially, then runs
//!   batched decode steps via `model.decode_batch()` — weights loaded once
//!   per step for all active sequences.
//!
//! When idle (no active sequences): blocks on condvar (zero CPU).
//! When busy: drains pending queue (mutex lock) after each decode step.

// ── Submodules (split for ≤500 LoC files) ──────────────────────────────────
mod cfg_jump_forward;
mod confidence;
mod decode_logits_content;
mod decode_logits_seq;
mod decode_logits_step;
mod decode_step;
mod emit_step;
mod helpers;
mod lifecycle;
mod logprobs;
mod mod_helpers;
mod mtp_gate;
mod mtp_step;
mod phase_continue_prefills;
mod phase_promote_prefills;
mod phase_start_prefills;
mod prefill_a_step;
mod prefill_b_step;
mod proposal_lifecycle;
mod repetition;
mod rollback;
mod sample_step;
pub mod snapshot;
mod spec_policy_accept;
mod spec_step;
mod spec_timing;
mod ssm_decode_ring;
mod think_spec_accept;
mod thinking_efficiency;
mod types;
mod verify_csk_step;
mod verify_csk_step_k2;
mod verify_dflash_batched_step;
mod verify_dflash_step;
mod verify_k2_step;
mod verify_k3_step;
mod verify_k4_step;

pub use cfg_jump_forward::{
    build_delim_table, build_forced_ids, cfg_jf_enabled, set_delim_table, set_forced_ids,
};
use confidence::*;
use decode_logits_content::*;
use decode_logits_seq::*;
use decode_logits_step::*;
use decode_step::*;
use emit_step::*;
pub use helpers::set_boundary_token_mask;
pub use helpers::set_enable_loop_watchdog;
pub use helpers::set_enable_thinking_loop_watchdog;
pub use helpers::set_im_start_hard_stop;
pub use helpers::set_numeric_token_mask;
use helpers::*;
pub use helpers::{CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN};
pub use helpers::{WatchdogParams, resolve_max_inter_tool_prose, set_watchdog_params};
use lifecycle::*;
use logprobs::*;
use mod_helpers::*;
use mtp_gate::{ArmKind, ArmSpec, MtpGate};
use mtp_step::*;
use phase_continue_prefills::continue_in_progress_prefills;
use phase_start_prefills::start_new_requests;
use prefill_a_step::*;
use prefill_b_step::*;
use repetition::*;
use rollback::{RollbackOutcome, rollback_to_boundary};
use sample_step::*;
use spec_policy_accept::*;
use spec_step::*;
use ssm_decode_ring::SsmDecodeRing;
use think_spec_accept::*;
pub use thinking_efficiency::{
    ADAPTIVE_PROBE_TOKENS as ADAPTIVE_PROBE_TOKENS_LOG, apply_think_logit_shaping,
    build_hesitation_ids, parse_config as parse_think_efficiency_config,
    set_think_efficiency_config, think_efficiency_config, top1_confidence,
};
use types::*;
use verify_csk_step::*;
use verify_csk_step_k2::*;
use verify_dflash_batched_step::*;
use verify_dflash_step::*;
use verify_k2_step::*;
use verify_k3_step::*;
use verify_k4_step::*;

// Re-exports threaded through `use super::*;` in sibling step files —
// keep these imports here even though `run` itself doesn't reference all
// of them directly (see scheduler/decode_step.rs etc.).
use anyhow::Result;
use parking_lot::{Condvar, Mutex};
use spark_model::traits::{EP_CMD_VERIFY_KGAMMA, EP_VERIFY_KGAMMA_ABORT, Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_spill::KvSpillManager;
use spark_runtime::sampler::{SamplingParams, sample_with_params, sample_with_params_history};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::api::{GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::ngram::NgramProposer;
use crate::scheduling_policy::SchedulingPolicy;

/// Build the speculation gate's arm set, or `None` to leave it disarmed.
///
/// Disarmed is the DEFAULT and must stay that way: `--mtp-gate` absent means
/// the champion configuration runs byte-for-byte as it did before the gate
/// existed, which is what makes an A/B against it meaningful.
///
/// `mode` is the resolved `--mtp-gate` value:
///   - `None`            → no gate (default)
///   - `Some("force")`   → no gate, and say so (diagnostic; verify keeps
///                          flowing even where the gate would measure it
///                          net-negative)
///   - `Some("auto")`    → arbitrate over every arm this build can offer
///   - `Some("dflash")`  → pin proposer arm 0
///   - `Some("mtp")`     → pin proposer arm 1
///
/// Optional extra arms, opt-in because neither is measured yet:
///   - `ATLAS_MTP_GATE_GAMMA_ARM=N` adds a γ-capped variant of arm 0. See
///     `speculative::set_dflash_gamma_override` for why a runtime cap is not
///     the same thing as serving with `--dflash-gamma N`.
///   - `ATLAS_MTP_GATE_SERIAL_ARM=1` adds a no-speculation arm. Measured floor
///     is ~13 tok/s on every task, i.e. dominated by both proposer arms on
///     this model — hence off by default.
fn build_speculation_gate(model: &dyn Model, use_mtp: bool, mode: Option<&str>) -> Option<MtpGate> {
    let mode = mode?;
    if !use_mtp {
        tracing::warn!("--mtp-gate {mode} ignored: speculation is not active for this run");
        return None;
    }
    if mode == "force" {
        tracing::warn!(
            "--mtp-gate force: speculation gate DISARMED (diagnostic; verify runs even \
             where the gate would measure it net-negative)"
        );
        return None;
    }

    let arms_available = model.proposer_arm_count();
    let primary_is_dflash = model.proposer_is_dflash();
    let primary_name = if primary_is_dflash { "dflash" } else { "mtp" };
    // The alternate arm is the in-checkpoint MTP head, which is measured
    // monotonically worse with more drafts on this model family (K=2 beats
    // K=4 on every benchmark task; K=8 falls below the no-speculation floor).
    // The run's `num_drafts` belongs to the DFlash arm — with a block-16
    // checkpoint it is the actual trained draft count γ=15 — so the MTP arm
    // gets its own, overridable for a
    // model whose head really does want a wider K.
    let alt_num_drafts = std::env::var("ATLAS_MTP_GATE_ALT_DRAFTS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(1);

    // Pinned arms are a one-arm "gate": no probing, no switching, but the
    // arm-selection plumbing still runs so `--mtp-gate mtp` is a real A/B
    // against `--mtp-gate dflash` on one binary and one set of weights.
    match mode {
        "dflash" => {
            return Some(MtpGate::new(vec![ArmSpec::spec(
                "pinned-dflash",
                mtp_gate::PROPOSER_ARM_PRIMARY,
                0,
                0,
            )]));
        }
        "mtp" => {
            if arms_available < 2 {
                tracing::warn!(
                    "--mtp-gate mtp: this build has only one proposer arm ({primary_name}); \
                     pass --dflash AND --speculative to build both. Pinning arm 0 instead."
                );
                return Some(MtpGate::new(vec![ArmSpec::spec(
                    "pinned-arm0",
                    mtp_gate::PROPOSER_ARM_PRIMARY,
                    0,
                    0,
                )]));
            }
            return Some(MtpGate::new(vec![ArmSpec::spec(
                "pinned-mtp",
                mtp_gate::PROPOSER_ARM_ALT,
                0,
                alt_num_drafts,
            )]));
        }
        "auto" => {}
        other => {
            tracing::error!("--mtp-gate {other}: unknown mode, gate DISARMED");
            return None;
        }
    }

    let mut arms = vec![ArmSpec::spec(
        if primary_is_dflash { "dflash" } else { "mtp" },
        mtp_gate::PROPOSER_ARM_PRIMARY,
        0,
        0,
    )];
    if arms_available >= 2 {
        arms.push(ArmSpec::spec(
            "mtp-alt",
            mtp_gate::PROPOSER_ARM_ALT,
            0,
            alt_num_drafts,
        ));
    }
    if primary_is_dflash
        && let Ok(v) = std::env::var("ATLAS_MTP_GATE_GAMMA_ARM")
        && let Ok(g) = v.trim().parse::<usize>()
        && g > 0
    {
        arms.push(ArmSpec::spec(
            "dflash-gamma-capped",
            mtp_gate::PROPOSER_ARM_PRIMARY,
            g,
            0,
        ));
    }
    if std::env::var("ATLAS_MTP_GATE_SERIAL_ARM").ok().as_deref() == Some("1") {
        arms.push(ArmSpec::serial("serial"));
    }

    if arms.len() < 2 {
        tracing::warn!(
            "--mtp-gate auto: only one arm is available ({primary_name}) — the gate will \
             measure but can never switch. Pass --dflash AND --speculative to build both \
             proposers, or set ATLAS_MTP_GATE_SERIAL_ARM=1 to arbitrate against plain decode."
        );
    }
    Some(MtpGate::new(arms))
}

/// Point the model at `arm`'s proposer and γ cap, moving every live sequence's
/// parked proposer state across with it.
///
/// The state swap is what makes an arm change safe: each proposer owns a
/// differently-typed per-sequence state and readers downcast
/// `seq.proposer_state` to the type they expect, so repointing the model
/// without moving the states would hand the incoming drafter the outgoing
/// one's buffers.
fn select_gate_arm(model: &dyn Model, active: &mut [ActiveSeq], arm: ArmSpec) {
    // `num_drafts` is read at the dispatch site, not here — this function only
    // repoints the model. The `..` keeps it that way without re-binding it.
    let ArmKind::Spec {
        proposer_arm,
        draft_cap,
        ..
    } = arm.kind
    else {
        // Serial arm: nothing to select. The γ override is left alone because
        // no propose will run this step.
        return;
    };
    let previous = model.proposer_arm();
    if previous != proposer_arm {
        let effective = model.set_proposer_arm(proposer_arm);
        if effective != previous {
            for a in active.iter_mut() {
                model.swap_proposer_state(&mut a.seq);
            }
        }
    }
    spark_model::speculative::set_dflash_gamma_override(draft_cap);
}

/// Run the scheduler loop on the current thread.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: Box<dyn Model>,
    request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
    eos_tokens: Vec<u32>,
    max_batch_size: usize,
    use_speculative: bool,
    num_drafts: usize,
    policy: Box<dyn SchedulingPolicy>,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    swap_space_gb: usize,
    high_speed_swap_cfg: Option<spark_storage::HighSpeedSwapConfig>,
    block_size: usize,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: Vec<u32>,
    mut grammar_engine: Option<GrammarEngine>,
    adaptive_sampling: bool,
    mut session_manager: crate::session_manager::SessionSsmManager,
    spontaneous_think_budget: u32,
    // Resolved `--mtp-gate` value; `None` (the default) leaves the speculation
    // gate disarmed and the run byte-identical to before. (A plain comment,
    // not a doc comment: rustc rejects `///` on a function parameter.)
    mtp_gate_mode: Option<String>,
) {
    match spec_timing::configure(max_batch_size) {
        Ok(true) => tracing::info!("DFlash spec-cycle schema 2 timing armed (C=1, async=0)"),
        Ok(false) => {}
        Err(error) => tracing::error!("DFlash spec-cycle schema 2 timing disabled: {error}"),
    }
    model
        .bind_gpu_to_thread()
        .expect("Failed to bind CUDA context to scheduler thread");
    // Diagnostic/correctness escape hatch: keep the proposer and all of its
    // model-side buffers loaded, but route every request through the ordinary
    // decode scheduler.  This is deliberately evaluated once at scheduler
    // startup so it cannot change state-machine arms in the middle of a
    // sequence.
    let force_disable_speculation =
        std::env::var("ATLAS_DISABLE_SPECULATION").ok().as_deref() == Some("1");
    let use_mtp = use_speculative && model.has_proposer() && !force_disable_speculation;
    if force_disable_speculation && use_speculative && model.has_proposer() {
        tracing::warn!(
            "ATLAS_DISABLE_SPECULATION=1: proposer remains loaded; using ordinary decode scheduling"
        );
    }
    let num_drafts = if use_mtp || use_self_speculative || use_ngram_speculative {
        num_drafts.max(1)
    } else {
        0
    };
    let chunked = max_prefill_tokens > 0;
    let mut ngram_proposer = if use_ngram_speculative {
        Some(NgramProposer::new(4)) // 4-gram context
    } else {
        None
    };
    // Throughput-arbitrated speculation gate. `None` unless `--mtp-gate` was
    // passed, so the default path below is exactly the pre-gate `step_mtp`
    // call. `last_gate_arm` tracks the arm the PREVIOUS step actually ran, so
    // the transition cleanup fires on probe excursions too and not only on
    // committed switches.
    let primary_proposer_is_dflash = model.proposer_is_dflash();
    let mut spec_gate = build_speculation_gate(&*model, use_mtp, mtp_gate_mode.as_deref());
    let entry_pin =
        mtp_gate::parse_entry_pin_tokens(std::env::var("ATLAS_SPEC_ENTRY_PIN").ok().as_deref());
    if use_mtp {
        tracing::info!(
            env = "ATLAS_SPEC_ENTRY_PIN",
            tokens = entry_pin.tokens,
            source = entry_pin.source,
            gate_armed = spec_gate.is_some(),
            "spec-entry dispatch provenance"
        );
    }
    // Hoisted out of the step loop: the arm set is fixed at construction, so
    // "can this gate ever change what we run?" is a constant for the whole
    // serve. False for `--mtp-gate dflash|mtp` (a pinned arm) and for `auto`
    // on a build that only produced one proposer. When false the scheduler
    // skips ALL of the gate's measurement below — see the step dispatch.
    let gate_arbitrates = spec_gate.as_ref().is_some_and(MtpGate::arbitrates);
    let mut last_gate_arm: Option<usize> = None;
    // A Serial arm leaves the previously selected proposer installed. Keep its
    // own request width as well: the DFlash primary normally uses γ=15 while
    // the native MTP alternate uses one draft.
    let mut last_spec_num_drafts = num_drafts;
    let mut last_step_was_entry_pin = false;
    tracing::info!(
        "Scheduler started (batched mode, max_batch={max_batch_size}, mtp={}, ngram={}, num_drafts={num_drafts}, policy={}, chunked_prefill={}, max_prefill_tokens={})",
        use_mtp,
        use_ngram_speculative,
        policy.name(),
        chunked,
        if chunked { max_prefill_tokens } else { 0 },
    );

    let pending = Arc::new((
        Mutex::new(PendingQueue {
            requests: Vec::new(),
            closed: false,
        }),
        Condvar::new(),
    ));

    // Receiver thread: blocks on channel, signals scheduler via condvar.
    let p = Arc::clone(&pending);
    std::thread::spawn(move || {
        let mut rx = request_rx;
        while let Some(req) = rx.blocking_recv() {
            p.0.lock().requests.push(req);
            p.1.notify_one();
        }
        p.0.lock().closed = true;
        p.1.notify_one();
    });

    // Dedicated CUDA stream + event for prefill compute-copy overlap.
    let prefill_stream = model
        .create_stream()
        .expect("Failed to create prefill CUDA stream");
    let prefill_event = model
        .create_event()
        .expect("Failed to create prefill CUDA event");

    let mut active: Vec<ActiveSeq> = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = Vec::new();
    let mut swapped: Vec<SwappedSeq> = Vec::new();
    let mut spill_manager: Option<KvSpillManager> = if swap_space_gb > 0 {
        let max_bytes = swap_space_gb as u64 * 1024 * 1024 * 1024;
        match KvSpillManager::new(PathBuf::from("/tmp/atlas-swap"), max_bytes) {
            Ok(mgr) => {
                tracing::info!("Swap space: {swap_space_gb} GB at /tmp/atlas-swap/");
                Some(mgr)
            }
            Err(e) => {
                tracing::error!("Failed to initialize swap space: {e:#}");
                None
            }
        }
    } else {
        None
    };

    install_high_speed_swap(&*model, high_speed_swap_cfg);

    // Native MTP verifies DURING `<think>` spans by default. DFlash keeps the
    // ATLAS_THINK_SPEC=1 opt-in because its wide forward is a distinct
    // numerical path. The post-verify accept filter
    // (`think_spec_accept::dflash_thinking_accept`) re-derives the
    // plain-path token per position, so output stays byte-identical to
    // `step_decode_only`. Disqualified when the model emits fp32 decode
    // logits (the filter's row D2H assumes BF16). Adaptive sampling is
    // threaded through the same per-row function, including state updates.
    // FP32 verify logits remain serial because the current per-row oracle reads
    // BF16. `dflash_spec_think` is resolved once so request scheduling cannot
    // change if the environment is mutated mid-serve.
    let dflash_spec_think = think_spec_enabled();
    let think_ctx = ThinkSpecCtx {
        enabled: !model.decode_logits_fp32(),
        code_fence_token,
        reflection_suppress_ids: &reflection_suppress_ids,
        adaptive_sampling,
    };
    let mut snapshot_steps: u64 = 0;
    loop {
        // ── Drain pending → start prefill (chunked or full) ──
        let new_reqs =
            drain_pending_requests(&pending, &active, &prefilling, &*policy, max_batch_size);
        // Publish only when the dashboard is active. Plain/benchmark mode
        // pays one atomic branch and never takes the snapshot mutex.
        if crate::tui::is_active() {
            snapshot_steps += 1;
            let (mtp_mode, delivered_tps) = match spec_gate.as_ref() {
                Some(g) => g.observe(),
                None => (snapshot::MtpModeSnap::Off, 0.0),
            };
            snapshot::publish(snapshot::SchedulerSnapshot {
                active_seqs: active.len() as u32,
                prefilling_seqs: prefilling.len() as u32,
                swapped_seqs: swapped.len() as u32,
                pending_len: new_reqs.len() as u32,
                kv_blocks_free: model.num_free_blocks() as u32,
                kv_blocks_total: model.num_total_blocks() as u32,
                ssm_slots_used: session_manager.session_count() as u32,
                ssm_slots_total: session_manager.total_slots() as u32,
                mtp_mode,
                delivered_tps,
                steps_total: snapshot_steps,
                published_at: std::time::Instant::now(),
            });
        }
        if new_reqs.is_empty() && active.is_empty() && prefilling.is_empty() {
            // Receiver thread was closed (shutdown).
            let pending_closed = pending.0.lock().closed;
            if pending_closed {
                break;
            }
        }

        // ── Swap-out: evict active sequences to disk when blocks run low ──
        if let Some(ref mut spill) = spill_manager {
            for req in &new_reqs {
                let prompt_len = req.prompt_len();
                let blocks_needed = prompt_len / block_size + 1;
                while model.num_free_blocks() < blocks_needed && !active.is_empty() {
                    let victim_idx = active
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.grammar_state.is_none())
                        .max_by_key(|(_, a)| a.seq.block_table.len())
                        .map(|(i, _)| i);
                    let Some(victim_idx) = victim_idx else {
                        tracing::warn!("No swappable sequences (all grammar-active)");
                        break;
                    };
                    match swap_out_sequence(&*model, &mut active, victim_idx, spill) {
                        Ok(s) => {
                            tracing::info!(
                                "Swap-out: evicted seq (seq_len={}, blocks={}) to disk",
                                s.seq_len,
                                s.num_blocks,
                            );
                            swapped.push(s);
                        }
                        Err(e) => {
                            tracing::error!("Swap-out failed: {e:#}");
                            break;
                        }
                    }
                }
            }
        }

        // ── Start new requests ──
        start_new_requests(
            &*model,
            new_reqs,
            chunked,
            max_prefill_tokens,
            max_batch_tokens,
            &eos_tokens,
            prefill_stream,
            prefill_event,
            &mut grammar_engine,
            spontaneous_think_budget,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            &mut active,
            &mut prefilling,
        );

        // ── Continue in-progress prefills ──
        let did_mixed_step = continue_in_progress_prefills(
            &*model,
            &*policy,
            &mut active,
            &mut prefilling,
            max_prefill_tokens,
            prefill_stream,
            prefill_event,
            use_mtp,
            use_self_speculative,
            use_ngram_speculative,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            &reflection_suppress_ids,
            adaptive_sampling,
        );

        if active.is_empty() {
            continue;
        }

        // Retirement consumes this in lockstep with `active`. Ordinary,
        // bootstrap and serial finishes stay cacheable; `step_mtp` clears only
        // entries that finish inside an over-planned speculative verify frame.
        let mut cache_on_finish = vec![true; active.len()];

        // Skip decode when mixed_forward already processed decode logits.
        if !did_mixed_step {
            // Ensure any in-flight prefill work on the prefill stream is complete
            // before decode starts on the default stream.
            if !prefilling.is_empty() {
                let _ = model.record_event(prefill_event, prefill_stream);
                let _ = model.stream_wait_event(model.default_stream(), prefill_event);
            }

            if use_ngram_speculative && active.len() == 1 && active[0].grammar_state.is_none() {
                // N-gram speculative: CPU proposer + CUDA-graphed K=2 verify.
                if let Some(ref mut proposer) = ngram_proposer {
                    let was_verify = !active[0].pending_drafts.is_empty();
                    step_ngram(&*model, &mut active, proposer);
                    if was_verify && active[0].finished {
                        cache_on_finish[0] = false;
                    }
                }
            } else if use_self_speculative && active.len() == 1 && active[0].grammar_state.is_none()
            {
                // Self-speculative: draft via layer-skipping, verify with full model.
                step_self_spec(&*model, &mut active, num_drafts);
                if active[0].finished {
                    // The self-spec helper may have over-planned a dense verify
                    // frame before the terminal emission. Conservatively skip
                    // the final cache insert; sequence teardown is unchanged.
                    cache_on_finish[0] = false;
                }
            } else if use_mtp {
                let thinking_step_allowed = if let Some(gate) = spec_gate.as_ref() {
                    mtp_gate::arm_allows_thinking(
                        gate.next_arm().kind,
                        primary_proposer_is_dflash,
                        dflash_spec_think,
                        think_ctx.enabled,
                    )
                } else {
                    let current = ArmKind::Spec {
                        proposer_arm: if primary_proposer_is_dflash {
                            mtp_gate::PROPOSER_ARM_PRIMARY
                        } else {
                            model.proposer_arm()
                        },
                        draft_cap: 0,
                        num_drafts,
                    };
                    mtp_gate::arm_allows_thinking(
                        current,
                        primary_proposer_is_dflash,
                        dflash_spec_think,
                        think_ctx.enabled,
                    )
                };
                if active.iter().all(|a| {
                    // Native MTP may verify while thinking; DFlash requires
                    // ATLAS_THINK_SPEC=1. Both use the same post-verify policy
                    // oracle (see think_spec_accept.rs). suppress_tool_call /
                    // disable_mtp gates are unchanged.
                    (!a.inside_thinking || thinking_step_allowed)
                        && !a.suppress_tool_call
                        && !a.disable_mtp
                }) {
                    // MTP speculative decode for ALL active sequences.
                    //
                    // Concurrent-decode fix (2026-05-22): previously gated on
                    // `active.len() == 1`, which forced n>=2 into
                    // `step_decode_only`. That path runs the SSM
                    // `decode_multi_seq_inner` which sequentially calls
                    // per-seq `decode()` (see qwen3_ssm/trait_decode_multi_seq.rs
                    // — comment: "delegate to per-sequence single decode") with
                    // no CUDA graphs and no MTP, collapsing throughput to ~14
                    // tok/s aggregate at c=2 (down from 28 at c=1).
                    //
                    // `step_mtp` already loops bootstrap+verify across all
                    // active sequences (see mtp_step.rs lines 21,96). Each
                    // sequence reuses its own per-slot CUDA graph for K=3
                    // verify (graph cache is keyed on `seq.slot_idx` in
                    // verify_c.rs), so n>=2 captures one graph per slot on
                    // first iteration and replays on every subsequent step.
                    //
                    // Per-seq guards (inside_thinking / suppress_tool_call /
                    // disable_mtp) are checked across ALL active sequences
                    // because step_mtp doesn't conditionally bootstrap per
                    // active flag; if any seq disables MTP we fall back to
                    // batched decode_only for the whole batch this tick.
                    //
                    // ── Speculation gate (`--mtp-gate`) ──
                    //
                    // When armed, the gate picks which arm to run for THIS step —
                    // the external DFlash/DDTree drafter, the native MTP head, a
                    // γ-capped drafter, or plain decode — by comparing DELIVERED
                    // tok/s per arm over 16-step windows with hysteresis and
                    // periodic probing. Every arm emits real, correct tokens, so
                    // arbitration never wastes work. See `mtp_gate` for why this
                    // measures throughput and not draft acceptance.
                    //
                    // Disarmed (the default) this is exactly the previous call.
                    if let Some(gate) = spec_gate.as_mut() {
                        let arm_idx = gate.next_arm_index();
                        let arm = gate.next_arm();
                        let pin_active = matches!(arm.kind, ArmKind::Serial)
                            && mtp_gate::entry_pin_active(
                                entry_pin.tokens,
                                active.iter().map(|a| {
                                    (a.think_ended, a.inside_thinking, a.post_think_gate_steps)
                                }),
                            );
                        let pinned_spec_width = mtp_gate::entry_pin_spec_width(
                            arm.kind,
                            pin_active,
                            last_spec_num_drafts,
                        );
                        // Everything the gate measures exists to feed arbitration,
                        // so a gate that cannot arbitrate must not pay for it. A
                        // PINNED arm therefore skips the depth scan, both
                        // `seq_len` scans, the `Instant` pair and the window
                        // accounting, and what remains between it and the disarmed
                        // path is one `Option` compare plus a `match` on a value
                        // that never changes.
                        //
                        // `Some(before)` doubles as the "measure this step" flag.
                        let measure = if gate_arbitrates && pinned_spec_width.is_none() {
                            // One pass, two answers. Depth is the MAX `seq_len`
                            // (arm economics are depth-dependent: weight-bound at
                            // short context, KV/SSM-bound at depth). Delivered
                            // tokens are the SUM — counting only `active[0]` would
                            // under-report a speculative arm by a factor of n and
                            // bias the gate toward whichever arm is cheapest per
                            // step rather than fastest per token.
                            let (depth, before) =
                                active.iter().fold((0usize, 0usize), |(d, s), a| {
                                    (d.max(a.seq.seq_len), s + a.seq.seq_len)
                                });
                            gate.observe_depth(depth);
                            Some(before)
                        } else {
                            None
                        };

                        let arm_changed = last_gate_arm != Some(arm_idx);
                        if arm_changed {
                            // Arm transition. Drafts in flight belong to the
                            // OUTGOING arm and would be verified against the
                            // incoming one's assumptions, so drop them (and the
                            // paired DDTree payload) and order the previous
                            // verify's async live-state restore before the next
                            // step reads h_state/conv_state. The next speculative
                            // step then re-enters through `step_mtp`'s bootstrap
                            // phase, whose `trim_proposer_state(seq, 0, 0)` resets
                            // the drafter's RoPE conditioning — which is what
                            // makes re-entry lossless.
                            for a in active.iter_mut() {
                                a.pending_drafts.clear();
                                a.pending_tree_payload = None;
                            }
                            if let Err(e) = model.sync_secondary() {
                                tracing::error!("gate arm switch sync_secondary: {e:#}");
                            }
                            select_gate_arm(&*model, &mut active, arm);
                            tracing::debug!(
                                "speculation gate: running arm {} ({})",
                                arm_idx,
                                gate.arm_name(arm_idx)
                            );
                            last_gate_arm = Some(arm_idx);
                        } else if mtp_gate::entry_pin_exits_to_serial(
                            arm.kind,
                            last_step_was_entry_pin,
                            pinned_spec_width,
                        ) {
                            // The gate stayed on Serial while the entry pin ran the
                            // installed proposer, so no arm transition exists to
                            // drop that last pinned step's drafts or order its
                            // async recurrent-state restore.
                            for a in active.iter_mut() {
                                a.pending_drafts.clear();
                                a.pending_tree_payload = None;
                            }
                            if let Err(e) = model.sync_secondary() {
                                tracing::error!("entry-pin→serial sync_secondary: {e:#}");
                            }
                        }

                        // Taken as late as possible so the one-shot transition
                        // work above is not charged to the arm being switched TO.
                        let t0 = measure.map(|_| Instant::now());
                        if let Some(nd) = pinned_spec_width {
                            // Entry pins are correctness dispatches, not samples of
                            // the Serial arm. Run the proposer which remains selected
                            // across the serial dwell and leave gate statistics
                            // untouched. `step_mtp`'s post-think policy gate routes
                            // both native short drafts and DFlash γ blocks through
                            // the row-by-row sampler oracle.
                            step_mtp(&*model, &mut active, nd, &think_ctx, &mut cache_on_finish);
                        } else {
                            match arm.kind {
                                ArmKind::Serial => {
                                    step_decode_only(
                                        &*model,
                                        &mut active,
                                        think_end_token,
                                        think_start_token,
                                        code_fence_token,
                                        tool_call_start_token,
                                        tool_call_end_token,
                                        &reflection_suppress_ids,
                                        adaptive_sampling,
                                    );
                                }
                                ArmKind::Spec {
                                    num_drafts: arm_drafts,
                                    ..
                                } => {
                                    // 0 means "the run's configured value" — the
                                    // DFlash arm — so only the alternate arm departs
                                    // from it.
                                    let nd = if arm_drafts == 0 {
                                        num_drafts
                                    } else {
                                        arm_drafts
                                    };
                                    last_spec_num_drafts = nd;
                                    step_mtp(
                                        &*model,
                                        &mut active,
                                        nd,
                                        &think_ctx,
                                        &mut cache_on_finish,
                                    );
                                }
                            }
                        }
                        if let (Some(before), Some(t0)) = (measure, t0) {
                            let after: usize = active.iter().map(|a| a.seq.seq_len).sum();
                            gate.record_step(t0.elapsed(), after.saturating_sub(before));
                            // Drained rather than acted on: the arm change is
                            // applied from `next_arm_index` at the top of the next
                            // step (which also covers probe excursions, where no
                            // "switch" is reported). Taking it here keeps the
                            // one-shot from going stale and gives the switch a
                            // single owner. Only reachable when arbitrating —
                            // a pinned gate never produces a switch to drain.
                            let _ = gate.take_fresh_switch();
                        }
                        last_step_was_entry_pin = pinned_spec_width.is_some();
                    } else {
                        step_mtp(
                            &*model,
                            &mut active,
                            num_drafts,
                            &think_ctx,
                            &mut cache_on_finish,
                        );
                    }
                } else {
                    // Fall through to the ordinary decode path below.
                    if use_mtp {
                        for a in active.iter_mut() {
                            a.pending_drafts.clear();
                            a.pending_tree_payload = None;
                        }
                        if let Err(e) = model.sync_secondary() {
                            tracing::error!("mtp→decode sync_secondary: {e:#}");
                        }
                        last_step_was_entry_pin = false;
                    }
                    step_decode_only(
                        &*model,
                        &mut active,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        &reflection_suppress_ids,
                        adaptive_sampling,
                    );
                }
            } else {
                // Batch decode (no MTP). Clear stale drafts when transitioning out of MTP mode.
                if use_mtp {
                    for a in active.iter_mut() {
                        a.pending_drafts.clear();
                        // `pending_tree_payload` is paired with `pending_drafts`
                        // (types.rs:194) and mtp_step.rs:49-50 always clears the
                        // two together. Clearing only the drafts here let a stale
                        // DDTree payload survive an MTP→decode→MTP round trip and
                        // be applied against a different `seq_len`.
                        a.pending_tree_payload = None;
                    }
                    // MTP→decode-only transition: the last verify commit's
                    // live-state restore runs async on the secondary stream;
                    // order it before this decode reads h_state/conv_state
                    // (GPU-side event wait, zero CPU cost). Every other verify
                    // path already does this (verify_k3_step.rs:9,
                    // verify_k4_step.rs:10, verify_dflash_batched_step.rs:43,
                    // spec_step.rs:227, verify_csk_step_k2.rs:38) — this exit
                    // edge was the one that did not.
                    if let Err(e) = model.sync_secondary() {
                        tracing::error!("mtp→decode sync_secondary: {e:#}");
                    }
                    last_step_was_entry_pin = false;
                }
                step_decode_only(
                    &*model,
                    &mut active,
                    think_end_token,
                    think_start_token,
                    code_fence_token,
                    tool_call_start_token,
                    tool_call_end_token,
                    &reflection_suppress_ids,
                    adaptive_sampling,
                );
            }
        }

        retire_finished_sequences(&*model, &mut active, &mut cache_on_finish);

        // ── Swap-in: resume swapped sequences when blocks free up ──
        if let Some(ref mut spill) = spill_manager {
            let mut resumed_any = true;
            while resumed_any && !swapped.is_empty() && active.len() < max_batch_size {
                resumed_any = false;
                let free = model.num_free_blocks();
                if let Some(idx) = swapped.iter().position(|s| s.num_blocks <= free) {
                    let s = swapped.remove(idx);
                    match resume_swapped_seq(think_end_token, think_start_token, &*model, s, spill)
                    {
                        Ok(a) => {
                            tracing::info!(
                                "Swap-in: restored seq (seq_len={}, blocks={})",
                                a.seq.seq_len,
                                a.seq.block_table.len(),
                            );
                            active.push(a);
                            resumed_any = true;
                        }
                        Err(e) => {
                            tracing::error!("Swap-in failed: {e:#}");
                        }
                    }
                }
            }
        }
    }

    // Periodic session eviction: free SSM snapshots for expired sessions.
    {
        let freed_slots = session_manager.evict_expired();
        if !freed_slots.is_empty() {
            tracing::info!(
                "Session eviction: freed {} SSM snapshot slot(s), {} sessions active",
                freed_slots.len(),
                session_manager.session_count()
            );
        }
    }

    // Drain any remaining active sequences on shutdown.
    for mut a in active {
        finish_sequence(&*model, &mut a);
    }
    if let Some(ref mut spill) = spill_manager {
        for s in swapped {
            let _ = spill.remove_file(s.swap_id);
        }
    }
    for p in prefilling {
        let mut seq = p.seq;
        let _ = model.free_sequence(&mut seq);
        let _ = model.ep_broadcast_cmd(0xFFFFFFF1);
    }
    let _ = model.ep_broadcast_cmd(0xFFFFFFFF);
    tracing::info!("Scheduler stopped");
}
