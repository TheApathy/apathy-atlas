// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits: post-decode logits processing.

use super::*;

thread_local! {
    /// Reusable host staging buffer for the D2H logits copy on the sampling
    /// path. Hoisted out of the per-token `vec![0u8; n*vocab*elem]` to avoid an
    /// mmap/munmap + page-fault cycle every decoded token (the buffer is
    /// ~0.5-1 MB at a 250k vocab). Fully overwritten by `copy_logits_to_host`,
    /// so residual contents are irrelevant. Per-thread: the scheduler drives
    /// decode on one thread.
    static DECODE_LOGITS_HOST_SCRATCH: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// DIAG (ATLAS_DECODE_TIMING=1): localize the host-path decode cost. Splits the
/// per-token wall into `copy` (D2H of the full 248k-vocab logits + the GPU
/// forward-wait absorbed by that sync) vs `sample` (the host scalar loops over
/// 248k: BF16→FP32 expand + penalties + masks + argmax). Emits a 100-token
/// running summary. Zero-cost when the env var is unset (OnceLock-gated).
fn decode_timing_record(copy_us: u64, sample_us: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("ATLAS_DECODE_TIMING").is_ok()) {
        return;
    }
    static COPY: AtomicU64 = AtomicU64::new(0);
    static SAMPLE: AtomicU64 = AtomicU64::new(0);
    static CNT: AtomicU64 = AtomicU64::new(0);
    COPY.fetch_add(copy_us, Ordering::Relaxed);
    SAMPLE.fetch_add(sample_us, Ordering::Relaxed);
    let n = CNT.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(100) {
        let c = COPY.swap(0, Ordering::Relaxed);
        let s = SAMPLE.swap(0, Ordering::Relaxed);
        CNT.store(0, Ordering::Relaxed);
        tracing::info!(
            "DECODE_TIMING (last 100 host-path tokens): copy+fwd-wait={:.2}ms/tok sample(248k host)={:.2}ms/tok",
            c as f64 / 100_000.0,
            s as f64 / 100_000.0,
        );
    }
}

/// Sample and process decode logits for all active sequences.
///
/// Factored out of `step_decode_only` so that `mixed_forward` can reuse
/// the same sampling + token-processing logic without duplication (SSOT).
/// `logits` must point to `[n, vocab_size]` BF16 on device where n = active.len().
pub fn process_decode_logits(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    logits: DevicePtr,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
) {
    if let Err(error) = process_decode_logits_slice(
        model,
        active,
        logits,
        t0,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        adaptive_sampling,
    ) {
        for mut a in active.drain(..) {
            send_error(model, &mut a, &format!("{error:#}"));
        }
    }
}

/// Borrowed ordinary entry for a model-restored singleton replay. The caller
/// retains sequence/retirement ownership on transport errors; policy and live
/// transition behavior are identical to the normal Vec entry above.
pub(super) fn process_decode_logits_slice(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    logits: DevicePtr,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
) -> Result<()> {
    let n = active.len();

    let greedy_tokens = match super::ordinary_greedy::try_pick_batch(
        model,
        active,
        logits,
        &super::logit_processors::LogitsContext {
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
        },
        adaptive_sampling,
    ) {
        Ok(tokens) => tokens,
        Err(e) => {
            tracing::error!("ordinary greedy admission error: {e:#}");
            return Err(e);
        }
    };

    let new_tokens: Vec<(u32, Option<crate::api::TokenLogprobs>)> =
        if let Some(tokens) = greedy_tokens {
            // Neutral or proven-immune greedy picks, validated for the whole batch.
            tokens.into_iter().map(|tok| (tok, None)).collect()
        } else {
            // Host-side path: copy all batch logits to host, sample per-sequence.
            // Required when any sequence has temperature > 0 or grammar constraints.
            let vocab_size = model.vocab_size();
            // FP32 lm_head dispatch (Gemma-4 dense + ATLAS_GEMMA4_FP32_LMHEAD=1).
            // When the model writes FP32 logits to its decode-logits buffer, we
            // copy 4 bytes/element and skip the BF16→FP32 expansion. Earlier
            // bisection at model.rs:1192-1201 incorrectly concluded FP32 lm_head
            // had no effect on Gemma-4 because this dispatch was never wired —
            // the scheduler always read the (stale) BF16 logits buffer.
            // FP32 lm_head dispatch (Gemma-4 dense). When `use_fp32_logits` is
            // on, the per-token decode lm_head writes 4 bytes/element. The
            // passed `logits` pointer is whatever the most-recent forward
            // returned — that's already the correct buffer (prefill or decode).
            // We just need to read it with the matching width.
            let logits_fp32 = model.decode_logits_fp32();
            let elem_bytes = if logits_fp32 { 4 } else { 2 };
            let t_copy = std::time::Instant::now();
            // Reuse the per-thread staging buffer (restored at the end of this
            // block). `resize` only grows it; `copy_logits_to_host` overwrites
            // every byte so the residual/zero-fill is irrelevant.
            let mut buf = DECODE_LOGITS_HOST_SCRATCH.with_borrow_mut(std::mem::take);
            buf.resize(n * vocab_size * elem_bytes, 0);
            if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
                tracing::error!("copy_logits_to_host error: {e:#}");
                return Err(e);
            }
            let copy_us = t_copy.elapsed().as_micros() as u64;
            // SSOT: build the same `LogitsContext` the verify path passes
            // into `run_pipeline`, so `process_seq_logits` and the MTP
            // verify path share one pipeline-stage signature instead of
            // two divergent arg lists. `think_start_token` lives on the
            // per-seq `ActiveSeq` (read inside the pipeline stages), so it
            // is intentionally not carried in the context.
            let ctx = crate::scheduler::logit_processors::LogitsContext {
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
            };
            let t_sample = std::time::Instant::now();
            let sampled: Vec<(u32, Option<crate::api::TokenLogprobs>)> = active
                .iter_mut()
                .enumerate()
                .map(|(i, a)| {
                    process_seq_logits(
                        model,
                        a,
                        &buf,
                        i,
                        vocab_size,
                        elem_bytes,
                        logits_fp32,
                        &ctx,
                        adaptive_sampling,
                    )
                })
                .collect();
            decode_timing_record(copy_us, t_sample.elapsed().as_micros() as u64);
            // Return the staging buffer for reuse next token (its capacity is
            // preserved). The error path above intentionally drops it — that is
            // rare and only forfeits the cached capacity.
            DECODE_LOGITS_HOST_SCRATCH.with_borrow_mut(|slot| *slot = buf);
            sampled
        };
    let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let token_ids: Vec<u32> = new_tokens.iter().map(|(t, _)| *t).collect();
        tracing::debug!(
            "DECODE: n={n} step={step_ms:.1}ms ({:.1} tok/s) tokens={:?}",
            1000.0 * n as f64 / step_ms,
            token_ids,
        );
    }

    let now = Instant::now();
    let context = super::ordinary_transition::TransitionContext {
        logits: super::logit_processors::LogitsContext {
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
        },
        code_fence_token,
        max_seq_len: max_seq_len_ceiling(),
        now,
    };
    for (i, (tok, logprobs)) in new_tokens.into_iter().enumerate() {
        let a = &mut active[i];
        let mut effects = super::ordinary_transition::LiveEffects { model };
        if let Err(error) =
            super::ordinary_transition::advance_ordinary(a, tok, logprobs, &context, &mut effects)
        {
            mark_sequence_error(a, "ordinary decode transition", &error);
        }
    }
    Ok(())
}
