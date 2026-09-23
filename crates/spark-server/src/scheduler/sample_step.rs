// SPDX-License-Identifier: AGPL-3.0-only

//! Token sampling helpers (resample + sample + grammar-constrained sample).

use super::*;

/// A request's sampler settings for its first token, as decode applies them to every later
/// one (`ActiveSeq::sampling_params`). The first token used to get temperature/top-k/top-p
/// only, so logit_bias, top_n_sigma, min_p and the seed were ignored at position 0.
/// Penalties see no history yet; they are carried so the two paths cannot drift.
pub(super) fn request_sampling_params(req: &InferenceRequest) -> SamplingParams {
    let temperature = match req {
        InferenceRequest::Blocking { temperature, .. } => *temperature,
        InferenceRequest::Streaming { temperature, .. } => *temperature,
    };
    SamplingParams {
        temperature,
        top_k: req.top_k(),
        top_p: req.top_p(),
        top_n_sigma: req.top_n_sigma(),
        min_p: req.min_p(),
        logit_bias: req.logit_bias().to_vec(),
        repetition_penalty: req.repetition_penalty(),
        repetition_penalty_window: 256,
        presence_penalty: req.presence_penalty(),
        frequency_penalty: req.frequency_penalty(),
        lz_penalty: req.lz_penalty(),
        dry_multiplier: req.dry_multiplier(),
        dry_base: req.dry_base(),
        dry_allowed_length: req.dry_allowed_length(),
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: req.seed(),
    }
}

impl PrefillInProgress {
    /// [`request_sampling_params`] for a chunked prefill's first token.
    pub(super) fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            top_n_sigma: self.top_n_sigma,
            min_p: self.min_p,
            logit_bias: self.logit_bias.clone(),
            repetition_penalty: self.repetition_penalty,
            repetition_penalty_window: self.repetition_penalty_window,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            lz_penalty: self.lz_penalty,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_sequence_breakers: self.dry_sequence_breakers.clone(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: self.seed,
        }
    }
}

impl ActiveSeq {
    /// The sampler settings for this sequence's next token at `temperature`.
    ///
    /// Phase-gated (P3.1, 2026-04-25): inside the tool-call body (between `<tool_call>` and
    /// `</tool_call>`) the JSON is dense with legitimate short repetitions — `":"`, `","`,
    /// key tokens — that DRY/presence/frequency penalties would punish, breaking schema
    /// validity. XGrammar already guarantees the structure there, so the penalties are off;
    /// outside the body (free text + `<think>`), where prose loops live, the full preset
    /// applies. LZ is off whenever a grammar is active.
    pub(super) fn sampling_params(&self, temperature: f32) -> SamplingParams {
        let in_tool = self.inside_tool_body && !self.inside_thinking;
        SamplingParams {
            temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            top_n_sigma: self.top_n_sigma,
            min_p: self.min_p,
            logit_bias: self.logit_bias.clone(),
            repetition_penalty: if in_tool { 1.0 } else { self.repetition_penalty },
            repetition_penalty_window: self.repetition_penalty_window,
            presence_penalty: if in_tool { 0.0 } else { self.presence_penalty },
            frequency_penalty: if in_tool { 0.0 } else { self.frequency_penalty },
            lz_penalty: if self.grammar_state.is_some() { 0.0 } else { self.lz_penalty },
            dry_multiplier: if in_tool { 0.0 } else { self.dry_multiplier },
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_sequence_breakers: self.dry_sequence_breakers.clone(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            // Advance the seed per token: deterministic but varying.
            seed: self.seed.map(|s| s.wrapping_add(self.output_tokens.len() as u64)),
        }
    }
}

/// Whether sampling `params` after `history` can pick something other than the raw argmax
/// even at temperature 0: a logit bias, or a penalty with history to act on.
fn needs_sampler(params: &SamplingParams, history: &[u32]) -> bool {
    let penalties = (params.repetition_penalty != 1.0 && params.repetition_penalty > 0.0)
        || params.presence_penalty != 0.0
        || params.frequency_penalty != 0.0
        || params.lz_penalty != 0.0
        || params.dry_multiplier != 0.0;
    params.temperature != 0.0 || !params.logit_bias.is_empty() || (penalties && !history.is_empty())
}

/// Sample one token from device logits with the request's full sampler settings (`params`),
/// penalties acting on `history` (the tokens generated so far).
///
/// `suppress_ids`: token IDs to mask to -inf before sampling (e.g. EOS on first token).
pub fn sample_token(
    model: &dyn Model,
    logits: DevicePtr,
    params: &SamplingParams,
    suppress_ids: &[u32],
    history: &[u32],
) -> Result<u32> {
    if !needs_sampler(params, history) && suppress_ids.is_empty() {
        return model.argmax_on_device(logits, 0);
    }
    let vocab_size = model.vocab_size();
    // Read logits from device. Gemma-4 dense single-token decode produces FP32
    // logits via the FP32 lm_head + softcap path (margin between top-1 and
    // top-2 sits on a BF16 representable boundary at value 16-32, so storing
    // BF16 there flips the greedy argmax). Other paths still produce BF16
    // and need expansion. Dispatch by `logits_ptr_is_fp32`.
    let mut f32_logits: Vec<f32> = if model.logits_ptr_is_fp32(logits) {
        let mut buf = vec![0u8; vocab_size * 4];
        model.copy_logits_to_host(logits, &mut buf)?;
        // SAFETY: buf has length vocab_size * 4 and the device kernel wrote
        // little-endian f32 values; reinterpret is byte-equivalent on x86/arm.
        let f32_slice: &[f32] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, vocab_size) };
        f32_slice.to_vec()
    } else {
        let mut bf16_buf = vec![0u8; vocab_size * 2];
        model.copy_logits_to_host(logits, &mut bf16_buf)?;
        (0..vocab_size)
            .map(|i| {
                let lo = bf16_buf[i * 2];
                let hi = bf16_buf[i * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };
    Ok(sample_host_logits(&mut f32_logits, params, suppress_ids, history))
}

/// The host half of [`sample_token`]: mask `suppress_ids`, then sample with the full
/// `params`, or take the plain argmax when nothing in them can move it.
fn sample_host_logits(
    f32_logits: &mut [f32],
    params: &SamplingParams,
    suppress_ids: &[u32],
    history: &[u32],
) -> u32 {
    // Suppress EOS tokens on first token by setting to -inf.
    for &id in suppress_ids {
        if let Some(l) = f32_logits.get_mut(id as usize) {
            *l = f32::NEG_INFINITY;
        }
    }
    if !needs_sampler(params, history) {
        // Greedy argmax over FP32
        return f32_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    // SAFETY: an f32 slice viewed as its bytes; same length in bytes, no alignment demand.
    let f32_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, f32_logits.len() * 4)
    };
    sample_with_params_history(f32_bytes, params, history)
}

/// Sample one token from device logits with optional grammar constraint.
///
/// Like `sample_token` but also applies grammar bitmask when `grammar_state`
/// is provided. Always uses host-side sampling when grammar is active (can't
/// use GPU argmax since grammar bitmask is CPU-side).
pub fn sample_token_with_grammar(
    model: &dyn Model,
    logits: DevicePtr,
    params: &SamplingParams,
    suppress_ids: &[u32],
    history: &[u32],
    grammar_state: Option<&mut GrammarState>,
) -> Result<u32> {
    let Some(gs) = grammar_state else {
        return sample_token(model, logits, params, suppress_ids, history);
    };

    // ── Tier 3b: forced-token short-circuit (xgrammar "Coalescence") ──
    //
    // When the grammar admits exactly one legal next token (very common in
    // JSON tool-call syntax: literal `{`, key strings, `:`, `,`, closing
    // braces), xgrammar can compute the token directly without filling a
    // vocab-wide bitmask. Skip the D2H + BF16→f32 + mask + argmax loop
    // entirely — saves ~5 ms per forced position. On a typical tool-call
    // response, ~30-50% of positions are forced.
    if let Some(forced) = gs.forced_token() {
        return Ok(forced as u32);
    }

    let vocab_size = model.vocab_size();
    let mut bf16_buf = vec![0u8; vocab_size * 2];
    model.copy_logits_to_host(logits, &mut bf16_buf)?;
    gs.fill_bitmask();
    let bitmask = gs.bitmask_data();

    // ── Greedy fused fast path (temperature == 0) ──
    //
    // Earlier implementation made THREE sequential passes over `vocab_size`:
    // (1) BF16→f32 conversion, (2) apply_bitmask_to_logits, (3) max_by scan
    // with f32 partial_cmp — about 5 ms wall on the 248k-vocab aeon-ultimate
    // model. We fuse them into ONE pass, comparing BF16 values directly as
    // signed i16 (which preserves the natural ordering of finite BF16
    // values) so no f32 scratch buffer is needed. Plus we apply suppress_ids
    // post-hoc since they're typically a handful of token IDs.
    if !needs_sampler(params, history) {
        let bytes: &[u8] = &bf16_buf;
        let mut best_tok: u32 = 0;
        let mut best_val: i16 = i16::MIN;
        for tok in 0..vocab_size {
            let word = tok / 32;
            let bit = tok % 32;
            if word >= bitmask.len() || (bitmask[word] & (1i32 << bit)) == 0 {
                continue;
            }
            // Reinterpret BF16 bit pattern as signed i16 for total ordering
            // over finite values. Suppress-ids are filtered post-loop.
            let hi = u16::from_le_bytes([bytes[2 * tok], bytes[2 * tok + 1]]);
            let signed = hi as i16;
            if signed > best_val {
                best_val = signed;
                best_tok = tok as u32;
            }
        }
        // Suppress-id post-filter: rare hit path, recompute argmax only when
        // a suppressed token was chosen. Cheaper than per-token suppress
        // check inside the hot loop above.
        if suppress_ids.contains(&best_tok) {
            best_val = i16::MIN;
            best_tok = 0;
            for tok in 0..vocab_size {
                if suppress_ids.contains(&(tok as u32)) {
                    continue;
                }
                let word = tok / 32;
                let bit = tok % 32;
                if word >= bitmask.len() || (bitmask[word] & (1i32 << bit)) == 0 {
                    continue;
                }
                let hi = u16::from_le_bytes([bytes[2 * tok], bytes[2 * tok + 1]]);
                let signed = hi as i16;
                if signed > best_val {
                    best_val = signed;
                    best_tok = tok as u32;
                }
            }
        }
        return Ok(best_tok);
    }

    // Stochastic sampling path: needs f32 logits for the sampler. Keep the
    // original BF16→f32 + apply_bitmask_to_logits pattern here; this path
    // is rare in greedy production usage and the dispatching overhead is
    // dominated by `sample_with_params` regardless.
    let mut f32_logits: Vec<f32> = (0..vocab_size)
        .map(|i| {
            let lo = bf16_buf[i * 2];
            let hi = bf16_buf[i * 2 + 1];
            bf16_to_f32(lo, hi)
        })
        .collect();
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            f32_logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    gs.apply_bitmask_to_logits(&mut f32_logits);
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    Ok(sample_with_params_history(f32_bytes, params, history))
}

#[cfg(test)]
#[path = "sample_step_tests.rs"]
mod tests;
