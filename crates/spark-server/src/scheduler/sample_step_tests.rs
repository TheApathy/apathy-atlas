// SPDX-License-Identifier: AGPL-3.0-only

//! The first sampled token honors the request's full sampler settings. Each test's control is
//! what the first-token path used to build: temperature/top-k/top-p and nothing else.

use super::{needs_sampler, sample_host_logits};
use spark_runtime::sampler::SamplingParams;

fn params(temperature: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: None,
    }
}

/// The parameters the first-token path built before the fix, from a request's `full` set.
fn old_first_token(full: &SamplingParams) -> SamplingParams {
    SamplingParams { seed: full.seed, ..params(full.temperature) }
}

/// A logit_bias that forbids the argmax is honored at position 0, greedy.
#[test]
fn logit_bias_is_honored_on_the_first_token() {
    let logits = vec![1.0, 2.0, 5.0, 3.0];
    let full = SamplingParams { logit_bias: vec![(2, -100.0)], ..params(0.0) };
    assert!(needs_sampler(&full, &[]), "a bias must leave the device argmax fast path");
    assert_eq!(sample_host_logits(&mut logits.clone(), &full, &[], &[]), 3);
    // Control: the old first-token parameters pick the forbidden token.
    assert_eq!(sample_host_logits(&mut logits.clone(), &old_first_token(&full), &[], &[]), 2);
}

/// top_n_sigma filters a token the request excluded; the old path samples it.
#[test]
fn top_n_sigma_is_honored_on_the_first_token() {
    // Nine tokens at 5.0 and a probe at 3.5: sigma 0.45, so the 1-sigma cut (max - sigma =
    // 4.55) drops the probe, which plain T=1 picks ~2.4% of the time.
    let mut logits = vec![5.0f32; 10];
    logits[9] = 3.5;
    let picks = |p: &SamplingParams| -> Vec<u32> {
        (0..1000u64)
            .map(|seed| {
                let p = SamplingParams { seed: Some(seed), ..p.clone() };
                sample_host_logits(&mut logits.clone(), &p, &[], &[])
            })
            .collect()
    };
    let full = SamplingParams { top_n_sigma: 1.0, ..params(1.0) };
    assert!(!picks(&full).contains(&9), "sigma must drop the probe");
    assert!(picks(&old_first_token(&full)).contains(&9), "control: the old path samples the probe");
}

/// Plain greedy with no bias stays on the fast path; penalties only matter once there is
/// history for them to act on.
#[test]
fn only_settings_that_can_move_the_argmax_leave_the_fast_path() {
    let penalised = SamplingParams { repetition_penalty: 1.1, ..params(0.0) };
    assert!(!needs_sampler(&params(0.0), &[]));
    assert!(!needs_sampler(&penalised, &[]));
    assert!(needs_sampler(&penalised, &[7]));
    assert!(needs_sampler(&params(0.7), &[]));
}
