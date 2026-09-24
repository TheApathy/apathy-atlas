// SPDX-License-Identifier: AGPL-3.0-only

//! Sampled (T > 0) DSpark acceptance: speculative rejection sampling, exactly as the Python
//! engine's non-lean loop (`engine/v41_engine.py:936-968`):
//!
//! ```text
//! for i in 0..B:
//!     accept d_i  iff  u_i < min(1, p_i(d_i) / max(q_i(d_i), 1e-20))
//!     else: next = sample(normalize(max(p_i - q_i, 0)))   (p_i itself if that is all zero); stop
//! all accepted: next = sample(p_B)
//! ```
//!
//! `p_i` is the verify row's distribution after the request's sampler transform (temperature,
//! top_p, penalties, grammar); `q_i` is the draft distribution the draft was sampled from
//! (`softmax(markov_logits / T)`, no top_p, as Python). The emitted tokens are then distributed
//! exactly as target-model sampling -- the property the unit tests below check statistically.
//!
//! Pure host code over probability vectors: the scheduler owns the transform and the RNG.

use anyhow::{Result, ensure};

/// Outcome of one sampled speculative step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledAccept {
    /// Leading drafts accepted (0..=B).
    pub accepted: usize,
    /// The token after the accepted drafts (residual sample on a rejection, else from p_B).
    pub next: u32,
}

/// Inverse-CDF sample of `w` (non-negative weights, need not be normalised) at uniform `u` in [0,1).
pub fn sample_weights(w: &[f32], u: f64) -> u32 {
    let total: f64 = w.iter().map(|&x| x as f64).sum();
    let target = u * total;
    let mut acc = 0.0f64;
    let mut last_nonzero = 0u32;
    for (i, &x) in w.iter().enumerate() {
        if x > 0.0 {
            last_nonzero = i as u32;
        }
        acc += x as f64;
        if acc > target {
            return i as u32;
        }
    }
    last_nonzero
}

/// Speculative rejection sampling over one verify block.
///
/// * `p` — B+1 target rows (row i = distribution after [token, d_1..d_i]); each sums to ~1.
/// * `q` — B draft rows (the distributions d_1..d_B were sampled from).
/// * `drafts` — B draft ids.
/// * `u_accept` — B uniforms for the accept tests; `u_next` — one uniform for the final sample.
pub fn speculative_accept(p: &[&[f32]], q: &[&[f32]], drafts: &[u32], u_accept: &[f64], u_next: f64) -> Result<SampledAccept> {
    let b = drafts.len();
    ensure!(p.len() == b + 1 && q.len() == b && u_accept.len() == b, "speculative_accept: shapes p {} q {} drafts {b} u {}", p.len(), q.len(), u_accept.len());
    for i in 0..b {
        let d = drafts[i] as usize;
        let (pi, qi) = (p[i], q[i]);
        ensure!(d < pi.len() && pi.len() == qi.len(), "speculative_accept: draft {d} / vocab {} {}", pi.len(), qi.len());
        let ratio = (pi[d] as f64 / (qi[d] as f64).max(1e-20)).min(1.0);
        if u_accept[i] < ratio {
            continue;
        }
        let resid: Vec<f32> = pi.iter().zip(qi).map(|(&a, &c)| (a - c).max(0.0)).collect();
        let next = if resid.iter().map(|&x| x as f64).sum::<f64>() > 0.0 { sample_weights(&resid, u_next) } else { sample_weights(pi, u_next) };
        return Ok(SampledAccept { accepted: i, next });
    }
    Ok(SampledAccept { accepted: b, next: sample_weights(p[b], u_next) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 -> uniform [0,1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn norm(v: &[f32]) -> Vec<f32> {
        let s: f32 = v.iter().sum();
        v.iter().map(|x| x / s).collect()
    }

    /// Pearson chi-square of `counts` against probabilities `p` (categories with p > 0).
    fn chi2(counts: &[u64], p: &[f32], n: u64) -> (f64, usize) {
        let mut stat = 0.0;
        let mut k = 0;
        for (c, &pi) in counts.iter().zip(p) {
            if pi > 0.0 {
                let e = n as f64 * pi as f64;
                stat += (*c as f64 - e).powi(2) / e;
                k += 1;
            }
        }
        (stat, k - 1)
    }

    // chi-square critical values at p = 0.01 for df = 7 (8 categories).
    const CRIT_DF7_P01: f64 = 18.475;

    /// Run `n` single-draft steps (B = 1); return the histogram of the FIRST emitted token.
    fn first_token_hist(p0: &[f32], p1: &[f32], q0: &[f32], n: u64, accept_always: bool, seed: u64) -> Vec<u64> {
        let mut rng = Rng(seed);
        let mut h = vec![0u64; p0.len()];
        for _ in 0..n {
            let d = sample_weights(q0, rng.next());
            let (ua, un) = (rng.next(), rng.next());
            let r = if accept_always {
                SampledAccept { accepted: 1, next: 0 }
            } else {
                speculative_accept(&[p0, p1], &[q0], &[d], &[ua], un).unwrap()
            };
            let first = if r.accepted >= 1 { d } else { r.next };
            h[first as usize] += 1;
        }
        h
    }

    /// The emitted first token must be distributed as the TARGET p (chi-square p > 0.01 over 1e6
    /// trials), for a draft q far from p. CONTROL: accepting every draft emits q's distribution,
    /// which must FAIL the same test -- otherwise the test could not detect a broken accept rule.
    #[test]
    fn emitted_token_is_distributed_as_the_target_and_the_accept_all_control_fails() {
        let p0 = norm(&[0.30, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.02]);
        let p1 = norm(&[0.125; 8]);
        let q0 = norm(&[0.05, 0.05, 0.30, 0.25, 0.05, 0.05, 0.05, 0.20]);
        let n = 1_000_000u64;
        let h = first_token_hist(&p0, &p1, &q0, n, false, 7);
        let (stat, df) = chi2(&h, &p0, n);
        assert_eq!(df, 7);
        assert!(stat < CRIT_DF7_P01, "sampled DSpark does not reproduce p: chi2 {stat:.2} >= {CRIT_DF7_P01} (hist {h:?})");
        let hc = first_token_hist(&p0, &p1, &q0, n, true, 7);
        let (sc, _) = chi2(&hc, &p0, n);
        assert!(sc > 1000.0 * CRIT_DF7_P01, "the accept-all control did not fail (chi2 {sc:.2}): the test cannot see a broken rule");
    }

    /// Two positions: the SECOND emitted token, conditional on the first being the accepted draft,
    /// must follow p1 (the row after the draft) -- checks the continuation, not just position 0.
    #[test]
    fn continuation_after_an_accepted_draft_follows_the_next_target_row() {
        let p0 = norm(&[0.40, 0.30, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03]);
        let p1 = norm(&[0.02, 0.08, 0.10, 0.30, 0.20, 0.15, 0.10, 0.05]);
        let q0 = norm(&[0.50, 0.20, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03]);
        let mut rng = Rng(11);
        let mut h = vec![0u64; 8];
        let mut n = 0u64;
        for _ in 0..1_000_000 {
            let d = sample_weights(&q0, rng.next());
            let r = speculative_accept(&[&p0, &p1], &[&q0], &[d], &[rng.next()], rng.next()).unwrap();
            if r.accepted == 1 && d == 0 {
                h[r.next as usize] += 1;
                n += 1;
            }
        }
        let (stat, _) = chi2(&h, &p1, n);
        assert!(n > 100_000, "too few accepted samples ({n}) for a meaningful test");
        assert!(stat < CRIT_DF7_P01, "the bonus after an accepted draft does not follow p1: chi2 {stat:.2} (hist {h:?})");
    }

    /// T -> 0: one-hot p and q reduce to the greedy rule (accept iff the draft is the argmax).
    #[test]
    fn one_hot_distributions_reduce_to_greedy_acceptance() {
        let onehot = |i: usize| -> Vec<f32> { (0..8).map(|j| if j == i { 1.0 } else { 0.0 }).collect() };
        let (p0, p1, p2) = (onehot(3), onehot(5), onehot(1));
        let (q0, q1) = (onehot(3), onehot(6));
        let r = speculative_accept(&[&p0, &p1, &p2], &[&q0, &q1], &[3, 6], &[0.999, 0.0], 0.5).unwrap();
        assert_eq!(r, SampledAccept { accepted: 1, next: 5 }, "draft 1 (=argmax) accepted, draft 2 rejected, next = p1's argmax");
        let r = speculative_accept(&[&p0, &p1, &p2], &[&q0, &onehot(5)], &[3, 5], &[0.5, 0.5], 0.5).unwrap();
        assert_eq!(r, SampledAccept { accepted: 2, next: 1 }, "all accepted, next = p2's argmax");
    }
}
