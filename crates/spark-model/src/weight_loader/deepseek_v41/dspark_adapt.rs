//! Adaptive DSpark verify length: how many of the B drafts to verify this step.
//!
//! Output is greedy-exact for every k (the verify pass decides every emitted token), so this is a
//! pure speed policy. It keeps a per-position conditional acceptance estimate
//! c_i = P(a >= i | a >= i-1): decayed hit/trial counts (a step observes position i only when
//! k >= i and a >= i-1) shrunk toward a prior, so a position left unobserved drifts back to the
//! prior instead of keeping a stale value. It picks the k that maximises expected tokens per ms:
//! (1 + sum_{i<=k} prod_{j<=i} c_j) / (draft_ms + verify_ms[k]).
//! Every `PROBE` steps it verifies all B drafts so a position it stopped verifying is re-measured.
//! Deterministic given the acceptance history.

use anyhow::{ensure, Result};

use super::dspark::B;

/// Verify all B drafts at least once every PROBE steps (16 cost ~1% on chat in k124v: each probe
/// is a ~40 ms longer step).
const PROBE: usize = 32;
/// Per-step decay of the hit/trial counts (an effective window of ~20 steps).
const DECAY: f64 = 0.95;
/// Prior mean of each c_i and its weight in pseudo-trials. Chosen by replaying the measured
/// k124k/k124c k=5 acceptance sequences through the policy: within 0.3% of the best fixed k on
/// both chat (best k=2..3) and code (best k=5); an optimistic 1.0 start lost 3% on chat.
const PRIOR: f64 = 0.7;
const PRIOR_WEIGHT: f64 = 2.0;
/// `DSV41_DSPARK_K=k` pins the verify length (1..=B); unset = adaptive.
pub const FIXED_K_ENV: &str = "DSV41_DSPARK_K";
/// `DSV41_DSPARK_COSTS=draft,v1,..,vB` overrides the cost table (ms).
pub const COSTS_ENV: &str = "DSV41_DSPARK_COSTS";

/// Step cost in ms: `draft` plus `verify[k-1]` for a k-draft verify (k + 1 rows).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Costs {
    pub draft: f64,
    pub verify: [f64; B],
}

impl Costs {
    /// Measured at keep=124 with segmented graphs (k124k sweep, code_task, 128 tokens, median
    /// step minus draft): verify+accept+commit per k = 1..5.
    pub const DEFAULT: Costs = Costs { draft: 12.9, verify: [84.1, 100.6, 113.2, 130.5, 139.5] };

    pub fn parse(s: &str) -> Result<Self> {
        let v: Vec<f64> = s.split(',').map(|x| x.trim().parse::<f64>()).collect::<Result<_, _>>()?;
        ensure!(v.len() == B + 1, "{COSTS_ENV}: want draft + {B} verify costs, got {}", v.len());
        ensure!(v.iter().all(|&x| x.is_finite() && x > 0.0), "{COSTS_ENV}: costs must be positive");
        let mut verify = [0.0; B];
        verify.copy_from_slice(&v[1..]);
        Ok(Self { draft: v[0], verify })
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveK {
    costs: Costs,
    fixed: Option<usize>,
    /// Decayed hits and trials of position i at [i-1].
    hits: [f64; B],
    trials: [f64; B],
    step: usize,
}

impl AdaptiveK {
    pub fn new(costs: Costs, fixed: Option<usize>) -> Result<Self> {
        if let Some(k) = fixed {
            ensure!((1..=B).contains(&k), "{FIXED_K_ENV}={k}: must be 1..={B}");
        }
        Ok(Self { costs, fixed, hits: [0.0; B], trials: [0.0; B], step: 0 })
    }

    /// From `DSV41_DSPARK_K` / `DSV41_DSPARK_COSTS`.
    pub fn from_env() -> Result<Self> {
        let costs = match std::env::var(COSTS_ENV) {
            Ok(s) => Costs::parse(&s)?,
            Err(_) => Costs::DEFAULT,
        };
        let fixed = match std::env::var(FIXED_K_ENV) {
            Ok(s) => Some(s.trim().parse::<usize>().map_err(|e| anyhow::anyhow!("{FIXED_K_ENV}={s}: {e}"))?),
            Err(_) => None,
        };
        Self::new(costs, fixed)
    }

    /// The verify length for the next step.
    pub fn choose(&self) -> usize {
        if let Some(k) = self.fixed {
            return k;
        }
        if self.step % PROBE == 0 {
            return B;
        }
        let (mut best_k, mut best_rate) = (B, f64::MIN);
        let cond = self.cond();
        let (mut reach, mut expected) = (1.0, 1.0);
        for k in 1..=B {
            reach *= cond[k - 1];
            expected += reach;
            let rate = expected / (self.costs.draft + self.costs.verify[k - 1]);
            // Strict >: on a tie keep the shorter verify.
            if rate > best_rate {
                best_rate = rate;
                best_k = k;
            }
        }
        best_k
    }

    /// Record a step that verified `k` drafts and accepted `a` of them.
    pub fn observe(&mut self, k: usize, a: usize) {
        debug_assert!(a <= k && k <= B);
        for i in 0..B {
            self.hits[i] *= DECAY;
            self.trials[i] *= DECAY;
        }
        // Position i was tested iff i <= k and every earlier draft was accepted (a >= i-1).
        for i in 1..=k.min(a + 1) {
            self.trials[i - 1] += 1.0;
            if a >= i {
                self.hits[i - 1] += 1.0;
            }
        }
        self.step += 1;
    }

    /// The current estimate of c_i at [i-1].
    pub fn cond(&self) -> [f64; B] {
        std::array::from_fn(|i| (self.hits[i] + PRIOR * PRIOR_WEIGHT) / (self.trials[i] + PRIOR_WEIGHT))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(mut p: AdaptiveK, accept: impl Fn(usize) -> usize, steps: usize) -> (AdaptiveK, Vec<usize>) {
        let mut ks = Vec::new();
        for _ in 0..steps {
            let k = p.choose();
            p.observe(k, accept(k));
            ks.push(k);
        }
        (p, ks)
    }

    #[test]
    fn always_accepting_keeps_the_full_block() {
        let (_, ks) = run(AdaptiveK::new(Costs::DEFAULT, None).unwrap(), |k| k, 64);
        assert!(ks[8..].iter().all(|&k| k == B), "{ks:?}");
    }

    #[test]
    fn never_accepting_shrinks_to_one_and_still_probes() {
        let (p, ks) = run(AdaptiveK::new(Costs::DEFAULT, None).unwrap(), |_| 0, 64);
        assert!(p.cond()[0] < 0.1, "{:?}", p.cond());
        // After the estimate decays, only the probe steps verify the full block.
        let tail: Vec<usize> = ks[4..].to_vec();
        assert!(tail.iter().enumerate().all(|(i, &k)| if (4 + i) % PROBE == 0 { k == B } else { k == 1 }), "{tail:?}");
    }

    #[test]
    fn a_cheap_long_verify_is_worth_it_and_an_expensive_one_is_not() {
        // Acceptance 1,1,0,...: the first two drafts always land, the third never.
        let acc = |k: usize| k.min(2);
        let flat = Costs { draft: 10.0, verify: [100.0; B] };
        let (_, ks) = run(AdaptiveK::new(flat, None).unwrap(), acc, 48);
        // Flat cost: a longer verify never costs more, so the policy keeps the full block.
        assert_eq!(ks[40], B, "{ks:?}");
        let steep = Costs { draft: 10.0, verify: [60.0, 200.0, 210.0, 220.0, 230.0] };
        let (_, ks) = run(AdaptiveK::new(steep, None).unwrap(), acc, 48);
        // 2 tokens / 70 ms beats 3 / 210 ms.
        assert_eq!(ks[40], 1, "{ks:?}");
    }

    #[test]
    fn unobserved_positions_are_not_updated() {
        let mut p = AdaptiveK::new(Costs::DEFAULT, None).unwrap();
        p.observe(2, 0);
        let c = p.cond();
        assert!(c[0] < PRIOR);
        assert!(c[1..].iter().all(|&x| (x - PRIOR).abs() < 1e-12), "position 2 was never tested (draft 1 rejected): {c:?}");
    }

    #[test]
    fn fixed_k_is_honoured_and_validated() {
        let p = AdaptiveK::new(Costs::DEFAULT, Some(3)).unwrap();
        assert_eq!(p.choose(), 3);
        assert!(AdaptiveK::new(Costs::DEFAULT, Some(0)).is_err());
        assert!(AdaptiveK::new(Costs::DEFAULT, Some(B + 1)).is_err());
    }

    #[test]
    fn policy_is_send_and_sync() {
        fn check<T: Send + Sync + 'static>() {}
        check::<AdaptiveK>();
    }

    #[test]
    fn costs_parse() {
        assert_eq!(Costs::parse("1,2,3,4,5,6").unwrap(), Costs { draft: 1.0, verify: [2.0, 3.0, 4.0, 5.0, 6.0] });
        assert!(Costs::parse("1,2,3").is_err());
        assert!(Costs::parse("1,2,3,4,5,-6").is_err());
    }
}
