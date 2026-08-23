//! Exact-output-safe speculative-width routing.
//!
//! The verifier remains authoritative; this module only chooses how many
//! drafts to offer. Scores are delivered tokens (accepted drafts + target
//! bonus) per complete propose/verify wall-clock second, so verification cost
//! and long-context KV degradation are represented directly.

#[derive(Debug, Clone)]
pub struct ThroughputRouter {
    ewma_tps: [f64; 65],
    samples: [u32; 65],
    probe_cursor: usize,
    decisions: u64,
}

impl Default for ThroughputRouter {
    fn default() -> Self {
        Self {
            ewma_tps: [0.0; 65],
            samples: [0; 65],
            probe_cursor: 0,
            decisions: 0,
        }
    }
}

impl ThroughputRouter {
    pub fn observe(&mut self, width: usize, delivered: usize, elapsed_secs: f64, alpha: f64) {
        if !(1..self.ewma_tps.len()).contains(&width)
            || delivered == 0
            || !elapsed_secs.is_finite()
            || elapsed_secs <= 0.0
        {
            return;
        }
        let sample = delivered as f64 / elapsed_secs;
        let alpha = alpha.clamp(0.01, 1.0);
        self.ewma_tps[width] = if self.samples[width] == 0 {
            sample
        } else {
            alpha * sample + (1.0 - alpha) * self.ewma_tps[width]
        };
        self.samples[width] = self.samples[width].saturating_add(1);
    }

    pub fn choose(&mut self, candidates: &[usize], max_width: usize, probe_interval: u64) -> usize {
        let mut valid: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&w| w > 0 && w <= max_width && w < self.ewma_tps.len())
            .collect();
        valid.sort_unstable();
        valid.dedup();
        if valid.is_empty() {
            return max_width.max(1);
        }
        self.decisions = self.decisions.saturating_add(1);

        // Warm every shape once. Afterwards periodically re-sample candidates
        // to follow phase changes and increasing KV cost.
        if let Some(&width) = valid.iter().find(|&&w| self.samples[w] == 0) {
            return width;
        }
        if probe_interval > 0 && self.decisions.is_multiple_of(probe_interval) {
            let width = valid[self.probe_cursor % valid.len()];
            self.probe_cursor = self.probe_cursor.wrapping_add(1);
            return width;
        }
        valid
            .into_iter()
            .max_by(|&a, &b| self.ewma_tps[a].total_cmp(&self.ewma_tps[b]))
            .unwrap_or(max_width.max(1))
    }

    pub fn score(&self, width: usize) -> Option<f64> {
        (width < self.samples.len() && self.samples[width] > 0).then_some(self.ewma_tps[width])
    }
}

pub fn parse_widths(value: Option<&str>, max_width: usize) -> Vec<usize> {
    let source = value.unwrap_or("2,4,6,8,12,15");
    let mut widths: Vec<usize> = source
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&w| w > 0 && w <= max_width)
        .collect();
    widths.push(max_width.max(1));
    widths.sort_unstable();
    widths.dedup();
    widths
}

/// Climb/drop adaptive draft-depth controller (ported from llama.cpp
/// PR #27210 `common/speculative-adaptive.h`, draft-mtp-adaptive).
///
/// Hysteresis state machine with a climb counter and a weighted
/// drop-pressure accumulator. The depth climbs one step after
/// `climb_threshold(depth)` consecutive verifies that accepted EVERY
/// drafted token; any miss adds `(n_draft - n_accepted)` to a
/// drop-pressure accumulator, and the depth drops one step once it
/// exceeds `drop_pressure(depth)`. High depths fall quickly (a total
/// miss adds N), low depths hold, and at the floor no pressure
/// accumulates at all. This exploits the bursty acceptance pattern on
/// coding workloads: while the drafter is in sync, depth climbs to the
/// wide end (amortizing the fixed target weight stream); on drift it
/// falls back fast.
#[derive(Debug, Clone)]
pub struct ClimbDropRouter {
    n_cur: usize,
    n_climb: usize,
    n_drop: usize,
    floor: usize,
    cap: usize,
    decisions: u64,
}

impl Default for ClimbDropRouter {
    fn default() -> Self {
        Self {
            n_cur: 0,
            n_climb: 0,
            n_drop: 0,
            floor: 1,
            cap: 1,
            decisions: 0,
        }
    }
}

impl ClimbDropRouter {
    /// Consecutive full accepts needed to climb one step from depth N;
    /// low at the floor and at depth, high in the middle where
    /// acceptance is marginal.
    pub const fn climb_threshold(depth: usize) -> usize {
        match depth {
            1 => 2,
            2 => 4,
            3 => 6,
            4 => 5,
            5 => 4,
            6 => 3,
            _ => 2, // depth >= 7
        }
    }

    /// Accumulated (n_draft - n_accepted) needed to drop one step from
    /// depth N; scaled by depth with a floor so shallow depths do not
    /// collapse too fast.
    pub const fn drop_pressure(depth: usize) -> usize {
        if depth * 5 < 20 { 20 } else { depth * 5 }
    }

    pub fn reset(&mut self, n_max: usize, n_min_adaptive: usize) {
        let cap = n_max.max(1);
        let floor = n_min_adaptive.max(1);
        self.cap = cap;
        self.floor = floor;
        self.n_cur = floor.min(cap);
        self.n_climb = 0;
        self.n_drop = 0;
        self.decisions = 0;
    }

    /// Feed one verification result: `n_draft` is how many drafts were
    /// offered, `n_accepted` how many the target accepted.
    pub fn update(&mut self, n_draft: usize, n_accepted: usize) {
        if n_draft == 0 {
            return;
        }
        let cap = self.cap.max(1);
        let floor = self.floor.max(1);
        if n_accepted == n_draft {
            self.n_drop = 0;
            if self.n_cur < cap {
                self.n_climb += 1;
                if self.n_climb >= Self::climb_threshold(self.n_cur) {
                    self.n_cur += 1;
                    self.n_climb = 0;
                }
            }
        } else {
            self.n_climb = 0;
            if self.n_cur > floor {
                self.n_drop += n_draft.saturating_sub(n_accepted);
                if self.n_drop >= Self::drop_pressure(self.n_cur) {
                    self.n_cur -= 1;
                    self.n_drop = 0;
                }
            }
        }
        self.decisions = self.decisions.saturating_add(1);
    }

    pub fn choose(&self) -> usize {
        self.n_cur.clamp(1, self.cap.max(1))
    }

    pub fn score(&self) -> usize {
        self.n_cur
    }

    pub fn decisions(&self) -> u64 {
        self.decisions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_delivered_throughput_not_accept_count() {
        let mut router = ThroughputRouter::default();
        router.observe(4, 4, 0.10, 0.3); // 40 tok/s
        router.observe(8, 7, 0.25, 0.3); // 28 tok/s despite more accepts
        assert_eq!(router.choose(&[4, 8], 8, 0), 4);
    }

    #[test]
    fn ewma_reacts_to_late_context_slowdown() {
        let mut router = ThroughputRouter::default();
        router.observe(4, 4, 0.10, 1.0);
        router.observe(8, 7, 0.10, 1.0);
        assert_eq!(router.choose(&[4, 8], 8, 0), 8);
        router.observe(8, 7, 0.30, 1.0);
        assert_eq!(router.choose(&[4, 8], 8, 0), 4);
    }

    #[test]
    fn parser_clamps_and_keeps_physical_max() {
        assert_eq!(parse_widths(Some("0,4,4,99,bad"), 8), vec![4, 8]);
    }

    #[test]
    fn climbdrop_climbs_on_consecutive_full_accepts() {
        let mut r = ClimbDropRouter::default();
        r.reset(15, 4); // floor 4, cap 15
        assert_eq!(r.choose(), 4);
        // Full accepts: threshold at depth 4 is 5, so 5 full accepts climb.
        for _ in 0..5 {
            r.update(4, 4);
        }
        assert_eq!(r.choose(), 5);
        // A miss resets the climb streak and accumulates pressure.
        r.update(5, 3); // +2 pressure, below drop_pressure(5)=25
        assert_eq!(r.choose(), 5);
    }

    #[test]
    fn climbdrop_drops_on_weighted_pressure() {
        let mut r = ClimbDropRouter::default();
        r.reset(15, 4);
        // Climb to depth 8 with full accepts (threshold >= 7 is 2).
        while r.choose() < 8 {
            let d = r.choose();
            for _ in 0..ClimbDropRouter::climb_threshold(d) {
                r.update(d, d);
            }
            assert!(r.choose() >= d, "depth must not fall on full accepts");
        }
        assert_eq!(r.choose(), 8);
        // drop_pressure(8) = 40: a total miss (0/8) adds 8; 5 misses drop one.
        for _ in 0..5 {
            r.update(8, 0);
        }
        assert_eq!(r.choose(), 7);
        // At the floor, pressure does not accumulate.
        while r.choose() > 4 {
            let d = r.choose();
            for _ in 0..ClimbDropRouter::drop_pressure(d) {
                r.update(d, 0);
            }
        }
        assert_eq!(r.choose(), 4);
        for _ in 0..100 {
            r.update(4, 0);
        }
        assert_eq!(r.choose(), 4);
    }

    #[test]
    fn climbdrop_respects_cap_and_floor() {
        let mut r = ClimbDropRouter::default();
        r.reset(8, 3);
        assert_eq!(r.choose(), 3);
        // Even unlimited full accepts never exceed the cap.
        for _ in 0..200 {
            let d = r.choose();
            r.update(d, d);
        }
        assert_eq!(r.choose(), 8);
    }
}
