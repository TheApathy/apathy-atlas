// SPDX-License-Identifier: AGPL-3.0-only

//! The Benchmarks pane, over data this box actually produces.
//!
//! NOT A PORT OF UPSTREAM'S TAB, DELIBERATELY. That tab drives
//! `avarok-plugin`, a 295-file benchmark framework — descriptors, a plugin
//! model, param schemas, hardware sensitivity, gate baselines, an artifact
//! store. It is not in this tree, and porting it to get one tab both dwarfs
//! "port the TUI" and duplicates the `bench/` harnesses this campaign already
//! measures with.
//!
//! Worse, it would be the WRONG measurement. A plugin-backed tab would show
//! numbers produced by a different harness than the one whose controls,
//! prewarm assertions and lock discipline the results on this box depend on —
//! and somebody would trust them.
//!
//! So this pane shows the two things upstream's could not:
//!
//! 1. **Whether the box is fit to measure on.** A run taken while another job
//!    holds the GPU lock, or during a build, is not a slower number — it is a
//!    wrong one. This is the state that invalidates a measurement, in one
//!    place, before it is taken.
//! 2. **What the harnesses recorded**, read from their own `timing.txt`
//!    files, so the pane cannot disagree with the files under review.

use std::path::{Path, PathBuf};

/// Is the box fit to measure on right now?
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxState {
    /// PID holding the GPU lock, if any. `None` means free.
    pub lock_holder: Option<u32>,
    /// PIDs queued behind it, in the kernel's grant order.
    pub waiters: Vec<u32>,
    /// Host memory available, MiB. A load can need tens of GB.
    pub mem_available_mib: u64,
    /// `cargo`/`rustc` processes running. A build during a measurement window
    /// slows GPU work on this SoC by roughly 1.15x.
    pub builders: usize,
}

impl BoxState {
    /// One line naming every reason NOT to measure, or that it is clear.
    ///
    /// Reports EVERY blocker rather than the first: an operator who clears one
    /// and hits the next learned nothing the first message could have told
    /// them, which is the same rule `validate_serve_args` follows.
    pub fn verdict(&self) -> String {
        let mut blockers = Vec::new();
        if let Some(pid) = self.lock_holder {
            blockers.push(format!("GPU lock held by pid {pid}"));
        }
        if !self.waiters.is_empty() {
            blockers.push(format!("{} job(s) queued behind it", self.waiters.len()));
        }
        if self.builders > 0 {
            blockers.push(format!("{} build process(es) running", self.builders));
        }
        if self.mem_available_mib < 40 * 1024 {
            blockers.push(format!(
                "only {:.1} GB host memory free",
                self.mem_available_mib as f64 / 1024.0
            ));
        }
        if blockers.is_empty() {
            "box is clear — a measurement taken now is not contended".into()
        } else {
            format!("DO NOT MEASURE: {}", blockers.join(" · "))
        }
    }
}

/// Read the lock table for `lock_path`'s inode.
///
/// `/proc/locks`, not `fuser`: `fuser` returns an undifferentiated pid list
/// and cannot say which one HOLDS and which are WAITING, so it cannot answer
/// "who runs next". `/proc/locks` marks waiters with a leading `->`.
///
/// A holder pid that is dead is NOT a stale lock — the lock survives through a
/// descriptor inherited by a live child, which is the normal shape for a
/// harness that re-execs. Reported as held, because it is.
pub fn read_lock(lock_path: &Path, proc_locks: &str) -> (Option<u32>, Vec<u32>) {
    let Ok(meta) = std::fs::metadata(lock_path) else {
        return (None, Vec::new());
    };
    let ino = {
        use std::os::unix::fs::MetadataExt as _;
        meta.ino()
    };
    let (mut holder, mut waiters) = (None, Vec::new());
    for line in proc_locks.lines() {
        // `116: FLOCK  ADVISORY  WRITE 346214 103:02:16529524 0 EOF`
        // `116: ->    FLOCK  ADVISORY  WRITE 409729 ...`
        let Some(rest) = line.split_once(':').map(|(_, r)| r) else {
            continue;
        };
        let is_waiter = rest.trim_start().starts_with("->");
        let fields: Vec<&str> = rest.split_whitespace().collect();
        let Some(inode_field) = fields
            .iter()
            .find(|f| f.contains(':') && f.contains('.') == false)
        else {
            continue;
        };
        let Some(found) = inode_field
            .rsplit(':')
            .next()
            .and_then(|i| i.parse::<u64>().ok())
        else {
            continue;
        };
        if found != ino {
            continue;
        }
        let pid = fields
            .iter()
            .rev()
            .find_map(|f| f.parse::<u32>().ok().filter(|p| *p > 1));
        let Some(pid) = pid else { continue };
        if is_waiter {
            waiters.push(pid);
        } else {
            holder = Some(pid);
        }
    }
    (holder, waiters)
}

/// Sample the box. Cheap enough for the pane's refresh tick.
pub fn box_state(lock_path: &Path) -> BoxState {
    let proc_locks = std::fs::read_to_string("/proc/locks").unwrap_or_default();
    let (lock_holder, waiters) = read_lock(lock_path, &proc_locks);
    let mem_available_mib = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map(|kib| kib / 1024)
        .unwrap_or(0);
    let builders = std::fs::read_dir("/proc")
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    let comm = e.path().join("comm");
                    std::fs::read_to_string(comm)
                        .map(|c| {
                            let c = c.trim();
                            c == "cargo" || c == "rustc" || c == "nvcc" || c == "ptxas"
                        })
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0);
    BoxState {
        lock_holder,
        waiters,
        mem_available_mib,
        builders,
    }
}

/// One harness arm, as its own `timing.txt` recorded it.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmSummary {
    pub name: String,
    pub trials: usize,
    /// Median tok/s. The harnesses report medians, so the pane does too —
    /// quoting a mean beside a file that records a median is two numbers for
    /// one quantity.
    pub median_tok_s: f64,
    pub min_tok_s: f64,
    pub max_tok_s: f64,
}

/// Parse a harness `timing.txt`.
///
/// Returns `None` for a file with no complete trial lines, rather than an arm
/// with zero trials: an arm that produced nothing is not an arm that measured
/// zero, and the pane must not render it as a result.
pub fn parse_timing(text: &str, name: &str) -> Option<ArmSummary> {
    let mut v: Vec<f64> = text
        .lines()
        .filter(|l| l.starts_with("trial-"))
        .filter_map(|l| l.rsplit_once("tok_s=")?.1.trim().parse::<f64>().ok())
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let median = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    };
    Some(ArmSummary {
        name: name.to_string(),
        trials: n,
        median_tok_s: median,
        min_tok_s: v[0],
        max_tok_s: v[n - 1],
    })
}

/// Every arm under `runs_dir`, newest first.
pub fn scan_runs(runs_dir: &Path) -> Vec<ArmSummary> {
    let Ok(rd) = std::fs::read_dir(runs_dir) else {
        return Vec::new();
    };
    let mut out: Vec<(std::time::SystemTime, ArmSummary)> = Vec::new();
    for e in rd.flatten() {
        let t = e.path().join("timing.txt");
        let Ok(text) = std::fs::read_to_string(&t) else {
            continue;
        };
        let when = t
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        if let Some(a) = parse_timing(&text, &e.file_name().to_string_lossy()) {
            out.push((when, a));
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out.into_iter().map(|(_, a)| a).collect()
}

/// Where the harnesses write, from `ATLAS_BENCH_RUNS` (colon-separated).
pub fn runs_dirs() -> Vec<PathBuf> {
    std::env::var("ATLAS_BENCH_RUNS")
        .map(|v| {
            v.split(':')
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default()
}

/// THE SPREAD OF ARMS THAT SHOULD BE THE SAME NUMBER.
///
/// The single most useful figure this campaign produced: identical control
/// arms on GLM spread 2.16%, which is the floor under every delta measured on
/// that harness. An effect smaller than this is not a small effect, it is
/// invisible — and three kernel sweeps were spent before anyone computed it.
///
/// `None` for fewer than two arms: a spread needs two, and one arm reported as
/// a 0% spread would be the same two-sample underestimate that hid this for a
/// night.
pub fn control_spread_pct(controls: &[ArmSummary]) -> Option<f64> {
    if controls.len() < 2 {
        return None;
    }
    let (mut lo, mut hi, mut sum) = (f64::MAX, f64::MIN, 0.0);
    for c in controls {
        lo = lo.min(c.median_tok_s);
        hi = hi.max(c.median_tok_s);
        sum += c.median_tok_s;
    }
    let mean = sum / controls.len() as f64;
    (mean > 0.0).then(|| (hi - lo) / mean * 100.0)
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod tests;
