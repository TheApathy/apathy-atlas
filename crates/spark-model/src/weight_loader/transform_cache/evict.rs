// SPDX-License-Identifier: AGPL-3.0-only

//! LRU eviction for the post-transform weight cache.
//!
//! Every model variant gets its own fingerprint directory, and at ~12.7 GiB
//! apiece a few checkpoint swaps will fill a disk. This module reclaims the
//! least-recently-used ones once the cache root exceeds a budget.
//!
//! **On by default** whenever the weight cache itself is enabled — the whole
//! point of a budget is that the disk does not fill up, which only works if
//! pruning happens without anyone remembering to ask. `ATLAS_WEIGHT_CACHE_EVICT=0`
//! opts out, after which nothing here ever deletes a directory.
//!
//! Env:
//!   - `ATLAS_WEIGHT_CACHE_EVICT=0` — opt out of deletion; the pass then only
//!     reports what is on disk. Also accepts `false` / `off` / `no`.
//!   - `ATLAS_WEIGHT_CACHE_MAX_GIB=<n>` — total budget, default 32 (about two
//!     variants at our size). The pass runs until the root fits.
//!   - `ATLAS_WEIGHT_CACHE_KEEP=<n>` — additionally cap the directory count.
//!     Default 0 = size-based only.
//!
//! Recency comes from a `last_used` marker holding a unix timestamp, rewritten
//! on every hit and after every successful build (same tmp-file + rename as the
//! index, so a torn write can never produce a half-parsed timestamp).
//!
//! ## Why two different grace periods
//!
//! Another server may be running against this cache root right now, and
//! deleting a directory out from under it would take down a live engine. Two
//! rules keep that from happening, both keyed on the newest mtime in the
//! directory:
//!
//!   - A directory with a valid index is skipped while it has been touched
//!     within [`INDEXED_GRACE_SECS`] (10 min). A live server touches
//!     `last_used` at startup, so anything recently started is protected.
//!   - A directory with NO valid index gets a much longer [`TORN_GRACE_SECS`]
//!     (1 h) instead. This is the dangerous case: an index-less directory is
//!     exactly what a build in progress looks like, and a cold build writes
//!     GiB of blob before it publishes an index. Ten minutes is not enough
//!     headroom for that; an hour of no writes means the builder is gone.
//!
//! Past its grace period an index-less directory is reclaimed regardless of
//! budget. It can never produce a hit — no index means no slot table — so it
//! is pure dead weight, and leaving it would defeat the point of having a
//! budget at all.

use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::CACHE_FORMAT_VERSION;

/// Quiescence required before an indexed directory may be evicted.
const INDEXED_GRACE_SECS: u64 = 10 * 60;
/// Quiescence required before an index-less directory may be evicted. Longer
/// than the indexed grace because an index-less directory is indistinguishable
/// from a cold build that has not published yet.
const TORN_GRACE_SECS: u64 = 60 * 60;

const DEFAULT_MAX_GIB: u64 = 32;
const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;

/// One fingerprint directory under the cache root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirInfo {
    pub fingerprint: String,
    pub path: PathBuf,
    /// Sum of every file's length in the directory.
    pub bytes: u64,
    /// Unix seconds from the `last_used` marker, `None` when absent/unreadable.
    pub last_used: Option<u64>,
    /// Newest mtime of any file, unix seconds. Drives the grace-period guard.
    pub newest_mtime: u64,
    /// Whether `index.json` parses AND matches the current format version.
    /// A stale-version index can never be served, so it counts as torn.
    pub has_index: bool,
}

/// What an eviction pass did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct EvictionReport {
    pub evicted: usize,
    pub reclaimed_bytes: u64,
    pub failed: usize,
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

pub fn budget_bytes() -> u64 {
    env_u64("ATLAS_WEIGHT_CACHE_MAX_GIB", DEFAULT_MAX_GIB).saturating_mul(BYTES_PER_GIB)
}

/// Whether eviction may delete anything. **On by default** — the point of the
/// budget is that the disk never fills up, so pruning has to happen without
/// anyone remembering to ask for it. `ATLAS_WEIGHT_CACHE_EVICT=0` is the
/// explicit opt-out.
///
/// The opt-out is matched leniently (`0`, `false`, `off`, `no`, any case)
/// while everything else means enabled. Leniency here can only ever result in
/// *fewer* deletions, so a user who typed `=false` meaning "stop deleting my
/// caches" gets what they intended rather than a surprise reclaim.
///
/// Deliberately absent from the fingerprint's env list: eviction changes which
/// caches exist on disk, never what a cached buffer contains, so toggling it
/// must not invalidate a valid cache.
pub fn eviction_enabled() -> bool {
    match std::env::var("ATLAS_WEIGHT_CACHE_EVICT") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

pub fn keep_limit() -> usize {
    env_u64("ATLAS_WEIGHT_CACHE_KEEP", 0) as usize
}

// ─────────────────────────── marker file ───────────────────────────

/// Record "this cache was used now". Same tmp + rename discipline as the
/// index so a crash mid-write leaves the previous timestamp intact rather
/// than a truncated one that would parse as a much older time.
pub fn touch_last_used(dir: &Path, now: u64) {
    let tmp = dir.join("last_used.tmp");
    let final_path = dir.join("last_used");
    if fs::write(&tmp, now.to_string().as_bytes()).is_ok()
        && let Err(e) = fs::rename(&tmp, &final_path)
    {
        tracing::debug!("weight cache: could not update last_used ({e})");
        let _ = fs::remove_file(&tmp);
    }
}

fn read_last_used(dir: &Path) -> Option<u64> {
    fs::read_to_string(dir.join("last_used"))
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

// ─────────────────────────── scanning ───────────────────────────

/// Total file bytes and newest mtime in a directory (flat — cache
/// directories only ever hold files).
pub fn dir_size_and_mtime(dir: &Path) -> (u64, u64) {
    let mut bytes = 0u64;
    let mut newest = 0u64;
    let Ok(entries) = fs::read_dir(dir) else {
        return (0, 0);
    };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        bytes += meta.len();
        if let Ok(mtime) = meta.modified()
            && let Ok(d) = mtime.duration_since(UNIX_EPOCH)
        {
            newest = newest.max(d.as_secs());
        }
    }
    (bytes, newest)
}

fn has_current_index(dir: &Path) -> bool {
    fs::read(dir.join("index.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<super::CacheIndex>(&b).ok())
        .is_some_and(|i| i.format_version == CACHE_FORMAT_VERSION)
}

/// Enumerate every fingerprint directory under the cache root.
pub fn scan_root(root: &Path) -> Vec<DirInfo> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(fingerprint) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let (bytes, newest_mtime) = dir_size_and_mtime(&path);
        out.push(DirInfo {
            fingerprint: fingerprint.to_string(),
            last_used: read_last_used(&path),
            has_index: has_current_index(&path),
            bytes,
            newest_mtime,
            path,
        });
    }
    out
}

// ─────────────────────────── policy ───────────────────────────

/// Decide which directories to delete, as indices into `dirs`.
///
/// Pure: no filesystem access, so the policy is testable without touching
/// disk. Order of preference is torn directories first (they can never be
/// served), then least-recently-used. `active` is never selected, and a
/// directory inside its grace period is never selected.
pub fn select_victims(
    dirs: &[DirInfo],
    active: &str,
    budget: u64,
    keep: usize,
    now: u64,
) -> Vec<usize> {
    let mut candidates: Vec<usize> = (0..dirs.len())
        .filter(|&i| {
            let d = &dirs[i];
            if d.fingerprint == active {
                return false;
            }
            let idle = now.saturating_sub(d.newest_mtime);
            let grace = if d.has_index {
                INDEXED_GRACE_SECS
            } else {
                TORN_GRACE_SECS
            };
            idle >= grace
        })
        .collect();

    // Torn first (has_index false sorts before true), then oldest last_used.
    // A missing marker sorts as epoch 0, i.e. evict before anything dated.
    // Fingerprint breaks ties so the pass is deterministic.
    candidates.sort_by(|&a, &b| {
        let (x, y) = (&dirs[a], &dirs[b]);
        x.has_index
            .cmp(&y.has_index)
            .then(x.last_used.unwrap_or(0).cmp(&y.last_used.unwrap_or(0)))
            .then(x.fingerprint.cmp(&y.fingerprint))
    });

    let mut total: u64 = dirs.iter().map(|d| d.bytes).sum();
    let mut count = dirs.len();
    let mut victims = Vec::new();
    for i in candidates {
        // Torn directories are reclaimed whether or not we are over budget:
        // they hold no servable data, so keeping them only wastes disk.
        let unusable = !dirs[i].has_index;
        let over_size = total > budget;
        let over_count = keep > 0 && count > keep;
        if !unusable && !over_size && !over_count {
            break;
        }
        victims.push(i);
        total = total.saturating_sub(dirs[i].bytes);
        count = count.saturating_sub(1);
    }
    victims
}

/// Human-readable recency for the eviction log line.
pub fn describe_age(last_used: Option<u64>, now: u64) -> String {
    let Some(t) = last_used else {
        return "never".to_string();
    };
    let secs = now.saturating_sub(t);
    if secs < 60 {
        "just now".to_string()
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GIB as f64
}

/// Run one eviction pass over `root`, never touching `active`.
///
/// `enabled` is passed in rather than read from the environment here so the
/// gate is testable without mutating process-global state. When it is false
/// this function reports what is on disk and deletes nothing at all.
///
/// Failures are logged and counted, not propagated: a cache we could not
/// prune is a disk-space problem, never a correctness one, and it must not
/// take down a server that has already loaded its weights.
pub fn run(
    root: &Path,
    active: &str,
    budget: u64,
    keep: usize,
    now: u64,
    enabled: bool,
) -> Result<EvictionReport> {
    let dirs = scan_root(root);
    if dirs.is_empty() {
        return Ok(EvictionReport::default());
    }
    let total: u64 = dirs.iter().map(|d| d.bytes).sum();

    if !enabled {
        // Say so out loud: an operator watching the cache grow needs to know
        // nothing is cleaning it, and which flag turns cleaning on.
        tracing::info!(
            "weight cache: eviction OFF (ATLAS_WEIGHT_CACHE_EVICT opts out) — \
             {} dirs, {:.2} GiB retained, nothing will be deleted",
            dirs.len(),
            gib(total),
        );
        return Ok(EvictionReport::default());
    }
    tracing::info!(
        "weight cache: eviction ON — budget {:.2} GiB{}, currently {} dirs / {:.2} GiB",
        gib(budget),
        if keep > 0 {
            format!(", keep at most {keep} dirs")
        } else {
            String::new()
        },
        dirs.len(),
        gib(total),
    );

    let victims = select_victims(&dirs, active, budget, keep, now);
    if victims.is_empty() {
        tracing::debug!(
            "weight cache: {} dirs, {:.2} GiB, budget {:.2} GiB — nothing to evict",
            dirs.len(),
            gib(total),
            gib(budget),
        );
        return Ok(EvictionReport::default());
    }

    let mut report = EvictionReport::default();
    for i in victims {
        let d = &dirs[i];
        match fs::remove_dir_all(&d.path) {
            Ok(()) => {
                tracing::info!(
                    "weight cache EVICT: {} ({:.2} GiB, last used {})",
                    d.fingerprint,
                    gib(d.bytes),
                    describe_age(d.last_used, now),
                );
                report.evicted += 1;
                report.reclaimed_bytes += d.bytes;
            }
            Err(e) => {
                tracing::warn!("weight cache: could not evict {} ({e})", d.fingerprint);
                report.failed += 1;
            }
        }
    }
    tracing::info!(
        "weight cache eviction: {} dirs removed, {:.2} GiB reclaimed, \
         {:.2} GiB now in use (budget {:.2} GiB){}",
        report.evicted,
        gib(report.reclaimed_bytes),
        gib(total.saturating_sub(report.reclaimed_bytes)),
        gib(budget),
        if report.failed > 0 {
            format!(", {} failed", report.failed)
        } else {
            String::new()
        },
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: u64 = 3600;
    /// A realistic unix time, so the multi-hour offsets below stay positive.
    const NOW: u64 = 1_700_000_000;

    fn dir(fp: &str, gib_size: u64, last_used: Option<u64>, idle: u64, has_index: bool) -> DirInfo {
        DirInfo {
            fingerprint: fp.to_string(),
            path: PathBuf::from("/nonexistent").join(fp),
            bytes: gib_size * BYTES_PER_GIB,
            last_used,
            newest_mtime: NOW.saturating_sub(idle),
            has_index,
        }
    }

    /// Indexed, long idle, dated marker — the ordinary eviction candidate.
    fn old(fp: &str, gib_size: u64, used_ago: u64) -> DirInfo {
        dir(
            fp,
            gib_size,
            Some(NOW.saturating_sub(used_ago)),
            24 * HOUR,
            true,
        )
    }

    #[test]
    fn evicts_least_recently_used_first() {
        let dirs = vec![
            old("aaa", 10, 2 * HOUR),
            old("bbb", 10, 90 * HOUR), // oldest
            old("ccc", 10, 10 * HOUR),
        ];
        // 32 GiB budget, 30 GiB present: under budget, nothing to do.
        assert!(select_victims(&dirs, "zzz", 32 * BYTES_PER_GIB, 0, NOW).is_empty());

        // 25 GiB budget: must drop 5 GiB, so exactly one dir, the oldest.
        let v = select_victims(&dirs, "zzz", 25 * BYTES_PER_GIB, 0, NOW);
        assert_eq!(v.len(), 1);
        assert_eq!(dirs[v[0]].fingerprint, "bbb");

        // 15 GiB budget: two must go, oldest then next-oldest.
        let v = select_victims(&dirs, "zzz", 15 * BYTES_PER_GIB, 0, NOW);
        let names: Vec<_> = v.iter().map(|&i| dirs[i].fingerprint.as_str()).collect();
        assert_eq!(names, vec!["bbb", "ccc"]);
    }

    #[test]
    fn active_dir_is_never_evicted_even_when_over_budget() {
        let dirs = vec![old("active", 100, 999 * HOUR), old("other", 1, 1)];
        // Budget of 0 demands evicting everything; active must still survive.
        let v = select_victims(&dirs, "active", 0, 0, NOW);
        let names: Vec<_> = v.iter().map(|&i| dirs[i].fingerprint.as_str()).collect();
        assert_eq!(names, vec!["other"]);
        assert!(
            !names.contains(&"active"),
            "the in-use cache must never be deleted"
        );
    }

    #[test]
    fn torn_dirs_go_first_and_go_even_when_under_budget() {
        let dirs = vec![
            old("indexed_old", 1, 500 * HOUR),
            dir("torn", 5, None, 2 * HOUR, false),
        ];
        // Wildly under budget, yet the torn dir is still reclaimed because it
        // can never be served, and the indexed one is left alone.
        let v = select_victims(&dirs, "zzz", 1024 * BYTES_PER_GIB, 0, NOW);
        let names: Vec<_> = v.iter().map(|&i| dirs[i].fingerprint.as_str()).collect();
        assert_eq!(names, vec!["torn"]);
    }

    #[test]
    fn torn_dirs_are_ordered_before_indexed_ones() {
        let dirs = vec![
            old("indexed", 10, 999 * HOUR), // far older by last_used
            dir("torn", 10, Some(NOW), 2 * HOUR, false),
        ];
        let v = select_victims(&dirs, "zzz", 0, 0, NOW);
        let names: Vec<_> = v.iter().map(|&i| dirs[i].fingerprint.as_str()).collect();
        assert_eq!(
            names,
            vec!["torn", "indexed"],
            "torn wins on ordering despite a newer last_used"
        );
    }

    #[test]
    fn recently_touched_dirs_are_protected_from_a_concurrent_build() {
        // Indexed but touched 1 minute ago: another server just started on it.
        let dirs = vec![old("stale", 10, 999 * HOUR), {
            let mut d = old("live", 10, 999 * HOUR);
            d.newest_mtime = NOW - 60;
            d
        }];
        let v = select_victims(&dirs, "zzz", 0, 0, NOW);
        let names: Vec<_> = v.iter().map(|&i| dirs[i].fingerprint.as_str()).collect();
        assert_eq!(names, vec!["stale"], "a dir touched 1m ago must be spared");

        // Same dir 11 minutes idle clears the indexed grace period.
        let mut dirs2 = dirs;
        dirs2[1].newest_mtime = NOW - (INDEXED_GRACE_SECS + 60);
        assert_eq!(select_victims(&dirs2, "zzz", 0, 0, NOW).len(), 2);
    }

    #[test]
    fn in_progress_build_survives_the_indexed_grace_but_not_the_torn_one() {
        // Index-less and 30 minutes idle — past the 10m indexed grace but
        // inside the 1h torn grace, i.e. plausibly a cold build in flight.
        let building = dir("building", 5, None, 30 * 60, false);
        assert!(
            select_victims(&[building.clone()], "zzz", 0, 0, NOW).is_empty(),
            "a half-written build must not be deleted out from under its writer"
        );

        // Two hours idle: the writer is gone, reclaim it.
        let abandoned = dir("abandoned", 5, None, 2 * HOUR, false);
        assert_eq!(select_victims(&[abandoned], "zzz", 0, 0, NOW).len(), 1);
    }

    #[test]
    fn keep_limit_caps_directory_count_independently_of_size() {
        let dirs = vec![
            old("a", 1, 10 * HOUR),
            old("b", 1, 30 * HOUR),
            old("c", 1, 20 * HOUR),
        ];
        let huge = 1024 * BYTES_PER_GIB;
        // keep=0 disables the count cap; 3 GiB is far under budget.
        assert!(select_victims(&dirs, "zzz", huge, 0, NOW).is_empty());
        // keep=2 drops exactly the oldest one.
        let v = select_victims(&dirs, "zzz", huge, 2, NOW);
        let names: Vec<_> = v.iter().map(|&i| dirs[i].fingerprint.as_str()).collect();
        assert_eq!(names, vec!["b"]);
        // keep=1 drops the two oldest, leaving one.
        let v = select_victims(&dirs, "zzz", huge, 1, NOW);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn missing_marker_sorts_as_oldest() {
        let dirs = vec![
            old("dated", 10, 5 * HOUR),
            dir("undated", 10, None, 24 * HOUR, true),
        ];
        let v = select_victims(&dirs, "zzz", 15 * BYTES_PER_GIB, 0, NOW);
        let names: Vec<_> = v.iter().map(|&i| dirs[i].fingerprint.as_str()).collect();
        assert_eq!(names, vec!["undated"]);
    }

    #[test]
    fn describes_recency_in_human_units() {
        assert_eq!(describe_age(None, NOW), "never");
        assert_eq!(describe_age(Some(NOW), NOW), "just now");
        assert_eq!(describe_age(Some(NOW - 300), NOW), "5m ago");
        assert_eq!(describe_age(Some(NOW - 3 * HOUR), NOW), "3h ago");
        assert_eq!(describe_age(Some(NOW - 50 * HOUR), NOW), "2d ago");
        // A marker from the future must not underflow into a huge age.
        assert_eq!(describe_age(Some(NOW + 500), NOW), "just now");
    }

    // ── filesystem-backed tests (tempdirs, no GPU) ──

    fn tempdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "atlas-evict-test-{tag}-{}-{}",
            std::process::id(),
            unix_now(),
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn size_accounting_sums_every_file_in_the_dir() {
        let root = tempdir("size");
        let d = root.join("fp0");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("blob.bin"), vec![0u8; 4096]).unwrap();
        fs::write(d.join("index.json"), vec![0u8; 100]).unwrap();

        let (bytes, mtime) = dir_size_and_mtime(&d);
        assert_eq!(bytes, 4196, "must sum blob + index, not just the blob");
        assert!(mtime > 0, "mtime must be populated");

        // A subdirectory is not counted as a file.
        fs::create_dir_all(d.join("nested")).unwrap();
        assert_eq!(dir_size_and_mtime(&d).0, 4196);

        // Missing directory reads as empty rather than panicking.
        assert_eq!(dir_size_and_mtime(&root.join("absent")), (0, 0));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn last_used_marker_round_trips() {
        let root = tempdir("marker");
        assert_eq!(read_last_used(&root), None, "absent marker reads as None");

        touch_last_used(&root, 1_234_567);
        assert_eq!(read_last_used(&root), Some(1_234_567));
        assert!(
            !root.join("last_used.tmp").exists(),
            "the temp file must be renamed away, not left behind"
        );

        touch_last_used(&root, 7_654_321);
        assert_eq!(read_last_used(&root), Some(7_654_321), "must overwrite");

        fs::write(root.join("last_used"), b"not-a-number").unwrap();
        assert_eq!(read_last_used(&root), None, "garbage reads as None");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn scan_root_classifies_indexed_and_torn_dirs() {
        let root = tempdir("scan");

        let good = root.join("aaaa");
        fs::create_dir_all(&good).unwrap();
        fs::write(good.join("blob.bin"), vec![0u8; 512]).unwrap();
        let index = super::super::CacheIndex {
            format_version: CACHE_FORMAT_VERSION,
            fingerprint: "aaaa".into(),
            blob_len: 512,
            entries: Vec::new(),
        };
        fs::write(good.join("index.json"), serde_json::to_vec(&index).unwrap()).unwrap();
        touch_last_used(&good, 42);

        // Torn: blob but no index at all.
        let torn = root.join("bbbb");
        fs::create_dir_all(&torn).unwrap();
        fs::write(torn.join("blob.bin"), vec![0u8; 256]).unwrap();

        // Stale format version counts as torn — it can never be served.
        let stale = root.join("cccc");
        fs::create_dir_all(&stale).unwrap();
        let stale_index = super::super::CacheIndex {
            format_version: CACHE_FORMAT_VERSION + 1,
            ..index
        };
        fs::write(
            stale.join("index.json"),
            serde_json::to_vec(&stale_index).unwrap(),
        )
        .unwrap();

        let mut found = scan_root(&root);
        found.sort_by(|a, b| a.fingerprint.cmp(&b.fingerprint));
        assert_eq!(found.len(), 3);

        assert_eq!(found[0].fingerprint, "aaaa");
        assert!(found[0].has_index);
        assert_eq!(found[0].last_used, Some(42));
        assert!(found[0].bytes >= 512);

        assert_eq!(found[1].fingerprint, "bbbb");
        assert!(!found[1].has_index, "no index.json = torn");
        assert_eq!(found[1].last_used, None);

        assert_eq!(found[2].fingerprint, "cccc");
        assert!(!found[2].has_index, "stale format version = torn");

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn run_deletes_only_the_selected_dirs() {
        let root = tempdir("run");
        for name in ["active", "torn"] {
            let d = root.join(name);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("blob.bin"), vec![0u8; 128]).unwrap();
        }
        // Both are index-less, but `active` is immune and `torn` is only
        // eligible once it clears the 1h torn grace. Backdate via `now`.
        let far_future = unix_now() + 2 * HOUR;
        let report = run(&root, "active", 0, 0, far_future, true).unwrap();

        assert_eq!(report.evicted, 1);
        assert_eq!(report.failed, 0);
        assert!(report.reclaimed_bytes >= 128);
        assert!(root.join("active").exists(), "active dir must survive");
        assert!(!root.join("torn").exists(), "torn dir must be gone");

        // Second pass has nothing left to do.
        assert_eq!(
            run(&root, "active", 0, 0, far_future, true)
                .unwrap()
                .evicted,
            0
        );

        // An absent root is a no-op, not an error.
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(
            run(&root, "active", 0, 0, far_future, true)
                .unwrap()
                .evicted,
            0
        );
    }

    #[test]
    fn opted_out_eviction_deletes_nothing_even_far_over_budget() {
        let root = tempdir("disabled");
        for name in ["aaa", "bbb", "ccc"] {
            let d = root.join(name);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("blob.bin"), vec![0u8; 4096]).unwrap();
        }
        // The harshest possible demand: budget 0, keep 1, no active dir to
        // protect, and every dir well past both grace periods.
        let far_future = unix_now() + 2 * HOUR;
        let report = run(&root, "", 0, 1, far_future, false).unwrap();

        assert_eq!(
            report,
            EvictionReport::default(),
            "opted-out eviction must report no work at all"
        );
        for name in ["aaa", "bbb", "ccc"] {
            assert!(
                root.join(name).exists(),
                "{name} was deleted despite ATLAS_WEIGHT_CACHE_EVICT opt-out"
            );
        }

        // Same call with the gate open proves the fixture really was
        // evictable — otherwise the assertions above would pass vacuously.
        let report = run(&root, "", 0, 1, far_future, true).unwrap();
        assert_eq!(report.evicted, 3);
        assert_eq!(report.failed, 0);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn eviction_is_on_by_default_and_opts_out_via_env() {
        // SAFETY: single-threaded test; this is the only reader of the var.
        unsafe { std::env::remove_var("ATLAS_WEIGHT_CACHE_EVICT") };
        assert!(eviction_enabled(), "must default to ON when unset");

        for off in ["0", "false", "off", "no", "FALSE", " Off "] {
            unsafe { std::env::set_var("ATLAS_WEIGHT_CACHE_EVICT", off) };
            assert!(!eviction_enabled(), "{off:?} must opt out");
        }

        for on in ["1", "true", "yes", "anything-else"] {
            unsafe { std::env::set_var("ATLAS_WEIGHT_CACHE_EVICT", on) };
            assert!(eviction_enabled(), "{on:?} must leave eviction on");
        }

        unsafe { std::env::remove_var("ATLAS_WEIGHT_CACHE_EVICT") };
    }

    #[test]
    fn budget_and_keep_fall_back_to_defaults_on_garbage() {
        // SAFETY: single-threaded test; these vars have no other reader.
        unsafe { std::env::set_var("ATLAS_WEIGHT_CACHE_MAX_GIB", "not-a-number") };
        assert_eq!(budget_bytes(), DEFAULT_MAX_GIB * BYTES_PER_GIB);

        unsafe { std::env::set_var("ATLAS_WEIGHT_CACHE_MAX_GIB", " 64 ") };
        assert_eq!(budget_bytes(), 64 * BYTES_PER_GIB, "whitespace is trimmed");

        unsafe { std::env::remove_var("ATLAS_WEIGHT_CACHE_MAX_GIB") };
        assert_eq!(budget_bytes(), DEFAULT_MAX_GIB * BYTES_PER_GIB);
        assert_eq!(keep_limit(), 0, "keep is off by default");
    }
}
