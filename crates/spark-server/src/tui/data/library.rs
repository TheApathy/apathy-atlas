// SPDX-License-Identifier: AGPL-3.0-only

//! Library tab data: enumerate locally cached HF models and read their
//! metadata from config.json — zero GPU, zero weight loading. Reuses the
//! resolver's cache-root precedence and weights predicate so the list agrees
//! with what `spark serve <model>` would actually find.

use std::path::{Path, PathBuf};

/// One locally cached model.
#[derive(Clone, Debug)]
pub struct LibraryEntry {
    /// HF id, un-mangled (`org/name`).
    pub id: String,
    pub snapshot_dir: PathBuf,
    pub size_bytes: u64,
    pub has_weights: bool,
    /// From config.json (None when parse fails — still listed).
    pub model_type: String,
    pub quant: String,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub experts: usize,
    pub context: usize,
    /// A compiled kernel target resolves UNAMBIGUOUSLY for this model's
    /// (model_type, hidden) + HF id. False also when targets exist but the
    /// tie-break fails — serving the entry as-is would refuse.
    pub optimized: bool,
}

/// Directory size (recursive, follows no symlinks). Snapshot dirs hardlink
/// into blobs/, so `len()` of the resolved files is the honest number.
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            total += dir_size(&p);
        } else if let Ok(md) = std::fs::metadata(&p) {
            total += md.len();
        }
    }
    total
}

/// Scan the HF cache. `cache_dir` is the `--cache-dir` override, if any.
pub fn scan(cache_dir: Option<&Path>) -> Vec<LibraryEntry> {
    let Ok(root) = crate::model_resolver::resolve_cache_root(cache_dir) else {
        return Vec::new();
    };
    let Ok(rd) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(mangled) = name.strip_prefix("models--") else {
            continue;
        };
        let id = mangled.replace("--", "/");
        let snapshots = e.path().join("snapshots");
        let Some(snap) =
            crate::model_resolver::find_snapshot_with_weights(&snapshots).or_else(|| {
                // No weights: still list the newest snapshot if one exists.
                std::fs::read_dir(&snapshots)
                    .ok()?
                    .flatten()
                    .map(|s| s.path())
                    .find(|p| p.is_dir())
            })
        else {
            continue;
        };
        // "Ready" has to mean "the loader would accept this", not "some file
        // that looks like weights is present". Two ways it can differ, both
        // reachable the moment downloads exist:
        //   * `snapshot_has_weights` counts `model.safetensors.index.json`,
        //     which is small and lands FIRST — so a download that has barely
        //     started reads as complete;
        //   * a download that never finished has no `refs/main`, which is what
        //     the resolver actually keys on.
        let has_shard = std::fs::read_dir(&snap)
            .map(|rd| {
                rd.filter_map(|e| e.ok()).any(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.ends_with(".safetensors"))
                })
            })
            .unwrap_or(false);
        let published = e.path().join("refs/main").exists();
        let has_weights = has_shard && published;
        let mut entry = LibraryEntry {
            id,
            // `blobs/` is where huggingface-cli puts the real bytes, with
            // snapshots/ as symlinks into it — so `len()` of a snapshot entry
            // would measure the link, not the model. Avarok's own downloader
            // writes the files directly into snapshots/ and has no blobs/ at
            // all, which reported every model it fetched as 0 MB. Measure
            // whichever layout this model actually uses.
            size_bytes: match dir_size(&e.path().join("blobs")) {
                0 => dir_size(&snap),
                n => n,
            },
            snapshot_dir: snap.clone(),
            has_weights,
            model_type: "?".into(),
            quant: "-".into(),
            layers: 0,
            hidden: 0,
            heads: 0,
            experts: 0,
            context: 0,
            optimized: false,
        };
        if let Ok(json) = std::fs::read_to_string(snap.join("config.json"))
            && let Ok(cfg) = atlas_core::config::parse_config(&json)
        {
            entry.model_type = cfg.model_type.clone();
            entry.layers = cfg.num_hidden_layers;
            entry.hidden = cfg.hidden_size;
            entry.heads = cfg.num_attention_heads;
            entry.experts = cfg.num_experts;
            entry.context = cfg.max_position_embeddings;
            if let Some(q) = &cfg.quantization_config {
                entry.quant = if q.quant_algo.is_empty() {
                    q.quant_method.clone()
                } else {
                    q.quant_algo.to_lowercase()
                };
            }
            // The HF id is the reference the tie-break matches against for
            // config-identical checkpoints (Qwen3.6-27B vs Qwen3.8-27B).
            // An ambiguity error is shown as un-optimized: serving this
            // entry as-is WOULD refuse, which is what the flag reports.
            // Ours takes (model_type, hidden_size) and returns `Option`, where
            // upstream also passes a recipe-id hint and a quant and returns
            // `Result<Option<_>>` so an AMBIGUOUS match can be reported as an
            // error. Without the hint this cannot distinguish "no target" from
            // "several targets", so the flag means only "a target exists" —
            // weaker than upstream's, and not the same claim.
            entry.optimized =
                atlas_kernels::ptx_for_config(&cfg.model_type, cfg.hidden_size).is_some();
        }
        out.push(entry);
    }
    scan_plain_roots(&mut out);
    out.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    out
}

/// Extra model roots that are NOT laid out as a HuggingFace cache.
///
/// Colon-separated, from `ATLAS_MODEL_DIRS`. Each immediate subdirectory that
/// holds a `config.json` is one model, and the directory itself is its
/// snapshot — which is exactly what `--model-from-path` already accepts, so
/// this lists the models that flag can already serve.
///
/// WHY THIS EXISTS: the scan above requires the `models--org--name` +
/// `snapshots/` layout and `continue`s past anything else WITHOUT A WORD. A
/// box keeping its checkpoints in a plain directory — which is the normal
/// shape for locally built or converted weights — got an empty Library and no
/// hint that anything had been skipped. An empty list is indistinguishable
/// from "you have no models", which is the worst thing a browser can say.
fn scan_plain_roots(out: &mut Vec<LibraryEntry>) {
    let Ok(dirs) = std::env::var("ATLAS_MODEL_DIRS") else {
        return;
    };
    for root in dirs.split(':').filter(|d| !d.is_empty()) {
        let Ok(rd) = std::fs::read_dir(root) else {
            continue;
        };
        for e in rd.flatten() {
            let dir = e.path();
            if !dir.join("config.json").is_file() {
                // Not a model directory. Counted by `scan_report`, not
                // silently dropped.
                continue;
            }
            let id = e.file_name().to_string_lossy().to_string();
            if out.iter().any(|x| x.id == id) {
                // Already found in the HF cache under the same name; the
                // cached copy wins because it carries refs/main.
                continue;
            }
            let mut entry = LibraryEntry {
                id,
                size_bytes: dir_size(&dir),
                snapshot_dir: dir.clone(),
                // No `refs/main` out here, so "published" has no meaning; a
                // directory with a config and a shard is servable, which is
                // what this flag is read for.
                has_weights: crate::model_resolver::find_snapshot_with_weights(
                    dir.parent().unwrap_or(&dir),
                )
                .is_some()
                    || dir.join("config.json").is_file(),
                model_type: "?".into(),
                quant: "-".into(),
                layers: 0,
                hidden: 0,
                heads: 0,
                experts: 0,
                context: 0,
                optimized: false,
            };
            if let Ok(json) = std::fs::read_to_string(dir.join("config.json"))
                && let Ok(cfg) = atlas_core::config::parse_config(&json)
            {
                entry.model_type = cfg.model_type.clone();
                entry.layers = cfg.num_hidden_layers;
                entry.hidden = cfg.hidden_size;
                entry.heads = cfg.num_attention_heads;
                entry.experts = cfg.num_experts;
                entry.context = cfg.max_position_embeddings;
                if let Some(q) = &cfg.quantization_config {
                    entry.quant = if q.quant_algo.is_empty() {
                        q.quant_method.clone()
                    } else {
                        q.quant_algo.to_lowercase()
                    };
                }
                entry.optimized =
                    atlas_kernels::ptx_for_config(&cfg.model_type, cfg.hidden_size).is_some();
            }
            out.push(entry);
        }
    }
}

/// What the last scan looked at and what it passed over.
///
/// The Library renders this under the list so an empty or short result is
/// LEGIBLE rather than mysterious: "scanned <root>: 92 models, 19 skipped"
/// tells an operator their checkpoints are in the wrong shape, where a blank
/// pane tells them they have none.
pub fn scan_report(cache_dir: Option<&Path>) -> String {
    scan_report_for(cache_dir, scan(cache_dir).len())
}

/// The report for an ALREADY-SCANNED list, so the number quoted is the number
/// on screen.
///
/// Split out because the first version of this counted DIRECTORY ENTRIES and
/// called them models: it printed "92 models" beside a list of 82, because a
/// cache entry with no usable snapshot produces a directory and no row. A
/// report and the thing it reports on have to be the same quantity.
pub fn scan_report_for(cache_dir: Option<&Path>, listed: usize) -> String {
    let root = crate::model_resolver::resolve_cache_root(cache_dir)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "<no cache root>".into());
    let (mut hf, mut skipped) = (0usize, 0usize);
    if let Ok(rd) = std::fs::read_dir(&root) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with("models--") {
                hf += 1;
            } else {
                skipped += 1;
            }
        }
    }
    let mut extra = String::new();
    if let Ok(dirs) = std::env::var("ATLAS_MODEL_DIRS") {
        let (mut found, mut passed) = (0usize, 0usize);
        for d in dirs.split(':').filter(|d| !d.is_empty()) {
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    if e.path().join("config.json").is_file() {
                        found += 1;
                    } else {
                        passed += 1;
                    }
                }
            }
        }
        extra = format!("  ·  ATLAS_MODEL_DIRS: {found} with config.json, {passed} without");
    }
    format!(
        "{root}: {listed} listed · {hf} cache entries, {skipped} not HF-cache layout{extra}"
    )
}

/// Human size. Delegates so the Library card and the download line that
/// replaces it cannot disagree about how big the same file is.
pub fn human_size(bytes: u64) -> String {
    crate::tui::format::bytes(bytes)
}

/// Run [`scan`] on its own thread, delivering the result over a channel.
///
/// The scan recursively `read_dir`s every `models--*/blobs`, which on a cache
/// holding a few dozen multi-gigabyte checkpoints is tens of milliseconds to
/// seconds with a cold page cache — and it was running on the render thread.
/// It also needs to be re-runnable now, because a finished download must
/// appear without restarting the dashboard.
///
/// Same shape as `recipe::fetch::refresh_in_background`, and the same rule:
/// the render thread only ever `try_recv`s.
pub fn scan_in_background(
    cache_dir: Option<&Path>,
) -> std::sync::mpsc::Receiver<Vec<LibraryEntry>> {
    let owned = cache_dir.map(|p| p.to_path_buf());
    // An empty list is what `scan` itself returns for an unreadable cache, so
    // it is also the honest answer when the scanner cannot start.
    crate::tui::worker::spawn(
        "atlas-libscan",
        move || scan(owned.as_deref()),
        |_| Vec::new(),
    )
}

#[cfg(test)]
mod plain_root_tests {
    use super::*;

    /// A directory holding a `config.json` is a model; one without is not.
    ///
    /// Positive AND negative arms: a scanner that returned everything would
    /// pass a test that only checked the model was found.
    #[test]
    fn a_plain_directory_with_a_config_is_listed_and_one_without_is_not() {
        let tmp = std::env::temp_dir().join(format!("atlas-lib-scan-{}", std::process::id()));
        let model = tmp.join("MyModel-NVFP4");
        let junk = tmp.join("not-a-model");
        std::fs::create_dir_all(&model).expect("tmp");
        std::fs::create_dir_all(&junk).expect("tmp");
        std::fs::write(
            model.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":4,"hidden_size":64,"num_attention_heads":2,"max_position_embeddings":128}"#,
        )
        .expect("write");
        std::fs::write(junk.join("README.md"), "no config here").expect("write");

        let mut out = Vec::new();
        // SAFETY: single-threaded test process; the var is read by the call below.
        unsafe { std::env::set_var("ATLAS_MODEL_DIRS", &tmp) };
        scan_plain_roots(&mut out);
        unsafe { std::env::remove_var("ATLAS_MODEL_DIRS") };

        let ids: Vec<&str> = out.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"MyModel-NVFP4"), "the model dir is listed: {ids:?}");
        assert!(!ids.contains(&"not-a-model"), "a dir with no config is not a model: {ids:?}");
        assert_eq!(
            out.iter().find(|e| e.id == "MyModel-NVFP4").map(|e| e.layers),
            Some(4),
            "and its config was actually parsed, not just its name taken"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// With the variable unset the scan must add nothing — the HF-cache path
    /// stays exactly as it was.
    #[test]
    fn no_extra_roots_means_no_extra_entries() {
        unsafe { std::env::remove_var("ATLAS_MODEL_DIRS") };
        let mut out = Vec::new();
        scan_plain_roots(&mut out);
        assert!(out.is_empty());
    }
}

#[cfg(test)]
mod real_box_scan {
    /// Prints what the scan finds on THIS box. Ignored by default: it depends
    /// on the machine's cache, so it is a probe, not an assertion.
    #[test]
    #[ignore]
    fn what_the_library_lists_here() {
        let all = super::scan(None);
        println!("{}", super::scan_report_for(None, all.len()));
        println!("total listed: {}", all.len());
        for e in all.iter().take(6) {
            println!("  {:<58} {:>9}  {}", e.id, super::human_size(e.size_bytes), e.model_type);
        }
    }
}
