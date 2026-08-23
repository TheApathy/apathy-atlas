// SPDX-License-Identifier: AGPL-3.0-only

//! Startup kernel-resolution audit.
//!
//! Every `GpuBackend::kernel(module, func)` lookup is recorded here with
//! whether it resolved and WHERE it was issued from. A MISSING optional kernel
//! (`layers::try_kernel` → `KernelHandle(0)`) silently falls back to a slower
//! dispatch path with no error — this tree has **167 `try_kernel` call sites**
//! and roughly forty runtime levers gated on `env == "1" && handle.0 != 0`, so
//! the second half of each of those conjunctions has never been checked. The
//! symptom of a miss is a throughput number nobody has a baseline for.
//!
//! Every kernel lookup in Atlas is EAGER: each one sits in a layer constructor
//! on the `serve_phases::build_model` path. By the time the model is built the
//! audit therefore holds the COMPLETE `(module, func)` set this model asks for,
//! which is what makes one boot yield the whole list for a target and makes
//! `--check-kernels` a usable sweep rather than a sampler.
//!
//! Ported from upstream `crates/spark-runtime/src/kernel_audit.rs` (`7761b965`)
//! in reduced form. **Deliberately not ported**, because each needs machinery
//! this tree does not have:
//!
//! * `seal` / late-miss `std::process::abort()` — upstream's fail-closed boot
//!   gate. It rides on `run_metrics`, which does not exist here, and it turns a
//!   silent fallback into a process abort. That is a behaviour change to a
//!   server currently under benchmark, and it is not what `--check-kernels`
//!   needs; this module is read-only.
//! * The required-vs-expected split against a target's MODEL.toml
//!   `[expected_absent]`. This tree's `atlas_kernels::TargetPtxSet` has no
//!   `expected_absent` (nor `shadowed_dropped`, nor `KERNEL_SET_HASH`) — they
//!   are upstream build-codegen products. Consequence: the report below lists
//!   EVERY unresolved lookup, including probes for architectures this model
//!   does not have. See UPSTREAM-PORT.md.

use std::collections::BTreeMap;
use std::panic::Location;
use std::sync::Mutex;

/// Raw lookup log for the live model. Appended by [`record`] from
/// `GpuBackend::kernel`; deduped by [`audit_rows`].
///
/// A plain static rather than a field on a metrics mailbox: this tree has no
/// `run_metrics`, and a model swap calls [`reset`] explicitly.
static LOG: Mutex<Vec<(String, String, bool, &'static Location<'static>)>> = Mutex::new(Vec::new());

/// One deduped kernel-resolution row.
///
/// `site` is the DISPATCH SITE — `file:line` of the `.kernel(…)` /
/// `try_kernel(…)` call, captured through `#[track_caller]`. A bare
/// `module::func` list is not actionable: the same module name is looked up
/// from a dozen constructors, and the fix is always "go to that line".
#[derive(Clone, Debug)]
pub struct AuditRow {
    pub module: String,
    pub func: String,
    /// True if ANY lookup of this `(module, func)` resolved.
    pub loaded: bool,
    /// Dispatch site of the first lookup of this pair.
    pub site: &'static Location<'static>,
}

impl AuditRow {
    /// `module::func` — the name the report prints.
    pub fn name(&self) -> String {
        format!("{}::{}", self.module, self.func)
    }
}

/// Record one kernel lookup. Cheap; called from `GpuBackend::kernel`.
///
/// `site` is the caller's `Location`, which the backend obtains from its own
/// `#[track_caller]` frame — this function cannot take it implicitly, because
/// its own caller is the backend, not the dispatch site.
pub fn record(module: &str, func: &str, loaded: bool, site: &'static Location<'static>) {
    if let Ok(mut v) = LOG.lock() {
        v.push((module.to_string(), func.to_string(), loaded, site));
    }
}

/// Clear the log for a new model load. A swap would otherwise leave the report
/// showing both models' lookups with no way to tell them apart.
pub fn reset() {
    if let Ok(mut v) = LOG.lock() {
        v.clear();
    }
}

/// Unresolved lookups for the live model — the count `--check-kernels` exits
/// with.
pub fn unresolved_lookups() -> u64 {
    audit_rows().iter().filter(|r| !r.loaded).count() as u64
}

/// Structured resolution rows: deduped `(module, func)`, sorted, `loaded` true
/// if ANY lookup of that pair resolved.
pub fn audit_rows() -> Vec<AuditRow> {
    let mut resolved: BTreeMap<(String, String), (bool, &'static Location<'static>)> =
        BTreeMap::new();
    if let Ok(v) = LOG.lock() {
        for (m, f, ok, site) in v.iter() {
            let e = resolved
                .entry((m.clone(), f.clone()))
                .or_insert((false, *site));
            e.0 = e.0 || *ok;
        }
    }
    resolved
        .into_iter()
        .map(|((module, func), (loaded, site))| AuditRow {
            module,
            func,
            loaded,
            site,
        })
        .collect()
}

/// The failed lookups.
pub fn failed_rows() -> Vec<AuditRow> {
    audit_rows().into_iter().filter(|r| !r.loaded).collect()
}

/// Per-module rollup of the embedded kernel set against what this model asked
/// for. `embedded` is the loaded target's `TargetPtxSet::modules`.
///
/// PLAIN ASCII on purpose: this is read through `docker logs`, `journalctl`
/// and a non-TTY pipe far more often than on a terminal, and box-drawing
/// survives none of them.
pub fn render_kernel_table(embedded: &[(&str, &str)]) -> String {
    let rows = audit_rows();
    let mut mod_resolved: BTreeMap<&str, bool> = BTreeMap::new();
    for r in &rows {
        let e = mod_resolved.entry(r.module.as_str()).or_insert(false);
        *e = *e || r.loaded;
    }

    let mut out = format!(
        "\n-- Kernel load audit -- {} modules embedded, {} distinct lookups --\n",
        embedded.len(),
        rows.len()
    );
    out.push_str(&format!("   {:<40} {}\n", "MODULE", "RESOLUTION"));
    out.push_str(&format!("   {}\n", "-".repeat(64)));
    let mut sorted: Vec<&&str> = embedded.iter().map(|(m, _)| m).collect();
    sorted.sort();
    for m in sorted {
        let res = match mod_resolved.get(*m) {
            Some(true) => "used",
            Some(false) => "** every lookup FAILED **",
            // Embedded but never requested by this model's dispatch. Not a
            // defect: one PTX bundle serves several architectures.
            None => "-",
        };
        out.push_str(&format!("   {m:<40} {res}\n"));
    }
    out
}

/// The unresolved-lookup report: the enumerated list FIRST, then the
/// remediation block ONCE at the end.
pub fn unresolved_report(failed: &[AuditRow], model: &str, arch: &str, quant: &str) -> String {
    let mut out = format!(
        "{} unresolved kernel lookup(s) for ({model}, {arch}, {quant}). Each one resolved to \
         handle 0, so its dispatch site is on a silent fallback path:\n",
        failed.len()
    );
    for (i, r) in failed.iter().enumerate() {
        out.push_str(&format!(
            "  {}. {}  at {}:{}\n",
            i + 1,
            r.name(),
            r.site.file(),
            r.site.line(),
        ));
    }
    out.push_str(
        "\nEach line is EITHER a probe this model should never have issued (gate it on config, \
         the way `qwen3_attention::init` does) OR a kernel that should have been compiled and \
         was not. Without a MODEL.toml [expected_absent] declaration this report cannot tell \
         the two apart, so triage the list once and gate the benign probes at their sites.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(module: &str, func: &str, loaded: bool) -> AuditRow {
        AuditRow {
            module: module.to_string(),
            func: func.to_string(),
            loaded,
            site: Location::caller(),
        }
    }

    /// POSITIVE: the report names the dispatch site, not just the kernel. A
    /// bare `module::func` is not actionable — the same module is looked up
    /// from a dozen constructors and the fix is always "go to that line".
    #[test]
    fn the_report_carries_the_dispatch_site() {
        let failed = vec![row("w4a16_gemv", "w4a16_gemv_sw", false)];
        let text = unresolved_report(&failed, "qwen3.8-27b", "sm_121f", "nvfp4");
        assert!(text.contains("w4a16_gemv::w4a16_gemv_sw"));
        assert!(text.contains("kernel_audit.rs:"), "missing site: {text}");
        assert!(text.starts_with("1 unresolved kernel lookup(s)"));
    }

    /// POSITIVE: a pair looked up from two sites where one resolves is NOT a
    /// miss. `try_kernel` probes the same symbol from several constructors and
    /// only some models reach each; reporting the pair as failed because one
    /// caller missed would make the list mostly noise.
    #[test]
    fn a_pair_that_resolved_anywhere_is_not_a_miss() {
        reset();
        let site = Location::caller();
        record("m", "f", false, site);
        record("m", "f", true, site);
        record("m", "g", false, site);
        let rows = audit_rows();
        assert_eq!(rows.len(), 2, "deduped by (module, func)");
        assert!(rows.iter().find(|r| r.func == "f").unwrap().loaded);
        assert!(!rows.iter().find(|r| r.func == "g").unwrap().loaded);
        assert_eq!(unresolved_lookups(), 1);
        reset();
    }

    /// POSITIVE: an embedded module nobody asked for reads as `-`, not as a
    /// failure. One PTX bundle serves several architectures, so "embedded and
    /// unused" is the normal case and must not look like a defect.
    #[test]
    fn an_unrequested_embedded_module_is_not_a_failure() {
        reset();
        let table = render_kernel_table(&[("mla_absorb", "<ptx>"), ("w4a16_gemv", "<ptx>")]);
        assert!(table.contains("mla_absorb"));
        assert!(!table.contains("FAILED"));
        reset();
    }
}
