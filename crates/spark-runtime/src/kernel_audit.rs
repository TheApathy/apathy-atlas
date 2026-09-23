// SPDX-License-Identifier: AGPL-3.0-only

//! Startup kernel-resolution audit + embedded-kernel-set table.
//!
//! Two halves, both printed once at model-load time:
//!  1. The EMBEDDED kernel set — every `(module, ptx)` compiled into this
//!     binary, with a per-kernel PTX content hash and the overall kernel-set
//!     hash. The count here is ground truth (e.g. 98 vs 99 modules), and the
//!     hashes pin exactly which kernel binary is loaded — so a stale/dropped
//!     kernel from a build-codegen regression is visible at a glance.
//!  2. The RESOLUTION audit — every `GpuBackend::kernel(module, func)` lookup
//!     and whether it resolved. A MISSING optional kernel (`try_kernel` →
//!     handle 0) silently falls back to a slower dispatch path with no error;
//!     this surfaces it (see the 2026-06-04 pipelined-GEMM regression where
//!     `w8a16_gemm_pipelined` resolved to 0 and QKVZ fell back to the ~4.6×
//!     slower `w8a16_gemm`).

use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, Mutex};

const MAX_AUDIT_ROWS: usize = 4096;

#[derive(Default)]
struct AuditState {
    modules: HashMap<String, HashMap<String, bool>>,
    rows: usize,
    truncated: bool,
}

impl AuditState {
    fn record(&mut self, module: &str, func: &str, loaded: bool, limit: usize) {
        if let Some(functions) = self.modules.get_mut(module) {
            if let Some(previous) = functions.get_mut(func) {
                *previous |= loaded;
                return;
            }
            if self.rows >= limit {
                self.truncated = true;
                return;
            }
            functions.insert(func.to_owned(), loaded);
            self.rows += 1;
            return;
        }
        if self.rows >= limit {
            self.truncated = true;
            return;
        }
        self.modules.insert(
            module.to_owned(),
            HashMap::from([(func.to_owned(), loaded)]),
        );
        self.rows += 1;
    }

    fn snapshot(&self) -> (Vec<(String, String, bool)>, bool) {
        let mut rows: Vec<_> = self
            .modules
            .iter()
            .flat_map(|(module, functions)| {
                functions
                    .iter()
                    .map(move |(func, loaded)| (module.clone(), func.clone(), *loaded))
            })
            .collect();
        rows.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
        (rows, self.truncated)
    }
}

/// Deduplicated `(module, func, loaded)` resolution state. Runtime lookup may
/// call `record` on every launch, so retaining lookup history is unbounded.
static AUDIT: LazyLock<Mutex<AuditState>> = LazyLock::new(|| Mutex::new(AuditState::default()));

/// Record one kernel lookup. Cheap; called from `GpuBackend::kernel`.
pub fn record(module: &str, func: &str, loaded: bool) {
    if let Ok(mut audit) = AUDIT.lock() {
        audit.record(module, func, loaded, MAX_AUDIT_ROWS);
    }
}

fn audit_snapshot() -> (Vec<(String, String, bool)>, bool) {
    AUDIT
        .lock()
        .map(|audit| audit.snapshot())
        .unwrap_or_default()
}

/// FNV-1a 64-bit content fingerprint → 12 hex chars (matches build.rs).
fn ptx_hash(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

/// Structured resolution rows for observers (e.g. the TUI kernel table):
/// deduped `(module, func, loaded)`, sorted. `loaded` is true if ANY lookup
/// of that (module, func) resolved.
pub fn audit_rows() -> Vec<(String, String, bool)> {
    audit_snapshot().0
}

/// True when distinct lookup names exceeded the defensive audit bound.
pub fn audit_truncated() -> bool {
    AUDIT.lock().map(|audit| audit.truncated).unwrap_or(false)
}

/// Render the embedded kernel set (`embedded` = the binary's `ptx_modules()`,
/// passed in since spark-runtime doesn't depend on atlas-kernels) plus the
/// runtime resolution overlay. `set_hash` is `atlas_kernels::KERNEL_SET_HASH`.
pub fn render_kernel_table(embedded: &[(&str, &[u8])], set_hash: &str) -> String {
    // Dedup resolution audit: (module, func) → loaded (true if ever true).
    let (rows, truncated) = audit_snapshot();
    let resolved: BTreeMap<(String, String), bool> = rows
        .into_iter()
        .map(|(module, func, loaded)| ((module, func), loaded))
        .collect();
    // Per-module resolution rollup: any-loaded / any-requested.
    let mut mod_resolved: BTreeMap<&str, (bool, bool)> = BTreeMap::new(); // (requested, loaded)
    for ((m, _f), ok) in &resolved {
        let e = mod_resolved.entry(m.as_str()).or_insert((false, false));
        e.0 = true;
        e.1 = e.1 || *ok;
    }

    let mut out = String::new();
    out.push_str(&format!(
        "\n┌─ Kernel load audit ─ {} kernels embedded · set-hash {} ─\n",
        embedded.len(),
        set_hash
    ));
    out.push_str(&format!(
        "│ {:<34} {:<14} {}\n",
        "MODULE (operation)", "PTX-HASH", "RESOLUTION"
    ));
    out.push_str(&format!("│ {}\n", "─".repeat(74)));
    let mut sorted: Vec<&(&str, &[u8])> = embedded.iter().collect();
    sorted.sort_by_key(|(m, _)| *m);
    for (m, blob) in sorted {
        // Blob is the raw kernel bytes (PTX text or AMD/Metal binary);
        // FNV-1a over the bytes directly — matches build.rs's set hash.
        let h = ptx_hash(blob);
        let res = match mod_resolved.get(m) {
            Some((_req, true)) => "used",
            Some((_req, false)) => "** lookup FAILED **",
            None => "-", // embedded but not requested by this model's dispatch
        };
        out.push_str(&format!("│ {m:<34} {h:<14} {res}\n"));
    }
    out.push_str("└─");

    // Explicit MISSING list: (module, func) requested but never resolved →
    // silent slower-fallback dispatch. The actionable debug signal.
    let missing: Vec<&(String, String)> = resolved
        .iter()
        .filter(|(_, ok)| !**ok)
        .map(|(k, _)| k)
        .collect();
    if !missing.is_empty() {
        out.push_str(&format!(
            "\n⚠ {} kernel lookup(s) MISSING — slower fallback dispatch in use:\n",
            missing.len()
        ));
        for (m, f) in &missing {
            out.push_str(&format!("    - {m}::{f}\n"));
        }
    }
    if truncated {
        out.push_str(&format!(
            "\n⚠ Kernel resolution audit truncated at {MAX_AUDIT_ROWS} distinct lookups.\n"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::AuditState;

    #[test]
    fn repeated_lookup_keeps_one_row_and_ors_loaded_status() {
        let mut audit = AuditState::default();
        for _ in 0..10_000 {
            audit.record("module", "function", false, 4);
        }
        audit.record("module", "function", true, 4);

        assert_eq!(
            audit.snapshot(),
            (vec![("module".into(), "function".into(), true)], false)
        );
        assert_eq!(audit.rows, 1);
    }

    #[test]
    fn distinct_lookup_bound_is_sticky_and_existing_rows_still_update() {
        let mut audit = AuditState::default();
        audit.record("a", "one", false, 2);
        audit.record("a", "two", true, 2);
        audit.record("b", "three", true, 2);
        audit.record("a", "one", true, 2);

        assert_eq!(
            audit.snapshot(),
            (
                vec![
                    ("a".into(), "one".into(), true),
                    ("a".into(), "two".into(), true),
                ],
                true,
            )
        );
        assert_eq!(audit.rows, 2);
    }
}
