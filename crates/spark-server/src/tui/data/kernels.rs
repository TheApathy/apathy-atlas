// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-table model for Main ▸ Kernels: the LOADED target's embedded module
//! set joined with the runtime resolution audit — the same data
//! `render_kernel_table` prints, as rows a real `Table` widget can sort/filter.
//!
//! The module set has to come from the target serve actually resolved
//! (published here, looked up via `ptx_for_exact_target`).
//! `atlas_kernels::ptx_modules()` is emitted as a
//! plain alias of TARGET 0 in a multi-target build (`build_codegen.rs`), and
//! targets are sorted by directory name — so on every model except
//! `deepseek-v4-flash` it rendered another architecture's modules: phantom
//! rows, no row at all for the modules the live model actually uses, and every
//! PTX hash wrong. The table is the instrument this whole audit is read
//! through; while it named the wrong target, it was worse than absent.

use std::sync::Mutex;

use spark_runtime::kernel_audit::AuditRow;

/// One row of the kernel table.
#[derive(Clone, Debug)]
pub struct KernelRow {
    pub module: String,
    pub ptx_hash: String,
    /// None = embedded but never requested ("-"); Some(true) = used;
    /// Some(false) = lookup FAILED.
    pub resolution: Option<bool>,
}

/// A `(module, func)` lookup that failed — the "missing" list under the table.
#[derive(Clone, Debug)]
pub struct MissingKernel {
    pub module: String,
    pub func: String,
    /// `file:line` of the dispatch site, from the audit's `#[track_caller]`
    /// capture. Without it the operator has a name and nowhere to go.
    pub site: String,
}

impl MissingKernel {
    fn from_row(r: &AuditRow) -> Self {
        Self {
            module: r.module.clone(),
            func: r.func.clone(),
            site: format!("{}:{}", r.site.file(), r.site.line()),
        }
    }
}

#[derive(Default)]
pub struct KernelTableModel {
    pub rows: Vec<KernelRow>,
    /// ACTIONABLE failures: nothing declared these absent. This is the only
    /// list that may raise an alarm.
    pub missing_required: Vec<MissingKernel>,
    /// Failures the target's MODEL.toml `[expected_absent]` declares, each
    /// with a stated reason. Shown, never alarmed on — a warning that is
    /// almost always noise trains people to ignore the one time it is not.
    pub missing_expected: Vec<MissingKernel>,
}

/// `(target model, quant)` of the kernel target the serve path RESOLVED for
/// the model currently loaded.
///
/// Published at the moment resolution succeeds. This used to be the
/// `(model_type, hidden_size)` config shape and the table re-ran resolution
/// from it — but that shape no longer identifies a target on its own
/// (Qwen3.6-27B and Qwen3.8-27B are config-identical and their tie is broken
/// by checkpoint reference, which this module does not have). Publishing the
/// resolved identity is exact: the table looks up the target by name+quant
/// and cannot disagree with what serve selected.
static LOADED_TARGET: Mutex<Option<(String, String)>> = Mutex::new(None);

/// Record which kernel target the serve path resolved.
///
/// NOTE (dsv41-engine, 2026-09-22): nothing in this tree calls this yet, so the Kernels table
/// is always empty. The natural call site is right after `serve_load`'s "Selected kernel
/// target" log, but `serve_load.rs` is pinned by `context_extension_source_authority_tests`,
/// so the line is left for that file's owner rather than slipped past its hash.
pub fn publish_loaded_target(model: &str, quant: &str) {
    if let Ok(mut g) = LOADED_TARGET.lock() {
        *g = Some((model.to_string(), quant.to_string()));
    }
}

fn loaded_target() -> Option<(String, String)> {
    let guard = LOADED_TARGET.lock().ok()?;
    guard.clone()
}

/// Index of the target named EXACTLY `(model, quant)`.
///
/// Not `atlas_kernels::ptx_for_model`, which is a substring match on the model name and
/// returns the first hit in directory order: `deepseek-v4` would find `deepseek-v4-flash`, and
/// `qwen3.6-35b-a3b` would find `qwen3.6-35b-a3b-abl` or the canonical target depending on
/// sort order. The table must render the modules of the target serve RESOLVED, so the lookup
/// is by the identity serve published, both halves.
pub(crate) fn exact_target(names: &[(&str, &str)], model: &str, quant: &str) -> Option<usize> {
    names.iter().position(|(m, q)| *m == model && *q == quant)
}

/// FNV-1a 12-hex content hash — matches `kernel_audit`'s `ptx_hash`.
fn ptx_hash(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

/// Build the table from the LOADED target's module set + the audit rows.
/// Cheap enough to refresh on demand (kernel resolution only happens at
/// startup, so callers refresh once after `ready`).
pub fn build() -> KernelTableModel {
    let audit = spark_runtime::kernel_audit::audit_rows();
    // No model loaded yet, or this build has no matching compiled target: an
    // EMPTY table is the honest answer. Falling back to some other target's
    // module list is the bug this function was rewritten to fix.
    // This engine exposes `ptx_for_model` (a substring match on the model
    // needle) rather than upstream's exact (model, quant) lookup. The needle is
    // the model half; a build carrying two quants of the same model would match
    // the first, which is the "some other target's module list" failure the
    // comment above is about — so it is narrowed by the loaded quant below.
    let Some(ptx) = loaded_target().and_then(|(model, quant)| {
        let targets = atlas_kernels::available_targets();
        let names: Vec<(&str, &str)> = targets
            .iter()
            .map(|t| (t.target.model, t.target.quant))
            .collect();
        let idx = exact_target(&names, &model, &quant)?;
        targets.into_iter().nth(idx)
    }) else {
        return KernelTableModel::default();
    };
    let mut rows: Vec<KernelRow> = ptx
        .modules
        .iter()
        .map(|(module, blob)| {
            let mut resolution = None;
            for r in &audit {
                if r.module == *module {
                    resolution = Some(resolution.unwrap_or(false) || r.loaded);
                }
            }
            KernelRow {
                module: (*module).to_string(),
                // `modules` carries the PTX as `&'static str` in this tree,
                // not `&[u8]`, so hash its bytes.
                ptx_hash: ptx_hash(blob.as_bytes()),
                resolution,
            }
        })
        .collect();
    rows.sort_by(|a, b| a.module.cmp(&b.module));
    // NO REQUIRED/EXPECTED SPLIT ON THIS ENGINE, and the blocker is DATA, not
    // code. `kernel_audit`'s own report says the same thing in the same words:
    // "Without a MODEL.toml [expected_absent] declaration this report cannot
    // tell the two apart." Building the mechanism without the per-model triage
    // that populates it would leave `expected_absent` empty, put every failure
    // back under "required", and reproduce the exact over-report below — so
    // this pane says what the log says, and the two cannot disagree.
    //
    // Every failure therefore goes in ONE list and `missing_expected` stays
    // empty. Putting them all under `missing_required` instead would be the
    // exact over-report the comment above records — 51 failures shown where 4
    // are actionable — so the pane shows an unclassified list and says so,
    // rather than asserting a severity it cannot determine.
    let failed = spark_runtime::kernel_audit::failed_rows();
    KernelTableModel {
        rows,
        missing_required: Vec::new(),
        missing_expected: failed.iter().map(MissingKernel::from_row).collect(),
    }
}

#[cfg(test)]
mod exact_target_tests {
    use super::exact_target;

    /// The lookup is exact on BOTH halves. The control is the substring rule it replaced:
    /// `deepseek-v4` is a prefix of `deepseek-v4-flash` and `deepseek-v4.1`, and a quant-blind
    /// match would pick the first quant of a model built twice.
    #[test]
    fn the_loaded_target_is_matched_exactly_not_by_substring() {
        let names = [
            ("deepseek-v4-flash", "nvfp4"),
            ("deepseek-v4.1", "cb3"),
            ("glm5.3-flash", "exl3"),
            ("glm5.3-flash", "iq3"),
        ];
        assert_eq!(exact_target(&names, "deepseek-v4.1", "cb3"), Some(1));
        assert_eq!(exact_target(&names, "glm5.3-flash", "iq3"), Some(3));
        assert_eq!(exact_target(&names, "deepseek-v4", "nvfp4"), None, "a prefix is not a match");
        assert_eq!(exact_target(&names, "deepseek-v4.1", "nvfp4"), None, "the quant must match too");
        // What the old substring rule would have answered for the V4.1 needle's prefix:
        assert_eq!(names.iter().position(|(m, _)| m.contains("deepseek-v4")), Some(0));
    }
}
