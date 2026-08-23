// SPDX-License-Identifier: AGPL-3.0-only

//! The `--check-kernels` dry-run kernel-resolution audit.
//!
//! Every kernel lookup in Atlas is EAGER: each `.kernel(…)` / `try_kernel(…)`
//! site sits in a constructor on the `build_model` path. By the time this runs,
//! `spark_runtime::kernel_audit` therefore holds the COMPLETE `(module, func)`
//! set this model asks for — which is what makes ONE BOOT yield the whole list
//! for a target, and what makes `--check-kernels` a usable sweep rather than a
//! sampler.
//!
//! Ported from upstream `crates/spark-server/src/main_modules/serve_phases/
//! kernel_gate.rs` (`7761b965`), **diagnostic half only**. Upstream's
//! `audit_and_gate` also seals the audit, aborts on a post-seal lookup, and
//! refuses to serve when anything is unresolved unless
//! `--dangerously-allow-unresolved-kernel-lookups` is passed. None of that is
//! here: the seal rides on `run_metrics` (absent in this tree), and turning a
//! silent fallback into a refusal-to-serve is a behaviour change to a server
//! that is currently mid-campaign. A normal serve is entirely unaffected by
//! this file. See `qwen38/analysis/UPSTREAM-PORT.md`.

use spark_runtime::kernel_audit;

/// POSIX exit statuses are 8 bits. An unclamped count of exactly 256 would be
/// reported as 0 — a catastrophically broken model reading as a clean pass,
/// which is the worst possible failure for a tool whose only job is to be
/// trustworthy. The clamp is announced in the output whenever it bites, so the
/// number in `$?` is never silently wrong.
const MAX_EXIT_CODE: usize = 255;

/// Print the audit and exit with the unresolved count. Never returns.
///
/// Owning the exit here keeps the count and the status in one place — routing
/// it back through `Result` would collapse every count to anyhow's 1.
pub(crate) fn check_and_exit(ptx_set: &atlas_kernels::TargetPtxSet) -> ! {
    use std::io::Write as _;

    let modules: Vec<(&str, &str)> = ptx_set.modules.iter().map(|(m, p)| (*m, *p)).collect();
    tracing::info!("{}", kernel_audit::render_kernel_table(&modules));

    let rows = kernel_audit::audit_rows();
    let failed = kernel_audit::failed_rows();
    let target = &ptx_set.target;
    let n = failed.len();

    if n == 0 {
        tracing::info!(
            "kernel check PASSED for ({}, {}, {}): {} distinct lookups, all resolved",
            target.model,
            target.arch,
            target.quant,
            rows.len(),
        );
    } else {
        tracing::error!(
            "{}",
            kernel_audit::unresolved_report(&failed, target.model, target.arch, target.quant)
        );
    }

    let code = exit_code_for(n);
    if code != n {
        // Unmissable, on both streams: `$?` is about to under-report.
        let msg = format!("{n} unresolved kernels (exit code clamped to {MAX_EXIT_CODE})");
        tracing::error!("{msg}");
        println!("{msg}");
    }
    // Machine-readable result on ONE line, after the human report, so a sweep
    // across every target aggregates without parsing prose.
    println!("{}", check_json(&rows, &failed, ptx_set, code));
    // `exit` runs no destructors, so flush what a pipe would otherwise lose.
    let _ = std::io::stdout().flush();
    std::process::exit(code as i32);
}

/// The process status for `n` unresolved kernels.
///
/// The contract is "the exit code IS the count", so this is identity up to the
/// 8-bit POSIX ceiling. The clamp exists because 256 would be reported as 0 —
/// a catastrophically broken model reading as a clean pass. Clamping to 255
/// keeps a broken target non-zero, and the caller announces whenever the clamp
/// bit so `$?` is never silently wrong.
fn exit_code_for(n: usize) -> usize {
    n.min(MAX_EXIT_CODE)
}

/// One compact JSON object summarising the check. `ok` is the exit-code twin.
fn check_json(
    rows: &[kernel_audit::AuditRow],
    failed: &[kernel_audit::AuditRow],
    ptx_set: &atlas_kernels::TargetPtxSet,
    exit_code: usize,
) -> String {
    let unresolved: Vec<serde_json::Value> = failed
        .iter()
        .map(|r| {
            serde_json::json!({
                "kernel": r.name(),
                "site": format!("{}:{}", r.site.file(), r.site.line()),
            })
        })
        .collect();
    serde_json::json!({
        "atlas_kernel_check": {
            "model": ptx_set.target.model,
            "arch": ptx_set.target.arch,
            "quant": ptx_set.target.quant,
            "modules_embedded": ptx_set.modules.len(),
            "lookups": rows.len(),
            "unresolved": failed.len(),
            "ok": failed.is_empty(),
            // The status this process is about to exit with. Differs from
            // `unresolved` only when the 8-bit ceiling clamped it.
            "exit_code": exit_code,
            "unresolved_kernels": unresolved,
        }
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::{MAX_EXIT_CODE, exit_code_for};

    #[test]
    fn the_exit_code_is_the_unresolved_count() {
        // The stated contract: `$?` equals the number of unresolved kernels.
        for n in [0usize, 1, 2, 15, 42, 254, 255] {
            assert_eq!(exit_code_for(n), n, "exit code must equal the count");
        }
    }

    #[test]
    fn a_count_of_256_does_not_report_as_a_clean_pass() {
        // ★ The reason the clamp exists. POSIX statuses are 8 bits, so an
        // unclamped 256 arrives as 0 — the most broken possible target reading
        // as "every lookup resolved". Anything at or above the ceiling must
        // stay non-zero.
        assert_eq!(exit_code_for(256), MAX_EXIT_CODE);
        assert_eq!(exit_code_for(1000), MAX_EXIT_CODE);
        for n in [256usize, 512, 4096] {
            assert_ne!(exit_code_for(n) % 256, 0, "{n} must not read as success");
        }
    }
}
