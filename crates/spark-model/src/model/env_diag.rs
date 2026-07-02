// SPDX-License-Identifier: AGPL-3.0-only

//! OnceLock-cached reads for diagnostic / runtime-toggle ATLAS_* env vars
//! that fire on per-token / per-decode hot paths.
//!
//! `std::env::var` does a getenv syscall + cstring conversion + alloc on every
//! call. For paths that read the same variable thousands of times per response
//! (decode loops, per-block dispatchers), the cumulative cost is non-trivial.
//! Helpers here cache the first read for the process lifetime — matching the
//! pattern used by `crate::layers::*_enabled()` for compute-path gates.

use std::sync::OnceLock;

/// `ATLAS_DUMP_HIDDEN` env var, cached. Returns the path string when set
/// (non-empty), `None` otherwise. Hot-path callers in `decode_a` /
/// `prefill_b` previously re-read this on every token.
#[inline]
pub fn dump_hidden_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var("ATLAS_DUMP_HIDDEN")
            .ok()
            .filter(|s| !s.is_empty())
    })
    .as_deref()
}

/// `ATLAS_DUMP_HIDDEN` set or unset (boolean form), cached. Used by callers
/// that only care about the on/off signal, not the path.
#[inline]
pub fn dump_hidden_enabled() -> bool {
    dump_hidden_path().is_some()
}

/// `ATLAS_DIAG_GEMMA4=1` or `=true`, cached. Per-decode-step Gemma-4
/// degeneration diagnostic — off by default in production.
#[inline]
pub fn diag_gemma4_enabled() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_DIAG_GEMMA4")
            .ok()
            .is_some_and(|v| v == "1" || v == "true")
    })
}

/// `ATLAS_MLA_PERSEQ_FALLBACK=1` or `=true`, cached. Disables the batched
/// MLA decode path in favor of a per-sequence loop — diagnostic-only.
#[inline]
pub fn mla_perseq_fallback_enabled() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_MLA_PERSEQ_FALLBACK")
            .ok()
            .is_some_and(|v| v == "1" || v == "true")
    })
}

/// `ATLAS_CONC_HSD=1` or `=true`, cached. Switches the n-seq decode SSM
/// dispatch to a concurrent host-side dispatch pattern — diagnostic-only.
#[inline]
pub fn conc_hsd_enabled() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_CONC_HSD")
            .ok()
            .is_some_and(|v| v == "1" || v == "true")
    })
}

/// `ATLAS_DFLASH_CAPTURE_THINKING=1` or `=true`, cached. Default-OFF.
///
/// When set, the thinking-phase plain-decode path (which bypasses the
/// DFlash propose/verify cycle, so its per-token target-hidden capture in
/// `dflash_hidden_save[0]` is normally never appended to `ctx_hidden_acc`)
/// ALSO appends each thinking token's captured 5-layer target hidden into
/// the per-seq `ctx_hidden_acc` accumulator at its absolute slot. This
/// fills the otherwise-ZERO ctx region spanning the thinking span so that,
/// when the answer phase begins, the DFlash drafter conditions on REAL
/// reasoning-context hidden states instead of zero-norm keys.
///
/// Drafter-conditioning only — target verify is unchanged, so committed
/// tokens stay byte-identical (raises acceptance, not output).
#[inline]
pub fn dflash_capture_thinking_enabled() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_CAPTURE_THINKING")
            .ok()
            .is_some_and(|v| v == "1" || v == "true")
    })
}
