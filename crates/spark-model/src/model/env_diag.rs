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

use anyhow::{Result, bail};

#[inline]
fn diag_bool_value(raw: Option<&str>) -> bool {
    raw.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Fail-closed K=1 controls for cross-sequence DFlash target bisection.
///
/// Only named diagnostic families are allowed. FFN+layer-norms tests the
/// partial layer contribution found by C1; adding LM-head serialization tests
/// the remaining output projection while deliberately leaving final norm wide.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DflashSerialControls {
    pub ffn: bool,
    pub layer_norms: bool,
    pub final_norm: bool,
    pub lm_head: bool,
}

impl DflashSerialControls {
    pub fn current() -> Self {
        static FFN: OnceLock<bool> = OnceLock::new();
        static LAYER_NORMS: OnceLock<bool> = OnceLock::new();
        static FINAL_NORM: OnceLock<bool> = OnceLock::new();
        static LM_HEAD: OnceLock<bool> = OnceLock::new();

        Self {
            ffn: *FFN.get_or_init(|| {
                diag_bool_value(std::env::var("ATLAS_DFLASH_SERIAL_FFN").ok().as_deref())
            }),
            layer_norms: *LAYER_NORMS.get_or_init(|| {
                diag_bool_value(
                    std::env::var("ATLAS_DFLASH_SERIAL_LAYER_NORMS")
                        .ok()
                        .as_deref(),
                )
            }),
            final_norm: *FINAL_NORM.get_or_init(|| {
                diag_bool_value(
                    std::env::var("ATLAS_DFLASH_SERIAL_FINAL_NORM")
                        .ok()
                        .as_deref(),
                )
            }),
            lm_head: *LM_HEAD.get_or_init(|| {
                diag_bool_value(std::env::var("ATLAS_DFLASH_SERIAL_LM_HEAD").ok().as_deref())
            }),
        }
    }

    /// Returns the one active family, or `None` with normal defaults.
    pub fn active_family(self) -> Result<Option<&'static str>> {
        match (self.ffn, self.layer_norms, self.final_norm, self.lm_head) {
            (false, false, false, false) => Ok(None),
            (true, false, false, false) => Ok(Some("ffn")),
            (false, true, false, false) => Ok(Some("layer_norms")),
            (true, true, false, false) => Ok(Some("ffn_layer_norms")),
            (true, true, false, true) => Ok(Some("ffn_layer_norms_lm_head")),
            (false, false, true, false) => Ok(Some("final_norm")),
            (false, false, false, true) => Ok(Some("lm_head")),
            _ => bail!(
                "DFLASH_K1_BISECT invalid serial-family combination; only \
                 FFN+LAYER_NORMS and FFN+LAYER_NORMS+LM_HEAD may be combined"
            ),
        }
    }
}

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

/// `ATLAS_DUMP_CTX_HIDDEN` env var, cached. Returns the append-file path when
/// set (non-empty), `None` otherwise. This is the DFlash drafter-retrain
/// teacher-forced capture: after a prefill completes, the per-sequence
/// `ctx_hidden_acc` (all-position × 5-capture-layer hidden states, computed
/// on the NVFP4 serving path) is dumped to this file, one record per request.
/// Distinct from `ATLAS_DUMP_HIDDEN` (verify-path, generation-time).
#[inline]
pub fn dump_ctx_hidden_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var("ATLAS_DUMP_CTX_HIDDEN")
            .ok()
            .filter(|s| !s.is_empty())
    })
    .as_deref()
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

#[cfg(test)]
mod tests {
    use super::{DflashSerialControls, diag_bool_value};

    #[test]
    fn dflash_serial_bool_parser_is_explicit() {
        assert!(!diag_bool_value(None));
        assert!(diag_bool_value(Some("1")));
        assert!(diag_bool_value(Some("true")));
        assert!(diag_bool_value(Some("TRUE")));
        assert!(!diag_bool_value(Some("0")));
        assert!(!diag_bool_value(Some("yes")));
        assert!(!diag_bool_value(Some(" true ")));
    }

    #[test]
    fn dflash_serial_controls_require_one_family_at_most() {
        assert_eq!(
            DflashSerialControls::default().active_family().unwrap(),
            None
        );
        assert_eq!(
            DflashSerialControls {
                ffn: true,
                ..Default::default()
            }
            .active_family()
            .unwrap(),
            Some("ffn")
        );
        assert_eq!(
            DflashSerialControls {
                layer_norms: true,
                ..Default::default()
            }
            .active_family()
            .unwrap(),
            Some("layer_norms")
        );
        assert_eq!(
            DflashSerialControls {
                ffn: true,
                layer_norms: true,
                ..Default::default()
            }
            .active_family()
            .unwrap(),
            Some("ffn_layer_norms")
        );
        assert_eq!(
            DflashSerialControls {
                ffn: true,
                layer_norms: true,
                lm_head: true,
                ..Default::default()
            }
            .active_family()
            .unwrap(),
            Some("ffn_layer_norms_lm_head")
        );
        assert_eq!(
            DflashSerialControls {
                final_norm: true,
                ..Default::default()
            }
            .active_family()
            .unwrap(),
            Some("final_norm")
        );
        assert_eq!(
            DflashSerialControls {
                lm_head: true,
                ..Default::default()
            }
            .active_family()
            .unwrap(),
            Some("lm_head")
        );
        assert!(
            DflashSerialControls {
                ffn: true,
                layer_norms: true,
                final_norm: true,
                lm_head: false,
            }
            .active_family()
            .is_err()
        );
    }
}
