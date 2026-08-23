// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed attestations for DFlash's diagnostic serial SSM paths.

use std::sync::Once;

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SerialPath {
    Qkvz,
    Ba,
    Recurrent,
    Out,
    Ffn,
    LayerNorms,
}

impl SerialPath {
    const fn canonical(self) -> &'static str {
        match self {
            Self::Qkvz => "ssm_qkvz",
            Self::Ba => "ssm_ba",
            Self::Recurrent => "ssm_recurrent",
            Self::Out => "ssm_out",
            Self::Ffn => "ffn",
            Self::LayerNorms => "layer_norms",
        }
    }

    fn proof_latch(self) -> &'static Once {
        static QKVZ: Once = Once::new();
        static BA: Once = Once::new();
        static RECURRENT: Once = Once::new();
        static OUT: Once = Once::new();
        static FFN: Once = Once::new();
        static LAYER_NORMS: Once = Once::new();

        match self {
            Self::Qkvz => &QKVZ,
            Self::Ba => &BA,
            Self::Recurrent => &RECURRENT,
            Self::Out => &OUT,
            Self::Ffn => &FFN,
            Self::LayerNorms => &LAYER_NORMS,
        }
    }

    fn proof_line(self, engaged: bool) -> String {
        format!(
            "DFLASH_SERIAL_PATH_PROOF path={} requested=true engaged={engaged}",
            self.canonical()
        )
    }
}

/// Validate a requested diagnostic path before the layer launches any work.
/// A request applies only to multi-row verify; ordinary K=1 oracle calls are
/// intentionally unaffected and are the implementation being matched.
pub(super) fn require_serial_path(
    path: SerialPath,
    requested: bool,
    num_tokens: usize,
    applicable: bool,
    requirement: &str,
) -> Result<bool> {
    if !requested || num_tokens <= 1 {
        return Ok(false);
    }
    if !applicable {
        bail!("{} requirement={requirement}", path.proof_line(false));
    }
    Ok(true)
}

/// Emit exactly one proof per process and canonical path. Call only from the
/// branch that has actually selected the requested serial implementation.
pub(super) fn prove_serial_path(path: SerialPath) {
    path.proof_latch()
        .call_once(|| tracing::warn!("{}", path.proof_line(true)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_proof_schema_is_exact() {
        let paths = [
            (SerialPath::Qkvz, "ssm_qkvz"),
            (SerialPath::Ba, "ssm_ba"),
            (SerialPath::Recurrent, "ssm_recurrent"),
            (SerialPath::Out, "ssm_out"),
            (SerialPath::Ffn, "ffn"),
            (SerialPath::LayerNorms, "layer_norms"),
        ];
        for (path, canonical) in paths {
            assert_eq!(path.canonical(), canonical);
            assert_eq!(
                path.proof_line(true),
                format!("DFLASH_SERIAL_PATH_PROOF path={canonical} requested=true engaged=true")
            );
        }
    }

    #[test]
    fn unrequested_and_k1_calls_do_not_engage() {
        assert!(!require_serial_path(SerialPath::Qkvz, false, 5, false, "unused").unwrap());
        assert!(!require_serial_path(SerialPath::Qkvz, true, 1, false, "unused").unwrap());
    }

    #[test]
    fn applicable_multirow_request_engages() {
        assert!(require_serial_path(SerialPath::Recurrent, true, 5, true, "unused").unwrap());
    }

    #[test]
    fn inapplicable_multirow_request_fails_closed() {
        for path in [
            SerialPath::Qkvz,
            SerialPath::Ba,
            SerialPath::Recurrent,
            SerialPath::Out,
            SerialPath::Ffn,
            SerialPath::LayerNorms,
        ] {
            let error = require_serial_path(path, true, 5, false, "missing prerequisite")
                .unwrap_err()
                .to_string();
            assert_eq!(
                error,
                format!(
                    "DFLASH_SERIAL_PATH_PROOF path={} requested=true engaged=false \
                     requirement=missing prerequisite",
                    path.canonical()
                )
            );
        }
    }
}
