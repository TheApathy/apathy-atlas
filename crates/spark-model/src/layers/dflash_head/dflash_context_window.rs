// SPDX-License-Identifier: AGPL-3.0-only

//! Pure, fail-closed resolution of the DFlash resident attention window.

use std::fmt;

/// The qualified resident ring used by the native Flash-Next V3 and DFlash2
/// checkpoints. Absolute positions may continue to 1M; only this tail is
/// resident in the drafter.
pub(crate) const NATIVE_FLASH_NEXT_WINDOW: usize = 4096;
pub(crate) const MAX_ABSOLUTE_CONTEXT: usize = 1_048_576;

const NATIVE_ARCHITECTURES: [&str; 2] = ["DFlashDraftModel", "DFlash2DraftModel"];
const NATIVE_TARGET_TAPS: [usize; 8] = [1, 7, 13, 20, 26, 33, 39, 46];

#[allow(clippy::too_many_arguments)]
pub(crate) fn is_native_flash_next(
    architectures: &[String],
    model_type: Option<&str>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_target_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    target_layer_ids: &[usize],
) -> bool {
    matches!(
        architectures,
        [architecture] if NATIVE_ARCHITECTURES.contains(&architecture.as_str())
    ) && model_type == Some("qwen3")
        && hidden_size == 2560
        && intermediate_size == 8704
        && num_hidden_layers == 6
        && num_target_layers == 48
        && num_attention_heads == 20
        && num_key_value_heads == 4
        && head_dim == 128
        && vocab_size == 248_320
        && target_layer_ids == NATIVE_TARGET_TAPS
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContextWindowError {
    ZeroMaxSequenceLength,
    AbsoluteContextTooLong { max_seq_len: usize },
    OutsideResidentRange { requested: usize },
    NativeFlashNextSequenceTooShort { max_seq_len: usize },
    NativeFlashNextMismatch { requested: usize },
}

impl fmt::Display for ContextWindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMaxSequenceLength => {
                f.write_str("DFlash requires max_seq_len greater than zero")
            }
            Self::AbsoluteContextTooLong { max_seq_len } => write!(
                f,
                "DFlash absolute context {max_seq_len} exceeds the qualified limit {MAX_ABSOLUTE_CONTEXT}"
            ),
            Self::OutsideResidentRange { requested } => write!(
                f,
                "DFlash resident context window {requested} is outside the qualified range 1..={NATIVE_FLASH_NEXT_WINDOW}"
            ),
            Self::NativeFlashNextSequenceTooShort { max_seq_len } => write!(
                f,
                "native Flash-Next V3/DFlash2 requires max_seq_len at least {NATIVE_FLASH_NEXT_WINDOW}; got {max_seq_len}"
            ),
            Self::NativeFlashNextMismatch { requested } => write!(
                f,
                "native Flash-Next V3/DFlash2 requires resident context window {NATIVE_FLASH_NEXT_WINDOW}; got {requested}"
            ),
        }
    }
}

/// Resolve the single value used for allocation and runtime attention.
///
/// `None` preserves the legacy CLI meaning of full attention, but only when
/// the configured maximum sequence length fits inside the qualified resident
/// ring. It never silently truncates a requested full window.
pub(crate) fn resolve_context_window(
    requested: Option<usize>,
    max_seq_len: usize,
    native_flash_next: bool,
) -> Result<usize, ContextWindowError> {
    if max_seq_len == 0 {
        return Err(ContextWindowError::ZeroMaxSequenceLength);
    }
    if max_seq_len > MAX_ABSOLUTE_CONTEXT {
        return Err(ContextWindowError::AbsoluteContextTooLong { max_seq_len });
    }
    if native_flash_next && max_seq_len < NATIVE_FLASH_NEXT_WINDOW {
        return Err(ContextWindowError::NativeFlashNextSequenceTooShort { max_seq_len });
    }

    let resolved = requested.unwrap_or(max_seq_len);
    if !(1..=NATIVE_FLASH_NEXT_WINDOW).contains(&resolved) {
        return Err(ContextWindowError::OutsideResidentRange {
            requested: resolved,
        });
    }
    if native_flash_next && resolved != NATIVE_FLASH_NEXT_WINDOW {
        return Err(ContextWindowError::NativeFlashNextMismatch {
            requested: resolved,
        });
    }
    Ok(resolved)
}
