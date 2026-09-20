// SPDX-License-Identifier: AGPL-3.0-only

//! Layer-major text prefill for the single-sequence GLM-5.3 EXL3 target.
//!
//! The exact-wide verifier remains a causal target forward over 2..=8 rows.
//! The explicit layer-major prompt mode extends the same transition to M2048,
//! batching row-independent projections while retaining causal state updates.
//! Mixed vision/token prompts use either the exact-wide M<=8 input seam or an
//! independently gated layer-major fill. Both layer-major paths reuse the same
//! bounded DFlash2 capture owner and checked ingestion.

use std::ops::Range;

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::layers::ops::{with_glm53_exact_wide_prefill, with_glm53_layer_major_prefill};

use super::target_model_exl3::Glm53Exl3Model;

const MAX_WIDE_ROWS: usize = 8;
const MAX_LAYER_MAJOR_ROWS: usize = 2_048;
const VOCAB: usize = 154_880;
const BF16_BYTES: usize = 2;

#[cfg(test)]
#[path = "prefill_request_mode_tests.rs"]
mod request_mode_tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct WidePrefillConfig {
    rows: usize,
    layer_major: bool,
}

impl WidePrefillConfig {
    pub(super) fn layer_major_rows(self) -> Option<usize> {
        self.layer_major.then_some(self.rows)
    }

    pub(super) fn is_layer_major(self) -> bool {
        self.layer_major
    }
}

pub(super) fn requested_wide_prefill(
    has_prepared_images: bool,
) -> Result<Option<WidePrefillConfig>> {
    // Large-row text and bounded mixed-image execution are independent
    // opt-ins. Never forward a text-only layer-major config to mixed inputs.
    let layer_major = std::env::var("ATLAS_GLM53_LAYER_MAJOR_PREFILL").as_deref() == Ok("1");
    let mixed_layer_major =
        std::env::var("ATLAS_GLM53_LAYER_MAJOR_VISION_PREFILL").as_deref() == Ok("1");
    ensure!(
        !mixed_layer_major || layer_major,
        "ATLAS_GLM53_LAYER_MAJOR_VISION_PREFILL requires ATLAS_GLM53_LAYER_MAJOR_PREFILL=1"
    );
    if layer_major && (!has_prepared_images || mixed_layer_major) {
        let rows = std::env::var("ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(MAX_LAYER_MAJOR_ROWS);
        ensure!(
            matches!(rows, 16 | 32 | 64 | 128 | 256 | 512 | 1_024 | 2_048),
            "ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS must be 16, 32, 64, 128, 256, 512, 1024, or 2048"
        );
        return Ok(Some(WidePrefillConfig {
            rows,
            layer_major: true,
        }));
    }
    if std::env::var_os("ATLAS_GLM53_WIDE_PREFILL").is_none() {
        return Ok(None);
    }
    let rows = std::env::var("ATLAS_GLM53_WIDE_PREFILL_ROWS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(MAX_WIDE_ROWS);
    ensure!(
        matches!(rows, 2 | 4 | 8),
        "ATLAS_GLM53_WIDE_PREFILL_ROWS must be 2, 4, or 8"
    );
    Ok(Some(WidePrefillConfig {
        rows,
        layer_major: false,
    }))
}

pub(super) fn mixed_prefill_rows(config: Option<WidePrefillConfig>) -> Result<usize> {
    let Some(config) = config else {
        return Ok(1);
    };
    ensure!(
        if config.layer_major {
            matches!(config.rows, 16 | 32 | 64 | 128 | 256 | 512 | 1_024 | 2_048)
        } else {
            config.rows <= MAX_WIDE_ROWS && matches!(config.rows, 2 | 4 | 8)
        },
        "GLM mixed image prefill row configuration is invalid"
    );
    Ok(config.rows)
}

pub(super) fn validate_prefill_mode(
    config: Option<WidePrefillConfig>,
    has_dflash2: bool,
) -> Result<()> {
    ensure!(
        !config.is_some_and(|config| config.layer_major) || !has_dflash2,
        "GLM layer-major prefill currently requires target-only serving"
    );
    Ok(())
}

fn chunk_ranges(tokens: usize, max_rows: usize) -> Result<Vec<Range<usize>>> {
    ensure!(tokens != 0, "GLM EXL3 wide prefill needs a token");
    ensure!(
        matches!(
            max_rows,
            2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1_024 | 2_048
        ),
        "GLM EXL3 prefill rows must be 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, or 2048"
    );
    let mut chunks = Vec::with_capacity(tokens.div_ceil(max_rows));
    let mut start = 0usize;
    while start < tokens {
        let remaining = tokens - start;
        let chunk_rows = remaining.min(max_rows);
        let end = start + chunk_rows;
        chunks.push(start..end);
        start = end;
    }
    Ok(chunks)
}

impl Glm53Exl3Model {
    pub(super) fn prefill_tokens_wide(
        &self,
        tokens: &[u32],
        config: WidePrefillConfig,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(!tokens.is_empty(), "GLM EXL3 wide prefill needs a token");
        let large_capture = config.layer_major && self.has_dflash2();
        if large_capture {
            ensure!(
                self.preflight_prefill_capture(Some(config), tokens.len())?,
                "GLM layer-major DFlash2 requires checked capture admission"
            );
        }
        let mut final_logits = self.logits_ptr();
        for range in chunk_ranges(tokens.len(), config.rows)? {
            let chunk = &tokens[range];
            if chunk.len() == 1 {
                final_logits = self.walk(chunk[0], stream)?;
                continue;
            }
            let logits = if large_capture {
                self.prefill_layer_major_dflash2(chunk, config.rows, stream)?
            } else if config.layer_major {
                with_glm53_layer_major_prefill(|| self.verify_tokens_full(chunk, stream))?
            } else {
                with_glm53_exact_wide_prefill(|| self.verify_tokens_full(chunk, stream))?
            };
            if !large_capture {
                self.observe_committed_wide_rows(u32::try_from(chunk.len())?, stream)?;
            }
            final_logits = logits.offset((chunk.len() - 1) * VOCAB * BF16_BYTES);
        }
        Ok(final_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_mode_preserves_serial_default_and_only_existing_small_opt_ins() {
        assert_eq!(mixed_prefill_rows(None).unwrap(), 1);
        for rows in [2, 4, 8] {
            assert_eq!(
                mixed_prefill_rows(Some(WidePrefillConfig {
                    rows,
                    layer_major: false,
                }))
                .unwrap(),
                rows
            );
        }
        for rows in [0, 1, 3, 7, 9, 16, 2048, usize::MAX] {
            assert!(
                mixed_prefill_rows(Some(WidePrefillConfig {
                    rows,
                    layer_major: false,
                }))
                .is_err()
            );
        }
        assert!(
            mixed_prefill_rows(Some(WidePrefillConfig {
                rows: 8,
                layer_major: true,
            }))
            .is_err()
        );
        assert!(
            validate_prefill_mode(
                Some(WidePrefillConfig {
                    rows: 2048,
                    layer_major: true,
                }),
                true
            )
            .is_err()
        );
    }

    #[test]
    fn plans_full_chunks_and_a_single_tail_without_gaps() {
        assert_eq!(chunk_ranges(1, 8).unwrap(), vec![0..1]);
        assert_eq!(chunk_ranges(8, 8).unwrap(), vec![0..8]);
        assert_eq!(chunk_ranges(9, 8).unwrap(), vec![0..8, 8..9]);
        assert_eq!(chunk_ranges(17, 8).unwrap(), vec![0..8, 8..16, 16..17]);
        assert_eq!(chunk_ranges(9, 4).unwrap(), vec![0..4, 4..8, 8..9]);
        assert_eq!(chunk_ranges(5, 2).unwrap(), vec![0..2, 2..4, 4..5]);
        assert_eq!(chunk_ranges(129, 128).unwrap(), vec![0..128, 128..129]);
        assert_eq!(
            chunk_ranges(2_049, 2_048).unwrap(),
            vec![0..2_048, 2_048..2_049]
        );
    }

    #[test]
    fn rejects_an_empty_prompt() {
        assert!(chunk_ranges(0, 8).is_err());
        assert!(chunk_ranges(1, 1).is_err());
        assert!(chunk_ranges(1, 3).is_err());
        assert!(chunk_ranges(1, 2_049).is_err());
    }
}
