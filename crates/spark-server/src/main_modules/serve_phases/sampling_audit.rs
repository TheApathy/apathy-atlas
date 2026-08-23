// SPDX-License-Identifier: AGPL-3.0-only

//! Startup-only audit logging for the effective per-category sampling policy.

use atlas_kernels::{SamplingCategory, SamplingPresets};

#[derive(Debug, Clone, Copy, PartialEq)]
struct SamplingPresetAudit {
    category: &'static str,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repetition_penalty: f32,
    lz_penalty: f32,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: u32,
}

impl SamplingPresetAudit {
    fn new(category: &'static str, preset: &SamplingCategory, min_p: f32) -> Self {
        Self {
            category,
            temperature: preset.temperature,
            top_p: preset.top_p,
            top_k: preset.top_k,
            min_p,
            presence_penalty: preset.presence_penalty,
            frequency_penalty: preset.frequency_penalty,
            repetition_penalty: preset.repetition_penalty,
            lz_penalty: preset.lz_penalty,
            dry_multiplier: preset.dry_multiplier,
            dry_base: preset.dry_base,
            dry_allowed_length: preset.dry_allowed_length,
        }
    }
}

fn audit_rows(presets: &SamplingPresets, resolved_min_p: f32) -> [SamplingPresetAudit; 4] {
    [
        SamplingPresetAudit::new("thinking_text", &presets.thinking_text, resolved_min_p),
        SamplingPresetAudit::new("thinking_coding", &presets.thinking_coding, resolved_min_p),
        SamplingPresetAudit::new("non_thinking", &presets.non_thinking, resolved_min_p),
        SamplingPresetAudit::new("tools", &presets.tools, resolved_min_p),
    ]
}

pub(crate) fn log_sampling_presets(presets: &SamplingPresets, resolved_min_p: f32) {
    for row in audit_rows(presets, resolved_min_p) {
        tracing::info!(
            category = row.category,
            temperature = %row.temperature,
            top_p = %row.top_p,
            top_k = row.top_k,
            min_p = %row.min_p,
            presence_penalty = %row.presence_penalty,
            frequency_penalty = %row.frequency_penalty,
            repetition_penalty = %row.repetition_penalty,
            lz_penalty = %row.lz_penalty,
            dry_multiplier = %row.dry_multiplier,
            dry_base = %row.dry_base,
            dry_allowed_length = row.dry_allowed_length,
            "Resolved sampling preset"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn category(seed: f32) -> SamplingCategory {
        SamplingCategory {
            temperature: seed + 0.25,
            top_p: seed + 0.5,
            top_k: (seed * 10.0) as u32 + 3,
            min_p: Some(seed + 0.625),
            presence_penalty: seed + 0.75,
            frequency_penalty: seed + 1.0,
            repetition_penalty: seed + 1.25,
            lz_penalty: seed + 1.5,
            dry_multiplier: seed + 1.75,
            dry_base: seed + 2.0,
            dry_allowed_length: (seed * 10.0) as u32 + 2,
        }
    }

    #[test]
    fn audit_has_exactly_one_row_for_every_sampling_category() {
        let presets = SamplingPresets {
            thinking_text: category(1.0),
            thinking_coding: category(2.0),
            non_thinking: category(3.0),
            tools: category(4.0),
        };

        let rows = audit_rows(&presets, 0.08);
        assert_eq!(
            rows.map(|row| row.category),
            ["thinking_text", "thinking_coding", "non_thinking", "tools"]
        );
        assert_eq!(
            rows[0],
            SamplingPresetAudit {
                category: "thinking_text",
                temperature: 1.25,
                top_p: 1.5,
                top_k: 13,
                min_p: 0.08,
                presence_penalty: 1.75,
                frequency_penalty: 2.0,
                repetition_penalty: 2.25,
                lz_penalty: 2.5,
                dry_multiplier: 2.75,
                dry_base: 3.0,
                dry_allowed_length: 12,
            }
        );
        assert_eq!(rows.map(|row| row.temperature), [1.25, 2.25, 3.25, 4.25]);
    }

    #[test]
    fn audit_uses_the_effective_server_min_p() {
        let mut presets = SamplingPresets::default();
        presets.non_thinking.min_p = Some(0.31);

        let rows = audit_rows(&presets, 0.08);

        assert!(rows.iter().all(|row| row.min_p == 0.08));
    }
}
