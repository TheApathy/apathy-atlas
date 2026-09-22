// SPDX-License-Identifier: AGPL-3.0-only
//! Metadata contract for the verified V4-Flash native indexer, not execution support.
//! Geometry follows official inference/model.py at 6821d6ad3681a4b137b066b76094fa82ebd0a380.
use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::ModelConfig;

const MAX_LAYERS: usize = 4096;

/// Which DeepSeek release an indexer block is being admitted for.
///
/// These are NOT interchangeable. V4-Flash-0731 and V4.1-Flash-Next differ in indexer
/// geometry AND in what `compress_ratios` means:
///
/// | | V4-Flash-0731 | V4.1-Flash-Next |
/// |---|---|---|
/// | (heads, head_dim, top_k) | (64, 128, 512) | (32, 128, 512) |
/// | (hidden, q_lora, qk_rope) | (4096, 1024, 64) | (5120, 1280, 64) |
/// | `compress_ratios` alphabet | {0, 4, 128} | {0, 1, 2} |
///
/// **UNRESOLVED, and it must be settled before any attention kernel READS these ratios.**
/// 0731's {0, 4, 128} read as literal compression factors. V4.1's {0, 1, 2} cannot be
/// literal factors (a ratio of 1 is a no-op and 2 is implausibly weak next to 128), so they
/// are presumably a MODE index — but nothing on this box states the mapping, and guessing it
/// would silently select the wrong attention path per layer. This enum admits the V4.1
/// alphabet for CONFIG PARSING ONLY (S0's deliverable is that shapes load); it deliberately
/// does not assign the values a meaning.
///
/// A second oddity, recorded not resolved: V4.1's `compress_ratios` has **43 entries for a
/// 40-layer model** — 43 being exactly 0731's layer count. Either the array is stale in the
/// checkpoint or the trailing entries are meaningful. The length check tolerates it; no
/// consumer should index past `num_hidden_layers` until this is answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IndexerVariant {
    V4Flash,
    V41FlashNext,
}

impl IndexerVariant {
    fn admits_ratio(self, ratio: u64) -> bool {
        match self {
            Self::V4Flash => matches!(ratio, 0 | 4 | 128),
            // Deliberately enumerated, NOT written as `0..=2`. These are an ALPHABET of
            // mode indices whose meaning is unknown (see `IndexerVariant`), not a numeric
            // range — a range spelling would imply an ordering and a continuity we have no
            // evidence for, and would silently admit a future `3`.
            #[allow(clippy::manual_range_patterns)]
            Self::V41FlashNext => matches!(ratio, 0 | 1 | 2),
        }
    }

    /// ((num_heads, head_dim, top_k), (hidden, q_lora_rank, qk_rope_head_dim))
    fn admitted_geometry(self) -> ((usize, usize, usize), (usize, usize, usize)) {
        match self {
            Self::V4Flash => ((64, 128, 512), (4096, 1024, 64)),
            Self::V41FlashNext => ((32, 128, 512), (5120, 1280, 64)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4IndexerConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub top_k: usize,
}

impl DeepSeekV4IndexerConfig {
    pub(super) fn parse_flat(raw: &Value) -> Result<Option<Self>> {
        Self::parse_flat_variant(raw, IndexerVariant::V4Flash)
    }

    /// DeepSeek-V4.1-Flash-Next admission.
    ///
    /// V4.1 is a DIFFERENT geometry and a DIFFERENT `compress_ratios` ENCODING from
    /// V4-Flash-0731, so it gets its own admission set rather than widening the V4-Flash
    /// whitelist (which exists to guard the one geometry that was verified against
    /// official inference/model.py). See `IndexerVariant` for the unresolved semantics.
    pub(super) fn parse_flat_v41(raw: &Value) -> Result<Option<Self>> {
        Self::parse_flat_variant(raw, IndexerVariant::V41FlashNext)
    }

    fn parse_flat_variant(raw: &Value, variant: IndexerVariant) -> Result<Option<Self>> {
        let object = raw
            .as_object()
            .context("DeepSeek config must be an object")?;
        let keys = ["index_n_heads", "index_head_dim", "index_topk"];
        if !keys.iter().any(|key| object.contains_key(*key)) {
            return Ok(None);
        }
        let integer = |key: &str| -> Result<usize> {
            let number = object
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("DeepSeek indexer requires integer {key}"))?;
            let number = usize::try_from(number).context("Indexer dimension exceeds host size")?;
            ensure!(number > 0, "DeepSeek indexer {key} must be positive");
            Ok(number)
        };
        let indexer = Self {
            num_heads: integer("index_n_heads")?,
            head_dim: integer("index_head_dim")?,
            top_k: integer("index_topk")?,
        };
        let layers = integer("num_hidden_layers")?;
        indexer.validate_geometry(
            integer("hidden_size")?,
            integer("q_lora_rank")?,
            integer("qk_rope_head_dim")?,
            layers,
            variant,
        )?;
        let ratios = object
            .get("compress_ratios")
            .and_then(Value::as_array)
            .context("DeepSeek indexer requires a compression schedule")?;
        ensure!(
            (layers..=MAX_LAYERS).contains(&ratios.len()),
            "DeepSeek indexer compression schedule is truncated or overlarge"
        );
        for (layer, ratio) in ratios.iter().enumerate() {
            ensure!(
                ratio.as_u64().is_some_and(|r| variant.admits_ratio(r)),
                "Unsupported DeepSeek indexer compression ratio at layer {layer}"
            );
        }
        Ok(Some(indexer))
    }

    /// Revalidate public config at the loader boundary; never infer absent fields.
    pub fn validate_model(&self, config: &ModelConfig) -> Result<()> {
        let variant = match config.model_type.as_str() {
            "deepseek_v4" => IndexerVariant::V4Flash,
            "deepseek_v41" => IndexerVariant::V41FlashNext,
            other => anyhow::bail!("Indexer requires DeepSeek-V4 or V4.1, got {other}"),
        };
        self.validate_geometry(
            config.hidden_size,
            config.q_lora_rank,
            config.qk_rope_head_dim,
            config.num_hidden_layers,
            variant,
        )?;
        ensure!(
            (config.num_hidden_layers..=MAX_LAYERS).contains(&config.compress_ratios.len()),
            "DeepSeek indexer compression schedule is truncated or overlarge"
        );
        ensure!(
            config
                .compress_ratios
                .iter()
                .all(|ratio| variant.admits_ratio(*ratio as u64)),
            "Unsupported DeepSeek indexer compression ratio"
        );
        Ok(())
    }

    fn validate_geometry(
        &self,
        hidden: usize,
        q_rank: usize,
        rope: usize,
        layers: usize,
        variant: IndexerVariant,
    ) -> Result<()> {
        let (heads, target) = variant.admitted_geometry();
        ensure!(
            (self.num_heads, self.head_dim, self.top_k) == heads,
            "Unsupported DeepSeek indexer geometry for {variant:?}: admitted {heads:?}, got {:?}",
            (self.num_heads, self.head_dim, self.top_k)
        );
        ensure!(
            (hidden, q_rank, rope) == target,
            "Unsupported DeepSeek indexer target geometry for {variant:?}: admitted {target:?}, got {:?}",
            (hidden, q_rank, rope)
        );
        ensure!(
            (1..=MAX_LAYERS).contains(&layers),
            "Invalid DeepSeek indexer layer count"
        );
        Ok(())
    }
}
