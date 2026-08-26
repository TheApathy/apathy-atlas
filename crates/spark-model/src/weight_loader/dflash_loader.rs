// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter weight loader.
//!
//! Loads `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash`-style drafter checkpoints into
//! the typed [`DflashWeights`] structure consumed by
//! [`crate::layers::BlockDiffusionDraftHead`]. The drafter is a small
//! Qwen3-architecture transformer (8 layers, hidden=2048, GQA 32:4) with
//! these distinctive parts vs. a vanilla Qwen3:
//!
//!  * `model.fc` — `[len(target_layer_ids) * target_hidden, draft_hidden]`
//!    BF16 projection that maps the stack of captured target hidden states
//!    into the drafter's input space.
//!  * `model.hidden_norm` — RMSNorm applied to the projected target context
//!    before mixing with token embeddings.
//!  * `lm_head` — drafter ships its own (NOT tied to target's), so
//!    `tie_word_embeddings=false`.
//!  * Optional `d2t` — draft-vocab → target-vocab id remap (absent when
//!    drafter shares vocab with target, as in Qwen3.6-35B-A3B-DFlash where
//!    both = 248320).
//!  * Special `mask_token_id` (`248070` for Qwen3.6-DFlash) used for the γ
//!    "to-be-predicted" positions in block diffusion.
//!
//! Under TP the drafter is **not sharded** — it's small (~1–2 GB BF16),
//! every rank loads the full set. Mirrors the existing MTP-under-TP pattern
//! (`MTP loads ALL experts on every rank — no EP all_reduce needed`).

use anyhow::{Context, Result};
use serde::Deserialize;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::dflash_validation::validate_dflash_store;
use crate::weight_map::{DenseWeight, dense};

/// Checkpoint-family contract for the otherwise shared DFlash/DSpark loader.
///
/// The family distinction is semantic, not cosmetic (Markov/confidence heads
/// belong only to DSpark). Legacy DFlash counts the known anchor in
/// `block_size`; published DSpark checkpoints count only the draft rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterCheckpointFamily {
    Dflash,
    Dspark,
}

/// Width contract for a checkpoint-backed parallel draft pass.
///
/// Current vLLM DSpark treats `num_speculative_tokens=2*block_size` as one
/// wider query block, not two feedback passes.  Keep that experimental parity
/// escape hatch both explicit and bounded: generic DFlash never expands, and
/// DSpark accepts only the exact 2x case when the caller opted in.
fn draft_width_is_supported(
    family: DrafterCheckpointFamily,
    trained_drafts: usize,
    requested_drafts: usize,
    allow_dspark_2x: bool,
) -> bool {
    requested_drafts <= trained_drafts
        || (family == DrafterCheckpointFamily::Dspark
            && allow_dspark_2x
            && requested_drafts == trained_drafts.saturating_mul(2))
}

/// DSpark target-verify planner mode. SGLang PR 34966 defaults to `static`,
/// which verifies the full gamma+1 window and does not consume confidence.
/// The dynamic modes need ragged layouts and a hardware-profiled SPS table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsparkVerifyMode {
    Static,
    CapAccept,
    Compact,
}

impl std::fmt::Display for DsparkVerifyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static => f.write_str("static"),
            Self::CapAccept => f.write_str("cap-accept"),
            Self::Compact => f.write_str("compact"),
        }
    }
}

impl std::str::FromStr for DsparkVerifyMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "static" => Ok(Self::Static),
            "cap-accept" => Ok(Self::CapAccept),
            "compact" => Ok(Self::Compact),
            _ => anyhow::bail!(
                "unsupported DSpark verify mode {value:?}; expected static, cap-accept, or compact"
            ),
        }
    }
}

/// Resolved learned-confidence feature contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsparkConfidenceConfig {
    pub with_markov: bool,
    pub input_dim: usize,
}

impl std::fmt::Display for DrafterCheckpointFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dflash => f.write_str("dflash"),
            Self::Dspark => f.write_str("dspark"),
        }
    }
}

/// Drafter HF `config.json` (subset Atlas consumes). Mirrors
/// `z-lab/Qwen3.6-35B-A3B-DFlash/config.json` field names verbatim so
/// `serde_json::from_str` works directly on the raw file.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashConfig {
    /// HF model class. `DSparkDraftModel` is load-bearing: it selects the
    /// DSpark Markov/confidence feature contract.
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub draft_vocab_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Family-specific training width. DFlash uses `B = anchor + drafts`;
    /// DSpark uses `B = drafts` and adds the target bonus only at verification.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// DFlash-specific nested config object.
    #[serde(default)]
    pub dflash_config: Option<DflashSubConfig>,
    /// Per-layer attention type. Qwen3.6-27B-DFlash ships
    /// `["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"]`
    /// — first 4 layers are SWA, last is full. Older drafters omit the field
    /// (treated as all `full_attention`).
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// SWA span (only meaningful when `layer_types` contains `sliding_attention`).
    /// Qwen3.6-27B-DFlash sets this to 2048.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Whether the drafter's block attention is causal.
    ///
    /// DFlash2 checkpoints ship `"is_causal": false` — the draft block is
    /// attended BIDIRECTIONALLY, because every masked position is predicted
    /// simultaneously and each may see all the others. Until 2026-08-20 this
    /// field was not parsed at all, so `from_weights.rs` derived causality
    /// purely from `layer_types` (`causals.push(is_sliding)`). DFlash2's
    /// `layer_types` is five `sliding_attention` entries and no
    /// `full_attention`, so every layer was run CAUSAL — the opposite of how
    /// the checkpoint was trained. Measured effect: acceptance 2.75/7 (39%)
    /// against 5.05/7 (72%) for the causally-trained qwen38-v2 drafter, i.e.
    /// dflash2 read as a large regression (28.28 vs 41.22 tok/s) when the
    /// mismatch alone could account for it.
    ///
    /// `None` (field absent) keeps the historical `layer_types`-derived
    /// behaviour, so every existing drafter is bit-identical.
    #[serde(default)]
    pub is_causal: Option<bool>,
    /// RoPE rotation base. Qwen3.6-27B-DFlash and 35B-DFlash both ship 10_000_000.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// RoPE scaling block. Older Qwen configs call this `rope_scaling`, while
    /// Transformers 5 / Qwen3.8 serializes the same block as
    /// `rope_parameters`. Accept both spellings: silently treating a Qwen3.8
    /// DSpark checkpoint as vanilla RoPE destroys draft acceptance.
    #[serde(default, alias = "rope_parameters")]
    pub rope_scaling: Option<DflashRopeScaling>,
    /// DSpark low-rank Markov head rank `r` (config key `markov_rank`).
    /// `0` (or absent) ⇒ no Markov head. The DSpark-AEON-draft checkpoint
    /// ships `markov_rank: 256` with a rank-256 `VanillaMarkov` head that
    /// adds a per-position bigram logit bias `B(prev) = W2(W1[prev])` and
    /// samples the block LEFT-TO-RIGHT (each position's chosen token biases
    /// the next). See `models/DSpark-AEON-draft/markov_head.py`.
    #[serde(default)]
    pub markov_rank: usize,
    /// DSpark Markov head variant (config key `markov_head_type`). Only
    /// `"vanilla"` is supported in Atlas (the +16-18% accepted-length default
    /// that ships in DSpark); `"gated"` requires the backbone hidden per
    /// position which Atlas's argmax-only propose path does not surface, so
    /// it falls back to vanilla with a warning.
    #[serde(default = "default_markov_head_type")]
    pub markov_head_type: String,
    /// Learned DSpark acceptance-confidence head. Atlas must not silently
    /// ignore this: the head changes dynamic/ragged verify scheduling.
    #[serde(default)]
    pub enable_confidence_head: Option<bool>,
    #[serde(default)]
    pub confidence_head_with_markov: Option<bool>,
}

impl DflashConfig {
    /// Resolve the training block width. Generic DFlash drafters (z-lab)
    /// serialize `block_size` at the config root; DFlash 2 (incoai) nests it
    /// inside `dflash_config` (block = 1 anchor + drafts). The nested value
    /// wins when present.
    pub fn resolved_block_size(&self) -> usize {
        self.dflash_config
            .as_ref()
            .and_then(|sub| (sub.block_size.is_some()).then(|| sub.block_size.expect("checked")))
            .unwrap_or(self.block_size)
    }

    /// True when the checkpoint declares the DFlash 2 contract: a
    /// `dflash_config.conv_kernel_size > 1` plus a candidate selector width.
    /// Both the conv and the selector must be executed together — the released
    /// checkpoint trains them jointly and neither can be dropped silently.
    pub fn is_dflash2(&self) -> bool {
        self.dflash_config.as_ref().is_some_and(|sub| {
            sub.conv_kernel_size > 1 && sub.selector_rank > 0 && sub.selector_top_k > 0
        })
    }

    /// Resolve the checkpoint family from two independent, explicit HF
    /// declarations. A disagreement fails closed instead of selecting block
    /// semantics from incidental tensors such as the Markov weights.
    pub fn checkpoint_family(&self) -> Result<DrafterCheckpointFamily> {
        let architecture_is_dspark = self
            .architectures
            .iter()
            .any(|name| name == "DSparkDraftModel");
        let projector = self
            .dflash_config
            .as_ref()
            .and_then(|sub| sub.projector_type.as_deref());
        let projector_is_dspark = projector == Some("dspark");

        match (architecture_is_dspark, projector_is_dspark) {
            (true, true) => Ok(DrafterCheckpointFamily::Dspark),
            (false, false) => Ok(DrafterCheckpointFamily::Dflash),
            (true, false) => anyhow::bail!(
                "Drafter architecture is DSparkDraftModel but \
                 dflash_config.projector_type={projector:?}; expected `dspark`"
            ),
            (false, true) => anyhow::bail!(
                "Drafter declares dflash_config.projector_type=`dspark` but architectures \
                 does not contain `DSparkDraftModel`"
            ),
        }
    }

    fn resolve_confidence_bool(
        &self,
        name: &str,
        top: Option<bool>,
        nested: Option<bool>,
        default: bool,
    ) -> Result<bool> {
        if let (Some(top), Some(nested)) = (top, nested) {
            anyhow::ensure!(
                top == nested,
                "DSpark confidence config disagrees for `{name}`: top-level={top}, \
                 dflash_config={nested}"
            );
        }
        Ok(top.or(nested).unwrap_or(default))
    }

    /// Resolve the trained confidence feature shape. This validates config
    /// semantics only; runtime planner support is checked separately.
    pub fn confidence_head_config(&self) -> Result<Option<DsparkConfidenceConfig>> {
        let family = self.checkpoint_family()?;
        let nested = self.dflash_config.as_ref();
        let enabled = self.resolve_confidence_bool(
            "enable_confidence_head",
            self.enable_confidence_head,
            nested.and_then(|sub| sub.enable_confidence_head),
            false,
        )?;
        let with_markov = self.resolve_confidence_bool(
            "confidence_head_with_markov",
            self.confidence_head_with_markov,
            nested.and_then(|sub| sub.confidence_head_with_markov),
            self.markov_rank > 0,
        )?;
        if !enabled {
            anyhow::ensure!(
                !with_markov
                    || (self.confidence_head_with_markov.is_none()
                        && nested
                            .and_then(|sub| sub.confidence_head_with_markov)
                            .is_none()),
                "confidence_head_with_markov=true requires enable_confidence_head=true"
            );
            return Ok(None);
        }
        anyhow::ensure!(
            family == DrafterCheckpointFamily::Dspark,
            "learned confidence-head semantics are supported only for explicit DSpark checkpoints"
        );
        if with_markov {
            anyhow::ensure!(
                self.markov_rank > 0,
                "confidence_head_with_markov=true requires markov_rank > 0"
            );
        }
        let input_dim = self
            .hidden_size
            .checked_add(if with_markov { self.markov_rank } else { 0 })
            .ok_or_else(|| anyhow::anyhow!("DSpark confidence input width overflow"))?;
        Ok(Some(DsparkConfidenceConfig {
            with_markov,
            input_dim,
        }))
    }

    /// Validate features whose silent omission would change checkpoint
    /// semantics. Kept separate from tensor validation so the server rejects
    /// an unsupported config before loading its multi-GB weight file.
    pub fn validate_supported_semantics(&self) -> Result<DrafterCheckpointFamily> {
        let family = self.checkpoint_family()?;
        if family == DrafterCheckpointFamily::Dspark {
            anyhow::ensure!(
                self.markov_rank > 0,
                "DSparkDraftModel requires markov_rank > 0"
            );
            anyhow::ensure!(
                self.markov_head_type.eq_ignore_ascii_case("vanilla"),
                "Atlas DSpark supports only markov_head_type=`vanilla`; got {:?}",
                self.markov_head_type
            );
        }
        self.confidence_head_config()?;
        Ok(family)
    }

    /// Keep dynamic confidence scheduling fail-closed until Atlas has the
    /// ragged verify layout and profiled SPS cost table used by SGLang.
    pub fn validate_verify_mode(&self, mode: DsparkVerifyMode) -> Result<()> {
        let family = self.validate_supported_semantics()?;
        if family == DrafterCheckpointFamily::Dflash {
            anyhow::ensure!(
                mode == DsparkVerifyMode::Static,
                "DSpark verify mode {mode} cannot be applied to a DFlash checkpoint"
            );
            return Ok(());
        }
        anyhow::ensure!(
            mode == DsparkVerifyMode::Static,
            "DSpark verify mode {mode} is not implemented in Atlas: dynamic cap-accept/compact \
             requires ragged verify layouts plus a hardware-profiled SPS cost table; use \
             --dspark-verify-mode static (full gamma+1 verify, matching SGLang PR 34966)"
        );
        Ok(())
    }

    /// Resolve the requested proposal count under this checkpoint family's
    /// training-width contract.
    pub fn resolve_draft_count(&self, requested: Option<usize>) -> Result<usize> {
        let family = self.validate_supported_semantics()?;
        let block_size = self.resolved_block_size();
        let trained_drafts = match family {
            DrafterCheckpointFamily::Dflash => {
                anyhow::ensure!(
                    block_size >= 2,
                    "DFlash checkpoint block_size={} cannot contain an anchor plus a draft",
                    block_size
                );
                block_size - 1
            }
            DrafterCheckpointFamily::Dspark => {
                anyhow::ensure!(
                    block_size >= 1,
                    "DSpark checkpoint block_size must contain at least one draft"
                );
                block_size
            }
        };
        let drafts = requested.unwrap_or(trained_drafts);
        anyhow::ensure!(drafts > 0, "{family} draft count must be non-zero");
        let dspark_width_multiple = family == DrafterCheckpointFamily::Dspark
            && std::env::var("ATLAS_DSPARK_WIDTH_MULTIPLE").ok().as_deref() == Some("1");
        anyhow::ensure!(
            draft_width_is_supported(family, trained_drafts, drafts, dspark_width_multiple,),
            "{family} requested {drafts} drafts, but checkpoint block_size={} supports at most \
             {trained_drafts} by default. DSpark permits exactly a 2x trained-width single \
             parallel pass only with ATLAS_DSPARK_WIDTH_MULTIPLE=1; use --dflash-gamma \
             {trained_drafts} or smaller otherwise",
            self.block_size
        );
        if drafts < trained_drafts {
            tracing::info!(
                "{family} draft width capped to {drafts}; checkpoint supports {trained_drafts} \
                 trained drafts (block_size={})",
                self.block_size
            );
        }
        Ok(drafts)
    }
}

fn default_markov_head_type() -> String {
    "vanilla".to_string()
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

/// HF `rope_scaling` block. Qwen3.5/3.6 DFlash drafters that DO use scaling
/// set `rope_type = "yarn"`. The 27B drafter omits the entire block.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeScaling {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// Transformers 5 places the RoPE base inside `rope_parameters` instead
    /// of at top level. Older configs leave this absent and use
    /// [`DflashConfig::rope_theta`].
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Explicit YaRN cos/sin multiplier. When absent, HF derives
    /// `1 + 0.1 * ln(factor)`.
    #[serde(default)]
    pub attention_factor: Option<f32>,
}

fn default_block_size() -> usize {
    16
}

/// Nested `dflash_config` block in the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashSubConfig {
    /// DFlash 2 serializes the training block width here (top-level
    /// `block_size` is absent in incoai checkpoints). `None` for z-lab
    /// drafters which use the config-root `block_size`.
    #[serde(default)]
    pub block_size: Option<usize>,
    /// Token id used to fill the γ "to-be-predicted" positions during draft
    /// inference. `248070` for Qwen3.6-DFlash.
    pub mask_token_id: u32,
    /// Target-model layer indices to capture intermediate hidden states from.
    /// `[1, 10, 19, 28, 37]` for Qwen3.6-35B-A3B-DFlash. Order matters:
    /// shallow-to-deep concatenation is what `fc` expects.
    pub target_layer_ids: Vec<usize>,
    /// Published SpecForge DSpark checkpoints use `projector_type="dspark"`.
    /// It must agree with the HF `architectures` declaration.
    #[serde(default)]
    pub projector_type: Option<String>,
    #[serde(default)]
    pub enable_confidence_head: Option<bool>,
    #[serde(default)]
    pub confidence_head_with_markov: Option<bool>,
    /// Some experimental EAGLE-3.1 exporters normalize every captured target
    /// layer independently before concatenation and `fc`. Atlas does not yet
    /// load or execute those per-capture norms, so such checkpoints must fail
    /// closed instead of silently serving a different computation graph.
    #[serde(default)]
    pub fc_layernorm: bool,
    /// DFlash 2 two-tap dynamic grouped causal conv (incoai/Qwen3.8-27B-DFlash2).
    /// Number of taps in `GroupedDynamicCausalConv` (2 for the released
    /// checkpoint). When > 1, every drafter layer carries an `attention_conv`
    /// and an `mlp_conv`, each with `base_kernel [2, kernel_size, H]` and
    /// `kernel_projection [2 * kernel_size * groups, H]` where
    /// `groups = H / conv_group_size`.
    #[serde(default)]
    pub conv_kernel_size: usize,
    /// DFlash 2 conv channel group — every `conv_group_size` channels share
    /// one dynamic coefficient (16 for the released checkpoint).
    #[serde(default)]
    pub conv_group_size: usize,
    /// DFlash 2 candidate-selector embedding rank (256 for the released
    /// checkpoint). `candidate_selector.hidden_projection` is
    /// `[rank, H]`; both codebooks are `[vocab, rank]`.
    #[serde(default)]
    pub selector_rank: usize,
    /// DFlash 2 candidate-selector beam width — top-k candidates per
    /// position scored by the codebook walk (16 for the released
    /// checkpoint).
    #[serde(default)]
    pub selector_top_k: usize,
}

/// Raw weight bundle for the DFlash drafter, post-load.
///
/// Verified against `z-lab/Qwen3.6-35B-A3B-DFlash` (commit 42d3b34, May 2026):
/// the checkpoint ships 91 BF16 tensors — `fc.weight`, `hidden_norm.weight`,
/// `norm.weight`, plus 11 weights per drafter layer × 8 layers. **No
/// `embed_tokens` or `lm_head` are in the checkpoint** — the drafter shares
/// the target's embedding and LM head at construction time. This matches the
/// vLLM PR #40898 flow: when those keys are absent, vLLM's `AutoWeightsLoader`
/// adds them to `skip_substrs`, leaving the runtime to slot in the target's
/// pointers.
#[allow(dead_code)]
pub struct DflashWeights {
    pub config: DflashConfig,

    /// `[draft_hidden, len(target_layer_ids) * target_hidden]`.
    /// Qwen3.6-35B-A3B-DFlash: `[2048, 10240]`.
    pub fc: DenseWeight,
    /// Serialized FC geometry retained for fail-closed target pairing.
    pub fc_shape: [usize; 2],
    /// `[draft_hidden]` — RMSNorm applied to the projected target context
    /// before mixing with token embeddings.
    pub hidden_norm: DenseWeight,
    /// `[draft_hidden]` — final RMSNorm before LM head.
    pub norm: DenseWeight,

    pub layers: Vec<DflashLayerWeights>,

    /// Present iff the drafter has a draft-id → target-id mapping (i.e.
    /// `draft_vocab_size != target_vocab_size`). Absent for
    /// Qwen3.6-35B-A3B-DFlash (both vocabs = 248320).
    pub draft_id_to_target_id: Option<Vec<i64>>,

    /// DSpark VanillaMarkov head weights, present iff the checkpoint ships
    /// `markov_head.markov_w1.weight` + `markov_head.markov_w2.weight` AND
    /// `config.markov_rank > 0`. `None` for the vanilla DFlash drafters.
    pub markov: Option<MarkovWeights>,

    /// DSpark learned acceptance-confidence projection. In the only supported
    /// planner mode (`static`) Atlas validates and loads these tensors for
    /// checkpoint integrity, but deliberately does not consume their output:
    /// this matches SGLang PR 34966's static verify-all path. Dynamic modes
    /// fail closed before weight loading until ragged verify and SPS costing
    /// are implemented.
    pub confidence: Option<DsparkConfidenceWeights>,

    /// DFlash 2 candidate selector (3 tensors), present iff the checkpoint
    /// ships `candidate_selector.*` AND `config.dflash_config` declares
    /// `selector_rank > 0`. `None` for plain DFlash / DSpark drafters.
    pub selector: Option<DflashSelectorWeights>,
}

/// DSpark VanillaMarkov head weights (both BF16, loaded verbatim on device).
///
/// `markov_w1` is an `nn.Embedding(V, r)` — the checkpoint stores it as
/// `[V, r]` so a row `markov_w1[prev]` is a contiguous `[r]` slice (a plain
/// embedding gather). `markov_w2` is an `nn.Linear(r, V, bias=False)` — the
/// checkpoint stores its weight as `[V, r]` (out=V, in=r), i.e. the per-token
/// bias is `B(prev) = markov_w2 @ markov_w1[prev]` = `[V, r] @ [r] = [V]`,
/// which is exactly a `dense_gemv(input=w1row[r], weight=w2[V, r], n=V, k=r)`.
#[derive(Debug, Clone, Copy)]
pub struct MarkovWeights {
    /// `nn.Embedding(vocab, rank)` weight — `[vocab, rank]` BF16.
    pub w1: DenseWeight,
    /// `nn.Linear(rank, vocab, bias=False)` weight — `[vocab, rank]` BF16.
    pub w2: DenseWeight,
    /// Low-rank dimension `r` (256 for DSpark-AEON-draft).
    pub rank: usize,
}

/// Learned DSpark confidence affine (`sigmoid([hidden; markov] @ W^T + b)`).
#[derive(Debug, Clone, Copy)]
pub struct DsparkConfidenceWeights {
    /// `nn.Linear(input_dim, 1)` weight — `[1, input_dim]` BF16.
    pub proj: DenseWeight,
    /// Scalar bias — `[1]` BF16.
    pub bias: DenseWeight,
    /// `hidden_size + markov_rank` when `with_markov`, else `hidden_size`.
    pub input_dim: usize,
    pub with_markov: bool,
}

/// DFlash 2 `GroupedDynamicCausalConv` weights (one per sublayer).
///
/// `base_kernel` is `[2, kernel_size, hidden]` (stage 0 = prepare,
/// stage 1 = finish) and `kernel_projection` is
/// `[2 * kernel_size * groups, hidden]` with `groups = hidden / group_size`.
/// Both are BF16 in the released checkpoint.
#[derive(Debug, Clone)]
pub struct DflashConvWeights {
    /// `[2, kernel_size, hidden]` BF16.
    pub base_kernel: DenseWeight,
    /// `[2 * kernel_size * groups, hidden]` BF16.
    pub kernel_projection: DenseWeight,
    /// `kernel_size` (2 for the released checkpoint).
    pub kernel_size: usize,
    /// `hidden / conv_group_size`.
    pub groups: usize,
    /// `conv_group_size` (16 for the released checkpoint).
    pub group_size: usize,
}

/// DFlash 2 candidate-selector weights.
///
/// `hidden_projection` is `[rank, hidden]`; `predecessor_codebook` and
/// `successor_codebook` are both `[vocab, rank]` BF16 embedding tables.
#[derive(Debug, Clone)]
pub struct DflashSelectorWeights {
    /// `[rank, hidden]` BF16.
    pub hidden_projection: DenseWeight,
    /// `[vocab, rank]` BF16.
    pub predecessor_codebook: DenseWeight,
    /// `[vocab, rank]` BF16.
    pub successor_codebook: DenseWeight,
    /// Selector embedding rank (256 for the released checkpoint).
    pub rank: usize,
    /// Selector beam width — top-k candidates per position (16).
    pub top_k: usize,
}

/// Per-drafter-layer raw weights (BF16). Same shape across all 8 layers.
#[allow(dead_code)]
pub struct DflashLayerWeights {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
    /// DFlash 2 two-tap conv wrapping the attention sublayer. `None` for
    /// non-DFlash2 drafters.
    pub attention_conv: Option<DflashConvWeights>,
    /// DFlash 2 two-tap conv wrapping the MLP sublayer. `None` for
    /// non-DFlash2 drafters.
    pub mlp_conv: Option<DflashConvWeights>,
}

/// Probe a [`WeightStore`] for the presence of DFlash drafter weights.
/// Returns true if the store contains the unique `fc.weight` tensor that
/// DFlash drafters ship — a lightweight detection that doesn't load any
/// data. Both bare-key and `model.`-prefixed layouts are accepted; the
/// canonical `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash` checkpoints ship the
/// bare layout (verified against commit 42d3b34, May 2026).
pub fn store_has_dflash_weights(store: &WeightStore) -> bool {
    store.contains("fc.weight") || store.contains("model.fc.weight")
}

/// Probe a [`WeightStore`] for the DSpark VanillaMarkov head tensors. Both
/// bare and `model.`-prefixed layouts are accepted (the DSpark-AEON-draft
/// checkpoint ships bare: `markov_head.markov_w1.weight`).
pub fn store_has_markov_head(store: &WeightStore) -> bool {
    store.contains("markov_head.markov_w1.weight")
        || store.contains("model.markov_head.markov_w1.weight")
}

/// Parse a DFlash drafter's `config.json` into a [`DflashConfig`]. Used by
/// `main.rs` after fetching the drafter's HF metadata to size the runtime
/// `BlockDiffusionDraftHead` (layer count, head_dim, vocab_size, the
/// `target_layer_ids` capture indices).
pub fn parse_dflash_config(json: &str) -> Result<DflashConfig> {
    let config: DflashConfig =
        serde_json::from_str(json).context("Parsing DFlash drafter config.json")?;
    if config
        .dflash_config
        .as_ref()
        .is_some_and(|sub| sub.fc_layernorm)
    {
        anyhow::bail!(
            "DFlash checkpoint requires dflash_config.fc_layernorm=true, but Atlas does not yet \
             load or execute the per-capture pre_fc_norms weights"
        );
    }
    config.validate_supported_semantics()?;
    Ok(config)
}

/// Load DFlash drafter weights from a separate [`WeightStore`] pointing at
/// the drafter checkpoint.
///
/// The drafter ships its weights at the **root** of the safetensors file
/// (no `model.` prefix), in the same naming convention as a vanilla Qwen3
/// transformer minus `embed_tokens` and `lm_head`. Atlas's runtime fills
/// those two from the *target* model's embedding / LM head at construction
/// time — exactly mirroring vLLM's "absent in checkpoint → skip_substrs →
/// share with parent" flow.
///
/// The probed key list (verified against `z-lab/Qwen3.6-35B-A3B-DFlash`):
///
/// ```text
///   fc.weight                                              [H, 5*H_target]
///   hidden_norm.weight                                     [H]
///   norm.weight                                            [H]
///   layers.{0..L-1}.input_layernorm.weight                 [H]
///   layers.{0..L-1}.post_attention_layernorm.weight        [H]
///   layers.{0..L-1}.self_attn.q_proj.weight                [Q*Hd, H]
///   layers.{0..L-1}.self_attn.k_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.v_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.o_proj.weight                [H, Q*Hd]
///   layers.{0..L-1}.self_attn.q_norm.weight                [Hd]
///   layers.{0..L-1}.self_attn.k_norm.weight                [Hd]
///   layers.{0..L-1}.mlp.gate_proj.weight                   [I, H]
///   layers.{0..L-1}.mlp.up_proj.weight                     [I, H]
///   layers.{0..L-1}.mlp.down_proj.weight                   [H, I]
/// ```
///
/// where `H=2048`, `H_target=2048`, `Q=32`, `Kv=4`, `Hd=128`, `I=6144`,
/// `L=8` for Qwen3.6-35B-A3B-DFlash.
///
/// Under TP the drafter is replicated, not sharded — `tp_size>1` produces
/// the same per-rank result as `tp_size=1`. Memory cost: ~948 MB BF16
/// per rank, trivially below the 119 GB GB10 budget.
pub fn load_dflash_weights(
    drafter_store: &WeightStore,
    drafter_config: &DflashConfig,
    _gpu: &dyn GpuBackend,
    _tp_size: usize,
) -> Result<Option<DflashWeights>> {
    let Some(prefix) = validate_dflash_store(drafter_store, drafter_config)? else {
        tracing::debug!("DFlash drafter store has no `fc.weight` — skipping");
        return Ok(None);
    };

    let fc_name = format!("{prefix}fc.weight");
    let fc_tensor = drafter_store.get(&fc_name)?;
    anyhow::ensure!(
        fc_tensor.shape.len() == 2,
        "DFlash drafter fc.weight must be rank 2, got {:?}",
        fc_tensor.shape
    );
    let fc_shape = [fc_tensor.shape[0], fc_tensor.shape[1]];
    let fc = dense(drafter_store, &fc_name).context("DFlash drafter: load fc.weight")?;
    let hidden_norm = dense(drafter_store, &format!("{prefix}hidden_norm.weight"))
        .context("DFlash drafter: load hidden_norm.weight")?;
    let norm = dense(drafter_store, &format!("{prefix}norm.weight"))
        .context("DFlash drafter: load norm.weight")?;

    let layer_count = drafter_config.num_hidden_layers;
    let hidden = drafter_config.hidden_size;
    let dflash2 = drafter_config.is_dflash2();
    let conv_kernel = drafter_config
        .dflash_config
        .as_ref()
        .map(|c| c.conv_kernel_size)
        .unwrap_or(0);
    let conv_group = drafter_config
        .dflash_config
        .as_ref()
        .map(|c| c.conv_group_size)
        .unwrap_or(0);
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");
        // DFlash 2 two-tap conv: `layers.{i}.attention_conv.base_kernel`
        // `[2, kernel, hidden]` + `kernel_projection.weight`
        // `[2*kernel*groups, hidden]`, same for `mlp_conv`.
        let attention_conv = if dflash2 {
            let base_kernel = dense(drafter_store, &format!("{lp}.attention_conv.base_kernel"))
                .context("DFlash2 drafter: load attention_conv.base_kernel")?;
            let kernel_projection = dense(
                drafter_store,
                &format!("{lp}.attention_conv.kernel_projection.weight"),
            )
            .context("DFlash2 drafter: load attention_conv.kernel_projection.weight")?;
            let groups = hidden / conv_group.max(1);
            Some(DflashConvWeights {
                base_kernel,
                kernel_projection,
                kernel_size: conv_kernel.max(1),
                groups,
                group_size: conv_group.max(1),
            })
        } else {
            None
        };
        let mlp_conv = if dflash2 {
            let base_kernel = dense(drafter_store, &format!("{lp}.mlp_conv.base_kernel"))
                .context("DFlash2 drafter: load mlp_conv.base_kernel")?;
            let kernel_projection = dense(
                drafter_store,
                &format!("{lp}.mlp_conv.kernel_projection.weight"),
            )
            .context("DFlash2 drafter: load mlp_conv.kernel_projection.weight")?;
            let groups = hidden / conv_group.max(1);
            Some(DflashConvWeights {
                base_kernel,
                kernel_projection,
                kernel_size: conv_kernel.max(1),
                groups,
                group_size: conv_group.max(1),
            })
        } else {
            None
        };
        let layer = DflashLayerWeights {
            input_layernorm: dense(drafter_store, &format!("{lp}.input_layernorm.weight"))?,
            post_attention_layernorm: dense(
                drafter_store,
                &format!("{lp}.post_attention_layernorm.weight"),
            )?,
            q_proj: dense(drafter_store, &format!("{lp}.self_attn.q_proj.weight"))?,
            k_proj: dense(drafter_store, &format!("{lp}.self_attn.k_proj.weight"))?,
            v_proj: dense(drafter_store, &format!("{lp}.self_attn.v_proj.weight"))?,
            o_proj: dense(drafter_store, &format!("{lp}.self_attn.o_proj.weight"))?,
            q_norm: dense(drafter_store, &format!("{lp}.self_attn.q_norm.weight"))?,
            k_norm: dense(drafter_store, &format!("{lp}.self_attn.k_norm.weight"))?,
            gate_proj: dense(drafter_store, &format!("{lp}.mlp.gate_proj.weight"))?,
            up_proj: dense(drafter_store, &format!("{lp}.mlp.up_proj.weight"))?,
            down_proj: dense(drafter_store, &format!("{lp}.mlp.down_proj.weight"))?,
            attention_conv,
            mlp_conv,
        };
        layers.push(layer);
    }

    // DFlash 2 candidate selector: `candidate_selector.hidden_projection`
    // `[rank, hidden]` + both codebooks `[vocab, rank]`. Loaded only when the
    // config declares the selector contract AND the tensors are present.
    let selector = if dflash2 {
        let rank = drafter_config
            .dflash_config
            .as_ref()
            .map(|c| c.selector_rank)
            .unwrap_or(0);
        let top_k = drafter_config
            .dflash_config
            .as_ref()
            .map(|c| c.selector_top_k)
            .unwrap_or(0);
        let hidden_projection = dense(
            drafter_store,
            &format!("{prefix}candidate_selector.hidden_projection.weight"),
        )
        .context("DFlash2 drafter: load candidate_selector.hidden_projection.weight")?;
        let predecessor_codebook = dense(
            drafter_store,
            &format!("{prefix}candidate_selector.predecessor_codebook"),
        )
        .context("DFlash2 drafter: load candidate_selector.predecessor_codebook")?;
        let successor_codebook = dense(
            drafter_store,
            &format!("{prefix}candidate_selector.successor_codebook"),
        )
        .context("DFlash2 drafter: load candidate_selector.successor_codebook")?;
        Some(DflashSelectorWeights {
            hidden_projection,
            predecessor_codebook,
            successor_codebook,
            rank,
            top_k,
        })
    } else {
        None
    };

    // `d2t` (draft-id → target-id) is absent from Qwen3.6-DFlash because
    // both vocabs are 248320. If a future drafter ships a smaller vocab
    // (vLLM supports this via `draft_vocab_size`), the int64 mapping table
    // would land here. Probing first to keep this loader compatible.
    let draft_id_to_target_id = if drafter_store.contains(&format!("{prefix}d2t"))
        || drafter_store.contains(&format!("{prefix}draft_id_to_target_id"))
    {
        // Mapping is loaded into device memory by upstream paths — for now
        // we just record presence. Phase 2.5 will copy it to a host Vec<i64>
        // when the head needs it for logit remapping.
        tracing::warn!(
            "DFlash drafter has draft-id→target-id mapping; remapping path is not yet wired (Phase 2.5 follow-up)"
        );
        Some(Vec::new())
    } else {
        None
    };

    // DSpark VanillaMarkov head (config-driven, like the rest of the drafter):
    // load W1/W2 iff `config.markov_rank > 0` AND the checkpoint actually ships
    // the tensors. ATLAS_DFLASH_MARKOV=0 disables the head at load time even
    // when present (A/B toggle); default auto-on when the checkpoint has it.
    let markov_disabled = std::env::var("ATLAS_DFLASH_MARKOV").ok().as_deref() == Some("0");
    let markov = if drafter_config.markov_rank > 0
        && store_has_markov_head(drafter_store)
        && !markov_disabled
    {
        let head_type = drafter_config.markov_head_type.to_lowercase();
        if head_type != "vanilla" {
            tracing::warn!(
                "DFlash Markov head type={head_type:?} is not fully supported in Atlas \
                 (only 'vanilla'); the gated variant needs the per-position backbone \
                 hidden which the argmax-only propose path does not surface — falling \
                 back to the VanillaMarkov bias (ignores the gate)."
            );
        }
        let w1 = dense(
            drafter_store,
            &format!("{prefix}markov_head.markov_w1.weight"),
        )
        .context("DFlash Markov head: load markov_w1.weight")?;
        let w2 = dense(
            drafter_store,
            &format!("{prefix}markov_head.markov_w2.weight"),
        )
        .context("DFlash Markov head: load markov_w2.weight")?;
        tracing::info!(
            "DFlash DSpark Markov head loaded: rank={}, type={} (bigram logit bias applied \
             LEFT-TO-RIGHT before per-position argmax; set ATLAS_DFLASH_MARKOV=0 to disable)",
            drafter_config.markov_rank,
            head_type,
        );
        Some(MarkovWeights {
            w1,
            w2,
            rank: drafter_config.markov_rank,
        })
    } else {
        if drafter_config.markov_rank > 0 && !store_has_markov_head(drafter_store) {
            tracing::warn!(
                "DFlash config has markov_rank={} but the checkpoint ships no \
                 `markov_head.markov_w1.weight` — Markov head disabled",
                drafter_config.markov_rank,
            );
        }
        if markov_disabled && store_has_markov_head(drafter_store) {
            tracing::info!("DFlash Markov head present but ATLAS_DFLASH_MARKOV=0 — disabled");
        }
        None
    };

    // In SGLang PR 34966's default `static` mode the learned confidence head
    // is not constructed or evaluated: every proposal is verified. Atlas
    // still validates and loads the public checkpoint's tensors so corrupt or
    // mismatched exports cannot masquerade as supported. Dynamic planners are
    // rejected separately by `validate_verify_mode`.
    let confidence = if let Some(confidence) = drafter_config.confidence_head_config()? {
        let proj = dense(
            drafter_store,
            &format!("{prefix}confidence_head.proj.weight"),
        )
        .context("DSpark confidence head: load proj.weight")?;
        let bias = dense(drafter_store, &format!("{prefix}confidence_head.proj.bias"))
            .context("DSpark confidence head: load proj.bias")?;
        Some(DsparkConfidenceWeights {
            proj,
            bias,
            input_dim: confidence.input_dim,
            with_markov: confidence.with_markov,
        })
    } else {
        None
    };

    tracing::info!(
        "Drafter weights loaded: family={}, {} layers, hidden={}, vocab={}, \
         checkpoint_block_size={}, target_layers={:?}, markov={}, confidence={}, \
         dflash2={} selector={} conv_per_layer={}",
        drafter_config.checkpoint_family()?,
        layers.len(),
        drafter_config.hidden_size,
        drafter_config.vocab_size,
        drafter_config.resolved_block_size(),
        drafter_config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.as_slice())
            .unwrap_or(&[]),
        markov.is_some(),
        confidence.is_some(),
        dflash2,
        selector.is_some(),
        layers.iter().filter(|l| l.attention_conv.is_some()).count(),
    );

    Ok(Some(DflashWeights {
        config: drafter_config.clone(),
        fc,
        fc_shape,
        hidden_norm,
        norm,
        layers,
        draft_id_to_target_id,
        markov,
        confidence,
        selector,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_CONFIG_FIELDS: &str = r#"
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "intermediate_size": 10240,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 248320,
        "block_size": 7,
        "markov_rank": 256,
        "markov_head_type": "vanilla"
    "#;

    /// Smoke-test the DFlash drafter `config.json` parser against the live
    /// `z-lab/Qwen3.6-35B-A3B-DFlash` checkpoint downloaded into the user's
    /// HF cache. Skipped when the cache directory isn't populated — keeps
    /// CI hermetic. Asserts the locked drafter dimensions: 8 layers,
    /// hidden=2048, vocab=248320, B=16/γ=15, mask=248070,
    /// layer_ids=[1,10,19,28,37].
    #[test]
    fn parse_qwen3_6_35b_dflash_config() {
        const SNAP: &str = "/workspace/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719/config.json";
        let json = match std::fs::read_to_string(SNAP) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: drafter snapshot not in cache");
                return;
            }
        };
        let config = parse_dflash_config(&json).expect("parse drafter config");
        assert_eq!(config.num_hidden_layers, 8);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 248320);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.block_size, 16);
        let sub = config.dflash_config.expect("dflash_config present");
        assert_eq!(sub.mask_token_id, 248070);
        assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 28, 37]);
    }

    #[test]
    fn rejects_unimplemented_per_capture_fc_layernorm() {
        let json = r#"{
            "hidden_size": 5120,
            "num_hidden_layers": 6,
            "intermediate_size": 17408,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 248320,
            "dflash_config": {
                "mask_token_id": 248077,
                "target_layer_ids": [1, 10, 18, 27, 35, 44, 52, 61],
                "fc_layernorm": true
            }
        }"#;
        let err = parse_dflash_config(json).expect_err("fc_layernorm must fail closed");
        assert!(err.to_string().contains("fc_layernorm"));
    }

    #[test]
    fn resolves_specforge_dspark_only_from_agreeing_explicit_declarations() {
        let json = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark"
                }}
            }}"#
        );
        let config = parse_dflash_config(&json).expect("supported headless DSpark config");
        assert_eq!(
            config.checkpoint_family().unwrap(),
            DrafterCheckpointFamily::Dspark
        );
        let gamma = config.resolve_draft_count(None).unwrap();
        assert_eq!(gamma, 7);
        assert_eq!(gamma + 1, 8);
        assert_eq!(config.resolve_draft_count(Some(4)).unwrap(), 4);
        assert_eq!(config.resolve_draft_count(Some(7)).unwrap(), 7);
        assert!(config.resolve_draft_count(Some(8)).is_err());
        assert!(config.resolve_draft_count(Some(0)).is_err());
        let mut one_draft = config;
        one_draft.block_size = 1;
        assert_eq!(one_draft.resolve_draft_count(None).unwrap(), 1);
    }

    #[test]
    fn dspark_two_x_width_is_explicit_bounded_and_family_specific() {
        assert!(draft_width_is_supported(
            DrafterCheckpointFamily::Dspark,
            7,
            14,
            true
        ));
        assert!(!draft_width_is_supported(
            DrafterCheckpointFamily::Dspark,
            7,
            14,
            false
        ));
        assert!(!draft_width_is_supported(
            DrafterCheckpointFamily::Dspark,
            7,
            15,
            true
        ));
        assert!(!draft_width_is_supported(
            DrafterCheckpointFamily::Dflash,
            7,
            14,
            true
        ));
        assert!(draft_width_is_supported(
            DrafterCheckpointFamily::Dflash,
            7,
            7,
            false
        ));
    }

    #[test]
    fn preserves_legacy_dflash_family_without_dspark_markers() {
        let json = format!(
            r#"{{
                "architectures": ["DFlashDraftModel"],
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248070,
                    "target_layer_ids": [1, 10, 19, 28, 37]
                }}
            }}"#
        );
        let config = parse_dflash_config(&json).expect("legacy DFlash config");
        assert_eq!(
            config.checkpoint_family().unwrap(),
            DrafterCheckpointFamily::Dflash
        );
        config
            .validate_verify_mode(DsparkVerifyMode::Static)
            .expect("static preserves legacy DFlash");
        let error = config
            .validate_verify_mode(DsparkVerifyMode::Compact)
            .expect_err("DSpark dynamic planner must not be applied to DFlash");
        assert!(error.to_string().contains("DFlash checkpoint"), "{error:#}");
        let mut b16 = config;
        b16.block_size = 16;
        assert_eq!(b16.resolve_draft_count(None).unwrap(), 15);
        assert_eq!(b16.resolve_draft_count(Some(4)).unwrap(), 4);
        assert!(b16.resolve_draft_count(Some(16)).is_err());
        assert!(b16.resolve_draft_count(Some(0)).is_err());
        b16.block_size = 1;
        assert!(b16.resolve_draft_count(None).is_err());
    }

    #[test]
    fn rejects_ambiguous_dspark_family_markers() {
        let architecture_only = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52]
                }}
            }}"#
        );
        let projector_only = format!(
            r#"{{
                "architectures": ["DFlashDraftModel"],
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark"
                }}
            }}"#
        );
        for json in [architecture_only, projector_only] {
            let error = parse_dflash_config(&json).expect_err("ambiguous family must fail closed");
            assert!(
                error.to_string().contains("DSparkDraftModel")
                    || error.to_string().contains("projector_type"),
                "{error:#}"
            );
        }
    }

    #[test]
    fn resolves_public_dspark_confidence_contract_without_silent_fallback() {
        let json = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                "enable_confidence_head": true,
                "confidence_head_with_markov": true,
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark",
                    "enable_confidence_head": true,
                    "confidence_head_with_markov": true
                }}
            }}"#
        );
        let config = parse_dflash_config(&json).expect("public DSpark confidence config");
        assert_eq!(
            config.confidence_head_config().unwrap(),
            Some(DsparkConfidenceConfig {
                with_markov: true,
                input_dim: 5376,
            })
        );
        config
            .validate_verify_mode(DsparkVerifyMode::Static)
            .expect("static verify-all is supported");
        for mode in [DsparkVerifyMode::CapAccept, DsparkVerifyMode::Compact] {
            let error = config
                .validate_verify_mode(mode)
                .expect_err("dynamic mode must fail closed");
            assert!(error.to_string().contains("ragged verify"), "{error:#}");
            assert!(error.to_string().contains("SPS"), "{error:#}");
        }
    }

    #[test]
    fn rejects_confidence_config_disagreement_and_invalid_markov_dependency() {
        let disagreement = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                "enable_confidence_head": true,
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark",
                    "enable_confidence_head": false
                }}
            }}"#
        );
        let error = parse_dflash_config(&disagreement).expect_err("must not select one value");
        assert!(error.to_string().contains("disagrees"), "{error:#}");

        let no_head = format!(
            r#"{{
                "architectures": ["DSparkDraftModel"],
                "enable_confidence_head": false,
                "confidence_head_with_markov": true,
                {BASE_CONFIG_FIELDS},
                "dflash_config": {{
                    "mask_token_id": 248077,
                    "target_layer_ids": [4, 16, 28, 40, 52],
                    "projector_type": "dspark"
                }}
            }}"#
        );
        let error = parse_dflash_config(&no_head).expect_err("Markov feature needs head");
        assert!(
            error
                .to_string()
                .contains("requires enable_confidence_head=true"),
            "{error:#}"
        );
    }

    #[test]
    fn dspark_verify_mode_parser_never_falls_back() {
        assert_eq!(
            "static".parse::<DsparkVerifyMode>().unwrap(),
            DsparkVerifyMode::Static
        );
        assert_eq!(
            "cap-accept".parse::<DsparkVerifyMode>().unwrap(),
            DsparkVerifyMode::CapAccept
        );
        assert_eq!(
            "compact".parse::<DsparkVerifyMode>().unwrap(),
            DsparkVerifyMode::Compact
        );
        assert!("dynamic".parse::<DsparkVerifyMode>().is_err());
        assert!("STATIC".parse::<DsparkVerifyMode>().is_err());
    }
}
