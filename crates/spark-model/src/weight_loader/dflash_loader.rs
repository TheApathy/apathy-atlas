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

use crate::weight_map::{DenseWeight, dense};

/// Drafter HF `config.json` (subset Atlas consumes). Mirrors
/// `z-lab/Qwen3.6-35B-A3B-DFlash/config.json` field names verbatim so
/// `serde_json::from_str` works directly on the raw file.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashConfig {
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
    /// HF `architectures` list. `["DFlashLagunaForCausalLM"]` for the poolside
    /// Laguna drafter (fused `qkv_proj`, per-head gating, sliding window);
    /// `["DFlashSpeculator"]`/absent for the Qwen3.6-DFlash drafters. Used to
    /// pick the fused-qkv Laguna load path in [`load_dflash_weights`].
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Sliding-window attention span. The Laguna drafter runs all layers as
    /// `sliding_attention` with window 512. `None`/0 ⇒ full attention (the
    /// Qwen3.6-DFlash drafters). Carried onto the head so the drafter
    /// attention can honour the window.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Block size γ. Qwen3.6-DFlash ships `block_size: 16`.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// DFlash-specific nested config object.
    #[serde(default)]
    pub dflash_config: Option<DflashSubConfig>,
    /// Drafter base RoPE θ. Defaults to 10M (matches Qwen3.6-DFlash).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// HF-style `rope_scaling` block. `None` ⇒ plain RoPE (the v2 2026-04-27
    /// Qwen3.6-DFlash drafter ships `rope_scaling: null`). When present and
    /// `rope_type == "yarn"`, the drafter's YaRN parameters are used to
    /// build the inv_freq table at construction time.
    #[serde(default)]
    pub rope_scaling: Option<DflashRopeScaling>,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

/// Subset of HF `rope_scaling` block consumed by Atlas. Mirrors the field
/// names in `transformers`' Qwen3 config so `serde_json::from_str` works
/// directly on the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeScaling {
    /// Currently only `"yarn"` is recognised; anything else falls back to
    /// plain RoPE with a warning logged at construction time.
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<f32>,
}

fn default_block_size() -> usize {
    16
}

/// Nested `dflash_config` block in the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashSubConfig {
    /// Token id used to fill the γ "to-be-predicted" positions during draft
    /// inference. `248070` for Qwen3.6-DFlash.
    pub mask_token_id: u32,
    /// Target-model layer indices to capture intermediate hidden states from.
    /// `[1, 10, 19, 28, 37]` for Qwen3.6-35B-A3B-DFlash, `[1,10,19,29,38,47]`
    /// for the Laguna drafter. Order matters: shallow-to-deep concatenation is
    /// what `fc` expects.
    pub target_layer_ids: Vec<usize>,
    /// `true` for the poolside Laguna drafter (autoregressive/causal drafter
    /// attention). Absent (⇒ `false`) for the Qwen3.6-DFlash drafters, which
    /// use bidirectional block-diffusion γ-block attention (`is_causal=false`
    /// throughout `forward_block_layer*`). Captured here so the head knows the
    /// drafter's intended mask; the forward-path causal switch is a follow-up
    /// (see [`load_dflash_weights`] Laguna branch notes).
    #[serde(default)]
    pub causal: bool,
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

    /// Per-captured-target-hidden RMSNorms (`aux_hidden_norms.{0..L_t-1}.weight`,
    /// each `[target_hidden]`). Present ONLY on the poolside **Laguna** drafter,
    /// which conditions `fc` by first RMS-normalising each captured target
    /// hidden with its own norm (the Laguna spelling of DFlash's per-capture
    /// conditioning). Empty `Vec` for the Qwen3.6-DFlash drafters (which apply
    /// only the single post-`fc` `hidden_norm`). Ordered shallow→deep to match
    /// the `dflash_config.target_layer_ids` capture order and the `fc` input
    /// concatenation layout.
    pub aux_hidden_norms: Vec<DenseWeight>,
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
    /// Per-head attention output gate (`self_attn.g_proj.weight`, `[Q, H]`).
    /// Present ONLY on the Laguna drafter (`gating: per-head`). `None` for the
    /// Qwen3.6-DFlash drafters. Loaded so the checkpoint tensor is not silently
    /// dropped; applying the gate in the forward path is a follow-up (the
    /// current `forward_block_layer*` does not gate attention output).
    pub g_proj: Option<DenseWeight>,
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

/// True when the drafter checkpoint uses the poolside **Laguna** DFlash
/// layout: a FUSED `self_attn.qkv_proj.weight` per layer instead of the
/// separate `q_proj`/`k_proj`/`v_proj` the Qwen3.6-DFlash drafters ship.
///
/// Detection is layout-based (presence of `qkv_proj`) rather than trusting
/// `config.architectures`, so a re-quantized / renamed re-upload still routes
/// correctly. Both bare and `model.`-prefixed layouts are probed.
pub fn store_has_laguna_dflash_weights(store: &WeightStore) -> bool {
    store.contains("layers.0.self_attn.qkv_proj.weight")
        || store.contains("model.layers.0.self_attn.qkv_proj.weight")
}

/// Parse a DFlash drafter's `config.json` into a [`DflashConfig`]. Used by
/// `main.rs` after fetching the drafter's HF metadata to size the runtime
/// `BlockDiffusionDraftHead` (layer count, head_dim, vocab_size, the
/// `target_layer_ids` capture indices).
pub fn parse_dflash_config(json: &str) -> Result<DflashConfig> {
    serde_json::from_str(json).context("Parsing DFlash drafter config.json")
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
    if !store_has_dflash_weights(drafter_store) {
        tracing::debug!("DFlash drafter store has no `fc.weight` — skipping");
        return Ok(None);
    }

    // Route the poolside Laguna drafter (fused qkv_proj + per-head gate +
    // sliding window) to its own loader. Layout-based detection so a renamed
    // re-upload still routes correctly.
    if store_has_laguna_dflash_weights(drafter_store) {
        return load_laguna_dflash_weights(drafter_store, drafter_config);
    }

    // Detect bare vs. `model.`-prefixed layout. `z-lab` checkpoints use
    // bare; we accept either to be robust against a hypothetical re-upload
    // that uses the prefixed layout.
    let prefix = if drafter_store.contains("model.fc.weight") {
        "model."
    } else {
        ""
    };

    let fc = dense(drafter_store, &format!("{prefix}fc.weight"))
        .context("DFlash drafter: load fc.weight")?;
    let hidden_norm = dense(drafter_store, &format!("{prefix}hidden_norm.weight"))
        .context("DFlash drafter: load hidden_norm.weight")?;
    let norm = dense(drafter_store, &format!("{prefix}norm.weight"))
        .context("DFlash drafter: load norm.weight")?;

    let layer_count = drafter_config.num_hidden_layers;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");
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
            // Qwen3.6-DFlash has no per-head attention gate.
            g_proj: None,
        };
        layers.push(layer);
    }

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

    tracing::info!(
        "DFlash drafter loaded: {} layers, hidden={}, vocab={}, γ={}, target_layers={:?}",
        layers.len(),
        drafter_config.hidden_size,
        drafter_config.vocab_size,
        drafter_config.block_size,
        drafter_config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.as_slice())
            .unwrap_or(&[]),
    );

    Ok(Some(DflashWeights {
        config: drafter_config.clone(),
        fc,
        hidden_norm,
        norm,
        layers,
        draft_id_to_target_id,
        // Qwen3.6-DFlash drafters have no per-capture aux norms — they apply
        // only the single post-`fc` `hidden_norm`.
        aux_hidden_norms: Vec::new(),
    }))
}

/// Load the poolside **Laguna** DFlash drafter (`DFlashLagunaForCausalLM`).
///
/// Differs from [`load_dflash_weights`]' Qwen3.6-DFlash path in five ways
/// (see `Laguna/notes/03-dflash-drafter-wiring.md`), all handled here at load
/// time so the runtime [`crate::layers::BlockDiffusionDraftHead`] consumes the
/// **same** [`DflashWeights`]/[`DflashLayerWeights`] shape as the Qwen3 path:
///
///  1. **Fused `qkv_proj`** — one `[Q·Hd + 2·Kv·Hd, H]` tensor per layer
///     (`[11264, 3072]` for the S-2.1 drafter). We split it into three
///     `DenseWeight`s by row-major byte offset into the *same* device
///     allocation: `q = rows[0, Q·Hd)`, `k = rows[Q·Hd, Q·Hd+Kv·Hd)`,
///     `v = rows[Q·Hd+Kv·Hd, ..)`. No copy — the sub-tensors are read-only
///     GEMM operands and each row (`H` contiguous BF16 elems) stays contiguous.
///  2. **q_norm/k_norm** — same key names as Qwen3; loaded identically.
///  3. **`causal: true`** — captured in `dflash_config.causal` (the forward
///     causal switch is a follow-up; today's `forward_block_layer*` runs the
///     drafter bidirectionally).
///  4. **sliding_window 512** — captured in `config.sliding_window` and carried
///     onto the head (forward honouring is a follow-up).
///  5. **per-head `g_proj`** — loaded into `DflashLayerWeights::g_proj`
///     (gate application in the forward is a follow-up).
///
/// The `fc`, `hidden_norm`, `norm`, and 6 `aux_hidden_norms.*` tensors map
/// straight across (the `aux_hidden_norms` feed the `fc` capture stack; they
/// are the Laguna spelling of the per-capture-point norms).
fn load_laguna_dflash_weights(
    store: &WeightStore,
    config: &DflashConfig,
) -> Result<Option<DflashWeights>> {
    let prefix = if store.contains("model.fc.weight") {
        "model."
    } else {
        ""
    };

    let fc = dense(store, &format!("{prefix}fc.weight")).context("Laguna DFlash: load fc.weight")?;
    let hidden_norm = dense(store, &format!("{prefix}hidden_norm.weight"))
        .context("Laguna DFlash: load hidden_norm.weight")?;
    let norm =
        dense(store, &format!("{prefix}norm.weight")).context("Laguna DFlash: load norm.weight")?;

    let head_dim = config.head_dim;
    let q_dim = config.num_attention_heads * head_dim; // 72*128 = 9216
    let kv_dim = config.num_key_value_heads * head_dim; // 8*128  = 1024
    let hidden = config.hidden_size; // 3072

    let layer_count = config.num_hidden_layers;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");

        // ── Split fused qkv_proj `[q_dim + 2*kv_dim, hidden]` ────────────
        let qkv_key = format!("{lp}.self_attn.qkv_proj.weight");
        let qkv = store
            .get(&qkv_key)
            .with_context(|| format!("Laguna DFlash: load {qkv_key}"))?;
        let expected_rows = q_dim + 2 * kv_dim;
        // Shape is `[rows, hidden]`; validate before slicing so a config/
        // checkpoint mismatch fails loudly instead of producing garbage.
        anyhow::ensure!(
            qkv.shape == vec![expected_rows, hidden],
            "Laguna DFlash: {qkv_key} shape {:?} != expected [{expected_rows}, {hidden}] \
             (q_dim={q_dim}, kv_dim={kv_dim} from config heads {}/{} × head_dim {head_dim})",
            qkv.shape,
            config.num_attention_heads,
            config.num_key_value_heads,
        );
        let elem = qkv.dtype.byte_size(); // BF16 ⇒ 2
        let row_bytes = hidden * elem;
        let base = qkv.ptr;
        // Row-major: Q occupies the first `q_dim` rows, then K, then V.
        let q_proj = DenseWeight { weight: base };
        let k_proj = DenseWeight {
            weight: base.offset(q_dim * row_bytes),
        };
        let v_proj = DenseWeight {
            weight: base.offset((q_dim + kv_dim) * row_bytes),
        };

        // Optional per-head attention gate (Laguna `gating: per-head`).
        let g_key = format!("{lp}.self_attn.g_proj.weight");
        let g_proj = if store.contains(&g_key) {
            Some(dense(store, &g_key).with_context(|| format!("Laguna DFlash: load {g_key}"))?)
        } else {
            None
        };

        let layer = DflashLayerWeights {
            input_layernorm: dense(store, &format!("{lp}.input_layernorm.weight"))?,
            post_attention_layernorm: dense(
                store,
                &format!("{lp}.post_attention_layernorm.weight"),
            )?,
            q_proj,
            k_proj,
            v_proj,
            o_proj: dense(store, &format!("{lp}.self_attn.o_proj.weight"))?,
            q_norm: dense(store, &format!("{lp}.self_attn.q_norm.weight"))?,
            k_norm: dense(store, &format!("{lp}.self_attn.k_norm.weight"))?,
            gate_proj: dense(store, &format!("{lp}.mlp.gate_proj.weight"))?,
            up_proj: dense(store, &format!("{lp}.mlp.up_proj.weight"))?,
            down_proj: dense(store, &format!("{lp}.mlp.down_proj.weight"))?,
            g_proj,
        };
        layers.push(layer);
    }

    // Laguna drafter shares vocab with target (draft_vocab_size == vocab_size),
    // so no d2t remap table ships.
    let draft_id_to_target_id = None;

    // ── Per-capture aux_hidden_norms (`aux_hidden_norms.{0..L_t-1}.weight`) ──
    // One RMSNorm per captured target hidden. The Laguna drafter RMS-normalises
    // each captured target hidden with its own norm BEFORE the `fc` projection
    // (per-capture conditioning). Ordered shallow→deep to match the
    // `target_layer_ids` capture order and the `fc` input concatenation.
    // Count is derived from the checkpoint (probe until absent) so a drafter
    // with a different capture depth still loads; we cross-check against
    // `target_layer_ids` below.
    let mut aux_hidden_norms: Vec<DenseWeight> = Vec::new();
    loop {
        let k = aux_hidden_norms.len();
        let key = format!("{prefix}aux_hidden_norms.{k}.weight");
        if !store.contains(&key) {
            break;
        }
        aux_hidden_norms
            .push(dense(store, &key).with_context(|| format!("Laguna DFlash: load {key}"))?);
    }
    if let Some(sub) = config.dflash_config.as_ref() {
        let n_targets = sub.target_layer_ids.len();
        if !aux_hidden_norms.is_empty() && aux_hidden_norms.len() != n_targets {
            tracing::warn!(
                "Laguna DFlash: found {} aux_hidden_norms but {} target_layer_ids — \
                 per-capture conditioning applies to the first {} captures only",
                aux_hidden_norms.len(),
                n_targets,
                aux_hidden_norms.len().min(n_targets),
            );
        }
    }

    let (causal, target_layers) = config
        .dflash_config
        .as_ref()
        .map(|c| (c.causal, c.target_layer_ids.as_slice()))
        .unwrap_or((false, &[]));
    tracing::info!(
        "Laguna DFlash drafter loaded: {} layers, hidden={}, GQA {}/{}, head_dim={}, \
         vocab={}, γ={}, sliding_window={:?}, causal={}, target_layers={:?}, gated_attn={}, \
         aux_hidden_norms={}",
        layers.len(),
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        head_dim,
        config.vocab_size,
        config.block_size,
        config.sliding_window,
        causal,
        target_layers,
        layers.first().map(|l| l.g_proj.is_some()).unwrap_or(false),
        aux_hidden_norms.len(),
    );
    if causal {
        tracing::info!(
            "Laguna DFlash drafter is causal=true — the γ-block attention runs with the \
             causal mask wired in `forward_block_layer*` (ctx prefix fully visible, γ block \
             causal within itself). Use the CONTIG attention path (ATLAS_DFLASH_CONTIG_ATTN=1) \
             for the faithful causal γ-block; the paged-indirect path stays bidirectional."
        );
    }
    if config.sliding_window.is_some() {
        tracing::warn!(
            "Laguna DFlash drafter uses sliding_window={:?}; the drafter attention does not yet \
             honour the window (loads full-attention). Follow-up.",
            config.sliding_window
        );
    }
    if layers.first().map(|l| l.g_proj.is_some()).unwrap_or(false) {
        tracing::warn!(
            "Laguna DFlash drafter has per-head g_proj gates; loaded but NOT applied in the \
             current forward path. Follow-up."
        );
    }

    Ok(Some(DflashWeights {
        config: config.clone(),
        fc,
        hidden_norm,
        norm,
        layers,
        draft_id_to_target_id,
        aux_hidden_norms,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke-test the DFlash drafter `config.json` parser against the live
    /// `z-lab/Qwen3.6-35B-A3B-DFlash` checkpoint downloaded into the user's
    /// HF cache. Skipped when the cache directory isn't populated — keeps
    /// CI hermetic. Asserts the locked drafter dimensions: 8 layers,
    /// hidden=2048, vocab=248320, γ=16, mask=248070, layer_ids=[1,10,19,28,37].
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

    /// Smoke-test the parser against the live `poolside/Laguna-S-2.1-DFlash`
    /// checkpoint in the user's HF cache. Locks the Laguna drafter dimensions
    /// and the new config fields the Laguna load path relies on
    /// (`architectures`, `sliding_window`, `dflash_config.causal`). Cache-gated
    /// like the Qwen3 test above.
    #[test]
    fn parse_laguna_s_2_1_dflash_config() {
        let snap_path = std::env::var("ATLAS_TEST_DFLASH_CONFIG").unwrap_or_default();
        let json = match std::fs::read_to_string(&snap_path) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: Laguna drafter snapshot not in cache");
                return;
            }
        };
        let config = parse_dflash_config(&json).expect("parse Laguna drafter config");
        assert_eq!(config.num_hidden_layers, 6);
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_attention_heads, 72);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 100352);
        assert_eq!(config.block_size, 16);
        assert_eq!(config.architectures, vec!["DFlashLagunaForCausalLM"]);
        assert_eq!(config.sliding_window, Some(512));
        assert!((config.rope_theta - 10_000.0).abs() < 1.0);
        let sub = config.dflash_config.expect("dflash_config present");
        assert_eq!(sub.mask_token_id, 12);
        assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 29, 38, 47]);
        assert!(sub.causal);
        // Verify the fused-qkv row split the loader will apply.
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        assert_eq!(q_dim + 2 * kv_dim, 11264);
    }
}
