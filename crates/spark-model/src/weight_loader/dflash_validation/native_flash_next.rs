// SPDX-License-Identifier: AGPL-3.0-only

//! Exact structural identity for the native Qwen3.8-Flash-Next V3 drafter.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, ensure};

use super::{TensorMetadataSource, require_tensor};
use crate::weight_loader::dflash_loader::{DflashConfig, DrafterCheckpointFamily};

const HIDDEN: usize = 2_560;
const INTERMEDIATE: usize = 8_704;
const LAYERS: usize = 6;
const TARGET_LAYERS: usize = 48;
const QUERY_HEADS: usize = 20;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const VOCAB: usize = 248_320;
const MASK_TOKEN: u32 = 248_077;
const BLOCK: usize = 16;
const DRAFTS: usize = 15;
const V3_TENSOR_COUNT: usize = 69;
const V3_PARAMETER_COUNT: usize = 547_918_336;
const DFLASH2_TENSOR_COUNT: usize = 96;
const DFLASH2_PARAMETER_COUNT: usize = 695_497_216;
const CONV_KERNEL: usize = 2;
const CONV_GROUP: usize = 16;
const SELECTOR_RANK: usize = 256;
const SELECTOR_TOP_K: usize = 16;
const SELECTOR_VOCAB: usize = 248_077;
const SLIDING_WINDOW: usize = 4_096;
const TAPS: [usize; 8] = [1, 7, 13, 20, 26, 33, 39, 46];
const DFLASH2_LAYER_TYPES: [&str; LAYERS] = [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeKind {
    V3,
    Dflash2,
}

impl NativeKind {
    fn label(self) -> &'static str {
        match self {
            Self::V3 => "V3",
            Self::Dflash2 => "DFlash2",
        }
    }

    fn tensor_count(self) -> usize {
        match self {
            Self::V3 => V3_TENSOR_COUNT,
            Self::Dflash2 => DFLASH2_TENSOR_COUNT,
        }
    }

    fn parameter_count(self) -> usize {
        match self {
            Self::V3 => V3_PARAMETER_COUNT,
            Self::Dflash2 => DFLASH2_PARAMETER_COUNT,
        }
    }
}

fn looks_like_native_config(config: &DflashConfig) -> bool {
    // A two-of-three quorum catches any one-field mutation of the native
    // identity without claiming every future 48-layer or H=2560 drafter.
    let target_depth = config.num_target_layers == TARGET_LAYERS;
    let draft_geometry = config.hidden_size == HIDDEN
        && config.intermediate_size == INTERMEDIATE
        && config.num_hidden_layers == LAYERS;
    let capture_taps = config
        .dflash_config
        .as_ref()
        .is_some_and(|sub| sub.target_layer_ids == TAPS);
    let dflash2_selector_marker = matches!(
        config.architectures.as_slice(),
        [architecture] if architecture == "DFlash2DraftModel"
    ) && config
        .dflash_config
        .as_ref()
        .is_some_and(|sub| sub.selector_vocab_size.is_some());
    usize::from(target_depth) + usize::from(draft_geometry) + usize::from(capture_taps) >= 2
        || dflash2_selector_marker
}

fn validate_native_config(config: &DflashConfig) -> Result<NativeKind> {
    let kind = match config.architectures.as_slice() {
        [architecture] if architecture == "DFlashDraftModel" => NativeKind::V3,
        [architecture] if architecture == "DFlash2DraftModel" => NativeKind::Dflash2,
        _ => anyhow::bail!(
            "native Flash-Next architectures must be exactly [DFlashDraftModel] or [DFlash2DraftModel]"
        ),
    };
    ensure!(
        config.model_type.as_deref() == Some("qwen3"),
        "native Flash-Next {} model_type must be explicitly `qwen3`",
        kind.label()
    );
    ensure!(
        config.checkpoint_family()? == DrafterCheckpointFamily::Dflash,
        "native Flash-Next {} must use the DFlash checkpoint family",
        kind.label()
    );
    for (name, actual, expected) in [
        ("hidden_size", config.hidden_size, HIDDEN),
        ("intermediate_size", config.intermediate_size, INTERMEDIATE),
        ("num_hidden_layers", config.num_hidden_layers, LAYERS),
        ("num_target_layers", config.num_target_layers, TARGET_LAYERS),
        (
            "num_attention_heads",
            config.num_attention_heads,
            QUERY_HEADS,
        ),
        ("num_key_value_heads", config.num_key_value_heads, KV_HEADS),
        ("head_dim", config.head_dim, HEAD_DIM),
        ("vocab_size", config.vocab_size, VOCAB),
    ] {
        ensure!(
            actual == expected,
            "native Flash-Next {} {name}={actual}; expected {expected}",
            kind.label()
        );
    }
    ensure!(
        config.root_block_size_explicit && config.block_size == BLOCK,
        "native Flash-Next {} root block_size must be explicitly {BLOCK}",
        kind.label()
    );
    let sub = config.dflash_config.as_ref().ok_or_else(|| {
        anyhow::anyhow!("native Flash-Next {} requires dflash_config", kind.label())
    })?;
    ensure!(
        sub.block_size == Some(BLOCK),
        "native Flash-Next {} nested block_size must be explicitly {BLOCK}",
        kind.label()
    );
    ensure!(
        config.resolved_block_size() == BLOCK && config.resolve_draft_count(None)? == DRAFTS,
        "native Flash-Next {} requires physical K{BLOCK}/gamma{DRAFTS}",
        kind.label()
    );
    ensure!(
        config.is_causal == Some(false),
        "native Flash-Next {} is_causal must be explicitly false",
        kind.label()
    );
    ensure!(
        !sub.fc_layernorm,
        "native Flash-Next {} fc_layernorm must be false",
        kind.label()
    );
    ensure!(
        sub.mask_token_id == MASK_TOKEN,
        "native Flash-Next {} mask_token_id must be {MASK_TOKEN}",
        kind.label()
    );
    ensure!(
        sub.target_layer_ids == TAPS,
        "native Flash-Next {} target_layer_ids must be {TAPS:?}",
        kind.label()
    );
    ensure!(
        sub.projector_type.is_none(),
        "native Flash-Next {} projector_type must be absent",
        kind.label()
    );
    ensure!(
        config.markov_rank == 0 && config.confidence_head_config()?.is_none(),
        "native Flash-Next {} forbids DSpark Markov and confidence semantics",
        kind.label()
    );
    ensure!(
        config.draft_vocab_size.is_none(),
        "native Flash-Next {} forbids a draft-to-target vocabulary remap",
        kind.label()
    );
    ensure!(
        !config.tie_word_embeddings,
        "native Flash-Next {} tie_word_embeddings must be false",
        kind.label()
    );
    match kind {
        NativeKind::V3 => ensure!(
            sub.conv_kernel_size == 0
                && sub.conv_group_size == 0
                && sub.selector_rank == 0
                && sub.selector_top_k == 0
                && sub.selector_vocab_size.is_none(),
            "native Flash-Next V3 forbids DFlash2 convolution and selector semantics"
        ),
        NativeKind::Dflash2 => {
            ensure!(
                sub.conv_kernel_size == CONV_KERNEL,
                "native Flash-Next DFlash2 conv_kernel_size must be {CONV_KERNEL}"
            );
            ensure!(
                sub.conv_group_size == CONV_GROUP,
                "native Flash-Next DFlash2 conv_group_size must be {CONV_GROUP}"
            );
            ensure!(
                sub.selector_rank == SELECTOR_RANK,
                "native Flash-Next DFlash2 selector_rank must be {SELECTOR_RANK}"
            );
            ensure!(
                sub.selector_top_k == SELECTOR_TOP_K,
                "native Flash-Next DFlash2 selector_top_k must be {SELECTOR_TOP_K}"
            );
            ensure!(
                sub.selector_vocab_size == Some(SELECTOR_VOCAB),
                "native Flash-Next DFlash2 selector_vocab_size must be explicitly {SELECTOR_VOCAB}"
            );
            ensure!(
                sub.unknown_fields.is_empty(),
                "native Flash-Next DFlash2 has unknown dflash_config fields: {:?}",
                sub.unknown_fields.keys().collect::<Vec<_>>()
            );
            ensure!(
                config.sliding_window == Some(SLIDING_WINDOW),
                "native Flash-Next DFlash2 sliding_window must be explicitly {SLIDING_WINDOW}"
            );
            let layer_types_match = config.layer_types.as_ref().is_some_and(|layer_types| {
                layer_types
                    .iter()
                    .map(String::as_str)
                    .eq(DFLASH2_LAYER_TYPES)
            });
            ensure!(
                layer_types_match,
                "native Flash-Next DFlash2 layer_types must be exactly {DFLASH2_LAYER_TYPES:?}"
            );
        }
    }
    Ok(kind)
}

pub(super) fn validate_config_if_candidate(config: &DflashConfig) -> Result<()> {
    if looks_like_native_config(config) {
        validate_native_config(config)?;
    }
    Ok(())
}

fn expected_manifest(kind: NativeKind) -> BTreeMap<String, Vec<usize>> {
    let mut expected = BTreeMap::from([
        ("fc.weight".into(), vec![HIDDEN, HIDDEN * TAPS.len()]),
        ("hidden_norm.weight".into(), vec![HIDDEN]),
        ("norm.weight".into(), vec![HIDDEN]),
    ]);
    let per_layer = [
        ("input_layernorm.weight", vec![HIDDEN]),
        ("post_attention_layernorm.weight", vec![HIDDEN]),
        (
            "self_attn.q_proj.weight",
            vec![QUERY_HEADS * HEAD_DIM, HIDDEN],
        ),
        ("self_attn.k_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
        ("self_attn.v_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
        (
            "self_attn.o_proj.weight",
            vec![HIDDEN, QUERY_HEADS * HEAD_DIM],
        ),
        ("self_attn.q_norm.weight", vec![HEAD_DIM]),
        ("self_attn.k_norm.weight", vec![HEAD_DIM]),
        ("mlp.gate_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("mlp.up_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("mlp.down_proj.weight", vec![HIDDEN, INTERMEDIATE]),
    ];
    for layer in 0..LAYERS {
        for (suffix, shape) in &per_layer {
            expected.insert(format!("layers.{layer}.{suffix}"), shape.clone());
        }
        if kind == NativeKind::Dflash2 {
            let groups = HIDDEN / CONV_GROUP;
            for sublayer in ["attention_conv", "mlp_conv"] {
                expected.insert(
                    format!("layers.{layer}.{sublayer}.base_kernel"),
                    vec![2, CONV_KERNEL, HIDDEN],
                );
                expected.insert(
                    format!("layers.{layer}.{sublayer}.kernel_projection.weight"),
                    vec![2 * CONV_KERNEL * groups, HIDDEN],
                );
            }
        }
    }
    if kind == NativeKind::Dflash2 {
        expected.insert(
            "candidate_selector.hidden_projection.weight".into(),
            vec![SELECTOR_RANK, HIDDEN],
        );
        expected.insert(
            "candidate_selector.predecessor_codebook".into(),
            vec![VOCAB, SELECTOR_RANK],
        );
        expected.insert(
            "candidate_selector.successor_codebook".into(),
            vec![VOCAB, SELECTOR_RANK],
        );
    }
    expected
}

pub(super) fn finiteness_manifest_if_candidate(
    config: &DflashConfig,
) -> Result<Option<BTreeMap<String, Vec<usize>>>> {
    if !looks_like_native_config(config) {
        return Ok(None);
    }
    let kind = validate_native_config(config)?;
    Ok(Some(expected_manifest(kind)))
}

pub(super) fn validate_store_if_candidate(
    source: &dyn TensorMetadataSource,
    prefix: &str,
    config: &DflashConfig,
) -> Result<()> {
    let native_fc = source
        .metadata(&format!("{prefix}fc.weight"))
        .is_some_and(|tensor| tensor.shape == [HIDDEN, HIDDEN * TAPS.len()]);
    if !native_fc && !looks_like_native_config(config) {
        return Ok(());
    }
    let kind = validate_native_config(config)?;
    ensure!(
        prefix.is_empty(),
        "native Flash-Next {} tensors must use the canonical bare names",
        kind.label()
    );
    let expected = expected_manifest(kind);
    let observed: BTreeSet<_> = source.names().map(str::to_owned).collect();
    let expected_names: BTreeSet<_> = expected.keys().cloned().collect();
    let missing: Vec<_> = expected_names.difference(&observed).cloned().collect();
    let unexpected: Vec<_> = observed.difference(&expected_names).cloned().collect();
    ensure!(
        missing.is_empty() && unexpected.is_empty(),
        "native Flash-Next {} tensor manifest mismatch: missing={missing:?}, unexpected={unexpected:?}",
        kind.label()
    );
    ensure!(
        expected.len() == kind.tensor_count(),
        "internal native tensor-count drift"
    );
    let mut parameters = 0usize;
    for (name, shape) in &expected {
        require_tensor(source, name, shape)?;
        let elements = shape.iter().try_fold(1usize, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or_else(|| anyhow::anyhow!("native tensor `{name}` size overflow"))
        })?;
        parameters = parameters
            .checked_add(elements)
            .ok_or_else(|| anyhow::anyhow!("native parameter-count overflow"))?;
    }
    ensure!(
        parameters == kind.parameter_count(),
        "native Flash-Next {} has {parameters} structural parameters; expected {}",
        kind.label(),
        kind.parameter_count()
    );
    Ok(())
}
