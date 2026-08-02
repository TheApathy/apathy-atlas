// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash **DSpark** block-drafter loader.
//!
//! The official `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint ships a
//! trained speculative block drafter in shards 46–48 (~10.9 GB, pure `mtp.*`):
//! three full V4 layers (`mtp.{0,1,2}` — MLA attention, mHC, 256-expert
//! native-MXFP4 MoE) plus the DSpark-specific heads. It is NOT the legacy
//! single-module MTP head (`num_nextn_predict_layers = 1` checkpoints): the
//! stages carry no `enorm`/`e_proj` combiner; instead stage 0 fuses the
//! TARGET's captured hiddens (layers 40/41/42, mean over hc streams,
//! concatenated to [3·h]) through `main_proj`/`main_norm`, and stage 2 carries
//! the Markov + confidence heads. Reference implementation:
//! `inference/model.py` in the official repo (`DSparkBlock` et al.); full port
//! design in `docs/dspark_port.md`; offline acceptance (3.81 tok/step ungated)
//! measured by `bench/deepseek-v4/dspark_probe/`.
//!
//! Each stage is loaded PIECEWISE — MoE via [`super::assemble::assemble_moe`],
//! mHC sites via `load_hc_site`, attention weights dequanted dense — rather
//! than as an assembled `TransformerLayer`: the drafter's attention is a
//! 5-row bidirectional block over a 128-entry `main_kv` ring, which the
//! layer's causal paged decode cannot express, so the propose forward
//! (`layers::dspark_head`) drives every op itself.
// Dead-code allowance is temporary: consumed by the forthcoming
// `DsparkDraftHead` proposer + factory wiring (docs/dspark_port.md, tasks
// #12–#13). Mirrors how `DeepseekV4MtpModule` landed ahead of its proposer.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::MoeLayer;
use crate::layers::qwen3_attention::{HcHeadWeights, HcSiteWeights};
use crate::weight_map::{DenseWeight, dense_auto};

/// DSpark hyper-parameters. The 0731 checkpoint does not repeat these in the
/// drafter shards, so the caller parses them from the checkpoint's
/// `config.json` (`dspark_block_size` etc.); [`DsparkParams::V4_FLASH_0731`]
/// is the shipped configuration for callers that load the shards standalone.
#[derive(Debug, Clone)]
pub struct DsparkParams {
    /// Draft tokens proposed per step (block rows). 0731: 5.
    pub block_size: usize,
    /// Filler token embedded into the non-committed block rows. 0731: 128799.
    pub noise_token_id: u32,
    /// Rank of the Markov bigram head. 0731: 256.
    pub markov_rank: usize,
    /// Target layers whose post-layer hc-mean hiddens feed `main_proj`,
    /// concatenated in order. 0731: [40, 41, 42].
    pub target_layer_ids: Vec<usize>,
    /// Drafter attention sliding window (ring capacity of `main_kv` rows
    /// per stage). 0731: 128.
    pub window: usize,
}

impl DsparkParams {
    pub const V4_FLASH_0731: fn() -> DsparkParams = || DsparkParams {
        block_size: 5,
        noise_token_id: 128799,
        markov_rank: 256,
        target_layer_ids: vec![40, 41, 42],
        window: 128,
    };
}

/// One drafter stage: MoE + mHC sites + dequanted-dense attention weights,
/// driven directly by the propose forward.
pub struct DsparkStage {
    /// 256-expert native-MXFP4 routed MoE + FP8→NVFP4 shared expert, built by
    /// the same `assemble_moe` the target layers use — its batched decode
    /// paths (`forward_kn`) run the 5-row block FFN.
    pub moe: MoeLayer,
    /// mHC mixing at the attention site (`hc_attn_fn/scale/base`).
    pub hc_attn: HcSiteWeights,
    /// mHC mixing at the FFN site (`hc_ffn_fn/scale/base`).
    pub hc_ffn: HcSiteWeights,
    /// Pre-attention RMSNorm (`attn_norm`).
    pub attn_norm: DenseWeight,
    /// Pre-FFN RMSNorm (`ffn_norm`).
    pub ffn_norm: DenseWeight,
    // ── MLA attention weights, dequanted to dense BF16 ──
    /// `[q_lora, h]` = [1024, 4096].
    pub wq_a: DenseWeight,
    /// `[heads·head_dim, q_lora]` = [32768, 1024].
    pub wq_b: DenseWeight,
    /// Q-LoRA RMSNorm `[q_lora]`.
    pub q_norm: DenseWeight,
    /// `[head_dim, h]` = [512, 4096] — MQA: one shared KV row.
    pub wkv: DenseWeight,
    /// KV RMSNorm `[head_dim]`.
    pub kv_norm: DenseWeight,
    /// Grouped O-LoRA down: `[groups·o_lora, heads·head_dim/groups]` = [8192, 4096]; group g
    /// reads attn cols `[g·(heads·head_dim/groups) ..)`.
    pub wo_a: DenseWeight,
    /// `[h, groups·o_lora]` = [4096, 8192].
    pub wo_b: DenseWeight,
    /// Per-head attention sink logits `[heads]` F32.
    pub attn_sink: DenseWeight,
}

/// The loaded DSpark drafter: 3 stages + target-fusion projection + heads.
/// Embedding and lm_head are shared with the target model and supplied at
/// proposer-build time (they are not present in the drafter shards).
pub struct DsparkDrafterModule {
    pub stages: Vec<DsparkStage>,
    /// `[h, target_layers·h]` — fuses the concatenated target captures.
    pub main_proj: DenseWeight,
    /// RMSNorm over the fused vector.
    pub main_norm: DenseWeight,
    /// Final RMSNorm (last stage) before the shared lm_head.
    pub norm: DenseWeight,
    /// Last stage's head hyper-connection collapse (`hc_head_*`).
    pub hc_head: Option<HcHeadWeights>,
    /// Markov bigram head, `[vocab, markov_rank]` each. `w1` is a gather
    /// table (embedding-style), `w2` a head (`logits += w2 · w1[prev]`).
    pub markov_w1: DenseWeight,
    pub markov_w2: DenseWeight,
    /// `[1, h + markov_rank]` — per-draft confidence logit (computed in F32;
    /// sigmoid > threshold keeps the chain).
    pub confidence_proj: DenseWeight,
    pub params: DsparkParams,
}

/// True iff `store` holds a DSpark drafter (vs a DFlash drafter or nothing).
/// `main_proj` only exists in the DSpark layout, so it is the cheapest marker.
pub fn store_is_dspark(store: &WeightStore) -> bool {
    store.contains("mtp.0.main_proj.weight")
}

/// Loads the DSpark drafter from its own store (the 0731 drafter shards).
///
/// `target_config` is the TARGET model's config; the drafter differs only in
/// expert count (the drafter MoE is the unpruned 256-expert set even when the
/// target is REAP-pruned), which is read from the gate shape rather than
/// trusted from config. Everything else — h, heads, lora ranks, hc_mult,
/// rope — is validated V4-Flash geometry shared with the target.
pub fn load_dspark_drafter(
    store: &WeightStore,
    target_config: &ModelConfig,
    params: DsparkParams,
    gpu: &dyn GpuBackend,
) -> Result<DsparkDrafterModule> {
    if !store_is_dspark(store) {
        bail!("drafter store has no mtp.0.main_proj.weight — not a DSpark checkpoint");
    }
    let h = target_config.hidden_size;

    // The drafter ships the unpruned expert set; the target may be REAP-pruned
    // (144 on DeepSeek-V4-Flash-162B). The gate is `[num_experts, h]` BF16, so
    // its shape is the authoritative count.
    let gate = store
        .get("mtp.0.ffn.gate.weight")
        .context("DSpark drafter store is missing mtp.0.ffn.gate.weight")?;
    let drafter_experts = gate.shape.first().copied().unwrap_or(0);
    if drafter_experts == 0 || h == 0 {
        bail!(
            "DSpark gate shape {:?} / hidden {h} — refusing to load",
            gate.shape
        );
    }
    let mut config = target_config.clone();
    config.num_experts = drafter_experts;

    let n_stages = (0..)
        .take_while(|s| store.contains(&format!("mtp.{s}.attn_norm.weight")))
        .count();
    if n_stages == 0 {
        bail!("DSpark drafter store has main_proj but no mtp.0.attn_norm.weight");
    }

    let qctx = crate::weight_map::QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    };

    let mut stages = Vec::with_capacity(n_stages);
    let mut last_hc_head: Option<HcHeadWeights> = None;
    for s in 0..n_stages {
        let prefix = format!("mtp.{s}");
        let ap = format!("{prefix}.attn");

        // `layer_idx = num_hidden_layers` ⇒ past every hash-routed layer, so
        // the MoE takes the learned-gate path (the drafter has no tid2eid).
        let moe = super::assemble::assemble_moe(
            store,
            &prefix,
            config.num_hidden_layers,
            true, // force_all_experts — drafter runs no-EP on rank 0
            &config,
            gpu,
            qctx,
        )
        .with_context(|| format!("assembling DSpark stage {s} MoE"))?;

        let hc_attn = super::assemble::load_hc_site(store, &prefix, "attn", &config, gpu)?;
        let hc_ffn = super::assemble::load_hc_site(store, &prefix, "ffn", &config, gpu)?;

        // Only the FINAL stage carries a head hyper-connection.
        if s == n_stages - 1 && config.hc_mult > 0 {
            let hc = config.hc_mult;
            let hc_dim = hc * h;
            last_hc_head = Some(HcHeadWeights {
                hc_fn: super::assemble::load_hc_f32(
                    store,
                    &[format!("{prefix}.hc_head_fn")],
                    hc * hc_dim,
                    gpu,
                )?,
                hc_base: super::assemble::load_hc_f32(
                    store,
                    &[format!("{prefix}.hc_head_base")],
                    hc,
                    gpu,
                )?,
                hc_scale: super::assemble::load_hc_f32(
                    store,
                    &[format!("{prefix}.hc_head_scale")],
                    1,
                    gpu,
                )?,
            });
        }

        stages.push(DsparkStage {
            moe,
            hc_attn,
            hc_ffn,
            attn_norm: dense_auto(store, &format!("{prefix}.attn_norm.weight"), gpu)?,
            ffn_norm: dense_auto(store, &format!("{prefix}.ffn_norm.weight"), gpu)?,
            wq_a: dense_auto(store, &format!("{ap}.wq_a.weight"), gpu)?,
            wq_b: dense_auto(store, &format!("{ap}.wq_b.weight"), gpu)?,
            q_norm: dense_auto(store, &format!("{ap}.q_norm.weight"), gpu)?,
            wkv: dense_auto(store, &format!("{ap}.wkv.weight"), gpu)?,
            kv_norm: dense_auto(store, &format!("{ap}.kv_norm.weight"), gpu)?,
            wo_a: dense_auto(store, &format!("{ap}.wo_a.weight"), gpu)?,
            wo_b: dense_auto(store, &format!("{ap}.wo_b.weight"), gpu)?,
            attn_sink: dense_auto(store, &format!("{ap}.attn_sink"), gpu)?,
        });
    }

    // ── DSpark-specific heads ──
    // Stage 0: target fusion. Last stage: final norm + Markov + confidence.
    // `dense_auto` dequants the FP8 block-scaled main_proj.
    let last = n_stages - 1;
    let main_proj = dense_auto(store, "mtp.0.main_proj.weight", gpu)?;
    let main_norm = dense_auto(store, "mtp.0.main_norm.weight", gpu)?;
    let norm = dense_auto(store, &format!("mtp.{last}.norm.weight"), gpu)?;
    let markov_w1 = dense_auto(store, &format!("mtp.{last}.markov_head.markov_w1.weight"), gpu)?;
    let markov_w2 = dense_auto(store, &format!("mtp.{last}.markov_head.markov_w2.weight"), gpu)?;
    let confidence_proj =
        dense_auto(store, &format!("mtp.{last}.confidence_head.proj.weight"), gpu)?;

    tracing::info!(
        "DSpark drafter loaded: {n_stages} V4 stages ({drafter_experts}-expert MoE) + \
         main_proj [{h}, {}] + Markov(rank {}) + confidence head; block_size={} \
         window={} target_layers={:?}",
        params.target_layer_ids.len() * h,
        params.markov_rank,
        params.block_size,
        params.window,
        params.target_layer_ids,
    );

    Ok(DsparkDrafterModule {
        stages,
        main_proj,
        main_norm,
        norm,
        hc_head: last_hc_head,
        markov_w1,
        markov_w2,
        confidence_proj,
        params,
    })
}
