// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1's built-in DSpark drafter: the three `mtp.{0,1,2}` blocks plus the seed and
//! head tensors (`engine/model.py::MTPWeights`, `dspark_seed`, `dspark_draft`).
//!
//! Each block is a full V4.1 block with a window-only attention (compress_ratio 0), a
//! 128-expert top-3 MoE whose experts are **FP4** (packed e2m1, low nibble = even element, one
//! UE8M0 scale per 32 K), and the usual FP8 shared expert. `mtp.0` adds `main_proj`/`main_norm`
//! (the seed from the hc-mean of L37-39); `mtp.2` adds the final `norm`, the Markov head and the
//! confidence head.
//!
//! ## Where the 7.2 GB of draft experts live
//! In the `WeightStore`. Serving skips `mtp.*` unless `ATLAS_DSV41_DSPARK=1`
//! (`skip_tensor_for_serving`), so the drafter costs nothing when speculation is off; when it
//! is on, the store owns the bytes and frees them with the store — the same ownership as every
//! other dense tensor. The "arena" is therefore a table of store pointers, all 128 experts
//! resident, slot == expert id: there is no residency mask to get wrong here.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::attn_block::V41AttnWeights;
use super::fwd::{SharedExpert, V41Dims};
use super::ops::{Fp8Linear, HcParams, bf16_tensor, f32_tensor};

pub const MTP_BLOCKS: usize = 3;
/// `dspark_n_routed_experts`.
pub const MTP_EXPERTS: usize = 128;
/// `dspark_num_experts_per_tok`.
pub const MTP_TOP_K: usize = 3;
/// `dspark_block_size`.
pub const DSPARK_BLOCK: usize = 5;
/// `dspark_noise_token_id`.
pub const DSPARK_NOISE_TOKEN: u32 = 128_799;
/// `dspark_target_layer_ids`: the main layers whose INPUT stream's hc-mean seeds the drafter.
pub const DSPARK_TARGET_LAYERS: [usize; 3] = [37, 38, 39];
/// `dspark_markov_rank`.
pub const MARKOV_RANK: usize = 256;

/// An FP4 e2m1 weight `[n, k]`: packed `[n, k/2]` bytes (low nibble = even k) + UE8M0 `[n, k/32]`.
#[derive(Clone, Copy, Debug)]
pub struct Fp4Linear {
    pub weight: DevicePtr,
    pub scale: DevicePtr,
    pub n: usize,
    pub k: usize,
}

impl Fp4Linear {
    pub fn load(store: &WeightStore, prefix: &str, n: usize, k: usize) -> Result<Self> {
        let w = store.get(&format!("{prefix}.weight"))?;
        let s = store.get(&format!("{prefix}.scale"))?;
        ensure!(w.dtype == WeightDtype::UInt8, "{prefix}.weight: expected packed FP4 (I8/U8), got {:?}", w.dtype);
        ensure!(w.shape == [n, k / 2], "{prefix}.weight: expected [{n}, {}], got {:?}", k / 2, w.shape);
        ensure!(s.dtype == WeightDtype::FP8E8M0, "{prefix}.scale: expected F8_E8M0, got {:?}", s.dtype);
        ensure!(s.shape == [n, k / 32], "{prefix}.scale: expected per-32 scales [{n}, {}], got {:?}", k / 32, s.shape);
        Ok(Self { weight: w.ptr, scale: s.ptr, n, k })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MtpExpert {
    pub w1: Fp4Linear,
    pub w2: Fp4Linear,
    pub w3: Fp4Linear,
}

/// `ffn.gate`: weight bf16 `[128, hidden]` (the reference widens it to fp32), biases fp32 `[128]`.
#[derive(Clone, Copy, Debug)]
pub struct MtpRouter {
    pub weight: DevicePtr,
    pub bias: DevicePtr,
    pub bias_vl: DevicePtr,
}

pub struct MtpBlock {
    pub k: usize,
    pub attn: V41AttnWeights,
    pub attn_norm: DevicePtr,
    pub ffn_norm: DevicePtr,
    pub hc_attn: HcParams,
    pub hc_ffn: HcParams,
    pub shared: SharedExpert,
    pub router: MtpRouter,
    /// All 128, slot == expert id.
    pub experts: Vec<MtpExpert>,
}

/// The seed and head tensors spread over mtp.0 and mtp.2.
pub struct MtpHead {
    /// `mtp.0.main_proj` fp8 `[hidden, 3 * hidden]` on the hc-mean of L37-39.
    pub main_proj: Fp8Linear,
    pub main_norm: DevicePtr,
    /// `mtp.2.norm` before the LM head.
    pub norm: DevicePtr,
    /// `markov_head.embed` / `.head`, bf16 `[vocab, 256]`.
    pub markov_embed: DevicePtr,
    pub markov_head: DevicePtr,
    /// `confidence_head.proj` bf16 `[1, hidden + 256]` (reported, not acted on, in production).
    pub conf_proj: DevicePtr,
}

pub struct DsparkWeights {
    pub blocks: Vec<MtpBlock>,
    pub head: MtpHead,
}

impl DsparkWeights {
    /// Every name resolves against the store with an exact dtype and shape, or this fails.
    pub fn load(store: &WeightStore, dims: &V41Dims, vocab: usize) -> Result<Self> {
        let (d, inter) = (dims.hidden, dims.moe_inter);
        let blocks = (0..MTP_BLOCKS)
            .map(|k| {
                let p = format!("mtp.{k}");
                let experts = (0..MTP_EXPERTS)
                    .map(|e| {
                        let ep = format!("{p}.ffn.experts.{e}");
                        Ok(MtpExpert {
                            w1: Fp4Linear::load(store, &format!("{ep}.w1"), inter, d)?,
                            w2: Fp4Linear::load(store, &format!("{ep}.w2"), d, inter)?,
                            w3: Fp4Linear::load(store, &format!("{ep}.w3"), inter, d)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(MtpBlock {
                    k,
                    // The block index a main-layer-shaped consumer sees: n_layers + k, as the
                    // reference's MTPWeights.layer.
                    attn: V41AttnWeights::load_prefixed(store, &format!("{p}.attn"), 40 + k, d)?,
                    attn_norm: bf16_tensor(store, &format!("{p}.attn_norm.weight"), &[d])?,
                    ffn_norm: bf16_tensor(store, &format!("{p}.ffn_norm.weight"), &[d])?,
                    hc_attn: HcParams::load(store, &format!("{p}.hc_attn"), dims.hc, d)?,
                    hc_ffn: HcParams::load(store, &format!("{p}.hc_ffn"), dims.hc, d)?,
                    shared: SharedExpert::load_prefixed(store, &format!("{p}.ffn.shared_experts"), dims)?,
                    router: MtpRouter {
                        weight: bf16_tensor(store, &format!("{p}.ffn.gate.weight"), &[MTP_EXPERTS, d])?,
                        bias: f32_tensor(store, &format!("{p}.ffn.gate.bias"), &[MTP_EXPERTS])?,
                        bias_vl: f32_tensor(store, &format!("{p}.ffn.gate.bias_vl"), &[MTP_EXPERTS])?,
                    },
                    experts,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let head = MtpHead {
            main_proj: Fp8Linear::load(store, "mtp.0.main_proj", d, DSPARK_TARGET_LAYERS.len() * d)?,
            main_norm: bf16_tensor(store, "mtp.0.main_norm.weight", &[d])?,
            norm: bf16_tensor(store, "mtp.2.norm.weight", &[d])?,
            markov_embed: bf16_tensor(store, "mtp.2.markov_head.embed.weight", &[vocab, MARKOV_RANK])?,
            markov_head: bf16_tensor(store, "mtp.2.markov_head.head.weight", &[vocab, MARKOV_RANK])?,
            conf_proj: bf16_tensor(store, "mtp.2.confidence_head.proj.weight", &[1, d + MARKOV_RANK])?,
        };
        Ok(Self { blocks, head })
    }
}

/// Bytes the drafter adds to the serving store (`mtp.*`), for the memory plan.
pub fn mtp_store_bytes(store: &WeightStore) -> usize {
    store.names().filter(|n| n.starts_with("mtp.")).filter_map(|n| store.get(n).ok()).map(|t| t.byte_size()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const INDEX: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/model.safetensors.index.json";

    /// Every name `DsparkWeights::load` builds exists in the shipped index, for all 3 blocks and
    /// all 128 experts — string-formatted names are otherwise invisible until a serve.
    #[test]
    fn every_mtp_name_resolves_against_the_shipped_index() {
        let Ok(raw) = std::fs::read_to_string(INDEX) else {
            eprintln!("skipping: {INDEX} not present");
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let map = v["weight_map"].as_object().unwrap();
        let mut want = vec![
            "mtp.0.main_proj.weight".to_string(),
            "mtp.0.main_proj.scale".into(),
            "mtp.0.main_norm.weight".into(),
            "mtp.2.norm.weight".into(),
            "mtp.2.markov_head.embed.weight".into(),
            "mtp.2.markov_head.head.weight".into(),
            "mtp.2.confidence_head.proj.weight".into(),
        ];
        for k in 0..MTP_BLOCKS {
            let p = format!("mtp.{k}");
            for t in ["attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight", "attn.wo_a.weight", "attn.wo_b.weight",
                "attn.q_norm.weight", "attn.kv_norm.weight", "attn.attn_sink", "attn_norm.weight", "ffn_norm.weight",
                "hc_attn_fn", "hc_ffn_fn", "ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl",
                "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight", "ffn.shared_experts.w3.weight"] {
                want.push(format!("{p}.{t}"));
            }
            for e in 0..MTP_EXPERTS {
                for w in ["w1", "w2", "w3"] {
                    want.push(format!("{p}.ffn.experts.{e}.{w}.weight"));
                    want.push(format!("{p}.ffn.experts.{e}.{w}.scale"));
                }
            }
        }
        let missing: Vec<&String> = want.iter().filter(|n| !map.contains_key(n.as_str())).collect();
        assert!(missing.is_empty(), "{} MTP names missing, e.g. {:?}", missing.len(), &missing[..missing.len().min(5)]);
        // Negative control: a name the loader must NOT find (the 129th expert) is absent, so the
        // check above can fail.
        assert!(!map.contains_key("mtp.0.ffn.experts.128.w1.weight"));
    }
}
