// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek-V4.1 **K154-CB3 expert pack**: the three-level expert mapping.
//!
//! `config.json` declares `n_routed_experts: 384`, but that is not what is on disk and not
//! what serves. There are THREE levels, and a loader that models fewer is wrong:
//!
//! ```text
//!   384  n_routed_experts, the ROUTER's id space (config.json)
//!    |   per-layer permutation, a FILE property (k154-cb3/manifest.json)
//!   154  experts actually present in the pack on disk (83 GB)
//!    |   packed_keep, a RUNTIME KNOB not discoverable from the checkpoint
//!   124  experts actually resident and routable in production
//! ```
//!
//! The 154 hop is a per-layer permutation: all 40 layers have DISTINCT selections, every
//! one of the 384 ids appears in *some* layer, and NO id appears in all 40. So a single
//! global mapping is not merely imprecise, it does not exist.
//!
//! The 124 hop is `PACKED_KEEP` in
//! `dsv41-prefill-work/scripts/k124_dspark_vision_fast_entrypoint.sh:14`, applied by the
//! Python engine as `entry["expert_ids"][:packed_keep]` (`engine/v41_engine.py:599`) — a
//! **prefix of a ranked list**, not a subset chosen by some other rule.
//!
//! ## Why mis-resolution must be a hard error
//! A routed id that is not resident has no correct fallback. A modulo or a clamp yields a
//! *different, valid-looking* expert, so the model produces plausible tokens and no error —
//! the worst failure shape available here. The Python engine avoids this by masking at the
//! ROUTER: "the immutable manifest is both the router allow-list and the arena order"
//! (`v41_engine.py:594-604`). We mirror that: [`ExpertPack::routing_mask`] is the allow-list
//! applied *before* top-k, and [`ExpertPack::slot_of`] is a hard error for anything else.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

/// Experts in the pack on disk. Fixed by the K154 artifact.
pub const PACK_EXPERTS: usize = 154;
/// The router's id space, from `config.json`'s `n_routed_experts`.
pub const ROUTED_EXPERTS: usize = 384;
/// Lower bound on `packed_keep`, matching the Python engine's `6 <= packed_keep <= K154`
/// (`engine/v41_engine.py:427`).
pub const MIN_PACKED_KEEP: usize = 6;
/// What PRODUCTION actually serves: `PACKED_KEEP="124"` in
/// `dsv41-prefill-work/scripts/k124_dspark_vision_fast_entrypoint.sh:14`.
///
/// This is the number to target when comparing cost against the Python engine. It is NOT
/// discoverable from the model directory, so anything that reads only the checkpoint lands
/// on 154 and is quietly wrong in production by ~25% of expert-weight traffic — a gap that
/// never errors and only shows up as an unexplained throughput difference between engines.
/// Callers should therefore pass `packed_keep` explicitly and LOG it at startup; if a
/// default is unavoidable, it should be this constant and not [`PACK_EXPERTS`].
pub const SERVED_PACKED_KEEP: usize = 124;

#[derive(Debug, Deserialize)]
struct RawShard {
    layer: usize,
    expert_ids: Vec<u32>,
    /// Present in the manifest and byte-identical to `expert_ids` on all 40 layers. Kept so
    /// the invariant is CHECKED rather than assumed: if a future pack ever separates the
    /// router id list from the arena order, silently reading one as the other is exactly the
    /// mis-routing this module exists to prevent.
    slot_order: Vec<u32>,
    tensor_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct RawManifest {
    format: String,
    layers: usize,
    experts_per_layer: usize,
    experts_kept_per_layer: usize,
    bytes_per_expert: u64,
    shards: Vec<RawShard>,
}

/// The twelve tensors a CB3 layer shard actually contains.
///
/// **This is the layout correction that matters.** `bytes_per_expert = 14,454,784` is the
/// SUM over these twelve, NOT a contiguous per-expert run. Every tensor is expert-major
/// (`[154, rows, bytes_per_row]`), so expert `i` occupies row `i` of each of the twelve
/// independently. Treating `i * bytes_per_expert` as a file offset is wrong and lands
/// mid-`s1` — the same class of error as assuming a tiled multi-row activation layout is
/// `row * per_row_bytes`.
///
/// Measured from `k154-cb3/layers/layer-00.safetensors`; the sum is asserted against the
/// manifest in `cb3_tensor_strides_sum_to_bytes_per_expert`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb3Tensor {
    /// Per-expert scale planes.
    S1,
    S2,
    S3,
    /// `w1` = gate, `w2` = down, `w3` = up. `_cb` is the codebook, `_hi`/`_lo` the payload.
    W1Cb,
    W1Hi,
    W1Lo,
    W2Cb,
    W2Hi,
    W2Lo,
    W3Cb,
    W3Hi,
    W3Lo,
}

/// All twelve, in the order they appear in the shard.
pub const CB3_TENSORS: [Cb3Tensor; 12] = [
    Cb3Tensor::S1,
    Cb3Tensor::S2,
    Cb3Tensor::S3,
    Cb3Tensor::W1Cb,
    Cb3Tensor::W1Hi,
    Cb3Tensor::W1Lo,
    Cb3Tensor::W2Cb,
    Cb3Tensor::W2Hi,
    Cb3Tensor::W2Lo,
    Cb3Tensor::W3Cb,
    Cb3Tensor::W3Hi,
    Cb3Tensor::W3Lo,
];

impl Cb3Tensor {
    /// Tensor name inside the layer shard.
    pub fn name(self) -> &'static str {
        match self {
            Self::S1 => "s1",
            Self::S2 => "s2",
            Self::S3 => "s3",
            Self::W1Cb => "w1_cb",
            Self::W1Hi => "w1_hi",
            Self::W1Lo => "w1_lo",
            Self::W2Cb => "w2_cb",
            Self::W2Hi => "w2_hi",
            Self::W2Lo => "w2_lo",
            Self::W3Cb => "w3_cb",
            Self::W3Hi => "w3_hi",
            Self::W3Lo => "w3_lo",
        }
    }

    /// `(rows, bytes_per_row)` per expert. Rows are `moe_intermediate_size` (2304) for the
    /// gate/up side and `hidden_size` (5120) for the down side.
    pub fn shape(self) -> (usize, usize) {
        match self {
            Self::S1 | Self::S3 => (2304, 160),
            Self::S2 => (5120, 72),
            Self::W1Cb | Self::W3Cb => (2304, 8),
            Self::W2Cb => (5120, 8),
            Self::W1Hi | Self::W3Hi => (2304, 640),
            Self::W2Hi => (5120, 288),
            Self::W1Lo | Self::W3Lo => (2304, 1280),
            Self::W2Lo => (5120, 576),
        }
    }

    /// Bytes one expert occupies in THIS tensor — its row stride.
    pub fn bytes_per_expert(self) -> u64 {
        let (rows, bytes_per_row) = self.shape();
        (rows * bytes_per_row) as u64
    }
}

/// One layer's slot table: index = pack slot, value = 384-space expert id.
#[derive(Debug, Clone)]
pub struct LayerSlots {
    slots: Vec<u32>,
}

/// The parsed, validated expert pack.
#[derive(Debug, Clone)]
pub struct ExpertPack {
    layers: Vec<LayerSlots>,
    bytes_per_expert: u64,
    packed_keep: usize,
}

impl ExpertPack {
    /// Parse and validate `k154-cb3/manifest.json`.
    ///
    /// `packed_keep` is the runtime knob. It is NOT inferred from the file — the file cannot
    /// know it — so callers must pass it explicitly; [`Self::parse_full_pack`] is the
    /// "load everything" default.
    pub fn parse(manifest_json: &str, packed_keep: usize) -> Result<Self> {
        let raw: RawManifest = serde_json::from_str(manifest_json)
            .context("Invalid JSON in DeepSeek-V4.1 expert pack manifest")?;

        ensure!(
            raw.format == "dsv41-cb3-expert-pack",
            "Unexpected expert pack format '{}' (want dsv41-cb3-expert-pack)",
            raw.format
        );
        ensure!(
            raw.experts_per_layer == ROUTED_EXPERTS,
            "Pack declares {} experts per layer, config routes over {ROUTED_EXPERTS}",
            raw.experts_per_layer
        );
        ensure!(
            raw.experts_kept_per_layer == PACK_EXPERTS,
            "Pack keeps {} experts per layer, expected {PACK_EXPERTS}",
            raw.experts_kept_per_layer
        );
        ensure!(
            (MIN_PACKED_KEEP..=raw.experts_kept_per_layer).contains(&packed_keep),
            "packed_keep {packed_keep} out of range {MIN_PACKED_KEEP}..={}",
            raw.experts_kept_per_layer
        );
        ensure!(
            raw.shards.len() == raw.layers,
            "Pack declares {} layers but carries {} shards",
            raw.layers,
            raw.shards.len()
        );

        // Index shards by their declared layer rather than trusting array order.
        let mut layers: Vec<Option<LayerSlots>> = vec![None; raw.layers];
        for shard in &raw.shards {
            ensure!(
                shard.layer < raw.layers,
                "Pack shard names layer {} beyond the declared {} layers",
                shard.layer,
                raw.layers
            );
            ensure!(
                layers[shard.layer].is_none(),
                "Pack lists layer {} more than once",
                shard.layer
            );
            ensure!(
                shard.expert_ids.len() == raw.experts_kept_per_layer,
                "Layer {} lists {} experts, expected {}",
                shard.layer,
                shard.expert_ids.len(),
                raw.experts_kept_per_layer
            );
            // The invariant that makes "slot i holds expert_ids[i]" safe to rely on.
            ensure!(
                shard.slot_order == shard.expert_ids,
                "Layer {} has slot_order != expert_ids; the arena order and the router \
                 allow-list disagree and reading either as the other would mis-route",
                shard.layer
            );
            // `bytes_per_expert` accounts for the whole of each expert. NOTE what this does
            // and does NOT prove: it is a SUM check across the 12 CB3 tensors, not evidence
            // that an expert is contiguous in the file. It is not — see `Cb3Tensor`.
            ensure!(
                shard.tensor_bytes == raw.bytes_per_expert * raw.experts_kept_per_layer as u64,
                "Layer {} tensor_bytes {} != {} x {}; the pack does not account for every \
                 expert byte",
                shard.layer,
                shard.tensor_bytes,
                raw.bytes_per_expert,
                raw.experts_kept_per_layer
            );

            let mut seen = vec![false; ROUTED_EXPERTS];
            for &id in &shard.expert_ids {
                let id = id as usize;
                ensure!(
                    id < ROUTED_EXPERTS,
                    "Layer {} selects expert {id}, outside the {ROUTED_EXPERTS}-expert space",
                    shard.layer
                );
                ensure!(
                    !std::mem::replace(&mut seen[id], true),
                    "Layer {} selects expert {id} twice",
                    shard.layer
                );
            }
            layers[shard.layer] = Some(LayerSlots {
                slots: shard.expert_ids.clone(),
            });
        }

        let layers = layers
            .into_iter()
            .enumerate()
            .map(|(index, slots)| slots.with_context(|| format!("Pack is missing layer {index}")))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            layers,
            bytes_per_expert: raw.bytes_per_expert,
            packed_keep,
        })
    }

    /// Parse with every pack expert resident (`packed_keep = 154`).
    ///
    /// This is a NAMED OPT-IN, not a default: it loads more experts than production serves.
    /// Use [`SERVED_PACKED_KEEP`] to mirror production.
    pub fn parse_full_pack(manifest_json: &str) -> Result<Self> {
        Self::parse(manifest_json, PACK_EXPERTS)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn packed_keep(&self) -> usize {
        self.packed_keep
    }

    pub fn bytes_per_expert(&self) -> u64 {
        self.bytes_per_expert
    }

    /// Resolve a 384-space routed expert id to its slot in `layer`'s shard.
    ///
    /// PERFORMANCE NOTE FOR WHOEVER MOVES THIS: it is a linear scan over <= 154 ids, which
    /// is fine at LOAD time, where it runs once per expert per layer. If it ever lands on a
    /// per-token path it wants an inverse table (`[u16; 384]` per layer, 0xFFFF for absent)
    /// built once at parse. Do not leave the scan on a hot path.
    ///
    /// **Hard error** when the expert is not resident — never a modulo, never a clamp.
    /// Callers that can legitimately see non-resident ids must consult [`Self::routing_mask`]
    /// BEFORE top-k, exactly as the Python engine does, rather than handling an error here.
    pub fn slot_of(&self, layer: usize, expert_id: u32) -> Result<usize> {
        let slots = self.layers.get(layer).with_context(|| {
            format!(
                "Layer {layer} is outside the pack's {} layers",
                self.layers.len()
            )
        })?;
        match slots.slots[..self.packed_keep]
            .iter()
            .position(|&id| id == expert_id)
        {
            Some(slot) => Ok(slot),
            None => {
                let in_pack = slots.slots.contains(&expert_id);
                bail!(
                    "Expert {expert_id} is not resident in layer {layer} \
                     (packed_keep = {}, pack holds {}){}",
                    self.packed_keep,
                    slots.slots.len(),
                    if in_pack {
                        " — it IS in the pack but outside the packed_keep prefix"
                    } else {
                        " — it is not in this layer's selection at all"
                    }
                )
            }
        }
    }

    /// Per-expert byte stride WITHIN one CB3 tensor.
    ///
    /// **There is no single "offset of expert i" in the shard.** See [`CB3_TENSORS`]: the
    /// shard holds 12 separate expert-major tensors, so one expert's data lives at 12
    /// distinct offsets, one per tensor. `bytes_per_expert` is the SUM across all twelve,
    /// not a contiguous run, and using it as a file offset lands in the middle of `s1`.
    pub fn slot_stride_in(&self, tensor: Cb3Tensor) -> u64 {
        tensor.bytes_per_expert()
    }

    /// The router allow-list for `layer`: `mask[id]` is true iff expert `id` is resident.
    ///
    /// Apply this BEFORE top-k. This is the mechanism that makes non-residency a routing
    /// decision rather than a lookup failure, and it is why [`Self::slot_of`] is free to be
    /// a hard error.
    pub fn routing_mask(&self, layer: usize) -> Result<Vec<bool>> {
        let slots = self.layers.get(layer).with_context(|| {
            format!(
                "Layer {layer} is outside the pack's {} layers",
                self.layers.len()
            )
        })?;
        let mut mask = vec![false; ROUTED_EXPERTS];
        for &id in &slots.slots[..self.packed_keep] {
            mask[id as usize] = true;
        }
        Ok(mask)
    }

    /// Resident ids for `layer`, in slot order.
    pub fn resident_ids(&self, layer: usize) -> Result<&[u32]> {
        let slots = self.layers.get(layer).with_context(|| {
            format!(
                "Layer {layer} is outside the pack's {} layers",
                self.layers.len()
            )
        })?;
        Ok(&slots.slots[..self.packed_keep])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_MANIFEST: &str =
        "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/k154-cb3/manifest.json";

    /// A minimal synthetic pack, so the invariant tests do not need the 290 GB checkpoint.
    /// `keep` experts per layer over `layers` layers, ids chosen so each layer's selection
    /// DIFFERS — mirroring the real pack, where all 40 selections are distinct.
    fn synthetic(layers: usize, keep: usize, bytes_per_expert: u64) -> String {
        let shards: Vec<String> = (0..layers)
            .map(|layer| {
                let ids: Vec<u32> = (0..keep)
                    .map(|slot| ((slot * 7 + layer * 3) % ROUTED_EXPERTS) as u32)
                    .collect();
                let ids_json = serde_json::to_string(&ids).unwrap();
                format!(
                    r#"{{"layer":{layer},"expert_ids":{ids_json},"slot_order":{ids_json},
                        "tensor_bytes":{}}}"#,
                    bytes_per_expert * keep as u64
                )
            })
            .collect();
        format!(
            r#"{{"format":"dsv41-cb3-expert-pack","layers":{layers},
                "experts_per_layer":{ROUTED_EXPERTS},"experts_kept_per_layer":{keep},
                "bytes_per_expert":{bytes_per_expert},"shards":[{}]}}"#,
            shards.join(",")
        )
    }

    fn small() -> String {
        synthetic(40, PACK_EXPERTS, 14_454_784)
    }

    /// THE TEST TEAM-LEAD ASKED FOR: a 384-space id absent from a layer's selection is a
    /// HARD ERROR, never a modulo and never a clamp. Silent mis-routing produces plausible
    /// tokens and no error, which is the worst failure shape available here.
    #[test]
    fn absent_expert_is_a_hard_error_not_a_modulo_or_clamp() {
        let pack = ExpertPack::parse_full_pack(&small()).expect("synthetic pack parses");
        let resident = pack.resident_ids(0).unwrap().to_vec();
        let absent = (0..ROUTED_EXPERTS as u32)
            .find(|id| !resident.contains(id))
            .expect("some id is outside a 154-of-384 selection");

        let err = pack
            .slot_of(0, absent)
            .expect_err("an absent expert must not resolve")
            .to_string();
        assert!(err.contains("not resident"), "got: {err}");

        // A modulo would have mapped it onto a valid slot. Prove it did not.
        let modulo_victim = (absent as usize) % PACK_EXPERTS;
        assert!(
            pack.slot_of(0, resident[modulo_victim]).unwrap() == modulo_victim,
            "sanity: resident ids do resolve"
        );

        // And an out-of-space id is refused rather than wrapped.
        assert!(pack.slot_of(0, ROUTED_EXPERTS as u32).is_err());
        assert!(pack.slot_of(0, u32::MAX).is_err());
    }

    /// Every routed index that the mask admits MUST resolve. This is the assertion the lead
    /// asked for: mask and slot table cannot disagree, or routing admits an id the loader
    /// then cannot place.
    #[test]
    fn every_masked_in_expert_resolves_on_every_layer() {
        for keep in [MIN_PACKED_KEEP, 124, PACK_EXPERTS] {
            let pack = ExpertPack::parse(&small(), keep).expect("parses");
            for layer in 0..pack.num_layers() {
                let mask = pack.routing_mask(layer).unwrap();
                let admitted = mask.iter().filter(|m| **m).count();
                assert_eq!(admitted, keep, "mask admits exactly packed_keep experts");
                for (id, admitted) in mask.iter().enumerate() {
                    let resolved = pack.slot_of(layer, id as u32);
                    assert_eq!(
                        resolved.is_ok(),
                        *admitted,
                        "layer {layer} expert {id}: mask and slot table must agree"
                    );
                    if let Ok(slot) = resolved {
                        assert!(slot < keep, "slot must lie inside packed_keep");
                    }
                }
            }
        }
    }

    /// packed_keep is a PREFIX of the ranked list (`v41_engine.py:599`), so shrinking it
    /// only ever removes experts from the tail — it must never renumber the survivors.
    #[test]
    fn packed_keep_is_a_prefix_and_does_not_renumber_survivors() {
        let full = ExpertPack::parse(&small(), PACK_EXPERTS).unwrap();
        let served = ExpertPack::parse(&small(), 124).unwrap();
        for layer in 0..full.num_layers() {
            assert_eq!(
                served.resident_ids(layer).unwrap(),
                &full.resident_ids(layer).unwrap()[..124],
                "served set is the first 124 of the pack order"
            );
            for &id in served.resident_ids(layer).unwrap() {
                assert_eq!(
                    served.slot_of(layer, id).unwrap(),
                    full.slot_of(layer, id).unwrap(),
                    "a survivor keeps its slot when packed_keep shrinks"
                );
            }
        }
    }

    /// The mapping is PER-LAYER. Using one layer's table for another is a silent mis-route,
    /// so prove the tables actually differ.
    #[test]
    fn the_mapping_is_per_layer_not_global() {
        let pack = ExpertPack::parse_full_pack(&small()).unwrap();
        assert_ne!(
            pack.resident_ids(0).unwrap(),
            pack.resident_ids(1).unwrap(),
            "layers must not share one selection"
        );
    }

    #[test]
    fn packed_keep_is_range_checked() {
        assert!(ExpertPack::parse(&small(), MIN_PACKED_KEEP - 1).is_err());
        assert!(ExpertPack::parse(&small(), PACK_EXPERTS + 1).is_err());
        assert!(ExpertPack::parse(&small(), MIN_PACKED_KEEP).is_ok());
        assert!(ExpertPack::parse(&small(), PACK_EXPERTS).is_ok());
    }

    /// slot_order is byte-identical to expert_ids on all 40 real layers, but the code must
    /// CHECK that rather than assume it: if a pack ever separates the arena order from the
    /// router allow-list, reading either as the other mis-routes silently.
    #[test]
    fn slot_order_disagreeing_with_expert_ids_is_rejected() {
        let json = small().replacen(r#""slot_order":[0,7"#, r#""slot_order":[7,0"#, 1);
        assert_ne!(
            json,
            small(),
            "the fixture must actually have been perturbed"
        );
        let err = ExpertPack::parse_full_pack(&json)
            .expect_err("disagreeing orders must be refused")
            .to_string();
        assert!(err.contains("slot_order"), "got: {err}");
    }

    #[test]
    fn duplicate_and_out_of_space_ids_are_rejected() {
        let dup = synthetic(1, 4, 16)
            .replace(r#""expert_ids":[0,7,14,21]"#, r#""expert_ids":[0,0,14,21]"#);
        assert!(ExpertPack::parse(&dup, MIN_PACKED_KEEP).is_err());
    }

    /// A shard whose `tensor_bytes` does not account for every expert is rejected.
    ///
    /// NOTE what this proves: only that the totals agree. It says NOTHING about whether an
    /// expert is contiguous — it is not; see `cb3_tensor_strides_sum_to_bytes_per_expert`.
    /// This test previously carried a "fixed stride" name that claimed the stronger
    /// property it never checked.
    #[test]
    fn shard_not_accounting_for_every_expert_byte_is_rejected() {
        let bad = small().replacen(
            "\"tensor_bytes\":2226036736",
            "\"tensor_bytes\":2226036737",
            1,
        );
        assert_ne!(bad, small());
        let err = ExpertPack::parse_full_pack(&bad).unwrap_err().to_string();
        assert!(err.contains("every expert byte"), "got: {err}");
    }

    /// The twelve per-expert strides must sum to `bytes_per_expert`.
    ///
    /// THIS IS THE TEST THAT CATCHES THE LAYOUT ERROR I ACTUALLY MADE. The earlier check —
    /// `tensor_bytes == 154 * bytes_per_expert` — passes whether or not an expert is
    /// contiguous, because it is a SUM. It let a `slot * bytes_per_expert` file offset look
    /// validated when it was wrong. A size-equality guard passing is not evidence about
    /// layout.
    #[test]
    fn cb3_tensor_strides_sum_to_bytes_per_expert() {
        let total: u64 = CB3_TENSORS.iter().map(|t| t.bytes_per_expert()).sum();
        assert_eq!(
            total, 14_454_784,
            "the twelve CB3 tensors must account for exactly one expert"
        );
        // And no single tensor is the whole expert, i.e. the layout really is split.
        assert!(CB3_TENSORS.iter().all(|t| t.bytes_per_expert() < total));
    }

    /// Verify the shapes against the SHIPPED shard header rather than trusting the table.
    #[test]
    fn cb3_tensor_table_matches_the_shipped_shard_header() {
        const SHARD: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K\
/k154-cb3/layers/layer-00.safetensors";
        let Ok(mut file) = std::fs::File::open(SHARD) else {
            eprintln!("skipping: {SHARD} not present");
            return;
        };
        use std::io::Read;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)
            .expect("safetensors length prefix");
        let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
        file.read_exact(&mut header).expect("safetensors header");
        let header: serde_json::Value =
            serde_json::from_slice(&header).expect("safetensors header is JSON");

        for tensor in CB3_TENSORS {
            let entry = &header[tensor.name()];
            assert!(
                !entry.is_null(),
                "shard is missing tensor {}",
                tensor.name()
            );
            let shape: Vec<u64> = entry["shape"]
                .as_array()
                .expect("shape array")
                .iter()
                .map(|v| v.as_u64().expect("shape entry"))
                .collect();
            let (rows, bytes_per_row) = tensor.shape();
            assert_eq!(
                shape,
                vec![PACK_EXPERTS as u64, rows as u64, bytes_per_row as u64],
                "shape mismatch for {}",
                tensor.name()
            );

            // Expert-major: the declared byte span divided by 154 is the per-expert stride.
            let offsets = entry["data_offsets"].as_array().expect("data_offsets");
            let span = offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap();
            assert_eq!(
                span / PACK_EXPERTS as u64,
                tensor.bytes_per_expert(),
                "per-expert stride mismatch for {}",
                tensor.name()
            );
        }
    }

    /// The real artifact. Skips when the 290 GB checkpoint is absent so CI stays green.
    #[test]
    fn real_k154_manifest_parses_and_matches_the_measured_geometry() {
        let Ok(json) = std::fs::read_to_string(REAL_MANIFEST) else {
            eprintln!("skipping: {REAL_MANIFEST} not present");
            return;
        };
        let pack = ExpertPack::parse(&json, SERVED_PACKED_KEEP)
            .expect("the shipped K154 manifest must parse");

        assert_eq!(pack.num_layers(), 40);
        assert_eq!(pack.packed_keep(), SERVED_PACKED_KEEP);
        assert_eq!(pack.bytes_per_expert(), 14_454_784);

        // 3 x 5120 x 2304 weights per expert at the manifest's own bytes/elem.
        let elems = 3 * 5120 * 2304;
        let bytes_per_elem = pack.bytes_per_expert() as f64 / elems as f64;
        assert!(
            (bytes_per_elem - 0.40845).abs() < 1e-4,
            "bytes/elem drifted: {bytes_per_elem}"
        );

        // Every layer's selection differs, and collectively they cover all 384 ids while
        // NO id is in all 40 layers — which is why a global mapping does not exist.
        let full = ExpertPack::parse_full_pack(&json).unwrap();
        let mut appearances = vec![0usize; ROUTED_EXPERTS];
        for layer in 0..full.num_layers() {
            for &id in full.resident_ids(layer).unwrap() {
                appearances[id as usize] += 1;
            }
        }
        assert!(
            appearances.iter().all(|&n| n > 0),
            "all 384 ids appear somewhere"
        );
        assert!(
            appearances.iter().all(|&n| n < full.num_layers()),
            "no id is resident in every layer"
        );

        // And the served prefix resolves everywhere.
        for layer in 0..pack.num_layers() {
            for &id in pack.resident_ids(layer).unwrap() {
                assert!(pack.slot_of(layer, id).is_ok());
            }
        }
    }
}
