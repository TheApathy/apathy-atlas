// SPDX-License-Identifier: AGPL-3.0-only

//! Reader for the DeepSeek-V4.1 **K154-CB3 expert pack** shards.
//!
//! Pairs with [`atlas_core::config::ExpertPack`], which owns the 384 -> 154 -> 124 expert
//! mapping. This module owns only the bytes: given a layer and a 384-space routed expert
//! id, produce that expert's CB3 payload out of `k154-cb3/layers/layer-NN.safetensors`.
//!
//! ## The layout, which is not the obvious one
//! A shard is **twelve separate expert-major tensors**, not 154 contiguous expert blocks.
//! Each is `[154, rows, bytes_per_row]`, so one expert is twelve row-slices at twelve
//! distinct offsets. `bytes_per_expert` (14,454,784) is their SUM. Indexing an expert as
//! `slot * bytes_per_expert` lands mid-`s1`; see [`atlas_core::config::Cb3Tensor`].
//!
//! ## No GPU, and no CB3 kernel
//! This reads and validates bytes. It does not decode CB3, and nothing in-tree can yet:
//! the weights it returns are still packed. Upload and decode belong to the loader, which
//! needs kernels that do not exist.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::{CB3_TENSORS, Cb3Tensor, ExpertPack};
use memmap2::Mmap;

/// One expert's twelve packed CB3 byte ranges, borrowed from the mapped shard.
pub struct ExpertBytes<'a> {
    /// Indexed in [`CB3_TENSORS`] order.
    planes: [&'a [u8]; 12],
    expert_id: u32,
    slot: usize,
}

impl<'a> ExpertBytes<'a> {
    /// Packed bytes for one CB3 tensor of this expert.
    pub fn plane(&self, tensor: Cb3Tensor) -> &'a [u8] {
        let index = CB3_TENSORS
            .iter()
            .position(|candidate| *candidate == tensor)
            .expect("CB3_TENSORS is exhaustive over Cb3Tensor");
        self.planes[index]
    }

    /// The 384-space routed id this expert answers to.
    pub fn expert_id(&self) -> u32 {
        self.expert_id
    }

    /// Its slot within the layer's pack.
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Total packed bytes across all twelve planes. Equals the manifest's
    /// `bytes_per_expert`, which is asserted on construction.
    pub fn total_bytes(&self) -> usize {
        self.planes.iter().map(|plane| plane.len()).sum()
    }
}

/// A memory-mapped CB3 layer shard.
///
/// Mapped rather than read: a shard is ~2.2 GB and a full pack is 83 GB, so reading an
/// expert must not pull in the layer. `Mmap` also leaves residency to the page cache
/// instead of this process.
pub struct LayerShard {
    map: Mmap,
    /// Byte offset of each tensor's data, in [`CB3_TENSORS`] order, relative to file start.
    plane_offsets: [usize; 12],
    layer: usize,
}

impl LayerShard {
    /// Shard path for `layer` under a `k154-cb3` directory.
    pub fn path_for(pack_dir: &Path, layer: usize) -> PathBuf {
        pack_dir
            .join("layers")
            .join(format!("layer-{layer:02}.safetensors"))
    }

    /// Map and validate one layer shard against the parsed pack.
    pub fn open(pack_dir: &Path, layer: usize, pack: &ExpertPack) -> Result<Self> {
        let path = Self::path_for(pack_dir, layer);
        let file = File::open(&path)
            .with_context(|| format!("Failed to open CB3 shard {}", path.display()))?;
        // SAFETY: the shard is a read-only artifact. A concurrent writer truncating it
        // would be undefined, but nothing in this system writes to the checkpoint, and the
        // pack is content-addressed by sha256 in the manifest.
        let map = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap CB3 shard {}", path.display()))?;

        ensure!(
            map.len() >= 8,
            "CB3 shard {} is too short to hold a safetensors header",
            path.display()
        );
        let header_len = u64::from_le_bytes(map[..8].try_into().expect("8 bytes")) as usize;
        let data_start = 8usize
            .checked_add(header_len)
            .context("CB3 shard header length overflows")?;
        ensure!(
            data_start <= map.len(),
            "CB3 shard {} header runs past end of file",
            path.display()
        );
        let header: serde_json::Value = serde_json::from_slice(&map[8..data_start])
            .with_context(|| format!("CB3 shard {} header is not JSON", path.display()))?;

        let experts = pack.pack_experts();
        let mut plane_offsets = [0usize; 12];
        for (index, tensor) in CB3_TENSORS.iter().enumerate() {
            let entry = &header[tensor.name()];
            ensure!(
                !entry.is_null(),
                "CB3 shard for layer {layer} is missing tensor {}",
                tensor.name()
            );

            // Shape must be [experts, rows, bytes_per_row] exactly. A mismatch means the
            // stride table and the file disagree, and every offset below would be wrong.
            let shape: Vec<u64> = entry["shape"]
                .as_array()
                .with_context(|| format!("tensor {} has no shape", tensor.name()))?
                .iter()
                .map(|value| value.as_u64().context("shape entry is not an integer"))
                .collect::<Result<_>>()?;
            let (rows, bytes_per_row) = tensor.shape();
            ensure!(
                shape == vec![experts as u64, rows as u64, bytes_per_row as u64],
                "CB3 layer {layer} tensor {} has shape {:?}, expected [{experts}, {rows}, {bytes_per_row}]",
                tensor.name(),
                shape
            );

            let offsets = entry["data_offsets"]
                .as_array()
                .with_context(|| format!("tensor {} has no data_offsets", tensor.name()))?;
            ensure!(offsets.len() == 2, "data_offsets must be a pair");
            let begin = offsets[0].as_u64().context("data_offset start")? as usize;
            let end = offsets[1].as_u64().context("data_offset end")? as usize;
            let span = end
                .checked_sub(begin)
                .context("CB3 data_offsets are not ascending")?;
            ensure!(
                span == experts * tensor.bytes_per_expert() as usize,
                "CB3 layer {layer} tensor {} spans {span} bytes, expected {} x {}",
                tensor.name(),
                experts,
                tensor.bytes_per_expert()
            );
            let absolute = data_start
                .checked_add(end)
                .context("CB3 tensor extent overflows")?;
            ensure!(
                absolute <= map.len(),
                "CB3 layer {layer} tensor {} runs past end of shard",
                tensor.name()
            );
            plane_offsets[index] = data_start + begin;
        }

        Ok(Self {
            map,
            plane_offsets,
            layer,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Borrow one expert's twelve packed planes, addressed by its **384-space routed id**.
    ///
    /// Resolution goes through [`ExpertPack::slot_of`], so an id that is not resident is a
    /// hard error here too — never a modulo, never a clamp. Callers on a routing path
    /// should consult `ExpertPack::routing_mask` before top-k so this cannot be reached.
    pub fn expert(&self, pack: &ExpertPack, expert_id: u32) -> Result<ExpertBytes<'_>> {
        let slot = pack.slot_of(self.layer, expert_id)?;
        self.expert_by_slot(pack, slot, expert_id)
    }

    fn expert_by_slot(
        &self,
        pack: &ExpertPack,
        slot: usize,
        expert_id: u32,
    ) -> Result<ExpertBytes<'_>> {
        let mut planes: [&[u8]; 12] = [&[]; 12];
        for (index, tensor) in CB3_TENSORS.iter().enumerate() {
            let stride = tensor.bytes_per_expert() as usize;
            // The row slice for THIS expert inside THIS tensor. There is no single offset
            // for "expert i" in the shard; there are twelve, and this is one of them.
            let begin = self.plane_offsets[index] + slot * stride;
            let end = begin + stride;
            if end > self.map.len() {
                bail!(
                    "CB3 layer {} tensor {} slot {slot} runs past end of shard",
                    self.layer,
                    tensor.name()
                );
            }
            planes[index] = &self.map[begin..end];
        }

        let bytes = ExpertBytes {
            planes,
            expert_id,
            slot,
        };
        // The twelve planes must account for exactly one expert, as the manifest declares.
        ensure!(
            bytes.total_bytes() as u64 == pack.bytes_per_expert(),
            "CB3 expert {expert_id} assembled {} bytes, manifest says {}",
            bytes.total_bytes(),
            pack.bytes_per_expert()
        );
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::SERVED_PACKED_KEEP;

    const PACK_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/k154-cb3";

    /// Load the shipped pack, or skip when the 290 GB checkpoint is absent.
    fn load_pack(keep: usize) -> Option<ExpertPack> {
        let manifest = std::fs::read_to_string(Path::new(PACK_DIR).join("manifest.json")).ok()?;
        Some(ExpertPack::parse(&manifest, keep).expect("shipped manifest must parse"))
    }

    /// Reads REAL CB3 bytes out of a REAL shard. This is the first point in the Rust port
    /// where checkpoint expert payload is actually touched.
    #[test]
    fn reads_real_cb3_expert_bytes_from_the_shipped_shard() {
        let Some(pack) = load_pack(SERVED_PACKED_KEEP) else {
            eprintln!("skipping: {PACK_DIR} not present");
            return;
        };
        let shard = LayerShard::open(Path::new(PACK_DIR), 0, &pack).expect("layer 0 must map");

        let resident = pack.resident_ids(0).unwrap().to_vec();
        let first = resident[0];
        let bytes = shard
            .expert(&pack, first)
            .expect("resident expert must resolve");

        assert_eq!(bytes.slot(), 0, "first resident id occupies slot 0");
        assert_eq!(bytes.expert_id(), first);
        assert_eq!(bytes.total_bytes() as u64, pack.bytes_per_expert());

        // Every plane has exactly its declared per-expert stride.
        for tensor in CB3_TENSORS {
            assert_eq!(
                bytes.plane(tensor).len() as u64,
                tensor.bytes_per_expert(),
                "plane {} wrong length",
                tensor.name()
            );
        }

        // The payload must not be trivially empty -- a wrong offset into a sparse region
        // could yield all zeros and still have the right LENGTH, so check content too.
        let w1_lo = bytes.plane(Cb3Tensor::W1Lo);
        assert!(
            w1_lo.iter().any(|byte| *byte != 0),
            "w1_lo payload is entirely zero, which suggests a bad offset"
        );
    }

    /// Distinct experts must yield DISTINCT bytes.
    ///
    /// This is the test that would catch a stride error that preserved lengths: if the
    /// per-expert stride were wrong, neighbouring slots would overlap and could compare
    /// equal. Length checks alone cannot see that.
    #[test]
    fn distinct_slots_yield_distinct_payloads() {
        let Some(pack) = load_pack(SERVED_PACKED_KEEP) else {
            eprintln!("skipping: {PACK_DIR} not present");
            return;
        };
        let shard = LayerShard::open(Path::new(PACK_DIR), 0, &pack).expect("layer 0 must map");
        let resident = pack.resident_ids(0).unwrap().to_vec();

        let a = shard.expert(&pack, resident[0]).unwrap();
        let b = shard.expert(&pack, resident[1]).unwrap();
        assert_ne!(
            a.plane(Cb3Tensor::W1Lo),
            b.plane(Cb3Tensor::W1Lo),
            "adjacent experts share bytes; the per-expert stride is wrong"
        );

        // And the last resident slot is reachable -- an off-by-one in the stride would run
        // past the tensor and fail here rather than silently aliasing.
        let last = *resident.last().unwrap();
        let tail = shard
            .expert(&pack, last)
            .expect("last resident expert must resolve");
        assert_eq!(tail.slot(), SERVED_PACKED_KEEP - 1);
    }

    /// A non-resident id is a hard error at the READER too, not only in the mapping.
    #[test]
    fn non_resident_expert_is_refused_by_the_reader() {
        let Some(pack) = load_pack(SERVED_PACKED_KEEP) else {
            eprintln!("skipping: {PACK_DIR} not present");
            return;
        };
        let shard = LayerShard::open(Path::new(PACK_DIR), 0, &pack).expect("layer 0 must map");

        let resident = pack.resident_ids(0).unwrap().to_vec();
        let absent = (0..384u32)
            .find(|id| !resident.contains(id))
            .expect("some id is outside a 124-of-384 selection");
        assert!(shard.expert(&pack, absent).is_err());

        // An id in the PACK but outside packed_keep is also refused -- the bytes exist on
        // disk, which is exactly why this must not silently succeed.
        let full = load_pack(atlas_core::config::PACK_EXPERTS).unwrap();
        let in_pack_not_served = full.resident_ids(0).unwrap()[SERVED_PACKED_KEEP];
        assert!(shard.expert(&pack, in_pack_not_served).is_err());
    }

    /// The per-layer mapping is real: the SAME routed id lands on different slots in
    /// different layers, and a layer that does not hold it refuses.
    #[test]
    fn the_same_expert_id_maps_differently_per_layer() {
        let Some(pack) = load_pack(atlas_core::config::PACK_EXPERTS) else {
            eprintln!("skipping: {PACK_DIR} not present");
            return;
        };
        let mut differing = 0;
        for layer in 1..pack.num_layers() {
            let id = pack.resident_ids(0).unwrap()[0];
            match (pack.slot_of(0, id), pack.slot_of(layer, id)) {
                (Ok(a), Ok(b)) if a != b => differing += 1,
                (Ok(_), Err(_)) => differing += 1,
                _ => {}
            }
        }
        assert!(
            differing > 0,
            "a routed id must not occupy the same slot in every layer"
        );
    }
}
