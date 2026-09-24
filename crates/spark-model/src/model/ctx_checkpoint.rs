// SPDX-License-Identifier: AGPL-3.0-only

//! Context checkpoint capture/restore for hybrid (GDN + attention) models.
//!
//! State of a sequence after prefilling `C` prompt tokens, as sections:
//! - `kv.{l}.k` / `kv.{l}.v`: the first `ceil(C / block_size)` paged blocks
//!   of attention layer `l`, in logical order (block ids are reassigned on
//!   restore). Rows past `C` in the last block are never read.
//! - `ssm.{j}.h` / `ssm.{j}.conv`: recurrent and conv state of GDN layer `j`
//!   (h always FP32, widened like the disk-swap path).
//! - `dflash.ring`: the resident DFlash target-hidden rows, chronological.
//!   Correctness does not depend on it (the target verifies every token),
//!   but without it the drafter refuses to propose until it refills.
//!
//! Restore targets a freshly allocated sequence and leaves it exactly as a
//! prefill of the same `C` tokens would: the next `prefill_chunk` starts at
//! `chunk_start = C`, the same as the second chunk of a split prefill.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::LayerType;
use spark_runtime::ctx_store::{AlignedBuf, CtxSection, CtxSnapshot};
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::layers::DflashProposerState;
use crate::traits::SequenceState;

/// Consecutive runs `(logical_start, physical_start, len)` of a block table.
fn block_runs(table: &[u32]) -> Vec<(usize, u32, usize)> {
    let mut runs: Vec<(usize, u32, usize)> = Vec::new();
    for (i, &b) in table.iter().enumerate() {
        match runs.last_mut() {
            Some((_, p, n)) if *p as usize + *n == b as usize => *n += 1,
            _ => runs.push((i, b, 1)),
        }
    }
    runs
}

/// Ring spans `(row, slot, rows)` covering absolute positions `[start, end)`
/// of a ring with `capacity` slots: at most two, split at the wrap.
fn ring_spans(start: usize, end: usize, capacity: usize) -> Vec<(usize, usize, usize)> {
    let mut spans = Vec::new();
    let mut p = start;
    while p < end {
        let slot = p % capacity;
        let rows = (end - p).min(capacity - slot);
        spans.push((p - start, slot, rows));
        p += rows;
    }
    spans
}

fn dflash_state(seq: &SequenceState) -> Option<&DflashProposerState> {
    seq.proposer_state
        .as_deref()
        .and_then(|p| p.as_any().downcast_ref::<DflashProposerState>())
}

/// Sections a gate control deliberately leaves unrestored
/// (`ATLAS_CTX_GATE_SKIP=<substring>`). Never set in production.
fn gate_skip() -> Option<String> {
    std::env::var("ATLAS_CTX_GATE_SKIP").ok().filter(|s| !s.is_empty())
}

impl TransformerModel {
    /// Why this model/sequence cannot be checkpointed, or `None` if it can.
    fn ctx_unsupported(&self, seq: Option<&SequenceState>) -> Option<&'static str> {
        if self.comm.is_some() {
            return Some("expert/tensor parallel");
        }
        if self.config.is_qwen4_exp() {
            return Some("qwen4 QSA side-cache is not captured yet");
        }
        if crate::traits::Model::is_mla(self) {
            return Some("MLA prefill cannot continue from a chunk boundary");
        }
        if (0..self.config.num_hidden_layers).any(|i| {
            !matches!(
                self.config.layer_type(i),
                LayerType::FullAttention | LayerType::LinearAttention
            )
        }) {
            return Some("layer kinds other than attention/GDN");
        }
        if self.kv_cache.lock().config().cache_blocks_per_seq.is_some() {
            return Some("high-speed-swap sliding KV");
        }
        let seq = seq?;
        if !seq.rotary_positions.is_identity() || self.tokens_have_vision_pad(&seq.tokens) {
            return Some("vision prompt");
        }
        if seq.proposer_state_alt.is_some() {
            return Some("two proposer arms");
        }
        if seq.proposer_state.is_some() && dflash_state(seq).is_none() {
            return Some("non-DFlash proposer state (MTP KV is not captured yet)");
        }
        None
    }

    /// Geometry folded into the checkpoint key; `None` when unsupported.
    pub(crate) fn ctx_geometry_dispatch(&self) -> Option<String> {
        if let Some(why) = self.ctx_unsupported(None) {
            tracing::info!("ctx-cache: unsupported for this model ({why})");
            return None;
        }
        let kv = self.kv_cache.lock();
        let strides: Vec<String> = (0..kv.num_layers())
            .map(|l| {
                format!(
                    "{:?}/{}/{}",
                    kv.dtype_for_layer(l),
                    kv.k_block_stride_bytes_for_layer(l),
                    kv.v_block_stride_bytes_for_layer(l)
                )
            })
            .collect();
        Some(format!(
            "arch={} layers={} bs={} kv=[{}] ssm={}x{}+{} capture={:?}",
            self.config.model_type,
            self.config.num_hidden_layers,
            kv.block_size(),
            strides.join(","),
            self.ssm_pool.num_ssm_layers,
            self.ssm_pool.h_bytes,
            self.ssm_pool.conv_bytes,
            self.dflash_capture_layers,
        ))
    }

    fn ssm_states(seq: &SequenceState) -> Vec<&SsmLayerState> {
        seq.layer_states
            .iter()
            .filter_map(|s| s.as_any().downcast_ref::<SsmLayerState>())
            .collect()
    }

    pub(crate) fn ctx_capture_dispatch(
        &self,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<Option<CtxSnapshot>> {
        if let Some(why) = self.ctx_unsupported(Some(seq)) {
            tracing::debug!("ctx-cache: capture skipped ({why})");
            return Ok(None);
        }
        let c = seq.seq_len;
        ensure!(
            c > 0 && seq.tokens.len() == c,
            "ctx capture: seq_len {c} != tokens {}",
            seq.tokens.len()
        );
        let gpu = self.gpu.as_ref();
        gpu.synchronize(stream)?;
        let mut sections = Vec::new();
        let kv = self.kv_cache.lock();
        let bs = kv.block_size();
        let nb = c.div_ceil(bs);
        ensure!(
            seq.block_table.len() >= nb,
            "ctx capture: {} blocks for {c} tokens",
            seq.block_table.len()
        );
        let runs = block_runs(&seq.block_table[..nb]);
        for l in 0..kv.num_layers() {
            for (side, stride) in [
                ("k", kv.k_block_stride_bytes_for_layer(l)),
                ("v", kv.v_block_stride_bytes_for_layer(l)),
            ] {
                let mut buf = AlignedBuf::zeroed(nb * stride)?;
                for &(logical, phys, n) in &runs {
                    let src = if side == "k" {
                        kv.k_cache_ptr(l, phys)
                    } else {
                        kv.v_cache_ptr(l, phys)
                    };
                    let dst = &mut buf.as_mut_slice()[logical * stride..(logical + n) * stride];
                    gpu.copy_d2h(src, dst)?;
                }
                sections.push(CtxSection { name: format!("kv.{l}.{side}"), data: buf });
            }
        }
        drop(kv);
        let ssm = Self::ssm_states(seq);
        ensure!(ssm.len() == self.ssm_pool.num_ssm_layers, "ctx capture: SSM layer count");
        for (j, s) in ssm.iter().enumerate() {
            let mut h = AlignedBuf::zeroed(self.ssm_pool.h_bytes)?;
            let src = if s.h_is_f16 {
                self.widen_h_to_f32_scratch(gpu, s.h_state)?
            } else {
                s.h_state
            };
            gpu.copy_d2h(src, h.as_mut_slice())?;
            let mut conv = AlignedBuf::zeroed(self.ssm_pool.conv_bytes)?;
            gpu.copy_d2h(s.conv_state, conv.as_mut_slice())?;
            sections.push(CtxSection { name: format!("ssm.{j}.h"), data: h });
            sections.push(CtxSection { name: format!("ssm.{j}.conv"), data: conv });
        }
        let mut meta = vec![
            ("c".to_string(), c as u64),
            ("block_size".to_string(), bs as u64),
            ("blocks".to_string(), nb as u64),
            ("ssm_layers".to_string(), ssm.len() as u64),
        ];
        if let Some(d) = dflash_state(seq) {
            ensure!(
                d.ctx_len == c,
                "ctx capture: DFlash ring cursor {} != seq_len {c}",
                d.ctx_len
            );
            let resident = d.ctx_resident_len;
            let sb = d.ctx_slot_bytes;
            let mut ring = AlignedBuf::zeroed(resident * sb)?;
            for (row, slot, rows) in ring_spans(c - resident, c, d.ctx_capacity) {
                let dst = &mut ring.as_mut_slice()[row * sb..(row + rows) * sb];
                gpu.copy_d2h(d.ctx_hidden_acc.offset(slot * sb), dst)?;
            }
            meta.extend([
                ("dflash.resident".to_string(), resident as u64),
                ("dflash.capacity".to_string(), d.ctx_capacity as u64),
                ("dflash.slot_bytes".to_string(), d.ctx_slot_bytes as u64),
                ("dflash.max_ctx".to_string(), d.max_ctx_len as u64),
            ]);
            sections.push(CtxSection { name: "dflash.ring".to_string(), data: ring });
        }
        Ok(Some(CtxSnapshot { tokens: seq.tokens.clone(), meta, sections }))
    }

    pub(crate) fn ctx_restore_dispatch(
        &self,
        snap: &CtxSnapshot,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        if let Some(why) = self.ctx_unsupported(Some(seq)) {
            bail!("ctx restore unsupported: {why}");
        }
        ensure!(
            seq.seq_len == 0 && seq.block_table.is_empty() && seq.tokens.is_empty(),
            "ctx restore needs a freshly allocated sequence"
        );
        let c = usize::try_from(snap.meta_req("c")?)?;
        ensure!(c == snap.tokens.len() && c > 0, "ctx restore: bad token count");
        let skip = gate_skip();
        let skipped = |name: &str| skip.as_deref().is_some_and(|s| name.contains(s));
        let gpu = self.gpu.as_ref();

        // Validate every section's size against the live geometry BEFORE
        // touching device state, so a mismatch leaves the sequence fresh.
        let mut kv = self.kv_cache.lock();
        let bs = kv.block_size();
        let nb = c.div_ceil(bs);
        ensure!(
            snap.meta_req("block_size")? as usize == bs && snap.meta_req("blocks")? as usize == nb,
            "ctx restore: block geometry changed"
        );
        let mut kv_sections = Vec::with_capacity(kv.num_layers());
        for l in 0..kv.num_layers() {
            let k = snap.section_exact(&format!("kv.{l}.k"), nb * kv.k_block_stride_bytes_for_layer(l))?;
            let v = snap.section_exact(&format!("kv.{l}.v"), nb * kv.v_block_stride_bytes_for_layer(l))?;
            kv_sections.push((k, v));
        }
        ensure!(
            snap.section(&format!("kv.{}.k", kv.num_layers())).is_none(),
            "ctx restore: checkpoint has more attention layers"
        );
        let n_ssm = self.ssm_pool.num_ssm_layers;
        ensure!(snap.meta_req("ssm_layers")? as usize == n_ssm, "ctx restore: SSM layer count");
        let mut ssm_sections = Vec::with_capacity(n_ssm);
        for j in 0..n_ssm {
            ssm_sections.push((
                snap.section_exact(&format!("ssm.{j}.h"), self.ssm_pool.h_bytes)?,
                snap.section_exact(&format!("ssm.{j}.conv"), self.ssm_pool.conv_bytes)?,
            ));
        }
        let ring = match dflash_state(seq) {
            Some(d) => {
                let resident = snap.meta("dflash.resident").context(
                    "ctx restore: checkpoint has no DFlash ring but this server drafts with DFlash",
                )? as usize;
                ensure!(
                    snap.meta_req("dflash.capacity")? as usize == d.ctx_capacity
                        && snap.meta_req("dflash.slot_bytes")? as usize == d.ctx_slot_bytes
                        && snap.meta_req("dflash.max_ctx")? as usize == d.max_ctx_len
                        && resident == c.min(d.ctx_capacity),
                    "ctx restore: DFlash ring geometry changed"
                );
                Some((
                    resident,
                    snap.section_exact("dflash.ring", resident * d.ctx_slot_bytes)?,
                ))
            }
            None => None,
        };

        // Allocate blocks (with prefix-cache eviction fallback), then upload.
        super::block_mgmt::ensure_blocks_through_prefill(
            seq,
            nb - 1,
            &mut kv,
            self.prefix_cache.as_ref(),
            gpu,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let runs = block_runs(&seq.block_table[..nb]);
        for (l, (k, v)) in kv_sections.iter().enumerate() {
            for (side, data, stride) in [
                ("k", k, kv.k_block_stride_bytes_for_layer(l)),
                ("v", v, kv.v_block_stride_bytes_for_layer(l)),
            ] {
                if skipped(&format!("kv.{l}.{side}")) {
                    continue;
                }
                for &(logical, phys, n) in &runs {
                    let dst: DevicePtr = if side == "k" {
                        kv.k_cache_ptr(l, phys)
                    } else {
                        kv.v_cache_ptr(l, phys)
                    };
                    gpu.copy_h2d(&data.as_slice()[logical * stride..(logical + n) * stride], dst)?;
                }
            }
        }
        drop(kv);
        let ssm = Self::ssm_states(seq);
        ensure!(ssm.len() == n_ssm, "ctx restore: sequence SSM layer count");
        for (j, (s, (h, conv))) in ssm.iter().zip(&ssm_sections).enumerate() {
            ensure!(!s.h_is_f16, "ctx restore: fresh sequence holds an FP16 h-state");
            if !skipped(&format!("ssm.{j}.h")) {
                gpu.copy_h2d(h.as_slice(), s.h_state)?;
            }
            if !skipped(&format!("ssm.{j}.conv")) {
                gpu.copy_h2d(conv.as_slice(), s.conv_state)?;
            }
        }
        if let Some((resident, data)) = ring {
            let d = seq
                .proposer_state
                .as_deref_mut()
                .and_then(|p| p.as_any_mut().downcast_mut::<DflashProposerState>())
                .context("ctx restore: DFlash state vanished")?;
            if !skipped("dflash.ring") {
                let sb = d.ctx_slot_bytes;
                for (row, slot, rows) in ring_spans(c - resident, c, d.ctx_capacity) {
                    let src = &data.as_slice()[row * sb..(row + rows) * sb];
                    gpu.copy_h2d(src, d.ctx_hidden_acc.offset(slot * sb))?;
                }
            }
            let state = crate::layers::dflash_head::ring_window::RingState::from_lengths(
                d.max_ctx_len,
                d.ctx_capacity,
                c,
                resident,
            )?;
            d.apply_ctx_ring_state(state)?;
        }
        gpu.synchronize(stream)?;
        if let Some(s) = skip {
            tracing::warn!(
                "ctx-cache: GATE CONTROL — sections matching '{s}' were NOT restored; \
                 this output is deliberately wrong"
            );
        }
        seq.tokens = snap.tokens.clone();
        seq.seq_len = c;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{block_runs, ring_spans};

    #[test]
    fn ring_spans_split_only_at_the_wrap() {
        assert_eq!(ring_spans(0, 100, 4096), vec![(0, 0, 100)]);
        assert_eq!(ring_spans(4000, 8096, 4096), vec![(0, 4000, 96), (96, 0, 4000)]);
        assert_eq!(ring_spans(8192, 12288, 4096), vec![(0, 0, 4096)]);
        assert!(ring_spans(5, 5, 4096).is_empty());
    }

    #[test]
    fn runs_coalesce_consecutive_physical_blocks() {
        assert_eq!(
            block_runs(&[4, 5, 6, 9, 10, 2]),
            vec![(0, 4, 3), (3, 9, 2), (5, 2, 1)]
        );
        assert!(block_runs(&[]).is_empty());
        // Descending ids never coalesce.
        assert_eq!(block_runs(&[3, 2]), vec![(0, 3, 1), (1, 2, 1)]);
    }
}
