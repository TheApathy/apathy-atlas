// SPDX-License-Identifier: AGPL-3.0-only

//! One-shot, typed capture for the pinned Vision decoder's 12-row text L0.

use crate::layers::vision_capture_files::{create_root, validate_path, write_new};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
#[cfg(test)]
use std::path::{Path, PathBuf};
use std::{collections::BTreeMap, ffi::OsStr, fs::File};

use crate::layer::ForwardContext;

const MAX_BYTES: usize = 8 * 1024 * 1024;
const VOCAB: u32 = 129_280;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum Stage {
    Embed,
    HcExpanded,
    HcPreAttn,
    PostAttn,
    CombAttn,
    NormAttn,
    AttentionOut,
    HcPostAttn,
    HcPreFfn,
    PostFfn,
    CombFfn,
    NormFfn,
    MoeOut,
    HcPostFfn,
}
const STAGES: [Stage; 14] = [
    Stage::Embed,
    Stage::HcExpanded,
    Stage::HcPreAttn,
    Stage::PostAttn,
    Stage::CombAttn,
    Stage::NormAttn,
    Stage::AttentionOut,
    Stage::HcPostAttn,
    Stage::HcPreFfn,
    Stage::PostFfn,
    Stage::CombFfn,
    Stage::NormFfn,
    Stage::MoeOut,
    Stage::HcPostFfn,
];

impl Stage {
    fn spec(self) -> (&'static str, &'static str, &'static [usize], usize) {
        let name = [
            "embed",
            "hc_expanded",
            "hc_pre_attn",
            "post_attn",
            "comb_attn",
            "norm_attn",
            "attention_out",
            "hc_post_attn",
            "hc_pre_ffn",
            "post_ffn",
            "comb_ffn",
            "norm_ffn",
            "moe_out",
            "hc_post_ffn",
        ][self as usize];
        let (dtype, shape, width): (_, &[usize], _) = match self {
            Self::HcExpanded | Self::HcPostAttn | Self::HcPostFfn => ("F32", &[12, 4, 4096], 4),
            Self::PostAttn | Self::PostFfn => ("F32", &[12, 4], 4),
            Self::CombAttn | Self::CombFfn => ("F32", &[12, 4, 4], 4),
            _ => ("BF16", &[12, 4096], 2),
        };
        (name, dtype, shape, shape.iter().product::<usize>() * width)
    }
}

#[derive(Clone, Copy)]
struct Admission {
    vision: bool,
    first: bool,
    c1: bool,
    eager: bool,
    rows: usize,
    geometry: bool,
}
impl Admission {
    fn validate(self) -> Result<()> {
        ensure!(
            self.vision && self.first && self.c1 && self.eager && self.rows == 12 && self.geometry,
            "ATLAS_VISION_L0_DUMP requires actual Vision, text-only C1, initial uncached 12-row L0 with H4096/HC4/vocab129280"
        );
        Ok(())
    }
}

pub(super) struct Capture {
    root: File,
    ids: Vec<u32>,
    tensors: BTreeMap<String, Value>,
    next: usize,
    total: usize,
    numerics: [f64; 3],
}

impl Capture {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin(
        layer: usize,
        rows: usize,
        start: usize,
        write_start: usize,
        batched: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<Self>> {
        if layer != 0 {
            return Ok(None);
        }
        let Some(path) = std::env::var_os("ATLAS_VISION_L0_DUMP") else {
            return Ok(None);
        };
        let c = ctx.config;
        Admission {
            vision: c.deepseek_vision.is_some(),
            first: start == 0 && write_start == 0,
            c1: !batched
                && c.ep_world_size <= 1
                && c.tp_world_size <= 1
                && ctx.comm.is_none()
                && ctx.attn_metadata.is_none_or(|meta| meta.num_seqs == 1),
            eager: !ctx.graph_capture,
            rows,
            geometry: c.hidden_size == 4096 && c.hc_mult == 4 && c.vocab_size == VOCAB as usize,
        }
        .validate()?;
        validate_path(&path)?;
        let ids = ctx
            .token_ids
            .filter(|ptr| !ptr.is_null())
            .context("Vision L0 capture requires stable token IDs")?;
        let bytes = read_gpu(ctx.gpu, ids, 48, stream)?;
        let ids = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Self::start(
            &path,
            ids,
            [c.rms_norm_eps, c.hc_eps as f64, c.hc_sinkhorn_iters as f64],
        )
        .map(Some)
    }

    fn start(path: &OsStr, ids: Vec<u32>, numerics: [f64; 3]) -> Result<Self> {
        ensure!(
            ids.len() == 12 && ids.iter().all(|&id| id < VOCAB),
            "Vision L0 capture requires exactly 12 text token IDs"
        );
        ensure!(
            numerics.iter().all(|n| n.is_finite() && *n > 0.0),
            "invalid Vision L0 numerics"
        );
        let capture = Self {
            root: create_root(path)?,
            ids,
            tensors: BTreeMap::new(),
            next: 0,
            total: 48,
            numerics,
        };
        let bytes: Vec<_> = capture.ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        capture.write_new("token_ids.bin", &bytes)?;
        Ok(capture)
    }

    fn write_new(&self, name: &str, bytes: &[u8]) -> Result<()> {
        write_new(&self.root, name, bytes)
    }

    pub(super) fn stage(
        &mut self,
        stage: Stage,
        gpu: &dyn GpuBackend,
        ptr: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            STAGES.get(self.next) == Some(&stage),
            "Vision L0 capture stage repeated or out of order"
        );
        let (name, dtype, shape, bytes) = stage.spec();
        ensure!(
            bytes <= 768 * 1024
                && self
                    .total
                    .checked_add(bytes)
                    .is_some_and(|n| n <= MAX_BYTES),
            "Vision L0 capture byte budget exceeded"
        );
        let data = read_gpu(gpu, ptr, bytes, stream)?;
        let file = format!("{name}.bin");
        self.write_new(&file, &data)?;
        self.tensors.insert(
            name.into(),
            json!({"file":file,"dtype":dtype,"shape":shape,"bytes":bytes}),
        );
        self.total += bytes;
        self.next += 1;
        Ok(())
    }

    pub(super) fn finish(self) -> Result<()> {
        ensure!(
            self.next == STAGES.len() && self.tensors.len() == STAGES.len(),
            "incomplete Vision L0 capture"
        );
        let manifest = json!({"schema":"atlas-vision-l0-dump-v1","status":"COMPLETE",
            "layer_index":0,"token_count":12,"hidden_size":4096,"hc_mult":4,"vocab_size":VOCAB,
            "token_ids":self.ids,"token_ids_file":"token_ids.bin","token_ids_bytes":48,
            "native_rms_norm_eps":self.numerics[0],"native_hc_eps":self.numerics[1],
            "native_sinkhorn_iters":self.numerics[2] as usize,"byte_order":"little",
            "payload_bytes":self.total,"tensors":self.tensors});
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        ensure!(
            self.total + bytes.len() <= MAX_BYTES,
            "Vision L0 manifest exceeds total byte budget"
        );
        self.write_new("manifest.json", &bytes)?;
        self.root.sync_all()?;
        Ok(())
    }
}

fn read_gpu(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize, stream: u64) -> Result<Vec<u8>> {
    ensure!(
        !ptr.is_null() && bytes > 0 && bytes <= 768 * 1024,
        "invalid Vision L0 capture extent/pointer"
    );
    gpu.synchronize(stream)?;
    let mut data = vec![0; bytes];
    gpu.copy_d2h(ptr, &mut data)?;
    Ok(data)
}

#[cfg(all(test, target_os = "linux"))]
#[path = "../../../../tests/vision_l0_dump/unit.rs"]
mod tests;
