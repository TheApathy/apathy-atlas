// SPDX-License-Identifier: AGPL-3.0-only

//! One-shot, borrowed observer for the exact Vision L0 shared/routed chain.

use super::{DevicePtr, ForwardContext, Fp8ExpertWeight, GpuBackend, KernelHandle, MoeLayer};
use crate::layers::vision_capture_files::{create_root, open_new, validate_path, write_new};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, ffi::OsStr, fs::File, io::Write};

const IDS: [u32; 12] = [
    0, 128803, 19905, 418, 9045, 20370, 305, 5760, 3006, 16, 128804, 128822,
];
const PAYLOAD_BYTES: usize = 25_909_296;
const MAX_BYTES: usize = 32 * 1024 * 1024;
const CHUNK: usize = 512 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Stage {
    FfnInput,
    SharedInput,
    W1,
    W1Scale,
    W3,
    W3Scale,
    W2,
    W2Scale,
    SharedGate,
    SharedUp,
    SharedActivation,
    SharedDown,
    SharedAfterRouted,
    RoutedOnly,
    MoeBlended,
}
const STAGES: [Stage; 15] = [
    Stage::FfnInput,
    Stage::SharedInput,
    Stage::W1,
    Stage::W1Scale,
    Stage::W3,
    Stage::W3Scale,
    Stage::W2,
    Stage::W2Scale,
    Stage::SharedGate,
    Stage::SharedUp,
    Stage::SharedActivation,
    Stage::SharedDown,
    Stage::SharedAfterRouted,
    Stage::RoutedOnly,
    Stage::MoeBlended,
];
impl Stage {
    fn spec(self) -> (&'static str, &'static str, &'static [usize], usize) {
        let name = [
            "ffn_input",
            "shared_input",
            "shared_w1_fp8",
            "shared_w1_scale_f32",
            "shared_w3_fp8",
            "shared_w3_scale_f32",
            "shared_w2_fp8",
            "shared_w2_scale_f32",
            "shared_gate",
            "shared_up",
            "shared_activation",
            "shared_down",
            "shared_after_routed",
            "routed_only",
            "moe_blended",
        ][self as usize];
        let (dtype, shape, width): (_, &[usize], _) = match self {
            Self::W1 | Self::W3 => ("F8_E4M3", &[2048, 4096], 1),
            Self::W2 => ("F8_E4M3", &[4096, 2048], 1),
            Self::W1Scale | Self::W3Scale => ("F32", &[16, 32], 4),
            Self::W2Scale => ("F32", &[32, 16], 4),
            Self::SharedGate | Self::SharedUp | Self::SharedActivation => ("BF16", &[12, 2048], 2),
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
    geometry: bool,
}
impl Admission {
    fn validate(self) -> Result<()> {
        ensure!(
            self.vision && self.first && self.c1 && self.eager && self.geometry,
            "ATLAS_VISION_MOE_L0_DUMP requires actual Vision, initial uncached exact12 text rows, C1 eager H4096/I2048/HC4/vocab129280"
        );
        Ok(())
    }
}

pub(crate) struct MoeCapture {
    root: File,
    tensors: BTreeMap<String, Value>,
    next: usize,
    total: usize,
    dispatch: Option<Value>,
}
impl MoeCapture {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin(
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
        let Some(path) = std::env::var_os("ATLAS_VISION_MOE_L0_DUMP") else {
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
                && ctx.attn_metadata.is_none_or(|m| m.num_seqs == 1),
            eager: !ctx.graph_capture,
            geometry: rows == 12
                && c.hidden_size == 4096
                && c.shared_expert_intermediate_size == 2048
                && c.moe_intermediate_size == 2048
                && c.hc_mult == 4
                && c.vocab_size == 129280
                && c.num_experts == 256
                && c.num_experts_per_tok == 6,
        }
        .validate()?;
        validate_path(&path)?;
        let ptr = ctx
            .token_ids
            .filter(|p| !p.is_null())
            .context("MoE capture requires stable IDs")?;
        ctx.gpu.synchronize(stream)?;
        let mut bytes = [0; 48];
        ctx.gpu.copy_d2h(ptr, &mut bytes)?;
        Self::start(&path, &bytes).map(Some)
    }

    fn start(path: &OsStr, ids: &[u8]) -> Result<Self> {
        let expected: Vec<_> = IDS.iter().flat_map(|id| id.to_le_bytes()).collect();
        ensure!(
            ids == expected,
            "MoE capture requires the pinned exact12 text IDs"
        );
        let root = create_root(path)?;
        write_new(&root, "token_ids.bin", ids)?;
        Ok(Self {
            root,
            tensors: BTreeMap::new(),
            next: 0,
            total: 48,
            dispatch: None,
        })
    }

    pub(crate) fn validate_moe(&self, moe: &MoeLayer) -> Result<()> {
        let exl3 = moe.exl3.as_ref().context("MoE capture requires EXL3")?;
        let p = &exl3.prefill;
        let unfused_p1 = !p.direct
            && !p.direct_m128
            && !p.direct_k64
            && !p.direct_n128
            && !p.direct_n256
            && !p.persistent
            && !p.fused_post
            && !p.dual_pre
            && !p.w2a8_requested
            && !p.w2a8_fused_gu_down_requested
            && !p.w2a8_fused_gu_down_n256_requested
            && !p.w2a8_n256_down_requested
            && !p.fused_unpermute
            && !p.fused_blend
            && !p.fused_blend_requested;
        validate_mode(
            unfused_p1,
            moe.native_shared_fp8.is_some(),
            moe.weights.shared_expert_gate.weight.is_null(),
            moe.bf16_gate_weight_ptrs.is_none()
                && moe.fp8_gate_weight_ptrs.is_none()
                && moe.pre_expert_norm.is_none(),
        )
    }

    pub(crate) fn native_weights(
        &mut self,
        weights: Fp8ExpertWeight,
        handles: [KernelHandle; 3],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.dispatch.is_none() && handles.iter().all(|h| h.0 != 0),
            "invalid/repeated native receipt"
        );
        for (weight, n, k, stage, scale) in [
            (weights.gate_proj, 2048, 4096, Stage::W1, Stage::W1Scale),
            (weights.up_proj, 2048, 4096, Stage::W3, Stage::W3Scale),
            (weights.down_proj, 4096, 2048, Stage::W2, Stage::W2Scale),
        ] {
            ensure!(
                weight.n == n
                    && weight.k == k
                    && weight.scale_format == crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "MoE capture native weight geometry/format mismatch"
            );
            self.stage(stage, gpu, weight.weight, stream)?;
            self.stage(scale, gpu, weight.row_scale, stream)?;
        }
        self.dispatch = Some(
            json!({"rows":12,"hidden_size":4096,"intermediate_size":2048,
            "routed_mode":"EXL3_P1_UNFUSED","shared_expert_gate":"0x0",
            "weight_format":"F8_E4M3","scale_format":"F32_BLOCK_128x128",
            "activation":"silu_gate_max10_up_clamp10","stream":format!("{stream:#x}"),
            "kernels":{
                "gemv":{"module":"w8a16_gemv","entry":"w8a16_gemv","handle":format!("{:#x}",handles[0].0)},
                "gemm":{"module":"w8a16_gemm","entry":"w8a16_gemm","handle":format!("{:#x}",handles[1].0)},
                "activation":{"module":"moe_silu_mul","entry":"moe_silu_mul","handle":format!("{:#x}",handles[2].0)}}}),
        );
        Ok(())
    }

    pub(crate) fn stage(
        &mut self,
        stage: Stage,
        gpu: &dyn GpuBackend,
        ptr: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ensure!(!ptr.is_null(), "MoE capture rejects null device pointer");
        gpu.synchronize(stream)?;
        self.write_stage(stage, ptr, |offset, bytes| {
            gpu.copy_d2h(ptr.offset(offset), bytes)
        })
    }

    fn write_stage(
        &mut self,
        stage: Stage,
        ptr: DevicePtr,
        mut read: impl FnMut(usize, &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            STAGES.get(self.next) == Some(&stage),
            "MoE capture stage repeated or out of order"
        );
        let (name, dtype, shape, bytes) = stage.spec();
        ensure!(
            self.total
                .checked_add(bytes)
                .is_some_and(|n| n <= PAYLOAD_BYTES),
            "MoE capture payload budget exceeded"
        );
        let file = format!("{name}.bin");
        let mut out = open_new(&self.root, &file)?;
        let mut buffer = vec![0; bytes.min(CHUNK)];
        let mut offset = 0;
        while offset < bytes {
            let count = buffer.len().min(bytes - offset);
            read(offset, &mut buffer[..count])?;
            out.write_all(&buffer[..count])?;
            offset += count;
        }
        out.sync_all()?;
        self.tensors.insert(
            name.into(),
            json!({"file":file,"dtype":dtype,"shape":shape,
            "bytes":bytes,"device_ptr":format!("{:#x}",ptr.0)}),
        );
        self.total += bytes;
        self.next += 1;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<()> {
        ensure!(
            self.next == STAGES.len()
                && self.tensors.len() == STAGES.len()
                && self.total == PAYLOAD_BYTES
                && self.dispatch.is_some(),
            "incomplete MoE capture"
        );
        let bytes = serde_json::to_vec_pretty(&json!({"schema":"atlas-vision-moe-l0-dump-v1",
            "status":"COMPLETE","layer_index":0,"token_count":12,"hidden_size":4096,
            "intermediate_size":2048,"hc_mult":4,"vocab_size":129280,"token_ids":IDS,
            "token_ids_file":"token_ids.bin","token_ids_bytes":48,"byte_order":"little",
            "payload_bytes":self.total,"tensors":self.tensors,"native_dispatch":self.dispatch}))?;
        ensure!(
            bytes.len() <= 64 * 1024 && self.total + bytes.len() <= MAX_BYTES,
            "MoE capture manifest budget exceeded"
        );
        write_new(&self.root, "manifest.json", &bytes)?;
        self.root.sync_all()?;
        Ok(())
    }
}

fn validate_mode(p1: bool, native: bool, null_gate: bool, plain_routed: bool) -> Result<()> {
    ensure!(
        p1 && native && null_gate && plain_routed,
        "MoE capture requires EXL3 P1 unfused, native FP8 shared, null shared gate and no alternate routed path"
    );
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
#[path = "../../../tests/vision_moe_l0_dump/unit.rs"]
mod tests;
