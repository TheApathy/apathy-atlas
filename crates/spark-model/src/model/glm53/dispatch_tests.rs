// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use anyhow::Result;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;

const TAIL_CAP: usize = spark_runtime::kv_cache::GLM53_DSA_TAIL_CAPACITY as usize;
use crate::layers::{Glm53TargetFfnKind, Glm53TargetGeometry, Glm53TargetSchedule};
use crate::model::glm53::arena::Glm53ArenaPlan;
use crate::model::glm53::t1_state_transaction::Glm53T1StateLayout;
use crate::weight_loader::{
    GLM53_MAX_CONTEXT_TOKENS, Glm53AttentionWeights, Glm53ContextPlan, Glm53DsaStorage,
    Glm53DsaWeights, Glm53FfnWeights, Glm53GgufF32, Glm53GgufMatrix, Glm53GgufMatrixBank,
    Glm53KdaWeights, Glm53MoeWeights,
};
use spark_runtime::weights::gguf::{GgmlType, GgufDeviceTensor};

#[test]
fn exl3_last_row_head_selector_is_fail_closed_and_scope_bound() {
    let last = glm53_exl3_lm_head_slice(1_875, true, None).unwrap();
    assert_eq!(last.rows, 1);
    assert_eq!(last.input_row, 1_874);
    assert_eq!(last.output_row, 1_874);

    let full = glm53_exl3_lm_head_slice(1_875, true, Some("0")).unwrap();
    assert_eq!(full.rows, 1_875);
    assert_eq!(full.input_row, 0);
    assert_eq!(full.output_row, 0);

    assert_eq!(
        glm53_exl3_lm_head_slice(1_875, false, Some("1")).unwrap(),
        full
    );
    assert_eq!(
        glm53_exl3_lm_head_slice(1_875, true, Some("1")).unwrap(),
        last
    );

    let singleton = glm53_exl3_lm_head_slice(1, true, Some("1")).unwrap();
    assert_eq!(singleton.rows, 1);
    assert_eq!(singleton.input_row, 0);
    assert_eq!(singleton.output_row, 0);

    assert!(glm53_exl3_lm_head_slice(1_875, true, Some("yes")).is_err());
    assert!(glm53_exl3_lm_head_slice(0, true, Some("1")).is_err());
}

/// Records which kernel *symbol* ran, in order, so a test can assert the
/// composed sequence rather than a launch count. Distinct handles per symbol
/// are the whole point: with one shared handle an out-of-order composition
/// still looks correct.
struct RecordingGpu {
    inner: MockGpuBackend,
    symbols: Mutex<HashMap<u64, String>>,
    next: Mutex<u64>,
    launched: Mutex<Vec<String>>,
}

impl RecordingGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            symbols: Mutex::new(HashMap::new()),
            next: Mutex::new(1),
            launched: Mutex::new(Vec::new()),
        }
    }
    fn sequence(&self) -> Vec<String> {
        self.launched.lock().unwrap().clone()
    }
    fn launch_count(&self) -> usize {
        self.launched.lock().unwrap().len()
    }
}

impl GpuBackend for RecordingGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.inner.free(ptr)
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(src, dst)
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(src, dst)
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn launch(
        &self,
        func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        let name = self
            .symbols
            .lock()
            .unwrap()
            .get(&func.0)
            .cloned()
            .unwrap_or_else(|| format!("unknown:{}", func.0));
        self.launched.lock().unwrap().push(name);
        Ok(())
    }
    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, _module: &str, name: &str) -> Result<KernelHandle> {
        let mut symbols = self.symbols.lock().unwrap();
        if let Some((handle, _)) = symbols.iter().find(|(_, known)| known.as_str() == name) {
            return Ok(KernelHandle(*handle));
        }
        let mut next = self.next.lock().unwrap();
        let handle = *next;
        *next += 1;
        symbols.insert(handle, name.to_string());
        Ok(KernelHandle(handle))
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
}

const BASE: u64 = 0x1_0000_0000;
const MHC_BASE: u64 = 0x8_0000_0000;
const WEIGHT_BASE: u64 = 0x10_0000_0000;
const TRANSIENT: u64 = 0x20_0000_0000;
const CAPTURE_BASE: u64 = 0x30_0000_0000;

fn fake_tensor(cursor: &mut u64, dims: &[u64], kind: GgmlType) -> GgufDeviceTensor {
    let byte_len = if kind == GgmlType::F32 {
        dims.iter().product::<u64>() as usize * 4
    } else {
        let plan = crate::layers::ops::GgmlIqMmqPlan::new(kind, 1, dims[1] as u32, dims[0] as u32)
            .unwrap();
        plan.weight_bytes * dims.get(2).copied().unwrap_or(1) as usize
    };
    let ptr = DevicePtr(*cursor);
    // Stride past the payload so no two fabricated weights alias.
    *cursor += byte_len as u64 + 0x1_0000;
    GgufDeviceTensor {
        ptr,
        dimensions: dims.to_vec(),
        ggml_type: kind,
        byte_len,
        alloc_bytes: spark_runtime::weights::gguf::mmq_tensor_alloc_bytes(kind, &dims, byte_len)
            .expect("test tensor slack"),
    }
}

/// One layer of MoE weights, shaped exactly like the pinned GLM schema.
fn moe_weights(cursor: &mut u64) -> Glm53MoeWeights {
    let router_t = fake_tensor(cursor, &[4096, 288], GgmlType::F32);
    let bias_t = fake_tensor(cursor, &[288], GgmlType::F32);
    let gate_t = fake_tensor(cursor, &[4096, 2048, 288], GgmlType::IQ2_XXS);
    let up_t = fake_tensor(cursor, &[4096, 2048, 288], GgmlType::IQ2_XXS);
    let down_t = fake_tensor(cursor, &[2048, 4096, 288], GgmlType::IQ3_XXS);
    let shared_gate_t = fake_tensor(cursor, &[4096, 2048], GgmlType::Q5_K);
    let shared_up_t = fake_tensor(cursor, &[4096, 2048], GgmlType::Q5_K);
    let shared_down_t = fake_tensor(cursor, &[2048, 4096], GgmlType::Q6_K);
    Glm53MoeWeights {
        router: Glm53GgufF32::new(&router_t, &[4096, 288]).unwrap(),
        expert_bias: Glm53GgufF32::new(&bias_t, &[288]).unwrap(),
        gate_experts: Glm53GgufMatrixBank::new(&gate_t).unwrap(),
        up_experts: Glm53GgufMatrixBank::new(&up_t).unwrap(),
        down_experts: Glm53GgufMatrixBank::new(&down_t).unwrap(),
        shared_gate: Glm53GgufMatrix::new(&shared_gate_t).unwrap(),
        shared_up: Glm53GgufMatrix::new(&shared_up_t).unwrap(),
        shared_down: Glm53GgufMatrix::new(&shared_down_t).unwrap(),
    }
}

/// 45 layers: dense for 0..3, MoE thereafter, matching the schedule's census.
fn ffn_weights_only() -> Vec<Glm53FfnWeights> {
    let mut cursor = 0x100_0000_0000u64;
    (0..45)
        .map(|layer| {
            if layer < 3 {
                // Dense layers are not dispatchable yet; a MoE placeholder here
                // would make the schedule/weights mismatch untestable, so build
                // real dense weights.
                let gate_t = fake_tensor(&mut cursor, &[4096, 12288], GgmlType::Q5_K);
                let up_t = fake_tensor(&mut cursor, &[4096, 12288], GgmlType::Q5_K);
                let down_t = fake_tensor(&mut cursor, &[12288, 4096], GgmlType::Q6_K);
                Glm53FfnWeights::Dense(crate::weight_loader::Glm53DenseFfnWeights {
                    gate: Glm53GgufMatrix::new(&gate_t).unwrap(),
                    up: Glm53GgufMatrix::new(&up_t).unwrap(),
                    down: Glm53GgufMatrix::new(&down_t).unwrap(),
                })
            } else {
                Glm53FfnWeights::Moe(moe_weights(&mut cursor))
            }
        })
        .collect()
}

fn f32_tensor_at(c: &mut u64, dims: &[u64]) -> GgufDeviceTensor {
    let elements: u64 = dims.iter().product();
    let ptr = DevicePtr(*c);
    *c += elements * 4 + 0x1_0000;
    GgufDeviceTensor {
        ptr,
        dimensions: dims.to_vec(),
        ggml_type: GgmlType::F32,
        byte_len: (elements * 4) as usize,
        alloc_bytes: spark_runtime::weights::gguf::mmq_tensor_alloc_bytes(
            GgmlType::F32,
            dims,
            (elements * 4) as usize,
        )
        .expect("test tensor slack"),
    }
}

/// 45 layers of attention weights, KDA except where `layer % 4 == 3`.
fn attention_weights_only() -> Vec<Glm53AttentionWeights> {
    let mut c = 0x200_0000_0000u64;
    let q8 = GgmlType::Q8_0;
    (0..45)
        .map(|layer| {
            if layer % 4 == 3 {
                Glm53AttentionWeights::Dsa(Glm53DsaWeights {
                    k_b: Glm53GgufMatrixBank::new(&fake_tensor(&mut c, &[256, 512, 64], q8))
                        .unwrap(),
                    v_b: Glm53GgufMatrixBank::new(&fake_tensor(&mut c, &[512, 256, 64], q8))
                        .unwrap(),
                    kv_a_mqa: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 512], q8)).unwrap(),
                    kv_a_norm: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[512]), &[512]).unwrap(),
                    q_a: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 1536], q8)).unwrap(),
                    q_a_norm: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[1536]), &[1536]).unwrap(),
                    q_b: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[1536, 16384], q8)).unwrap(),
                    output: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[16384, 4096], q8)).unwrap(),
                    indexer_k: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 128], q8))
                        .unwrap(),
                    indexer_q_b: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[1536, 4096], q8))
                        .unwrap(),
                    indexer_k_norm: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[128]), &[128])
                        .unwrap(),
                    indexer_k_norm_bias: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[128]), &[128])
                        .unwrap(),
                    indexer_proj: Glm53GgufF32::new(
                        &f32_tensor_at(&mut c, &[4096, 32]),
                        &[4096, 32],
                    )
                    .unwrap(),
                    compressor_ape: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[128, 4]), &[128, 4])
                        .unwrap(),
                    compressor_gate: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 128], q8))
                        .unwrap(),
                })
            } else {
                Glm53AttentionWeights::Kda(Glm53KdaWeights {
                    q: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 8192], q8)).unwrap(),
                    k: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 8192], q8)).unwrap(),
                    v: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 8192], q8)).unwrap(),
                    output: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[8192, 4096], q8)).unwrap(),
                    conv_q: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[4, 1, 8192]), &[4, 1, 8192])
                        .unwrap(),
                    conv_k: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[4, 1, 8192]), &[4, 1, 8192])
                        .unwrap(),
                    conv_v: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[4, 1, 8192]), &[4, 1, 8192])
                        .unwrap(),
                    a: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[64]), &[64]).unwrap(),
                    beta: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 64], q8)).unwrap(),
                    dt_bias: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[8192]), &[8192]).unwrap(),
                    f_a: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 128], q8)).unwrap(),
                    f_b: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[128, 8192], q8)).unwrap(),
                    g_a: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, 128], q8)).unwrap(),
                    g_b: Glm53GgufMatrix::new(&fake_tensor(&mut c, &[128, 8192], q8)).unwrap(),
                    norm: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[128]), &[128]).unwrap(),
                })
            }
        })
        .collect()
}

const CAPACITY: u32 = 1_048_576;
const POSITION: u32 = 4_096;
const VOCAB: u64 = 154_880;

/// `output.weight`, the LM head.
fn lm_head() -> Glm53GgufMatrix {
    let mut c = 0x400_0000_0000u64;
    Glm53GgufMatrix::new(&fake_tensor(&mut c, &[4096, VOCAB], GgmlType::Q4_K)).unwrap()
}

fn attention_binding(gpu: &dyn GpuBackend) -> Glm53AttentionBinding {
    let context =
        Glm53ContextPlan::new(1, GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage::BF16).unwrap();
    let layout = Glm53T1StateLayout::exact().unwrap();
    let t1 = Glm53ArenaPlan::exact_full_1m_b1()
        .unwrap()
        .t1_transaction
        .offset_bytes;
    let at = |bytes: usize| GgmlIqBuffer {
        ptr: gpu.alloc(bytes.max(1)).unwrap(),
        bytes,
    };
    let conv_state = 3 * 8_192 * 4 * 4;
    let pools = (POSITION / 4) as usize;
    Glm53AttentionBinding {
        kda_states: (0..34)
            .map(|ordinal| {
                Glm53KdaScratchState::stage(DevicePtr(BASE), t1, &layout, &context, ordinal)
                    .unwrap()
            })
            .collect(),
        kda_conv: (0..34)
            .map(|_| Glm53KdaConvSlots {
                persistent_state_f32: at(conv_state),
                staged_state_f32: at(conv_state),
                published_ends_u32: at(4),
                published_nonces_u64: at(8),
                logical_lengths_u32: at(4),
            })
            .collect(),
        dsa_cache: (0..11)
            .map(|_| Glm53DsaCacheSlots {
                latent_cache_bf16: at(CAPACITY as usize * 512 * 2),
                latent_overlay_bf16: at(512 * 2),
                pool_keys_bf16: at(pools * 128 * 2),
                pool_validity_u8: at(pools),
                // Sized from the constant, never a literal.
                prior_tail_keys_bf16: at(TAIL_CAP * 128 * 2),
                prior_tail_gates_bf16: at(TAIL_CAP * 128 * 2),
                prior_tail_validity_u8: at(TAIL_CAP),
                out_tail_validity_u8: at(TAIL_CAP),
                sequence_lengths_u32: at(4),
                query_positions_u32: at(4),
                query_validity_u8: at(1),
                published_ends_u32: at(4),
                published_nonces_u64: at(8),
            })
            .collect(),
        geometry: Glm53DsaLayerGeometry {
            position: POSITION,
            capacity: CAPACITY,
            nonce: 0x51D3,
        },
    }
}

fn logits(gpu: &dyn GpuBackend) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: gpu.alloc(VOCAB as usize * 2).unwrap(),
        bytes: VOCAB as usize * 2,
    }
}

/// One hyper branch of fabricated weights.
fn hyper_branch(c: &mut u64) -> crate::weight_loader::Glm53HyperBranchWeights {
    crate::weight_loader::Glm53HyperBranchWeights {
        function: Glm53GgufMatrix::new(&fake_tensor(c, &[16384, 24], GgmlType::Q8_0)).unwrap(),
        base: Glm53GgufF32::new(&f32_tensor_at(c, &[24]), &[24]).unwrap(),
        scale: Glm53GgufF32::new(&f32_tensor_at(c, &[3]), &[3]).unwrap(),
    }
}

/// The catalog's 45 target layers, exactly as the GGUF loader yields them.
fn catalog_layers() -> Vec<crate::weight_loader::Glm53TargetLayerWeights> {
    let mut c = 0x300_0000_0000u64;
    let mut ffn = ffn_weights_only();
    let mut attention = attention_weights_only();
    (0..45)
        .map(|layer| crate::weight_loader::Glm53TargetLayerWeights {
            descriptor: crate::weight_loader::Glm53LayerDescriptor::new(layer as u32).unwrap(),
            norms: crate::weight_loader::Glm53LayerNorms {
                attention: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[4096]), &[4096]).unwrap(),
                ffn: Glm53GgufF32::new(&f32_tensor_at(&mut c, &[4096]), &[4096]).unwrap(),
            },
            hyper: crate::weight_loader::Glm53HyperWeights {
                attention: hyper_branch(&mut c),
                ffn: hyper_branch(&mut c),
            },
            attention: attention.remove(0),
            ffn: ffn.remove(0),
        })
        .collect()
}

/// Distinct, non-overlapping F32 operands per layer, so a mis-indexed layer
/// shows up as a wrong pointer rather than passing silently.
fn operands(expanded: &Glm53MhcExpanded) -> Vec<Glm53LayerHyperOperands> {
    (0..45)
        .map(|layer| {
            let at = |slot: u64| GgmlIqBuffer {
                ptr: DevicePtr(WEIGHT_BASE + (layer as u64 * 8 + slot) * 0x1_0000),
                bytes: 0,
            };
            Glm53LayerHyperOperands {
                attn_function: expanded.slot(layer, Glm53HyperBranch::Attention).unwrap(),
                attn_base: GgmlIqBuffer {
                    bytes: 24 * 4,
                    ..at(0)
                },
                attn_scale: GgmlIqBuffer {
                    bytes: 3 * 4,
                    ..at(1)
                },
                attn_norm: GgmlIqBuffer {
                    bytes: 4096 * 4,
                    ..at(2)
                },
                ffn_function: expanded.slot(layer, Glm53HyperBranch::Ffn).unwrap(),
                ffn_base: GgmlIqBuffer {
                    bytes: 24 * 4,
                    ..at(3)
                },
                ffn_scale: GgmlIqBuffer {
                    bytes: 3 * 4,
                    ..at(4)
                },
                ffn_norm: GgmlIqBuffer {
                    bytes: 4096 * 4,
                    ..at(5)
                },
            }
        })
        .collect()
}

fn fixture<'w>(
    gpu: &dyn GpuBackend,
    catalog: &'w [crate::weight_loader::Glm53TargetLayerWeights],
    head: &'w Glm53GgufMatrix,
) -> (Glm53TargetSchedule, Glm53Dispatcher<'w>, Glm53WalkScratch) {
    let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1)).unwrap();
    let bound = Glm53BoundWorkspace::bind(
        &schedule.workspace,
        DevicePtr(BASE),
        schedule.workspace.arena_bytes,
    )
    .unwrap();
    let expanded = Glm53MhcExpanded::bind(
        DevicePtr(MHC_BASE),
        crate::model::glm53::arena::GLM53_MHC_EXPANDED_F32_BYTES,
    )
    .unwrap();
    let output_norm = GgmlIqBuffer {
        ptr: DevicePtr(WEIGHT_BASE + 0xF00_0000),
        bytes: 4096 * 4,
    };
    let scratch = alloc_scratch(gpu);
    let captures = Glm53CaptureSlots::bind(
        DevicePtr(CAPTURE_BASE),
        crate::model::glm53::arena::GLM53_DFLASH_CAPTURE_BYTES,
    )
    .unwrap();
    let d = Glm53Dispatcher::new(
        gpu,
        1,
        bound,
        operands(&expanded),
        output_norm,
        scratch,
        captures,
        catalog,
        attention_binding(gpu),
        head,
        GgmlIqBuffer {
            ptr: gpu.alloc(VOCAB as usize * 2).unwrap(),
            bytes: VOCAB as usize * 2,
        },
    )
    .unwrap();
    (schedule, d, scratch)
}

/// Back the transient region with real mock memory so `copy_h2d`/`copy_d2h`
/// work — the serial MoE executor reads its route ids back to the host.
fn alloc_scratch(gpu: &dyn GpuBackend) -> Glm53WalkScratch {
    let bytes = Glm53WalkScratch::required_bytes();
    let raw = gpu.alloc(bytes as usize + 256).unwrap();
    let aligned = DevicePtr((raw.0 + 255) & !255);
    Glm53WalkScratch::bind(aligned, bytes).unwrap()
}

/// The eight routed experts the serial MoE executor decodes from device memory.
/// They must be unique and in range or the op refuses before any expert runs.
fn seed_route_ids(gpu: &dyn GpuBackend, scratch: &Glm53WalkScratch) {
    let dummy = GgmlIqBuffer {
        ptr: DevicePtr(0x9000_0000),
        bytes: 8_192,
    };
    let ids: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, scratch.moe_buffers(dummy, dummy).route_ids_u32.ptr)
        .unwrap();
}

/// ExpandMhc must actually reach the GPU — asserted by a recorded launch, not
/// by dispatch returning Ok.
#[test]
fn expand_mhc_issues_a_launch() {
    let gpu = RecordingGpu::new();
    let catalog = catalog_layers();
    let head = lm_head();
    let (_, d, _scratch) = fixture(&gpu, &catalog, &head);
    d.dispatch(&gpu, &Glm53TargetEvent::ExpandMhc, 0).unwrap();
    assert_eq!(gpu.sequence(), vec!["atlas_glm53_hc_expand"]);
}

/// OrderedMean is the inverse reduction and must also launch.
#[test]
fn ordered_mean_issues_a_launch() {
    let gpu = RecordingGpu::new();
    let catalog = catalog_layers();
    let head = lm_head();
    let (_, d, _scratch) = fixture(&gpu, &catalog, &head);
    d.dispatch(&gpu, &Glm53TargetEvent::OrderedMean, 0).unwrap();
    assert_eq!(gpu.sequence(), vec!["atlas_glm53_hc_mean"]);
}

/// The composed per-layer order must match the reference graph exactly:
/// pre, norm, [attention], post, pre, norm, [ffn], post.
#[test]
fn one_layer_composes_the_reference_hyper_sequence() {
    let gpu = RecordingGpu::new();
    let catalog = catalog_layers();
    let head = lm_head();
    let (_, d, _scratch) = fixture(&gpu, &catalog, &head);

    d.dispatch(&gpu, &Glm53TargetEvent::PreAttention { layer: 7 }, 0)
        .unwrap();
    // Attention would run here; it is not wired yet.
    d.dispatch(&gpu, &Glm53TargetEvent::PostAttention { layer: 7 }, 0)
        .unwrap();
    // FFN would run here.
    d.dispatch(&gpu, &Glm53TargetEvent::PostFfn { layer: 7 }, 0)
        .unwrap();

    assert_eq!(
        gpu.sequence(),
        vec![
            // PreAttention: hc_pre(attn) then attn_norm
            "atlas_glm53_hc_pre",
            "atlas_glm53_rms_norm",
            // PostAttention: hc_post(attn), then hc_pre(ffn) and ffn_norm
            "atlas_glm53_hc_post",
            "atlas_glm53_hc_pre",
            "atlas_glm53_rms_norm",
            // PostFfn: hc_post(ffn)
            "atlas_glm53_hc_post",
        ]
    );
}

/// FinalNormF32 applies `output_norm`, not a layer norm.
#[test]
fn final_norm_uses_the_output_norm_weight() {
    let gpu = RecordingGpu::new();
    let catalog = catalog_layers();
    let head = lm_head();
    let (_, d, _scratch) = fixture(&gpu, &catalog, &head);
    d.dispatch(&gpu, &Glm53TargetEvent::FinalNormF32, 0)
        .unwrap();
    assert_eq!(gpu.sequence(), vec!["atlas_glm53_rms_norm"]);
}

/// A layer outside the mHC range must refuse rather than index past its
/// operands. Layer 45 is the NextN block and has no hyper connections.
#[test]
fn layers_without_mhc_operands_refuse() {
    let gpu = RecordingGpu::new();
    let catalog = catalog_layers();
    let head = lm_head();
    let (_, d, _scratch) = fixture(&gpu, &catalog, &head);
    for event in [
        Glm53TargetEvent::PreAttention { layer: 45 },
        Glm53TargetEvent::PostAttention { layer: 45 },
        Glm53TargetEvent::PostFfn { layer: 99 },
    ] {
        assert!(
            d.dispatch(&gpu, &event, 0).is_err(),
            "{event:?} must refuse"
        );
    }
    assert_eq!(gpu.launch_count(), 0, "a refused event must not launch");
}

/// The dispatcher must reject a layer count that does not cover the walk.
#[test]
fn construction_requires_every_layer() {
    let gpu = RecordingGpu::new();
    let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1)).unwrap();
    let bound = Glm53BoundWorkspace::bind(
        &schedule.workspace,
        DevicePtr(BASE),
        schedule.workspace.arena_bytes,
    )
    .unwrap();
    let expanded = Glm53MhcExpanded::bind(
        DevicePtr(MHC_BASE),
        crate::model::glm53::arena::GLM53_MHC_EXPANDED_F32_BYTES,
    )
    .unwrap();
    let mut short = operands(&expanded);
    short.pop();
    let output_norm = GgmlIqBuffer {
        ptr: DevicePtr(WEIGHT_BASE + 0xF00_0000),
        bytes: 4096 * 4,
    };
    let head = lm_head();
    let scratch = alloc_scratch(&gpu);
    let captures = Glm53CaptureSlots::bind(
        DevicePtr(CAPTURE_BASE),
        crate::model::glm53::arena::GLM53_DFLASH_CAPTURE_BYTES,
    )
    .unwrap();
    let catalog = catalog_layers();
    assert!(
        Glm53Dispatcher::new(
            &gpu,
            1,
            bound,
            short,
            output_norm,
            scratch,
            captures,
            &catalog,
            attention_binding(&gpu),
            &head,
            logits(&gpu),
        )
        .is_err()
    );
    // A short FFN weight list must be refused for the same reason.
    let expanded2 = Glm53MhcExpanded::bind(
        DevicePtr(MHC_BASE),
        crate::model::glm53::arena::GLM53_MHC_EXPANDED_F32_BYTES,
    )
    .unwrap();
    assert!(
        Glm53Dispatcher::new(
            &gpu,
            1,
            bound,
            operands(&expanded2),
            output_norm,
            scratch,
            captures,
            &catalog[..44],
            attention_binding(&gpu),
            &head,
            logits(&gpu),
        )
        .is_err()
    );
    // A catalog whose layer topology disagrees with the schedule must be
    // refused: a KDA-scheduled layer handed DSA weights is silent corruption.
    let expanded3 = Glm53MhcExpanded::bind(
        DevicePtr(MHC_BASE),
        crate::model::glm53::arena::GLM53_MHC_EXPANDED_F32_BYTES,
    )
    .unwrap();
    let mut swapped = catalog_layers();
    swapped.swap(0, 3);
    assert!(
        Glm53Dispatcher::new(
            &gpu,
            1,
            bound,
            operands(&expanded3),
            output_norm,
            scratch,
            captures,
            &swapped,
            attention_binding(&gpu),
            &head,
            logits(&gpu),
        )
        .is_err()
    );

    // A logits buffer too small for the vocabulary must be refused.
    let expanded4 = Glm53MhcExpanded::bind(
        DevicePtr(MHC_BASE),
        crate::model::glm53::arena::GLM53_MHC_EXPANDED_F32_BYTES,
    )
    .unwrap();
    assert!(
        Glm53Dispatcher::new(
            &gpu,
            1,
            bound,
            operands(&expanded4),
            output_norm,
            scratch,
            captures,
            &catalog,
            attention_binding(&gpu),
            &head,
            GgmlIqBuffer {
                ptr: gpu.alloc(1024).unwrap(),
                bytes: 1024,
            },
        )
        .is_err()
    );
}

/// **The milestone test.** Every one of the 234 schedule events must dispatch,
/// in order, without a single refusal — this is the whole one-token walk.
///
/// It replaces the former `unwired_events_error_and_never_launch`, which
/// counted refusals and became vacuous the moment the last event was wired. The
/// property that matters now is total coverage, and the refusal path is still
/// guarded by `layers_without_mhc_operands_refuse` and the per-executor
/// shape checks.
#[test]
fn the_whole_schedule_walks_without_a_single_refusal() {
    let gpu = RecordingGpu::new();
    let catalog = catalog_layers();
    let head = lm_head();
    let (schedule, d, scratch) = fixture(&gpu, &catalog, &head);
    seed_route_ids(&gpu, &scratch);

    let mut dispatched = 0usize;
    for event in schedule.events() {
        d.dispatch(&gpu, event, 0)
            .unwrap_or_else(|error| panic!("event {event:?} refused: {error:#}"));
        dispatched += 1;
    }
    assert_eq!(dispatched, 234, "the walk must cover every schedule event");

    // Every event kind must be represented among the launches, and the walk
    // must be substantial: a silently-skipping dispatcher would show far fewer.
    assert!(
        gpu.launch_count() > 2_000,
        "a full walk should issue thousands of launches, got {}",
        gpu.launch_count()
    );
    let symbols = gpu.sequence();
    for required in [
        "atlas_glm53_hc_expand",
        "atlas_glm53_hc_pre",
        "atlas_glm53_hc_post",
        "atlas_glm53_hc_mean",
        "atlas_glm53_rms_norm",
        "atlas_glm53_kda_decode",
        "atlas_glm53_dsa_selected_attention_bf16",
        "atlas_glm53_router_logits_t4",
        "atlas_glm53_ordered_expert_reduce",
    ] {
        assert!(
            symbols.iter().any(|s| s == required),
            "the walk never ran {required}"
        );
    }
}

/// Dispatchable set and actual success must agree — no event may claim to be
/// dispatchable and then refuse, or vice versa.
///
/// MoE FFN is excluded here and covered by `moe_ffn_runs_the_serial_executor`:
/// it decodes eight routed expert ids *from device memory*, so against an
/// unseeded buffer it legitimately refuses. That refusal is a property of the
/// data, not of the wiring, and folding it in here would force the predicate
/// to lie about what is implemented.
#[test]
fn dispatchable_predicate_matches_behaviour() {
    let gpu = RecordingGpu::new();
    let catalog = catalog_layers();
    let head = lm_head();
    let (schedule, d, _scratch) = fixture(&gpu, &catalog, &head);
    for event in schedule.events() {
        if matches!(
            event,
            Glm53TargetEvent::Ffn {
                kind: Glm53TargetFfnKind::Moe,
                ..
            }
        ) {
            continue;
        }
        let ok = d.dispatch(&gpu, event, 0).is_ok();
        assert_eq!(
            ok,
            Glm53Dispatcher::dispatchable(event),
            "predicate disagrees with behaviour for {event:?}"
        );
    }
}
