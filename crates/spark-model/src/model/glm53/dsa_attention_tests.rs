// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::gguf::GgufDeviceTensor;

use super::*;

const EXL3_ABSORB_CUDA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_dsa_absorb.cu"
));
const DSA_HOST_SOURCE: &str = include_str!("dsa_attention.rs");

#[path = "dsa_tail_visibility_tests.rs"]
mod tail_visibility;

#[test]
fn exl3_absorption_tiles_every_m2048_row() {
    assert!(EXL3_ABSORB_CUDA.contains("const unsigned row_base = blockIdx.z * ROW_TILE;"));
    assert!(EXL3_ABSORB_CUDA.contains("const unsigned global_row = row_base + row;"));
    assert!(EXL3_ABSORB_CUDA.contains("const unsigned out_row0 = row_base + group;"));
}

#[test]
fn layer_major_uses_one_causal_wide_stage_while_exact_wide_stays_tokenwise() {
    let layer_major = DSA_HOST_SOURCE
        .find("if rows > 1 && layer_major {")
        .expect("layer-major DSA branch");
    let exact_wide = DSA_HOST_SOURCE[layer_major..]
        .find("if rows > 1 && exact_wide {")
        .map(|offset| layer_major + offset)
        .expect("legacy exact-wide DSA branch");
    let batched = &DSA_HOST_SOURCE[layer_major..exact_wide];
    assert!(batched.contains("self.precompute_wide_exl3_rows("));
    assert!(batched.contains(".checked_add(self.stage_inner("));
    assert!(!batched.contains("for row in"));

    let legacy = &DSA_HOST_SOURCE[exact_wide..];
    assert!(legacy.contains("for row in 0..rows as usize"));
}

#[test]
fn dense_causal_attention_is_limited_to_layer_major_full_selector_coverage() {
    let compact: String = DSA_HOST_SOURCE
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    assert!(compact.contains(
        "letdense_full_coverage=dense_full_coverage(rows,geometry.position,geometry.capacity,glm53_layer_major_prefill_active(),)?;"
    ));
    assert!(DSA_HOST_SOURCE.contains("if dense_full_coverage {"));
    assert!(DSA_HOST_SOURCE.contains("self.selected.launch_dense_causal("));
    assert!(DSA_HOST_SOURCE.contains("self.selected\n                .launch(gpu, selected_plan"));
}

const TAIL_CAP: usize = spark_runtime::kv_cache::GLM53_DSA_TAIL_CAPACITY as usize;
use crate::layers::ops::GgmlIqMmqPlan;
use crate::model::glm53::walk_scratch::Glm53WalkScratch;
use crate::weight_loader::{Glm53GgufF32, Glm53GgufMatrixBank};

/// Records every launch's symbol and its first two pointer arguments, so a test
/// can prove which buffer each stage read from and wrote to.
struct TraceGpu {
    inner: MockGpuBackend,
    symbols: Mutex<HashMap<u64, String>>,
    next: Mutex<u64>,
    launched: Mutex<Vec<(String, u64, u64, u64)>>,
    tail_visibility: tail_visibility::Recorder,
}

impl TraceGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            symbols: Mutex::new(HashMap::new()),
            next: Mutex::new(1),
            launched: Mutex::new(Vec::new()),
            tail_visibility: tail_visibility::Recorder::default(),
        }
    }
    fn launches(&self) -> Vec<(String, u64, u64, u64)> {
        self.launched.lock().unwrap().clone()
    }
    fn symbols_seen(&self) -> Vec<String> {
        self.launches().into_iter().map(|(n, _, _, _)| n).collect()
    }
}

impl GpuBackend for TraceGpu {
    fn alloc(&self, b: usize) -> Result<DevicePtr> {
        self.inner.alloc(b)
    }
    fn alloc_managed(&self, b: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(b)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn copy_h2d(&self, s: &[u8], d: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(s, d)
    }
    fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(s, d)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, b: usize) -> Result<()> {
        self.inner.copy_d2d(s, d, b)
    }
    fn memset(&self, p: DevicePtr, v: u8, b: usize) -> Result<()> {
        self.inner.memset(p, v, b)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, b: usize, _s: u64) -> Result<()> {
        self.inner.memset(p, v, b)
    }
    fn launch(
        &self,
        func: KernelHandle,
        _g: [u32; 3],
        _b: [u32; 3],
        _sh: u32,
        _st: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let name = self
            .symbols
            .lock()
            .unwrap()
            .get(&func.0)
            .cloned()
            .unwrap_or_default();
        // SAFETY: the launch builder wrote DevicePtr-sized leading arguments.
        let read = |i: usize| -> u64 {
            if params.len() > i {
                unsafe { *(params[i] as *const u64) }
            } else {
                0
            }
        };
        self.launched
            .lock()
            .unwrap()
            .push((name.clone(), read(0), read(1), read(2)));
        self.tail_visibility.record(&name, params, _st);
        Ok(())
    }
    fn synchronize(&self, _s: u64) -> Result<()> {
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, _m: &str, name: &str) -> Result<KernelHandle> {
        let mut symbols = self.symbols.lock().unwrap();
        if let Some((h, _)) = symbols.iter().find(|(_, k)| k.as_str() == name) {
            return Ok(KernelHandle(*h));
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

fn matrix_tensor(c: &mut u64, dims: &[u64], kind: GgmlType) -> GgufDeviceTensor {
    let plan = GgmlIqMmqPlan::new(kind, 1, dims[1] as u32, dims[0] as u32).unwrap();
    let per = plan.weight_bytes * dims.get(2).copied().unwrap_or(1) as usize;
    let ptr = DevicePtr(*c);
    *c += per as u64 + 0x1_0000;
    GgufDeviceTensor {
        ptr,
        dimensions: dims.to_vec(),
        ggml_type: kind,
        byte_len: per,
        alloc_bytes: spark_runtime::weights::gguf::mmq_tensor_alloc_bytes(kind, &dims, per)
            .expect("test tensor slack"),
    }
}

fn f32_tensor(c: &mut u64, dims: &[u64]) -> GgufDeviceTensor {
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

fn weights(q_rank: u64) -> Glm53DsaWeights {
    let mut c = 0x90_0000_0000u64;
    let q8 = GgmlType::Q8_0;
    Glm53DsaWeights {
        k_b: Glm53GgufMatrixBank::new(&matrix_tensor(&mut c, &[256, 512, 64], q8)).unwrap(),
        v_b: Glm53GgufMatrixBank::new(&matrix_tensor(&mut c, &[512, 256, 64], q8)).unwrap(),
        kv_a_mqa: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, 512], q8)).unwrap(),
        kv_a_norm: Glm53GgufF32::new(&f32_tensor(&mut c, &[512]), &[512]).unwrap(),
        q_a: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, q_rank], GgmlType::Q5_K)).unwrap(),
        q_a_norm: Glm53GgufF32::new(&f32_tensor(&mut c, &[q_rank]), &[q_rank]).unwrap(),
        q_b: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[q_rank, 16384], q8)).unwrap(),
        output: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[16384, 4096], GgmlType::Q5_K))
            .unwrap(),
        indexer_k: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, 128], q8)).unwrap(),
        indexer_q_b: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[q_rank, 4096], q8)).unwrap(),
        indexer_k_norm: Glm53GgufF32::new(&f32_tensor(&mut c, &[128]), &[128]).unwrap(),
        indexer_k_norm_bias: Glm53GgufF32::new(&f32_tensor(&mut c, &[128]), &[128]).unwrap(),
        indexer_proj: Glm53GgufF32::new(&f32_tensor(&mut c, &[4096, 32]), &[4096, 32]).unwrap(),
        compressor_ape: Glm53GgufF32::new(&f32_tensor(&mut c, &[128, 4]), &[128, 4]).unwrap(),
        compressor_gate: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, 128], q8)).unwrap(),
    }
}

const CAPACITY: u32 = 1_048_576;

fn cache(gpu: &dyn GpuBackend, position: u32) -> Glm53DsaCacheSlots {
    let at = |bytes: usize| GgmlIqBuffer {
        ptr: gpu.alloc(bytes.max(1)).unwrap(),
        bytes,
    };
    let pools = (position / 4) as usize;
    Glm53DsaCacheSlots {
        latent_cache_bf16: at(CAPACITY as usize * 512 * 2),
        latent_overlay_bf16: at(512 * 2),
        pool_keys_bf16: at(pools.max(1) * 128 * 2),
        pool_validity_u8: at(pools.max(1)),
        // Sized from the constant, not a literal: a test that hardcodes the old
        // capacity would keep passing while the real buffers were short.
        prior_tail_keys_bf16: at(TAIL_CAP * 128 * 2),
        prior_tail_gates_bf16: at(TAIL_CAP * 128 * 2),
        prior_tail_validity_u8: at(TAIL_CAP),
        out_tail_validity_u8: at(TAIL_CAP),
        sequence_lengths_u32: at(4),
        query_positions_u32: at(4),
        query_validity_u8: at(1),
        published_ends_u32: at(4),
        published_nonces_u64: at(8),
    }
}

fn scratch(gpu: &dyn GpuBackend) -> Glm53WalkScratch {
    let raw = gpu
        .alloc(Glm53WalkScratch::required_bytes() as usize + 256)
        .unwrap();
    Glm53WalkScratch::bind(
        DevicePtr((raw.0 + 255) & !255),
        Glm53WalkScratch::required_bytes(),
    )
    .unwrap()
}

fn run_with(gpu: &TraceGpu, s: &Glm53WalkScratch, position: u32, q_rank: u64) -> Result<u32> {
    let kernels = Glm53DsaAttentionKernels::load(gpu).unwrap();
    let input = GgmlIqBuffer {
        ptr: gpu.alloc(8_192).unwrap(),
        bytes: 8_192,
    };
    let output = GgmlIqBuffer {
        ptr: gpu.alloc(8_192).unwrap(),
        bytes: 8_192,
    };
    kernels.stage(
        gpu,
        &weights(q_rank),
        input,
        s.dsa_buffers(),
        cache(gpu, position),
        Glm53DsaLayerGeometry {
            position,
            capacity: CAPACITY,
            nonce: 0x1234,
        },
        output,
        7,
    )
}

/// Every stage of the reference pipeline must run, in one layer.
#[test]
fn a_layer_runs_the_whole_reference_pipeline() {
    let gpu = TraceGpu::new();
    let s = scratch(&gpu);
    run_with(&gpu, &s, 4_096, 1_536).unwrap();
    let symbols = gpu.symbols_seen();
    for required in [
        "atlas_glm53_dsa_absolute_rms_norm_bf16",
        "atlas_glm53_dsa_biased_layer_norm_bf16",
        "atlas_glm53_dsa_index_projection_f32_bf16",
        "atlas_glm53_dsa_pool_k4",
        "atlas_glm53_dsa_score_bf16",
        "atlas_glm53_dsa_topk_k4",
        "atlas_glm53_dsa_selected_attention_bf16",
        "atlas_glm53_dsa_latent_append_bf16_stage",
    ] {
        assert!(symbols.iter().any(|s| s == required), "missing {required}");
    }
}

/// **The load-bearing test.** The latent append must write the transaction
/// overlay, never the persistent latent cache. Writing persistent here would
/// publish an unaccepted token into the cache with no way to roll it back.
#[test]
fn the_latent_append_targets_the_overlay_never_the_persistent_cache() {
    let gpu = TraceGpu::new();
    let s = scratch(&gpu);
    let slots = cache(&gpu, 4_096);
    let kernels = Glm53DsaAttentionKernels::load(&gpu).unwrap();
    let input = GgmlIqBuffer {
        ptr: gpu.alloc(8_192).unwrap(),
        bytes: 8_192,
    };
    let output = GgmlIqBuffer {
        ptr: gpu.alloc(8_192).unwrap(),
        bytes: 8_192,
    };
    kernels
        .stage(
            &gpu,
            &weights(1_536),
            input,
            s.dsa_buffers(),
            slots,
            Glm53DsaLayerGeometry {
                position: 4_096,
                capacity: CAPACITY,
                nonce: 0x1234,
            },
            output,
            7,
        )
        .unwrap();

    let append = gpu
        .launches()
        .into_iter()
        .find(|(n, _, _, _)| n == "atlas_glm53_dsa_latent_append_bf16_stage")
        .expect("the latent append must run");
    // Args are (source, transaction_overlay, ..).
    assert_eq!(append.1, s.dsa_buffers().kv_cmpr_norm_bf16.ptr.0);
    assert_eq!(
        append.2, slots.latent_overlay_bf16.ptr.0,
        "the append must target the overlay"
    );
    assert_ne!(
        append.2, slots.latent_cache_bf16.ptr.0,
        "the append must never target the persistent latent cache"
    );
}

/// Absorption must cover all 64 heads on both sides, each into its own slice.
/// A shared destination would silently collapse every head onto one.
#[test]
fn absorption_covers_every_head_on_both_sides_into_distinct_slices() {
    let gpu = TraceGpu::new();
    let s = scratch(&gpu);
    run_with(&gpu, &s, 4_096, 1_536).unwrap();

    let buffers = s.dsa_buffers();
    // The matmul kernel's second pointer argument is its output for our
    // per-head calls, so collect destinations landing in each region.
    let launches = gpu.launches();
    let in_range = |base: GgmlIqBuffer, span: u64, value: u64| {
        value >= base.ptr.0 && value < base.ptr.0 + span
    };
    let absorbed: std::collections::BTreeSet<u64> = launches
        .iter()
        .flat_map(|(_, a, b, c)| [*a, *b, *c])
        .filter(|v| in_range(buffers.absorbed_q_bf16, 64 * 512 * 2, *v))
        .collect();
    let unabsorbed: std::collections::BTreeSet<u64> = launches
        .iter()
        .flat_map(|(_, a, b, c)| [*a, *b, *c])
        .filter(|v| in_range(buffers.unabsorbed_bf16, 64 * 256 * 2, *v))
        .collect();
    assert_eq!(
        absorbed.len(),
        64,
        "every head must absorb into its own slice"
    );
    assert_eq!(
        unabsorbed.len(),
        64,
        "every head must un-absorb into its own slice"
    );
    assert_eq!(GLM53_DSA_ABSORPTION_MATMULS, 128);
}

/// A checkpoint whose query rank disagrees with the pinned geometry must fail
/// before anything is enqueued.
#[test]
fn a_wrong_query_rank_is_refused_before_any_effect() {
    let gpu = TraceGpu::new();
    let s = scratch(&gpu);
    let error =
        run_with(&gpu, &s, 4_096, 1_024).expect_err("a 1024 query rank must not be accepted");
    assert!(format!("{error:#}").contains("expected [4096, 1536]"));
    assert!(gpu.launches().is_empty(), "a refused layer must not launch");
}

#[test]
fn geometry_rejects_a_zero_nonce_and_an_out_of_range_position() {
    assert!(
        Glm53DsaLayerGeometry {
            position: 0,
            capacity: CAPACITY,
            nonce: 0,
        }
        .validate()
        .is_err()
    );
    assert!(
        Glm53DsaLayerGeometry {
            position: CAPACITY,
            capacity: CAPACITY,
            nonce: 1,
        }
        .validate()
        .is_err()
    );
    let ok = Glm53DsaLayerGeometry {
        position: 4_096,
        capacity: CAPACITY,
        nonce: 1,
    };
    assert!(ok.validate().is_ok());
    // Pools are complete groups of four; a partial trailing group stays raw.
    assert_eq!(ok.usable_pools(), 1_024);
}

#[test]
fn exact_wide_precompute_row_slices_are_disjoint_and_exact() {
    let whole = GgmlIqBuffer {
        ptr: DevicePtr(0x12_0000),
        bytes: 4 * 3_072,
    };
    let rows = (0..4)
        .map(|row| {
            Glm53DsaAttentionKernels::exact_row_slice(whole, 4, row, 3_072, "query")
                .expect("valid exact row")
        })
        .collect::<Vec<_>>();
    assert_eq!(rows[0].ptr, whole.ptr);
    assert_eq!(rows[3].ptr, whole.ptr.offset(3 * 3_072));
    assert!(rows.iter().all(|row| row.bytes == 3_072));
    assert!(
        Glm53DsaAttentionKernels::exact_row_slice(whole, 4, 4, 3_072, "query").is_err(),
        "an out-of-range row must fail"
    );
    assert!(
        Glm53DsaAttentionKernels::exact_row_slice(
            GgmlIqBuffer {
                bytes: whole.bytes - 2,
                ..whole
            },
            4,
            0,
            3_072,
            "query",
        )
        .is_err(),
        "a prefix buffer must not masquerade as the exact wide slab"
    );
}

/// `usable_pools` counts PUBLISHED pools, which is floor(q / KPOOL).
///
/// The completing pool IS counted at the step that completes it, because the
/// walk now publishes its row into the cache before score/topk. Counting it
/// while publishing at the END of the step reads an unwritten row -- measured,
/// that took p3 from 3.16% to 44.98% -- so the count and the publish position
/// are one change and this test pins the count half of it.
///
/// Not counting it is the defect that produced the period-4 peaks: the raw
/// tail cannot cover the completing group either, so at q = 7, 11, 15, ...
/// three positions were absent from selected_indices altogether. Observed in
/// dump_dsafix layer 3: p7 selected [0,1,2,3] and nothing more.
///
/// The coverage arithmetic that follows is exact: tail occupancy is
/// (q+1) - KPOOL*pools = (q+1) mod KPOOL, whose maximum is KPOOL-1. That is
/// GLM53_DSA_TAIL_CAPACITY, and it agrees with the reference model in
/// `glm53_dsa_topk_tests.rs`, which covers q = 7 as pools {0,1} plus a
/// three-slot tail plus the current position via query metadata.
#[test]
fn usable_pools_counts_published_pools_and_the_tail_covers_the_remainder() {
    let at = |position: u32| {
        Glm53DsaLayerGeometry {
            position,
            capacity: CAPACITY,
            nonce: 1,
        }
        .usable_pools()
    };

    // Published-pool count: the pool completing at q IS counted at q, because
    // its row is published before score/topk.
    assert_eq!(at(0), 0);
    assert_eq!(at(2), 0, "no pool is complete before p3");
    assert_eq!(
        at(3),
        1,
        "pool 0 completes AT p3 and is published before scoring"
    );
    assert_eq!(at(4), 1);
    assert_eq!(at(7), 2, "pool 1 covers 4..7 and completes AT p7");
    assert_eq!(at(8), 2);
    assert_eq!(at(27), 7, "pool 6 covers 24..27 and completes AT p27");

    // The period-4 positions, stated as the coverage they must produce. These
    // are the six that measured 44-63%: at each, the completing pool is the
    // ONLY route by which its group reaches selection.
    for position in [7u32, 11, 15, 19, 23, 27] {
        assert_eq!(
            4 * at(position),
            position + 1,
            "p{position} must be fully pooled with an empty tail"
        );
    }

    // The coverage requirement the tail capacity must satisfy, at every
    // position: pooled + tail == q+1, and tail never exceeds capacity.
    let capacity = spark_runtime::kv_cache::GLM53_DSA_TAIL_CAPACITY;
    let mut worst = 0;
    for position in 0..64u32 {
        let pools = at(position);
        let pooled = 4 * pools;
        let tail = position + 1 - pooled;
        assert_eq!(pooled + tail, position + 1, "coverage gap at {position}");
        worst = worst.max(tail);
        assert!(
            tail <= capacity,
            "position {position} needs {tail} tail slots, capacity is {capacity}"
        );
    }
    // The bound is TIGHT: some position really does need all KPOOL-1 slots, so
    // the capacity is exactly right rather than merely sufficient.
    assert_eq!(
        worst, capacity,
        "tail capacity must be exactly the worst case"
    );
    assert_eq!(capacity, 3, "the reference model carries three tail slots");
}
