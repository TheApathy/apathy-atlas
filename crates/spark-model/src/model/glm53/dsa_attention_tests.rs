// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::gguf::GgufDeviceTensor;

use super::*;
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
}

impl TraceGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            symbols: Mutex::new(HashMap::new()),
            next: Mutex::new(1),
            launched: Mutex::new(Vec::new()),
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
            .push((name, read(0), read(1), read(2)));
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
        prior_tail_keys_bf16: at(3 * 128 * 2),
        prior_tail_gates_bf16: at(3 * 128 * 2),
        prior_tail_validity_u8: at(3),
        out_tail_validity_u8: at(3),
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

/// The pool count must follow the causality rule `4k+3 <= q`, so the usable
/// count is floor((q+1)/KPOOL).
///
/// The pre-existing pin used position 4096, which is 0 mod 4 -- the one residue
/// where the old `q / KPOOL` and the correct `(q+1) / KPOOL` AGREE. It could
/// never have caught this. These cases walk every residue.
#[test]
fn usable_pool_count_follows_the_causality_rule_at_every_residue() {
    let at = |position: u32| Glm53DsaLayerGeometry {
        position,
        capacity: CAPACITY,
        nonce: 1,
    }
    .usable_pools();

    // q ≡ 3 (mod 4) is where the old arithmetic was low by one: the pool that
    // just completed was dropped, losing the four most recent pooled tokens.
    assert_eq!(at(3), 1, "p3 completes pool 0 (positions 0..3)");
    assert_eq!(at(7), 2, "p7 completes pool 1");
    assert_eq!(at(11), 3);
    assert_eq!(at(27), 7);

    // The other three residues are unchanged by the fix.
    assert_eq!(at(0), 0);
    assert_eq!(at(4), 1);
    assert_eq!(at(5), 1);
    assert_eq!(at(6), 1);

    // Every usable pool's last token must have arrived, at every position.
    for position in 0..64u32 {
        let pools = at(position);
        assert!(
            pools == 0 || 4 * pools - 1 <= position,
            "pool {} at position {position} has not received its last token",
            pools - 1
        );
        // And it must be maximal: the next pool must NOT yet be usable.
        assert!(
            4 * (pools + 1) - 1 > position,
            "pool {pools} is usable at position {position} but was not counted"
        );
    }
}
