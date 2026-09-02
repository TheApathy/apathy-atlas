// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::Result;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::gguf::{GgmlType, GgufDeviceTensor};

use super::*;
use crate::weight_loader::{Glm53GgufF32, Glm53GgufMatrix, Glm53GgufMatrixBank};

struct TraceGpu {
    inner: MockGpuBackend,
    capturing: AtomicBool,
    d2h: AtomicUsize,
    syncs: AtomicUsize,
    launches: Mutex<Vec<(u64, u64)>>,
}

impl TraceGpu {
    fn new(capturing: bool) -> Self {
        Self {
            inner: MockGpuBackend::new(),
            capturing: AtomicBool::new(capturing),
            d2h: AtomicUsize::new(0),
            syncs: AtomicUsize::new(0),
            launches: Mutex::new(Vec::new()),
        }
    }

    fn counts(&self) -> (usize, usize, usize) {
        (
            self.d2h.load(Ordering::Relaxed),
            self.syncs.load(Ordering::Relaxed),
            self.launches.lock().unwrap().len(),
        )
    }
}

impl GpuBackend for TraceGpu {
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
    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], _stream: u64) -> Result<()> {
        self.d2h.fetch_add(1, Ordering::Relaxed);
        self.syncs.fetch_add(1, Ordering::Relaxed);
        self.inner.copy_d2h(src, dst)
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }
    fn launch(
        &self,
        func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let first = params
            .first()
            .map_or(0, |ptr| unsafe { *(*ptr as *const u64) });
        self.launches.lock().unwrap().push((func.0, first));
        Ok(())
    }
    fn stream_is_capturing(&self, _stream: u64) -> bool {
        self.capturing.load(Ordering::Relaxed)
    }
    fn synchronize(&self, _stream: u64) -> Result<()> {
        self.syncs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, _module: &str, name: &str) -> Result<KernelHandle> {
        let handle = if name.contains("quantize") {
            1
        } else if name.contains("q4_k_mmq") {
            20
        } else if name.contains("q5_k_mmq") {
            21
        } else if name.contains("iq2_xs_mmq") {
            22
        } else if name.contains("iq3_xxs_mmq") {
            23
        } else if name.contains("mmq") {
            2
        } else if name.contains("clamped_swiglu") {
            3
        } else if name.contains("ordered_expert_reduce") {
            4
        } else {
            99
        };
        Ok(KernelHandle(handle))
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
}

#[derive(Clone, Copy)]
struct WeightTrace {
    gate: u64,
    up: u64,
    down: u64,
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    shared_gate: u64,
    shared_up: u64,
    shared_down: u64,
}

fn fake_tensor(cursor: &mut u64, dimensions: &[u64], kind: GgmlType) -> GgufDeviceTensor {
    let byte_len = if kind == GgmlType::F32 {
        dimensions.iter().product::<u64>() as usize * 4
    } else {
        let plan = GgmlIqMmqPlan::new(kind, 1, dimensions[1] as u32, dimensions[0] as u32).unwrap();
        plan.weight_bytes * dimensions.get(2).copied().unwrap_or(1) as usize
    };
    let ptr = DevicePtr(*cursor);
    *cursor += byte_len as u64 + 0x1_0000;
    GgufDeviceTensor {
        ptr,
        dimensions: dimensions.to_vec(),
        ggml_type: kind,
        byte_len,
    }
}

fn fake_weights() -> (Glm53MoeWeights, WeightTrace) {
    fake_weights_with(2048, [GgmlType::IQ3_S; 6])
}

fn fake_weights_with_shared(shared_intermediate: u64) -> (Glm53MoeWeights, WeightTrace) {
    fake_weights_with(shared_intermediate, [GgmlType::IQ3_S; 6])
}

fn fake_weights_with(
    shared_intermediate: u64,
    kinds: [GgmlType; 6],
) -> (Glm53MoeWeights, WeightTrace) {
    let mut cursor = 0x10_0000_0000u64;
    let router_t = fake_tensor(&mut cursor, &[4096, 288], GgmlType::F32);
    let bias_t = fake_tensor(&mut cursor, &[288], GgmlType::F32);
    let gate_t = fake_tensor(&mut cursor, &[4096, 2048, 288], kinds[0]);
    let up_t = fake_tensor(&mut cursor, &[4096, 2048, 288], kinds[1]);
    let down_t = fake_tensor(&mut cursor, &[2048, 4096, 288], kinds[2]);
    let shared_gate_t = fake_tensor(&mut cursor, &[4096, shared_intermediate], kinds[3]);
    let shared_up_t = fake_tensor(&mut cursor, &[4096, shared_intermediate], kinds[4]);
    let shared_down_t = fake_tensor(&mut cursor, &[shared_intermediate, 4096], kinds[5]);
    let gate = Glm53GgufMatrixBank::new(&gate_t).unwrap();
    let up = Glm53GgufMatrixBank::new(&up_t).unwrap();
    let down = Glm53GgufMatrixBank::new(&down_t).unwrap();
    let trace = WeightTrace {
        gate: gate_t.ptr.0,
        up: up_t.ptr.0,
        down: down_t.ptr.0,
        gate_stride: gate.expert(0).unwrap().buffer().bytes,
        up_stride: up.expert(0).unwrap().buffer().bytes,
        down_stride: down.expert(0).unwrap().buffer().bytes,
        shared_gate: shared_gate_t.ptr.0,
        shared_up: shared_up_t.ptr.0,
        shared_down: shared_down_t.ptr.0,
    };
    (
        Glm53MoeWeights {
            router: Glm53GgufF32::new(&router_t, &[4096, 288]).unwrap(),
            expert_bias: Glm53GgufF32::new(&bias_t, &[288]).unwrap(),
            gate_experts: gate,
            up_experts: up,
            down_experts: down,
            shared_gate: Glm53GgufMatrix::new(&shared_gate_t).unwrap(),
            shared_up: Glm53GgufMatrix::new(&shared_up_t).unwrap(),
            shared_down: Glm53GgufMatrix::new(&shared_down_t).unwrap(),
        },
        trace,
    )
}

fn alloc_buffer(gpu: &TraceGpu, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: gpu.alloc(bytes).unwrap(),
        bytes,
    }
}

fn buffers(gpu: &TraceGpu) -> Glm53SerialMoeBuffers {
    Glm53SerialMoeBuffers {
        input_bf16: alloc_buffer(gpu, HIDDEN_BYTES),
        route_ids_u32: alloc_buffer(gpu, ROUTE_BYTES),
        route_weights_f32: alloc_buffer(gpu, ROUTE_BYTES),
        q8_activation: alloc_buffer(gpu, MAX_Q8_ACTIVATION_BYTES),
        expert_gate_bf16: alloc_buffer(gpu, EXPERT_INTERMEDIATE_BYTES),
        expert_up_bf16: alloc_buffer(gpu, EXPERT_INTERMEDIATE_BYTES),
        expert_swiglu_bf16: alloc_buffer(gpu, EXPERT_INTERMEDIATE_BYTES),
        routed_bf16: alloc_buffer(gpu, ROUTED_BYTES),
        shared_gate_bf16: alloc_buffer(gpu, SHARED_INTERMEDIATE_BYTES),
        shared_up_bf16: alloc_buffer(gpu, SHARED_INTERMEDIATE_BYTES),
        shared_swiglu_bf16: alloc_buffer(gpu, SHARED_INTERMEDIATE_BYTES),
        shared_bf16: alloc_buffer(gpu, HIDDEN_BYTES),
        output_bf16: alloc_buffer(gpu, HIDDEN_BYTES),
    }
}

fn put_ids(gpu: &TraceGpu, buffers: Glm53SerialMoeBuffers, ids: [u32; 8]) {
    let mut raw = [0u8; ROUTE_BYTES];
    for (slot, id) in ids.into_iter().enumerate() {
        raw[slot * 4..slot * 4 + 4].copy_from_slice(&id.to_le_bytes());
    }
    gpu.copy_h2d(&raw, buffers.route_ids_u32.ptr).unwrap();
}

#[test]
fn graph_capture_rejects_before_copy_or_launch() {
    let gpu = TraceGpu::new(true);
    let kernels = Glm53SerialMoeKernels::load(&gpu).unwrap();
    let (weights, _) = fake_weights();
    let buffers = buffers(&gpu);
    assert!(kernels.execute(&gpu, &weights, buffers, 7).is_err());
    assert_eq!(gpu.counts(), (0, 0, 0));
}

#[test]
fn forged_extent_and_overlap_reject_before_copy_or_launch() {
    let gpu = TraceGpu::new(false);
    let kernels = Glm53SerialMoeKernels::load(&gpu).unwrap();
    let (weights, _) = fake_weights();
    let valid = buffers(&gpu);
    let overlap = Glm53SerialMoeBuffers {
        output_bf16: GgmlIqBuffer {
            ptr: valid.routed_bf16.ptr,
            bytes: HIDDEN_BYTES,
        },
        ..valid
    };
    assert!(kernels.execute(&gpu, &weights, overlap, 7).is_err());
    assert_eq!(gpu.counts(), (0, 0, 0));
}

#[test]
fn old_shared_i12288_rejects_before_effects_and_correct_scratch_is_pinned() {
    assert_eq!(SHARED_INTERMEDIATE, 2048);
    assert_eq!(SHARED_INTERMEDIATE_BYTES, 4096);
    assert_eq!(MAX_Q8_ACTIVATION_BYTES, 4096 / 128 * 144);
    let gpu = TraceGpu::new(false);
    let kernels = Glm53SerialMoeKernels::load(&gpu).unwrap();
    let (wrong_weights, _) = fake_weights_with_shared(12288);
    let buffers = buffers(&gpu);
    assert!(kernels.execute(&gpu, &wrong_weights, buffers, 7).is_err());
    assert_eq!(gpu.counts(), (0, 0, 0));
}

#[test]
fn bad_ids_stop_after_the_only_ordered_copy_and_sync() {
    for ids in [[0, 1, 2, 3, 4, 5, 6, 6], [0, 1, 2, 3, 4, 5, 6, 288]] {
        let gpu = TraceGpu::new(false);
        let kernels = Glm53SerialMoeKernels::load(&gpu).unwrap();
        let (weights, _) = fake_weights();
        let buffers = buffers(&gpu);
        put_ids(&gpu, buffers, ids);
        assert!(kernels.execute(&gpu, &weights, buffers, 7).is_err());
        assert_eq!(gpu.counts(), (1, 1, 0));
    }
}

#[test]
fn q2_recipe_kinds_dispatch_exactly_and_unsupported_stays_closed() {
    let gpu = TraceGpu::new(false);
    let kernels = Glm53SerialMoeKernels::load(&gpu).unwrap();
    // IQ2_XXS is admitted: it is the dominant expert type in the UD-IQ2_XXS
    // recipe, which is the checkpoint that fits a single Spark once resident.
    assert!(kernels.mmq(GgmlType::IQ2_XXS).is_ok());
    // Types with no pinned GLM specialization must still fail closed.
    for unsupported in [GgmlType::IQ1_S, GgmlType::IQ1_M, GgmlType::Q4_0] {
        assert!(kernels.mmq(unsupported).is_err());
    }
    assert_eq!(gpu.counts(), (0, 0, 0));

    let (weights, weights_trace) = fake_weights_with(
        2048,
        [
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::IQ2_XS,
            GgmlType::IQ3_XXS,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
        ],
    );
    let buffers = buffers(&gpu);
    let ids = [9, 1, 200, 3, 287, 0, 55, 8];
    put_ids(&gpu, buffers, ids);
    let receipt = kernels.execute(&gpu, &weights, buffers, 7).unwrap();
    assert_eq!(receipt.route_ids, ids);
    assert_eq!(gpu.counts(), (1, 1, 64));

    let launches = gpu.launches.lock().unwrap();
    let routed_group = [1u64, 20, 1, 21, 3, 1, 22];
    for slot in 0..8 {
        assert_eq!(
            launches[slot * 7..slot * 7 + 7]
                .iter()
                .map(|entry| entry.0)
                .collect::<Vec<_>>(),
            routed_group,
        );
    }
    assert_eq!(
        launches[56..63]
            .iter()
            .map(|entry| entry.0)
            .collect::<Vec<_>>(),
        [1u64, 23, 1, 20, 3, 1, 21],
    );
    assert_eq!(launches[63].0, 4);
    let mmq_weights = launches
        .iter()
        .filter(|entry| (20..=23).contains(&entry.0))
        .map(|entry| entry.1)
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for id in ids {
        expected.push(weights_trace.gate + id as u64 * weights_trace.gate_stride as u64);
        expected.push(weights_trace.up + id as u64 * weights_trace.up_stride as u64);
        expected.push(weights_trace.down + id as u64 * weights_trace.down_stride as u64);
    }
    expected.extend([
        weights_trace.shared_gate,
        weights_trace.shared_up,
        weights_trace.shared_down,
    ]);
    assert_eq!(mmq_weights, expected);
}

#[test]
fn valid_routes_preserve_slot_order_and_exact_effect_count() {
    let gpu = TraceGpu::new(false);
    let kernels = Glm53SerialMoeKernels::load(&gpu).unwrap();
    let (weights, weights_trace) = fake_weights();
    let buffers = buffers(&gpu);
    let ids = [7, 1, 200, 3, 287, 0, 55, 8];
    put_ids(&gpu, buffers, ids);
    let receipt = kernels.execute(&gpu, &weights, buffers, 7).unwrap();
    assert_eq!(receipt.route_ids, ids);
    assert_eq!(gpu.counts(), (1, 1, 64));
    assert_eq!(
        (
            receipt.d2h_copies,
            receipt.mandatory_stream_syncs,
            receipt.kernel_launches,
        ),
        (1, 1, 64)
    );

    let launches = gpu.launches.lock().unwrap();
    let group = [1u64, 2, 1, 2, 3, 1, 2];
    for at in 0..9 {
        assert_eq!(
            launches[at * 7..at * 7 + 7]
                .iter()
                .map(|x| x.0)
                .collect::<Vec<_>>(),
            group
        );
    }
    assert_eq!(launches[63].0, 4);
    let mmq_weights = launches
        .iter()
        .filter(|x| x.0 == 2)
        .map(|x| x.1)
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for id in ids {
        expected.push(weights_trace.gate + id as u64 * weights_trace.gate_stride as u64);
        expected.push(weights_trace.up + id as u64 * weights_trace.up_stride as u64);
        expected.push(weights_trace.down + id as u64 * weights_trace.down_stride as u64);
    }
    expected.extend([
        weights_trace.shared_gate,
        weights_trace.shared_up,
        weights_trace.shared_down,
    ]);
    assert_eq!(mmq_weights, expected);
}

#[test]
fn pointer_bearing_weights_are_borrowed_through_execution() {
    let source = include_str!("glm53_moe_serial.rs");
    let borrowed = |source: &str| {
        source.matches("weights: &Glm53MoeWeights").count() == 3
            && source.matches("matrix: &Glm53GgufMatrix").count() == 2
            && source.matches("bank: &Glm53GgufMatrixBank").count() == 2
            && source.contains("self.linear(\n                gpu,\n                &gate,")
            && source.contains("self.bank_plan(&weights.gate_experts")
            && source.contains("validate_ranges(weights, buffers)")
    };
    assert!(borrowed(source));
    for (from, to) in [
        ("weights: &Glm53MoeWeights", "weights: Glm53MoeWeights"),
        ("matrix: &Glm53GgufMatrix", "matrix: Glm53GgufMatrix"),
        ("bank: &Glm53GgufMatrixBank", "bank: Glm53GgufMatrixBank"),
    ] {
        let mutant = source.replacen(from, to, 1);
        assert_ne!(mutant, source);
        assert!(!borrowed(&mutant));
    }
}
