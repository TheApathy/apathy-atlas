// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::c_void;
use std::sync::Mutex;

use anyhow::Result;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::gguf::GgufDeviceTensor;

use super::*;
use crate::model::glm53::arena::GLM53_T1_TRANSIENT_BYTES;
use crate::model::glm53::walk_scratch::Glm53WalkScratch;

/// Counts launches and records host copies / syncs, because a dense layer must
/// perform neither — that is the whole reason it does not reuse the serial MoE
/// executor.
struct TraceGpu {
    inner: MockGpuBackend,
    launches: Mutex<usize>,
    d2h: Mutex<usize>,
    syncs: Mutex<usize>,
}

impl TraceGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            launches: Mutex::new(0),
            d2h: Mutex::new(0),
            syncs: Mutex::new(0),
        }
    }
    fn counts(&self) -> (usize, usize, usize) {
        (
            *self.launches.lock().unwrap(),
            *self.d2h.lock().unwrap(),
            *self.syncs.lock().unwrap(),
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
        *self.d2h.lock().unwrap() += 1;
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
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        *self.launches.lock().unwrap() += 1;
        Ok(())
    }
    fn synchronize(&self, _stream: u64) -> Result<()> {
        *self.syncs.lock().unwrap() += 1;
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, _module: &str, _name: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(1))
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
}

fn tensor(cursor: &mut u64, dims: &[u64], kind: GgmlType) -> GgufDeviceTensor {
    let plan = GgmlIqMmqPlan::new(kind, 1, dims[1] as u32, dims[0] as u32).unwrap();
    let ptr = DevicePtr(*cursor);
    *cursor += plan.weight_bytes as u64 + 0x1_0000;
    GgufDeviceTensor {
        ptr,
        dimensions: dims.to_vec(),
        ggml_type: kind,
        byte_len: plan.weight_bytes,
        alloc_bytes: spark_runtime::weights::gguf::mmq_tensor_alloc_bytes(
            kind,
            &dims,
            plan.weight_bytes,
        )
        .expect("test tensor slack"),
    }
}

fn weights(intermediate: u64) -> Glm53DenseFfnWeights {
    let mut cursor = 0x50_0000_0000u64;
    Glm53DenseFfnWeights {
        gate: Glm53GgufMatrix::new(&tensor(&mut cursor, &[4096, intermediate], GgmlType::Q5_K))
            .unwrap(),
        up: Glm53GgufMatrix::new(&tensor(&mut cursor, &[4096, intermediate], GgmlType::Q5_K))
            .unwrap(),
        down: Glm53GgufMatrix::new(&tensor(&mut cursor, &[intermediate, 4096], GgmlType::Q6_K))
            .unwrap(),
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

fn hidden(gpu: &dyn GpuBackend) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: gpu.alloc(8_192).unwrap(),
        bytes: 8_192,
    }
}

/// The dense stack is three matmuls and one SwiGLU — seven launches, since each
/// matmul quantizes its activation first — with **no** host
/// copy and no stream sync — unlike the serial MoE path, which must read its
/// routed expert ids back to the host.
#[test]
fn dense_layer_is_seven_launches_with_no_host_copy_or_sync() {
    let gpu = TraceGpu::new();
    let kernels = Glm53DenseFfnKernels::load(&gpu).unwrap();
    let s = scratch(&gpu);
    let (input, output) = (hidden(&gpu), hidden(&gpu));

    let launches = kernels
        .execute(&gpu, &weights(12_288), input, s.dense_buffers(), output, 7)
        .unwrap();

    assert_eq!(launches, GLM53_DENSE_FFN_KERNEL_LAUNCHES);
    assert_eq!(
        gpu.counts(),
        (GLM53_DENSE_FFN_KERNEL_LAUNCHES as usize, 0, 0),
        "dense FFN must not copy to host or synchronize"
    );
    assert_eq!(GLM53_DENSE_FFN_MATMULS, 3);
}

/// A checkpoint whose dense layer is not 4096 -> 12288 -> 4096 must fail before
/// any kernel runs. Running a SwiGLU over the wrong width would produce fluent,
/// wrong output rather than an error.
#[test]
fn a_wrong_intermediate_width_is_refused_before_any_launch() {
    let gpu = TraceGpu::new();
    let kernels = Glm53DenseFfnKernels::load(&gpu).unwrap();
    let s = scratch(&gpu);
    let (input, output) = (hidden(&gpu), hidden(&gpu));

    // 2,048 is the MoE expert width, not the dense width.
    let error = kernels
        .execute(&gpu, &weights(2_048), input, s.dense_buffers(), output, 7)
        .expect_err("the MoE expert width must not be accepted as dense");
    let message = format!("{error:#}");
    assert!(message.contains("expected [4096, 12288]"), "{message}");
    assert_eq!(gpu.counts(), (0, 0, 0), "a refused layer must not launch");
}

/// Every quant type the pinned GLM schema can put on a dense layer must resolve
/// to a kernel; anything else must fail closed rather than pick a wrong one.
#[test]
fn every_schema_quant_resolves_and_unsupported_types_fail_closed() {
    let gpu = TraceGpu::new();
    let kernels = Glm53DenseFfnKernels::load(&gpu).unwrap();
    for kind in [
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q8_0,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::IQ2_S,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
        GgmlType::IQ4_XS,
    ] {
        assert!(kernels.mmq(kind).is_ok(), "{kind:?} must resolve");
    }
    for kind in [GgmlType::IQ1_S, GgmlType::IQ1_M, GgmlType::Q4_0] {
        assert!(kernels.mmq(kind).is_err(), "{kind:?} must fail closed");
    }
}
