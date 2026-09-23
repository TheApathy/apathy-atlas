// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::gguf::GgufDeviceTensor;

use super::*;
use crate::layers::ops::GgmlIqMmqPlan;
use crate::model::glm53::arena::{GLM53_T1_TRANSIENT_BYTES, Glm53ArenaPlan};
use crate::model::glm53::t1_state_transaction::Glm53T1StateLayout;
use crate::model::glm53::walk_scratch::Glm53WalkScratch;
use crate::weight_loader::{
    GLM53_MAX_CONTEXT_TOKENS, Glm53ContextPlan, Glm53DsaStorage, Glm53GgufF32,
};

const ARENA: u64 = 0x1_0000_0000;
const CONV_STATE_BYTES: usize = 3 * 8_192 * 4 * 4;

/// Records launches by symbol and keeps every kernel's first pointer argument,
/// so a test can prove which buffer the recurrence was handed.
struct TraceGpu {
    inner: MockGpuBackend,
    symbols: Mutex<HashMap<u64, String>>,
    next: Mutex<u64>,
    launched: Mutex<Vec<(String, u64)>>,
    copies: Mutex<Vec<(u64, u64, usize)>>,
}

impl TraceGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            symbols: Mutex::new(HashMap::new()),
            next: Mutex::new(1),
            launched: Mutex::new(Vec::new()),
            copies: Mutex::new(Vec::new()),
        }
    }
    fn launches(&self) -> Vec<(String, u64)> {
        self.launched.lock().unwrap().clone()
    }
    fn first_arg_of(&self, symbol: &str) -> Option<u64> {
        self.launches()
            .into_iter()
            .find(|(name, _)| name == symbol)
            .map(|(_, arg)| arg)
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
        self.copies.lock().unwrap().push((s.0, d.0, b));
        Ok(())
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
        // SAFETY: the launch builder wrote a DevicePtr-sized first argument.
        let first = if params.is_empty() {
            0
        } else {
            unsafe { *(params[0] as *const u64) }
        };
        self.launched.lock().unwrap().push((name, first));
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
        if let Some((handle, _)) = symbols.iter().find(|(_, k)| k.as_str() == name) {
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

fn matrix_tensor(cursor: &mut u64, dims: &[u64], kind: GgmlType) -> GgufDeviceTensor {
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

fn f32_tensor(cursor: &mut u64, dims: &[u64]) -> GgufDeviceTensor {
    let elements: u64 = dims.iter().product();
    let ptr = DevicePtr(*cursor);
    *cursor += elements * 4 + 0x1_0000;
    GgufDeviceTensor {
        ptr,
        dimensions: dims.to_vec(),
        ggml_type: GgmlType::F32,
        byte_len: (elements * 4) as usize,
        // F32 never reaches the MMQ weight path, so the slack is zero -- routed
        // through the same helper the loader uses rather than hardcoding 0, so
        // this fixture stays honest if the tensor's type ever changes.
        alloc_bytes: spark_runtime::weights::gguf::mmq_tensor_alloc_bytes(
            GgmlType::F32,
            dims,
            (elements * 4) as usize,
        )
        .expect("F32 fixture slack"),
    }
}

fn weights(qkv: u64) -> Glm53KdaWeights {
    let mut c = 0x80_0000_0000u64;
    Glm53KdaWeights {
        q: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, qkv], GgmlType::Q5_K)).unwrap(),
        k: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, qkv], GgmlType::Q5_K)).unwrap(),
        v: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, qkv], GgmlType::Q5_K)).unwrap(),
        output: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[8192, 4096], GgmlType::Q5_K))
            .unwrap(),
        conv_q: Glm53GgufF32::new(&f32_tensor(&mut c, &[4, 1, 8192]), &[4, 1, 8192]).unwrap(),
        conv_k: Glm53GgufF32::new(&f32_tensor(&mut c, &[4, 1, 8192]), &[4, 1, 8192]).unwrap(),
        conv_v: Glm53GgufF32::new(&f32_tensor(&mut c, &[4, 1, 8192]), &[4, 1, 8192]).unwrap(),
        a: Glm53GgufF32::new(&f32_tensor(&mut c, &[64]), &[64]).unwrap(),
        beta: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, 64], GgmlType::Q8_0)).unwrap(),
        dt_bias: Glm53GgufF32::new(&f32_tensor(&mut c, &[8192]), &[8192]).unwrap(),
        f_a: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, 128], GgmlType::Q8_0)).unwrap(),
        f_b: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[128, 8192], GgmlType::Q8_0)).unwrap(),
        g_a: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[4096, 128], GgmlType::Q8_0)).unwrap(),
        g_b: Glm53GgufMatrix::new(&matrix_tensor(&mut c, &[128, 8192], GgmlType::Q8_0)).unwrap(),
        norm: Glm53GgufF32::new(&f32_tensor(&mut c, &[128]), &[128]).unwrap(),
    }
}

fn conv_slots(gpu: &dyn GpuBackend) -> Glm53KdaConvSlots {
    let at = |bytes: usize| GgmlIqBuffer {
        ptr: gpu.alloc(bytes).unwrap(),
        bytes,
    };
    Glm53KdaConvSlots {
        persistent_state_f32: at(CONV_STATE_BYTES),
        staged_state_f32: at(CONV_STATE_BYTES),
        published_ends_u32: at(4),
        published_nonces_u64: at(8),
        logical_lengths_u32: at(4),
    }
}

fn scratch_state() -> Glm53KdaScratchState {
    let context =
        Glm53ContextPlan::new(1, GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage::BF16).unwrap();
    let layout = Glm53T1StateLayout::exact().unwrap();
    let t1 = Glm53ArenaPlan::exact_full_1m_b1()
        .unwrap()
        .t1_transaction
        .offset_bytes;
    Glm53KdaScratchState::stage(DevicePtr(ARENA), t1, &layout, &context, 3).unwrap()
}

fn walk_scratch(gpu: &dyn GpuBackend) -> Glm53WalkScratch {
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

/// **The load-bearing test.** The recurrence must be handed the staged scratch
/// buffer, never persistent state, and must be preceded by a prime copy in the
/// persistent -> scratch direction. Getting this backwards is the P0 that
/// motivates the whole binding.
#[test]
fn the_recurrence_runs_against_scratch_and_is_primed_from_persistent() {
    let gpu = TraceGpu::new();
    let kernels = Glm53KdaAttentionKernels::load(&gpu).unwrap();
    let s = walk_scratch(&gpu);
    let state = scratch_state();
    let (input, output) = (hidden(&gpu), hidden(&gpu));

    kernels
        .stage(
            &gpu,
            &weights(8_192),
            input,
            s.kda_buffers(),
            &state,
            conv_slots(&gpu),
            0,
            1_048_576,
            0xABCD,
            output,
            7,
        )
        .unwrap();

    let decode_state = gpu
        .first_arg_of("atlas_glm53_kda_decode")
        .expect("the recurrence must run");
    assert_eq!(
        decode_state,
        state.buffer().ptr.0,
        "the recurrence must mutate scratch"
    );
    assert_ne!(
        decode_state,
        state.persistent().ptr.0,
        "the recurrence must never be handed persistent state"
    );

    // The prime copy must precede it, persistent -> scratch.
    let copies = gpu.copies.lock().unwrap().clone();
    assert!(
        copies.contains(&(
            state.persistent().ptr.0,
            state.buffer().ptr.0,
            state.buffer().bytes
        )),
        "scratch must be primed from persistent before the recurrence"
    );
}

/// The launch budget is pinned so a dropped stage shows up as a count mismatch
/// rather than as silently missing work.
#[test]
fn a_layer_issues_the_exact_pinned_launch_budget() {
    let gpu = TraceGpu::new();
    let kernels = Glm53KdaAttentionKernels::load(&gpu).unwrap();
    let s = walk_scratch(&gpu);
    let state = scratch_state();
    let (input, output) = (hidden(&gpu), hidden(&gpu));

    let launches = kernels
        .stage(
            &gpu,
            &weights(8_192),
            input,
            s.kda_buffers(),
            &state,
            conv_slots(&gpu),
            0,
            1_048_576,
            0xABCD,
            output,
            7,
        )
        .unwrap();

    assert_eq!(launches, GLM53_KDA_KERNEL_LAUNCHES);
    assert_eq!(gpu.launches().len(), GLM53_KDA_KERNEL_LAUNCHES as usize);

    // Every stage of the reference sequence must appear.
    let symbols: Vec<String> = gpu.launches().into_iter().map(|(name, _)| name).collect();
    for required in [
        "atlas_glm53_kda_conv_f32_stage",
        "atlas_glm53_kda_conv_finalize",
        "atlas_glm53_kda_forget_gate",
        "atlas_glm53_kda_beta_sigmoid",
        "atlas_glm53_kda_decode",
        "atlas_glm53_kda_gated_rms_norm",
    ] {
        assert!(
            symbols.iter().any(|s| s == required),
            "missing {required} in {symbols:?}"
        );
    }
}

/// A checkpoint whose projections disagree with the pinned geometry must fail
/// before anything is enqueued — including before the prime copy.
#[test]
fn a_wrong_projection_width_is_refused_before_any_effect() {
    let gpu = TraceGpu::new();
    let kernels = Glm53KdaAttentionKernels::load(&gpu).unwrap();
    let s = walk_scratch(&gpu);
    let state = scratch_state();
    let (input, output) = (hidden(&gpu), hidden(&gpu));

    let error = kernels
        .stage(
            &gpu,
            &weights(4_096),
            input,
            s.kda_buffers(),
            &state,
            conv_slots(&gpu),
            0,
            1_048_576,
            0xABCD,
            output,
            7,
        )
        .expect_err("a 4096-wide QKV must not be accepted");
    assert!(format!("{error:#}").contains("expected [4096, 8192]"));
    assert!(gpu.launches().is_empty(), "a refused layer must not launch");
    assert!(
        gpu.copies.lock().unwrap().is_empty(),
        "a refused layer must not even prime"
    );
}

/// A zero nonce is not a valid transaction and must be refused.
#[test]
fn a_zero_transaction_nonce_is_refused() {
    let gpu = TraceGpu::new();
    let kernels = Glm53KdaAttentionKernels::load(&gpu).unwrap();
    let s = walk_scratch(&gpu);
    let state = scratch_state();
    let (input, output) = (hidden(&gpu), hidden(&gpu));

    assert!(
        kernels
            .stage(
                &gpu,
                &weights(8_192),
                input,
                s.kda_buffers(),
                &state,
                conv_slots(&gpu),
                0,
                1_048_576,
                0,
                output,
                7,
            )
            .is_err()
    );
}
