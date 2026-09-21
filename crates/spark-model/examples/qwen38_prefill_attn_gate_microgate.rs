// SPDX-License-Identifier: AGPL-3.0-only

//! Raw fail-closed gate for dense-Qwen3.8 chunk-0 BR64 attention + sigmoid gate.
//!
//! The candidate is compared byte-for-byte with the shipped BR64 attention
//! followed by `sigmoid_gate_mul_batched`. Inputs are immutable, outputs carry
//! distinct unwritten sentinels, and every allocation has 4-KiB redzones.
//! Timing is permitted only after the complete parity matrix succeeds.
//!
//! ```text
//! ATLAS_PREFILL_ATTN_GATE_MICROGATE_FULL=1 \
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --features cuda \
//!   --example qwen38_prefill_attn_gate_microgate
//! ```
//!
//! Add `ATLAS_PREFILL_ATTN_GATE_MICROGATE_TIMING=1` to require alternating
//! median and p90 wins at 2K and 8K. Never run this on an unreserved device.

use std::ffi::{CStr, c_char, c_void};

use anyhow::{Context, Result, bail, ensure};
use spark_model::layers::ops::{
    prefill_attention_64, prefill_attention_64_gate_fused, sigmoid_gate_mul_batched,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const NQ: u32 = 24;
const NKV: u32 = 4;
const HD: u32 = 256;
const GATE_STRIDE: u32 = 12_288;
const OUTPUT_DIM: u32 = NQ * HD;
const REDZONE: usize = 4 * 1024;
const TIMING_LAUNCHES_PER_SAMPLE: usize = 8;
const ATTR_MULTIPROCESSOR_COUNT: u32 = 16;
const ATTR_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
const ATTR_COMPUTE_CAPABILITY_MINOR: u32 = 76;
const GB10_SMS: i32 = 48;
const FUNC_ATTR_MAX_THREADS_PER_BLOCK: u32 = 0;
const FUNC_ATTR_SHARED_SIZE_BYTES: u32 = 1;
const FUNC_ATTR_LOCAL_SIZE_BYTES: u32 = 3;
const FUNC_ATTR_NUM_REGS: u32 = 4;

unsafe extern "C" {
    fn cuCtxGetDevice(device: *mut i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attribute: u32, device: i32) -> i32;
    fn cuDeviceGetName(name: *mut c_char, length: i32, device: i32) -> i32;
    fn cuFuncGetAttribute(value: *mut i32, attribute: u32, function: *mut c_void) -> i32;
    fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> i32;
    fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cuEventSynchronize(event: *mut c_void) -> i32;
    fn cuEventElapsedTime(milliseconds: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    #[link_name = "cuEventDestroy_v2"]
    fn cuEventDestroy(event: *mut c_void) -> i32;
}

const ATTN_FINITE: [u16; 16] = [
    0xc080, 0xc000, 0xbf80, 0xbf00, 0xbe80, 0x8080, 0x8000, 0x0000, 0x0080, 0x3d00, 0x3e80, 0x3f00,
    0x3f80, 0x4000, 0x4040, 0x4080,
];
const ATTN_CANCELLATION: [u16; 16] = [
    0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x3e80, 0xbe80, 0x3d00, 0xbd00, 0x3c00, 0xbc00, 0x0080, 0x8080,
    0x0001, 0x8001, 0x0000, 0x8000,
];
const GATE_FINITE: [u16; 16] = [
    0xc180, 0xc100, 0xc000, 0xbf80, 0xbf00, 0xbe80, 0x8000, 0x0000, 0x3e80, 0x3f00, 0x3f80, 0x4000,
    0x4040, 0x4100, 0x4180, 0x7f7f,
];
const GATE_EXTREME: [u16; 12] = [
    0xff80, 0x7f80, 0xff7f, 0x7f7f, 0xffc1, 0x7fc1, 0x8000, 0x0000, 0x8001, 0x0001, 0xc180, 0x4180,
];

#[derive(Clone, Copy, Debug)]
enum Fixture {
    Finite,
    Cancellation,
    ExtremeGate,
}

impl Fixture {
    const fn label(self) -> &'static str {
        match self {
            Self::Finite => "finite",
            Self::Cancellation => "cancellation",
            Self::ExtremeGate => "extreme-gate",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    tokens: u32,
    sliding_window: u32,
    fixture: Fixture,
}

impl Case {
    fn label(self) -> String {
        format!(
            "n{}_win{}_{}",
            self.tokens,
            self.sliding_window,
            self.fixture.label()
        )
    }
}

struct Kernels {
    attention: KernelHandle,
    gate: KernelHandle,
    fused: KernelHandle,
}

struct DeviceIdentity {
    ordinal: i32,
    name: String,
    major: i32,
    minor: i32,
    multiprocessors: i32,
}

struct CudaEventPair {
    start: *mut c_void,
    end: *mut c_void,
}

#[derive(Debug)]
struct FunctionResources {
    max_threads: i32,
    shared_bytes: i32,
    local_bytes: i32,
    registers: i32,
}

fn function_resources(kernel: KernelHandle) -> Result<FunctionResources> {
    let function = kernel.0 as usize as *mut c_void;
    let attribute = |kind: u32, label: &str| -> Result<i32> {
        let mut value = -1;
        let status = unsafe { cuFuncGetAttribute(&mut value, kind, function) };
        ensure!(
            status == 0 && value >= 0,
            "cuFuncGetAttribute({label}) failed: status={status} value={value}"
        );
        Ok(value)
    };
    Ok(FunctionResources {
        max_threads: attribute(FUNC_ATTR_MAX_THREADS_PER_BLOCK, "MAX_THREADS_PER_BLOCK")?,
        shared_bytes: attribute(FUNC_ATTR_SHARED_SIZE_BYTES, "SHARED_SIZE_BYTES")?,
        local_bytes: attribute(FUNC_ATTR_LOCAL_SIZE_BYTES, "LOCAL_SIZE_BYTES")?,
        registers: attribute(FUNC_ATTR_NUM_REGS, "NUM_REGS")?,
    })
}

impl CudaEventPair {
    fn new() -> Result<Self> {
        let mut start = std::ptr::null_mut();
        let mut end = std::ptr::null_mut();
        let start_status = unsafe { cuEventCreate(&mut start, 0) };
        ensure!(
            start_status == 0,
            "cuEventCreate(start) failed: {start_status}"
        );
        let end_status = unsafe { cuEventCreate(&mut end, 0) };
        if end_status != 0 {
            let _ = unsafe { cuEventDestroy(start) };
            bail!("cuEventCreate(end) failed: {end_status}");
        }
        Ok(Self { start, end })
    }

    fn measure(&self, stream: u64, launch: impl FnOnce() -> Result<()>) -> Result<f64> {
        let stream = stream as usize as *mut c_void;
        let status = unsafe { cuEventRecord(self.start, stream) };
        ensure!(status == 0, "cuEventRecord(start) failed: {status}");
        launch()?;
        let status = unsafe { cuEventRecord(self.end, stream) };
        ensure!(status == 0, "cuEventRecord(end) failed: {status}");
        let status = unsafe { cuEventSynchronize(self.end) };
        ensure!(status == 0, "cuEventSynchronize(end) failed: {status}");
        let mut milliseconds = 0.0f32;
        let status = unsafe { cuEventElapsedTime(&mut milliseconds, self.start, self.end) };
        ensure!(status == 0, "cuEventElapsedTime failed: {status}");
        Ok(f64::from(milliseconds))
    }

    fn destroy(self) -> Result<()> {
        let start_status = unsafe { cuEventDestroy(self.start) };
        let end_status = unsafe { cuEventDestroy(self.end) };
        ensure!(
            start_status == 0,
            "cuEventDestroy(start) failed: {start_status}"
        );
        ensure!(end_status == 0, "cuEventDestroy(end) failed: {end_status}");
        Ok(())
    }
}

fn exact_kernel_bundle() -> Result<(String, Vec<(&'static str, &'static str)>)> {
    let model = std::env::var("ATLAS_TARGET_MODEL")
        .context("ATLAS_TARGET_MODEL must be set for raw qualification")?;
    let quant = std::env::var("ATLAS_TARGET_QUANT")
        .context("ATLAS_TARGET_QUANT must be set for raw qualification")?;
    ensure!(
        model == "qwen3.8-27b" && quant == "nvfp4",
        "raw qualification requires ATLAS_TARGET_MODEL=qwen3.8-27b and \
         ATLAS_TARGET_QUANT=nvfp4"
    );
    let mut matches: Vec<_> = atlas_kernels::available_targets()
        .into_iter()
        .filter(|set| {
            set.target.arch == "sm_121"
                && set.target.model == "qwen3.8-27b"
                && set.target.quant == "nvfp4"
        })
        .collect();
    ensure!(
        matches.len() == 1,
        "embedded kernel bundle must contain exactly one exact \
         (sm_121,qwen3.8-27b,nvfp4) target, found {}",
        matches.len()
    );
    let matched = matches.pop().context("exact target disappeared")?;
    Ok((matched.target.to_string(), matched.modules))
}

fn current_device_identity() -> Result<DeviceIdentity> {
    let mut ordinal = -1;
    let status = unsafe { cuCtxGetDevice(&mut ordinal) };
    ensure!(
        status == 0 && ordinal >= 0,
        "cuCtxGetDevice failed: {status}"
    );

    let attribute = |kind: u32, label: &str| -> Result<i32> {
        let mut value = -1;
        let status = unsafe { cuDeviceGetAttribute(&mut value, kind, ordinal) };
        ensure!(
            status == 0 && value >= 0,
            "cuDeviceGetAttribute({label}) failed: status={status} value={value}"
        );
        Ok(value)
    };
    let major = attribute(ATTR_COMPUTE_CAPABILITY_MAJOR, "CC_MAJOR")?;
    let minor = attribute(ATTR_COMPUTE_CAPABILITY_MINOR, "CC_MINOR")?;
    let multiprocessors = attribute(ATTR_MULTIPROCESSOR_COUNT, "MULTIPROCESSOR_COUNT")?;
    let mut name = [0 as c_char; 256];
    let status = unsafe { cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, ordinal) };
    ensure!(status == 0, "cuDeviceGetName failed: {status}");
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    ensure!(
        major == 12
            && minor == 1
            && multiprocessors == GB10_SMS
            && name.to_ascii_uppercase().contains("GB10"),
        "raw qualification requires GB10 SM121 with 48 SMs, got \
         name={name:?} cc={major}.{minor} sms={multiprocessors}"
    );
    Ok(DeviceIdentity {
        ordinal,
        name,
        major,
        minor,
        multiprocessors,
    })
}

struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    image: Vec<u8>,
    redzone_byte: u8,
}

impl Guarded {
    fn input(gpu: &dyn GpuBackend, payload: &[u8], redzone_byte: u8) -> Result<Self> {
        let mut image = vec![redzone_byte; REDZONE + payload.len() + REDZONE];
        image[REDZONE..REDZONE + payload.len()].copy_from_slice(payload);
        let allocation = gpu.alloc(image.len())?;
        gpu.copy_h2d(&image, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            image,
            redzone_byte,
        })
    }

    fn output(
        gpu: &dyn GpuBackend,
        payload_len: usize,
        payload_byte: u8,
        redzone_byte: u8,
    ) -> Result<Self> {
        Self::input(gpu, &vec![payload_byte; payload_len], redzone_byte)
    }

    fn ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.image, self.allocation)
    }

    fn reset_payload_fill(&self, gpu: &dyn GpuBackend, payload_byte: u8) -> Result<()> {
        let mut image = self.image.clone();
        image[REDZONE..REDZONE + self.payload_len].fill(payload_byte);
        gpu.copy_h2d(&image, self.allocation)
    }

    fn read_all(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut actual = vec![0; self.image.len()];
        gpu.copy_d2h(self.allocation, &mut actual)?;
        Ok(actual)
    }

    fn payload(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        let actual = self.read_all(gpu)?;
        ensure!(
            actual[..REDZONE]
                .iter()
                .all(|&byte| byte == self.redzone_byte),
            "{label}: leading 4-KiB redzone changed"
        );
        let suffix = REDZONE + self.payload_len;
        ensure!(
            actual[suffix..]
                .iter()
                .all(|&byte| byte == self.redzone_byte),
            "{label}: trailing 4-KiB redzone changed"
        );
        Ok(actual[REDZONE..suffix].to_vec())
    }

    fn immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        ensure!(self.read_all(gpu)? == self.image, "{label}: input changed");
        Ok(())
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}

fn bf16_pattern(count: usize, salt: usize, fixture: Fixture) -> Vec<u8> {
    let pattern = match fixture {
        Fixture::Cancellation => &ATTN_CANCELLATION[..],
        Fixture::Finite | Fixture::ExtremeGate => &ATTN_FINITE[..],
    };
    let mut bytes = vec![0; count * 2];
    for (index, chunk) in bytes.chunks_exact_mut(2).enumerate() {
        let word = pattern[(index.wrapping_mul(17).wrapping_add(salt)) % pattern.len()];
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn gate_fixture(case: Case) -> Vec<u8> {
    let words = case.tokens as usize * GATE_STRIDE as usize;
    let mut bytes = vec![0x5a; words * 2];
    for token in 0..case.tokens as usize {
        let row = token * GATE_STRIDE as usize;
        for dim in 0..OUTPUT_DIM as usize {
            let word = GATE_FINITE
                [(dim.wrapping_mul(13).wrapping_add(token.wrapping_mul(29))) % GATE_FINITE.len()];
            let byte = (row + dim) * 2;
            bytes[byte..byte + 2].copy_from_slice(&word.to_le_bytes());
        }
    }
    if matches!(case.fixture, Fixture::ExtremeGate) {
        let boundary_dims = [
            0usize,
            1,
            HD as usize - 1,
            HD as usize,
            HD as usize + 1,
            OUTPUT_DIM as usize / 2 - 1,
            OUTPUT_DIM as usize / 2,
            OUTPUT_DIM as usize - 2,
            OUTPUT_DIM as usize - 1,
        ];
        for (index, word) in GATE_EXTREME.into_iter().enumerate() {
            let token = match index % 4 {
                0 => 0,
                1 => 1.min(case.tokens as usize - 1),
                2 => case.tokens as usize / 2,
                _ => case.tokens as usize - 1,
            };
            let dim = boundary_dims[index % boundary_dims.len()];
            let byte = (token * GATE_STRIDE as usize + dim) * 2;
            bytes[byte..byte + 2].copy_from_slice(&word.to_le_bytes());
        }
    }
    bytes
}

#[allow(clippy::too_many_arguments)]
fn launch_parent(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    case: Case,
) -> Result<()> {
    prefill_attention_64(
        gpu,
        kernels.attention,
        q,
        k,
        v,
        output,
        case.tokens,
        1,
        NQ,
        NKV,
        HD,
        1.0 / (HD as f32).sqrt(),
        true,
        case.sliding_window,
        stream,
    )?;
    sigmoid_gate_mul_batched(
        gpu,
        kernels.gate,
        output,
        gate,
        output,
        OUTPUT_DIM,
        GATE_STRIDE,
        case.tokens,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_fused(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    case: Case,
) -> Result<()> {
    prefill_attention_64_gate_fused(
        gpu,
        kernels.fused,
        q,
        k,
        v,
        output,
        gate,
        case.tokens,
        1,
        NQ,
        NKV,
        HD,
        1.0 / (HD as f32).sqrt(),
        true,
        case.sliding_window,
        GATE_STRIDE,
        stream,
    )
}

fn require_equal(label: &str, parent: &[u8], candidate: &[u8]) -> Result<()> {
    if parent != candidate {
        let index = parent
            .iter()
            .zip(candidate)
            .position(|(left, right)| left != right)
            .context("mismatch without an index")?;
        let word = index / 2;
        let token = word / OUTPUT_DIM as usize;
        let dim = word % OUTPUT_DIM as usize;
        let head = dim / HD as usize;
        let column = dim % HD as usize;
        bail!(
            "{label}: byte {index} (token={token} head={head} column={column} lane={}) \
             differs: parent=0x{:02x} candidate=0x{:02x}; checksums parent={:016x} \
             candidate={:016x}",
            index % 2,
            parent[index],
            candidate[index],
            fnv1a64(parent),
            fnv1a64(candidate),
        );
    }
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn bf16_word(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])
}

fn bf16_to_f32(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

fn run_case(gpu: &dyn GpuBackend, stream: u64, kernels: &Kernels, case: Case) -> Result<()> {
    let label = case.label();
    let q_words = case.tokens as usize * NQ as usize * HD as usize;
    let kv_words = case.tokens as usize * NKV as usize * HD as usize;
    let output_bytes = q_words * 2;
    let q = Guarded::input(gpu, &bf16_pattern(q_words, 3, case.fixture), 0x91)?;
    let k = Guarded::input(gpu, &bf16_pattern(kv_words, 7, case.fixture), 0x92)?;
    let v = Guarded::input(gpu, &bf16_pattern(kv_words, 11, case.fixture), 0x93)?;
    let gate = Guarded::input(gpu, &gate_fixture(case), 0x94)?;
    let parent = Guarded::output(gpu, output_bytes, 0x35, 0xa5)?;
    let candidate = Guarded::output(gpu, output_bytes, 0xca, 0x5a)?;

    launch_parent(
        gpu,
        stream,
        kernels,
        q.ptr(),
        k.ptr(),
        v.ptr(),
        gate.ptr(),
        parent.ptr(),
        case,
    )?;
    launch_fused(
        gpu,
        stream,
        kernels,
        q.ptr(),
        k.ptr(),
        v.ptr(),
        gate.ptr(),
        candidate.ptr(),
        case,
    )?;
    gpu.synchronize(stream)?;

    let parent_bytes = parent.payload(gpu, &format!("{label}/parent"))?;
    let candidate_bytes = candidate.payload(gpu, &format!("{label}/candidate"))?;
    if parent_bytes != candidate_bytes {
        let mismatch_byte = parent_bytes
            .iter()
            .zip(&candidate_bytes)
            .position(|(left, right)| left != right)
            .context("mismatch without an index")?;
        let mismatch_word = mismatch_byte / 2;
        let token = mismatch_word / OUTPUT_DIM as usize;
        let dim = mismatch_word % OUTPUT_DIM as usize;
        let attention = Guarded::output(gpu, output_bytes, 0x6d, 0x3c)?;
        prefill_attention_64(
            gpu,
            kernels.attention,
            q.ptr(),
            k.ptr(),
            v.ptr(),
            attention.ptr(),
            case.tokens,
            1,
            NQ,
            NKV,
            HD,
            1.0 / (HD as f32).sqrt(),
            true,
            case.sliding_window,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let attention_bytes = attention.payload(gpu, &format!("{label}/attention-only"))?;
        let gate_bytes = gate.payload(gpu, &format!("{label}/gate-diagnostic"))?;
        let attention_word = bf16_word(&attention_bytes, mismatch_word);
        let gate_word = bf16_word(&gate_bytes, token * GATE_STRIDE as usize + dim);
        let parent_word = bf16_word(&parent_bytes, mismatch_word);
        let candidate_word = bf16_word(&candidate_bytes, mismatch_word);
        let x = bf16_to_f32(attention_word);
        let g = bf16_to_f32(gate_word);
        eprintln!(
            "DIAGNOSTIC {label}: word={mismatch_word} token={token} dim={dim} \
             attention=0x{attention_word:04x}({x:?}) gate=0x{gate_word:04x}({g:?}) \
             parent=0x{parent_word:04x}({:?}) candidate=0x{candidate_word:04x}({:?}) \
             host_f32={:?}",
            bf16_to_f32(parent_word),
            bf16_to_f32(candidate_word),
            x * (1.0 / (1.0 + (-g).exp())),
        );
        attention.free(gpu)?;
    }
    require_equal(&label, &parent_bytes, &candidate_bytes)?;

    // Repeat in reverse launch order from a third/fourth distinct payload
    // fill. Equal first and second results prove neither path accidentally
    // inherits an unwritten byte from its initialization.
    parent.reset_payload_fill(gpu, 0xe7)?;
    candidate.reset_payload_fill(gpu, 0x19)?;
    launch_fused(
        gpu,
        stream,
        kernels,
        q.ptr(),
        k.ptr(),
        v.ptr(),
        gate.ptr(),
        candidate.ptr(),
        case,
    )?;
    launch_parent(
        gpu,
        stream,
        kernels,
        q.ptr(),
        k.ptr(),
        v.ptr(),
        gate.ptr(),
        parent.ptr(),
        case,
    )?;
    gpu.synchronize(stream)?;
    let parent_replay = parent.payload(gpu, &format!("{label}/parent-replay"))?;
    let candidate_replay = candidate.payload(gpu, &format!("{label}/candidate-replay"))?;
    require_equal(
        &format!("{label}/replay"),
        &parent_replay,
        &candidate_replay,
    )?;
    require_equal(
        &format!("{label}/parent-fill-independence"),
        &parent_bytes,
        &parent_replay,
    )?;
    require_equal(
        &format!("{label}/candidate-fill-independence"),
        &candidate_bytes,
        &candidate_replay,
    )?;
    q.immutable(gpu, &format!("{label}/Q"))?;
    k.immutable(gpu, &format!("{label}/K"))?;
    v.immutable(gpu, &format!("{label}/V"))?;
    gate.immutable(gpu, &format!("{label}/gate"))?;
    println!(
        "PASS {label}: {} parent+gate/fused bytes identical across distinct-fill replay; checksum={:016x}",
        parent_bytes.len(),
        fnv1a64(&parent_bytes)
    );

    q.free(gpu)?;
    k.free(gpu)?;
    v.free(gpu)?;
    gate.free(gpu)?;
    parent.free(gpu)?;
    candidate.free(gpu)
}

fn percentile_90(sorted: &[f64]) -> f64 {
    sorted[(sorted.len() * 9).div_ceil(10).saturating_sub(1)]
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn time_case(gpu: &dyn GpuBackend, stream: u64, kernels: &Kernels, case: Case) -> Result<()> {
    let q_words = case.tokens as usize * NQ as usize * HD as usize;
    let kv_words = case.tokens as usize * NKV as usize * HD as usize;
    let q = Guarded::input(gpu, &bf16_pattern(q_words, 3, case.fixture), 0x91)?;
    let k = Guarded::input(gpu, &bf16_pattern(kv_words, 7, case.fixture), 0x92)?;
    let v = Guarded::input(gpu, &bf16_pattern(kv_words, 11, case.fixture), 0x93)?;
    let gate = Guarded::input(gpu, &gate_fixture(case), 0x94)?;
    let output = Guarded::output(gpu, q_words * 2, 0x35, 0xa5)?;
    let events = CudaEventPair::new()?;

    let measure_once = |candidate: bool| -> Result<f64> {
        output.reset(gpu)?;
        gpu.synchronize(stream)?;
        let total_ms = events.measure(stream, || {
            for _ in 0..TIMING_LAUNCHES_PER_SAMPLE {
                if candidate {
                    launch_fused(
                        gpu,
                        stream,
                        kernels,
                        q.ptr(),
                        k.ptr(),
                        v.ptr(),
                        gate.ptr(),
                        output.ptr(),
                        case,
                    )?;
                } else {
                    launch_parent(
                        gpu,
                        stream,
                        kernels,
                        q.ptr(),
                        k.ptr(),
                        v.ptr(),
                        gate.ptr(),
                        output.ptr(),
                        case,
                    )?;
                }
            }
            Ok(())
        })?;
        Ok(total_ms / TIMING_LAUNCHES_PER_SAMPLE as f64)
    };

    for round in 0..3 {
        if round % 2 == 0 {
            let _ = measure_once(false)?;
            let _ = measure_once(true)?;
        } else {
            let _ = measure_once(true)?;
            let _ = measure_once(false)?;
        }
    }
    let mut parent = Vec::with_capacity(21);
    let mut candidate = Vec::with_capacity(21);
    let mut paired_savings = Vec::with_capacity(21);
    for round in 0..21 {
        let (parent_ms, candidate_ms) = if round % 2 == 0 {
            (measure_once(false)?, measure_once(true)?)
        } else {
            let candidate_ms = measure_once(true)?;
            let parent_ms = measure_once(false)?;
            (parent_ms, candidate_ms)
        };
        parent.push(parent_ms);
        candidate.push(candidate_ms);
        paired_savings.push(parent_ms - candidate_ms);
    }
    parent.sort_by(f64::total_cmp);
    candidate.sort_by(f64::total_cmp);
    let parent_median = parent[parent.len() / 2];
    let candidate_median = candidate[candidate.len() / 2];
    let parent_p90 = percentile_90(&parent);
    let candidate_p90 = percentile_90(&candidate);
    let median_saving = median(paired_savings.clone());
    let mad = median(
        paired_savings
            .iter()
            .map(|saving| (saving - median_saving).abs())
            .collect(),
    );
    let conservative_saving = median_saving - 3.0 * mad;
    let aggregate_saving = median_saving * 16.0;
    let required_aggregate_saving = if case.tokens == 2_048 { 0.5 } else { 2.0 };
    println!(
        "TIMING {} batch={TIMING_LAUNCHES_PER_SAMPLE}: median parent={parent_median:.3} ms fused={candidate_median:.3} ms; \
         p90 parent={parent_p90:.3} ms fused={candidate_p90:.3} ms; paired median \
         saving={median_saving:.3} ms MAD={mad:.3} ms conservative={conservative_saving:.3} ms; \
         modeled x16 saving={aggregate_saving:.3} ms",
        case.label()
    );
    ensure!(
        candidate_median < parent_median && candidate_p90 < parent_p90,
        "{}: fused route must beat parent at median and p90; stop promotion",
        case.label()
    );
    ensure!(
        conservative_saving > 0.0,
        "{}: paired median saving does not clear the 3*MAD noise bound; stop promotion",
        case.label()
    );
    ensure!(
        aggregate_saving >= required_aggregate_saving,
        "{}: modeled 16-layer saving {aggregate_saving:.3} ms is below the required \
         {required_aggregate_saving:.3} ms; stop promotion",
        case.label()
    );

    q.free(gpu)?;
    k.free(gpu)?;
    v.free(gpu)?;
    gate.free(gpu)?;
    output.free(gpu)?;
    events.destroy()
}

fn strict_switch(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => bail!("{name} must be exactly 0 or 1"),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("{name} must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

fn main() -> Result<()> {
    let full = strict_switch("ATLAS_PREFILL_ATTN_GATE_MICROGATE_FULL")?;
    let timing = strict_switch("ATLAS_PREFILL_ATTN_GATE_MICROGATE_TIMING")?;
    ensure!(
        !timing || full,
        "timing requires ATLAS_PREFILL_ATTN_GATE_MICROGATE_FULL=1"
    );

    let (target_identity, modules) = exact_kernel_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let device = current_device_identity()?;
    let stream = gpu.create_stream()?;
    let kernels = Kernels {
        attention: gpu.kernel("inferspark_prefill", "inferspark_prefill_64")?,
        gate: gpu.kernel("residual_add", "sigmoid_gate_mul_batched")?,
        fused: gpu.kernel(
            "inferspark_prefill_gate_fused",
            "inferspark_prefill_64_gate_fused",
        )?,
    };
    let parent_resources = function_resources(kernels.attention)?;
    let fused_resources = function_resources(kernels.fused)?;
    ensure!(
        parent_resources.max_threads >= 256 && fused_resources.max_threads >= 256,
        "BR64 kernels must admit 256 threads: parent={parent_resources:?} fused={fused_resources:?}"
    );
    ensure!(
        fused_resources.registers <= parent_resources.registers
            && fused_resources.shared_bytes == parent_resources.shared_bytes
            && fused_resources.local_bytes <= parent_resources.local_bytes,
        "fused kernel resource regression: parent={parent_resources:?} fused={fused_resources:?}"
    );
    println!("RESOURCES parent={parent_resources:?} fused={fused_resources:?}");

    let smoke = [
        Case {
            tokens: 64,
            sliding_window: 0,
            fixture: Fixture::Finite,
        },
        Case {
            tokens: 129,
            sliding_window: 33,
            fixture: Fixture::ExtremeGate,
        },
    ];
    let complete = [
        Case {
            tokens: 63,
            sliding_window: 0,
            fixture: Fixture::Finite,
        },
        Case {
            tokens: 64,
            sliding_window: 0,
            fixture: Fixture::ExtremeGate,
        },
        Case {
            tokens: 65,
            sliding_window: 1,
            fixture: Fixture::Finite,
        },
        Case {
            tokens: 127,
            sliding_window: 31,
            fixture: Fixture::Finite,
        },
        Case {
            tokens: 128,
            sliding_window: 32,
            fixture: Fixture::Finite,
        },
        Case {
            tokens: 129,
            sliding_window: 33,
            fixture: Fixture::Cancellation,
        },
        Case {
            tokens: 2_048,
            sliding_window: 0,
            fixture: Fixture::Finite,
        },
        Case {
            tokens: 2_048,
            sliding_window: 4_096,
            fixture: Fixture::Cancellation,
        },
        Case {
            tokens: 8_192,
            sliding_window: 0,
            fixture: Fixture::Finite,
        },
        Case {
            tokens: 8_192,
            sliding_window: 4_096,
            fixture: Fixture::Cancellation,
        },
    ];
    let cases = if full { &complete[..] } else { &smoke[..] };
    let matrix_bytes: Vec<u8> = cases
        .iter()
        .flat_map(|case| {
            format!(
                "{}:{}:{};",
                case.tokens,
                case.sliding_window,
                case.fixture.label()
            )
            .into_bytes()
        })
        .collect();
    let matrix_checksum = fnv1a64(&matrix_bytes);
    for &case in cases {
        run_case(gpu, stream, &kernels, case)?;
    }

    if timing {
        for tokens in [2_048, 8_192] {
            time_case(
                gpu,
                stream,
                &kernels,
                Case {
                    tokens,
                    sliding_window: 0,
                    fixture: Fixture::Finite,
                },
            )?;
        }
    }
    if full && timing {
        println!(
            "FULL PARITY+TIMING PASS: cases={}/{} matrix={matrix_checksum:016x} \
             target={} device={} ordinal={} cc={}.{} sms={}",
            cases.len(),
            complete.len(),
            target_identity,
            device.name,
            device.ordinal,
            device.major,
            device.minor,
            device.multiprocessors,
        );
    } else if full {
        println!(
            "FULL PARITY PASS — TIMING NOT RUN: cases={}/{} matrix={matrix_checksum:016x} \
             target={} device={} ordinal={} cc={}.{} sms={}",
            cases.len(),
            complete.len(),
            target_identity,
            device.name,
            device.ordinal,
            device.major,
            device.minor,
            device.multiprocessors,
        );
    } else {
        println!(
            "SMOKE PASS ONLY — NOT QUALIFIED: cases={}/{} matrix={matrix_checksum:016x} \
             target={} device={} ordinal={} cc={}.{} sms={}",
            cases.len(),
            complete.len(),
            target_identity,
            device.name,
            device.ordinal,
            device.major,
            device.minor,
            device.multiprocessors,
        );
    }
    Ok(())
}
