// SPDX-License-Identifier: AGPL-3.0-only

//! Raw qualification gate for Qwen3.8 large-M FFN dual-fusion candidates.
//!
//! Three routes must produce identical BF16 bytes before timing is allowed:
//! the ordinary two-pipe-plus-SiLU parent, the up-only fused incumbent, and
//! The default candidate is `w4a16_gemm_pipe_dual_warp8`; set
//! `ATLAS_PREFILL_FFN_DUAL_MICROGATE_CANDIDATE=n32` for the N32 shadow. Every
//! allocation is guarded,
//! all inputs are checked for mutation, and the fused branch is checked with
//! both its production NULL C1 and a guarded non-NULL C1 that must stay intact.
//!
//! ```text
//! ATLAS_PREFILL_FFN_DUAL_MICROGATE_FULL=1 \
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --features cuda \
//!   --example qwen38_prefill_ffn_dual_fused_microgate
//! ```
//!
//! Add `ATLAS_PREFILL_FFN_DUAL_MICROGATE_TIMING=1` for parity-gated balanced
//! three-arm timing at M=2048 and M=8192. Run only on a reserved GB10.

#[allow(dead_code)]
#[path = "w4a16_exact_lm_head_microtest/data.rs"]
mod data;

use std::ffi::{CStr, c_char, c_void};

use anyhow::{Context, Result, bail, ensure};
use data::{Fixture, as_le_bytes, cancellation_fixture, random_fixture};
use spark_model::layers::ops;
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const HIDDEN: usize = 5_120;
const INTERMEDIATE: usize = 17_408;
const REDZONE: usize = 4 * 1_024;
const FULL_ROWS: [usize; 9] = [33, 63, 64, 65, 127, 128, 129, 2_048, 8_192];
const TIMING_ROWS: [usize; 2] = [2_048, 8_192];
const TIMING_PERMUTATIONS: [[Route; 3]; 6] = [
    [Route::Parent, Route::UpOnly, Route::Candidate],
    [Route::Parent, Route::Candidate, Route::UpOnly],
    [Route::UpOnly, Route::Parent, Route::Candidate],
    [Route::UpOnly, Route::Candidate, Route::Parent],
    [Route::Candidate, Route::Parent, Route::UpOnly],
    [Route::Candidate, Route::UpOnly, Route::Parent],
];
const TIMING_PERMUTATION_CYCLES: usize = 3;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputKind {
    Random,
    Cancellation,
}

impl InputKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Cancellation => "cancellation",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Parent,
    UpOnly,
    Candidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    Warp8,
    N32,
}

impl Candidate {
    fn from_env() -> Result<Self> {
        let Some(raw) = std::env::var_os("ATLAS_PREFILL_FFN_DUAL_MICROGATE_CANDIDATE") else {
            return Ok(Self::Warp8);
        };
        let value = raw.into_string().map_err(|_| {
            anyhow::anyhow!(
                "ATLAS_PREFILL_FFN_DUAL_MICROGATE_CANDIDATE must be valid UTF-8 and one of warp8|n32"
            )
        })?;
        match value.as_str() {
            "warp8" => Ok(Self::Warp8),
            "n32" => Ok(Self::N32),
            _ => bail!(
                "ATLAS_PREFILL_FFN_DUAL_MICROGATE_CANDIDATE must be one of warp8|n32, got {value:?}"
            ),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Warp8 => "warp8",
            Self::N32 => "n32",
        }
    }

    const fn symbol(self) -> &'static str {
        match self {
            Self::Warp8 => "w4a16_gemm_pipe_dual_warp8",
            Self::N32 => "w4a16_gemm_pipe_dual_n32",
        }
    }

    const fn launch_threads(self) -> u32 {
        match self {
            Self::Warp8 => 256,
            Self::N32 => 128,
        }
    }

    const fn n_tile(self) -> u32 {
        match self {
            Self::Warp8 => 64,
            Self::N32 => 32,
        }
    }

    const fn shared_ceiling(self) -> i32 {
        match self {
            Self::Warp8 => 41_024,
            Self::N32 => 22_336,
        }
    }
}

#[derive(Debug)]
struct DeviceIdentity {
    ordinal: i32,
    name: String,
    major: i32,
    minor: i32,
    multiprocessors: i32,
}

#[derive(Debug)]
struct FunctionResources {
    max_threads: i32,
    shared_bytes: i32,
    local_bytes: i32,
    registers: i32,
}

struct CudaEventPair {
    start: *mut c_void,
    end: *mut c_void,
}

impl CudaEventPair {
    fn new() -> Result<Self> {
        let mut start = std::ptr::null_mut();
        let mut end = std::ptr::null_mut();
        let status = unsafe { cuEventCreate(&mut start, 0) };
        ensure!(status == 0, "cuEventCreate(start) failed: {status}");
        let status = unsafe { cuEventCreate(&mut end, 0) };
        if status != 0 {
            let _ = unsafe { cuEventDestroy(start) };
            bail!("cuEventCreate(end) failed: {status}");
        }
        Ok(Self { start, end })
    }

    fn measure(&self, stream: u64, launch: impl FnOnce() -> Result<()>) -> Result<f64> {
        let stream_ptr = stream as usize as *mut c_void;
        let status = unsafe { cuEventRecord(self.start, stream_ptr) };
        ensure!(status == 0, "cuEventRecord(start) failed: {status}");
        launch()?;
        let status = unsafe { cuEventRecord(self.end, stream_ptr) };
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

struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    image: Vec<u8>,
}

impl Guarded {
    fn input(
        gpu: &dyn GpuBackend,
        payload: &[u8],
        prefix_canary: u8,
        suffix_canary: u8,
    ) -> Result<Self> {
        let mut image = vec![prefix_canary; REDZONE + payload.len() + REDZONE];
        image[REDZONE..REDZONE + payload.len()].copy_from_slice(payload);
        image[REDZONE + payload.len()..].fill(suffix_canary);
        let allocation = gpu.alloc(image.len())?;
        gpu.copy_h2d(&image, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            image,
        })
    }

    fn output(
        gpu: &dyn GpuBackend,
        payload_len: usize,
        fill: u8,
        prefix_canary: u8,
        suffix_canary: u8,
    ) -> Result<Self> {
        Self::input(gpu, &vec![fill; payload_len], prefix_canary, suffix_canary)
    }

    fn ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    fn reset_payload_fill(&self, gpu: &dyn GpuBackend, fill: u8) -> Result<()> {
        let mut image = self.image.clone();
        image[REDZONE..REDZONE + self.payload_len].fill(fill);
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
            actual[..REDZONE] == self.image[..REDZONE],
            "{label}: leading 4-KiB redzone changed"
        );
        let suffix = REDZONE + self.payload_len;
        ensure!(
            actual[suffix..] == self.image[suffix..],
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

struct UploadedWeight {
    packed: Guarded,
    scales: Guarded,
    quant: QuantizedWeight,
}

impl UploadedWeight {
    fn new(gpu: &dyn GpuBackend, fixture: &Fixture, scale2: f32, salt: u8) -> Result<Self> {
        ensure!(fixture.logical_n == INTERMEDIATE, "weight N mismatch");
        ensure!(
            fixture.physical_n == INTERMEDIATE,
            "weight physical N mismatch"
        );
        ensure!(fixture.k == HIDDEN, "weight K mismatch");
        let packed = Guarded::input(gpu, &fixture.packed, 0x80 ^ salt, 0x2d ^ salt)?;
        let scales = Guarded::input(gpu, &fixture.scales, 0x4b ^ salt, 0xe1 ^ salt)?;
        let quant = QuantizedWeight {
            weight: packed.ptr(),
            weight_scale: scales.ptr(),
            weight_scale_2: scale2,
            input_scale: DevicePtr::NULL,
        };
        Ok(Self {
            packed,
            scales,
            quant,
        })
    }

    fn immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        self.packed.immutable(gpu, &format!("{label}/packed"))?;
        self.scales.immutable(gpu, &format!("{label}/scales"))
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.packed.free(gpu)?;
        self.scales.free(gpu)
    }
}

struct Kernels {
    pipe: KernelHandle,
    up_only: KernelHandle,
    candidate: KernelHandle,
    candidate_kind: Candidate,
    silu: KernelHandle,
}

fn exact_kernel_bundle() -> Result<(String, Vec<(&'static str, &'static str)>)> {
    let model = std::env::var("ATLAS_TARGET_MODEL")
        .context("ATLAS_TARGET_MODEL must be set for raw qualification")?;
    let quant = std::env::var("ATLAS_TARGET_QUANT")
        .context("ATLAS_TARGET_QUANT must be set for raw qualification")?;
    ensure!(
        model == "qwen3.8-27b" && quant == "nvfp4",
        "raw qualification requires ATLAS_TARGET_MODEL=qwen3.8-27b and ATLAS_TARGET_QUANT=nvfp4"
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
        "embedded kernel bundle must contain exactly one (sm_121,qwen3.8-27b,nvfp4) target, found {}",
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
        "raw qualification requires GB10 SM121 with 48 SMs, got name={name:?} cc={major}.{minor} sms={multiprocessors}"
    );
    Ok(DeviceIdentity {
        ordinal,
        name,
        major,
        minor,
        multiprocessors,
    })
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

fn input_bytes(rows: usize, kind: InputKind) -> Vec<u8> {
    let fixture = match kind {
        InputKind::Random => random_fixture(rows, 4, HIDDEN, 0x51c0_f00d_0000_0000 ^ rows as u64),
        InputKind::Cancellation => cancellation_fixture(rows, 4, HIDDEN),
    };
    as_le_bytes(&fixture.activations)
}

#[allow(clippy::too_many_arguments)]
fn launch_parent(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    rows: usize,
) -> Result<()> {
    ops::w4a16_gemm_pipe(
        gpu,
        kernels.pipe,
        input,
        gate_weight,
        gate_out,
        rows as u32,
        INTERMEDIATE as u32,
        HIDDEN as u32,
        stream,
    )?;
    ops::w4a16_gemm_pipe(
        gpu,
        kernels.pipe,
        input,
        up_weight,
        up_out,
        rows as u32,
        INTERMEDIATE as u32,
        HIDDEN as u32,
        stream,
    )?;
    ops::silu_mul(
        gpu,
        kernels.silu,
        gate_out,
        up_out,
        gate_out,
        (rows * INTERMEDIATE) as u32,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_up_only(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    output: DevicePtr,
    rows: usize,
) -> Result<()> {
    ops::w4a16_gemm_pipe(
        gpu,
        kernels.pipe,
        input,
        gate_weight,
        output,
        rows as u32,
        INTERMEDIATE as u32,
        HIDDEN as u32,
        stream,
    )?;
    ops::w4a16_gemm_pipe_silu_mul(
        gpu,
        kernels.up_only,
        input,
        up_weight,
        output,
        rows as u32,
        INTERMEDIATE as u32,
        HIDDEN as u32,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_candidate(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    output: DevicePtr,
    output_second: DevicePtr,
    rows: usize,
) -> Result<()> {
    let candidate = kernels.candidate_kind;
    KernelLaunch::new(gpu, kernels.candidate)
        .grid([
            div_ceil(INTERMEDIATE as u32, candidate.n_tile()),
            div_ceil(rows as u32, 64),
            1,
        ])
        .block([candidate.launch_threads(), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight.weight)
        .arg_ptr(gate_weight.weight_scale)
        .arg_f32(gate_weight.weight_scale_2)
        .arg_ptr(up_weight.weight)
        .arg_ptr(up_weight.weight_scale)
        .arg_f32(up_weight.weight_scale_2)
        .arg_ptr(output)
        .arg_ptr(output_second)
        .arg_u32(1)
        .arg_u32(rows as u32)
        .arg_u32(INTERMEDIATE as u32)
        .arg_u32(HIDDEN as u32)
        .launch(stream)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn require_equal(label: &str, reference: &[u8], actual: &[u8]) -> Result<()> {
    if reference == actual {
        return Ok(());
    }
    let byte = reference
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .context("mismatch without an index")?;
    let element = byte / 2;
    let word = element * 2;
    let reference_word = u16::from_le_bytes([reference[word], reference[word + 1]]);
    let actual_word = u16::from_le_bytes([actual[word], actual[word + 1]]);
    bail!(
        "{label}: first mismatch byte={byte} row={} column={} reference=0x{reference_word:04x} actual=0x{actual_word:04x} reference_hash={:016x} actual_hash={:016x}",
        element / INTERMEDIATE,
        element % INTERMEDIATE,
        fnv1a64(reference),
        fnv1a64(actual),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    gate_weight: &UploadedWeight,
    up_weight: &UploadedWeight,
    rows: usize,
    kind: InputKind,
) -> Result<()> {
    let label = format!("M={rows}/{}", kind.label());
    let candidate_label = kernels.candidate_kind.label();
    let input = Guarded::input(gpu, &input_bytes(rows, kind), 0x31, 0x13)?;
    let output_bytes = rows * INTERMEDIATE * size_of::<u16>();
    let parent = Guarded::output(gpu, output_bytes, 0xa5, 0x81, 0x18)?;
    let parent_up = Guarded::output(gpu, output_bytes, 0xb6, 0x82, 0x28)?;
    let up_only = Guarded::output(gpu, output_bytes, 0x5a, 0x83, 0x38)?;
    let candidate = Guarded::output(gpu, output_bytes, 0xc3, 0x84, 0x48)?;

    launch_parent(
        gpu,
        stream,
        kernels,
        input.ptr(),
        &gate_weight.quant,
        &up_weight.quant,
        parent.ptr(),
        parent_up.ptr(),
        rows,
    )?;
    launch_up_only(
        gpu,
        stream,
        kernels,
        input.ptr(),
        &gate_weight.quant,
        &up_weight.quant,
        up_only.ptr(),
        rows,
    )?;
    launch_candidate(
        gpu,
        stream,
        kernels,
        input.ptr(),
        &gate_weight.quant,
        &up_weight.quant,
        candidate.ptr(),
        DevicePtr::NULL,
        rows,
    )?;
    gpu.synchronize(stream)?;

    let parent_bytes = parent.payload(gpu, &format!("{label}/parent"))?;
    let _ = parent_up.payload(gpu, &format!("{label}/parent-up-scratch"))?;
    let up_only_bytes = up_only.payload(gpu, &format!("{label}/up-only"))?;
    let candidate_bytes = candidate.payload(gpu, &format!("{label}/{candidate_label}"))?;
    require_equal(
        &format!("{label}/parent-vs-up-only"),
        &parent_bytes,
        &up_only_bytes,
    )?;
    require_equal(
        &format!("{label}/parent-vs-{candidate_label}"),
        &parent_bytes,
        &candidate_bytes,
    )?;

    if rows == 65 && kind == InputKind::Random {
        let ignored_c1 = Guarded::output(gpu, output_bytes, 0x69, 0x85, 0x58)?;
        candidate.reset_payload_fill(gpu, 0x7e)?;
        launch_candidate(
            gpu,
            stream,
            kernels,
            input.ptr(),
            &gate_weight.quant,
            &up_weight.quant,
            candidate.ptr(),
            ignored_c1.ptr(),
            rows,
        )?;
        gpu.synchronize(stream)?;
        let nonnull_candidate =
            candidate.payload(gpu, &format!("{label}/{candidate_label}-nonnull-c1"))?;
        require_equal(
            &format!("{label}/{candidate_label}-null-vs-nonnull-c1"),
            &candidate_bytes,
            &nonnull_candidate,
        )?;
        ignored_c1.immutable(gpu, &format!("{label}/ignored-c1"))?;

        parent.reset_payload_fill(gpu, 0xe7)?;
        parent_up.reset_payload_fill(gpu, 0xd6)?;
        up_only.reset_payload_fill(gpu, 0x19)?;
        candidate.reset_payload_fill(gpu, 0x2a)?;
        launch_candidate(
            gpu,
            stream,
            kernels,
            input.ptr(),
            &gate_weight.quant,
            &up_weight.quant,
            candidate.ptr(),
            DevicePtr::NULL,
            rows,
        )?;
        launch_up_only(
            gpu,
            stream,
            kernels,
            input.ptr(),
            &gate_weight.quant,
            &up_weight.quant,
            up_only.ptr(),
            rows,
        )?;
        launch_parent(
            gpu,
            stream,
            kernels,
            input.ptr(),
            &gate_weight.quant,
            &up_weight.quant,
            parent.ptr(),
            parent_up.ptr(),
            rows,
        )?;
        gpu.synchronize(stream)?;
        let parent_replay = parent.payload(gpu, &format!("{label}/parent-replay"))?;
        let up_only_replay = up_only.payload(gpu, &format!("{label}/up-only-replay"))?;
        let candidate_replay =
            candidate.payload(gpu, &format!("{label}/{candidate_label}-replay"))?;
        require_equal(
            &format!("{label}/parent-fill-independence"),
            &parent_bytes,
            &parent_replay,
        )?;
        require_equal(
            &format!("{label}/up-only-fill-independence"),
            &up_only_bytes,
            &up_only_replay,
        )?;
        require_equal(
            &format!("{label}/{candidate_label}-fill-independence"),
            &candidate_bytes,
            &candidate_replay,
        )?;
        ignored_c1.free(gpu)?;
    }

    input.immutable(gpu, &format!("{label}/A"))?;
    gate_weight.immutable(gpu, &format!("{label}/gate-weight"))?;
    up_weight.immutable(gpu, &format!("{label}/up-weight"))?;
    println!(
        "PASS {label}: {} bytes identical across parent/up-only/{candidate_label}; checksum={:016x}",
        parent_bytes.len(),
        fnv1a64(&parent_bytes)
    );

    input.free(gpu)?;
    parent.free(gpu)?;
    parent_up.free(gpu)?;
    up_only.free(gpu)?;
    candidate.free(gpu)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn percentile_90(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() * 9).div_ceil(10).saturating_sub(1)]
}

fn median_absolute_deviation(values: &[f64]) -> f64 {
    let center = median(values);
    let deviations: Vec<_> = values.iter().map(|value| (value - center).abs()).collect();
    median(&deviations)
}

#[derive(Debug)]
struct TimingReceipt {
    rows: usize,
    candidate: &'static str,
    parent_median: f64,
    up_only_median: f64,
    candidate_median: f64,
    parent_p90: f64,
    up_only_p90: f64,
    candidate_p90: f64,
    pd_median: f64,
    ud_median: f64,
    pd_mad: f64,
    ud_mad: f64,
    pd_lower: f64,
    ud_lower: f64,
    required_pd: f64,
    required_ud: f64,
    wins_median_and_p90: bool,
    clears_noise_bound: bool,
    clears_absolute_savings: bool,
}

impl TimingReceipt {
    const fn passed(&self) -> bool {
        self.wins_median_and_p90 && self.clears_noise_bound && self.clears_absolute_savings
    }

    fn failed_gates(&self) -> String {
        let mut failed = Vec::with_capacity(3);
        if !self.wins_median_and_p90 {
            failed.push("median+p90");
        }
        if !self.clears_noise_bound {
            failed.push("3*MAD");
        }
        if !self.clears_absolute_savings {
            failed.push("absolute-savings");
        }
        failed.join(",")
    }

    fn print(&self) {
        println!(
            "TIMING RECEIPT candidate={} M={}: median parent={:.3} up-only={:.3} candidate={:.3} ms; \
             p90 parent={:.3} up-only={:.3} candidate={:.3} ms; \
             paired P-C median={:.3} MAD={:.3} lower={:.3} required={:.3}; \
             U-C median={:.3} MAD={:.3} lower={:.3} required={:.3}; verdict={}{}",
            self.candidate,
            self.rows,
            self.parent_median,
            self.up_only_median,
            self.candidate_median,
            self.parent_p90,
            self.up_only_p90,
            self.candidate_p90,
            self.pd_median,
            self.pd_mad,
            self.pd_lower,
            self.required_pd,
            self.ud_median,
            self.ud_mad,
            self.ud_lower,
            self.required_ud,
            if self.passed() { "PASS" } else { "FAIL" },
            if self.passed() {
                String::new()
            } else {
                format!(" failed-gates={}", self.failed_gates())
            }
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_route(
    route: Route,
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    output: DevicePtr,
    scratch: DevicePtr,
    rows: usize,
) -> Result<()> {
    match route {
        Route::Parent => launch_parent(
            gpu,
            stream,
            kernels,
            input,
            gate_weight,
            up_weight,
            output,
            scratch,
            rows,
        ),
        Route::UpOnly => launch_up_only(
            gpu,
            stream,
            kernels,
            input,
            gate_weight,
            up_weight,
            output,
            rows,
        ),
        Route::Candidate => launch_candidate(
            gpu,
            stream,
            kernels,
            input,
            gate_weight,
            up_weight,
            output,
            DevicePtr::NULL,
            rows,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn time_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    gate_weight: &UploadedWeight,
    up_weight: &UploadedWeight,
    rows: usize,
) -> Result<TimingReceipt> {
    let input = Guarded::input(gpu, &input_bytes(rows, InputKind::Random), 0x31, 0x13)?;
    let output_bytes = rows * INTERMEDIATE * size_of::<u16>();
    let output = Guarded::output(gpu, output_bytes, 0x5a, 0x91, 0x19)?;
    let scratch = Guarded::output(gpu, output_bytes, 0xa5, 0x92, 0x29)?;
    let events = CudaEventPair::new()?;

    let measure = |route: Route| -> Result<f64> {
        events.measure(stream, || {
            launch_route(
                route,
                gpu,
                stream,
                kernels,
                input.ptr(),
                &gate_weight.quant,
                &up_weight.quant,
                output.ptr(),
                scratch.ptr(),
                rows,
            )
        })
    };

    for permutation in TIMING_PERMUTATIONS {
        for route in permutation {
            let _ = measure(route)?;
        }
    }

    let mut parent_ms = Vec::with_capacity(18);
    let mut up_only_ms = Vec::with_capacity(18);
    let mut candidate_ms = Vec::with_capacity(18);
    let mut parent_minus_candidate = Vec::with_capacity(18);
    let mut up_only_minus_candidate = Vec::with_capacity(18);
    for cycle in 0..TIMING_PERMUTATION_CYCLES {
        for permutation in TIMING_PERMUTATIONS {
            let mut round_parent = None;
            let mut round_up_only = None;
            let mut round_candidate = None;
            for route in permutation {
                let elapsed = measure(route)?;
                match route {
                    Route::Parent => round_parent = Some(elapsed),
                    Route::UpOnly => round_up_only = Some(elapsed),
                    Route::Candidate => round_candidate = Some(elapsed),
                }
            }
            let parent = round_parent.context("balanced timing round omitted parent")?;
            let up_only = round_up_only.context("balanced timing round omitted up-only")?;
            let candidate = round_candidate.context("balanced timing round omitted candidate")?;
            parent_ms.push(parent);
            up_only_ms.push(up_only);
            candidate_ms.push(candidate);
            parent_minus_candidate.push(parent - candidate);
            up_only_minus_candidate.push(up_only - candidate);
        }
        println!(
            "TIMING M={rows}: completed balanced permutation cycle {}",
            cycle + 1
        );
    }

    let parent_median = median(&parent_ms);
    let up_only_median = median(&up_only_ms);
    let candidate_median = median(&candidate_ms);
    let parent_p90 = percentile_90(&parent_ms);
    let up_only_p90 = percentile_90(&up_only_ms);
    let candidate_p90 = percentile_90(&candidate_ms);
    let pd_median = median(&parent_minus_candidate);
    let ud_median = median(&up_only_minus_candidate);
    let pd_mad = median_absolute_deviation(&parent_minus_candidate);
    let ud_mad = median_absolute_deviation(&up_only_minus_candidate);
    let pd_lower = pd_median - 3.0 * pd_mad;
    let ud_lower = ud_median - 3.0 * ud_mad;
    let (required_pd, required_ud) = if rows == 2_048 {
        (0.25, 0.125)
    } else {
        (1.0, 0.5)
    };
    let receipt = TimingReceipt {
        rows,
        candidate: kernels.candidate_kind.label(),
        parent_median,
        up_only_median,
        candidate_median,
        parent_p90,
        up_only_p90,
        candidate_p90,
        pd_median,
        ud_median,
        pd_mad,
        ud_mad,
        pd_lower,
        ud_lower,
        required_pd,
        required_ud,
        wins_median_and_p90: candidate_median < parent_median
            && candidate_median < up_only_median
            && candidate_p90 < parent_p90
            && candidate_p90 < up_only_p90,
        clears_noise_bound: pd_lower > 0.0 && ud_lower > 0.0,
        clears_absolute_savings: pd_median >= required_pd && ud_median >= required_ud,
    };

    input.immutable(gpu, &format!("timing-M={rows}/A"))?;
    gate_weight.immutable(gpu, &format!("timing-M={rows}/gate-weight"))?;
    up_weight.immutable(gpu, &format!("timing-M={rows}/up-weight"))?;
    let _ = output.payload(gpu, &format!("timing-M={rows}/output"))?;
    let _ = scratch.payload(gpu, &format!("timing-M={rows}/scratch"))?;
    input.free(gpu)?;
    output.free(gpu)?;
    scratch.free(gpu)?;
    events.destroy()?;
    Ok(receipt)
}

fn matrix_checksum(cases: &[(usize, InputKind)]) -> u64 {
    cases.iter().fold(0xcbf2_9ce4_8422_2325, |hash, case| {
        format!("{}:{};", case.0, case.1.label())
            .bytes()
            .fold(hash, |inner, byte| {
                (inner ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    })
}

fn main() -> Result<()> {
    let full = strict_switch("ATLAS_PREFILL_FFN_DUAL_MICROGATE_FULL")?;
    let timing = strict_switch("ATLAS_PREFILL_FFN_DUAL_MICROGATE_TIMING")?;
    let candidate_kind = Candidate::from_env()?;
    ensure!(
        !timing || full,
        "timing requires ATLAS_PREFILL_FFN_DUAL_MICROGATE_FULL=1"
    );

    let (target, modules) = exact_kernel_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let device = current_device_identity()?;
    let stream = gpu.create_stream()?;
    let kernels = Kernels {
        pipe: gpu.kernel("w4a16", "w4a16_gemm_pipe")?,
        up_only: gpu.kernel("w4a16", "w4a16_gemm_pipe_silu_mul")?,
        candidate: gpu.kernel("w4a16", candidate_kind.symbol())?,
        candidate_kind,
        silu: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
    };
    let pipe_resources = function_resources(kernels.pipe)?;
    let up_only_resources = function_resources(kernels.up_only)?;
    let candidate_resources = function_resources(kernels.candidate)?;
    ensure!(
        pipe_resources.max_threads >= 128 && up_only_resources.max_threads >= 128,
        "parent and up-only routes must admit 128 threads: pipe={pipe_resources:?} up-only={up_only_resources:?}"
    );
    ensure!(
        candidate_resources.max_threads == candidate_kind.launch_threads() as i32
            && candidate_resources.registers <= 128
            && candidate_resources.shared_bytes <= candidate_kind.shared_ceiling()
            && candidate_resources.local_bytes == 0,
        "{} resource/launch-contract regression: expected max_threads={}, registers<=128, shared<={}, local=0; actual={candidate_resources:?}",
        candidate_kind.label(),
        candidate_kind.launch_threads(),
        candidate_kind.shared_ceiling()
    );
    println!(
        "RESOURCES candidate={} pipe={pipe_resources:?} up-only={up_only_resources:?} candidate_resources={candidate_resources:?} launch_threads={} n_tile={}",
        candidate_kind.label(),
        candidate_kind.launch_threads(),
        candidate_kind.n_tile()
    );

    let gate_fixture = random_fixture(1, INTERMEDIATE, HIDDEN, 0x51c0_0000_4400_0001);
    let up_fixture = random_fixture(1, INTERMEDIATE, HIDDEN, 0x51c0_0000_4400_0002);
    let gate_weight = UploadedWeight::new(gpu, &gate_fixture, 0.75, 0x11)?;
    let up_weight = UploadedWeight::new(gpu, &up_fixture, 1.25, 0x22)?;

    let smoke = [(65usize, InputKind::Random)];
    let mut complete: Vec<_> = FULL_ROWS
        .into_iter()
        .map(|rows| (rows, InputKind::Random))
        .collect();
    complete.push((65, InputKind::Cancellation));
    let cases = if full { &complete[..] } else { &smoke[..] };
    for &(rows, kind) in cases {
        run_case(gpu, stream, &kernels, &gate_weight, &up_weight, rows, kind)?;
    }

    let timing_receipts = if timing {
        TIMING_ROWS
            .into_iter()
            .map(|rows| time_case(gpu, stream, &kernels, &gate_weight, &up_weight, rows))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };

    gate_weight.immutable(gpu, "final/gate-weight")?;
    up_weight.immutable(gpu, "final/up-weight")?;
    gate_weight.free(gpu)?;
    up_weight.free(gpu)?;
    for receipt in &timing_receipts {
        receipt.print();
    }
    let failed_timing: Vec<_> = timing_receipts
        .iter()
        .filter(|receipt| !receipt.passed())
        .map(|receipt| format!("M={}:{}", receipt.rows, receipt.failed_gates()))
        .collect();
    ensure!(
        failed_timing.is_empty(),
        "aggregate timing gate failed after both rows: {}",
        failed_timing.join("; ")
    );
    let checksum = matrix_checksum(cases);
    let identity = format!(
        "target={target} candidate={} device={:?} ordinal={} cc={}.{} sms={}",
        candidate_kind.label(),
        device.name,
        device.ordinal,
        device.major,
        device.minor,
        device.multiprocessors
    );
    if full && timing {
        println!(
            "QUALIFIED PARITY+TIMING PASS: cases={}/{} matrix={checksum:016x} {identity}",
            cases.len(),
            complete.len()
        );
    } else if full {
        println!(
            "QUALIFIED PARITY PASS: cases={}/{} matrix={checksum:016x} {identity}",
            cases.len(),
            complete.len()
        );
    } else {
        println!(
            "SMOKE PASS ONLY - NOT QUALIFIED: cases={}/{} matrix={checksum:016x} {identity}",
            cases.len(),
            complete.len()
        );
    }
    Ok(())
}
