// SPDX-License-Identifier: AGPL-3.0-only

//! Raw fail-closed gate for the exact M17 attention O-projection A-staging twin.
//!
//! Every case compares the candidate with both the exact dynamic-M parent and
//! independent ordinary-K1 row launches. Inputs/weights are immutable guarded
//! images; outputs have 4-KiB redzones, a non-zero base offset, and canary-only
//! inactive M17 rows. Timing is opt-in and cannot run before the full parity
//! matrix succeeds.
//!
//! Run only on a reserved GB10:
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --features cuda \
//!   --example w4a16_oproj_m17_astage_microgate
//! ```
//! Add `ATLAS_M17_OPROJ_ASTAGE_TIMING=1` for alternating parent/candidate
//! timing. The timing gate requires at least 0.5 ms modeled saving across the
//! sixteen Qwen3.8 attention layers.

#[allow(dead_code)]
#[path = "w4a16_exact_lm_head_microtest/data.rs"]
mod data;

use std::ffi::{CStr, c_char, c_void};

use anyhow::{Context, Result, bail, ensure};
use data::{Fixture, as_le_bytes, cancellation_fixture, fnv1a64, from_le_bytes, random_fixture};
use spark_model::layers::ops::{self, W4a16ExactLmHeadKernels};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const MAX_M: usize = 17;
const PROD_N: usize = 5_120;
const PROD_K: usize = 6_144;
const REDZONE: usize = 4 * 1_024;
const BASE_PAD_WORDS: usize = 37;
const TAIL_PAD_WORDS: usize = 43;
const ATTENTION_LAYERS: f64 = 16.0;
const TIMING_BATCH: usize = 16;
const TIMING_ROUNDS: usize = 21;
const GB10_SMS: i32 = 48;
const ATTR_MULTIPROCESSOR_COUNT: u32 = 16;
const ATTR_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
const ATTR_COMPUTE_CAPABILITY_MINOR: u32 = 76;
const FUNC_ATTR_MAX_THREADS_PER_BLOCK: u32 = 0;
const FUNC_ATTR_SHARED_SIZE_BYTES: u32 = 1;
const FUNC_ATTR_LOCAL_SIZE_BYTES: u32 = 3;
const FUNC_ATTR_NUM_REGS: u32 = 4;

unsafe extern "C" {
    fn cuCtxGetDevice(device: *mut i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attribute: u32, device: i32) -> i32;
    fn cuDeviceGetName(name: *mut c_char, length: i32, device: i32) -> i32;
    fn cuFuncGetAttribute(value: *mut i32, attribute: u32, function: *mut c_void) -> i32;
    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks: *mut i32,
        function: *mut c_void,
        block_size: i32,
        dynamic_smem: usize,
    ) -> i32;
    fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> i32;
    fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cuEventSynchronize(event: *mut c_void) -> i32;
    fn cuEventElapsedTime(milliseconds: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    #[link_name = "cuEventDestroy_v2"]
    fn cuEventDestroy(event: *mut c_void) -> i32;
}

#[derive(Clone, Copy, Debug)]
enum InputKind {
    Random,
    Cancellation,
    Extreme,
}

impl InputKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Cancellation => "cancellation",
            Self::Extreme => "signed-zero-subnormal-inf-nan",
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    rows: usize,
    n: usize,
    k: usize,
    kind: InputKind,
    seed: u64,
}

impl Case {
    fn label(self) -> String {
        format!(
            "M{}_N{}_K{}_{}",
            self.rows,
            self.n,
            self.k,
            self.kind.label()
        )
    }
}

#[derive(Clone, Copy)]
struct Kernels {
    serial: KernelHandle,
    parent: W4a16ExactLmHeadKernels,
    staged: KernelHandle,
}

struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    expected: Vec<u8>,
    redzone: u8,
}

impl Guarded {
    fn input(gpu: &dyn GpuBackend, payload: &[u8], redzone: u8) -> Result<Self> {
        let mut expected = vec![redzone; REDZONE + payload.len() + REDZONE];
        expected[REDZONE..REDZONE + payload.len()].copy_from_slice(payload);
        let allocation = gpu.alloc(expected.len())?;
        gpu.copy_h2d(&expected, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            expected,
            redzone,
        })
    }

    fn output(gpu: &dyn GpuBackend, active_width: usize, fill: u8, redzone: u8) -> Result<Self> {
        let payload_words = BASE_PAD_WORDS + MAX_M * active_width + TAIL_PAD_WORDS;
        Self::input(gpu, &vec![fill; payload_words * size_of::<u16>()], redzone)
    }

    fn ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    fn output_ptr(&self) -> DevicePtr {
        self.ptr().offset(BASE_PAD_WORDS * size_of::<u16>())
    }

    fn read(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<Vec<u8>> {
        let mut actual = vec![0u8; self.expected.len()];
        gpu.copy_d2h_on_stream(self.allocation, &mut actual, stream)?;
        ensure!(
            actual[..REDZONE].iter().all(|&byte| byte == self.redzone),
            "leading 4-KiB redzone changed"
        );
        let suffix = REDZONE + self.payload_len;
        ensure!(
            actual[suffix..].iter().all(|&byte| byte == self.redzone),
            "trailing 4-KiB redzone changed"
        );
        Ok(actual)
    }

    fn immutable(&self, gpu: &dyn GpuBackend, stream: u64, label: &str) -> Result<()> {
        let actual = self.read(gpu, stream)?;
        if let Some(index) = actual
            .iter()
            .zip(&self.expected)
            .position(|(actual, expected)| actual != expected)
        {
            bail!(
                "{label}: immutable image changed at byte {index}: actual=0x{:02x} expected=0x{:02x}",
                actual[index],
                self.expected[index]
            );
        }
        Ok(())
    }

    fn active_output(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        rows: usize,
        width: usize,
        fill: u8,
        label: &str,
    ) -> Result<Vec<u16>> {
        let actual = self.read(gpu, stream)?;
        let payload = &actual[REDZONE..REDZONE + self.payload_len];
        let active_start = BASE_PAD_WORDS * size_of::<u16>();
        let active_end = active_start + rows * width * size_of::<u16>();
        ensure!(
            payload[..active_start].iter().all(|&byte| byte == fill),
            "{label}: output base-offset guard changed"
        );
        ensure!(
            payload[active_end..].iter().all(|&byte| byte == fill),
            "{label}: inactive M17 row or trailing stride guard changed"
        );
        Ok(from_le_bytes(&payload[active_start..active_end]))
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}

struct Uploaded {
    input: Guarded,
    packed: Guarded,
    scales: Guarded,
    weight: QuantizedWeight,
}

impl Uploaded {
    fn new(gpu: &dyn GpuBackend, fixture: &Fixture) -> Result<Self> {
        let input = Guarded::input(gpu, &as_le_bytes(&fixture.activations), 0x91)?;
        let packed = Guarded::input(gpu, &fixture.packed, 0xa2)?;
        let scales = Guarded::input(gpu, &fixture.scales, 0xb3)?;
        let weight = QuantizedWeight {
            weight: packed.ptr(),
            weight_scale: scales.ptr(),
            weight_scale_2: 1.0,
            input_scale: DevicePtr::NULL,
        };
        Ok(Self {
            input,
            packed,
            scales,
            weight,
        })
    }

    fn immutable(&self, gpu: &dyn GpuBackend, stream: u64, label: &str) -> Result<()> {
        self.input.immutable(gpu, stream, &format!("{label}/A"))?;
        self.packed
            .immutable(gpu, stream, &format!("{label}/packed-weight"))?;
        self.scales
            .immutable(gpu, stream, &format!("{label}/weight-scales"))
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.input.free(gpu)?;
        self.packed.free(gpu)?;
        self.scales.free(gpu)
    }
}

struct CudaEvents {
    start: *mut c_void,
    end: *mut c_void,
}

impl CudaEvents {
    fn new() -> Result<Self> {
        let mut start = std::ptr::null_mut();
        let mut end = std::ptr::null_mut();
        ensure!(
            unsafe { cuEventCreate(&mut start, 0) } == 0,
            "create start event"
        );
        if unsafe { cuEventCreate(&mut end, 0) } != 0 {
            let _ = unsafe { cuEventDestroy(start) };
            bail!("create end event");
        }
        Ok(Self { start, end })
    }

    fn measure(&self, stream: u64, launch: impl FnOnce() -> Result<()>) -> Result<f64> {
        let raw_stream = stream as usize as *mut c_void;
        ensure!(
            unsafe { cuEventRecord(self.start, raw_stream) } == 0,
            "record start event"
        );
        launch()?;
        ensure!(
            unsafe { cuEventRecord(self.end, raw_stream) } == 0,
            "record end event"
        );
        ensure!(
            unsafe { cuEventSynchronize(self.end) } == 0,
            "synchronize end event"
        );
        let mut ms = 0.0f32;
        ensure!(
            unsafe { cuEventElapsedTime(&mut ms, self.start, self.end) } == 0,
            "elapsed event time"
        );
        Ok(f64::from(ms))
    }

    fn destroy(self) -> Result<()> {
        ensure!(
            unsafe { cuEventDestroy(self.start) } == 0,
            "destroy start event"
        );
        ensure!(
            unsafe { cuEventDestroy(self.end) } == 0,
            "destroy end event"
        );
        Ok(())
    }
}

fn exact_bundle() -> Result<(String, Vec<(&'static str, &'static str)>)> {
    ensure!(
        std::env::var("ATLAS_TARGET_MODEL").as_deref() == Ok("qwen3.8-27b")
            && std::env::var("ATLAS_TARGET_QUANT").as_deref() == Ok("nvfp4"),
        "qualification requires exact ATLAS_TARGET_MODEL=qwen3.8-27b and ATLAS_TARGET_QUANT=nvfp4"
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
        "expected one embedded exact target, found {}",
        matches.len()
    );
    let matched = matches.pop().context("exact target disappeared")?;
    Ok((matched.target.to_string(), matched.modules))
}

fn require_gb10() -> Result<String> {
    let mut device = -1;
    ensure!(
        unsafe { cuCtxGetDevice(&mut device) } == 0 && device >= 0,
        "active CUDA device"
    );
    let attr = |kind, label| -> Result<i32> {
        let mut value = -1;
        let status = unsafe { cuDeviceGetAttribute(&mut value, kind, device) };
        ensure!(
            status == 0 && value >= 0,
            "device attribute {label}: status={status}"
        );
        Ok(value)
    };
    let major = attr(ATTR_COMPUTE_CAPABILITY_MAJOR, "cc-major")?;
    let minor = attr(ATTR_COMPUTE_CAPABILITY_MINOR, "cc-minor")?;
    let sms = attr(ATTR_MULTIPROCESSOR_COUNT, "sms")?;
    let mut name = [0 as c_char; 256];
    ensure!(
        unsafe { cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, device) } == 0,
        "device name"
    );
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    ensure!(
        major == 12 && minor == 1 && sms == GB10_SMS && name.to_ascii_uppercase().contains("GB10"),
        "qualification requires GB10 CC12.1/48SM, got {name:?} cc={major}.{minor} sms={sms}"
    );
    Ok(format!(
        "ordinal={device},name={name},cc={major}.{minor},sms={sms}"
    ))
}

fn load_kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    let staged = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m17_astage")?;
    let parent = W4a16ExactLmHeadKernels::new(
        KernelHandle(0),
        KernelHandle(0),
        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m17")?,
        KernelHandle(0),
    )
    .with_m17_astage(staged);
    Ok(Kernels {
        serial: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
        parent,
        staged,
    })
}

fn require_resources(kernel: KernelHandle) -> Result<String> {
    let function = kernel.0 as usize as *mut c_void;
    let attr = |kind, label| -> Result<i32> {
        let mut value = -1;
        let status = unsafe { cuFuncGetAttribute(&mut value, kind, function) };
        ensure!(
            status == 0 && value >= 0,
            "function attribute {label}: status={status}"
        );
        Ok(value)
    };
    let threads = attr(FUNC_ATTR_MAX_THREADS_PER_BLOCK, "threads")?;
    let shared = attr(FUNC_ATTR_SHARED_SIZE_BYTES, "shared")?;
    let local = attr(FUNC_ATTR_LOCAL_SIZE_BYTES, "local")?;
    let registers = attr(FUNC_ATTR_NUM_REGS, "registers")?;
    let mut resident = -1;
    ensure!(
        unsafe { cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut resident, function, 256, 0) }
            == 0,
        "candidate occupancy query"
    );
    ensure!(
        threads >= 256 && shared == 35_424 && local == 0 && resident >= 2,
        "candidate resource gate failed: threads={threads} shared={shared} local={local} regs={registers} resident={resident}"
    );
    Ok(format!(
        "threads={threads},shared={shared},local={local},regs={registers},resident_ctas_per_sm={resident}"
    ))
}

fn strict_switch(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => bail!("{name} must be exactly 0 or 1"),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} must be valid UTF-8 and 0 or 1"),
    }
}

fn extreme_fixture(n: usize, k: usize, seed: u64) -> Fixture {
    let mut fixture = random_fixture(MAX_M, n, k, seed);
    let bits = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80,
        0xff80, 0x7fc1, 0xffc1, 0x7f81, 0xff81,
    ];
    for (index, value) in fixture.activations.iter_mut().enumerate() {
        *value = bits[(index * 13 + index / k * 7) % bits.len()];
    }
    fixture
}

fn fixture(case: Case) -> Fixture {
    match case.kind {
        InputKind::Random => random_fixture(MAX_M, case.n, case.k, case.seed),
        InputKind::Cancellation => cancellation_fixture(MAX_M, case.n, case.k),
        InputKind::Extreme => extreme_fixture(case.n, case.k, case.seed),
    }
}

fn launch_serial(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    uploaded: &Uploaded,
    output: DevicePtr,
    case: Case,
) -> Result<()> {
    for row in 0..case.rows {
        ops::w4a16_gemv(
            gpu,
            kernels.serial,
            uploaded.input.ptr().offset(row * case.k * size_of::<u16>()),
            &uploaded.weight,
            output.offset(row * case.n * size_of::<u16>()),
            case.n as u32,
            case.k as u32,
            stream,
        )?;
    }
    Ok(())
}

fn launch_parent(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    uploaded: &Uploaded,
    output: DevicePtr,
    case: Case,
) -> Result<()> {
    ops::w4a16_gemv_batch_logits_exact_with(
        gpu,
        kernels.parent,
        uploaded.input.ptr(),
        &uploaded.weight,
        output,
        case.rows as u32,
        case.n as u32,
        case.k as u32,
        stream,
        false,
    )
}

fn launch_staged(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    uploaded: &Uploaded,
    output: DevicePtr,
    case: Case,
) -> Result<()> {
    ops::w4a16_gemv_batch_logits_exact_m17_astage(
        gpu,
        kernels.staged,
        uploaded.input.ptr(),
        &uploaded.weight,
        output,
        case.rows as u32,
        case.n as u32,
        case.k as u32,
        stream,
    )
}

fn exact(label: &str, actual: &[u16], expected: &[u16], width: usize) -> Result<()> {
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        bail!(
            "{label}: BF16 mismatch flat={index} row={} col={} actual=0x{:04x} expected=0x{:04x}",
            index / width,
            index % width,
            actual[index],
            expected[index]
        );
    }
    ensure!(
        actual.len() == expected.len(),
        "{label}: output length mismatch"
    );
    Ok(())
}

fn run_case(gpu: &dyn GpuBackend, stream: u64, kernels: Kernels, case: Case) -> Result<()> {
    ensure!((9..=17).contains(&case.rows), "raw case outside M17 route");
    let label = case.label();
    let fixture = fixture(case);
    let uploaded = Uploaded::new(gpu, &fixture)?;
    let serial = Guarded::output(gpu, case.n, 0x51, 0xc1)?;
    let parent = Guarded::output(gpu, case.n, 0x62, 0xd2)?;
    let staged = Guarded::output(gpu, case.n, 0x73, 0xe3)?;
    let replay = Guarded::output(gpu, case.n, 0x84, 0xf4)?;

    launch_serial(gpu, stream, kernels, &uploaded, serial.output_ptr(), case)
        .with_context(|| format!("{label}/serial"))?;
    launch_parent(gpu, stream, kernels, &uploaded, parent.output_ptr(), case)
        .with_context(|| format!("{label}/parent"))?;
    launch_staged(gpu, stream, kernels, &uploaded, staged.output_ptr(), case)
        .with_context(|| format!("{label}/staged"))?;
    launch_staged(gpu, stream, kernels, &uploaded, replay.output_ptr(), case)
        .with_context(|| format!("{label}/staged-replay"))?;
    gpu.synchronize(stream)?;

    let serial_words = serial.active_output(gpu, stream, case.rows, case.n, 0x51, "serial")?;
    let parent_words = parent.active_output(gpu, stream, case.rows, case.n, 0x62, "parent")?;
    let staged_words = staged.active_output(gpu, stream, case.rows, case.n, 0x73, "staged")?;
    let replay_words = replay.active_output(gpu, stream, case.rows, case.n, 0x84, "replay")?;
    exact(
        &format!("{label}/parent-vs-serial"),
        &parent_words,
        &serial_words,
        case.n,
    )?;
    exact(
        &format!("{label}/staged-vs-serial"),
        &staged_words,
        &serial_words,
        case.n,
    )?;
    exact(
        &format!("{label}/determinism"),
        &replay_words,
        &staged_words,
        case.n,
    )?;
    uploaded.immutable(gpu, stream, &label)?;

    println!(
        "PARITY {label} physical_weight_n={} hash=fnv1a64:{:016x} parent=EXACT serial=EXACT deterministic=PASS redzones=PASS inactive_rows=PASS immutable=PASS",
        fixture.physical_n,
        fnv1a64(&staged_words)
    );
    serial.free(gpu)?;
    parent.free(gpu)?;
    staged.free(gpu)?;
    replay.free(gpu)?;
    uploaded.free(gpu)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn p90(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() * 9).div_ceil(10) - 1).min(values.len() - 1)]
}

fn time_candidate(gpu: &dyn GpuBackend, stream: u64, kernels: Kernels) -> Result<()> {
    let case = Case {
        rows: MAX_M,
        n: PROD_N,
        k: PROD_K,
        kind: InputKind::Random,
        seed: 0xa57a_0f00_5120_6144,
    };
    let uploaded = Uploaded::new(gpu, &fixture(case))?;
    let parent = Guarded::output(gpu, case.n, 0x35, 0xc5)?;
    let staged = Guarded::output(gpu, case.n, 0x46, 0xd6)?;
    for _ in 0..3 {
        launch_parent(gpu, stream, kernels, &uploaded, parent.output_ptr(), case)?;
        launch_staged(gpu, stream, kernels, &uploaded, staged.output_ptr(), case)?;
    }
    gpu.synchronize(stream)?;

    let events = CudaEvents::new()?;
    let mut parent_ms = Vec::with_capacity(TIMING_ROUNDS);
    let mut staged_ms = Vec::with_capacity(TIMING_ROUNDS);
    let mut paired = Vec::with_capacity(TIMING_ROUNDS);
    for round in 0..TIMING_ROUNDS {
        let measure_parent = || {
            events
                .measure(stream, || {
                    for _ in 0..TIMING_BATCH {
                        launch_parent(gpu, stream, kernels, &uploaded, parent.output_ptr(), case)?;
                    }
                    Ok(())
                })
                .map(|ms| ms / TIMING_BATCH as f64)
        };
        let measure_staged = || {
            events
                .measure(stream, || {
                    for _ in 0..TIMING_BATCH {
                        launch_staged(gpu, stream, kernels, &uploaded, staged.output_ptr(), case)?;
                    }
                    Ok(())
                })
                .map(|ms| ms / TIMING_BATCH as f64)
        };
        let (p, s) = if round % 2 == 0 {
            (measure_parent()?, measure_staged()?)
        } else {
            let s = measure_staged()?;
            let p = measure_parent()?;
            (p, s)
        };
        parent_ms.push(p);
        staged_ms.push(s);
        paired.push(p - s);
    }
    events.destroy()?;

    let p_med = median(parent_ms.clone());
    let s_med = median(staged_ms.clone());
    let p_p90 = p90(parent_ms);
    let s_p90 = p90(staged_ms);
    let paired_med = median(paired.clone());
    let mad = median(
        paired
            .iter()
            .map(|value| (value - paired_med).abs())
            .collect(),
    );
    let conservative = paired_med - 3.0 * mad;
    let modeled_frame_saving = (p_med - s_med) * ATTENTION_LAYERS;
    ensure!(
        s_med < p_med && s_p90 < p_p90,
        "candidate must beat parent at median and p90"
    );
    ensure!(
        conservative > 0.0,
        "paired timing does not clear median-3*MAD"
    );
    ensure!(
        modeled_frame_saving >= 0.5,
        "modeled full-attention saving {modeled_frame_saving:.3} ms is below 0.5 ms stop gate"
    );
    let p_words = parent.active_output(gpu, stream, case.rows, case.n, 0x35, "timed-parent")?;
    let s_words = staged.active_output(gpu, stream, case.rows, case.n, 0x46, "timed-staged")?;
    exact("post-timing parent-vs-staged", &s_words, &p_words, case.n)?;
    uploaded.immutable(gpu, stream, "post-timing")?;
    println!(
        "TIMING M17_N5120_K6144 batch={TIMING_BATCH} rounds={TIMING_ROUNDS} parent_median_ms={p_med:.6} staged_median_ms={s_med:.6} parent_p90_ms={p_p90:.6} staged_p90_ms={s_p90:.6} paired_median_saving_ms={paired_med:.6} mad_ms={mad:.6} conservative_ms={conservative:.6} modeled_x16_saving_ms={modeled_frame_saving:.6} gate=PASS"
    );
    parent.free(gpu)?;
    staged.free(gpu)?;
    uploaded.free(gpu)
}

fn main() -> Result<()> {
    let timing = strict_switch("ATLAS_M17_OPROJ_ASTAGE_TIMING")?;
    let (target, modules) = exact_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules).context("initialize exact Qwen3.8 bundle")?;
    let gpu: &dyn GpuBackend = &backend;
    let device = require_gb10()?;
    let stream = gpu.create_stream()?;
    let kernels = load_kernels(gpu)?;
    let resources = require_resources(kernels.staged)?;
    println!("PROVENANCE target={target} device={device} candidate_resources={resources}");

    let mut cases = Vec::new();
    for rows in 9..=17 {
        cases.push(Case {
            rows,
            n: PROD_N,
            k: PROD_K,
            kind: InputKind::Random,
            seed: 0x9000_5120_6144_0000 ^ rows as u64,
        });
    }
    for n in [1, 2, 3, 5_119] {
        cases.push(Case {
            rows: 17,
            n,
            k: PROD_K,
            kind: InputKind::Random,
            seed: 0x7a11_0000_6144_0000 ^ n as u64,
        });
    }
    for (rows, kind) in [
        (9, InputKind::Cancellation),
        (17, InputKind::Cancellation),
        (9, InputKind::Extreme),
        (17, InputKind::Extreme),
    ] {
        cases.push(Case {
            rows,
            n: 7,
            k: 528,
            kind,
            seed: 0xe871_0000_0528_0000 ^ rows as u64,
        });
    }

    for case in cases.iter().copied() {
        run_case(gpu, stream, kernels, case)?;
    }
    if timing {
        time_candidate(gpu, stream, kernels)?;
    }
    println!(
        "FINAL verdict=PASS cases={} matrix=M9..17/N5120/K6144+N1,2,3,5119+K528 timing={timing} target={target} device={device} production_route=false",
        cases.len()
    );
    Ok(())
}
