// SPDX-License-Identifier: AGPL-3.0-only

//! Raw quality and timing screen for Qwen3.8 original-layout BF16 MMA versus
//! transposed K-major FP8 MMA. The candidate is approximate by construction;
//! this program never claims bit-exact qualification. Run only on a reserved GB10.
//!
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//! ATLAS_PREFILL_FFN_TRANSPOSED_MICROGATE_TIMING=1 \
//! cargo run --release -p spark-model --features cuda \
//!   --example qwen38_prefill_ffn_transposed_microgate
//! ```

#[allow(dead_code)]
#[path = "w4a16_exact_lm_head_microtest/data.rs"]
mod data;

use std::ffi::{CStr, c_char, c_void};

use anyhow::{Context, Result, bail, ensure};
use data::{Fixture, as_le_bytes, random_fixture};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const REDZONE: usize = 4 * 1_024;
const SCREEN_ROWS: [usize; 5] = [127, 128, 129, 2_048, 8_192];
const TIMING_ROWS: [usize; 2] = [2_048, 8_192];
const TIMING_CYCLES: usize = 9;
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
    fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> i32;
    fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cuEventSynchronize(event: *mut c_void) -> i32;
    fn cuEventElapsedTime(milliseconds: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    #[link_name = "cuEventDestroy_v2"]
    fn cuEventDestroy(event: *mut c_void) -> i32;
}

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    k: usize,
    n: usize,
    seed: u64,
}

const SHAPES: [Shape; 2] = [
    Shape {
        label: "gate-up",
        k: 5_120,
        n: 17_408,
        seed: 0x51c0_1740_8005_1201,
    },
    Shape {
        label: "down",
        k: 17_408,
        n: 5_120,
        seed: 0x51c0_0512_0174_0802,
    },
];

#[derive(Clone, Copy)]
enum Route {
    Original,
    Transposed,
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
        let start = unsafe { cuEventDestroy(self.start) };
        let end = unsafe { cuEventDestroy(self.end) };
        ensure!(
            start == 0 && end == 0,
            "cuEventDestroy failed: {start}/{end}"
        );
        Ok(())
    }
}

struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    image: Vec<u8>,
}

impl Guarded {
    fn input(gpu: &dyn GpuBackend, payload: &[u8], prefix: u8, suffix: u8) -> Result<Self> {
        let mut image = vec![prefix; REDZONE + payload.len() + REDZONE];
        image[REDZONE..REDZONE + payload.len()].copy_from_slice(payload);
        image[REDZONE + payload.len()..].fill(suffix);
        let allocation = gpu.alloc(image.len())?;
        gpu.copy_h2d(&image, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            image,
        })
    }

    fn output(gpu: &dyn GpuBackend, payload_len: usize, fill: u8, salt: u8) -> Result<Self> {
        Self::input(gpu, &vec![fill; payload_len], 0xa5 ^ salt, 0x5a ^ salt)
    }

    fn ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.image.len()];
        gpu.copy_d2h(self.allocation, &mut bytes)?;
        Ok(bytes)
    }

    fn payload(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        let bytes = self.read(gpu)?;
        ensure!(
            bytes[..REDZONE] == self.image[..REDZONE],
            "{label}: prefix canary changed"
        );
        let suffix = REDZONE + self.payload_len;
        ensure!(
            bytes[suffix..] == self.image[suffix..],
            "{label}: suffix canary changed"
        );
        Ok(bytes[REDZONE..suffix].to_vec())
    }

    fn immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        ensure!(self.read(gpu)? == self.image, "{label}: bytes changed");
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
    fn new(
        gpu: &dyn GpuBackend,
        packed_bytes: &[u8],
        scale_bytes: &[u8],
        scale2: f32,
        salt: u8,
    ) -> Result<Self> {
        let packed = Guarded::input(gpu, packed_bytes, 0x91 ^ salt, 0x19 ^ salt)?;
        let scales = Guarded::input(gpu, scale_bytes, 0xc3 ^ salt, 0x3c ^ salt)?;
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

struct WeightPair {
    original: UploadedWeight,
    transposed: UploadedWeight,
}

impl WeightPair {
    fn new(gpu: &dyn GpuBackend, shape: Shape) -> Result<Self> {
        let fixture = random_fixture(1, shape.n, shape.k, shape.seed);
        ensure!(
            fixture.physical_n == shape.n,
            "{} physical N changed",
            shape.label
        );
        let (packed_t, scales_t) = transpose_weight(&fixture);
        let scale2 = 0.75f32;
        Ok(Self {
            original: UploadedWeight::new(gpu, &fixture.packed, &fixture.scales, scale2, 0x11)?,
            transposed: UploadedWeight::new(gpu, &packed_t, &scales_t, scale2, 0x22)?,
        })
    }

    fn immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        self.original.immutable(gpu, &format!("{label}/original"))?;
        self.transposed
            .immutable(gpu, &format!("{label}/transposed"))
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.original.free(gpu)?;
        self.transposed.free(gpu)
    }
}

fn transpose_weight(fixture: &Fixture) -> (Vec<u8>, Vec<u8>) {
    let n = fixture.logical_n;
    let k = fixture.k;
    let mut packed = vec![0u8; fixture.packed.len()];
    let mut scales = vec![0u8; fixture.scales.len()];
    for output in 0..n {
        for k_byte in 0..k / 2 {
            packed[k_byte * n + output] = fixture.packed[output * (k / 2) + k_byte];
        }
        for group in 0..k / 16 {
            scales[group * n + output] = fixture.scales[output * (k / 16) + group];
        }
    }
    (packed, scales)
}

struct Kernels {
    original: KernelHandle,
    transposed: KernelHandle,
}

fn strict_switch(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => bail!("{name} must be exactly 0 or 1"),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} must be valid UTF-8"),
    }
}

fn exact_kernel_bundle() -> Result<(String, Vec<(&'static str, &'static str)>)> {
    let model = std::env::var("ATLAS_TARGET_MODEL").context("ATLAS_TARGET_MODEL is required")?;
    let quant = std::env::var("ATLAS_TARGET_QUANT").context("ATLAS_TARGET_QUANT is required")?;
    ensure!(
        model == "qwen3.8-27b" && quant == "nvfp4",
        "requires ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4"
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
        "expected one exact embedded target, found {}",
        matches.len()
    );
    let target = matches.pop().context("exact target disappeared")?;
    Ok((target.target.to_string(), target.modules))
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
            "cuDeviceGetAttribute({label}) failed: {status}"
        );
        Ok(value)
    };
    let major = attribute(ATTR_COMPUTE_CAPABILITY_MAJOR, "CC_MAJOR")?;
    let minor = attribute(ATTR_COMPUTE_CAPABILITY_MINOR, "CC_MINOR")?;
    let multiprocessors = attribute(ATTR_MULTIPROCESSOR_COUNT, "SM_COUNT")?;
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
        "requires GB10 SM121/48SM, got {name:?} cc={major}.{minor} sms={multiprocessors}"
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
            "cuFuncGetAttribute({label}) failed: {status}"
        );
        Ok(value)
    };
    Ok(FunctionResources {
        max_threads: attribute(FUNC_ATTR_MAX_THREADS_PER_BLOCK, "MAX_THREADS")?,
        shared_bytes: attribute(FUNC_ATTR_SHARED_SIZE_BYTES, "SHARED")?,
        local_bytes: attribute(FUNC_ATTR_LOCAL_SIZE_BYTES, "LOCAL")?,
        registers: attribute(FUNC_ATTR_NUM_REGS, "REGISTERS")?,
    })
}

#[allow(clippy::too_many_arguments)]
fn launch(
    route: Route,
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    input: DevicePtr,
    weights: &WeightPair,
    output: DevicePtr,
    rows: usize,
    shape: Shape,
) -> Result<()> {
    match route {
        Route::Original => ops::w4a16_gemm_pipe(
            gpu,
            kernels.original,
            input,
            &weights.original.quant,
            output,
            rows as u32,
            shape.n as u32,
            shape.k as u32,
            stream,
        ),
        Route::Transposed => ops::w4a16_gemm_n128_m128(
            gpu,
            kernels.transposed,
            input,
            &weights.transposed.quant,
            output,
            rows as u32,
            shape.n as u32,
            shape.k as u32,
            stream,
        ),
    }
}

fn input_bytes(rows: usize, shape: Shape) -> Vec<u8> {
    let fixture = random_fixture(rows, 1, shape.k, shape.seed ^ rows as u64 ^ 0xa11c_e55a);
    as_le_bytes(&fixture.activations)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

struct QualityReceipt {
    mismatches: usize,
    first_mismatch: Option<(usize, u16, u16)>,
    cosine: f64,
    relative_rms: f64,
    max_abs_error: f64,
    non_finite: usize,
    original_hash: u64,
    transposed_hash: u64,
}

impl QualityReceipt {
    fn passed(&self) -> bool {
        self.non_finite == 0 && self.cosine >= 0.99
    }
}

fn compare_outputs(left: &[u8], right: &[u8]) -> Result<QualityReceipt> {
    ensure!(left.len() == right.len(), "output sizes differ");
    ensure!(left.len().is_multiple_of(2), "output is not BF16-aligned");
    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    let mut dot = 0.0f64;
    let mut original_sq = 0.0f64;
    let mut transposed_sq = 0.0f64;
    let mut error_sq = 0.0f64;
    let mut max_abs_error = 0.0f64;
    let mut non_finite = 0usize;
    for (element, (original, transposed)) in
        left.chunks_exact(2).zip(right.chunks_exact(2)).enumerate()
    {
        let original_bits = u16::from_le_bytes([original[0], original[1]]);
        let transposed_bits = u16::from_le_bytes([transposed[0], transposed[1]]);
        if original_bits != transposed_bits {
            mismatches += 1;
            first_mismatch.get_or_insert((element, original_bits, transposed_bits));
        }
        let original = f64::from(bf16::from_bits(original_bits).to_f32());
        let transposed = f64::from(bf16::from_bits(transposed_bits).to_f32());
        if !original.is_finite() || !transposed.is_finite() {
            non_finite += 1;
            continue;
        }
        let error = original - transposed;
        dot += original * transposed;
        original_sq += original * original;
        transposed_sq += transposed * transposed;
        error_sq += error * error;
        max_abs_error = max_abs_error.max(error.abs());
    }
    let cosine = dot / (original_sq.sqrt() * transposed_sq.sqrt());
    let relative_rms = (error_sq / original_sq).sqrt();
    Ok(QualityReceipt {
        mismatches,
        first_mismatch,
        cosine,
        relative_rms,
        max_abs_error,
        non_finite,
        original_hash: fnv1a64(left),
        transposed_hash: fnv1a64(right),
    })
}

fn run_quality_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    weights: &WeightPair,
    shape: Shape,
    rows: usize,
) -> Result<()> {
    let label = format!("{}/M={rows}/K={}/N={}", shape.label, shape.k, shape.n);
    let input = Guarded::input(gpu, &input_bytes(rows, shape), 0x31, 0x13)?;
    let output_bytes = rows * shape.n * 2;
    let original = Guarded::output(gpu, output_bytes, 0xa5, 0x11)?;
    let transposed = Guarded::output(gpu, output_bytes, 0x5a, 0x22)?;
    launch(
        Route::Original,
        gpu,
        stream,
        kernels,
        input.ptr(),
        weights,
        original.ptr(),
        rows,
        shape,
    )?;
    launch(
        Route::Transposed,
        gpu,
        stream,
        kernels,
        input.ptr(),
        weights,
        transposed.ptr(),
        rows,
        shape,
    )?;
    gpu.synchronize(stream)?;
    let original_bytes = original.payload(gpu, &format!("{label}/original"))?;
    let transposed_bytes = transposed.payload(gpu, &format!("{label}/transposed"))?;
    let quality = compare_outputs(&original_bytes, &transposed_bytes)?;
    input.immutable(gpu, &format!("{label}/A"))?;
    weights.immutable(gpu, &format!("{label}/weights"))?;
    let first = quality.first_mismatch.map_or_else(
        || "none".to_string(),
        |(element, original, transposed)| {
            format!(
                "row={} column={} original=0x{original:04x} transposed=0x{transposed:04x}",
                element / shape.n,
                element % shape.n
            )
        },
    );
    println!(
        "QUALITY {label}: mismatches={}/{} first_mismatch={} original_hash=fnv1a64:{:016x} transposed_hash=fnv1a64:{:016x} cosine={:.8} relative_rms={:.8} max_abs_error={:.6} non_finite={} canaries=PASS immutable=PASS verdict={}",
        quality.mismatches,
        original_bytes.len() / 2,
        first,
        quality.original_hash,
        quality.transposed_hash,
        quality.cosine,
        quality.relative_rms,
        quality.max_abs_error,
        quality.non_finite,
        if quality.passed() { "PASS" } else { "FAIL" }
    );
    ensure!(
        quality.passed(),
        "{label}: approximate transposed quality screen failed: cosine={:.8} non_finite={}",
        quality.cosine,
        quality.non_finite
    );
    input.free(gpu)?;
    original.free(gpu)?;
    transposed.free(gpu)
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn p90(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[(values.len() * 9).div_ceil(10).saturating_sub(1)]
}

fn mad(values: &[f64]) -> f64 {
    let center = median(values);
    median(
        &values
            .iter()
            .map(|value| (value - center).abs())
            .collect::<Vec<_>>(),
    )
}

#[derive(Debug)]
struct TimingReceipt {
    shape: &'static str,
    rows: usize,
    original_median: f64,
    transposed_median: f64,
    original_p90: f64,
    transposed_p90: f64,
    paired_median: f64,
    paired_mad: f64,
    paired_lower: f64,
}

impl TimingReceipt {
    fn passed(&self) -> bool {
        self.transposed_median < self.original_median
            && self.transposed_p90 < self.original_p90
            && self.paired_lower > 0.0
    }

    fn print(&self) {
        println!(
            "TIMING RECEIPT shape={} M={}: original median/p90={:.3}/{:.3} ms transposed median/p90={:.3}/{:.3} ms paired O-T median={:.3} MAD={:.3} lower3MAD={:.3} verdict={}",
            self.shape,
            self.rows,
            self.original_median,
            self.original_p90,
            self.transposed_median,
            self.transposed_p90,
            self.paired_median,
            self.paired_mad,
            self.paired_lower,
            if self.passed() { "PASS" } else { "FAIL" }
        );
    }
}

fn time_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    weights: &WeightPair,
    shape: Shape,
    rows: usize,
) -> Result<TimingReceipt> {
    let label = format!("timing/{}/M={rows}", shape.label);
    let input = Guarded::input(gpu, &input_bytes(rows, shape), 0x41, 0x14)?;
    let output_bytes = rows * shape.n * 2;
    let original = Guarded::output(gpu, output_bytes, 0xa6, 0x31)?;
    let transposed = Guarded::output(gpu, output_bytes, 0x6a, 0x32)?;
    let events = CudaEventPair::new()?;
    let measure = |route| {
        events.measure(stream, || {
            let output = match route {
                Route::Original => original.ptr(),
                Route::Transposed => transposed.ptr(),
            };
            launch(
                route,
                gpu,
                stream,
                kernels,
                input.ptr(),
                weights,
                output,
                rows,
                shape,
            )
        })
    };

    for order in [
        [Route::Original, Route::Transposed],
        [Route::Transposed, Route::Original],
    ] {
        for route in order {
            let _ = measure(route)?;
        }
    }
    let mut original_ms = Vec::with_capacity(TIMING_CYCLES * 2);
    let mut transposed_ms = Vec::with_capacity(TIMING_CYCLES * 2);
    let mut paired = Vec::with_capacity(TIMING_CYCLES * 2);
    for _ in 0..TIMING_CYCLES {
        for order in [
            [Route::Original, Route::Transposed],
            [Route::Transposed, Route::Original],
        ] {
            let mut o = None;
            let mut t = None;
            for route in order {
                let elapsed = measure(route)?;
                match route {
                    Route::Original => o = Some(elapsed),
                    Route::Transposed => t = Some(elapsed),
                }
            }
            let o = o.context("balanced round omitted original")?;
            let t = t.context("balanced round omitted transposed")?;
            original_ms.push(o);
            transposed_ms.push(t);
            paired.push(o - t);
        }
    }
    let paired_median = median(&paired);
    let paired_mad = mad(&paired);
    let receipt = TimingReceipt {
        shape: shape.label,
        rows,
        original_median: median(&original_ms),
        transposed_median: median(&transposed_ms),
        original_p90: p90(&original_ms),
        transposed_p90: p90(&transposed_ms),
        paired_median,
        paired_mad,
        paired_lower: paired_median - 3.0 * paired_mad,
    };
    input.immutable(gpu, &format!("{label}/A"))?;
    weights.immutable(gpu, &format!("{label}/weights"))?;
    let _ = original.payload(gpu, &format!("{label}/original"))?;
    let _ = transposed.payload(gpu, &format!("{label}/transposed"))?;
    input.free(gpu)?;
    original.free(gpu)?;
    transposed.free(gpu)?;
    events.destroy()?;
    Ok(receipt)
}

fn main() -> Result<()> {
    let timing = strict_switch("ATLAS_PREFILL_FFN_TRANSPOSED_MICROGATE_TIMING")?;
    let (target, modules) = exact_kernel_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let device = current_device_identity()?;
    let stream = gpu.create_stream()?;
    let kernels = Kernels {
        original: gpu.kernel("w4a16", "w4a16_gemm_pipe")?,
        transposed: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
    };
    let original_resources = function_resources(kernels.original)?;
    let transposed_resources = function_resources(kernels.transposed)?;
    ensure!(
        original_resources.max_threads == 640
            && transposed_resources.max_threads == 128
            && original_resources.local_bytes == 0
            && transposed_resources.local_bytes == 0
            && original_resources.registers > 0
            && transposed_resources.registers > 0
            && original_resources.shared_bytes > 0
            && transposed_resources.shared_bytes > 0,
        "resource contract requires original max_threads=640 and transposed max_threads=128 with zero local bytes: original={original_resources:?} transposed={transposed_resources:?}"
    );
    println!(
        "RESOURCES original={original_resources:?} transposed={transposed_resources:?} target={target} device={device:?}"
    );

    for shape in SHAPES {
        let weights = WeightPair::new(gpu, shape)?;
        for rows in SCREEN_ROWS {
            run_quality_case(gpu, stream, &kernels, &weights, shape, rows)?;
        }
        weights.free(gpu)?;
    }

    let mut receipts = Vec::new();
    if timing {
        for shape in SHAPES {
            let weights = WeightPair::new(gpu, shape)?;
            for rows in TIMING_ROWS {
                receipts.push(time_case(gpu, stream, &kernels, &weights, shape, rows)?);
            }
            weights.free(gpu)?;
        }
        for receipt in &receipts {
            receipt.print();
        }
        let failed: Vec<_> = receipts
            .iter()
            .filter(|receipt| !receipt.passed())
            .map(|receipt| format!("{}/M={}", receipt.shape, receipt.rows))
            .collect();
        ensure!(
            failed.is_empty(),
            "aggregate timing gate failed after all rows: {}",
            failed.join(",")
        );
    }

    let identity = format!(
        "target={target} device={:?} ordinal={} cc={}.{} sms={}",
        device.name, device.ordinal, device.major, device.minor, device.multiprocessors
    );
    if timing {
        println!(
            "APPROXIMATE_NOT_BIT_EXACT QUALITY+TIMING SCREEN PASS: cases=10 timing_receipts=4 {identity}"
        );
    } else {
        println!(
            "APPROXIMATE_NOT_BIT_EXACT QUALITY SCREEN PASS - TIMING NOT RUN: cases=10 {identity}"
        );
    }
    Ok(())
}
