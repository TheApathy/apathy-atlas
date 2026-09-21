// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed raw screen for extending Qwen3.8 `PREFILL_PROJ_FAST` target
//! projections from the W4A16 transposed parent to dynamic BF16 -> NVFP4 plus
//! K-major M256 W4A4. This example does not alter production routing.
//!
//! Attention Q/G/K/V share one quantized activation, matching the intended
//! production chain. The dynamic arm includes absmax, required host readback,
//! quantization, and every GEMM. A second mixed arm captures a finite positive
//! scale before timing for checkpoint-static attention and SSM-out, while SSM
//! QKVZ retains the complete dynamic path.
//!
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//! ATLAS_PREFILL_PROJ_W4A4_MICROGATE_TIMING=1 \
//! cargo run --release -p spark-model --features cuda \
//!   --example qwen38_prefill_proj_w4a4_microgate
//! ```

#[allow(dead_code)]
#[path = "w4a16_exact_lm_head_microtest/data.rs"]
mod data;

use std::ffi::{CStr, c_char, c_void};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use data::{Fixture, as_le_bytes, random_fixture};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const REDZONE: usize = 4 * 1_024;
const ROWS: [usize; 2] = [2_079, 8_192];
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
}

#[derive(Clone, Copy)]
struct Projection {
    label: &'static str,
    n: usize,
    seed: u64,
}

#[derive(Clone, Copy)]
struct ChainSpec {
    label: &'static str,
    k: usize,
    projections: &'static [Projection],
    layer_count: usize,
    checkpoint_static: bool,
    seed: u64,
}

const ATTENTION_QKV: [Projection; 3] = [
    Projection {
        label: "qg",
        n: 12_288,
        seed: 0x11,
    },
    Projection {
        label: "k",
        n: 1_024,
        seed: 0x12,
    },
    Projection {
        label: "v",
        n: 1_024,
        seed: 0x13,
    },
];
const ATTENTION_O: [Projection; 1] = [Projection {
    label: "o",
    n: 5_120,
    seed: 0x21,
}];
const SSM_QKVZ: [Projection; 1] = [Projection {
    label: "qkvz",
    n: 12_288,
    seed: 0x31,
}];
const SSM_OUT: [Projection; 1] = [Projection {
    label: "out",
    n: 5_120,
    seed: 0x41,
}];

const CHAINS: [ChainSpec; 4] = [
    ChainSpec {
        label: "attention-qgkv",
        k: 5_120,
        projections: &ATTENTION_QKV,
        layer_count: 16,
        checkpoint_static: true,
        seed: 0xa1,
    },
    ChainSpec {
        label: "attention-o",
        k: 6_144,
        projections: &ATTENTION_O,
        layer_count: 16,
        checkpoint_static: true,
        seed: 0xa2,
    },
    ChainSpec {
        label: "ssm-qkvz",
        k: 5_120,
        projections: &SSM_QKVZ,
        layer_count: 48,
        checkpoint_static: false,
        seed: 0xb1,
    },
    ChainSpec {
        label: "ssm-out",
        k: 4_096,
        projections: &SSM_OUT,
        layer_count: 48,
        checkpoint_static: true,
        seed: 0xb2,
    },
];

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

struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    image: Vec<u8>,
}

impl Guarded {
    fn input(gpu: &dyn GpuBackend, payload: &[u8], salt: u8) -> Result<Self> {
        let mut image = vec![0xa5 ^ salt; REDZONE + payload.len() + REDZONE];
        image[REDZONE..REDZONE + payload.len()].copy_from_slice(payload);
        image[REDZONE + payload.len()..].fill(0x5a ^ salt);
        let allocation = gpu.alloc(image.len())?;
        gpu.copy_h2d(&image, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            image,
        })
    }

    fn output(gpu: &dyn GpuBackend, payload_len: usize, salt: u8) -> Result<Self> {
        // Signaling-NaN BF16 catches unwritten output elements in the quality gate.
        let mut payload = vec![0u8; payload_len];
        for pair in payload.chunks_exact_mut(2) {
            pair.copy_from_slice(&0x7f81u16.to_le_bytes());
        }
        Self::input(gpu, &payload, salt)
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
        ensure!(
            self.read(gpu)? == self.image,
            "{label}: input bytes changed"
        );
        Ok(())
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}

struct Weight {
    packed: Guarded,
    scales: Guarded,
    quant: QuantizedWeight,
    projection: Projection,
}

impl Weight {
    fn new(gpu: &dyn GpuBackend, k: usize, projection: Projection) -> Result<Self> {
        let fixture = random_fixture(1, projection.n, k, 0x5100_0000 ^ projection.seed);
        let (packed, scales) = transpose_weight(&fixture);
        let packed = Guarded::input(gpu, &packed, projection.seed as u8)?;
        let scales = Guarded::input(gpu, &scales, projection.seed as u8 ^ 0x55)?;
        let quant = QuantizedWeight {
            weight: packed.ptr(),
            weight_scale: scales.ptr(),
            weight_scale_2: 0.75,
            input_scale: DevicePtr::NULL,
        };
        Ok(Self {
            packed,
            scales,
            quant,
            projection,
        })
    }

    fn immutable(&self, gpu: &dyn GpuBackend, chain: &str) -> Result<()> {
        self.packed
            .immutable(gpu, &format!("{chain}/{}/weight", self.projection.label))?;
        self.scales
            .immutable(gpu, &format!("{chain}/{}/scales", self.projection.label))
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.packed.free(gpu)?;
        self.scales.free(gpu)
    }
}

fn transpose_weight(fixture: &Fixture) -> (Vec<u8>, Vec<u8>) {
    let (n, k) = (fixture.logical_n, fixture.k);
    let mut packed = vec![0; fixture.packed.len()];
    let mut scales = vec![0; fixture.scales.len()];
    for row in 0..n {
        for byte in 0..k / 2 {
            packed[byte * n + row] = fixture.packed[row * (k / 2) + byte];
        }
        for group in 0..k / 16 {
            scales[group * n + row] = fixture.scales[row * (k / 16) + group];
        }
    }
    (packed, scales)
}

struct Kernels {
    parent: KernelHandle,
    absmax: KernelHandle,
    quantize: KernelHandle,
    candidate: KernelHandle,
}

struct ChainBuffers {
    input: Guarded,
    packed: Guarded,
    scales: Guarded,
    absmax: Guarded,
    weights: Vec<Weight>,
    parent_outputs: Vec<Guarded>,
    candidate_outputs: Vec<Guarded>,
}

impl ChainBuffers {
    fn new(gpu: &dyn GpuBackend, chain: ChainSpec, rows: usize) -> Result<Self> {
        let fixture = random_fixture(rows, 1, chain.k, chain.seed ^ rows as u64);
        let input = Guarded::input(gpu, &as_le_bytes(&fixture.activations), 0x01)?;
        let packed = Guarded::input(gpu, &vec![0x31; rows * chain.k / 2], 0x02)?;
        let scales = Guarded::input(gpu, &vec![0x42; rows * chain.k / 16], 0x03)?;
        let absmax = Guarded::input(gpu, &0.0f32.to_le_bytes(), 0x04)?;
        let mut weights = Vec::new();
        let mut parent_outputs = Vec::new();
        let mut candidate_outputs = Vec::new();
        for projection in chain.projections {
            weights.push(Weight::new(gpu, chain.k, *projection)?);
            let bytes = rows * projection.n * 2;
            parent_outputs.push(Guarded::output(gpu, bytes, 0x10 ^ projection.seed as u8)?);
            candidate_outputs.push(Guarded::output(gpu, bytes, 0x60 ^ projection.seed as u8)?);
        }
        Ok(Self {
            input,
            packed,
            scales,
            absmax,
            weights,
            parent_outputs,
            candidate_outputs,
        })
    }

    fn validate_and_free(&self, gpu: &dyn GpuBackend, chain: ChainSpec) -> Result<()> {
        self.input
            .immutable(gpu, &format!("{}/input", chain.label))?;
        for weight in &self.weights {
            weight.immutable(gpu, chain.label)?;
        }
        let _ = self
            .packed
            .payload(gpu, &format!("{}/activation-packed", chain.label))?;
        let _ = self
            .scales
            .payload(gpu, &format!("{}/activation-scales", chain.label))?;
        let _ = self
            .absmax
            .payload(gpu, &format!("{}/activation-absmax", chain.label))?;
        self.input.free(gpu)?;
        self.packed.free(gpu)?;
        self.scales.free(gpu)?;
        self.absmax.free(gpu)?;
        for weight in &self.weights {
            weight.free(gpu)?;
        }
        for output in &self.parent_outputs {
            output.free(gpu)?;
        }
        for output in &self.candidate_outputs {
            output.free(gpu)?;
        }
        Ok(())
    }
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
    ensure!(
        std::env::var("ATLAS_TARGET_MODEL").as_deref() == Ok("qwen3.8-27b"),
        "requires ATLAS_TARGET_MODEL=qwen3.8-27b"
    );
    ensure!(
        std::env::var("ATLAS_TARGET_QUANT").as_deref() == Ok("nvfp4"),
        "requires ATLAS_TARGET_QUANT=nvfp4"
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
    ensure!(
        unsafe { cuCtxGetDevice(&mut ordinal) } == 0 && ordinal >= 0,
        "cuCtxGetDevice failed"
    );
    let attribute = |kind, label| -> Result<i32> {
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
    ensure!(
        unsafe { cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, ordinal) } == 0,
        "cuDeviceGetName failed"
    );
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
    let attribute = |kind, label| -> Result<i32> {
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

fn launch_parent(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    chain: ChainSpec,
    rows: usize,
    b: &ChainBuffers,
) -> Result<()> {
    for ((weight, output), projection) in b
        .weights
        .iter()
        .zip(&b.parent_outputs)
        .zip(chain.projections)
    {
        ops::w4a16_gemm_n128_m128(
            gpu,
            kernels.parent,
            b.input.ptr(),
            &weight.quant,
            output.ptr(),
            rows as u32,
            projection.n as u32,
            chain.k as u32,
            stream,
        )?;
    }
    Ok(())
}

fn dynamic_scale(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    chain: ChainSpec,
    rows: usize,
    b: &ChainBuffers,
) -> Result<f32> {
    gpu.memset_async(b.absmax.ptr(), 0, 4, stream)?;
    ops::nvfp4_global_absmax(
        gpu,
        kernels.absmax,
        b.input.ptr(),
        b.absmax.ptr(),
        (rows * chain.k) as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let bytes = b.absmax.payload(gpu, &format!("{}/absmax", chain.label))?;
    let global_max = f32::from_le_bytes(bytes.try_into().expect("four-byte absmax"));
    ensure!(
        global_max.is_finite() && global_max >= 0.0,
        "{} invalid absmax {global_max}",
        chain.label
    );
    let scale2 = if global_max > 0.0 {
        global_max / (6.0 * 448.0)
    } else {
        1.0
    };
    ensure!(
        scale2.is_finite() && scale2 > 0.0,
        "{} scale2 must be finite and positive",
        chain.label
    );
    Ok(scale2)
}

fn quantize_candidate(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    chain: ChainSpec,
    rows: usize,
    b: &ChainBuffers,
    scale2: f32,
) -> Result<()> {
    ensure!(
        scale2.is_finite() && scale2 > 0.0,
        "{} admitted scale must be finite and positive",
        chain.label
    );
    ops::quantize_bf16_to_nvfp4(
        gpu,
        kernels.quantize,
        b.input.ptr(),
        b.packed.ptr(),
        b.scales.ptr(),
        scale2,
        rows as u32,
        chain.k as u32,
        stream,
    )
}

fn launch_candidate_gemms(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    chain: ChainSpec,
    rows: usize,
    b: &ChainBuffers,
    activation_scale2: f32,
) -> Result<()> {
    for ((weight, output), projection) in b
        .weights
        .iter()
        .zip(&b.candidate_outputs)
        .zip(chain.projections)
    {
        ops::nvfp4_nvfp4_gemm_kmajor_m256(
            gpu,
            kernels.candidate,
            b.packed.ptr(),
            b.scales.ptr(),
            weight.packed.ptr(),
            weight.scales.ptr(),
            activation_scale2 * weight.quant.weight_scale_2,
            output.ptr(),
            rows as u32,
            projection.n as u32,
            chain.k as u32,
            stream,
        )?;
    }
    Ok(())
}

fn launch_candidate_dynamic(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    chain: ChainSpec,
    rows: usize,
    b: &ChainBuffers,
) -> Result<()> {
    let scale2 = dynamic_scale(gpu, stream, kernels, chain, rows, b)?;
    quantize_candidate(gpu, stream, kernels, chain, rows, b, scale2)?;
    launch_candidate_gemms(gpu, stream, kernels, chain, rows, b, scale2)
}

fn launch_candidate_mixed(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    chain: ChainSpec,
    rows: usize,
    b: &ChainBuffers,
    admitted_scale: Option<f32>,
) -> Result<()> {
    if chain.checkpoint_static {
        let scale2 = admitted_scale.context("static-eligible chain omitted admitted scale")?;
        quantize_candidate(gpu, stream, kernels, chain, rows, b, scale2)?;
        launch_candidate_gemms(gpu, stream, kernels, chain, rows, b, scale2)
    } else {
        ensure!(
            admitted_scale.is_none(),
            "dynamic chain unexpectedly supplied a static scale"
        );
        launch_candidate_dynamic(gpu, stream, kernels, chain, rows, b)
    }
}

struct Quality {
    cosine: f64,
    relative_rms: f64,
    max_abs: f64,
    non_finite: usize,
    mismatches: usize,
    parent_hash: u64,
    candidate_hash: u64,
}

impl Quality {
    fn passed(&self) -> bool {
        self.non_finite == 0 && self.cosine >= 0.99 && self.relative_rms <= 0.15
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn compare(parent: &[u8], candidate: &[u8]) -> Result<Quality> {
    ensure!(
        parent.len() == candidate.len() && parent.len().is_multiple_of(2),
        "invalid output lengths"
    );
    let (mut dot, mut p2, mut c2, mut e2, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut non_finite, mut mismatches) = (0usize, 0usize);
    for (p, c) in parent.chunks_exact(2).zip(candidate.chunks_exact(2)) {
        let pb = u16::from_le_bytes([p[0], p[1]]);
        let cb = u16::from_le_bytes([c[0], c[1]]);
        mismatches += usize::from(pb != cb);
        let pf = f64::from(bf16::from_bits(pb).to_f32());
        let cf = f64::from(bf16::from_bits(cb).to_f32());
        if !pf.is_finite() || !cf.is_finite() {
            non_finite += 1;
            continue;
        }
        let error = pf - cf;
        dot += pf * cf;
        p2 += pf * pf;
        c2 += cf * cf;
        e2 += error * error;
        max_abs = max_abs.max(error.abs());
    }
    ensure!(p2 > 0.0 && c2 > 0.0, "degenerate output norm");
    Ok(Quality {
        cosine: dot / (p2.sqrt() * c2.sqrt()),
        relative_rms: (e2 / p2).sqrt(),
        max_abs,
        non_finite,
        mismatches,
        parent_hash: fnv1a64(parent),
        candidate_hash: fnv1a64(candidate),
    })
}

fn run_quality(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    chain: ChainSpec,
    rows: usize,
    b: &ChainBuffers,
) -> Result<()> {
    launch_parent(gpu, stream, kernels, chain, rows, b)?;
    launch_candidate_dynamic(gpu, stream, kernels, chain, rows, b)?;
    gpu.synchronize(stream)?;
    for ((parent, candidate), projection) in b
        .parent_outputs
        .iter()
        .zip(&b.candidate_outputs)
        .zip(chain.projections)
    {
        let label = format!(
            "{}/{}/M={rows}/K={}/N={}",
            chain.label, projection.label, chain.k, projection.n
        );
        let parent = parent.payload(gpu, &format!("{label}/parent"))?;
        let candidate = candidate.payload(gpu, &format!("{label}/candidate"))?;
        let q = compare(&parent, &candidate)?;
        println!(
            "QUALITY {label}: mismatches={}/{} parent_hash=fnv1a64:{:016x} candidate_hash=fnv1a64:{:016x} cosine={:.8} relative_rms={:.8} max_abs_error={:.6} non_finite={} canaries=PASS verdict={}",
            q.mismatches,
            parent.len() / 2,
            q.parent_hash,
            q.candidate_hash,
            q.cosine,
            q.relative_rms,
            q.max_abs,
            q.non_finite,
            if q.passed() { "PASS" } else { "FAIL" }
        );
        ensure!(
            q.passed(),
            "{label}: quality gate failed (cosine >= 0.99, relative_rms <= 0.15, finite required)"
        );
    }
    b.input
        .immutable(gpu, &format!("{}/M={rows}/input", chain.label))?;
    for weight in &b.weights {
        weight.immutable(gpu, chain.label)?;
    }
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn p90(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v[(v.len() * 9).div_ceil(10).saturating_sub(1)]
}
fn mad(values: &[f64]) -> f64 {
    let m = median(values);
    median(&values.iter().map(|x| (x - m).abs()).collect::<Vec<_>>())
}

struct Timing {
    arm: &'static str,
    label: &'static str,
    rows: usize,
    layers: usize,
    parent_median: f64,
    candidate_median: f64,
    parent_p90: f64,
    candidate_p90: f64,
    paired_median: f64,
    paired_mad: f64,
}
impl Timing {
    fn passed(&self) -> bool {
        self.candidate_median < self.parent_median
            && self.candidate_p90 < self.parent_p90
            && self.paired_median - 3.0 * self.paired_mad > 0.0
    }
    fn print(&self) {
        println!(
            "TIMING arm={} shape={} M={} layers={} parent median/p90={:.3}/{:.3} ms candidate_chain median/p90={:.3}/{:.3} ms paired P-C median={:.3} MAD={:.3} lower3MAD={:.3} verdict={}",
            self.arm,
            self.label,
            self.rows,
            self.layers,
            self.parent_median,
            self.parent_p90,
            self.candidate_median,
            self.candidate_p90,
            self.paired_median,
            self.paired_mad,
            self.paired_median - 3.0 * self.paired_mad,
            if self.passed() { "PASS" } else { "FAIL" }
        );
    }
}

fn time_chain(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    chain: ChainSpec,
    rows: usize,
    b: &ChainBuffers,
    arm: &'static str,
    admitted_scale: Option<f32>,
) -> Result<Timing> {
    let measure = |candidate: bool| -> Result<f64> {
        gpu.synchronize(stream)?;
        let start = Instant::now();
        if candidate {
            match arm {
                "dynamic" => launch_candidate_dynamic(gpu, stream, kernels, chain, rows, b)?,
                "mixed-static" => {
                    launch_candidate_mixed(gpu, stream, kernels, chain, rows, b, admitted_scale)?
                }
                _ => bail!("unknown timing arm {arm}"),
            }
        } else {
            launch_parent(gpu, stream, kernels, chain, rows, b)?;
        }
        gpu.synchronize(stream)?;
        Ok(start.elapsed().as_secs_f64() * 1_000.0)
    };
    for candidate in [false, true, true, false] {
        let _ = measure(candidate)?;
    }
    let (mut parent, mut candidate, mut paired) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..TIMING_CYCLES {
        for order in [[false, true], [true, false]] {
            let mut p = None;
            let mut c = None;
            for is_candidate in order {
                let ms = measure(is_candidate)?;
                if is_candidate {
                    c = Some(ms);
                } else {
                    p = Some(ms);
                }
            }
            let p = p.context("balanced round omitted parent")?;
            let c = c.context("balanced round omitted candidate")?;
            parent.push(p);
            candidate.push(c);
            paired.push(p - c);
        }
    }
    let pm = median(&paired);
    let md = mad(&paired);
    Ok(Timing {
        arm,
        label: chain.label,
        rows,
        layers: chain.layer_count,
        parent_median: median(&parent),
        candidate_median: median(&candidate),
        parent_p90: p90(&parent),
        candidate_p90: p90(&candidate),
        paired_median: pm,
        paired_mad: md,
    })
}

fn main() -> Result<()> {
    let timing = strict_switch("ATLAS_PREFILL_PROJ_W4A4_MICROGATE_TIMING")?;
    let (target, modules) = exact_kernel_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let device = current_device_identity()?;
    let stream = gpu.create_stream()?;
    let kernels = Kernels {
        parent: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
        absmax: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        candidate: gpu.kernel("nvfp4_cutlass", "nvfp4_nvfp4_gemm_kmajor_m256")?,
    };
    let parent_resources = function_resources(kernels.parent)?;
    let candidate_resources = function_resources(kernels.candidate)?;
    ensure!(
        parent_resources.max_threads >= 128
            && parent_resources.local_bytes == 0
            && candidate_resources.max_threads >= 512
            && candidate_resources.local_bytes == 0
            && candidate_resources.registers > 0
            && candidate_resources.registers <= 128
            && candidate_resources.shared_bytes > 0,
        "resource stop gate failed: parent={parent_resources:?} candidate={candidate_resources:?}"
    );
    println!(
        "RESOURCES parent={parent_resources:?} candidate={candidate_resources:?} target={target} device={device:?}"
    );

    let mut timings = Vec::new();
    for rows in ROWS {
        for chain in CHAINS {
            let buffers = ChainBuffers::new(gpu, chain, rows)?;
            run_quality(gpu, stream, &kernels, chain, rows, &buffers)?;
            if timing {
                timings.push(time_chain(
                    gpu, stream, &kernels, chain, rows, &buffers, "dynamic", None,
                )?);
                let admitted_scale = if chain.checkpoint_static {
                    let scale = dynamic_scale(gpu, stream, &kernels, chain, rows, &buffers)?;
                    ensure!(
                        scale.is_finite() && scale > 0.0,
                        "{} failed static-scale admission",
                        chain.label
                    );
                    println!(
                        "STATIC_SCALE shape={} M={} scale2={:.9} capture=OUTSIDE_TIMING verdict=PASS",
                        chain.label, rows, scale
                    );
                    Some(scale)
                } else {
                    None
                };
                timings.push(time_chain(
                    gpu,
                    stream,
                    &kernels,
                    chain,
                    rows,
                    &buffers,
                    "mixed-static",
                    admitted_scale,
                )?);
            }
            buffers.validate_and_free(gpu, chain)?;
        }
    }
    for receipt in &timings {
        receipt.print();
    }
    if timing {
        let mut aggregate_failed = Vec::new();
        for rows in ROWS {
            for arm in ["dynamic", "mixed-static"] {
                let row: Vec<_> = timings
                    .iter()
                    .filter(|r| r.rows == rows && r.arm == arm)
                    .collect();
                ensure!(
                    row.len() == CHAINS.len(),
                    "aggregate {arm}/M={rows} omitted a chain"
                );
                let parent: f64 = row.iter().map(|r| r.parent_median * r.layers as f64).sum();
                let candidate: f64 = row
                    .iter()
                    .map(|r| r.candidate_median * r.layers as f64)
                    .sum();
                let passed = candidate < parent;
                println!(
                    "AGGREGATE arm={arm} M={rows}: layer_weighted_parent={parent:.3} layer_weighted_candidate={candidate:.3} projected_saving={:.3} ms verdict={}",
                    parent - candidate,
                    if passed { "PASS" } else { "FAIL" }
                );
                if !passed {
                    aggregate_failed.push(format!("{arm}/M={rows}"));
                }
            }
        }
        let receipt_failed: Vec<_> = timings
            .iter()
            .filter(|r| !r.passed())
            .map(|r| format!("{}/{}/M={}", r.arm, r.label, r.rows))
            .collect();
        ensure!(
            aggregate_failed.is_empty() && receipt_failed.is_empty(),
            "timing gate failed after printing all receipts and aggregates: aggregates=[{}] receipts=[{}]",
            aggregate_failed.join(", "),
            receipt_failed.join(", ")
        );
    }
    println!(
        "APPROXIMATE_NOT_BIT_EXACT {} SCREEN PASS: quality_cases=12 timing_receipts={} target={} device={:?} ordinal={} cc={}.{} sms={}",
        if timing {
            "QUALITY+TIMING"
        } else {
            "QUALITY (TIMING NOT RUN)"
        },
        timings.len(),
        target,
        device.name,
        device.ordinal,
        device.major,
        device.minor,
        device.multiprocessors
    );
    Ok(())
}
