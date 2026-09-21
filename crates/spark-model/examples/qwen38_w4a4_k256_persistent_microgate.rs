// SPDX-License-Identifier: AGPL-3.0-only

//! Raw, default-unrouted qualification gate for the SM121 K256 static-
//! persistent W4A4 shadow.  It compares already-quantized inputs directly
//! against the promoted K-major M256 kernel, so byte equality is required.
//!
//! This program is intentionally not a production selector.  Running it uses
//! a CUDA device and must be separately authorized:
//!
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//! cargo run --release -p spark-model --features cuda \
//!   --example qwen38_w4a4_k256_persistent_microgate
//! ```
//!
//! Add `ATLAS_W4A4_K256_PERSISTENT_TIMING=1` for balanced AB/BA timing.

use std::ffi::{CStr, c_char, c_void};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const REDZONE: usize = 4 * 1_024;
const ROWS: [u32; 3] = [2_048, 2_079, 8_192];
const TIMING_ROUNDS: usize = 9;
const EXPECTED_SMS: i32 = 48;

const ATTR_MULTIPROCESSOR_COUNT: u32 = 16;
const ATTR_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
const ATTR_COMPUTE_CAPABILITY_MINOR: u32 = 76;
const ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: u32 = 97;
const FUNC_ATTR_MAX_THREADS_PER_BLOCK: u32 = 0;
const FUNC_ATTR_SHARED_SIZE_BYTES: u32 = 1;
const FUNC_ATTR_LOCAL_SIZE_BYTES: u32 = 3;
const FUNC_ATTR_NUM_REGS: u32 = 4;
const FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES: u32 = 8;
const CANDIDATE_DYNAMIC_SHARED_BYTES: u32 = 55_296;

unsafe extern "C" {
    fn cuCtxGetDevice(device: *mut i32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attribute: u32, device: i32) -> i32;
    fn cuDeviceGetName(name: *mut c_char, length: i32, device: i32) -> i32;
    fn cuFuncGetAttribute(value: *mut i32, attribute: u32, function: *mut c_void) -> i32;
    fn cuFuncSetAttribute(function: *mut c_void, attribute: u32, value: i32) -> i32;
}

#[derive(Clone, Copy)]
struct Projection {
    label: &'static str,
    n: u32,
    k: u32,
    seed: u64,
}

const PROJECTIONS: [Projection; 3] = [
    Projection {
        label: "gate",
        n: 17_408,
        k: 5_120,
        seed: 0x671a_7e01,
    },
    Projection {
        label: "up",
        n: 17_408,
        k: 5_120,
        seed: 0x671a_7e02,
    },
    Projection {
        label: "down",
        n: 5_120,
        k: 17_408,
        seed: 0xd04a_0003,
    },
];

#[derive(Debug)]
struct Resources {
    max_threads: i32,
    shared_bytes: i32,
    local_bytes: i32,
    registers: i32,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn packed_fp4(&mut self) -> u8 {
        const NIBBLES: [u8; 14] = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15];
        let lo = NIBBLES[(self.next() as usize) % NIBBLES.len()];
        let hi = NIBBLES[(self.next() as usize) % NIBBLES.len()];
        lo | (hi << 4)
    }

    fn scale(&mut self) -> u8 {
        const SCALES: [u8; 5] = [0x30, 0x34, 0x38, 0x3c, 0x40];
        SCALES[(self.next() as usize) % SCALES.len()]
    }
}

struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    prefix: Vec<u8>,
    suffix: Vec<u8>,
    immutable_payload: Option<Vec<u8>>,
}

impl Guarded {
    fn create(
        gpu: &dyn GpuBackend,
        stream: u64,
        payload: Vec<u8>,
        salt: u8,
        immutable: bool,
    ) -> Result<Self> {
        let prefix = vec![0xa5 ^ salt; REDZONE];
        let suffix = vec![0x5a ^ salt; REDZONE];
        let mut image = Vec::with_capacity(REDZONE + payload.len() + REDZONE);
        image.extend_from_slice(&prefix);
        image.extend_from_slice(&payload);
        image.extend_from_slice(&suffix);
        let allocation = gpu.alloc(image.len())?;
        // `alloc` may enqueue initialization.  Drain it before the synchronous
        // canary upload so a late memset cannot erase the guard image.
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&image, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            prefix,
            suffix,
            immutable_payload: immutable.then_some(payload),
        })
    }

    fn input(gpu: &dyn GpuBackend, stream: u64, payload: Vec<u8>, salt: u8) -> Result<Self> {
        Self::create(gpu, stream, payload, salt, true)
    }

    fn output(gpu: &dyn GpuBackend, stream: u64, bytes: usize, salt: u8) -> Result<Self> {
        let mut payload = vec![0u8; bytes];
        for pair in payload.chunks_exact_mut(2) {
            pair.copy_from_slice(&0x7f81u16.to_le_bytes());
        }
        Self::create(gpu, stream, payload, salt, false)
    }

    fn ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    fn check_guards(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        let mut prefix = vec![0; REDZONE];
        let mut suffix = vec![0; REDZONE];
        gpu.copy_d2h(self.allocation, &mut prefix)?;
        gpu.copy_d2h(self.ptr().offset(self.payload_len), &mut suffix)?;
        ensure!(prefix == self.prefix, "{label}: prefix canary changed");
        ensure!(suffix == self.suffix, "{label}: suffix canary changed");
        Ok(())
    }

    fn payload(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        self.check_guards(gpu, label)?;
        let mut payload = vec![0; self.payload_len];
        gpu.copy_d2h(self.ptr(), &mut payload)?;
        Ok(payload)
    }

    fn check_immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        let expected = self
            .immutable_payload
            .as_ref()
            .with_context(|| format!("{label}: not declared immutable"))?;
        ensure!(
            self.payload(gpu, label)? == *expected,
            "{label}: payload changed"
        );
        Ok(())
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}

fn strict_switch(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => bail!("{name} must be exactly 0 or 1"),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} must be UTF-8"),
    }
}

fn exact_bundle() -> Result<(String, Vec<(&'static str, &'static str)>)> {
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
        "expected one embedded SM121 target, found {}",
        matches.len()
    );
    let target = matches.pop().context("embedded target disappeared")?;
    Ok((target.target.to_string(), target.modules))
}

fn device_identity() -> Result<(String, i32, i32)> {
    let mut ordinal = -1;
    ensure!(
        unsafe { cuCtxGetDevice(&mut ordinal) } == 0 && ordinal >= 0,
        "cuCtxGetDevice failed"
    );
    let attr = |kind, label| -> Result<i32> {
        let mut value = -1;
        let status = unsafe { cuDeviceGetAttribute(&mut value, kind, ordinal) };
        ensure!(
            status == 0,
            "cuDeviceGetAttribute({label}) failed: {status}"
        );
        Ok(value)
    };
    let major = attr(ATTR_COMPUTE_CAPABILITY_MAJOR, "major")?;
    let minor = attr(ATTR_COMPUTE_CAPABILITY_MINOR, "minor")?;
    let sms = attr(ATTR_MULTIPROCESSOR_COUNT, "sms")?;
    let max_optin_shared = attr(ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, "max_optin_shared")?;
    let mut name = [0 as c_char; 256];
    ensure!(
        unsafe { cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, ordinal) } == 0,
        "cuDeviceGetName failed"
    );
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    ensure!(
        major == 12 && minor == 1 && sms == EXPECTED_SMS,
        "requires GB10 SM121/48SM, got {name} cc={major}.{minor} sms={sms}"
    );
    Ok((
        format!("{name} cc={major}.{minor} sms={sms} max_optin_shared={max_optin_shared}"),
        sms,
        max_optin_shared,
    ))
}

fn resources(kernel: KernelHandle) -> Result<Resources> {
    let function = kernel.0 as usize as *mut c_void;
    let attr = |kind, label| -> Result<i32> {
        let mut value = -1;
        let status = unsafe { cuFuncGetAttribute(&mut value, kind, function) };
        ensure!(status == 0, "cuFuncGetAttribute({label}) failed: {status}");
        Ok(value)
    };
    Ok(Resources {
        max_threads: attr(FUNC_ATTR_MAX_THREADS_PER_BLOCK, "max_threads")?,
        shared_bytes: attr(FUNC_ATTR_SHARED_SIZE_BYTES, "shared")?,
        local_bytes: attr(FUNC_ATTR_LOCAL_SIZE_BYTES, "local")?,
        registers: attr(FUNC_ATTR_NUM_REGS, "registers")?,
    })
}

#[allow(clippy::too_many_arguments)]
fn launch_candidate(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    stream: u64,
    sms: u32,
    a: DevicePtr,
    a_scale: DevicePtr,
    b: DevicePtr,
    b_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    ensure!(
        m > 0 && n.is_multiple_of(128) && k.is_multiple_of(256),
        "candidate geometry rejected M={m} N={n} K={k}"
    );
    let tiles = m.div_ceil(128) * (n / 128);
    KernelLaunch::new(gpu, kernel)
        .grid([tiles.min(sms), 1, 1])
        .block([256, 1, 1])
        .shared_mem(CANDIDATE_DYNAMIC_SHARED_BYTES)
        .arg_ptr(a)
        .arg_ptr(a_scale)
        .arg_ptr(b)
        .arg_ptr(b_scale)
        .arg_f32(0.001)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

fn deterministic_bytes(len: usize, seed: u64, scales: bool) -> Vec<u8> {
    let mut rng = Rng(seed);
    (0..len)
        .map(|_| {
            if scales {
                rng.scale()
            } else {
                rng.packed_fp4()
            }
        })
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn require_finite(label: &str, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len().is_multiple_of(2),
        "{label}: odd BF16 byte count"
    );
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        let value = bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32();
        ensure!(
            value.is_finite(),
            "{label}: nonfinite output at element {index}"
        );
    }
    Ok(())
}

fn require_equal(label: &str, expected: &[u8], actual: &[u8]) -> Result<()> {
    if expected == actual {
        return Ok(());
    }
    let byte = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a != b)
        .context("mismatched buffers lacked a differing byte")?;
    bail!(
        "{label}: exact parity failed at byte {byte} (element {}, expected=0x{:02x}, actual=0x{:02x})",
        byte / 2,
        expected[byte],
        actual[byte]
    )
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn p90(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() * 9).div_ceil(10)).saturating_sub(1);
    sorted[index]
}

fn mad(values: &[f64]) -> f64 {
    let centre = median(values);
    let deviations: Vec<_> = values.iter().map(|v| (v - centre).abs()).collect();
    median(&deviations)
}

#[allow(clippy::too_many_arguments)]
fn time_pair(
    gpu: &dyn GpuBackend,
    stream: u64,
    parent: KernelHandle,
    candidate: KernelHandle,
    sms: u32,
    a: DevicePtr,
    a_scale: DevicePtr,
    b: DevicePtr,
    b_scale: DevicePtr,
    parent_out: DevicePtr,
    candidate_out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(f64, f64, f64, f64, f64, f64)> {
    let measure = |is_candidate: bool| -> Result<f64> {
        gpu.synchronize(stream)?;
        let start = Instant::now();
        if is_candidate {
            launch_candidate(
                gpu,
                candidate,
                stream,
                sms,
                a,
                a_scale,
                b,
                b_scale,
                candidate_out,
                m,
                n,
                k,
            )?;
        } else {
            ops::nvfp4_nvfp4_gemm_kmajor_m256(
                gpu, parent, a, a_scale, b, b_scale, 0.001, parent_out, m, n, k, stream,
            )?;
        }
        gpu.synchronize(stream)?;
        Ok(start.elapsed().as_secs_f64() * 1_000.0)
    };
    let _ = measure(false)?;
    let _ = measure(true)?;
    let (mut p, mut c, mut delta) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..TIMING_ROUNDS {
        for order in [[false, true], [true, false]] {
            let (mut pv, mut cv) = (None, None);
            for arm in order {
                let value = measure(arm)?;
                if arm {
                    cv = Some(value)
                } else {
                    pv = Some(value)
                }
            }
            let pv = pv.context("ABBA parent omitted")?;
            let cv = cv.context("ABBA candidate omitted")?;
            p.push(pv);
            c.push(cv);
            delta.push(pv - cv);
        }
    }
    let delta_median = median(&delta);
    let delta_mad = mad(&delta);
    Ok((
        median(&p),
        median(&c),
        p90(&p),
        p90(&c),
        delta_median,
        delta_mad,
    ))
}

fn run_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    parent: KernelHandle,
    candidate: KernelHandle,
    sms: u32,
    projection: Projection,
    m: u32,
    timing: bool,
) -> Result<()> {
    let a = Guarded::input(
        gpu,
        stream,
        deterministic_bytes(
            (m as usize) * (projection.k as usize) / 2,
            projection.seed ^ m as u64,
            false,
        ),
        0x11,
    )?;
    let a_scale = Guarded::input(
        gpu,
        stream,
        deterministic_bytes(
            (m as usize) * (projection.k as usize) / 16,
            projection.seed ^ 0xa5a5 ^ m as u64,
            true,
        ),
        0x22,
    )?;
    let b = Guarded::input(
        gpu,
        stream,
        deterministic_bytes(
            (projection.k as usize / 2) * projection.n as usize,
            projection.seed ^ 0xb4b4,
            false,
        ),
        0x33,
    )?;
    let b_scale = Guarded::input(
        gpu,
        stream,
        deterministic_bytes(
            (projection.k as usize / 16) * projection.n as usize,
            projection.seed ^ 0xc3c3,
            true,
        ),
        0x44,
    )?;
    let output_bytes = m as usize * projection.n as usize * 2;
    let parent_out = Guarded::output(gpu, stream, output_bytes, 0x55)?;
    let candidate_out = Guarded::output(gpu, stream, output_bytes, 0x66)?;

    ops::nvfp4_nvfp4_gemm_kmajor_m256(
        gpu,
        parent,
        a.ptr(),
        a_scale.ptr(),
        b.ptr(),
        b_scale.ptr(),
        0.001,
        parent_out.ptr(),
        m,
        projection.n,
        projection.k,
        stream,
    )?;
    launch_candidate(
        gpu,
        candidate,
        stream,
        sms,
        a.ptr(),
        a_scale.ptr(),
        b.ptr(),
        b_scale.ptr(),
        candidate_out.ptr(),
        m,
        projection.n,
        projection.k,
    )?;
    gpu.synchronize(stream)?;

    let label = format!(
        "{}/M={m}/K={}/N={}",
        projection.label, projection.k, projection.n
    );
    let expected = parent_out.payload(gpu, &format!("{label}/parent"))?;
    let actual = candidate_out.payload(gpu, &format!("{label}/candidate"))?;
    require_finite(&format!("{label}/parent"), &expected)?;
    require_finite(&format!("{label}/candidate"), &actual)?;
    require_equal(&label, &expected, &actual)?;
    a.check_immutable(gpu, &format!("{label}/A"))?;
    a_scale.check_immutable(gpu, &format!("{label}/A-scale"))?;
    b.check_immutable(gpu, &format!("{label}/B"))?;
    b_scale.check_immutable(gpu, &format!("{label}/B-scale"))?;
    println!(
        "PARITY {label}: bytes={} hash=fnv1a64:{:016x} canaries=PASS immutable=PASS finite=PASS verdict=PASS",
        actual.len(),
        fnv1a64(&actual)
    );

    if timing {
        let (pm, cm, pp90, cp90, dm, dmad) = time_pair(
            gpu,
            stream,
            parent,
            candidate,
            sms,
            a.ptr(),
            a_scale.ptr(),
            b.ptr(),
            b_scale.ptr(),
            parent_out.ptr(),
            candidate_out.ptr(),
            m,
            projection.n,
            projection.k,
        )?;
        let lower_3mad = dm - 3.0 * dmad;
        let passed = cm < pm && cp90 < pp90 && lower_3mad > 0.0;
        println!(
            "TIMING {label}: parent_median={pm:.4} candidate_median={cm:.4} parent_p90={pp90:.4} candidate_p90={cp90:.4} paired_delta_median={dm:.4} paired_mad={dmad:.4} lower_3mad={lower_3mad:.4} verdict={}",
            if passed { "PASS" } else { "FAIL" }
        );
        let timed_parent = parent_out.payload(gpu, &format!("{label}/timed-parent"))?;
        let timed_candidate = candidate_out.payload(gpu, &format!("{label}/timed-candidate"))?;
        require_equal(&format!("{label}/timed"), &timed_parent, &timed_candidate)?;
        a.check_immutable(gpu, &format!("{label}/timed-A"))?;
        a_scale.check_immutable(gpu, &format!("{label}/timed-A-scale"))?;
        b.check_immutable(gpu, &format!("{label}/timed-B"))?;
        b_scale.check_immutable(gpu, &format!("{label}/timed-B-scale"))?;
        ensure!(passed, "{label}: strict timing gate failed");
    }

    a.free(gpu)?;
    a_scale.free(gpu)?;
    b.free(gpu)?;
    b_scale.free(gpu)?;
    parent_out.free(gpu)?;
    candidate_out.free(gpu)?;
    Ok(())
}

fn main() -> Result<()> {
    let timing = strict_switch("ATLAS_W4A4_K256_PERSISTENT_TIMING")?;
    let (target, modules) = exact_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let (device, sms, max_optin_shared) = device_identity()?;
    let stream = gpu.create_stream()?;
    let parent = gpu.kernel("nvfp4_cutlass", "nvfp4_nvfp4_gemm_kmajor_m256")?;
    let candidate = gpu.kernel(
        "cutlass_nvfp4_gemm_persistent",
        "nvfp4_nvfp4_gemm_kmajor_k256_persistent",
    )?;
    let candidate_function = candidate.0 as usize as *mut c_void;
    let optin_status = unsafe {
        cuFuncSetAttribute(
            candidate_function,
            FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            CANDIDATE_DYNAMIC_SHARED_BYTES as i32,
        )
    };
    ensure!(
        optin_status == 0,
        "candidate dynamic-shared opt-in to {CANDIDATE_DYNAMIC_SHARED_BYTES} bytes failed: {optin_status}"
    );
    let parent_resources = resources(parent)?;
    let candidate_resources = resources(candidate)?;
    ensure!(
        parent_resources.max_threads >= 512
            && parent_resources.local_bytes == 0
            && candidate_resources.max_threads >= 256
            && candidate_resources.registers > 0
            && candidate_resources.registers <= 128
            && candidate_resources.shared_bytes <= 1_024
            && candidate_resources.local_bytes == 0
            && max_optin_shared
                >= candidate_resources.shared_bytes + CANDIDATE_DYNAMIC_SHARED_BYTES as i32,
        "resource stop gate failed: parent={parent_resources:?} candidate={candidate_resources:?}"
    );
    println!(
        "RESOURCES target={target} device={device:?} parent={parent_resources:?} candidate={candidate_resources:?} candidate_dynamic_shared={CANDIDATE_DYNAMIC_SHARED_BYTES} schedule=static-persistent grid<=SMs TMA=false blocker=missing-host-CUtensorMap-ABI verdict=PASS"
    );

    for m in ROWS {
        for projection in PROJECTIONS {
            run_case(
                gpu, stream, parent, candidate, sms as u32, projection, m, timing,
            )?;
        }
    }
    println!("FINAL verdict=PASS timing={timing} production_route=false");
    Ok(())
}
