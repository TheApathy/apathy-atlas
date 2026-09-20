// SPDX-License-Identifier: AGPL-3.0-only

//! Isolated raw gate for the default-unrouted CUTLASS-layout NVFP4 quantizer.
//!
//! The reference is the installed FlashInfer 0.6.6 SM121 specialization's
//! exported `invokeFP4Quantization<__nv_bfloat16,16>` symbol. Timing is disabled
//! unless `ATLAS_FI_QUANTIZER_TIMING=1`; parity always runs first.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::cutlass_scale_layout::{
    NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const ROWS: [usize; 6] = [127, 128, 129, 2_048, 2_079, 8_192];
const COLS: [usize; 2] = [5_120, 17_408];
const REDZONE: usize = 4_096;
const TIMING_ROUNDS: usize = 9;
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;
const SWIZZLED_128X4: c_int = 0;
const FUNC_ATTR_MAX_THREADS_PER_BLOCK: u32 = 0;
const FUNC_ATTR_SHARED_SIZE_BYTES: u32 = 1;
const FUNC_ATTR_LOCAL_SIZE_BYTES: u32 = 3;
const FUNC_ATTR_NUM_REGS: u32 = 4;
const FLASHINFER_SYMBOL: &[u8] = b"_ZN12tensorrt_llm7kernels21invokeFP4QuantizationI13__nv_bfloat16Li16EEEviiiPKT_PKfPlPibN10flashinfer20QuantizationSFLayoutEibP11CUstream_st\0";

type FlashInferQuantize = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const f32,
    *mut i64,
    *mut i32,
    bool,
    c_int,
    c_int,
    bool,
    *mut c_void,
);

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
    fn cuFuncGetAttribute(value: *mut i32, attribute: u32, function: *mut c_void) -> i32;
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

struct FlashInfer {
    tvm_ffi_handle: *mut c_void,
    handle: *mut c_void,
    quantize: FlashInferQuantize,
}

impl FlashInfer {
    fn open(path: &Path, tvm_ffi_path: &Path) -> Result<Self> {
        ensure!(
            path.is_absolute(),
            "ATLAS_FI_QUANTIZER_LIB must be absolute"
        );
        ensure!(
            tvm_ffi_path.is_absolute(),
            "ATLAS_FI_TVM_FFI_LIB must be absolute"
        );
        let tvm_ffi_path = CString::new(tvm_ffi_path.as_os_str().as_encoded_bytes())
            .context("TVM FFI library path NUL")?;
        let tvm_ffi_handle = unsafe { dlopen(tvm_ffi_path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        ensure!(
            !tvm_ffi_handle.is_null(),
            "TVM FFI dlopen failed: {}",
            dl_error()
        );
        let path = CString::new(path.as_os_str().as_encoded_bytes()).context("library path NUL")?;
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            let error = dl_error();
            let _ = unsafe { dlclose(tvm_ffi_handle) };
            bail!("dlopen failed: {error}");
        }
        let symbol = unsafe { dlsym(handle, FLASHINFER_SYMBOL.as_ptr().cast()) };
        if symbol.is_null() {
            let error = dl_error();
            let _ = unsafe { dlclose(handle) };
            bail!("FlashInfer specialization missing: {error}");
        }
        Ok(Self {
            tvm_ffi_handle,
            handle,
            quantize: unsafe { std::mem::transmute::<*mut c_void, FlashInferQuantize>(symbol) },
        })
    }
}

impl Drop for FlashInfer {
    fn drop(&mut self) {
        let _ = unsafe { dlclose(self.handle) };
        let _ = unsafe { dlclose(self.tvm_ffi_handle) };
    }
}

fn dl_error() -> String {
    let pointer = unsafe { dlerror() };
    if pointer.is_null() {
        "<no dlerror>".to_owned()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

fn required_env(name: &str) -> Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} is required"))?;
    ensure!(!value.is_empty(), "{name} must not be empty");
    Ok(value)
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

fn exact_bundle() -> Result<Vec<(&'static str, &'static str)>> {
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
    ensure!(matches.len() == 1, "expected one exact SM121f target");
    Ok(matches.pop().context("target disappeared")?.modules)
}

fn device_ptr(pointer: DevicePtr) -> *mut c_void {
    pointer.0 as usize as *mut c_void
}

struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    prefix: Vec<u8>,
    suffix: Vec<u8>,
    immutable: Option<Vec<u8>>,
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
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&image, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            prefix,
            suffix,
            immutable: immutable.then_some(payload),
        })
    }

    fn input(gpu: &dyn GpuBackend, stream: u64, payload: Vec<u8>, salt: u8) -> Result<Self> {
        Self::create(gpu, stream, payload, salt, true)
    }

    fn output(gpu: &dyn GpuBackend, stream: u64, len: usize, salt: u8) -> Result<Self> {
        Self::create(gpu, stream, vec![0x7d; len], salt, false)
    }

    fn ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    fn read(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        let mut prefix = vec![0; REDZONE];
        let mut suffix = vec![0; REDZONE];
        gpu.copy_d2h(self.allocation, &mut prefix)?;
        gpu.copy_d2h(self.ptr().offset(self.payload_len), &mut suffix)?;
        ensure!(prefix == self.prefix, "{label}: prefix redzone changed");
        ensure!(suffix == self.suffix, "{label}: suffix redzone changed");
        let mut payload = vec![0; self.payload_len];
        gpu.copy_d2h(self.ptr(), &mut payload)?;
        Ok(payload)
    }

    fn check_immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        let expected = self.immutable.as_ref().context("not immutable")?;
        ensure!(
            self.read(gpu, label)? == *expected,
            "{label}: input changed"
        );
        Ok(())
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fixture {
    Production,
    Halfway,
    SignedZero,
}

impl Fixture {
    const ALL: [Self; 3] = [Self::Production, Self::Halfway, Self::SignedZero];
}

fn input_bytes(rows: usize, cols: usize, fixture: Fixture) -> Vec<u8> {
    // Every 16-value group contains +/-3 so global_scale=896 is exact, while
    // the other magnitudes deliberately avoid E2M1 half-way decision points.
    // Atlas's legacy software converter and SM121's RN instruction have
    // different tie policies; that pre-existing boundary is not attributable
    // to the scale-layout transform being qualified here.
    const PRODUCTION: [f32; 16] = [
        0.0, 0.0, 0.1, 0.1, 0.4, -0.4, 0.9, -0.9, 1.4, -1.4, 1.9, -1.9, 2.9, -2.9, 3.0, -3.0,
    ];
    const HALFWAY: [f32; 16] = [
        3.0, -3.0, 0.125, -0.125, 0.25, -0.25, 0.375, -0.375, 0.625, -0.625, 0.875, -0.875, 1.125,
        -1.125, 1.375, -1.375,
    ];
    const SIGNED_ZERO: [f32; 16] = [
        -0.0, 3.0, 0.5, -0.0, 0.1, -0.1, 0.124, -0.124, 0.249, -0.249, 0.4, -0.4, 0.9, -0.9, 3.0,
        -3.0,
    ];
    let values = match fixture {
        Fixture::Production => &PRODUCTION,
        Fixture::Halfway => &HALFWAY,
        Fixture::SignedZero => &SIGNED_ZERO,
    };
    let mut bytes = Vec::with_capacity(rows * cols * 2);
    for index in 0..rows * cols {
        let mixed = if fixture == Fixture::Production {
            index.wrapping_mul(1_103_515_245).wrapping_add(index >> 7)
        } else {
            index
        };
        let value = values[mixed % values.len()];
        bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    bytes
}

fn require_equal(label: &str, reference: &[u8], candidate: &[u8]) -> Result<()> {
    ensure!(
        reference.len() == candidate.len(),
        "{label}: length mismatch"
    );
    if let Some(index) = reference.iter().zip(candidate).position(|(a, b)| a != b) {
        bail!(
            "{label}: mismatch byte {index}: reference=0x{:02x} candidate=0x{:02x}",
            reference[index],
            candidate[index]
        );
    }
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn launch_candidate(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    packed: DevicePtr,
    scales: DevicePtr,
    global: DevicePtr,
    rows: usize,
    cols: usize,
    stream: u64,
) -> Result<()> {
    let padded_rows = rows.div_ceil(128) * 128;
    KernelLaunch::new(gpu, kernel)
        .grid([u32::try_from(padded_rows.min(96))?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_ptr(global)
        .arg_u32(u32::try_from(rows)?)
        .arg_u32(u32::try_from(cols)?)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_atlas_direct(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    packed: DevicePtr,
    scales: DevicePtr,
    scale2: f32,
    rows: usize,
    cols: usize,
    stream: u64,
) -> Result<()> {
    let padded_rows = rows.div_ceil(128) * 128;
    KernelLaunch::new(gpu, kernel)
        .grid([u32::try_from(padded_rows.min(96))?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_f32(scale2)
        .arg_u32(u32::try_from(rows)?)
        .arg_u32(u32::try_from(cols)?)
        .launch(stream)
}

fn main() -> Result<()> {
    let timing = strict_switch("ATLAS_FI_QUANTIZER_TIMING")?;
    let library = PathBuf::from(required_env("ATLAS_FI_QUANTIZER_LIB")?);
    let tvm_ffi_library = PathBuf::from(required_env("ATLAS_FI_TVM_FFI_LIB")?);
    let modules = exact_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let fi_direct = gpu.kernel(
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_cutlass_128x4",
    )?;
    let atlas_direct = gpu.kernel(
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4",
    )?;
    let atlas = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let fi_resources = function_resources(fi_direct)?;
    let atlas_direct_resources = function_resources(atlas_direct)?;
    ensure!(
        fi_resources.max_threads >= 256
            && fi_resources.local_bytes == 0
            && fi_resources.registers <= 128
            && atlas_direct_resources.max_threads >= 256
            && atlas_direct_resources.local_bytes == 0
            && atlas_direct_resources.registers <= 128,
        "resource gate failed: FI={fi_resources:?} Atlas-direct={atlas_direct_resources:?}"
    );
    println!("RESOURCES fi_direct={fi_resources:?} atlas_direct={atlas_direct_resources:?}");
    let flashinfer = FlashInfer::open(&library, &tvm_ffi_library)?;
    let sm_count = 48;

    for rows in ROWS {
        for cols in COLS {
            for fixture in Fixture::ALL {
                let padded_rows = rows.div_ceil(128) * 128;
                let packed_len = rows * cols / 2;
                let scale_len = padded_rows * cols / 16;
                let input = Guarded::input(gpu, stream, input_bytes(rows, cols, fixture), 0x11)?;
                let global_value = 896.0f32;
                let global =
                    Guarded::input(gpu, stream, global_value.to_le_bytes().to_vec(), 0x12)?;
                let reference_packed = Guarded::output(gpu, stream, packed_len, 0x21)?;
                let reference_scales = Guarded::output(gpu, stream, scale_len, 0x22)?;
                let candidate_packed = Guarded::output(gpu, stream, packed_len, 0x31)?;
                let candidate_scales = Guarded::output(gpu, stream, scale_len, 0x32)?;
                let atlas_packed = Guarded::output(gpu, stream, packed_len, 0x41)?;
                let atlas_scales = Guarded::output(gpu, stream, rows * cols / 16, 0x42)?;
                let atlas_direct_packed = Guarded::output(gpu, stream, packed_len, 0x51)?;
                let atlas_direct_scales = Guarded::output(gpu, stream, scale_len, 0x52)?;

                let invoke_reference = || unsafe {
                    (flashinfer.quantize)(
                        1,
                        rows as c_int,
                        cols as c_int,
                        device_ptr(input.ptr()),
                        device_ptr(global.ptr()).cast(),
                        device_ptr(reference_packed.ptr()).cast(),
                        device_ptr(reference_scales.ptr()).cast(),
                        false,
                        SWIZZLED_128X4,
                        sm_count,
                        false,
                        stream as usize as *mut c_void,
                    )
                };

                invoke_reference();
                launch_candidate(
                    gpu,
                    fi_direct,
                    input.ptr(),
                    candidate_packed.ptr(),
                    candidate_scales.ptr(),
                    global.ptr(),
                    rows,
                    cols,
                    stream,
                )?;
                ops::quantize_bf16_to_nvfp4(
                    gpu,
                    atlas,
                    input.ptr(),
                    atlas_packed.ptr(),
                    atlas_scales.ptr(),
                    global_value.recip(),
                    rows as u32,
                    cols as u32,
                    stream,
                )?;
                launch_atlas_direct(
                    gpu,
                    atlas_direct,
                    input.ptr(),
                    atlas_direct_packed.ptr(),
                    atlas_direct_scales.ptr(),
                    global_value.recip(),
                    rows,
                    cols,
                    stream,
                )?;
                gpu.synchronize(stream)?;
                let rp = reference_packed.read(gpu, "reference-packed")?;
                let rs = reference_scales.read(gpu, "reference-scales")?;
                let cp = candidate_packed.read(gpu, "candidate-packed")?;
                let cs = candidate_scales.read(gpu, "candidate-scales")?;
                let ap = atlas_packed.read(gpu, "atlas-packed")?;
                let atlas_logical = atlas_scales.read(gpu, "atlas-scales")?;
                let adp = atlas_direct_packed.read(gpu, "atlas-direct-packed")?;
                let ads = atlas_direct_scales.read(gpu, "atlas-direct-scales")?;
                require_equal("packed E2M1", &rp, &cp)?;
                require_equal("physical E4M3 scales", &rs, &cs)?;
                require_equal("Atlas-vs-Atlas-direct packed E2M1", &ap, &adp)?;
                let candidate_logical = deinterleave_nvfp4_scales_128x4(
                    &cs,
                    &[padded_rows, cols / NVFP4_GROUP_SIZE],
                    NVFP4_GROUP_SIZE,
                )?;
                require_equal(
                    "Atlas-vs-FI-direct logical E4M3 scales",
                    &atlas_logical,
                    &candidate_logical[..rows * cols / NVFP4_GROUP_SIZE],
                )?;
                let atlas_direct_logical = deinterleave_nvfp4_scales_128x4(
                    &ads,
                    &[padded_rows, cols / NVFP4_GROUP_SIZE],
                    NVFP4_GROUP_SIZE,
                )?;
                require_equal(
                    "Atlas-vs-Atlas-direct logical E4M3 scales",
                    &atlas_logical,
                    &atlas_direct_logical[..rows * cols / NVFP4_GROUP_SIZE],
                )?;
                if fixture == Fixture::Production {
                    require_equal("Atlas-vs-FI-direct packed E2M1", &ap, &cp)?;
                } else {
                    ensure!(ap != cp, "{fixture:?}: expected Atlas/FI nibble divergence");
                    if fixture == Fixture::SignedZero {
                        ensure!(
                            ap[0] == 0x70 && cp[0] == 0x78 && ap[1] == 0x02 && cp[1] == 0x82,
                            "signed-zero sentinels changed: Atlas/FI {:02x}/{:02x} and {:02x}/{:02x}",
                            ap[0],
                            cp[0],
                            ap[1],
                            cp[1]
                        );
                    }
                }
                ensure!(
                    candidate_logical[rows * cols / NVFP4_GROUP_SIZE..]
                        .iter()
                        .all(|byte| *byte == 0),
                    "direct-layout padded scale rows are not zero"
                );
                ensure!(
                    atlas_direct_logical[rows * cols / NVFP4_GROUP_SIZE..]
                        .iter()
                        .all(|byte| *byte == 0),
                    "Atlas-direct padded scale rows are not zero"
                );
                ensure!(
                    cs.iter().chain(&ads).all(|byte| byte & 0x7f != 0x7f),
                    "direct quantizer emitted E4M3 NaN"
                );
                input.check_immutable(gpu, "BF16 input")?;
                global.check_immutable(gpu, "global scale")?;
                println!(
                    "PARITY fixture={fixture:?} M={rows} K={cols} atlas_packed_hash=fnv1a64:{:016x} atlas_scale_hash=fnv1a64:{:016x} flashinfer_packed=EXACT flashinfer_physical_scale=EXACT atlas_direct_packed=EXACT atlas_direct_logical_scale=EXACT family_relation={} padded_rows={padded_rows} padded_scale_zero=PASS redzones=PASS immutable=PASS finite=PASS",
                    fnv1a64(&adp),
                    fnv1a64(&ads),
                    if fixture == Fixture::Production {
                        "EXACT"
                    } else {
                        "EXPECTED_DIVERGENCE"
                    },
                );

                if timing && fixture == Fixture::Production {
                    let measure = |arm: u8| -> Result<f64> {
                        gpu.synchronize(stream)?;
                        let start = Instant::now();
                        match arm {
                            0 => invoke_reference(),
                            1 => launch_candidate(
                                gpu,
                                fi_direct,
                                input.ptr(),
                                candidate_packed.ptr(),
                                candidate_scales.ptr(),
                                global.ptr(),
                                rows,
                                cols,
                                stream,
                            )?,
                            2 => ops::quantize_bf16_to_nvfp4(
                                gpu,
                                atlas,
                                input.ptr(),
                                atlas_packed.ptr(),
                                atlas_scales.ptr(),
                                global_value.recip(),
                                rows as u32,
                                cols as u32,
                                stream,
                            )?,
                            3 => launch_atlas_direct(
                                gpu,
                                atlas_direct,
                                input.ptr(),
                                atlas_direct_packed.ptr(),
                                atlas_direct_scales.ptr(),
                                global_value.recip(),
                                rows,
                                cols,
                                stream,
                            )?,
                            _ => bail!("invalid timing arm {arm}"),
                        }
                        gpu.synchronize(stream)?;
                        Ok(start.elapsed().as_secs_f64() * 1_000.0)
                    };
                    let (mut reference_ms, mut fi_direct_ms, mut atlas_ms, mut atlas_direct_ms) =
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    for _ in 0..TIMING_ROUNDS {
                        for order in [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]] {
                            let mut quartet = [0.0; 4];
                            for arm in order {
                                quartet[arm as usize] = measure(arm)?;
                            }
                            reference_ms.push(quartet[0]);
                            fi_direct_ms.push(quartet[1]);
                            atlas_ms.push(quartet[2]);
                            atlas_direct_ms.push(quartet[3]);
                        }
                    }
                    println!(
                        "TIMING M={rows} K={cols} flashinfer_median_ms={:.6} fi_direct_median_ms={:.6} atlas_logical_median_ms={:.6} atlas_direct_median_ms={:.6} atlas_logical_minus_atlas_direct_ms={:.6} evidence_only=true",
                        median(&reference_ms),
                        median(&fi_direct_ms),
                        median(&atlas_ms),
                        median(&atlas_direct_ms),
                        median(
                            &atlas_ms
                                .iter()
                                .zip(&atlas_direct_ms)
                                .map(|(a, c)| a - c)
                                .collect::<Vec<_>>()
                        ),
                    );
                    gpu.synchronize(stream)?;
                    require_equal(
                        "post-timing packed",
                        &reference_packed.read(gpu, "timed-reference-packed")?,
                        &candidate_packed.read(gpu, "timed-candidate-packed")?,
                    )?;
                    require_equal(
                        "post-timing scales",
                        &reference_scales.read(gpu, "timed-reference-scales")?,
                        &candidate_scales.read(gpu, "timed-candidate-scales")?,
                    )?;
                    require_equal(
                        "post-timing Atlas packed",
                        &atlas_packed.read(gpu, "timed-atlas-packed")?,
                        &atlas_direct_packed.read(gpu, "timed-atlas-direct-packed")?,
                    )?;
                    let timed_atlas_direct_scales =
                        atlas_direct_scales.read(gpu, "timed-atlas-direct-scales")?;
                    let timed_atlas_direct_logical = deinterleave_nvfp4_scales_128x4(
                        &timed_atlas_direct_scales,
                        &[padded_rows, cols / NVFP4_GROUP_SIZE],
                        NVFP4_GROUP_SIZE,
                    )?;
                    require_equal(
                        "post-timing Atlas logical scales",
                        &atlas_scales.read(gpu, "timed-atlas-scales")?,
                        &timed_atlas_direct_logical[..rows * cols / NVFP4_GROUP_SIZE],
                    )?;
                }

                for buffer in [
                    &input,
                    &global,
                    &reference_packed,
                    &reference_scales,
                    &candidate_packed,
                    &candidate_scales,
                    &atlas_packed,
                    &atlas_scales,
                    &atlas_direct_packed,
                    &atlas_direct_scales,
                ] {
                    buffer.free(gpu)?;
                }
            }
        }
    }
    println!("FINAL verdict=PASS exact_all_shapes=true timing={timing} production_route=false");
    Ok(())
}
