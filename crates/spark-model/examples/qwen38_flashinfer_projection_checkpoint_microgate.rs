// SPDX-License-Identifier: AGPL-3.0-only

//! Real-checkpoint projection parity gate for the isolated FlashInfer SM121 C ABI.
//!
//! This example deliberately bypasses the production model loader: `WeightStore`
//! uploads the complete checkpoint to GPU and therefore is not a safe raw-gate
//! seam. Instead, a bounded safetensors reader seeks only explicitly named
//! attention/SSM packed bytes, E4M3 scales, and global scalars. Merged QGKV or
//! QKVZ is admitted only when every component has bit-identical input scale and
//! derived alpha, because the native ABI consumes one scalar per GEMM.
//!
//! CPU-only checkpoint admission (does not initialize CUDA):
//!
//! ```text
//! ATLAS_FI_CHECKPOINT_DIR=/path/to/Qwen3.8-27B-NVFP4 \
//! ATLAS_FI_LAYER=3 ATLAS_FI_PROJECTION=attention-qgkv \
//! ATLAS_FI_METADATA_ONLY=1 \
//! cargo run -p spark-model --features cuda \
//!   --example qwen38_flashinfer_projection_checkpoint_microgate
//! ```
//!
//! GPU parity is a separately authorized action.  It additionally requires
//! `ATLAS_FI_CABI_LIB=/absolute/path/libatlas_fi_fp4_sm121.so`, exact Atlas
//! target variables, and `ATLAS_FI_M` in {2048,2079,8192}. Timing remains off
//! unless `ATLAS_FI_CHECKPOINT_TIMING=1` is explicitly supplied after parity.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::cutlass_scale_layout::{
    NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4, interleave_nvfp4_scales_128x4,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const REDZONE: usize = 4 * 1_024;
const TIMING_ROUNDS: usize = 9;
const STATUS_OK: c_int = 0;
const STATUS_CUTLASS_ERROR: c_int = -2;
const RTLD_NOW: c_int = 2;

type WorkspaceFn = unsafe extern "C" fn(c_int, c_int, c_int, c_int, c_int, *mut usize) -> c_int;
type GemmFn = unsafe extern "C" fn(
    c_int,
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const f32,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut c_void,
    usize,
    *mut c_void,
) -> c_int;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

struct Cabi {
    handle: *mut c_void,
    workspace: WorkspaceFn,
    gemm: GemmFn,
    last_error: LastErrorFn,
}

impl Cabi {
    fn open(path: &Path) -> Result<Self> {
        ensure!(path.is_absolute(), "ATLAS_FI_CABI_LIB must be absolute");
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .context("C ABI path contains a NUL byte")?;
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        ensure!(!handle.is_null(), "dlopen failed: {}", dl_error());
        let symbol = |name: &'static [u8]| -> Result<*mut c_void> {
            let pointer = unsafe { dlsym(handle, name.as_ptr().cast()) };
            ensure!(
                !pointer.is_null(),
                "dlsym({}) failed: {}",
                String::from_utf8_lossy(&name[..name.len() - 1]),
                dl_error()
            );
            Ok(pointer)
        };
        let workspace = symbol(b"atlas_fi_nvfp4_sm121_workspace_size\0")?;
        let gemm = symbol(b"atlas_fi_nvfp4_sm121_bf16\0")?;
        let last_error = symbol(b"atlas_fi_nvfp4_sm121_last_error\0")?;
        Ok(Self {
            handle,
            workspace: unsafe { std::mem::transmute::<*mut c_void, WorkspaceFn>(workspace) },
            gemm: unsafe { std::mem::transmute::<*mut c_void, GemmFn>(gemm) },
            last_error: unsafe { std::mem::transmute::<*mut c_void, LastErrorFn>(last_error) },
        })
    }

    fn error(&self) -> String {
        let pointer = unsafe { (self.last_error)() };
        if pointer.is_null() {
            "<null last_error>".to_owned()
        } else {
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Drop for Cabi {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = unsafe { dlclose(self.handle) };
        }
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

#[derive(Clone, Copy, Debug)]
enum Projection {
    AttentionQgkv,
    AttentionOutput,
    SsmQkvz,
    SsmOutput,
}

impl Projection {
    fn parse() -> Result<Self> {
        match required_env("ATLAS_FI_PROJECTION")?.as_str() {
            "attention-qgkv" => Ok(Self::AttentionQgkv),
            "attention-o" => Ok(Self::AttentionOutput),
            "ssm-qkvz" => Ok(Self::SsmQkvz),
            "ssm-o" => Ok(Self::SsmOutput),
            value => bail!(
                "ATLAS_FI_PROJECTION must be attention-qgkv, attention-o, ssm-qkvz, or ssm-o; got {value}"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::AttentionQgkv => "attention_qgkv",
            Self::AttentionOutput => "attention_o",
            Self::SsmQkvz => "ssm_qkvz",
            Self::SsmOutput => "ssm_o",
        }
    }

    fn shape(self) -> (usize, usize) {
        match self {
            Self::AttentionQgkv => (14_336, 5_120),
            Self::AttentionOutput => (5_120, 6_144),
            Self::SsmQkvz => (16_384, 5_120),
            Self::SsmOutput => (5_120, 6_144),
        }
    }
}

#[derive(Debug)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    offsets: [u64; 2],
}

struct TensorBytes {
    bytes: Vec<u8>,
    meta: TensorMeta,
    shard: PathBuf,
}

struct Checkpoint {
    root: PathBuf,
    index: serde_json::Map<String, serde_json::Value>,
}

impl Checkpoint {
    fn open(root: &Path) -> Result<Self> {
        ensure!(
            root.is_absolute(),
            "ATLAS_FI_CHECKPOINT_DIR must be absolute"
        );
        let root = root
            .canonicalize()
            .with_context(|| format!("canonicalize {}", root.display()))?;
        let index_path = root.join("model.safetensors.index.json");
        let value: serde_json::Value = serde_json::from_reader(
            File::open(&index_path).with_context(|| format!("open {}", index_path.display()))?,
        )
        .context("parse safetensors index")?;
        let index = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .context("index lacks object weight_map")?
            .clone();
        Ok(Self { root, index })
    }

    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn tensor(&self, name: &str) -> Result<TensorBytes> {
        let shard_name = self
            .index
            .get(name)
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("index missing tensor {name}"))?;
        let shard = self
            .root
            .join(shard_name)
            .canonicalize()
            .with_context(|| format!("canonicalize shard {shard_name}"))?;
        ensure!(
            shard.starts_with(&self.root),
            "index shard escapes checkpoint root: {}",
            shard.display()
        );
        let mut file = File::open(&shard).with_context(|| format!("open {}", shard.display()))?;
        let mut len8 = [0u8; 8];
        file.read_exact(&mut len8)?;
        let header_len = u64::from_le_bytes(len8);
        ensure!(
            header_len <= 128 << 20,
            "safetensors header too large: {header_len}"
        );
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;
        let value: serde_json::Value =
            serde_json::from_slice(&header).context("parse safetensors header")?;
        let tensor = value
            .get(name)
            .with_context(|| format!("shard header missing {name}"))?;
        let dtype = tensor
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("{name}: missing dtype"))?
            .to_owned();
        let shape = tensor
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .with_context(|| format!("{name}: missing shape"))?
            .iter()
            .map(|value| value.as_u64().context("non-u64 shape").map(|v| v as usize))
            .collect::<Result<Vec<_>>>()?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .with_context(|| format!("{name}: missing offsets"))?;
        ensure!(
            offsets.len() == 2,
            "{name}: offsets must contain two values"
        );
        let offsets = [
            offsets[0].as_u64().context("bad start offset")?,
            offsets[1].as_u64().context("bad end offset")?,
        ];
        ensure!(offsets[1] >= offsets[0], "{name}: reversed offsets");
        let data_start = 8u64
            .checked_add(header_len)
            .context("header offset overflow")?;
        let start = data_start
            .checked_add(offsets[0])
            .context("tensor offset overflow")?;
        let len = offsets[1] - offsets[0];
        ensure!(
            start.checked_add(len).context("tensor end overflow")? <= file.metadata()?.len(),
            "{name}: tensor range exceeds shard"
        );
        let len = usize::try_from(len).context("tensor too large for host")?;
        let mut bytes = vec![0u8; len];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut bytes)?;
        Ok(TensorBytes {
            bytes,
            meta: TensorMeta {
                dtype,
                shape,
                offsets,
            },
            shard,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScaleConvention {
    CompressedReciprocal,
    ModelOptIdentity,
}

struct ProjectionData {
    prefix: String,
    packed: TensorBytes,
    scales: TensorBytes,
    input_scale2: f32,
    weight_scale2: f32,
    convention: ScaleConvention,
}

fn scalar(tensor: TensorBytes, name: &str) -> Result<f32> {
    ensure!(tensor.meta.dtype == "F32", "{name}: expected F32");
    ensure!(
        tensor.meta.shape == [1] || tensor.meta.shape.is_empty(),
        "{name}: expected scalar shape, got {:?}",
        tensor.meta.shape
    );
    ensure!(tensor.bytes.len() == 4, "{name}: expected four bytes");
    let value = f32::from_le_bytes(tensor.bytes.try_into().expect("four checked bytes"));
    ensure!(
        value.is_finite() && value > 0.0,
        "{name}: invalid scalar {value}"
    );
    Ok(value)
}

fn load_projection(
    checkpoint: &Checkpoint,
    layer: usize,
    projection: Projection,
) -> Result<ProjectionData> {
    ensure!(layer < 64, "ATLAS_FI_LAYER must be in 0..64");
    let base = format!("model.language_model.layers.{layer}");
    match projection {
        Projection::AttentionQgkv => merge_projection(
            projection,
            [
                load_single_projection(
                    checkpoint,
                    format!("{base}.self_attn.q_proj"),
                    12_288,
                    5_120,
                )?,
                load_single_projection(
                    checkpoint,
                    format!("{base}.self_attn.k_proj"),
                    1_024,
                    5_120,
                )?,
                load_single_projection(
                    checkpoint,
                    format!("{base}.self_attn.v_proj"),
                    1_024,
                    5_120,
                )?,
            ],
        ),
        Projection::AttentionOutput => {
            load_single_projection(checkpoint, format!("{base}.self_attn.o_proj"), 5_120, 6_144)
        }
        Projection::SsmQkvz => merge_projection(
            projection,
            [
                load_single_projection(
                    checkpoint,
                    format!("{base}.linear_attn.in_proj_qkv"),
                    10_240,
                    5_120,
                )?,
                load_single_projection(
                    checkpoint,
                    format!("{base}.linear_attn.in_proj_z"),
                    6_144,
                    5_120,
                )?,
            ],
        ),
        Projection::SsmOutput => load_single_projection(
            checkpoint,
            format!("{base}.linear_attn.out_proj"),
            5_120,
            6_144,
        ),
    }
}

fn load_single_projection(
    checkpoint: &Checkpoint,
    prefix: String,
    n: usize,
    k: usize,
) -> Result<ProjectionData> {
    let compressed = checkpoint.contains(&format!("{prefix}.weight_packed"));
    let (packed_name, input_name, weight_name, convention) = if compressed {
        (
            format!("{prefix}.weight_packed"),
            format!("{prefix}.input_global_scale"),
            format!("{prefix}.weight_global_scale"),
            ScaleConvention::CompressedReciprocal,
        )
    } else {
        (
            format!("{prefix}.weight"),
            format!("{prefix}.input_scale"),
            format!("{prefix}.weight_scale_2"),
            ScaleConvention::ModelOptIdentity,
        )
    };
    let packed = checkpoint.tensor(&packed_name)?;
    let scales = checkpoint.tensor(&format!("{prefix}.weight_scale"))?;
    let raw_input = scalar(checkpoint.tensor(&input_name)?, &input_name)?;
    let raw_weight = scalar(checkpoint.tensor(&weight_name)?, &weight_name)?;
    let (input_scale2, weight_scale2) = match convention {
        ScaleConvention::CompressedReciprocal => (raw_input.recip(), raw_weight.recip()),
        ScaleConvention::ModelOptIdentity => (raw_input, raw_weight),
    };
    ensure!(
        input_scale2.is_finite()
            && input_scale2 > 0.0
            && weight_scale2.is_finite()
            && weight_scale2 > 0.0,
        "derived scales are invalid"
    );
    ensure!(packed.meta.dtype == "U8", "{packed_name}: expected U8");
    ensure!(
        packed.meta.shape == [n, k / 2],
        "{packed_name}: shape {:?}, expected [{n},{}]",
        packed.meta.shape,
        k / 2
    );
    ensure!(
        packed.bytes.len() == n * k / 2,
        "{packed_name}: byte length mismatch"
    );
    ensure!(
        matches!(scales.meta.dtype.as_str(), "F8_E4M3" | "U8"),
        "weight_scale: expected F8_E4M3/U8, got {}",
        scales.meta.dtype
    );
    ensure!(
        scales.meta.shape == [n, k / 16],
        "weight_scale: shape {:?}, expected [{n},{}]",
        scales.meta.shape,
        k / 16
    );
    ensure!(
        scales.bytes.len() == n * k / 16,
        "weight_scale: byte length mismatch"
    );
    Ok(ProjectionData {
        prefix,
        packed,
        scales,
        input_scale2,
        weight_scale2,
        convention,
    })
}

fn merge_projection<const PARTS: usize>(
    projection: Projection,
    parts: [ProjectionData; PARTS],
) -> Result<ProjectionData> {
    ensure!(PARTS > 1, "merged projection requires multiple components");
    let first = &parts[0];
    let input_bits = first.input_scale2.to_bits();
    let alpha_bits = (first.input_scale2 * first.weight_scale2).to_bits();
    let input_scale2 = first.input_scale2;
    let weight_scale2 = first.weight_scale2;
    let convention = first.convention;
    for part in &parts[1..] {
        ensure!(
            part.convention == first.convention,
            "{} merge mixes scale conventions",
            projection.name()
        );
        ensure!(
            part.input_scale2.to_bits() == input_bits,
            "{} merge requires bit-identical input scales: {} != {}",
            projection.name(),
            first.input_scale2,
            part.input_scale2
        );
        ensure!(
            (part.input_scale2 * part.weight_scale2).to_bits() == alpha_bits,
            "{} merge requires bit-identical derived alpha; component {} has distinct weight/global scale",
            projection.name(),
            part.prefix
        );
    }
    let (n, k) = projection.shape();
    let mut packed_bytes = Vec::with_capacity(n * k / 2);
    let mut scale_bytes = Vec::with_capacity(n * k / 16);
    let mut prefixes = Vec::with_capacity(PARTS);
    let mut shards = Vec::with_capacity(PARTS);
    for part in parts {
        prefixes.push(part.prefix);
        shards.push(part.packed.shard.display().to_string());
        packed_bytes.extend_from_slice(&part.packed.bytes);
        scale_bytes.extend_from_slice(&part.scales.bytes);
    }
    ensure!(
        packed_bytes.len() == n * k / 2,
        "merged packed length mismatch"
    );
    ensure!(
        scale_bytes.len() == n * k / 16,
        "merged scale length mismatch"
    );
    Ok(ProjectionData {
        prefix: prefixes.join("+"),
        packed: TensorBytes {
            bytes: packed_bytes,
            meta: TensorMeta {
                dtype: "U8".to_owned(),
                shape: vec![n, k / 2],
                offsets: [0, (n * k / 2) as u64],
            },
            shard: PathBuf::from(format!("merged:[{}]", shards.join(","))),
        },
        scales: TensorBytes {
            bytes: scale_bytes,
            meta: TensorMeta {
                dtype: "F8_E4M3".to_owned(),
                shape: vec![n, k / 16],
                offsets: [0, (n * k / 16) as u64],
            },
            shard: PathBuf::from("merged-scales"),
        },
        input_scale2,
        weight_scale2,
        convention,
    })
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

fn parse_usize(name: &str) -> Result<usize> {
    required_env(name)?
        .parse()
        .with_context(|| format!("{name} must be an unsigned integer"))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn transpose(input: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let mut output = vec![0u8; input.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = input[row * cols + col];
        }
    }
    output
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

    fn packed(&mut self) -> u8 {
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

    fn payload(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        let mut prefix = vec![0u8; REDZONE];
        let mut suffix = vec![0u8; REDZONE];
        gpu.copy_d2h(self.allocation, &mut prefix)?;
        gpu.copy_d2h(self.ptr().offset(self.payload_len), &mut suffix)?;
        ensure!(prefix == self.prefix, "{label}: prefix redzone changed");
        ensure!(suffix == self.suffix, "{label}: suffix redzone changed");
        let mut payload = vec![0u8; self.payload_len];
        gpu.copy_d2h(self.ptr(), &mut payload)?;
        Ok(payload)
    }

    fn check_immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        let expected = self.immutable.as_ref().context("buffer is not immutable")?;
        ensure!(
            self.payload(gpu, label)? == *expected,
            "{label}: input changed"
        );
        Ok(())
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}

fn device_ptr(pointer: DevicePtr) -> *mut c_void {
    pointer.0 as usize as *mut c_void
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
    ensure!(
        matches.len() == 1,
        "expected one embedded SM121 target, found {}",
        matches.len()
    );
    Ok(matches.pop().context("target disappeared")?.modules)
}

fn require_finite(label: &str, bytes: &[u8]) -> Result<()> {
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        let value = bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32();
        ensure!(value.is_finite(), "{label}: nonfinite BF16 at {index}");
    }
    Ok(())
}

fn require_equal(label: &str, reference: &[u8], candidate: &[u8]) -> Result<()> {
    if reference == candidate {
        return Ok(());
    }
    let byte = reference
        .iter()
        .zip(candidate)
        .position(|(a, b)| a != b)
        .context("mismatch without differing byte")?;
    bail!(
        "{label}: BF16 mismatch at byte {byte}/element {} reference=0x{:02x} candidate=0x{:02x}",
        byte / 2,
        reference[byte],
        candidate[byte]
    )
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<()> {
    let checkpoint_dir = PathBuf::from(required_env("ATLAS_FI_CHECKPOINT_DIR")?);
    let layer = parse_usize("ATLAS_FI_LAYER")?;
    let projection = Projection::parse()?;
    let checkpoint = Checkpoint::open(&checkpoint_dir)?;
    let data = load_projection(&checkpoint, layer, projection)?;
    let (n, k) = projection.shape();
    let physical_weight_scales = interleave_nvfp4_scales_128x4(
        &data.scales.bytes,
        &[n, k / NVFP4_GROUP_SIZE],
        NVFP4_GROUP_SIZE,
    )?;
    ensure!(
        deinterleave_nvfp4_scales_128x4(
            &physical_weight_scales,
            &[n, k / NVFP4_GROUP_SIZE],
            NVFP4_GROUP_SIZE,
        )? == data.scales.bytes,
        "weight scale 128x4 roundtrip failed"
    );
    println!(
        "CHECKPOINT prefix={} convention={:?} shard={} packed_shape={:?} scale_shape={:?} packed_offsets={:?} scale_offsets={:?} packed_hash=fnv1a64:{:016x} logical_scale_hash=fnv1a64:{:016x} physical_scale_hash=fnv1a64:{:016x} input_scale2={:.9e} weight_scale2={:.9e} scale_roundtrip=PASS",
        data.prefix,
        data.convention,
        data.packed.shard.display(),
        data.packed.meta.shape,
        data.scales.meta.shape,
        data.packed.meta.offsets,
        data.scales.meta.offsets,
        fnv1a64(&data.packed.bytes),
        fnv1a64(&data.scales.bytes),
        fnv1a64(&physical_weight_scales),
        data.input_scale2,
        data.weight_scale2,
    );
    if strict_switch("ATLAS_FI_METADATA_ONLY")? {
        println!("FINAL verdict=METADATA_PASS gpu_initialized=false parity=UNPROVEN timing=false");
        return Ok(());
    }

    let timing = strict_switch("ATLAS_FI_CHECKPOINT_TIMING")?;
    let m = parse_usize("ATLAS_FI_M")?;
    ensure!(
        matches!(m, 2_048 | 2_079 | 8_192),
        "ATLAS_FI_M must be 2048, 2079, or 8192"
    );
    let tactic = parse_usize("ATLAS_FI_TACTIC")?;
    ensure!(tactic < 6, "ATLAS_FI_TACTIC must be in 0..6");
    let library_path = PathBuf::from(required_env("ATLAS_FI_CABI_LIB")?);
    let modules = exact_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let cabi = Cabi::open(&library_path)?;
    let parent: KernelHandle = gpu.kernel("nvfp4_cutlass", "nvfp4_nvfp4_gemm_kmajor_m256")?;

    let m_pad = m.div_ceil(128) * 128;
    let mut rng = Rng(0xc0de_0000 ^ layer as u64 ^ m as u64 ^ n as u64 ^ k as u64);
    let a_packed: Vec<_> = (0..m * k / 2).map(|_| rng.packed()).collect();
    let a_scales_padded: Vec<_> = (0..m_pad * k / 16).map(|_| rng.scale()).collect();
    let a_scales_logical = a_scales_padded[..m * k / 16].to_vec();
    let a_scales_physical = interleave_nvfp4_scales_128x4(
        &a_scales_padded,
        &[m_pad, k / NVFP4_GROUP_SIZE],
        NVFP4_GROUP_SIZE,
    )?;
    let b_packed_t = transpose(&data.packed.bytes, n, k / 2);
    let b_scales_t = transpose(&data.scales.bytes, n, k / 16);
    let alpha = data.input_scale2 * data.weight_scale2;
    ensure!(alpha.is_finite() && alpha > 0.0, "combined scale invalid");

    let a = Guarded::input(gpu, stream, a_packed, 0x11)?;
    let a_sf_logical = Guarded::input(gpu, stream, a_scales_logical, 0x12)?;
    let a_sf_physical = Guarded::input(gpu, stream, a_scales_physical, 0x13)?;
    let b = Guarded::input(gpu, stream, data.packed.bytes, 0x21)?;
    let b_sf_physical = Guarded::input(gpu, stream, physical_weight_scales, 0x22)?;
    let b_t = Guarded::input(gpu, stream, b_packed_t, 0x23)?;
    let b_sf_t = Guarded::input(gpu, stream, b_scales_t, 0x24)?;
    let global = Guarded::input(gpu, stream, alpha.to_le_bytes().to_vec(), 0x25)?;
    let output_bytes = m * n * 2;
    let atlas_output = Guarded::output(gpu, stream, output_bytes, 0x31)?;
    let cabi_output = Guarded::output(gpu, stream, output_bytes, 0x32)?;

    let mut workspace_bytes = 0usize;
    let status = unsafe {
        (cabi.workspace)(
            tactic as c_int,
            m as c_int,
            n as c_int,
            k as c_int,
            1,
            &mut workspace_bytes,
        )
    };
    ensure!(
        status == STATUS_OK,
        "workspace query failed: {status} {}",
        cabi.error()
    );
    let workspace = if workspace_bytes == 0 {
        None
    } else {
        Some(Guarded::create(
            gpu,
            stream,
            vec![0u8; workspace_bytes],
            0x41,
            false,
        )?)
    };
    let invoke_cabi = || -> Result<()> {
        let status = unsafe {
            (cabi.gemm)(
                tactic as c_int,
                device_ptr(cabi_output.ptr()),
                device_ptr(a.ptr()),
                device_ptr(b.ptr()),
                device_ptr(a_sf_physical.ptr()),
                device_ptr(b_sf_physical.ptr()),
                device_ptr(global.ptr()).cast(),
                m as c_int,
                n as c_int,
                k as c_int,
                1,
                workspace
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |w| device_ptr(w.ptr())),
                workspace_bytes,
                stream as usize as *mut c_void,
            )
        };
        ensure!(
            status == STATUS_OK,
            "C ABI GEMM failed: {status} {}",
            cabi.error()
        );
        Ok(())
    };

    if workspace_bytes > 0 {
        let short = Guarded::create(gpu, stream, vec![0u8; workspace_bytes - 1], 0x42, false)?;
        let status = unsafe {
            (cabi.gemm)(
                tactic as c_int,
                device_ptr(cabi_output.ptr()),
                device_ptr(a.ptr()),
                device_ptr(b.ptr()),
                device_ptr(a_sf_physical.ptr()),
                device_ptr(b_sf_physical.ptr()),
                device_ptr(global.ptr()).cast(),
                m as c_int,
                n as c_int,
                k as c_int,
                1,
                device_ptr(short.ptr()),
                workspace_bytes - 1,
                stream as usize as *mut c_void,
            )
        };
        ensure!(
            status == STATUS_CUTLASS_ERROR,
            "short workspace was not rejected: {status} {}",
            cabi.error()
        );
        let untouched = cabi_output.payload(gpu, "short-workspace-output")?;
        ensure!(
            untouched
                .chunks_exact(2)
                .all(|p| p == 0x7f81u16.to_le_bytes()),
            "short workspace modified output"
        );
        short.free(gpu)?;
    }

    ops::nvfp4_nvfp4_gemm_kmajor_m256(
        gpu,
        parent,
        a.ptr(),
        a_sf_logical.ptr(),
        b_t.ptr(),
        b_sf_t.ptr(),
        alpha,
        atlas_output.ptr(),
        m as u32,
        n as u32,
        k as u32,
        stream,
    )?;
    invoke_cabi()?;
    gpu.synchronize(stream)?;
    let atlas = atlas_output.payload(gpu, "atlas-output")?;
    let candidate = cabi_output.payload(gpu, "cabi-output")?;
    require_finite("atlas-output", &atlas)?;
    require_finite("cabi-output", &candidate)?;
    require_equal("Atlas-vs-CABI", &atlas, &candidate)?;
    for (buffer, label) in [
        (&a, "A"),
        (&a_sf_logical, "A-sf-logical"),
        (&a_sf_physical, "A-sf-physical"),
        (&b, "B-checkpoint"),
        (&b_sf_physical, "B-sf-physical"),
        (&b_t, "B-atlas-transposed"),
        (&b_sf_t, "B-sf-atlas-transposed"),
        (&global, "global-scale"),
    ] {
        buffer.check_immutable(gpu, label)?;
    }
    println!(
        "PARITY layer={layer} projection={} M={m} N={n} K={k} tactic={tactic} workspace_bytes={workspace_bytes} output_hash=fnv1a64:{:016x} exact_bf16=PASS finite=PASS redzones=PASS immutable=PASS",
        projection.name(),
        fnv1a64(&candidate),
    );

    if timing {
        let measure = |candidate_arm: bool| -> Result<f64> {
            gpu.synchronize(stream)?;
            let start = Instant::now();
            if candidate_arm {
                invoke_cabi()?;
            } else {
                ops::nvfp4_nvfp4_gemm_kmajor_m256(
                    gpu,
                    parent,
                    a.ptr(),
                    a_sf_logical.ptr(),
                    b_t.ptr(),
                    b_sf_t.ptr(),
                    alpha,
                    atlas_output.ptr(),
                    m as u32,
                    n as u32,
                    k as u32,
                    stream,
                )?;
            }
            gpu.synchronize(stream)?;
            Ok(start.elapsed().as_secs_f64() * 1_000.0)
        };
        let _ = measure(false)?;
        let _ = measure(true)?;
        let (mut parent_ms, mut candidate_ms, mut deltas) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..TIMING_ROUNDS {
            for order in [[false, true], [true, false]] {
                let (mut p, mut c) = (None, None);
                for arm in order {
                    let value = measure(arm)?;
                    if arm {
                        c = Some(value)
                    } else {
                        p = Some(value)
                    }
                }
                let p = p.context("ABBA parent omitted")?;
                let c = c.context("ABBA candidate omitted")?;
                parent_ms.push(p);
                candidate_ms.push(c);
                deltas.push(p - c);
            }
        }
        println!(
            "TIMING parent_median_ms={:.4} candidate_median_ms={:.4} paired_delta_median_ms={:.4} qualification=EVIDENCE_ONLY",
            median(&parent_ms),
            median(&candidate_ms),
            median(&deltas),
        );
        let timed_atlas = atlas_output.payload(gpu, "timed-atlas")?;
        let timed_cabi = cabi_output.payload(gpu, "timed-cabi")?;
        require_equal("post-timing Atlas-vs-CABI", &timed_atlas, &timed_cabi)?;
    }

    for buffer in [
        &a,
        &a_sf_logical,
        &a_sf_physical,
        &b,
        &b_sf_physical,
        &b_t,
        &b_sf_t,
        &global,
        &atlas_output,
        &cabi_output,
    ] {
        buffer.free(gpu)?;
    }
    if let Some(workspace) = workspace.as_ref() {
        let _ = workspace.payload(gpu, "workspace")?;
        workspace.free(gpu)?;
    }
    println!("FINAL verdict=PASS parity=EXACT timing={timing} production_route=false");
    Ok(())
}
