// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed raw gate for the large-M attention K/V dual-projection route.
//!
//! The production parent launches `w4a16_gemm` once for K and once for V.
//! The candidate launches `w4a16_gemm_pipe_dual(..., fuse_silu=false)` once
//! with the same A and independent original-layout NVFP4 K/V weights. Every
//! BF16 output byte must match before timing is permitted. Inputs, weights,
//! scales, and 4-KiB redzones are checked for mutation; parent and candidate
//! outputs start with different payload sentinels so a common missed write
//! cannot compare equal.
//!
//! GPU qualification command (run only on a reserved device):
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --features cuda \
//!   --example w4a16_attention_kv_dual_microgate
//! ```
//!
//! Set `ATLAS_PREFILL_KV_DUAL_MICROGATE_TIMING=1` to run parity-gated ABBA
//! timing at M=2048 and M=8192 and fail unless the dual launch wins both.

#[allow(dead_code)]
#[path = "w4a16_exact_lm_head_microtest/data.rs"]
mod data;

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use data::{Fixture, as_le_bytes, cancellation_fixture, random_fixture};
use spark_model::layers::ops;
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const HIDDEN: usize = 5_120;
const KV_DIM: usize = 1_024;
const REDZONE: usize = 4 * 1_024;
const OUTER_CANARY: u8 = 0xc7;
const PARENT_FILL: u8 = 0xa5;
const DUAL_FILL: u8 = 0x5a;
const TIMING_ROUNDS: usize = 3;

#[derive(Clone, Copy)]
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

    fn output(gpu: &dyn GpuBackend, payload_len: usize, fill: u8) -> Result<Self> {
        Self::input(gpu, &vec![fill; payload_len], OUTER_CANARY, OUTER_CANARY)
    }

    fn payload_ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.image, self.allocation)
    }

    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; self.image.len()];
        gpu.copy_d2h(self.allocation, &mut bytes)?;
        Ok(bytes)
    }

    fn verify_immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        ensure!(
            self.read(gpu)? == self.image,
            "{label}: input bytes changed"
        );
        Ok(())
    }

    fn output_payload(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        let bytes = self.read(gpu)?;
        ensure!(
            bytes[..REDZONE] == self.image[..REDZONE],
            "{label}: leading 4-KiB redzone changed"
        );
        let suffix = REDZONE + self.payload_len;
        ensure!(
            bytes[suffix..] == self.image[suffix..],
            "{label}: trailing 4-KiB redzone changed"
        );
        Ok(bytes[REDZONE..suffix].to_vec())
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
        ensure!(fixture.logical_n == KV_DIM, "weight N mismatch");
        ensure!(fixture.k == HIDDEN, "weight K mismatch");
        let packed = Guarded::input(
            gpu,
            &fixture.packed,
            OUTER_CANARY ^ salt,
            OUTER_CANARY.wrapping_add(salt),
        )?;
        let scales = Guarded::input(
            gpu,
            &fixture.scales,
            OUTER_CANARY.wrapping_add(salt),
            OUTER_CANARY ^ salt,
        )?;
        let quant = QuantizedWeight {
            weight: packed.payload_ptr(),
            weight_scale: scales.payload_ptr(),
            weight_scale_2: scale2,
            input_scale: DevicePtr::NULL,
        };
        Ok(Self {
            packed,
            scales,
            quant,
        })
    }

    fn verify_immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        self.packed
            .verify_immutable(gpu, &format!("{label}/packed"))?;
        self.scales
            .verify_immutable(gpu, &format!("{label}/scales"))
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.packed.free(gpu)?;
        self.scales.free(gpu)
    }
}

struct Kernels {
    parent: KernelHandle,
    dual: KernelHandle,
}

struct Outputs {
    k: Guarded,
    v: Guarded,
}

impl Outputs {
    fn new(gpu: &dyn GpuBackend, payload_len: usize, fill: u8) -> Result<Self> {
        Ok(Self {
            k: Guarded::output(gpu, payload_len, fill)?,
            v: Guarded::output(gpu, payload_len, fill)?,
        })
    }

    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.k.reset(gpu)?;
        self.v.reset(gpu)
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.k.free(gpu)?;
        self.v.free(gpu)
    }
}

fn input_bytes(rows: usize, kind: InputKind) -> Vec<u8> {
    let fixture = match kind {
        InputKind::Random => random_fixture(rows, 4, HIDDEN, 0x51c0_0000_0000_0000 ^ rows as u64),
        InputKind::Cancellation => cancellation_fixture(rows, 4, HIDDEN),
    };
    as_le_bytes(&fixture.activations)
}

#[allow(clippy::too_many_arguments)]
fn launch_parent(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernel: KernelHandle,
    input: DevicePtr,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    outputs: &Outputs,
    rows: usize,
) -> Result<()> {
    ops::w4a16_gemm(
        gpu,
        kernel,
        input,
        k_weight,
        outputs.k.payload_ptr(),
        rows as u32,
        KV_DIM as u32,
        HIDDEN as u32,
        stream,
    )?;
    ops::w4a16_gemm(
        gpu,
        kernel,
        input,
        v_weight,
        outputs.v.payload_ptr(),
        rows as u32,
        KV_DIM as u32,
        HIDDEN as u32,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_dual(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernel: KernelHandle,
    input: DevicePtr,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    outputs: &Outputs,
    rows: usize,
) -> Result<()> {
    ops::w4a16_gemm_pipe_dual(
        gpu,
        kernel,
        input,
        k_weight,
        v_weight,
        outputs.k.payload_ptr(),
        outputs.v.payload_ptr(),
        false,
        rows as u32,
        KV_DIM as u32,
        HIDDEN as u32,
        stream,
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn require_equal(label: &str, parent: &[u8], dual: &[u8]) -> Result<()> {
    if parent == dual {
        return Ok(());
    }
    let first = parent
        .iter()
        .zip(dual)
        .position(|(lhs, rhs)| lhs != rhs)
        .context("mismatch search returned no index")?;
    let word = first & !1;
    let parent_word = u16::from_le_bytes([parent[word], parent[word + 1]]);
    let dual_word = u16::from_le_bytes([dual[word], dual[word + 1]]);
    bail!(
        "{label}: first mismatch byte={first} element={} parent=0x{parent_word:04x} dual=0x{dual_word:04x} parent_hash={:016x} dual_hash={:016x}",
        first / 2,
        fnv1a64(parent),
        fnv1a64(dual)
    )
}

#[allow(clippy::too_many_arguments)]
fn measure_parent(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    input: DevicePtr,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    outputs: &Outputs,
    rows: usize,
) -> Result<f64> {
    outputs.reset(gpu)?;
    let start = Instant::now();
    launch_parent(
        gpu,
        stream,
        kernels.parent,
        input,
        k_weight,
        v_weight,
        outputs,
        rows,
    )?;
    gpu.synchronize(stream)?;
    Ok(start.elapsed().as_secs_f64() * 1_000.0)
}

#[allow(clippy::too_many_arguments)]
fn measure_dual(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    input: DevicePtr,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    outputs: &Outputs,
    rows: usize,
) -> Result<f64> {
    outputs.reset(gpu)?;
    let start = Instant::now();
    launch_dual(
        gpu,
        stream,
        kernels.dual,
        input,
        k_weight,
        v_weight,
        outputs,
        rows,
    )?;
    gpu.synchronize(stream)?;
    Ok(start.elapsed().as_secs_f64() * 1_000.0)
}

#[allow(clippy::too_many_arguments)]
fn time_abba(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    parent_input: DevicePtr,
    dual_input: DevicePtr,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    parent: &Outputs,
    dual: &Outputs,
    rows: usize,
) -> Result<()> {
    let _ = measure_parent(
        gpu,
        stream,
        kernels,
        parent_input,
        k_weight,
        v_weight,
        parent,
        rows,
    )?;
    let _ = measure_dual(
        gpu, stream, kernels, dual_input, k_weight, v_weight, dual, rows,
    )?;
    let mut parent_ms = Vec::with_capacity(TIMING_ROUNDS * 2);
    let mut dual_ms = Vec::with_capacity(TIMING_ROUNDS * 2);
    for _ in 0..TIMING_ROUNDS {
        parent_ms.push(measure_parent(
            gpu,
            stream,
            kernels,
            parent_input,
            k_weight,
            v_weight,
            parent,
            rows,
        )?);
        dual_ms.push(measure_dual(
            gpu, stream, kernels, dual_input, k_weight, v_weight, dual, rows,
        )?);
        dual_ms.push(measure_dual(
            gpu, stream, kernels, dual_input, k_weight, v_weight, dual, rows,
        )?);
        parent_ms.push(measure_parent(
            gpu,
            stream,
            kernels,
            parent_input,
            k_weight,
            v_weight,
            parent,
            rows,
        )?);
    }
    parent_ms.sort_by(f64::total_cmp);
    dual_ms.sort_by(f64::total_cmp);
    let parent_median = parent_ms[parent_ms.len() / 2];
    let dual_median = dual_ms[dual_ms.len() / 2];
    println!(
        "TIMING M={rows}: parent-K+V={parent_median:.3} ms dual-KV={dual_median:.3} ms speedup={:.3}x",
        parent_median / dual_median
    );
    ensure!(
        dual_median < parent_median,
        "M={rows}: dual K/V did not beat two-launch parent; stop promotion"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    k_weight: &UploadedWeight,
    v_weight: &UploadedWeight,
    rows: usize,
    kind: InputKind,
    timing: bool,
) -> Result<()> {
    let label = format!("M={rows}/{}", kind.label());
    let bytes = input_bytes(rows, kind);
    // Different suffix bytes expose any masked-tail read dependence between
    // parent and candidate while leaving every valid A byte identical.
    let parent_input = Guarded::input(gpu, &bytes, 0x31, 0x13)?;
    let dual_input = Guarded::input(gpu, &bytes, 0x31, 0x9b)?;
    let output_len = rows * KV_DIM * size_of::<u16>();
    let parent = Outputs::new(gpu, output_len, PARENT_FILL)?;
    let dual = Outputs::new(gpu, output_len, DUAL_FILL)?;

    launch_parent(
        gpu,
        stream,
        kernels.parent,
        parent_input.payload_ptr(),
        &k_weight.quant,
        &v_weight.quant,
        &parent,
        rows,
    )?;
    launch_dual(
        gpu,
        stream,
        kernels.dual,
        dual_input.payload_ptr(),
        &k_weight.quant,
        &v_weight.quant,
        &dual,
        rows,
    )?;
    gpu.synchronize(stream)?;

    let parent_k = parent.k.output_payload(gpu, &format!("{label}/parent-K"))?;
    let parent_v = parent.v.output_payload(gpu, &format!("{label}/parent-V"))?;
    let dual_k = dual.k.output_payload(gpu, &format!("{label}/dual-K"))?;
    let dual_v = dual.v.output_payload(gpu, &format!("{label}/dual-V"))?;
    require_equal(&format!("{label}/K"), &parent_k, &dual_k)?;
    require_equal(&format!("{label}/V"), &parent_v, &dual_v)?;
    parent_input.verify_immutable(gpu, &format!("{label}/parent-A"))?;
    dual_input.verify_immutable(gpu, &format!("{label}/dual-A"))?;
    k_weight.verify_immutable(gpu, &format!("{label}/K-weight"))?;
    v_weight.verify_immutable(gpu, &format!("{label}/V-weight"))?;
    println!(
        "PASS {label}: {} K bytes + {} V bytes",
        parent_k.len(),
        parent_v.len()
    );

    if timing && kind.label() == InputKind::Random.label() && matches!(rows, 2_048 | 8_192) {
        time_abba(
            gpu,
            stream,
            kernels,
            parent_input.payload_ptr(),
            dual_input.payload_ptr(),
            &k_weight.quant,
            &v_weight.quant,
            &parent,
            &dual,
            rows,
        )?;
    }

    parent_input.free(gpu)?;
    dual_input.free(gpu)?;
    parent.free(gpu)?;
    dual.free(gpu)
}

fn timing_requested() -> Result<bool> {
    match std::env::var("ATLAS_PREFILL_KV_DUAL_MICROGATE_TIMING") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => bail!("ATLAS_PREFILL_KV_DUAL_MICROGATE_TIMING must be exactly 0 or 1"),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("ATLAS_PREFILL_KV_DUAL_MICROGATE_TIMING must be valid UTF-8")
        }
    }
}

fn main() -> Result<()> {
    let timing = timing_requested()?;
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend with compiled Qwen3.8 kernels")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernels = Kernels {
        parent: gpu.kernel("w4a16", "w4a16_gemm")?,
        dual: gpu.kernel("w4a16", "w4a16_gemm_pipe_dual")?,
    };

    let k_fixture = random_fixture(1, KV_DIM, HIDDEN, 0x51c0_0000_0400_0012);
    let v_fixture = random_fixture(1, KV_DIM, HIDDEN, 0x51c0_0000_0400_0013);
    let k_weight = UploadedWeight::new(gpu, &k_fixture, 0.75, 0x11)?;
    let v_weight = UploadedWeight::new(gpu, &v_fixture, 1.25, 0x22)?;

    println!(
        "attention K/V dual raw gate: N={KV_DIM} K={HIDDEN}, timing={} (parity always mandatory)",
        u8::from(timing)
    );
    for rows in [33, 63, 64, 65, 127, 128, 129, 2_048, 8_192] {
        run_case(
            gpu,
            stream,
            &kernels,
            &k_weight,
            &v_weight,
            rows,
            InputKind::Random,
            timing,
        )?;
    }
    for rows in [65, 129, 2_048] {
        run_case(
            gpu,
            stream,
            &kernels,
            &k_weight,
            &v_weight,
            rows,
            InputKind::Cancellation,
            false,
        )?;
    }

    k_weight.verify_immutable(gpu, "final/K-weight")?;
    v_weight.verify_immutable(gpu, "final/V-weight")?;
    k_weight.free(gpu)?;
    v_weight.free(gpu)?;
    println!("ALL PASS: dual K/V is byte-identical to two production parent launches");
    Ok(())
}
