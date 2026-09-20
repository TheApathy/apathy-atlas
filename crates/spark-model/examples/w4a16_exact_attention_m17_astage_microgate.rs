// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed raw gate for the M17 exact-attention activation-staging pair.
//!
//! The staged QG and dual-KV symbols must match both the proven multi-row
//! exact kernels and the serial K1 production oracle for every M=5..17. The
//! gate also checks production and padded row strides, partial N blocks,
//! partial 512-column activation waves, external redzones, interior holes,
//! unwritten M17 rows, and immutable input/weight buffers.
//!
//! GPU qualification command (do not run on an unreserved device):
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --features cuda \
//!   --example w4a16_exact_attention_m17_astage_microgate
//! ```

#[allow(dead_code)]
#[path = "w4a16_exact_lm_head_microtest/data.rs"]
mod data;

use anyhow::{Context, Result, bail, ensure};
use data::{Fixture, as_le_bytes, cancellation_fixture, fnv1a64, from_le_bytes, random_fixture};
use spark_model::layers::ops::{
    self, W4a16ExactAttentionKernels, W4a16ExactAttentionM17AStageKernels,
};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const MAX_M: usize = 17;
const PROD_K: usize = 5_120;
const PROD_NQ: usize = 24;
const PROD_HD: usize = 256;
const PROD_Q_N: usize = 2 * PROD_NQ * PROD_HD;
const PROD_KV_N: usize = 1_024;
const PROD_STRIDE: usize = 14_336;
const TAIL_NQ: usize = 3;
const TAIL_HD: usize = 5;
const TAIL_Q_N: usize = 2 * TAIL_NQ * TAIL_HD;
const REDZONE_BYTES: usize = 4 * 1_024;
const REDZONE_WORDS: usize = REDZONE_BYTES / size_of::<u16>();
const CANARY: u16 = 0xa55a;

struct Kernels {
    gemv: KernelHandle,
    gemv_qg: KernelHandle,
    baseline: W4a16ExactAttentionKernels,
    staged: W4a16ExactAttentionM17AStageKernels,
}

struct UploadedWeight {
    quant: QuantizedWeight,
    packed: Vec<u8>,
    scales: Vec<u8>,
}

struct TestSet {
    label: String,
    k: usize,
    nq: usize,
    hd: usize,
    q_n: usize,
    kv_n: usize,
    input: DevicePtr,
    input_bytes: Vec<u8>,
    q: UploadedWeight,
    k_weight: UploadedWeight,
    v: UploadedWeight,
}

struct GuardedOutput {
    allocation: DevicePtr,
    base_words: usize,
    stride: usize,
    payload_words: usize,
}

impl GuardedOutput {
    fn new(gpu: &dyn GpuBackend, base_words: usize, stride: usize) -> Result<Self> {
        let payload_words = base_words + MAX_M * stride + 16;
        let words = vec![CANARY; REDZONE_WORDS * 2 + payload_words];
        Ok(Self {
            allocation: upload(gpu, &as_le_bytes(&words))?,
            base_words,
            stride,
            payload_words,
        })
    }

    fn output_ptr(&self) -> DevicePtr {
        self.allocation
            .offset((REDZONE_WORDS + self.base_words) * size_of::<u16>())
    }

    fn read_active(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        label: &str,
        rows: usize,
        width: usize,
    ) -> Result<Vec<u16>> {
        ensure!(rows <= MAX_M, "{label}: rows exceed M17");
        ensure!(width <= self.stride, "{label}: width exceeds stride");
        let total_words = REDZONE_WORDS * 2 + self.payload_words;
        let mut bytes = vec![0u8; total_words * size_of::<u16>()];
        gpu.copy_d2h_on_stream(self.allocation, &mut bytes, stream)?;
        let words = from_le_bytes(&bytes);
        ensure!(
            words[..REDZONE_WORDS].iter().all(|&word| word == CANARY),
            "{label}: 4 KiB leading redzone modified"
        );
        let suffix = REDZONE_WORDS + self.payload_words;
        ensure!(
            words[suffix..].iter().all(|&word| word == CANARY),
            "{label}: 4 KiB trailing redzone modified"
        );

        let payload = &words[REDZONE_WORDS..suffix];
        let mut active = Vec::with_capacity(rows * width);
        for index in 0..self.payload_words {
            let relative = index.checked_sub(self.base_words);
            let written = relative.is_some_and(|relative| {
                let row = relative / self.stride;
                let col = relative % self.stride;
                row < rows && col < width
            });
            if written {
                active.push(payload[index]);
            } else if payload[index] != CANARY {
                bail!(
                    "{label}: output hole/unwritten row changed at payload word {index}: 0x{:04x}",
                    payload[index]
                );
            }
        }
        ensure!(
            active.len() == rows * width,
            "{label}: gathered {} words, expected {}",
            active.len(),
            rows * width
        );
        Ok(active)
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}

struct OutputTriplet {
    q: GuardedOutput,
    k: GuardedOutput,
    v: GuardedOutput,
}

impl OutputTriplet {
    fn new(
        gpu: &dyn GpuBackend,
        q_stride: usize,
        kv_stride: usize,
        k_base: usize,
        v_base: usize,
    ) -> Result<Self> {
        Ok(Self {
            q: GuardedOutput::new(gpu, 0, q_stride)?,
            k: GuardedOutput::new(gpu, k_base, kv_stride)?,
            v: GuardedOutput::new(gpu, v_base, kv_stride)?,
        })
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.q.free(gpu)?;
        self.k.free(gpu)?;
        self.v.free(gpu)
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn upload_weight(gpu: &dyn GpuBackend, fixture: &Fixture) -> Result<UploadedWeight> {
    let packed = fixture.packed.clone();
    let scales = fixture.scales.clone();
    Ok(UploadedWeight {
        quant: QuantizedWeight {
            weight: upload(gpu, &packed)?,
            weight_scale: upload(gpu, &scales)?,
            weight_scale_2: 1.0,
            input_scale: DevicePtr::NULL,
        },
        packed,
        scales,
    })
}

impl TestSet {
    fn new(
        gpu: &dyn GpuBackend,
        label: impl Into<String>,
        nq: usize,
        hd: usize,
        kv_n: usize,
        q_fixture: Fixture,
        k_fixture: Fixture,
        v_fixture: Fixture,
    ) -> Result<Self> {
        let q_n = 2 * nq * hd;
        ensure!(q_fixture.rows == MAX_M, "Q fixture must contain M17 inputs");
        ensure!(q_fixture.logical_n == q_n, "Q fixture width mismatch");
        ensure!(k_fixture.logical_n == kv_n, "K fixture width mismatch");
        ensure!(v_fixture.logical_n == kv_n, "V fixture width mismatch");
        ensure!(
            q_fixture.k == k_fixture.k && q_fixture.k == v_fixture.k,
            "fixture K mismatch"
        );
        let input_bytes = as_le_bytes(&q_fixture.activations);
        Ok(Self {
            label: label.into(),
            k: q_fixture.k,
            nq,
            hd,
            q_n,
            kv_n,
            input: upload(gpu, &input_bytes)?,
            input_bytes,
            q: upload_weight(gpu, &q_fixture)?,
            k_weight: upload_weight(gpu, &k_fixture)?,
            v: upload_weight(gpu, &v_fixture)?,
        })
    }

    fn assert_immutable(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        assert_device_bytes(gpu, stream, "input", self.input, &self.input_bytes)?;
        for (name, weight) in [("q", &self.q), ("k", &self.k_weight), ("v", &self.v)] {
            assert_device_bytes(
                gpu,
                stream,
                &format!("{name}.packed"),
                weight.quant.weight,
                &weight.packed,
            )?;
            assert_device_bytes(
                gpu,
                stream,
                &format!("{name}.scales"),
                weight.quant.weight_scale,
                &weight.scales,
            )?;
        }
        Ok(())
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for ptr in [
            self.input,
            self.q.quant.weight,
            self.q.quant.weight_scale,
            self.k_weight.quant.weight,
            self.k_weight.quant.weight_scale,
            self.v.quant.weight,
            self.v.quant.weight_scale,
        ] {
            gpu.free(ptr)?;
        }
        Ok(())
    }
}

fn assert_device_bytes(
    gpu: &dyn GpuBackend,
    stream: u64,
    label: &str,
    ptr: DevicePtr,
    expected: &[u8],
) -> Result<()> {
    let mut actual = vec![0u8; expected.len()];
    gpu.copy_d2h_on_stream(ptr, &mut actual, stream)?;
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        bail!(
            "immutable {label} changed at byte {index}: actual=0x{:02x}, expected=0x{:02x}",
            actual[index],
            expected[index]
        );
    }
    Ok(())
}

fn load_kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    Ok(Kernels {
        gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
        gemv_qg: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg")?,
        baseline: W4a16ExactAttentionKernels::new(
            gpu.kernel("w4a16_gemv_exact_attention", "w4a16_gemv_qg_exact_m17")?,
            gpu.kernel("w4a16_gemv_exact_attention", "w4a16_gemv_dual_kv_exact_m17")?,
        ),
        staged: W4a16ExactAttentionM17AStageKernels::new(
            gpu.kernel(
                "w4a16_gemv_exact_attention",
                "w4a16_gemv_qg_exact_m17_astage",
            )?,
            gpu.kernel(
                "w4a16_gemv_exact_attention",
                "w4a16_gemv_dual_kv_exact_m17_astage",
            )?,
        ),
    })
}

fn run_serial(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    set: &TestSet,
    rows: usize,
    outputs: &OutputTriplet,
) -> Result<()> {
    for row in 0..rows {
        let input = set.input.offset(row * set.k * size_of::<u16>());
        ops::w4a16_gemv_qg(
            gpu,
            kernels.gemv_qg,
            input,
            &set.q.quant,
            outputs.q.output_ptr().offset(row * outputs.q.stride * 2),
            set.q_n as u32,
            set.k as u32,
            set.nq as u32,
            set.hd as u32,
            stream,
        )?;
        for (weight, output, label) in [
            (&set.k_weight.quant, &outputs.k, "K"),
            (&set.v.quant, &outputs.v, "V"),
        ] {
            ops::w4a16_gemv(
                gpu,
                kernels.gemv,
                input,
                weight,
                output.output_ptr().offset(row * output.stride * 2),
                set.kv_n as u32,
                set.k as u32,
                stream,
            )
            .with_context(|| format!("serial {label} row={row}"))?;
        }
    }
    Ok(())
}

fn run_exact(
    gpu: &dyn GpuBackend,
    stream: u64,
    qg: KernelHandle,
    dual_kv: KernelHandle,
    set: &TestSet,
    rows: usize,
    outputs: &OutputTriplet,
) -> Result<()> {
    ops::w4a16_gemv_qg_exact(
        gpu,
        qg,
        set.input,
        &set.q.quant,
        outputs.q.output_ptr(),
        rows as u32,
        set.q_n as u32,
        set.k as u32,
        set.nq as u32,
        set.hd as u32,
        outputs.q.stride as u32,
        stream,
    )?;
    ops::w4a16_gemv_dual_kv_exact(
        gpu,
        dual_kv,
        set.input,
        &set.k_weight.quant,
        outputs.k.output_ptr(),
        &set.v.quant,
        outputs.v.output_ptr(),
        rows as u32,
        set.kv_n as u32,
        set.k as u32,
        outputs.k.stride as u32,
        stream,
    )
}

fn assert_exact(label: &str, actual: &[u16], expected: &[u16], width: usize) -> Result<()> {
    ensure!(actual.len() == expected.len(), "{label}: length mismatch");
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        bail!(
            "{label}: raw BF16 mismatch at flat={index}, row={}, col={}, actual=0x{:04x}, expected=0x{:04x}",
            index / width,
            index % width,
            actual[index],
            expected[index]
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    set: &TestSet,
    rows: usize,
    q_stride: usize,
    kv_stride: usize,
    k_base: usize,
    v_base: usize,
) -> Result<()> {
    let label = format!(
        "{} M={rows} K={} qN={} kvN={} qstride={q_stride} kvstride={kv_stride} kbase={k_base} vbase={v_base}",
        set.label, set.k, set.q_n, set.kv_n
    );
    let oracle = OutputTriplet::new(gpu, q_stride, kv_stride, k_base, v_base)?;
    let baseline = OutputTriplet::new(gpu, q_stride, kv_stride, k_base, v_base)?;
    let staged = OutputTriplet::new(gpu, q_stride, kv_stride, k_base, v_base)?;

    run_serial(gpu, stream, kernels, set, rows, &oracle)
        .with_context(|| format!("{label}: serial oracle"))?;
    run_exact(
        gpu,
        stream,
        kernels.baseline.qg_for_rows(rows),
        kernels.baseline.dual_kv_for_rows(rows),
        set,
        rows,
        &baseline,
    )
    .with_context(|| format!("{label}: baseline exact"))?;
    run_exact(
        gpu,
        stream,
        kernels.staged.qg(),
        kernels.staged.dual_kv(),
        set,
        rows,
        &staged,
    )
    .with_context(|| format!("{label}: staged exact"))?;
    gpu.synchronize(stream)?;

    let oq = oracle
        .q
        .read_active(gpu, stream, &format!("{label} oracle QG"), rows, set.q_n)?;
    let ok = oracle
        .k
        .read_active(gpu, stream, &format!("{label} oracle K"), rows, set.kv_n)?;
    let ov = oracle
        .v
        .read_active(gpu, stream, &format!("{label} oracle V"), rows, set.kv_n)?;
    let bq = baseline
        .q
        .read_active(gpu, stream, &format!("{label} baseline QG"), rows, set.q_n)?;
    let bk = baseline
        .k
        .read_active(gpu, stream, &format!("{label} baseline K"), rows, set.kv_n)?;
    let bv = baseline
        .v
        .read_active(gpu, stream, &format!("{label} baseline V"), rows, set.kv_n)?;
    let sq = staged
        .q
        .read_active(gpu, stream, &format!("{label} staged QG"), rows, set.q_n)?;
    let sk = staged
        .k
        .read_active(gpu, stream, &format!("{label} staged K"), rows, set.kv_n)?;
    let sv = staged
        .v
        .read_active(gpu, stream, &format!("{label} staged V"), rows, set.kv_n)?;

    let q_dim = set.q_n / 2;
    for row in 0..rows {
        let base = row * set.q_n;
        for (name, actual) in [("baseline", &bq), ("staged", &sq)] {
            assert_exact(
                &format!("{label}: {name} Q half"),
                &actual[base..base + q_dim],
                &oq[base..base + q_dim],
                q_dim,
            )?;
            assert_exact(
                &format!("{label}: {name} Gate half"),
                &actual[base + q_dim..base + set.q_n],
                &oq[base + q_dim..base + set.q_n],
                q_dim,
            )?;
        }
    }
    for (name, actual, expected) in [
        ("baseline K", &bk, &ok),
        ("baseline V", &bv, &ov),
        ("staged K", &sk, &ok),
        ("staged V", &sv, &ov),
    ] {
        assert_exact(&format!("{label}: {name}"), actual, expected, set.kv_n)?;
    }
    set.assert_immutable(gpu, stream)?;

    println!(
        "PASS {label} q={:016x} k={:016x} v={:016x}",
        fnv1a64(&sq),
        fnv1a64(&sk),
        fnv1a64(&sv)
    );
    oracle.free(gpu)?;
    baseline.free(gpu)?;
    staged.free(gpu)
}

fn random_set(
    gpu: &dyn GpuBackend,
    label: &str,
    k: usize,
    nq: usize,
    hd: usize,
    kv_n: usize,
    seed: u64,
) -> Result<TestSet> {
    let q_n = 2 * nq * hd;
    TestSet::new(
        gpu,
        label,
        nq,
        hd,
        kv_n,
        random_fixture(MAX_M, q_n, k, seed),
        random_fixture(1, kv_n, k, seed ^ 0x0b6d_75a1_4421_9001),
        random_fixture(1, kv_n, k, seed ^ 0xf451_223a_90ce_7003),
    )
}

fn cancellation_set(gpu: &dyn GpuBackend, k: usize) -> Result<TestSet> {
    TestSet::new(
        gpu,
        format!("cancellation-tail-k{k}"),
        TAIL_NQ,
        TAIL_HD,
        3,
        cancellation_fixture(MAX_M, TAIL_Q_N, k),
        random_fixture(1, 3, k, 0xcace_1100_0000_0001 ^ k as u64),
        random_fixture(1, 3, k, 0xcace_2200_0000_0002 ^ k as u64),
    )
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend with Qwen3.8 kernels")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernels = load_kernels(gpu).context("resolve baseline/staged attention kernels")?;
    ensure!(
        kernels.staged.complete(),
        "staged QG/dual-KV pair incomplete"
    );

    let production = random_set(
        gpu,
        "production-random",
        PROD_K,
        PROD_NQ,
        PROD_HD,
        PROD_KV_N,
        0xa57a_6e00_5120_0017,
    )?;
    let mut cases = 0usize;
    for rows in 5..=MAX_M {
        run_case(
            gpu,
            stream,
            &kernels,
            &production,
            rows,
            PROD_Q_N,
            PROD_KV_N,
            0,
            0,
        )?;
        run_case(
            gpu,
            stream,
            &kernels,
            &production,
            rows,
            PROD_STRIDE,
            PROD_STRIDE,
            PROD_Q_N,
            PROD_Q_N + PROD_KV_N,
        )?;
        cases += 2;
    }
    production.free(gpu)?;

    // K=512 fills exactly one staged wave; K=528 exercises a partial second
    // wave while remaining a legal multiple of the 16-element scale group.
    for k in [512usize, 528] {
        for kv_n in 1..=3 {
            let tail = random_set(
                gpu,
                &format!("tail-random-k{k}-kv{kv_n}"),
                k,
                TAIL_NQ,
                TAIL_HD,
                kv_n,
                0x7a11_0000_0000_0000 ^ ((k as u64) << 8) ^ kv_n as u64,
            )?;
            for rows in 5..=MAX_M {
                run_case(gpu, stream, &kernels, &tail, rows, TAIL_Q_N, kv_n, 0, 0)?;
                run_case(
                    gpu,
                    stream,
                    &kernels,
                    &tail,
                    rows,
                    TAIL_Q_N + 7,
                    kv_n + 7,
                    0,
                    0,
                )?;
                cases += 2;
            }
            tail.free(gpu)?;
        }

        let cancellation = cancellation_set(gpu, k)?;
        for rows in [5usize, MAX_M] {
            run_case(
                gpu,
                stream,
                &kernels,
                &cancellation,
                rows,
                TAIL_Q_N,
                3,
                0,
                0,
            )?;
            run_case(
                gpu,
                stream,
                &kernels,
                &cancellation,
                rows,
                TAIL_Q_N + 7,
                10,
                0,
                0,
            )?;
            cases += 2;
        }
        cancellation.free(gpu)?;
    }

    println!(
        "PASS {cases} M17 activation-staging raw cases: staged == baseline exact == serial K1"
    );
    Ok(())
}
