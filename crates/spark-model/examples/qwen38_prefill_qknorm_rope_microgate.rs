// SPDX-License-Identifier: AGPL-3.0-only

//! Raw fail-closed gate for the Qwen3.8 Q/G + Q/K norm + RoPE fusion.
//!
//! The candidate is compared byte-for-byte with the shipped three-launch
//! parent chain. Both scalar cache-skip positions and distinct T/H/W paged
//! positions are covered, together with 4-KiB redzones, Q/G row gaps, and
//! immutable weights/position inputs. Timing is optional and runs only after
//! parity succeeds.
//!
//! GPU qualification command (never run on an unreserved device):
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --features cuda \
//!   --example qwen38_prefill_qknorm_rope_microgate
//! ```
//!
//! Set `ATLAS_QKNORM_ROPE_MICROGATE_FULL=1` for the complete token/position
//! boundary matrix. Set `ATLAS_QKNORM_ROPE_MICROGATE_TIMING=1` to require the
//! candidate to beat its parent at 2K and 8K in both position modes.

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use spark_model::layers::ops::{
    deinterleave_qg_split_qnorm, qwen38_prefill_qknorm_rope, rms_norm, rope, rope_mrope_interleaved,
};
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const NQ: u32 = 24;
const NKV: u32 = 4;
const HD: u32 = 256;
const QG_WORDS: usize = NQ as usize * HD as usize * 2;
const Q_WORDS: usize = NQ as usize * HD as usize;
const K_WORDS: usize = NKV as usize * HD as usize;
const ROTARY: u32 = 64;
const THETA: f32 = 10_000_000.0;
const PROD_EPS: f32 = 1.0e-6;
const REDZONE: usize = 4 * 1024;
const CANARY_BYTE: u8 = 0xa5;
const GAP_CANARY: u16 = 0xa55a;

#[derive(Clone, Copy, Debug)]
enum PositionMode {
    Scalar,
    Mrope,
}

impl PositionMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Mrope => "mrope-thw",
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    tokens: u32,
    qg_stride: u32,
    eps: f32,
    mode: PositionMode,
    extreme: bool,
}

impl Case {
    fn label(self) -> String {
        format!(
            "n{}_stride{}_eps{:08x}_{}_{}",
            self.tokens,
            self.qg_stride,
            self.eps.to_bits(),
            self.mode.label(),
            if self.extreme { "extreme" } else { "finite" }
        )
    }
}

struct Kernels {
    deinterleave_qnorm: KernelHandle,
    rms_norm: KernelHandle,
    rope: KernelHandle,
    mrope: KernelHandle,
    candidate: KernelHandle,
}

struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    image: Vec<u8>,
}

impl Guarded {
    fn input(gpu: &dyn GpuBackend, payload: &[u8]) -> Result<Self> {
        let mut image = vec![CANARY_BYTE; REDZONE + payload.len() + REDZONE];
        image[REDZONE..REDZONE + payload.len()].copy_from_slice(payload);
        let allocation = gpu.alloc(image.len())?;
        gpu.copy_h2d(&image, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            image,
        })
    }

    fn output(gpu: &dyn GpuBackend, payload_len: usize) -> Result<Self> {
        Self::input(gpu, &vec![CANARY_BYTE; payload_len])
    }

    fn ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.image, self.allocation)
    }

    fn read_all(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.image.len()];
        gpu.copy_d2h(self.allocation, &mut bytes)?;
        Ok(bytes)
    }

    fn payload(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        let bytes = self.read_all(gpu)?;
        ensure!(
            bytes[..REDZONE].iter().all(|&byte| byte == CANARY_BYTE),
            "{label}: leading 4-KiB redzone changed"
        );
        let suffix = REDZONE + self.payload_len;
        ensure!(
            bytes[suffix..].iter().all(|&byte| byte == CANARY_BYTE),
            "{label}: trailing 4-KiB redzone changed"
        );
        Ok(bytes[REDZONE..suffix].to_vec())
    }

    fn immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        ensure!(self.read_all(gpu)? == self.image, "{label}: input changed");
        Ok(())
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}

struct MutableSet {
    qg: Guarded,
    q: Guarded,
    k: Guarded,
}

impl MutableSet {
    fn new(gpu: &dyn GpuBackend, qg: &[u8], k: &[u8], q_bytes: usize) -> Result<Self> {
        Ok(Self {
            qg: Guarded::input(gpu, qg)?,
            q: Guarded::output(gpu, q_bytes)?,
            k: Guarded::input(gpu, k)?,
        })
    }

    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.qg.reset(gpu)?;
        self.q.reset(gpu)?;
        self.k.reset(gpu)
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.qg.free(gpu)?;
        self.q.free(gpu)?;
        self.k.free(gpu)
    }
}

const FINITE_BF16_BITS: [u16; 18] = [
    0xc100, 0xc040, 0xbf80, 0xbf00, 0x8080, 0x8001, 0x8000, 0x0000, 0x0001, 0x0080, 0x3d00, 0x3f00,
    0x3f80, 0x3fc0, 0x4040, 0x40c0, 0x4100, 0x4180,
];
const EXTREME_BF16_BITS: [u16; 6] = [0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffc1];

fn words_to_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn u32_to_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn patterned_words(count: usize, salt: usize) -> Vec<u16> {
    (0..count)
        .map(|index| FINITE_BF16_BITS[(index.wrapping_mul(17) + salt) % FINITE_BF16_BITS.len()])
        .collect()
}

fn inject_extremes(words: &mut [u16]) {
    for (word, value) in words.iter_mut().zip(EXTREME_BF16_BITS) {
        *word = value;
    }
}

fn qg_fixture(case: Case) -> Vec<u8> {
    let stride = case.qg_stride as usize;
    let mut words = vec![GAP_CANARY; case.tokens as usize * stride];
    for token in 0..case.tokens as usize {
        let mut active = patterned_words(QG_WORDS, token.wrapping_mul(29));
        if case.extreme && token == 0 {
            inject_extremes(&mut active);
        }
        words[token * stride..token * stride + QG_WORDS].copy_from_slice(&active);
    }
    words_to_bytes(&words)
}

fn assert_qg_gaps(bytes: &[u8], case: Case, label: &str) -> Result<()> {
    let stride = case.qg_stride as usize;
    if stride == QG_WORDS {
        return Ok(());
    }
    for token in 0..case.tokens as usize {
        for word in QG_WORDS..stride {
            let byte = (token * stride + word) * 2;
            let actual = u16::from_le_bytes([bytes[byte], bytes[byte + 1]]);
            ensure!(
                actual == GAP_CANARY,
                "{label}: token {token} Q/G gap word {word} changed to 0x{actual:04x}"
            );
        }
    }
    Ok(())
}

fn position_fixtures(tokens: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    const BOUNDARIES: [u32; 10] = [
        0,
        1,
        31,
        32,
        262_143,
        1_048_576,
        16_777_215,
        16_777_216,
        16_777_217,
        u32::MAX - 1,
    ];
    let t: Vec<u32> = (0..tokens)
        .map(|index| BOUNDARIES[index % BOUNDARIES.len()])
        .collect();
    let h: Vec<u32> = (0..tokens)
        .map(|index| BOUNDARIES[(index * 3 + 1) % BOUNDARIES.len()])
        .collect();
    let w: Vec<u32> = (0..tokens)
        .map(|index| BOUNDARIES[(index * 7 + 2) % BOUNDARIES.len()])
        .collect();
    (t, h, w)
}

#[allow(clippy::too_many_arguments)]
fn launch_parent(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    buffers: &MutableSet,
    q_weight: DevicePtr,
    k_weight: DevicePtr,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    case: Case,
) -> Result<()> {
    deinterleave_qg_split_qnorm(
        gpu,
        kernels.deinterleave_qnorm,
        buffers.qg.ptr(),
        buffers.q.ptr(),
        q_weight,
        case.tokens,
        NQ,
        HD,
        case.qg_stride,
        case.eps,
        stream,
    )?;
    rms_norm(
        gpu,
        kernels.rms_norm,
        buffers.k.ptr(),
        &DenseWeight { weight: k_weight },
        buffers.k.ptr(),
        case.tokens * NKV,
        HD,
        case.eps,
        stream,
    )?;
    match case.mode {
        PositionMode::Scalar => rope(
            gpu,
            kernels.rope,
            buffers.q.ptr(),
            buffers.k.ptr(),
            pos_t,
            case.tokens,
            NQ,
            NKV,
            HD,
            ROTARY,
            THETA,
            stream,
        ),
        PositionMode::Mrope => rope_mrope_interleaved(
            gpu,
            kernels.mrope,
            buffers.q.ptr(),
            buffers.k.ptr(),
            pos_t,
            pos_h,
            pos_w,
            case.tokens,
            NQ,
            NKV,
            HD,
            ROTARY,
            THETA,
            stream,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_candidate(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: &Kernels,
    buffers: &MutableSet,
    q_weight: DevicePtr,
    k_weight: DevicePtr,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    case: Case,
) -> Result<()> {
    let (pos_h, pos_w) = match case.mode {
        PositionMode::Scalar => (pos_t, pos_t),
        PositionMode::Mrope => (pos_h, pos_w),
    };
    qwen38_prefill_qknorm_rope(
        gpu,
        kernels.candidate,
        buffers.qg.ptr(),
        buffers.q.ptr(),
        buffers.k.ptr(),
        q_weight,
        k_weight,
        pos_t,
        pos_h,
        pos_w,
        case.tokens,
        NQ,
        NKV,
        HD,
        case.qg_stride,
        ROTARY,
        case.eps,
        THETA,
        stream,
    )
}

fn first_mismatch(lhs: &[u8], rhs: &[u8]) -> Option<usize> {
    lhs.iter().zip(rhs).position(|(left, right)| left != right)
}

fn require_equal(label: &str, parent: &[u8], candidate: &[u8]) -> Result<()> {
    if parent != candidate {
        let index = first_mismatch(parent, candidate).context("mismatch without an index")?;
        bail!(
            "{label}: byte {index} differs: parent=0x{:02x} candidate=0x{:02x}",
            parent[index],
            candidate[index]
        );
    }
    Ok(())
}

fn run_case(gpu: &dyn GpuBackend, stream: u64, kernels: &Kernels, case: Case) -> Result<()> {
    ensure!(
        case.qg_stride as usize >= QG_WORDS,
        "Q/G stride is too small"
    );
    let label = case.label();
    let qg = qg_fixture(case);
    let mut k_words = patterned_words(case.tokens as usize * K_WORDS, 7);
    if case.extreme {
        inject_extremes(&mut k_words);
    }
    let k = words_to_bytes(&k_words);
    let q_weight_bytes = words_to_bytes(&patterned_words(HD as usize, 11));
    let k_weight_bytes = words_to_bytes(&patterned_words(HD as usize, 13));
    let (pos_t_values, pos_h_values, pos_w_values) = position_fixtures(case.tokens as usize);
    let q_weight = Guarded::input(gpu, &q_weight_bytes)?;
    let k_weight = Guarded::input(gpu, &k_weight_bytes)?;
    let pos_t = Guarded::input(gpu, &u32_to_bytes(&pos_t_values))?;
    let pos_h = Guarded::input(gpu, &u32_to_bytes(&pos_h_values))?;
    let pos_w = Guarded::input(gpu, &u32_to_bytes(&pos_w_values))?;
    let q_bytes = case.tokens as usize * Q_WORDS * 2;
    let parent = MutableSet::new(gpu, &qg, &k, q_bytes)?;
    let candidate = MutableSet::new(gpu, &qg, &k, q_bytes)?;

    launch_parent(
        gpu,
        stream,
        kernels,
        &parent,
        q_weight.ptr(),
        k_weight.ptr(),
        pos_t.ptr(),
        pos_h.ptr(),
        pos_w.ptr(),
        case,
    )?;
    launch_candidate(
        gpu,
        stream,
        kernels,
        &candidate,
        q_weight.ptr(),
        k_weight.ptr(),
        pos_t.ptr(),
        pos_h.ptr(),
        pos_w.ptr(),
        case,
    )?;
    gpu.synchronize(stream)?;

    let parent_qg = parent.qg.payload(gpu, &format!("{label}/parent-qg"))?;
    let candidate_qg = candidate
        .qg
        .payload(gpu, &format!("{label}/candidate-qg"))?;
    let parent_q = parent.q.payload(gpu, &format!("{label}/parent-q"))?;
    let candidate_q = candidate.q.payload(gpu, &format!("{label}/candidate-q"))?;
    let parent_k = parent.k.payload(gpu, &format!("{label}/parent-k"))?;
    let candidate_k = candidate.k.payload(gpu, &format!("{label}/candidate-k"))?;
    require_equal(&format!("{label}/qg"), &parent_qg, &candidate_qg)?;
    require_equal(&format!("{label}/q"), &parent_q, &candidate_q)?;
    require_equal(&format!("{label}/k"), &parent_k, &candidate_k)?;
    assert_qg_gaps(&parent_qg, case, &format!("{label}/parent"))?;
    assert_qg_gaps(&candidate_qg, case, &format!("{label}/candidate"))?;
    q_weight.immutable(gpu, &format!("{label}/q-weight"))?;
    k_weight.immutable(gpu, &format!("{label}/k-weight"))?;
    pos_t.immutable(gpu, &format!("{label}/pos-t"))?;
    pos_h.immutable(gpu, &format!("{label}/pos-h"))?;
    pos_w.immutable(gpu, &format!("{label}/pos-w"))?;
    println!(
        "PASS {label}: qg={} q={} k={} byte-identical",
        parent_qg.len(),
        parent_q.len(),
        parent_k.len()
    );

    parent.free(gpu)?;
    candidate.free(gpu)?;
    q_weight.free(gpu)?;
    k_weight.free(gpu)?;
    pos_t.free(gpu)?;
    pos_h.free(gpu)?;
    pos_w.free(gpu)
}

fn time_case(gpu: &dyn GpuBackend, stream: u64, kernels: &Kernels, case: Case) -> Result<()> {
    let qg = qg_fixture(case);
    let k = words_to_bytes(&patterned_words(case.tokens as usize * K_WORDS, 7));
    let q_weight = Guarded::input(gpu, &words_to_bytes(&patterned_words(HD as usize, 11)))?;
    let k_weight = Guarded::input(gpu, &words_to_bytes(&patterned_words(HD as usize, 13)))?;
    let (pos_t_values, pos_h_values, pos_w_values) = position_fixtures(case.tokens as usize);
    let pos_t = Guarded::input(gpu, &u32_to_bytes(&pos_t_values))?;
    let pos_h = Guarded::input(gpu, &u32_to_bytes(&pos_h_values))?;
    let pos_w = Guarded::input(gpu, &u32_to_bytes(&pos_w_values))?;
    let buffers = MutableSet::new(gpu, &qg, &k, case.tokens as usize * Q_WORDS * 2)?;

    let measure_once = |candidate: bool| -> Result<f64> {
        buffers.reset(gpu)?;
        gpu.synchronize(stream)?;
        let start = Instant::now();
        if candidate {
            launch_candidate(
                gpu,
                stream,
                kernels,
                &buffers,
                q_weight.ptr(),
                k_weight.ptr(),
                pos_t.ptr(),
                pos_h.ptr(),
                pos_w.ptr(),
                case,
            )?;
        } else {
            launch_parent(
                gpu,
                stream,
                kernels,
                &buffers,
                q_weight.ptr(),
                k_weight.ptr(),
                pos_t.ptr(),
                pos_h.ptr(),
                pos_w.ptr(),
                case,
            )?;
        }
        gpu.synchronize(stream)?;
        Ok(start.elapsed().as_secs_f64() * 1_000.0)
    };

    let _ = measure_once(false)?;
    let _ = measure_once(true)?;
    let mut parent_samples = Vec::with_capacity(21);
    let mut candidate_samples = Vec::with_capacity(21);
    for round in 0..21 {
        if round % 2 == 0 {
            parent_samples.push(measure_once(false)?);
            candidate_samples.push(measure_once(true)?);
        } else {
            candidate_samples.push(measure_once(true)?);
            parent_samples.push(measure_once(false)?);
        }
    }
    parent_samples.sort_by(f64::total_cmp);
    candidate_samples.sort_by(f64::total_cmp);
    let parent_ms = parent_samples[parent_samples.len() / 2];
    let candidate_ms = candidate_samples[candidate_samples.len() / 2];
    let parent_p90 = parent_samples[parent_samples.len() * 9 / 10];
    let candidate_p90 = candidate_samples[candidate_samples.len() * 9 / 10];
    println!(
        "TIMING {}: median parent={parent_ms:.3} ms candidate={candidate_ms:.3} ms \
         ratio={:.4}; p90 parent={parent_p90:.3} ms candidate={candidate_p90:.3} ms",
        case.label(),
        candidate_ms / parent_ms
    );
    ensure!(
        candidate_ms < parent_ms,
        "{}: candidate did not beat parent; stop promotion",
        case.label()
    );

    buffers.free(gpu)?;
    q_weight.free(gpu)?;
    k_weight.free(gpu)?;
    pos_t.free(gpu)?;
    pos_h.free(gpu)?;
    pos_w.free(gpu)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernels = Kernels {
        deinterleave_qnorm: gpu.kernel("ssm_preprocess", "deinterleave_qg_split_qnorm")?,
        rms_norm: gpu.kernel("norm", "rms_norm")?,
        rope: gpu.kernel("rope", "rope_forward")?,
        mrope: gpu.kernel("rope_mrope_interleaved", "rope_forward_mrope_interleaved")?,
        candidate: gpu.kernel("qwen38_prefill_qknorm_rope", "qwen38_prefill_qknorm_rope")?,
    };

    let smoke = [
        Case {
            tokens: 31,
            qg_stride: QG_WORDS as u32,
            eps: PROD_EPS,
            mode: PositionMode::Scalar,
            extreme: false,
        },
        Case {
            tokens: 129,
            qg_stride: QG_WORDS as u32 + 17,
            eps: PROD_EPS,
            mode: PositionMode::Mrope,
            extreme: false,
        },
    ];
    let mut full = Vec::new();
    for tokens in [1, 2, 3, 31, 32, 33, 127, 128, 129, 2048, 8192] {
        for mode in [PositionMode::Scalar, PositionMode::Mrope] {
            full.push(Case {
                tokens,
                qg_stride: if tokens & 1 == 0 {
                    QG_WORDS as u32
                } else {
                    QG_WORDS as u32 + 17
                },
                eps: if tokens == 3 { 0.0 } else { PROD_EPS },
                mode,
                extreme: tokens == 33,
            });
        }
    }
    let cases: &[Case] = if std::env::var("ATLAS_QKNORM_ROPE_MICROGATE_FULL")
        .ok()
        .as_deref()
        == Some("1")
    {
        &full
    } else {
        &smoke
    };
    for &case in cases {
        run_case(gpu, stream, &kernels, case)?;
    }

    if std::env::var("ATLAS_QKNORM_ROPE_MICROGATE_TIMING")
        .ok()
        .as_deref()
        == Some("1")
    {
        for tokens in [2048, 8192] {
            for mode in [PositionMode::Scalar, PositionMode::Mrope] {
                let case = Case {
                    tokens,
                    qg_stride: QG_WORDS as u32,
                    eps: PROD_EPS,
                    mode,
                    extreme: false,
                };
                run_case(gpu, stream, &kernels, case)?;
                time_case(gpu, stream, &kernels, case)?;
            }
        }
    }
    println!("ALL PASS: Qwen3.8 fused Q/K norm + RoPE matches the parent chain");
    Ok(())
}
