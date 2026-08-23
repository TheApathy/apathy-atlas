// SPDX-License-Identifier: AGPL-3.0-only

//! Live CUDA exactness oracle for DSpark's device-resident Markov add-argmax.
//!
//! This compares `argmax_add_bf16` with the former host algorithm: convert the
//! two BF16 operands to FP32, add, then scan left-to-right with strict `>`.
//! The fixture covers a non-1024 vocabulary, exact ties, the public
//! 248077-logical/248320-padded boundary, and the public gamma=7 proposal
//! width. Seven gamma rows are enqueued on one stream and downloaded once.
//!
//! This is deliberately a live post-build gate, not a unit-test substitute:
//!
//!   cargo run --release -p spark-model --example dspark_argmax_add_microtest
//!
//! Exit 0 means every device token exactly matched the CPU oracle. A missing
//! kernel, launch failure, or token mismatch is a hard failure.

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const PUBLIC_LOGICAL_VOCAB: usize = 248_077;
const PUBLIC_PADDED_VOCAB: usize = 248_320;
const PUBLIC_GAMMA: usize = 7;

#[derive(Clone)]
struct Row {
    base: Vec<u16>,
    bias: Vec<u16>,
}

fn bits(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}

fn filled(len: usize, value: f32) -> Vec<u16> {
    vec![bits(value); len]
}

fn u16s_to_le(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn upload(gpu: &dyn GpuBackend, values: &[u16]) -> Result<DevicePtr> {
    let bytes = u16s_to_le(values);
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

/// Exact oracle for the host code replaced by the device-resident P2 path.
fn host_oracle(base: &[u16], bias: &[u16], logical_vocab: usize) -> u32 {
    let mut best_token = 0u32;
    let mut best_value = f32::NEG_INFINITY;
    for token in 0..logical_vocab {
        let value = bf16::from_bits(base[token]).to_f32() + bf16::from_bits(bias[token]).to_f32();
        if value > best_value {
            best_value = value;
            best_token = token as u32;
        }
    }
    best_token
}

fn run_rows(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernel: KernelHandle,
    label: &str,
    rows: &[Row],
    logical_vocab: usize,
    physical_stride: usize,
) -> Result<Vec<u32>> {
    ensure!(
        !rows.is_empty(),
        "{label}: fixture must contain at least one row"
    );
    ensure!(
        logical_vocab <= physical_stride,
        "{label}: logical vocab {logical_vocab} exceeds physical stride {physical_stride}"
    );
    for (row_index, row) in rows.iter().enumerate() {
        ensure!(
            row.base.len() == physical_stride && row.bias.len() == physical_stride,
            "{label}: row {row_index} does not match physical stride {physical_stride}"
        );
    }

    let base: Vec<u16> = rows
        .iter()
        .flat_map(|row| row.base.iter().copied())
        .collect();
    let bias: Vec<u16> = rows
        .iter()
        .flat_map(|row| row.bias.iter().copied())
        .collect();
    let base_dev = upload(gpu, &base)?;
    let bias_dev = upload(gpu, &bias)?;
    let out_dev = gpu.alloc(rows.len() * size_of::<u32>())?;

    for row in 0..rows.len() {
        let row_bytes = row * physical_stride * size_of::<u16>();
        ops::argmax_add_bf16(
            gpu,
            kernel,
            base_dev.offset(row_bytes),
            bias_dev.offset(row_bytes),
            out_dev.offset(row * size_of::<u32>()),
            logical_vocab as u32,
            stream,
        )
        .with_context(|| format!("{label}: argmax_add_bf16 launch for row {row}"))?;
    }

    // Mirrors the P2 hot path: all row kernels are ordered on one producer
    // stream, followed by one terminal gamma*u32 D2H and synchronization.
    let mut output_bytes = vec![0u8; rows.len() * size_of::<u32>()];
    gpu.copy_d2h_on_stream(out_dev, &mut output_bytes, stream)
        .with_context(|| format!("{label}: terminal token readback"))?;

    for ptr in [base_dev, bias_dev, out_dev] {
        gpu.free(ptr)?;
    }

    let actual: Vec<u32> = output_bytes
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    let expected: Vec<u32> = rows
        .iter()
        .map(|row| host_oracle(&row.base, &row.bias, logical_vocab))
        .collect();
    if actual != expected {
        bail!("{label}: GPU tokens {actual:?} != former host oracle {expected:?}");
    }
    println!("PASS {label}: {actual:?}");
    Ok(actual)
}

fn non_1024_case() -> (Vec<Row>, usize, usize) {
    let vocab = 1_037;
    let mut row = Row {
        base: filled(vocab, -8.0),
        bias: filled(vocab, 0.0),
    };
    row.base[vocab - 1] = bits(9.0);
    (vec![row], vocab, vocab)
}

fn exact_tie_case() -> (Vec<Row>, usize, usize) {
    let vocab = 2_053;
    let mut row = Row {
        base: filled(vocab, -8.0),
        bias: filled(vocab, 0.0),
    };
    // IDs 5 and 1029 share a reduction lane; ID 1028 reaches the same value
    // through another lane/tree branch. The exact answer must be ID 5.
    for token in [5, 1_028, 1_029] {
        row.base[token] = bits(3.0);
        row.bias[token] = bits(4.0);
    }
    (vec![row], vocab, vocab)
}

fn padded_tail_case() -> (Vec<Row>, usize, usize) {
    let mut row = Row {
        base: filled(PUBLIC_PADDED_VOCAB, -8.0),
        bias: filled(PUBLIC_PADDED_VOCAB, 0.0),
    };
    row.base[PUBLIC_LOGICAL_VOCAB - 1] = bits(11.0);
    // These padded-only scores would win if the kernel consumed the drafter
    // allocation width instead of the logical target vocabulary.
    row.base[PUBLIC_LOGICAL_VOCAB] = bits(50.0);
    row.bias[PUBLIC_PADDED_VOCAB - 1] = bits(100.0);
    (vec![row], PUBLIC_LOGICAL_VOCAB, PUBLIC_PADDED_VOCAB)
}

fn gamma7_case() -> (Vec<Row>, usize, usize, Vec<u32>) {
    let vocab = PUBLIC_LOGICAL_VOCAB;
    let mut predecessor = 42usize;
    let mut expected_chain = Vec::with_capacity(PUBLIC_GAMMA);
    let mut rows = Vec::with_capacity(PUBLIC_GAMMA);
    for position in 0..PUBLIC_GAMMA {
        let selected = (predecessor * 37 + 17 + position * 13) % vocab;
        let competitor = (selected + 137) % vocab;
        let mut row = Row {
            base: filled(vocab, -8.0),
            bias: filled(vocab, 0.0),
        };
        row.base[competitor] = bits(4.0);
        row.base[selected] = bits(2.0);
        row.bias[selected] = bits(5.0);
        rows.push(row);
        expected_chain.push(selected as u32);
        predecessor = selected;
    }
    (rows, vocab, vocab, expected_chain)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend with the compiled Atlas kernel set")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernel = gpu
        .kernel("argmax", "argmax_add_bf16")
        .context("required DSpark kernel argmax::argmax_add_bf16 is absent")?;

    let (rows, logical, physical) = non_1024_case();
    let tokens = run_rows(
        gpu,
        stream,
        kernel,
        "non-1024 logical vocab",
        &rows,
        logical,
        physical,
    )?;
    ensure!(tokens == [1_036], "non-1024 fixture construction drifted");

    let (rows, logical, physical) = exact_tie_case();
    let tokens = run_rows(
        gpu,
        stream,
        kernel,
        "exact lowest-ID tie",
        &rows,
        logical,
        physical,
    )?;
    ensure!(tokens == [5], "tie fixture itself did not select token 5");

    let (rows, logical, physical) = padded_tail_case();
    let tokens = run_rows(
        gpu,
        stream,
        kernel,
        "public padded-tail exclusion",
        &rows,
        logical,
        physical,
    )?;
    ensure!(
        tokens == [(PUBLIC_LOGICAL_VOCAB - 1) as u32],
        "padded-tail fixture selected outside the expected logical winner"
    );

    let (rows, logical, physical, expected_chain) = gamma7_case();
    let tokens = run_rows(
        gpu,
        stream,
        kernel,
        "public gamma7 chain",
        &rows,
        logical,
        physical,
    )?;
    ensure!(
        tokens == expected_chain,
        "gamma7 fixture construction drifted"
    );

    println!("PASS: argmax_add_bf16 exactly matches the former CPU oracle for all required cases");
    Ok(())
}
