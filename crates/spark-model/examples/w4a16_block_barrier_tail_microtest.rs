// SPDX-License-Identifier: AGPL-3.0-only

//! Isolated tail-safety and exact-byte oracle for W4A16 kernels whose CTA
//! cooperates through block-wide barriers.
//!
//! The parent starts one child per `(symbol, N)` so a divergent-barrier hang is
//! containable. Each child compares the partial final CTA against an aligned
//! launch of the same symbol using the same inputs and first logical weight
//! rows. Run only after the CUDA tail-safety fix has been compiled:
//!
//! ```text
//! cargo run --release -p spark-model --example w4a16_block_barrier_tail_microtest
//! cargo run --release -p spark-model --example w4a16_block_barrier_tail_microtest -- \
//!     --child w4a16_gemv_batch2 3
//! ```

#[path = "w4a16_block_barrier_tail_microtest/cases.rs"]
mod cases;
#[path = "w4a16_block_barrier_tail_microtest/data.rs"]
mod data;
#[path = "w4a16_block_barrier_tail_microtest/launch.rs"]
mod launch;

use std::env;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use cases::{CASES, Case, K, MODULE};
use data::{CANARY, GuardedOutput, gather, u16s_to_le, upload, written_mask};
use launch::launch;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const CHILD_TIMEOUT: Duration = Duration::from_secs(20);

fn run_child(case: Case, n: usize) -> Result<()> {
    ensure!(
        (1..=case.group).contains(&n),
        "N must be in 1..={} for {}",
        case.group,
        case.symbol
    );
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize Atlas CUDA backend")?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernel = gpu
        .kernel(MODULE, case.symbol)
        .with_context(|| format!("resolve {MODULE}::{}", case.symbol))?;

    // Identical activation rows make any legal non-strided Q/G mapped overlap
    // deterministic when partial N retains the aligned deinterleave params.
    let input = upload(gpu, &u16s_to_le(&vec![0x3f80u16; case.rows * K]))?;

    // K=16: eight packed FP4 bytes and one E4M3 scale per output row. Tail and
    // control share these buffers, making the first N weight rows exact peers.
    let packed0 = upload(gpu, &vec![0x22; case.group * (K / 2)])?;
    let scale0 = upload(gpu, &vec![0x38; case.group * (K / 16)])?;
    let packed1 = upload(gpu, &vec![0x11; case.group * (K / 2)])?;
    let scale1 = upload(gpu, &vec![0x38; case.group * (K / 16)])?;

    let mut aligned_outputs = Vec::with_capacity(case.projections());
    let mut tail_outputs = Vec::with_capacity(case.projections());
    for _ in 0..case.projections() {
        aligned_outputs.push(GuardedOutput::new(gpu, case.output_words())?);
        tail_outputs.push(GuardedOutput::new(gpu, case.output_words())?);
    }
    let output_ptr = |outputs: &[GuardedOutput], index: usize| {
        outputs
            .get(index)
            .map_or(DevicePtr::NULL, GuardedOutput::payload_ptr)
    };

    launch(
        gpu,
        stream,
        kernel,
        case,
        case.group,
        input,
        packed0,
        scale0,
        output_ptr(&aligned_outputs, 0),
        packed1,
        scale1,
        output_ptr(&aligned_outputs, 1),
    )?;
    launch(
        gpu,
        stream,
        kernel,
        case,
        n,
        input,
        packed0,
        scale0,
        output_ptr(&tail_outputs, 0),
        packed1,
        scale1,
        output_ptr(&tail_outputs, 1),
    )?;
    gpu.synchronize(stream)?;

    let aligned_mask = written_mask(case, case.group);
    let tail_mask = written_mask(case, n);
    for projection in 0..case.projections() {
        let aligned = aligned_outputs[projection].read_and_check(
            gpu,
            stream,
            &format!("{} aligned projection {projection}", case.symbol),
            &aligned_mask,
        )?;
        let tail = tail_outputs[projection].read_and_check(
            gpu,
            stream,
            &format!("{} N={n} projection {projection}", case.symbol),
            &tail_mask,
        )?;
        let expected = gather(case, case.group, n, &aligned);
        let actual = gather(case, n, n, &tail);
        ensure!(
            actual == expected,
            "{} N={n} projection {projection}: exact BF16 mismatch: actual={actual:04x?} expected={expected:04x?}",
            case.symbol
        );
        ensure!(
            actual.iter().any(|&word| word != CANARY),
            "{} N={n} projection {projection}: no output was written",
            case.symbol
        );
    }

    for output in aligned_outputs.iter().chain(&tail_outputs) {
        output.free(gpu)?;
    }
    for ptr in [input, packed0, scale0, packed1, scale1] {
        gpu.free(ptr)?;
    }
    println!(
        "PASS {} N={n} K={K} group={} rows={} abi={:?}",
        case.symbol, case.group, case.rows, case.abi
    );
    Ok(())
}

fn run_parent() -> Result<()> {
    let executable = env::current_exe().context("locate current example executable")?;
    let mut passed = 0usize;
    for case in CASES {
        for n in 1..=case.group {
            eprintln!("RUN {} N={n}/{}", case.symbol, case.group);
            let n_arg = n.to_string();
            let mut child = Command::new(&executable)
                .args(["--child", case.symbol, n_arg.as_str()])
                .stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .spawn()
                .with_context(|| format!("spawn {} N={n}", case.symbol))?;
            let deadline = Instant::now() + CHILD_TIMEOUT;
            loop {
                if let Some(status) = child.try_wait()? {
                    ensure!(
                        status.success(),
                        "{} N={n}: child exited with {status}",
                        case.symbol
                    );
                    passed += 1;
                    break;
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    bail!(
                        "{} N={n}: child exceeded {:?}; possible divergent block barrier",
                        case.symbol,
                        CHILD_TIMEOUT
                    );
                }
                thread::sleep(Duration::from_millis(25));
            }
        }
    }
    println!("PASS all {passed} isolated symbol/residue cases");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [] => run_parent(),
        [flag] if flag == "--list" => {
            for case in CASES {
                println!(
                    "{}: N=1..={} K={K} block={} rows={} abi={:?}",
                    case.symbol, case.group, case.block, case.rows, case.abi
                );
            }
            Ok(())
        }
        [flag, symbol, n] if flag == "--child" => {
            let case = CASES
                .iter()
                .copied()
                .find(|case| case.symbol == symbol.as_str())
                .with_context(|| format!("unknown W4A16 symbol {symbol:?}"))?;
            run_child(case, n.parse().context("parse child N")?)
        }
        _ => bail!("usage: w4a16_block_barrier_tail_microtest [--list | --child SYMBOL N]"),
    }
}
