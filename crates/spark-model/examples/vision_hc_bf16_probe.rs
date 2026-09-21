// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded numerical probe, no model/weights/timing. Root owns GPU execution.
//! cargo run --release -p spark-model --features gpu-examples --example
//! vision_hc_bf16_probe -- [--capture /absolute/V14/capture]

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "vision_hc_bf16_probe/inputs.rs"]
mod inputs;
use inputs::{H, HC, Inputs, rounded_bits};

struct Arena<'a> {
    gpu: &'a dyn GpuBackend,
    allocations: Vec<DevicePtr>,
}

impl Arena<'_> {
    fn upload(&mut self, data: &[u8]) -> Result<DevicePtr> {
        let ptr = self.gpu.alloc(data.len())?;
        self.allocations.push(ptr);
        self.gpu.copy_h2d(data, ptr)?;
        Ok(ptr)
    }
}

impl Drop for Arena<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.synchronize(self.gpu.default_stream());
        for ptr in self.allocations.drain(..) {
            let _ = self.gpu.free(ptr);
        }
    }
}

fn run(gpu: &dyn GpuBackend, old: KernelHandle, new: KernelHandle, input: &Inputs) -> Result<()> {
    ensure!([1, 12].contains(&input.rows), "unsupported row count");
    for (data, bytes, bf16) in [
        (&input.block, input.rows * H * 2, true),
        (&input.residual, input.rows * HC * H * 4, false),
        (&input.post, input.rows * HC * 4, false),
        (&input.comb, input.rows * HC * HC * 4, false),
    ] {
        ensure!(data.len() == bytes, "input size mismatch");
        for cell in data.chunks_exact(if bf16 { 2 } else { 4 }) {
            let value = if bf16 {
                f32::from_bits((u16::from_le_bytes(cell.try_into()?) as u32) << 16)
            } else {
                f32::from_le_bytes(cell.try_into()?)
            };
            ensure!(value.is_finite(), "nonfinite probe input");
        }
    }
    let mut arena = Arena {
        gpu,
        allocations: vec![],
    };
    let block = arena.upload(&input.block)?;
    let residual = arena.upload(&input.residual)?;
    let post = arena.upload(&input.post)?;
    let comb = arena.upload(&input.comb)?;
    let disjoint = arena.upload(&vec![0xa5; input.residual.len()])?;
    let mut reference: Option<Vec<u8>> = None;
    for shards in [1, 16] {
        for inplace in [false, true] {
            for (label, handle) in [("baseline", old), ("BF16-RNE", new)] {
                gpu.copy_h2d(&input.residual, residual)?;
                gpu.memset(disjoint, 0xa5, input.residual.len())?;
                let out = if inplace { residual } else { disjoint };
                KernelLaunch::new(gpu, handle)
                    .grid([input.rows as u32, shards, 1])
                    .block([256, 1, 1])
                    .arg_ptr(block)
                    .arg_ptr(residual)
                    .arg_ptr(post)
                    .arg_ptr(comb)
                    .arg_ptr(out)
                    .arg_u32(H as u32)
                    .arg_u32(HC as u32)
                    .launch(gpu.default_stream())?;
                gpu.synchronize(gpu.default_stream())?;
                let mut got = vec![0; input.residual.len()];
                gpu.copy_d2h(out, &mut got)?;
                if label == "baseline" {
                    if let Some(captured) = &input.captured_output {
                        ensure!(
                            &got == captured,
                            "baseline differs from captured HCpost output"
                        );
                    }
                    for cell in got.chunks_exact(4) {
                        ensure!(
                            f32::from_le_bytes(cell.try_into()?).is_finite(),
                            "nonfinite baseline output"
                        );
                    }
                    if let Some(expected) = &reference {
                        ensure!(
                            &got == expected,
                            "baseline changed across alias/shard controls"
                        );
                    } else {
                        reference = Some(got);
                    }
                } else {
                    let expected = reference.as_ref().unwrap();
                    for (i, (cell, base)) in got
                        .chunks_exact(4)
                        .zip(expected.chunks_exact(4))
                        .enumerate()
                    {
                        let actual = u32::from_le_bytes(cell.try_into()?);
                        let want = rounded_bits(u32::from_le_bytes(base.try_into()?));
                        ensure!(
                            actual == want,
                            "{} N{} shards{shards} inplace{inplace} cell{i}: {actual:08x} != RNE {want:08x}",
                            input.label,
                            input.rows
                        );
                    }
                }
                println!(
                    "PASS {} N{} shards{shards} inplace{inplace} {label} cells{}",
                    input.label,
                    input.rows,
                    input.rows * HC * H
                );
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.is_empty() || (args.len() == 2 && args[0] == "--capture"),
        "usage: vision_hc_bf16_probe [--capture /absolute/V14/capture]"
    );
    let mut cases = Vec::new();
    for rows in [1, 12] {
        for boundary in [true, false] {
            cases.push(inputs::synthetic(rows, boundary));
        }
    }
    if args.len() == 2 {
        cases.extend(inputs::captured(std::path::Path::new(&args[1]))?);
    }
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let old = gpu.kernel("hyper_connection", "hc_post")?;
    let new = gpu.kernel(
        "deepseek_vision_hc_post_bf16",
        "deepseek_vision_hc_post_bf16",
    )?;
    ensure!(
        old.0 != 0 && new.0 != 0 && old.0 != new.0,
        "missing/aliased probe handles"
    );
    for case in &cases {
        run(&gpu, old, new, case)?;
    }
    println!(
        "PASS all {} cases; boundary-only parity, no whole-model quality or speed claim",
        cases.len()
    );
    Ok(())
}
