// SPDX-License-Identifier: AGPL-3.0-only

//! Physical smoke test for the pinned GLM-5.3 EXL3 cooperative GEMM ABI.

use anyhow::{Context, Result, bail};
use spark_model::layers::ops::{
    Glm53Exl3Buffer, Glm53Exl3GemmBuffers, Glm53Exl3GemmKernel, Glm53Exl3GemmPlan, Glm53Exl3Output,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

struct Allocations<'a> {
    gpu: &'a dyn GpuBackend,
    pointers: Vec<DevicePtr>,
}

impl<'a> Allocations<'a> {
    fn new(gpu: &'a dyn GpuBackend) -> Self {
        Self {
            gpu,
            pointers: Vec::new(),
        }
    }

    fn zeroed(&mut self, bytes: usize) -> Result<Glm53Exl3Buffer> {
        let ptr = self.gpu.alloc(bytes)?;
        self.pointers.push(ptr);
        self.gpu.memset(ptr, 0, bytes)?;
        Ok(Glm53Exl3Buffer { ptr, bytes })
    }

    fn free_all(mut self) -> Result<()> {
        let mut first_error = None;
        for ptr in self.pointers.drain(..).rev() {
            if let Err(error) = self.gpu.free(ptr) {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error).context("free GLM EXL3 microtest allocation"),
            None => Ok(()),
        }
    }
}

fn run(gpu: &dyn GpuBackend) -> Result<()> {
    let plan = Glm53Exl3GemmPlan::new(1, 128, 128, 2, Glm53Exl3Output::F16)?;
    let kernel = Glm53Exl3GemmKernel::load(gpu, &plan)?;
    let stream = gpu.create_stream()?;
    let mut allocations = Allocations::new(gpu);
    let result = (|| {
        let input_f16 = allocations.zeroed(plan.input_bytes)?;
        let trellis_i16 = allocations.zeroed(plan.trellis_bytes)?;
        let output = allocations.zeroed(plan.output_bytes)?;
        let locks_i32 = allocations.zeroed(1024 * 1024 * size_of::<i32>())?;
        let scale_in_f16 = allocations.zeroed(plan.scale_in_bytes)?;
        let input_hadamard_f16 = allocations.zeroed(plan.input_bytes)?;
        let scale_out_f16 = allocations.zeroed(plan.scale_out_bytes)?;

        let before = vec![0x5au8; plan.output_bytes];
        gpu.copy_h2d(&before, output.ptr)?;
        kernel.launch(
            gpu,
            &plan,
            Glm53Exl3GemmBuffers {
                input_f16,
                trellis_i16,
                output,
                locks_i32,
                scale_in_f16,
                input_hadamard_f16,
                scale_out_f16,
            },
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut after = vec![0xffu8; plan.output_bytes];
        gpu.copy_d2h(output.ptr, &mut after)?;
        if after == before {
            bail!("GLM EXL3 cooperative kernel left the output sentinel untouched");
        }
        if let Some((offset, byte)) = after
            .iter()
            .copied()
            .enumerate()
            .find(|(_, byte)| *byte != 0)
        {
            bail!("GLM EXL3 zero-scale output is nonzero at byte {offset}: {byte:#04x}");
        }
        println!(
            "RESULT: PASS module={} shape={} grid={} block={} output_bytes={}",
            plan.module(),
            plan.shape_index,
            plan.grid,
            plan.block,
            plan.output_bytes
        );
        Ok(())
    })();
    let cleanup = allocations.free_all();
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(cleanup)) => Err(cleanup),
        (Err(error), Err(cleanup)) => {
            Err(error).context(format!("GLM EXL3 cleanup also failed: {cleanup:#}"))
        }
    }
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    run(&backend)
}
