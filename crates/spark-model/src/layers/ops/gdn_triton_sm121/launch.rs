// SPDX-License-Identifier: AGPL-3.0-only

use super::ffi;
use super::loader::LoadedFamily;
use super::manifest::{Abi, Grid, KERNELS, KernelSpec};
use super::types::{Buffers, Stream, WorkspaceLayout, preflight_buffers};
use anyhow::{Result, ensure};
use std::ffi::c_void;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {
    U64(u64),
    U32(u32),
    F32(f32),
}

impl ArgValue {
    fn abi(self) -> Abi {
        match self {
            Self::U64(_) => Abi::U64,
            Self::U32(_) => Abi::U32,
            Self::F32(_) => Abi::F32,
        }
    }

    fn as_mut_ptr(&mut self) -> *mut c_void {
        match self {
            Self::U64(value) => (value as *mut u64).cast(),
            Self::U32(value) => (value as *mut u32).cast(),
            Self::F32(value) => (value as *mut f32).cast(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KernelCall {
    pub role: &'static str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_shared: u32,
    args: Vec<ArgValue>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetadataContract {
    pub cu_seqlens_i32: [i32; 2],
    pub state_index_i32: [i32; 1],
    pub chunk_offsets_i64: [i64; 2],
    pub stride_init_state_i32: u32,
    pub scale_f32: f32,
}

pub struct PreparedLaunch {
    pub layout: WorkspaceLayout,
    pub metadata: MetadataContract,
    calls: Vec<KernelCall>,
    stream: Stream,
}

fn grid(spec: &KernelSpec, nt: u32) -> [u32; 3] {
    match spec.grid {
        Grid::Nt48 => [nt, 48, 1],
        Grid::Fixed4x48 => [4, 48, 1],
        Grid::Output => [2, nt, 48],
    }
}

fn validate_call(call: &KernelCall, spec: &KernelSpec, nt: u32) -> Result<()> {
    ensure!(call.role == spec.role, "kernel role/order drift");
    ensure!(call.grid == grid(spec, nt), "kernel grid drift");
    ensure!(call.block == [spec.block_x, 1, 1], "kernel block drift");
    ensure!(
        call.dynamic_shared == spec.dynamic_shared,
        "dynamic shared drift"
    );
    ensure!(call.args.len() == spec.abi.len(), "driver ABI arity drift");
    ensure!(
        call.args
            .iter()
            .copied()
            .map(ArgValue::abi)
            .eq(spec.abi.iter().copied()),
        "driver ABI type drift"
    );
    ensure!(
        call.args.ends_with(&[ArgValue::U64(0), ArgValue::U64(0)]),
        "two trailing null u64 scratch arguments required"
    );
    for (index, (value, kind)) in call.args.iter().zip(spec.abi).enumerate() {
        if *kind == Abi::U64 && index + 2 < call.args.len() {
            ensure!(*value != ArgValue::U64(0), "null semantic device pointer");
        }
    }
    Ok(())
}

pub fn prepare_launch(m: u32, buffers: Buffers, stream: Stream) -> Result<PreparedLaunch> {
    let layout = preflight_buffers(m, buffers, stream)?;
    let pointer = |name| layout.pointer(buffers.workspace, name).map(ArgValue::U64);
    let zero_scratch = || [ArgValue::U64(0), ArgValue::U64(0)];
    let mut calls = vec![
        KernelCall {
            role: "cumsum",
            grid: grid(&KERNELS[0], layout.nt),
            block: [256, 1, 1],
            dynamic_shared: 8,
            args: vec![
                pointer("log_gate_f32")?,
                pointer("g_cumsum_f32")?,
                pointer("cu_seqlens_i32")?,
                pointer("chunk_indices_i32")?,
                ArgValue::U32(m),
                zero_scratch()[0],
                zero_scratch()[1],
            ],
        },
        KernelCall {
            role: "kkt_bc16_solve",
            grid: grid(&KERNELS[1], layout.nt),
            block: [32, 1, 1],
            dynamic_shared: 7_168,
            args: vec![
                pointer("k_bf16")?,
                pointer("g_cumsum_f32")?,
                pointer("beta_f32")?,
                pointer("A_bf16")?,
                pointer("cu_seqlens_i32")?,
                pointer("chunk_indices_i32")?,
                ArgValue::U32(m),
                zero_scratch()[0],
                zero_scratch()[1],
            ],
        },
        KernelCall {
            role: "recompute_w_u",
            grid: grid(&KERNELS[2], layout.nt),
            block: [128, 1, 1],
            dynamic_shared: 28_672,
            args: vec![
                pointer("k_bf16")?,
                pointer("v_bf16")?,
                pointer("beta_f32")?,
                pointer("w_bf16")?,
                pointer("u_bf16")?,
                pointer("A_bf16")?,
                pointer("g_cumsum_f32")?,
                pointer("cu_seqlens_i32")?,
                pointer("chunk_indices_i32")?,
                ArgValue::U32(m),
                zero_scratch()[0],
                zero_scratch()[1],
            ],
        },
        KernelCall {
            role: "chunk_recurrence_state",
            grid: grid(&KERNELS[3], layout.nt),
            block: [128, 1, 1],
            dynamic_shared: 41_220,
            args: vec![
                pointer("k_bf16")?,
                pointer("u_bf16")?,
                pointer("w_bf16")?,
                pointer("v_new_bf16")?,
                pointer("g_cumsum_f32")?,
                pointer("h_bf16")?,
                pointer("state_hvk_f32")?,
                pointer("state_index_i32")?,
                ArgValue::U32(786_432),
                pointer("cu_seqlens_i32")?,
                pointer("chunk_offsets_i64")?,
                ArgValue::U32(m),
                zero_scratch()[0],
                zero_scratch()[1],
            ],
        },
        KernelCall {
            role: "output",
            grid: grid(&KERNELS[4], layout.nt),
            block: [128, 1, 1],
            dynamic_shared: 18_432,
            args: vec![
                pointer("q_bf16")?,
                pointer("k_bf16")?,
                pointer("v_new_bf16")?,
                pointer("h_bf16")?,
                pointer("g_cumsum_f32")?,
                ArgValue::U64(buffers.output_bf16.address),
                pointer("cu_seqlens_i32")?,
                pointer("chunk_indices_i32")?,
                ArgValue::F32(f32::from_bits(0x3db5_04f3)),
                ArgValue::U32(m),
                zero_scratch()[0],
                zero_scratch()[1],
            ],
        },
    ];
    ensure!(
        calls.len() == KERNELS.len(),
        "exact five-kernel plan required"
    );
    for (call, spec) in calls.iter().zip(&KERNELS) {
        validate_call(call, spec, layout.nt)?;
    }
    let metadata = MetadataContract {
        cu_seqlens_i32: [0, m as i32],
        state_index_i32: [0],
        chunk_offsets_i64: [0, i64::from(layout.nt)],
        stride_init_state_i32: 786_432,
        scale_f32: f32::from_bits(0x3db5_04f3),
    };
    Ok(PreparedLaunch {
        layout,
        metadata,
        calls: std::mem::take(&mut calls),
        stream,
    })
}

impl PreparedLaunch {
    /// The caller must enqueue the sealed adapters/A+output clears before this
    /// call and the state-out adapter/completion marker after it on `stream`.
    pub unsafe fn launch_five(&mut self, family: &LoadedFamily) -> Result<()> {
        unsafe { family.ensure_context() }?;
        for call in &mut self.calls {
            let kernel = family.kernel(call.role)?;
            validate_call(call, kernel.spec, self.layout.nt)?;
            let mut pointers = call
                .args
                .iter_mut()
                .map(ArgValue::as_mut_ptr)
                .collect::<Vec<_>>();
            unsafe {
                ffi::launch(
                    kernel.function,
                    call.grid,
                    call.block,
                    call.dynamic_shared,
                    self.stream.raw(),
                    &mut pointers,
                )
            }?;
        }
        Ok(())
    }
}
