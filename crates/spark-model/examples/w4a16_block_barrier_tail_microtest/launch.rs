// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::cases::{Abi, Case, K};

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernel: KernelHandle,
    case: Case,
    n: usize,
    input: DevicePtr,
    packed0: DevicePtr,
    scale0: DevicePtr,
    output0: DevicePtr,
    packed1: DevicePtr,
    scale1: DevicePtr,
    output1: DevicePtr,
) -> Result<()> {
    let base = || {
        KernelLaunch::new(gpu, kernel)
            .grid([n.div_ceil(case.group) as u32, 1, case.grid_z])
            .block([case.block, 1, 1])
    };
    match case.abi {
        Abi::Single => base()
            .arg_ptr(input)
            .arg_ptr(packed0)
            .arg_ptr(scale0)
            .arg_f32(1.0)
            .arg_ptr(output0)
            .arg_u32(n as u32)
            .arg_u32(K as u32)
            .launch(stream),
        Abi::Qg => base()
            .arg_ptr(input)
            .arg_ptr(packed0)
            .arg_ptr(scale0)
            .arg_f32(1.0)
            .arg_ptr(output0)
            .arg_u32(n as u32)
            .arg_u32(K as u32)
            .arg_u32(2) // num_heads
            .arg_u32(1) // head_dim
            .launch(stream),
        Abi::Qkvz => base()
            .arg_ptr(input)
            .arg_ptr(packed0)
            .arg_ptr(scale0)
            .arg_f32(1.0)
            .arg_ptr(output0)
            .arg_u32(n as u32)
            .arg_u32(K as u32)
            .arg_u32(1) // num_groups
            .arg_u32(1) // head_k_dim
            .arg_u32(1) // vheads_per_group
            .arg_u32(1) // head_v_dim
            .launch(stream),
        Abi::Dual => base()
            .arg_ptr(input)
            .arg_ptr(packed0)
            .arg_ptr(scale0)
            .arg_f32(1.0)
            .arg_ptr(output0)
            .arg_ptr(packed1)
            .arg_ptr(scale1)
            .arg_f32(1.0)
            .arg_ptr(output1)
            .arg_u32(n as u32)
            .arg_u32(K as u32)
            .launch(stream),
        Abi::QgStrided => base()
            .arg_ptr(input)
            .arg_ptr(packed0)
            .arg_ptr(scale0)
            .arg_f32(1.0)
            .arg_ptr(output0)
            .arg_u32(n as u32)
            .arg_u32(K as u32)
            .arg_u32(2)
            .arg_u32(1)
            .arg_u32(case.stride() as u32)
            .launch(stream),
        Abi::DualStrided => base()
            .arg_ptr(input)
            .arg_ptr(packed0)
            .arg_ptr(scale0)
            .arg_f32(1.0)
            .arg_ptr(output0)
            .arg_ptr(packed1)
            .arg_ptr(scale1)
            .arg_f32(1.0)
            .arg_ptr(output1)
            .arg_u32(n as u32)
            .arg_u32(K as u32)
            .arg_u32(case.stride() as u32)
            .launch(stream),
    }
}
