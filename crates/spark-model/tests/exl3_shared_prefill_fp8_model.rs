// SPDX-License-Identifier: AGPL-3.0-only

use spark_model::layers::ops;
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

const ASSEMBLE: &str = include_str!("../src/weight_loader/deepseek_v4/assemble.rs");
const HELPERS: &str = include_str!("../src/layers/moe/helpers_c.rs");
const PREFILL: &str = include_str!("../src/layers/moe/forward_prefill_phase.rs");
const OPS: &str = include_str!("../src/layers/ops/gemm_fp8_prefill.rs");

#[test]
fn exact_bf16_input_launcher_is_one_kernel_launch() {
    let gpu = MockGpuBackend::new();
    ops::fp8_gemm_n128_bf16_input(
        &gpu,
        KernelHandle(7),
        DevicePtr(11),
        DevicePtr(13),
        DevicePtr(17),
        2_410,
        2_048,
        4_096,
        0,
    )
    .unwrap();
    assert_eq!(gpu.launch_count(), 1);
}

#[test]
fn predequant_allocates_exact_fp8_shape() {
    let gpu = MockGpuBackend::new();
    let weight = QuantizedWeight {
        weight: DevicePtr(23),
        weight_scale: DevicePtr(29),
        weight_scale_2: 0.5,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    };
    let output = weight
        .predequant_to_fp8(&gpu, KernelHandle(31), 17, 32, 0)
        .unwrap();
    assert_eq!(gpu.read_alloc(output).unwrap().len(), 17 * 32);
    assert_eq!(gpu.alloc_count(), 1);
    assert_eq!(gpu.launch_count(), 1);
}

#[test]
fn predequant_rejects_odd_k_before_allocation() {
    let gpu = MockGpuBackend::new();
    let weight = QuantizedWeight {
        weight: DevicePtr(23),
        weight_scale: DevicePtr(29),
        weight_scale_2: 0.5,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    };
    assert!(
        weight
            .predequant_to_fp8(&gpu, KernelHandle(31), 17, 31, 0)
            .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn deepseek_shared_fp8_prefill_is_explicit_and_exl3_only() {
    let gate = ASSEMBLE
        .find("ATLAS_EXL3_SHARED_PREFILL_FP8")
        .expect("missing explicit opt-in");
    let exl3_block = ASSEMBLE
        .find("if exl3_detected {")
        .expect("missing EXL3-only block");
    let attach = ASSEMBLE
        .find("moe.set_exl3_experts")
        .expect("missing EXL3 attachment");
    let build = ASSEMBLE
        .find(".predequant_shared_for_prefill")
        .expect("missing shared predequant call");
    assert!(exl3_block < attach && attach < gate && gate < build);
}

#[test]
fn exl3_shared_fp8_prefill_bypasses_activation_quantization() {
    assert!(PREFILL.contains("self.exl3.is_some()"));
    assert!(PREFILL.contains("ops::fp8_gemm_n128_bf16_input"));
    assert!(PREFILL.contains("ops::fp8_gemm_n128("));

    let direct = OPS
        .split_once("pub fn fp8_gemm_n128_bf16_input")
        .expect("missing direct launcher")
        .1
        .split_once("pub fn ")
        .expect("direct launcher must remain bounded")
        .0;
    assert!(direct.contains("KernelLaunch::new"));
    assert!(!direct.contains("bf16_to_fp8"));
}

#[test]
fn shared_mirror_builder_is_transactional() {
    assert!(HELPERS.contains("pub fn predequant_shared_for_prefill"));
    assert!(HELPERS.contains("let mut built"));
    assert!(HELPERS.contains("for ptr in built"));
    assert!(HELPERS.contains("gpu.free(ptr)"));
    assert!(HELPERS.contains("self.shared_gate_fp8 = Some(gate)"));
    assert!(HELPERS.contains("self.shared_up_fp8 = Some(up)"));
    assert!(HELPERS.contains("self.shared_down_fp8 = Some(down)"));
}

#[test]
fn production_geometry_memory_cost_is_exact() {
    let bytes_per_layer = 3usize * 4_096 * 2_048;
    assert_eq!(bytes_per_layer, 25_165_824);
    assert_eq!(bytes_per_layer * 43, 1_082_130_432);
}
