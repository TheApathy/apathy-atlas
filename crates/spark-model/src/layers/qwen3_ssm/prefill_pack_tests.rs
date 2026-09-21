// SPDX-License-Identifier: AGPL-3.0-only

use super::trait_prefill_phase1::{
    fused_prefill_conv_l2_geometry, packed_gdn_kernel_ready, use_fused_prefill_conv_l2,
    use_packed_gdn_inputs,
};

#[test]
fn pack_route_requires_aligned_nonempty_qwen_layout() {
    assert_eq!(
        use_packed_gdn_inputs(true, true, 8192, 12288, 8192, 4096, 64),
        Ok(true)
    );
    assert_eq!(
        use_packed_gdn_inputs(false, true, 8192, 12288, 8192, 4096, 64),
        Ok(false)
    );
    assert_eq!(
        use_packed_gdn_inputs(true, true, 0, 12288, 8192, 4096, 64),
        Ok(false)
    );
    assert_eq!(
        use_packed_gdn_inputs(true, true, 65_536, 12288, 8192, 4096, 64),
        Ok(false)
    );
    assert_eq!(
        use_packed_gdn_inputs(true, true, 8192, 12289, 8192, 4096, 64),
        Ok(false)
    );
    assert_eq!(
        use_packed_gdn_inputs(true, true, 8192, 12288, 8191, 4096, 64),
        Ok(false)
    );
    assert_eq!(
        use_packed_gdn_inputs(true, true, 8192, 12288, 8192, 4096, 63),
        Ok(false)
    );
}

#[test]
fn requested_eligible_pack_requires_the_kernel_symbol() {
    assert!(use_packed_gdn_inputs(true, false, 8192, 12288, 8192, 4096, 64).is_err());
    assert_eq!(
        use_packed_gdn_inputs(false, false, 8192, 12288, 8192, 4096, 64),
        Ok(false)
    );
}

#[test]
fn exact_conv_l2_route_is_qwen_geometry_only_and_fails_closed() {
    assert_eq!(
        use_fused_prefill_conv_l2(true, true, true, 8192, 8192, 4, 4096, 128, 4096),
        Ok(true)
    );
    assert_eq!(
        use_fused_prefill_conv_l2(false, true, true, 8192, 8192, 4, 4096, 128, 4096),
        Ok(false)
    );
    assert!(use_fused_prefill_conv_l2(true, false, true, 8192, 8192, 4, 4096, 128, 4096).is_err());
    assert!(use_fused_prefill_conv_l2(true, true, false, 8192, 8192, 4, 4096, 128, 4096).is_err());
    assert_eq!(
        use_fused_prefill_conv_l2(true, true, false, 8192, 8192, 4, 4096, 64, 4096),
        Ok(false)
    );
    assert_eq!(
        use_fused_prefill_conv_l2(true, true, false, 8192, 8192, 4, 4097, 128, 4096),
        Ok(false)
    );
    assert_eq!(
        use_fused_prefill_conv_l2(true, true, false, 8192, 8192, 3, 4096, 128, 4096),
        Ok(false)
    );
}

#[test]
fn ineligible_fused_symbol_cannot_mask_a_missing_pack_symbol() {
    let fused_geometry = fused_prefill_conv_l2_geometry(8192, 8192, 4, 4096, 64, 4096);
    assert!(!fused_geometry, "head_dim=64 is not the exact fused route");
    assert!(!packed_gdn_kernel_ready(false, true, fused_geometry, true));
    assert!(use_packed_gdn_inputs(true, false, 8192, 12288, 8192, 4096, 64).is_err());

    let fused_geometry = fused_prefill_conv_l2_geometry(8192, 8192, 4, 4096, 128, 4096);
    assert!(fused_geometry);
    assert!(packed_gdn_kernel_ready(false, true, fused_geometry, true));
}

#[test]
fn cuda_source_keeps_conv_math_and_exact_z_assignment() {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/common/causal_conv1d.cu"
    ));
    let body = source
        .split_once("extern \"C\" __global__ void causal_conv1d_update_prefill_zcopy")
        .expect("prefill conv/Z kernel is missing")
        .1;
    assert!(body.contains("float new_val = (float)input"));
    assert!(body.contains("float acc = b_val + s[0]*w_reg[0]"));
    assert!(body.contains("__float2bfloat16(acc * sigmoid_acc)"));
    assert!(body.contains("z_output[(unsigned long long)t * z_dim + ch]"));
    assert!(body.contains("z_input[(unsigned long long)t * input_stride + ch]"));
}

#[test]
fn fused_conv_l2_source_retains_bf16_boundary_and_pairwise_tree() {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/common/causal_conv1d.cu"
    ));
    let body = source
        .split_once("extern \"C\" __global__ void causal_conv1d_update_prefill_l2norm_zcopy")
        .expect("exact prefill conv/L2/Z kernel is missing")
        .1
        .split_once("// ============================================================\n// CHUNK2")
        .expect("exact prefill conv/L2/Z kernel end marker is missing")
        .0;
    assert!(body.contains("head_dim != 128"));
    assert!(body.contains("d_conv != 4"));
    assert!(body.contains("__nv_bfloat16 silu_bf16 = __float2bfloat16(acc * sigmoid_acc)"));
    assert!(body.contains("sum_sq += v0 * v0 + v1 * v1"));
    assert!(body.contains("sum_sq = conv_l2_warp_reduce_sum(sum_sq)"));
    assert!(body.contains("value = conv_l2_warp_reduce_sum(value)"));
    assert!(body.contains("conv_l2_pack_bf16x2(v0 * inv_norm, v1 * inv_norm)"));
    assert!(body.contains("z_input[(unsigned long long)token * input_stride + ch]"));
}
