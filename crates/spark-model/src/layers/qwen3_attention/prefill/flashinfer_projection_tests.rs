// SPDX-License-Identifier: AGPL-3.0-only

#[cfg(all(feature = "cuda", target_os = "linux"))]
use super::flashinfer_projection::bind_activation_stream;
use super::{Qwen38FlashinferProjectionRoute, qwen38_flashinfer_projection_route};

fn route(requested: bool, m: u32, prepared: bool) -> Qwen38FlashinferProjectionRoute {
    // The predicate now takes `m_qualified` rather than `m`. With
    // ATLAS_FLASHINFER_PROJ_EXTRA_M unset (the default these tests assert),
    // only the frozen M=2079 entry qualifies.
    qwen38_flashinfer_projection_route(
        requested,
        true,
        true,
        true,
        true,
        m == 2_079,
        5_120,
        12_288,
        1_024,
        prepared,
    )
}

/// The extended projection range must stay OFF by default, so default behaviour
/// is bit-for-bit the frozen single-shape table.
#[test]
fn extended_projection_range_is_off_unless_requested() {
    use crate::weight_map::flashinfer_projection_admission::{
        Qwen38AttentionProjectionFamily, qwen38_proj_extra_m_enabled,
        select_qwen38_attention_projection_launch,
    };
    if qwen38_proj_extra_m_enabled() {
        return; // caller opted in; the negative assertion does not apply
    }
    for m in [128usize, 1_024, 2_016, 2_057, 4_096] {
        assert!(
            select_qwen38_attention_projection_launch(
                Qwen38AttentionProjectionFamily::MergedQkv,
                m
            )
            .is_err(),
            "M={m} must be unqualified while ATLAS_FLASHINFER_PROJ_EXTRA_M is unset"
        );
    }
    // And M8192 must stay rejected even WITH the flag: its rejection is on
    // documented output drift, not on missing qualification.
    assert!(
        select_qwen38_attention_projection_launch(
            Qwen38AttentionProjectionFamily::MergedQkv,
            8_192
        )
        .is_err()
    );
}

#[test]
fn route_is_default_off_exact_shape_only_and_fail_closed() {
    assert_eq!(
        route(false, 2_079, false),
        Qwen38FlashinferProjectionRoute::Disabled
    );
    assert_eq!(
        route(true, 2_079, true),
        Qwen38FlashinferProjectionRoute::Complete
    );
    assert_eq!(
        route(true, 8_192, true),
        Qwen38FlashinferProjectionRoute::Ineligible
    );
    assert_eq!(
        route(true, 2_079, false),
        Qwen38FlashinferProjectionRoute::Missing
    );
    for m in [0, 1, 2_048, 2_080, 8_191, 8_192, 8_193] {
        assert_eq!(
            route(true, m, false),
            Qwen38FlashinferProjectionRoute::Ineligible
        );
    }
    for candidate in [
        qwen38_flashinfer_projection_route(
            true, false, true, true, true, true, 5_120, 12_288, 1_024, true,
        ),
        qwen38_flashinfer_projection_route(
            true, true, false, true, true, true, 5_120, 12_288, 1_024, true,
        ),
        qwen38_flashinfer_projection_route(
            true, true, true, false, true, true, 5_120, 12_288, 1_024, true,
        ),
        qwen38_flashinfer_projection_route(
            true, true, true, true, false, true, 5_120, 12_288, 1_024, true,
        ),
        qwen38_flashinfer_projection_route(
            true, true, true, true, true, true, 5_119, 12_288, 1_024, true,
        ),
        qwen38_flashinfer_projection_route(
            true, true, true, true, true, true, 5_120, 12_287, 1_024, true,
        ),
        qwen38_flashinfer_projection_route(
            true, true, true, true, true, true, 5_120, 12_288, 1_023, true,
        ),
    ] {
        assert_eq!(candidate, Qwen38FlashinferProjectionRoute::Ineligible);
    }
}

#[test]
fn production_route_rejects_the_failed_m8192_shape() {
    let route = include_str!("mod.rs");
    let implementation = include_str!("flashinfer_projection.rs");
    // The predicate now takes a precomputed `m_qualified`, so the old literal
    // is gone; assert the INVARIANT it protected instead — M8192 must remain
    // unselectable, and the admission ceiling must not exceed the scratch bound.
    assert!(route.contains("|| !m_qualified"));
    let admission = include_str!("../../../weight_map/flashinfer_projection_admission.rs");
    assert!(admission.contains("const QWEN38_PROJ_EXTRA_M_MAX: usize = 2_079;"));
    assert!(implementation.contains("const MAX_M: usize = 2_079;"));
    assert!(implementation.contains("const MAX_M_PADDED: usize = 2_176;"));
    assert!(implementation.contains("m.div_ceil(128) * 128"));
    assert!(!implementation.contains("for m in [2_079, 8_192]"));
    assert!(implementation.contains("M8192 is intentionally ineligible"));
}

#[test]
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn activation_scratch_binds_first_runtime_stream_and_rejects_a_different_one() {
    let mut admitted = None;
    bind_activation_stream(&mut admitted, 0xb).unwrap();
    assert_eq!(admitted, Some(0xb));
    bind_activation_stream(&mut admitted, 0xb).unwrap();
    assert_eq!(admitted, Some(0xb));
    let error = bind_activation_stream(&mut admitted, 0xc).unwrap_err();
    assert!(error.to_string().contains("bound to CUDA stream 0xb"));
    assert_eq!(admitted, Some(0xb));
}

#[test]
fn construction_stream_is_not_retained_and_runtime_stream_is_guarded_before_effects() {
    let source = include_str!("flashinfer_projection.rs");
    let struct_start = source
        .find("struct FlashinferAttentionProjectionPrefill {")
        .unwrap();
    let struct_end = source[struct_start..]
        .find("\n}")
        .map(|offset| struct_start + offset)
        .unwrap();
    let route_struct = &source[struct_start..struct_end];
    assert!(route_struct.contains("activation_use_stream: Mutex<Option<u64>>"));
    assert!(!route_struct.contains("stream: u64"));

    let prepare_start = source
        .find("pub fn prepare_flashinfer_projection_prefill(")
        .unwrap();
    let qgkv_start = source
        .find("pub(super) fn try_flashinfer_projection_qgkv(")
        .unwrap();
    let prepare = &source[prepare_start..qgkv_start];
    assert!(prepare.contains("let stream = gpu.default_stream();"));
    assert!(prepare.contains("activation_use_stream: Mutex::new(None)"));

    let output_start = source
        .find("pub(super) fn try_flashinfer_projection_output(")
        .unwrap();
    let qgkv = &source[qgkv_start..output_start];
    let qgkv_prepare = qgkv
        .rfind("prepare_borrowed_zero_workspace(ctx.gpu, output_shape, stream)")
        .unwrap();
    let qgkv_guard = qgkv.find("route.lock_activation_use(stream)").unwrap();
    let qgkv_effect = qgkv
        .find("ops::quantize_bf16_to_nvfp4_atlas_128x4(")
        .unwrap();
    assert!(qgkv_prepare < qgkv_guard && qgkv_guard < qgkv_effect);
    assert!(!qgkv.contains("stream == route.stream"));

    let output_end = source[output_start..]
        .find("\n}\n\nfn log_success")
        .map(|offset| output_start + offset)
        .unwrap();
    let output = &source[output_start..output_end];
    let output_prepare = output
        .find("prepare_borrowed_zero_workspace(ctx.gpu, shape, stream)")
        .unwrap();
    let output_guard = output.find("route.lock_activation_use(stream)").unwrap();
    let output_effect = output
        .find("ops::quantize_bf16_to_nvfp4_atlas_128x4(")
        .unwrap();
    assert!(output_prepare < output_guard && output_guard < output_effect);
    assert!(!output.contains("stream == route.stream"));
}

#[test]
fn merged_row_layout_splits_exact_qg_k_v_boundaries() {
    const QG: usize = 12_288;
    const KV: usize = 1_024;
    const TOTAL: usize = QG + 2 * KV;
    let merged: Vec<u16> = (0..2 * TOTAL).map(|index| index as u16).collect();
    let mut qg = vec![0_u16; 2 * QG];
    let mut key = vec![0_u16; 2 * KV];
    let mut value = vec![0_u16; 2 * KV];
    for row in 0..2 {
        let source = &merged[row * TOTAL..(row + 1) * TOTAL];
        qg[row * QG..(row + 1) * QG].copy_from_slice(&source[..QG]);
        key[row * KV..(row + 1) * KV].copy_from_slice(&source[QG..QG + KV]);
        value[row * KV..(row + 1) * KV].copy_from_slice(&source[QG + KV..]);
    }
    for row in 0..2 {
        assert_eq!(
            &qg[row * QG..(row + 1) * QG],
            &merged[row * TOTAL..row * TOTAL + QG]
        );
        assert_eq!(
            &key[row * KV..(row + 1) * KV],
            &merged[row * TOTAL + QG..row * TOTAL + QG + KV]
        );
        assert_eq!(
            &value[row * KV..(row + 1) * KV],
            &merged[row * TOTAL + QG + KV..(row + 1) * TOTAL]
        );
    }
}

#[test]
fn split_kernel_and_qwen_manifest_freeze_vectorized_same_stream_abi() {
    let source =
        include_str!("../../../../../../kernels/gb10/common/flashinfer_projection_split.cu");
    assert!(source.contains("extern \"C\" __global__ void flashinfer_projection_split_qgkv"));
    assert!(source.contains("reinterpret_cast<const uint4 *>"));
    assert!(source.contains("qg[vector] = src[vector]"));
    assert!(source.contains("k[vector] = src[QG_VECTORS + vector]"));
    assert!(source.contains("v[vector] = src[QG_VECTORS + KV_VECTORS + vector]"));
    assert!(!source.contains("cudaMemcpy"));

    let manifest = include_str!("../../../../../../kernels/gb10/qwen3.8-27b/nvfp4/KERNEL.toml");
    assert!(manifest.contains("flashinfer_projection_split = \"flashinfer_projection_split\""));
}

#[test]
fn merged_qgkv_route_precedes_parent_projection_in_both_prefill_paths() {
    for source in [
        include_str!("cache_skip_qkv.rs"),
        include_str!("paged_qkv.rs"),
    ] {
        let route = source
            .find("self.try_flashinfer_projection_qgkv(")
            .expect("prefill path must attempt the admitted merged-QGKV route");
        let parent_selector = source
            .find("let dual_kv =")
            .expect("prefill path must retain the parent projection selector");
        let parent_q = source[parent_selector..]
            .find("Proj::Q")
            .or_else(|| source[parent_selector..].find("SkipProj::Q"))
            .expect("prefill path must retain the parent Q projection")
            + parent_selector;
        assert!(
            route < parent_selector,
            "explicit route must fail before parent selection"
        );
        assert!(
            route < parent_q,
            "complete route must replace parent Q/K/V projection"
        );
        assert!(
            source[route..parent_selector].contains("return Ok(())"),
            "complete merged-QGKV launch must not continue into the parent path"
        );
    }
}
