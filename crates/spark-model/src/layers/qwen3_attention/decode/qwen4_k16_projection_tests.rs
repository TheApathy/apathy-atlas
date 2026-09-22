// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    QWEN4_K16_EXACT_RECEIPT, Qwen4ExactQkvRowsRoute, Qwen4QkvProjectionExtents,
    parse_exact_k16_env_value, qwen4_exact_qkv_receipt, qwen4_exact_qkv_rows_route,
    qwen4_k16_target_attention_geometry, qwen4_qkv_projection_extents,
};

#[test]
fn k16_exact_qkv_admission_is_default_off_narrow_and_strict() {
    for rows in 0..=32 {
        assert_eq!(qwen4_exact_qkv_rows_route(rows, false, false), None);
    }

    assert_eq!(
        qwen4_exact_qkv_rows_route(5, true, false),
        Some(Qwen4ExactQkvRowsRoute::LegacyK5K9)
    );
    assert_eq!(
        qwen4_exact_qkv_rows_route(9, true, false),
        Some(Qwen4ExactQkvRowsRoute::LegacyK5K9)
    );
    assert_eq!(qwen4_exact_qkv_rows_route(16, true, false), None);

    assert_eq!(qwen4_exact_qkv_rows_route(5, false, true), None);
    assert_eq!(qwen4_exact_qkv_rows_route(9, false, true), None);
    assert_eq!(
        qwen4_exact_qkv_rows_route(16, false, true),
        Some(Qwen4ExactQkvRowsRoute::K16)
    );
    for rows in [0, 1, 4, 8, 15, 17, 32, usize::MAX] {
        assert_eq!(qwen4_exact_qkv_rows_route(rows, false, true), None);
    }

    assert!(!parse_exact_k16_env_value(None).unwrap());
    assert!(!parse_exact_k16_env_value(Some("0")).unwrap());
    assert!(parse_exact_k16_env_value(Some("1")).unwrap());
    for invalid in ["", "true", "false", " 1", "1 ", "2", "-1"] {
        let error = parse_exact_k16_env_value(Some(invalid)).unwrap_err();
        assert!(error.to_string().contains("must be exactly 0 or 1"));
    }

    assert_eq!(
        qwen4_exact_qkv_receipt(Qwen4ExactQkvRowsRoute::LegacyK5K9),
        None
    );
    assert_eq!(
        qwen4_exact_qkv_receipt(Qwen4ExactQkvRowsRoute::K16),
        Some(QWEN4_K16_EXACT_RECEIPT)
    );
}

#[test]
fn k16_target_geometry_and_projection_extents_are_exact() {
    assert_eq!(
        QWEN4_K16_EXACT_RECEIPT,
        "ENGAGED ATLAS_QWEN4_K16_EXACT: M16 qg+dual_kv+qk_norm"
    );
    assert!(qwen4_k16_target_attention_geometry(true, 2560, 24, 2, 256));
    assert!(!qwen4_k16_target_attention_geometry(true, 2560, 20, 4, 128));
    for geometry in [
        (false, 2560, 24, 2, 256),
        (true, 2559, 24, 2, 256),
        (true, 2560, 23, 2, 256),
        (true, 2560, 24, 4, 256),
        (true, 2560, 24, 2, 128),
    ] {
        assert!(!qwen4_k16_target_attention_geometry(
            geometry.0, geometry.1, geometry.2, geometry.3, geometry.4
        ));
    }

    assert_eq!(
        qwen4_qkv_projection_extents(24, 2, 256, 16),
        Some(Qwen4QkvProjectionExtents {
            q_dim: 6144,
            q_proj_dim: 12288,
            kv_dim: 512,
            q_proj_bytes: 24576,
            kv_bytes: 1024,
            row_stride_bf16: 13312,
            row_bytes: 26624,
            total_bytes: 425984,
        })
    );
    assert_eq!(qwen4_qkv_projection_extents(u32::MAX, 2, 256, 16), None);
    assert_eq!(qwen4_qkv_projection_extents(24, 2, 256, usize::MAX), None);
}

#[test]
fn k16_projection_source_preserves_atomic_qg_dual_kv_and_rowwise_norm_order() {
    let source = include_str!("qwen4_k5_projection.rs");
    let qg = source.find("ops::w4a16_gemv_qg_exact(").unwrap();
    let k_base = source
        .find("let k_base = chunk_output.offset(extents.q_proj_bytes);")
        .unwrap();
    let v_base = source
        .find("let v_base = k_base.offset(extents.kv_bytes);")
        .unwrap();
    let dual = source.find("ops::w4a16_gemv_dual_kv_exact(").unwrap();
    let norms = source.find("for row in 0..rows {").unwrap();
    let receipt = source
        .find("if let Some(receipt) = qwen4_exact_qkv_receipt(route)")
        .unwrap();
    let success = source.find("Ok(Some((qkv, extents.row_bytes)))").unwrap();

    assert!(
        qg < k_base
            && k_base < v_base
            && v_base < dual
            && dual < norms
            && norms < receipt
            && receipt < success
    );
    assert!(source.contains("self.w4a16_exact_qkv_kernels.qg_for_rows(projection_rows)"));
    assert!(source.contains(".dual_kv_for_rows(projection_rows)"));
    assert!(source.contains("ctx.config.is_qwen4_exp()"));
    assert!(source.contains("qwen4_k16_target_attention_geometry("));
    assert!(source.contains("let k_out = q_out.offset(extents.q_proj_bytes);"));
    assert!(source.contains("self.attn.q_norm"));
    assert!(source.contains("self.attn.k_norm"));
    assert!(source.contains("static K16_EXACT_ENGAGED: std::sync::Once"));
    assert!(source.contains(QWEN4_K16_EXACT_RECEIPT));
}
