// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the serving checkpoint's fixed-K2 EXL3 decode arm.

const KERNEL: &str = include_str!("../../../kernels/gb10/common/exl3_gemv.cu");
const K2_WRAPPER: &str = include_str!("../../../kernels/gb10/common/exl3_gemv_k2.cu");
const DISPATCH: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const TRANSPOSE: &str = include_str!("../src/layers/moe/helpers_a.rs");

fn lane_geom(lane: u32, bits: u32) -> (u32, u32, u32) {
    let b1 = (lane * 8 + 257) * bits;
    let b0 = b1 - 16;
    let b2 = b1 + 7 * bits;
    let i0 = b0 >> 5;
    let i2 = (b2 - 1) >> 5;
    (i0 % (8 * bits), i2 % (8 * bits), (i2 + 1) * 32 - b2)
}

#[test]
fn fixed_k2_wrapper_reuses_the_single_kernel_source() {
    assert!(K2_WRAPPER.contains("#define EXL3_FIXED_BITS 2"));
    assert!(K2_WRAPPER.contains("#include \"exl3_gemv.cu\""));
    assert!(!K2_WRAPPER.contains("exl3_gemv_m1_body("));
}

#[test]
fn every_decode_entry_rejects_a_bitrate_mismatch_and_passes_a_constant() {
    assert!(KERNEL.contains("#ifndef EXL3_FIXED_BITS"));
    assert!(KERNEL.contains("#define EXL3_REQUIRE_BITS(bits_)"));
    assert!(KERNEL.contains("#define EXL3_BIT_WIDTH(bits_) EXL3_FIXED_BITS"));
    assert!(KERNEL.matches("EXL3_REQUIRE_BITS(bits);").count() >= 6);
    assert!(KERNEL.matches("EXL3_BIT_WIDTH(bits)").count() >= 6);
}

#[test]
fn fixed_k2_is_selected_from_weight_metadata_with_an_explicit_opt_out() {
    assert!(DISPATCH.contains("let decode_module ="));
    assert!(DISPATCH.contains("gate.bits == 2 && std::env::var"));
    assert!(DISPATCH.contains("ATLAS_EXL3_FIXED_K2"));
    assert!(DISPATCH.contains("mrow_handles(gpu, decode_module"));
    assert!(DISPATCH.contains(".kernel(decode_module, \"exl3_gemv_m1_fused_gate_up\")"));
    assert!(DISPATCH.contains("gpu.kernel(decode_module, \"exl3_gemv_m1_fused_down\")"));
}

#[test]
fn exl3_trellis_never_enters_native_moe_transpose() {
    let guard = "self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Exl3Trellis";
    assert_eq!(TRANSPOSE.matches(guard).count(), 2);
    let guarded_return = format!("if {guard} {{\n            return Ok(());\n        }}");
    assert_eq!(TRANSPOSE.matches(&guarded_return).count(), 2);
    assert!(TRANSPOSE.contains("fn transpose_for_prefill_impl("));
    assert!(TRANSPOSE.contains("fn transpose_for_prefill_unified_inner("));
}

#[test]
fn k2_lane_geometry_has_the_expected_two_phase_pattern() {
    assert_eq!(lane_geom(0, 2), (15, 0, 16));
    for lane in 1..32 {
        let pair = (lane - 1) / 2;
        let expected = if lane % 2 == 1 {
            (pair, pair, 0)
        } else {
            (pair, pair + 1, 16)
        };
        assert_eq!(lane_geom(lane, 2), expected, "lane {lane}");
    }
}

#[test]
fn every_mrow_rung_fits_the_sm121_static_shared_memory_limit() {
    const STATIC_LIMIT: usize = 48 * 1024;
    const STAGE_BYTES: usize = 2 * 8 * 8 * 48 * 2;
    const ELECT_BYTES: usize = 4;
    assert!(KERNEL.contains("#define EXL3_M16_XCHUNKS 4"));
    let shared_bytes = |mrow: usize, xchunks: usize| {
        let x_bytes = mrow * xchunks * 64 * 4;
        let y_bytes = mrow * 128 * 4;
        let routing_bytes = mrow * 32 * 4 + mrow * 4 + 4;
        x_bytes + STAGE_BYTES + y_bytes + ELECT_BYTES + routing_bytes
    };
    assert_eq!(shared_bytes(16, 8), 0xd848);
    for (mrow, xchunks) in [(1, 8), (2, 8), (4, 8), (6, 8), (8, 8), (16, 4)] {
        let total = shared_bytes(mrow, xchunks);
        assert!(
            total <= STATIC_LIMIT,
            "MROW={mrow} needs {total} static shared-memory bytes"
        );
    }
}
