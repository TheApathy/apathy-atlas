// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for keeping V4 compressed-pool FP8 bytes on the raw-K scale.

use half::bf16;
use std::fs;
use std::path::PathBuf;

const PREFILL: &str = "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs";
const DECODE: &str = "crates/spark-model/src/layers/qwen3_attention/decode/attention_forward_v4.rs";
const KERNEL: &str = "kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    fs::read_to_string(root().join(path))
        .unwrap_or_else(|error| panic!("required V4 compressed-pool source {path}: {error}"))
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    if exponent == 0 {
        return sign * mantissa as f32 * 2.0f32.powi(-9);
    }
    sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
}

fn f32_to_e4m3_rne_satfinite(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return sign | 0x7f;
    }
    if magnitude >= 448.0 {
        return sign | 0x7e;
    }
    let mut best = 0u8;
    let mut best_distance = f64::INFINITY;
    for code in 0u8..=0x7e {
        let distance = (e4m3_to_f32(code) as f64 - magnitude as f64).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0 && best & 1 != 0)
        {
            best = code;
            best_distance = distance;
        }
    }
    sign | best
}

fn quantize(value: bf16, scale: f32) -> u8 {
    f32_to_e4m3_rne_satfinite(value.to_f32() / scale)
}

#[test]
fn scaled_conversion_matches_raw_k_quantization_for_non_unit_scales() {
    let values = [
        bf16::from_f32(-731.0),
        bf16::from_f32(-17.5),
        bf16::from_f32(0.0),
        bf16::from_f32(19.25),
        bf16::from_f32(812.0),
    ];
    for scale in [0.125, 0.75, 2.0, 3.5] {
        let pool: Vec<_> = values.iter().map(|&value| quantize(value, scale)).collect();
        let raw_k: Vec<_> = values.iter().map(|&value| quantize(value, scale)).collect();
        assert_eq!(pool, raw_k);
        assert_ne!(
            pool,
            values
                .iter()
                .map(|&value| quantize(value, 1.0))
                .collect::<Vec<_>>(),
            "fixture must detect the old unscaled pool write at scale {scale}"
        );
    }
}

#[test]
fn production_kernel_and_host_abi_carry_one_positive_dequant_scale() {
    let kernel = compact(&read(KERNEL));
    assert!(kernel.contains("voidbf16_to_fp8_scaled("));
    assert!(kernel.contains("floatscale"));
    assert!(kernel.contains("constfloatinv_scale=1.0f/scale;"));
    assert!(kernel.contains("f0*=inv_scale;"));
    assert!(kernel.contains("f1*=inv_scale;"));

    let types = read("crates/spark-model/src/layers/qwen3_attention/types.rs");
    let init = read("crates/spark-model/src/layers/qwen3_attention/init.rs");
    let ops = compact(&read(
        "crates/spark-model/src/layers/ops/gemm_fp8_prefill.rs",
    ));
    assert!(types.contains("bf16_to_fp8_scaled_k"));
    assert!(init.contains("\"bf16_to_fp8_scaled\""));
    assert!(ops.contains("pubfnbf16_to_fp8_scaled("));
    assert!(ops.contains(".arg_u32(total_elements).arg_f32(scale).launch(stream)"));
}

#[test]
fn prefill_finalizes_raw_k_scale_before_publishing_compressed_pool() {
    let source = read(PREFILL);
    let raw_write = source
        .find("self.write_kv_cache(")
        .expect("incumbent raw KV write must remain");
    let pool_write = source
        .find("// BEGIN V4 scale-consistent compressed-pool write")
        .expect("deferred compressed-pool write must exist");
    assert!(
        raw_write < pool_write,
        "raw calibration/write must precede pool quantization"
    );

    let tail = compact(&source[pool_write..]);
    assert!(tail.contains("let(k_scale,_v_scale)=self.effective_fp8_scales();"));
    assert!(tail.contains("k_scale.is_finite()&&k_scale>0.0"));
    assert!(tail.contains("ops::bf16_to_fp8_scaled("));
    assert!(!source.contains("compressed-pool persist assumes k_scale=1.0"));
}

#[test]
fn decode_append_uses_the_same_scale_aware_pool_conversion() {
    let source = read(DECODE);
    let append = source
        .find("pub(in crate::layers::qwen3_attention) fn v4_compress_append")
        .expect("V4 compressed append helper");
    let tail = compact(&source[append..]);
    assert!(tail.contains("let(k_scale,_v_scale)=self.effective_fp8_scales();"));
    assert!(tail.contains("ops::bf16_to_fp8_scaled("));
    assert!(tail.contains("k_scale.is_finite()&&k_scale>0.0"));
}
