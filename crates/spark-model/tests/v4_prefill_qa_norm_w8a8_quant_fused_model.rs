// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the isolated V4 q_a RMSNorm + W8A8 quantizer.

use half::bf16;
use std::fs;
use std::path::{Path, PathBuf};

const TOKENS: u64 = 2_410;
const Q_LORA: usize = 1_024;
const RMS_BLOCK: usize = 1_024;
const QUANT_BLOCK: usize = 256;
const LAYERS: u64 = 43;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn experiment() -> String {
    fs::read_to_string(
        repo_root().join("kernels/gb10/experiments/v4_prefill_qa_norm_w8a8_quant_fused.cu"),
    )
    .expect("isolated V4 q_a norm/W8A8 experiment must exist")
}

fn sass_gate() -> String {
    fs::read_to_string(
        repo_root().join("scripts/check-v4-prefill-qa-norm-w8a8-quant-fused-sass.sh"),
    )
    .expect("isolated V4 q_a norm/W8A8 SASS gate must exist")
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn xor_sum(values: &mut [f32; 32]) {
    for offset in [16, 8, 4, 2, 1] {
        let before = *values;
        for lane in 0..32 {
            values[lane] = before[lane] + before[lane ^ offset];
        }
    }
}

fn incumbent_norm(input: &[bf16], weight: &[bf16], eps: f32) -> Vec<bf16> {
    let mut partials = [0.0f32; RMS_BLOCK];
    for (tid, partial) in partials.iter_mut().enumerate().take(Q_LORA / 2) {
        let x0 = input[2 * tid].to_f32();
        let x1 = input[2 * tid + 1].to_f32();
        *partial += x0 * x0 + x1 * x1;
    }
    let mut warp_sums = [0.0f32; 32];
    for (warp, chunk) in partials.chunks_exact(32).enumerate() {
        let mut lanes = [0.0f32; 32];
        lanes.copy_from_slice(chunk);
        xor_sum(&mut lanes);
        warp_sums[warp] = lanes[0];
    }
    xor_sum(&mut warp_sums);
    let rms = 1.0f32 / (warp_sums[0] / Q_LORA as f32 + eps).sqrt();
    input
        .iter()
        .zip(weight)
        .map(|(x, w)| bf16::from_f32(x.to_f32() * rms * w.to_f32()))
        .collect()
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
    let mut distance = f64::INFINITY;
    for code in 0u8..=0x7e {
        let candidate = (e4m3_to_f32(code) as f64 - magnitude as f64).abs();
        if candidate < distance || (candidate == distance && code & 1 == 0 && best & 1 != 0) {
            best = code;
            distance = candidate;
        }
    }
    sign | best
}

fn quantize_incumbent(normed: &[bf16]) -> (f32, Vec<u8>) {
    let mut thread_max = [0.0f32; QUANT_BLOCK];
    for tid in 0..QUANT_BLOCK {
        for index in (tid..Q_LORA).step_by(QUANT_BLOCK) {
            thread_max[tid] = thread_max[tid].max(normed[index].to_f32().abs());
        }
    }
    let mut warp_max = [0.0f32; 8];
    for (warp, lanes) in thread_max.chunks_exact(32).enumerate() {
        warp_max[warp] = lanes.iter().copied().fold(0.0f32, f32::max);
    }
    let scale = warp_max.into_iter().fold(0.0f32, f32::max).max(1.0e-8) / 448.0;
    let bytes = normed
        .iter()
        .map(|value| f32_to_e4m3_rne_satfinite(value.to_f32() / scale))
        .collect();
    (scale, bytes)
}

fn fused_cpu(input: &[bf16], weight: &[bf16], eps: f32) -> (f32, Vec<u8>) {
    let reference = incumbent_norm(input, weight, eps);
    let mut shared = vec![bf16::ZERO; Q_LORA];
    // Model the fused CTA's pair-owned writes independently from the
    // incumbent's linear host projection above.
    for tid in 0..Q_LORA / 2 {
        shared[2 * tid] = reference[2 * tid];
        shared[2 * tid + 1] = reference[2 * tid + 1];
    }
    quantize_incumbent(&shared)
}

fn assert_tree_excludes(path: &Path) {
    for entry in fs::read_dir(path).expect("production tree must exist") {
        let path = entry.expect("production entry").path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) == Some("experiments") {
                continue;
            }
            assert_tree_excludes(&path);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("rs" | "toml" | "cu" | "cuh")
        ) {
            let source = fs::read_to_string(&path).expect("production source must be UTF-8");
            assert!(
                !source.contains("v4_prefill_qa_norm_w8a8_quant_fused"),
                "{}",
                path.display()
            );
        }
    }
}

#[test]
fn exact_shape_pointer_and_launch_contract_fail_closed() {
    let source = experiment();
    let flat = compact(&source);
    for contract in [
        "#define V4_QA_DIM 1024u",
        "#define V4_QA_BLOCK 1024u",
        "#define V4_QA_QUANT_BLOCK 256u",
        "__launch_bounds__(V4_QA_BLOCK)",
        "v4_prefill_qa_norm_w8a8_quant_fused",
    ] {
        assert!(source.contains(contract), "missing contract: {contract}");
    }
    assert!(flat.contains("num_tokens==0||input==nullptr||weight==nullptr"));
    assert!(flat.contains("output_fp8==nullptr||row_scale==nullptr"));
    assert!(flat.contains("hidden_size!=V4_QA_DIM||!isfinite(eps)||eps<=0.0f"));
    assert!(flat.contains("blockDim.x!=V4_QA_BLOCK||blockDim.y!=1||blockDim.z!=1"));
    assert!(flat.contains("gridDim.x!=num_tokens||gridDim.y!=1||gridDim.z!=1"));
}

#[test]
fn pair_to_strided_shared_remap_preserves_both_incumbent_reductions() {
    let source = compact(&experiment());
    assert!(source.contains("constunsignedintpair0=tid"));
    assert!(source.contains("normed[2u*pair0]"));
    assert!(source.contains("normed[2u*pair0+1u]"));
    assert!(source.contains("__float2bfloat16(x0*rms*w0)"));
    assert!(
        !source.contains("1.0f+w0"),
        "V4 q_a uses HF-vanilla weights"
    );
    assert!(source.contains("constunsignedintq0=tid"));
    assert!(source.contains("constunsignedintq3=tid+3u*V4_QA_QUANT_BLOCK"));
    assert!(source.contains("__shfl_xor_sync(0xFFFFFFFFu,value,offset)"));
    assert!(source.contains("__shfl_down_sync(0xFFFFFFFFu,amax,offset)"));
    let retained = source
        .find("y0_bits=__bfloat16_as_ushort(normed[q0])")
        .expect("quantizer must retain its BF16 input across the max reduction");
    let scale = source
        .find("constfloatinverse_scale=1.0f/reduction[0]")
        .expect("scale reciprocal");
    let encode = source
        .rfind("__ushort_as_bfloat16(y0_bits)")
        .expect("retained BF16 encode input");
    assert!(retained < scale && scale < encode);
    assert_eq!(source.matches("normed[q0]").count(), 1);

    let input: Vec<_> = (0..Q_LORA)
        .map(|index| bf16::from_f32((index as i32 % 127 - 63) as f32 * 0.03125))
        .collect();
    let weight: Vec<_> = (0..Q_LORA)
        .map(|index| bf16::from_f32(0.75 + (index % 19) as f32 * 0.015625))
        .collect();
    let normed = incumbent_norm(&input, &weight, 1.0e-6);
    assert_eq!(
        fused_cpu(&input, &weight, 1.0e-6),
        quantize_incumbent(&normed)
    );
}

#[test]
fn zeros_extremes_nonfinite_and_bf16_boundary_match_the_standalone_quantizer() {
    let zeros = vec![bf16::ZERO; Q_LORA];
    let mut extremes = vec![bf16::ZERO; Q_LORA];
    extremes[0] = bf16::from_f32(448.0);
    extremes[1] = bf16::from_f32(-448.0);
    extremes[2] = bf16::MAX;
    let mut nonfinite = vec![bf16::ZERO; Q_LORA];
    nonfinite[0] = bf16::INFINITY;
    nonfinite[1] = bf16::NEG_INFINITY;
    nonfinite[2] = bf16::NAN;
    let weight = vec![bf16::ONE; Q_LORA];
    for input in [&zeros, &extremes, &nonfinite] {
        let normed = incumbent_norm(input, &weight, 1.0e-6);
        assert_eq!(
            fused_cpu(input, &weight, 1.0e-6),
            quantize_incumbent(&normed)
        );
    }
    let (zero_scale, zero_bytes) = fused_cpu(&zeros, &weight, 1.0e-6);
    assert_eq!(zero_scale, 1.0e-8 / 448.0);
    assert!(zero_bytes.iter().all(|byte| *byte == 0));
    let (nonfinite_scale, nonfinite_bytes) = fused_cpu(&nonfinite, &weight, 1.0e-6);
    assert_eq!(nonfinite_scale, 1.0e-8 / 448.0);
    assert!(nonfinite_bytes.iter().all(|byte| byte & 0x7f == 0x7f));

    let fixed_max = bf16::ONE;
    let raw = f32::from_bits((1.0625f32 / 448.0).to_bits() + 1);
    let scale = fixed_max.to_f32() / 448.0;
    let before = f32_to_e4m3_rne_satfinite(raw / scale);
    let after = f32_to_e4m3_rne_satfinite(bf16::from_f32(raw).to_f32() / scale);
    assert_ne!(
        before, after,
        "the fused path must retain the BF16 boundary"
    );
}

#[test]
fn existing_fp8_eligibility_declines_before_any_arena_mutation() {
    let projection = fs::read_to_string(
        repo_root().join("crates/spark-model/src/layers/qwen3_attention/prefill/v4_fp8_proj.rs"),
    )
    .expect("V4 projection dispatch source");
    let function = projection
        .split("pub(super) fn try_w8a8_project_prefill")
        .nth(1)
        .expect("W8A8 eligibility function");
    let decline = function
        .find("return Ok(false);")
        .expect("decline before mutation");
    let arena = function
        .find("let a_fp8 = ctx.buffers.fp8_act();")
        .expect("arena mutation boundary");
    let quantize = function
        .find("ops::quantize_a_fp8_rows(")
        .expect("quantizer launch");
    assert!(decline < arena && arena < quantize);
    for guard in [
        "!v4_proj_fp8mma_enabled()",
        "self.w8a8_gemm_pipelined_k.0 == 0",
        "self.quantize_a_fp8_rows_k.0 == 0",
        "ctx.buffers.fp8_act_bytes() < (m as usize) * (k as usize)",
        "weight.scale_format != WeightQuantFormat::Fp8BlockScaled",
        "weight.n != n",
        "weight.k != k",
        "!n.is_multiple_of(FP8_BLOCK)",
        "!k.is_multiple_of(FP8_BLOCK)",
    ] {
        assert!(
            function.contains(guard),
            "missing eligibility guard: {guard}"
        );
    }
}

#[test]
fn structural_savings_are_exact_and_not_a_runtime_claim() {
    let avoided_bytes = TOKENS * Q_LORA as u64 * 6 * LAYERS;
    assert_eq!(avoided_bytes, 636_702_720);
    assert_eq!(avoided_bytes as f64 / 1_048_576.0, 607.207_031_25);
    assert_eq!(LAYERS, 43);
    assert_eq!(TOKENS * LAYERS, 103_630);
}

#[test]
fn sass_gate_pins_sm121a_resources_and_rejects_injected_flags() {
    let script = sass_gate();
    for contract in [
        "-arch=sm_121a",
        "--fmad=false",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing unreceipted nvcc flags",
        "expected_resource=",
        "spill stores",
        "spill loads",
        "ATOM|RED|LDL|STL",
        "shared_load",
        "shared_store",
        "source_sha256=",
        "cubin_sha256=",
    ] {
        assert!(
            script.contains(contract),
            "missing SASS gate contract: {contract}"
        );
    }
}

#[test]
fn experiment_has_no_registry_or_serving_reachability() {
    let root = repo_root();
    for path in [
        root.join("crates/spark-model/src"),
        root.join("crates/atlas-kernels"),
        root.join("kernels/gb10/common"),
        root.join("kernels/gb10/deepseek-v4-flash"),
    ] {
        assert_tree_excludes(&path);
    }
}
