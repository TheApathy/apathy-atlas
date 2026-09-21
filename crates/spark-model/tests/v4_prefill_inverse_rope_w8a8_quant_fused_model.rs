// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for isolated V4 inverse-RoPE + full-row W8A8 quantization.

use half::bf16;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const TOKENS: u64 = 2_410;
const LAYERS: u64 = 43;
const NQ: usize = 64;
const HEAD_DIM: usize = 512;
const NOPE_DIM: usize = 448;
const ROPE_DIM: usize = 64;
const ROPE_PAIRS: usize = 32;
const ROW_WIDTH: usize = NQ * HEAD_DIM;
const QUANT_BLOCK: usize = 256;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn experiment() -> String {
    fs::read_to_string(
        repo_root().join("kernels/gb10/experiments/v4_prefill_inverse_rope_w8a8_quant_fused.cu"),
    )
    .expect("isolated V4 inverse-RoPE/W8A8 experiment must exist")
}

fn sass_gate() -> String {
    fs::read_to_string(
        repo_root().join("scripts/check-v4-prefill-inverse-rope-w8a8-quant-fused-sass.sh"),
    )
    .expect("isolated V4 inverse-RoPE/W8A8 SASS gate must exist")
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
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

fn inverse_pair(x0: bf16, x1: bf16, position: u32, inv_freq: f32, mscale: f32) -> (bf16, bf16) {
    let x0 = x0.to_f32();
    let x1 = x1.to_f32();
    let angle = position as f32 * inv_freq;
    let cos_value = angle.cos() * mscale;
    let sin_value = angle.sin() * mscale;
    (
        bf16::from_f32(x0 * cos_value + x1 * sin_value),
        bf16::from_f32(x1 * cos_value - x0 * sin_value),
    )
}

fn quantize_from<F>(load: F) -> (f32, Vec<u8>)
where
    F: Fn(usize) -> bf16,
{
    let mut thread_max = [0.0_f32; QUANT_BLOCK];
    for (tid, maximum) in thread_max.iter_mut().enumerate() {
        for k in (tid..ROW_WIDTH).step_by(QUANT_BLOCK) {
            *maximum = maximum.max(load(k).to_f32().abs());
        }
    }
    let mut warp_max = [0.0_f32; 8];
    for (warp, lanes) in thread_max.chunks_exact(32).enumerate() {
        warp_max[warp] = lanes.iter().copied().fold(0.0_f32, f32::max);
    }
    let scale = warp_max.into_iter().fold(0.0_f32, f32::max).max(1.0e-8) / 448.0;
    let mut output = vec![0u8; ROW_WIDTH];
    for tid in 0..QUANT_BLOCK {
        for k in (tid..ROW_WIDTH).step_by(QUANT_BLOCK) {
            output[k] = f32_to_e4m3_rne_satfinite(load(k).to_f32() / scale);
        }
    }
    (scale, output)
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
                !source.contains("v4_prefill_inverse_rope_w8a8_quant_fused"),
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
        "#define V4_INV_QUANT_NQ 64u",
        "#define V4_INV_QUANT_HEAD_DIM 512u",
        "#define V4_INV_QUANT_NOPE_DIM 448u",
        "#define V4_INV_QUANT_ROPE_DIM 64u",
        "#define V4_INV_QUANT_ROW_WIDTH 32768u",
        "#define V4_INV_QUANT_BLOCK 256u",
        "__launch_bounds__(V4_INV_QUANT_BLOCK)",
        "v4_prefill_inverse_rope_w8a8_quant_fused",
    ] {
        assert!(source.contains(contract), "missing contract: {contract}");
    }
    assert!(flat.contains(
        "num_tokens==0||input==nullptr||positions==nullptr||inv_freq==nullptr||output_fp8==nullptr||row_scale==nullptr"
    ));
    assert!(flat.contains(
        "num_q_heads!=V4_INV_QUANT_NQ||head_dim!=V4_INV_QUANT_HEAD_DIM||nope_dim!=V4_INV_QUANT_NOPE_DIM||rotary_dim!=V4_INV_QUANT_ROPE_DIM"
    ));
    assert!(flat.contains(
        "row_width!=V4_INV_QUANT_ROW_WIDTH||quant_block!=V4_INV_QUANT_BLOCK||!isfinite(mscale)"
    ));
    assert!(flat.contains("blockDim.x!=V4_INV_QUANT_BLOCK||blockDim.y!=1||blockDim.z!=1"));
    assert!(flat.contains("gridDim.x!=num_tokens||gridDim.y!=1||gridDim.z!=1"));
}

#[test]
fn pair_and_scalar_ownership_are_exact_bijections() {
    let mut pairs = HashSet::new();
    let mut tail_values = HashSet::new();
    for tid in 0..QUANT_BLOCK {
        for linear_pair in (tid..NQ * ROPE_PAIRS).step_by(QUANT_BLOCK) {
            assert!(pairs.insert(linear_pair));
            assert!(tail_values.insert(2 * linear_pair));
            assert!(tail_values.insert(2 * linear_pair + 1));
        }
    }
    assert_eq!(pairs.len(), NQ * ROPE_PAIRS);
    assert_eq!(tail_values.len(), NQ * ROPE_DIM);

    let scalar_columns: HashSet<_> = (0..QUANT_BLOCK)
        .flat_map(|tid| (tid..ROW_WIDTH).step_by(QUANT_BLOCK))
        .collect();
    assert_eq!(scalar_columns.len(), ROW_WIDTH);
}

#[test]
fn formulas_bf16_seam_and_quantizer_order_match_the_two_incumbents() {
    let source = experiment();
    let flat = compact(&source);
    for contract in [
        "for(unsignedintlinear_pair=tid;linear_pair<V4_INV_QUANT_TOTAL_PAIRS;linear_pair+=V4_INV_QUANT_BLOCK)",
        "constfloatangle=static_cast<float>(positions[token])*inv_freq[pair]",
        "constfloatcos_value=cosf(angle)*mscale",
        "constfloatsin_value=sinf(angle)*mscale",
        "__float2bfloat16(x0*cos_value+x1*sin_value)",
        "__float2bfloat16(x1*cos_value-x0*sin_value)",
        "for(unsignedintk=tid;k<V4_INV_QUANT_ROW_WIDTH;k+=V4_INV_QUANT_BLOCK)",
        "amax=fmaxf(amax,fabsf(__bfloat162float(value)))",
        "__shfl_down_sync(0xFFFFFFFFu,amax,offset)",
        "for(unsignedintwarp=1;warp<V4_INV_QUANT_WARPS;++warp)",
        "fmaxf(block_max,V4_INV_QUANT_SCALE_FLOOR)/V4_INV_QUANT_E4M3_MAX",
        "constfloatinverse_scale=1.0f/warp_max[0]",
        "const__nv_fp8_e4m3quantized(__bfloat162float(value)*inverse_scale)",
    ] {
        assert!(
            flat.contains(contract),
            "missing numeric contract: {contract}"
        );
    }
    assert!(flat.contains("rotated_tail[2u*linear_pair]=__float2bfloat16"));
    assert!(flat.contains("rotated_tail[2u*linear_pair+1u]=__float2bfloat16"));
    assert_eq!(source.matches("__syncthreads();").count(), 3);
}

#[test]
fn cpu_fusion_matches_inverse_bf16_then_full_row_quantization() {
    let input: Vec<_> = (0..ROW_WIDTH)
        .map(|index| bf16::from_f32((index as i32 % 251 - 125) as f32 * 0.007_812_5))
        .collect();
    let position = 2_409_u32;
    let frequencies: Vec<_> = (0..ROPE_PAIRS)
        .map(|pair| 0.000_013 * (pair + 1) as f32)
        .collect();
    let mscale = 1.125_f32;

    let mut sequential = input.clone();
    let mut shared_tail = vec![bf16::ZERO; NQ * ROPE_DIM];
    for head in 0..NQ {
        for pair in 0..ROPE_PAIRS {
            let input_index = head * HEAD_DIM + NOPE_DIM + 2 * pair;
            let (first, second) = inverse_pair(
                input[input_index],
                input[input_index + 1],
                position,
                frequencies[pair],
                mscale,
            );
            sequential[input_index] = first;
            sequential[input_index + 1] = second;
            shared_tail[head * ROPE_DIM + 2 * pair] = first;
            shared_tail[head * ROPE_DIM + 2 * pair + 1] = second;
        }
    }

    let incumbent = quantize_from(|k| sequential[k]);
    let fused = quantize_from(|k| {
        let head = k / HEAD_DIM;
        let dimension = k % HEAD_DIM;
        if dimension >= NOPE_DIM {
            shared_tail[head * ROPE_DIM + dimension - NOPE_DIM]
        } else {
            input[k]
        }
    });
    assert_eq!(fused.0.to_bits(), incumbent.0.to_bits());
    assert_eq!(fused.1, incumbent.1);
    assert_eq!(&input[..NOPE_DIM], &sequential[..NOPE_DIM]);
    assert_ne!(input[NOPE_DIM..HEAD_DIM], sequential[NOPE_DIM..HEAD_DIM]);
}

#[test]
fn w8a8_inplace_eligibility_and_dead_attn_output_boundary_are_source_locked() {
    let projection = fs::read_to_string(
        repo_root().join("crates/spark-model/src/layers/qwen3_attention/prefill/v4_fp8_proj.rs"),
    )
    .expect("V4 grouped projection source");
    let grouped = projection
        .split("pub(super) fn v4_grouped_wo_a_prefill")
        .nth(1)
        .expect("grouped wo_a function");
    let eligibility = grouped.find("let w8a8_inplace = fp8.is_some()").unwrap();
    let quantize = grouped.find("ops::quantize_a_fp8_rows(").unwrap();
    let early_return = grouped.find("return Ok(());").unwrap();
    assert!(eligibility < quantize && quantize < early_return);
    for contract in [
        "v4_proj_fp8mma_enabled()",
        "self.w8a8_gemm_pipelined_ld_k.0 != 0",
        "self.quantize_a_fp8_rows_k.0 != 0",
        "ctx.buffers.fp8_act_bytes() >= (n as usize) * (input_width as usize)",
        "attn_out",
        "a_fp8",
        "a_scale",
    ] {
        assert!(grouped.contains(contract), "eligibility omits `{contract}`");
    }

    let prefill = fs::read_to_string(
        repo_root().join("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs"),
    )
    .expect("V4 prefill source");
    let inverse = prefill.find("self.rope_yarn_interleaved_inv_k").unwrap();
    let diagnostics = prefill[inverse..]
        .find("if diag_this {")
        .map(|offset| inverse + offset)
        .unwrap();
    let grouped_call = prefill.find("self.v4_grouped_wo_a_prefill(").unwrap();
    assert!(inverse < diagnostics && diagnostics < grouped_call);
    let diagnostic_window = &prefill[diagnostics..grouped_call];
    assert!(diagnostic_window.contains("diag_norm("));
    assert!(diagnostic_window.contains("attn_out"));

    let experiment = experiment();
    for boundary in [
        "decide the existing",
        "w8a8_inplace eligibility before launch",
        "require diag_this == false",
        "materialized inverse-rotated BF16 attn_out",
        "diagnostics",
        "must bypass it",
    ] {
        assert!(
            experiment.contains(boundary),
            "missing applicability boundary: {boundary}"
        );
    }
}

#[test]
fn structural_savings_distinguish_conservative_and_exact_global_traffic() {
    let tail_values = TOKENS * NQ as u64 * ROPE_DIM as u64;
    let one_bf16_transaction = tail_values * 2;
    let conservative = one_bf16_transaction * 2;
    let exact_global = one_bf16_transaction * 3;
    assert_eq!(tail_values, 9_871_360);
    assert_eq!(conservative, 39_485_440);
    assert_eq!(conservative * LAYERS, 1_697_873_920);
    assert_eq!(exact_global, 59_228_160);
    assert_eq!(exact_global * LAYERS, 2_546_810_880);
    assert_eq!((2 - 1) * LAYERS, 43);
    assert_eq!(TOKENS * NQ as u64 * LAYERS, 6_632_320);

    let inverse_rope =
        fs::read_to_string(repo_root().join("kernels/gb10/experiments/v4_prefill_rope_fused.cu"))
            .expect("incumbent direct inverse RoPE source");
    let inverse_helper = inverse_rope
        .split("void v4_rope_pair_inverse(")
        .nth(1)
        .expect("direct inverse RoPE helper")
        .split("// Grid")
        .next()
        .unwrap();
    assert_eq!(
        inverse_helper
            .matches("ptr[d0] = __float2bfloat16(y0)")
            .count(),
        1
    );
    assert_eq!(
        inverse_helper
            .matches("ptr[d1] = __float2bfloat16(y1)")
            .count(),
        1
    );
    let inverse_entry = inverse_rope
        .split("void v4_prefill_rope_fused_inverse(")
        .nth(1)
        .expect("direct inverse RoPE entrypoint");
    assert!(inverse_entry.contains("v4_rope_pair_inverse("));

    let quantizer =
        fs::read_to_string(repo_root().join("kernels/gb10/common/w8a8_gemm_pipelined.cu"))
            .expect("incumbent quantizer source");
    let function = quantizer.split("void quantize_a_fp8_rows(").nth(1).unwrap();
    assert_eq!(function.matches("row[k]").count(), 2);
}

#[test]
fn sass_gate_pins_sm121a_resources_topology_and_injected_flag_rejection() {
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
        "F2FP\\.SATFINITE\\.E4M3\\.F32",
        "SHFL\\.DOWN",
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
