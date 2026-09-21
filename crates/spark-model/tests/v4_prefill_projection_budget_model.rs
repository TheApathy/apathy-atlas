// SPDX-License-Identifier: AGPL-3.0-only

//! CPU arithmetic and source contracts for the major DeepSeek-V4 prefill projections.

const PROJECTION: &str = include_str!("../src/layers/qwen3_attention/prefill/v4_fp8_proj.rs");
const PREFILL: &str = include_str!("../src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
const ROW_QUANTIZER: &str = include_str!("../../../kernels/gb10/common/w8a8_gemm_pipelined.cu");
const HISTORICAL_CAMPAIGN: &str = include_str!("../../../docs/PREFILL-CAMPAIGN-2026-08-10.md");

const TOKENS: u64 = 2_410;
const LAYERS: u64 = 43;

#[derive(Clone, Copy)]
struct ProjectionShape {
    n: u64,
    k: u64,
    count: u64,
}

const MAJOR: [ProjectionShape; 3] = [
    ProjectionShape {
        n: 32_768,
        k: 1_024,
        count: 1,
    }, // wq_b
    ProjectionShape {
        n: 1_024,
        k: 4_096,
        count: 8,
    }, // grouped wo_a
    ProjectionShape {
        n: 4_096,
        k: 8_192,
        count: 1,
    }, // wo_b
];

fn flops_per_layer(shape: ProjectionShape) -> u64 {
    2 * TOKENS * shape.n * shape.k * shape.count
}

fn compact(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn exact_major_projection_math_is_20_863_tf_per_pass() {
    let each = MAJOR.map(flops_per_layer);
    assert_eq!(each, [161_732_362_240; 3]);

    let per_layer: u64 = each.into_iter().sum();
    let per_pass = per_layer * LAYERS;
    assert_eq!(per_layer, 485_197_086_720);
    assert_eq!(per_pass, 20_863_474_728_960);

    // Hardware-peak floors are prioritization arithmetic, not runtime claims.
    assert_eq!(per_pass / 250_000_000_000, 83); // whole milliseconds at 250 TF/s
    assert_eq!(per_pass / 500_000_000_000, 41); // whole milliseconds at 500 TF/s
}

#[test]
fn fp8_activation_quantization_has_an_explicit_traffic_floor() {
    // One BF16 read plus one E4M3 write, plus one f32 scale per input row
    // for each projection invocation: wq_b once, grouped wo_a once over the
    // full attn_out row before its eight strided GEMMs, and wo_b once.
    // Cache effects and kernel instructions are deliberately excluded, so
    // this remains a lower bound.
    let input_widths = 1_024 + 32_768 + 8_192;
    let elements_per_layer = TOKENS * input_widths;
    let activation_bytes = elements_per_layer * LAYERS * 3;
    let quantizations_per_layer = 1 + 1 + 1;
    let scale_bytes = TOKENS * LAYERS * quantizations_per_layer * size_of::<f32>() as u64;

    assert_eq!(activation_bytes, 13_052_405_760);
    assert_eq!(scale_bytes, 1_243_560);
    assert_eq!(activation_bytes / 273_000_000, 47); // whole ms at 273 GB/s
    assert_eq!((activation_bytes + scale_bytes) / 273_000_000, 47);

    let quantizer = compact(ROW_QUANTIZER);
    assert!(quantizer.contains("FP32 scale[M]"));
    assert!(quantizer.contains("row_scale[m] = warp_max[0]"));
    let grouped = PROJECTION
        .split("pub(super) fn v4_grouped_wo_a_prefill")
        .nth(1)
        .expect("grouped wo_a source");
    let arm_start = grouped
        .find("if w8a8_inplace {")
        .expect("grouped in-place W8A8 arm");
    let arm_tail = &grouped[arm_start..];
    let arm_end = arm_tail
        .find("return Ok(());")
        .expect("grouped in-place W8A8 return");
    let arm = &arm_tail[..arm_end];
    assert_eq!(arm.matches("ops::quantize_a_fp8_rows(").count(), 1);
    assert!(arm.find("ops::quantize_a_fp8_rows(") < arm.find("for group in 0..o_groups"));
}

#[test]
fn selectors_are_explicit_and_decline_before_quantization() {
    let src = compact(PROJECTION);

    assert!(src.contains("ATLAS_V4_PREFILL_CUBLASLT"));
    assert!(src.contains("std::env::var(\"ATLAS_V4_PREFILL_CUBLASLT\").as_deref() != Ok(\"0\")"));
    assert!(src.contains("weight_bf16.is_null() || v4_cublaslt_poisoned().load(Relaxed)"));
    assert!(src.contains("Err(e) => { v4_cublaslt_poisoned().store(true, Relaxed);"));

    assert!(src.contains("std::env::var(\"ATLAS_V4_PROJ_FP8MMA\").as_deref() == Ok(\"1\")"));
    for guard in [
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
            src.contains(guard),
            "missing FP8 eligibility guard: {guard}"
        );
    }

    let decline = src.find("return Ok(false);").expect("missing FP8 decline");
    let quantize = src
        .find("ops::quantize_a_fp8_rows(")
        .expect("missing activation quantization");
    assert!(
        decline < quantize,
        "eligibility must precede the first write"
    );
}

#[test]
fn projection_dispatch_retains_residency_and_fallback_contracts() {
    let src = compact(PROJECTION);
    assert!(src.contains("if !dense.weight.is_null()"));
    assert!(src.contains("if try_v4_cublas_prefill(input, dense.weight, output, m, n, k, stream, label) { return Ok(()); }"));
    assert!(src.contains("return ops::dense_gemm_bf16_pipelined("));
    assert!(src.contains("if self.try_w8a8_project_prefill(ctx, input, weight, output, m, n, k, stream)? { return Ok(()); }"));
    assert!(src.contains("ops::w8a16_gemm_pipelined("));
    assert!(src.contains("RELEASE_BF16=0 + ATLAS_V4_PREFILL_CUBLASLT=1"));
    assert!(src.contains("+~8 GiB resident"));

    let prefill = compact(PREFILL);
    assert!(prefill.contains("self.v4_project_prefill( ctx, q_latent, &mla.wq_b"));
    assert!(prefill.contains("self.v4_grouped_wo_a_prefill("));
    assert!(prefill.contains("self.v4_project_prefill( ctx, o_latent, &mla.wo_b"));
}

#[test]
fn historical_850_ms_bucket_is_not_current_remaining_cost() {
    assert!(HISTORICAL_CAMPAIGN.contains("Prefill campaign 2026-08-09/10"));
    assert!(HISTORICAL_CAMPAIGN.contains("| o_proj + wq_b | ~0.85 s |"));
    assert!(PROJECTION.contains("≈480 ms"));
    assert!(PROJECTION.contains("less per pass"));

    // Current standalone cuBLASLt timings for the three major families sum to
    // 229.792 ms/pass. The old 850 ms combined waterfall bucket predates this
    // dispatch and includes a different execution state; treating it as the
    // current remaining cost would overstate this modeled kernel sum by >3x.
    let current_cublas_us_per_layer = 1_920 + 1_920 + 8 * 188;
    let current_cublas_us_per_pass = current_cublas_us_per_layer * LAYERS;
    assert_eq!(current_cublas_us_per_pass, 229_792);
    assert!(850_000 > 3 * current_cublas_us_per_pass);
}
