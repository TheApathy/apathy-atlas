// SPDX-License-Identifier: AGPL-3.0-only

use super::{Qwen38QkNormRopeRoute as Route, parse_qwen38_qknorm_rope, qwen38_qknorm_rope_route};

fn route(overrides: impl FnOnce(&mut [bool; 14])) -> Route {
    // exact-model, single, standard, gated, qnorm, knorm, no-full,
    // no-vnorm, mrope, proportional, theta-valid, parent, symbol, requested
    let mut flags = [true; 14];
    flags[9] = false;
    overrides(&mut flags);
    qwen38_qknorm_rope_route(
        flags[13], flags[0], flags[1], flags[2], flags[3], flags[4], flags[5], flags[6], flags[7],
        flags[8], flags[9], 8192, 24, 4, 256, 12_288, 64, flags[10], flags[11], flags[12],
    )
}

#[test]
fn selector_is_exact_qwen_single_sequence_and_fail_closed() {
    assert_eq!(route(|flags| flags[13] = false), Route::Disabled);
    for index in 0..11 {
        assert_eq!(
            route(|flags| flags[index] = !flags[index]),
            Route::Ineligible,
            "eligibility flag {index}"
        );
    }
    assert_eq!(route(|flags| flags[11] = false), Route::Missing);
    assert_eq!(route(|flags| flags[12] = false), Route::Missing);
    assert_eq!(route(|_| {}), Route::Complete);

    for (tokens, nq, nkv, hd, stride, rotary) in [
        (0, 24, 4, 256, 12_288, 64),
        (8192, 23, 4, 256, 12_288, 64),
        (8192, 24, 3, 256, 12_288, 64),
        (8192, 24, 4, 128, 12_288, 64),
        (8192, 24, 4, 256, 12_287, 64),
        (8192, 24, 4, 256, 12_288, 128),
    ] {
        assert_eq!(
            qwen38_qknorm_rope_route(
                true, true, true, true, true, true, true, true, true, true, false, tokens, nq, nkv,
                hd, stride, rotary, true, true, true,
            ),
            Route::Ineligible
        );
    }
}

#[test]
fn dispatch_is_atomic_and_path_attribution_is_distinct() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paged = std::fs::read_to_string(
        root.join("crates/spark-model/src/layers/qwen3_attention/prefill/paged.rs"),
    )
    .unwrap();
    let cache_skip = std::fs::read_to_string(
        root.join("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip.rs"),
    )
    .unwrap();

    for source in [&paged, &cache_skip] {
        for identity in [
            "ctx.config.model_type == \"qwen3_5\"",
            "ctx.config.num_experts == 0",
            "ctx.config.hidden_size == 5120",
            "ctx.config.tp_world_size.max(1) == 1",
            "ctx.config.mrope_section == [11, 11, 10]",
        ] {
            assert!(
                source.contains(identity),
                "missing identity guard: {identity}"
            );
        }
        assert!(source.contains("if !qknorm_rope_fused {\n            if self.gated"));
        assert!(
            source.contains("if !qknorm_rope_fused {\n            if let Some(ref k_norm_full)")
        );
        assert!(source.contains("if qknorm_rope_fused {"));

        let complete = source.find("Qwen38QkNormRopeRoute::Complete =>").unwrap();
        let missing = source[complete..]
            .find("Qwen38QkNormRopeRoute::Missing =>")
            .map(|offset| complete + offset)
            .unwrap();
        let legacy_q = source.find("if !qknorm_rope_fused {").unwrap();
        assert!(complete < missing && missing < legacy_q);
    }

    assert!(paged.contains("fused_meta.positions_h"));
    assert!(paged.contains("fused_meta.positions_w"));
    assert!(paged.contains("mark_qwen38_qknorm_rope_paged_engaged"));
    assert!(cache_skip.contains(
        "meta.positions,\n                    meta.positions,\n                    meta.positions,"
    ));
    assert!(cache_skip.contains("mark_qwen38_qknorm_rope_cache_skip_engaged"));
}

#[test]
fn host_launch_abi_is_ordered() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let host =
        std::fs::read_to_string(root.join("crates/spark-model/src/layers/ops/ssm_preproc.rs"))
            .unwrap();
    let start = host.find("pub fn qwen38_prefill_qknorm_rope(").unwrap();
    let body = &host[start..];
    let mut cursor = 0;
    for argument in [
        ".arg_ptr(qg_data)",
        ".arg_ptr(q_out)",
        ".arg_ptr(k_data)",
        ".arg_ptr(q_norm_weight)",
        ".arg_ptr(k_norm_weight)",
        ".arg_ptr(pos_t)",
        ".arg_ptr(pos_h)",
        ".arg_ptr(pos_w)",
        ".arg_u32(num_tokens)",
        ".arg_u32(num_q_heads)",
        ".arg_u32(num_kv_heads)",
        ".arg_u32(head_dim)",
        ".arg_u32(qg_stride)",
        ".arg_u32(rotary_dim)",
        ".arg_f32(eps)",
        ".arg_f32(theta)",
    ] {
        let next = body[cursor..]
            .find(argument)
            .unwrap_or_else(|| panic!("missing ordered ABI argument: {argument}"));
        cursor += next + argument.len();
    }
}

#[test]
fn explicit_flag_accepts_only_absent_zero_or_one() {
    assert_eq!(parse_qwen38_qknorm_rope(None), Ok(false));
    assert_eq!(parse_qwen38_qknorm_rope(Some("0")), Ok(false));
    assert_eq!(parse_qwen38_qknorm_rope(Some("1")), Ok(true));
    for invalid in ["", "true", "01", "2", " 1"] {
        assert!(parse_qwen38_qknorm_rope(Some(invalid)).is_err());
    }
}

#[test]
fn cuda_source_pins_parent_arithmetic_and_bf16_boundary() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cuda =
        std::fs::read_to_string(root.join("kernels/gb10/common/qwen38_prefill_qknorm_rope.cu"))
            .unwrap();
    let manifest =
        std::fs::read_to_string(root.join("kernels/gb10/qwen3.8-27b/nvfp4/KERNEL.toml")).unwrap();
    assert!(manifest.contains("qwen38_prefill_qknorm_rope = \"qwen38_prefill_qknorm_rope\""));
    for needle in [
        "__launch_bounds__(256, 2)",
        "value = __shfl_xor_sync(0xFFFFFFFF, value, 16) + value",
        "value += __shfl_xor_sync(0xFFFFFFFF, value, 16)",
        "sum_sq += value * value",
        "sum_sq += v0 * v0 + v1 * v1",
        "__float2bfloat16(value * rms * (1.0f + weight))",
        "q38_qknorm_pack_bf16x2",
        "const unsigned int section = pair_idx % 3",
        "const double freq_exp_d = (double)(2 * pair_idx) / (double)rotary_dim",
        "const float angle = (float)abs_pos * freq",
        "const float y0 = x0 * cos_val - x1 * sin_val",
        "const float y1 = x1 * cos_val + x0 * sin_val",
    ] {
        assert!(
            cuda.contains(needle),
            "missing arithmetic contract: {needle}"
        );
    }
    let q_norm = cuda.find("Store the normalized BF16 boundary").unwrap();
    let q_rope = cuda.find("Preserve the normalized BF16 values").unwrap();
    let reuse = cuda.find("The Q tile is dead").unwrap();
    assert!(q_norm < q_rope && q_rope < reuse);
    assert!(cuda[q_rope..reuse].contains("__syncthreads();"));
}
