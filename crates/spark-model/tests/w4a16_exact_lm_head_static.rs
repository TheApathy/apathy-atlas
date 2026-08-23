// SPDX-License-Identifier: AGPL-3.0-only

//! Offline source-contract guard for the exact multi-row NVFP4 LM-head GEMV.

const KERNEL: &str = include_str!("../../../kernels/gb10/common/w4a16_gemv.cu");
const WRAPPER: &str = include_str!("../src/layers/ops/gemv_exact_lm_head.rs");
const ROUTE: &str = include_str!("../src/layers/ops/gemv_exact_lm_head/route.rs");
const MICRO: &str = include_str!("../examples/w4a16_exact_lm_head_microtest.rs");

const OUTPUTS_PER_BLOCK: usize = 4;
const THREADS_PER_OUTPUT: usize = 64;
const BLOCK_THREADS: usize = OUTPUTS_PER_BLOCK * THREADS_PER_OUTPUT;

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"));
    let end_index = source[start_index..]
        .find(end)
        .map(|offset| start_index + offset)
        .unwrap_or_else(|| panic!("missing section end: {end}"));
    &source[start_index..end_index]
}

fn braced_body_after<'a>(source: &'a str, needle: &str) -> &'a str {
    let needle_index = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing source contract: {needle}"));
    let open_index = source[needle_index..]
        .find('{')
        .map(|offset| needle_index + offset)
        .unwrap_or_else(|| panic!("missing opening brace after: {needle}"));
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open_index..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open_index + 1..open_index + offset];
                }
            }
            _ => {}
        }
    }
    panic!("missing closing brace after: {needle}");
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|ch| !ch.is_whitespace() && *ch != '\\')
        .collect()
}

fn exact_body() -> &'static str {
    section(
        KERNEL,
        "template <int MAX_M>\n__device__ __forceinline__ void w4a16_gemv_batch_logits_exact_body(",
        "#define DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT",
    )
}

#[test]
fn exact_family_exports_all_register_tiers_and_wrapper_names() {
    for max_rows in [4, 8, 17, 32] {
        let symbol = format!("w4a16_gemv_batch_logits_exact_m{max_rows}");
        assert!(KERNEL.contains(&symbol), "CUDA missing {symbol}");
        assert!(ROUTE.contains(&symbol), "Rust route table missing {symbol}");
        assert!(
            KERNEL.contains(&format!("{symbol}, {max_rows})")),
            "{symbol} must instantiate MAX_M={max_rows}"
        );
    }
}

#[test]
fn tier_mapping_has_no_lossy_or_oversized_fallback() {
    let route = compact(ROUTE);
    for mapping in [
        "2..=4=>Some(ExactLmHeadTier::M4)",
        "5..=8=>Some(ExactLmHeadTier::M8)",
        "9..=17=>Some(ExactLmHeadTier::M17)",
        "18..=32=>Some(ExactLmHeadTier::M32)",
        "_=>None",
    ] {
        assert!(route.contains(mapping), "missing tier mapping: {mapping}");
    }
    let wrapper = compact(WRAPPER);
    assert!(wrapper.contains("ensure!(kernel.0!=0"));
    assert!(!wrapper.contains("w4a16_gemm_t_m32_n64"));
    assert!(!wrapper.contains("w4a16_gemv_batch3_logits("));
}

#[test]
fn exact_body_preserves_k1_association_and_two_ordered_updates() {
    let base = compact(section(
        KERNEL,
        "__device__ __forceinline__ float w4a16_gemv_partial(",
        "// W4A16 GEMV: C[n]",
    ));
    let exact = compact(exact_body());

    assert!(base.contains("k16=orig_lane;k16<K16;k16+=64u"));
    assert!(exact.contains("k16=lane;k16<K16;k16+=64u"));
    assert!(exact.contains(
        "acc[row]+=__bfloat162float(a_lo_bf)*w_lo[b];acc[row]+=__bfloat162float(a_hi_bf)*w_hi[b];"
    ));
    assert!(!exact.contains("*w_lo[b]+__bfloat162float"));
    assert!(!exact.contains("K8"));
    assert!(!exact.contains("__nv_fp8_e4m3(A"));
    assert!(!exact.contains("mma.sync"));
}

#[test]
fn weight_unpack_is_outside_the_activation_row_loop() {
    let exact = exact_body();
    let unpack = exact
        .find("float w_lo[8];")
        .expect("exact kernel must unpack one K16 weight chunk");
    let row_loop = exact[unpack..]
        .find("for (int row = 0; row < MAX_M; ++row) {")
        .map(|offset| unpack + offset)
        .expect("exact kernel must visit activation rows");
    let activation = exact
        .find("const __nv_bfloat16* __restrict__ A_row =")
        .expect("exact kernel must form a predicated activation row");
    assert!(unpack < row_loop);
    assert!(row_loop < activation);
    assert_eq!(exact.matches("packed8 =").count(), 1);
}

#[test]
fn tail_groups_cannot_return_or_form_global_row_addresses() {
    let exact = exact_body();
    assert!(!exact.contains("return;"));
    assert_eq!(exact.matches("__syncthreads();").count(), 2);
    assert!(exact.contains("const bool valid = rows_valid && n < N;"));

    let valid_body = braced_body_after(exact, "if (valid) {");
    for required in [
        "B_packed + weight_row + k16 * 8",
        "B_scale[scale_row + scale_group]",
        "A + (unsigned long long)row * K",
    ] {
        assert!(
            valid_body.contains(required),
            "missing guarded load: {required}"
        );
    }
    let before_valid = &exact[..exact.find("if (valid) {").unwrap()];
    assert!(!before_valid.contains("B_packed + weight_row"));
    assert!(!before_valid.contains("B_scale[scale_row"));

    let store = braced_body_after(exact, "if (valid && lane == 0) {");
    assert!(store.contains("C[(unsigned long long)row * N + n]"));
}

#[test]
fn every_n_mod_4_tail_keeps_all_threads_in_the_block() {
    for n in 1usize..=12 {
        let grid = n.div_ceil(OUTPUTS_PER_BLOCK);
        let base = (grid - 1) * OUTPUTS_PER_BLOCK;
        let valid_groups = (0..OUTPUTS_PER_BLOCK)
            .filter(|local_out| base + local_out < n)
            .count();
        assert_eq!(
            valid_groups * THREADS_PER_OUTPUT
                + (OUTPUTS_PER_BLOCK - valid_groups) * THREADS_PER_OUTPUT,
            BLOCK_THREADS
        );
    }
}

#[test]
fn ordinary_k1_variants_are_block_uniform_for_non_four_vocab_tails() {
    let base = compact(section(
        KERNEL,
        "extern \"C\" __global__ void w4a16_gemv(",
        "// W4A16 exact multi-row GEMV",
    ));
    let logits = compact(section(
        KERNEL,
        "extern \"C\" __global__ void w4a16_gemv_logits(",
        "// W4A16 double-GEMV",
    ));
    for (name, body) in [("bf16", base), ("fp32", logits)] {
        assert!(
            !body.contains("if(n>=N)return"),
            "unsafe {name} tail return"
        );
        assert!(body.contains("constboolvalid=n<N;"));
        assert_eq!(body.matches("__syncthreads();").count(), 2);
        assert!(body.contains("if(valid){"));
        assert!(body.contains("if(valid&&lane==0){"));
    }

    for n in [5usize, 6, 7, 9, 10, 11] {
        let base = (n.div_ceil(OUTPUTS_PER_BLOCK) - 1) * OUTPUTS_PER_BLOCK;
        let valid_groups = n - base;
        assert!((1..=3).contains(&valid_groups));
        assert_eq!(
            valid_groups * THREADS_PER_OUTPUT
                + (OUTPUTS_PER_BLOCK - valid_groups) * THREADS_PER_OUTPUT,
            BLOCK_THREADS
        );
    }
}

#[test]
fn device_oracle_exercises_k1_with_the_unpadded_effective_vocab() {
    let micro = compact(MICRO);
    assert!(micro.contains("serial_out.offset(row*fixture.logical_n*size_of::<u16>())"));
    assert!(micro.contains("fixture.logical_nasu32,fixture.kasu32,stream"));
    assert!(!micro.contains("crop_serial"));
    for width in [7, 11, 13, 17, 19, 23, 29] {
        assert!(
            MICRO.contains(&width.to_string()),
            "device fixture missing odd/non-four N={width}"
        );
    }
}

#[test]
fn legacy_batch3_is_a_non_vacuous_negative_control() {
    let legacy = compact(section(
        KERNEL,
        "extern \"C\" __global__ void w4a16_gemv_batch3(",
        "// W4A16 GEMV batch3 with inline Q/Gate deinterleave",
    ));
    let exact = compact(exact_body());
    assert!(legacy.contains("constunsignedintK8=K/8;"));
    assert!(legacy.contains("acc0+=__bfloat162float(a0_lo)*w_lo+__bfloat162float(a0_hi)*w_hi;"));
    assert!(!exact.contains("constunsignedintK8=K/8;"));
    assert!(!exact.contains("*w_lo+__bfloat162float"));
}

#[test]
fn output_reduction_and_abi_are_pinned() {
    let exact = compact(exact_body());
    assert!(exact.contains("__shfl_down_sync(0xFFFFFFFF,acc[row],offset)"));
    assert!(exact.contains("__float2bfloat16(smem[base]+smem[base+1])"));

    let macro_body = compact(section(
        KERNEL,
        "#define DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT",
        "#undef DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT",
    ));
    assert!(macro_body.contains(
        "constfloatscale2,__nv_bfloat16*__restrict__C,unsignedintM,unsignedintN,unsignedintK)"
    ));
    let wrapper = compact(WRAPPER);
    assert!(wrapper.contains(".arg_ptr(output).arg_u32(rows).arg_u32(n).arg_u32(k)"));
}
