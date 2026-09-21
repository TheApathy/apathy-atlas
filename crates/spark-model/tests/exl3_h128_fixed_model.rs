// SPDX-License-Identifier: AGPL-3.0-only

//! Source-only contracts for the DeepSeek-V4 fixed H4096 fused H128 post-pass.
//! The specialization may remove dynamic bounds, but must share every numeric
//! operation with the generic kernel and retain an explicit host fallback.

const KERNEL: &str = include_str!("../../../kernels/gb10/common/exl3_gemv.cu");
const K2_WRAPPER: &str = include_str!("../../../kernels/gb10/common/exl3_gemv_k2.cu");
const STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3_tail.rs");

#[test]
fn k2_wrapper_alone_emits_the_dsv4_fixed_hrow_entry() {
    assert!(K2_WRAPPER.contains("#define EXL3_HROW_DSV4_FIXED 1"));
    assert!(KERNEL.contains("#ifndef EXL3_HROW_DSV4_FIXED"));
    assert!(KERNEL.contains("#define EXL3_HROW_DSV4_FIXED 0"));
    assert!(KERNEL.contains("EXL3_HROW_DSV4_FIXED == 0 || EXL3_HROW_DSV4_FIXED == 1"));
    assert!(KERNEL.contains("exl3_h128_post_unpermute_rows("));
    assert!(KERNEL.contains("exl3_h128_post_unpermute_rows_h4096("));
}

#[test]
fn generic_and_fixed_entries_share_one_force_inlined_arithmetic_body() {
    assert!(KERNEL.contains("template <unsigned int FIXED_H>"));
    assert!(KERNEL.contains("__device__ __forceinline__ void exl3_h128_post_unpermute_rows_body"));
    assert!(KERNEL.contains("exl3_h128_post_unpermute_rows_body<0>"));
    assert!(KERNEL.contains("exl3_h128_post_unpermute_rows_body<4096>"));

    // There must be one weighted/Hadamard implementation, not a copied fixed
    // sibling whose rounding order can drift from the generic oracle.
    assert_eq!(
        KERNEL
            .matches("for (unsigned int k = 0; k < topk; ++k)")
            .count(),
        1
    );
}

#[test]
fn fixed_entry_rejects_every_nonexact_runtime_shape_or_grid() {
    assert!(KERNEL.contains("if (H != 4096) return;"));
    assert!(KERNEL.contains("gridDim.x != num_tokens"));
    assert!(KERNEL.contains("gridDim.y != 4"));
    assert!(KERNEL.contains("gridDim.z != 1"));
    assert!(KERNEL.contains("mov.u32 n, %ntid.x; setp.ne.u32 p, n, 256; @p exit;"));
    assert!(KERNEL.contains("mov.u32 n, %ntid.y; setp.ne.u32 p, n, 1; @p exit;"));
    assert!(KERNEL.contains("mov.u32 n, %ntid.z; setp.ne.u32 p, n, 1; @p exit;"));

    // Generic entry retains both runtime bounds for arbitrary H/grid shapes.
    assert!(KERNEL.contains("token >= num_tokens"));
    assert!(KERNEL.contains("chunk * 128 >= H"));
}

#[test]
fn model_state_loads_both_generic_and_fixed_hrow_handles() {
    assert!(STATE.contains("h128_post_unpermute_k: KernelHandle"));
    assert!(STATE.contains("h128_post_unpermute_h4096_k: KernelHandle"));
    let compact: String = STATE.split_whitespace().collect();
    assert!(compact.contains(
        "h128_post_unpermute_h4096_k:gpu.kernel(\"exl3_gemv_k2\",\"exl3_h128_post_unpermute_rows_h4096\")?"
    ));
    assert!(compact.contains(
        "h128_post_unpermute_k:gpu.kernel(\"exl3_gemv\",\"exl3_h128_post_unpermute_rows\")?"
    ));
}

#[test]
fn host_selects_fixed_only_for_exact_k2_h4096_with_opt_out() {
    assert!(STATE.contains("ATLAS_EXL3_HROW_FIXED_SHAPE"));
    assert!(STATE.contains("down.bits == 2"));
    assert!(STATE.contains("down.n == 4096"));
    assert!(DISPATCH.contains("hidden_size == 4096"));
    assert!(DISPATCH.contains("st.down.bits == 2"));
    assert!(DISPATCH.contains("h128_post_unpermute_h4096_k"));
    assert!(DISPATCH.contains("h128_post_unpermute_k"));
}

#[test]
fn host_launch_uses_exact_grid_for_fixed_and_generic_grid_on_mismatch() {
    assert!(DISPATCH.contains("[num_tokens, 4, 1]"));
    assert!(DISPATCH.contains("[num_tokens, hidden_size.div_ceil(H128_COLS_PER_BLOCK), 1]"));
    let fixed = DISPATCH
        .find("h128_post_unpermute_h4096_k")
        .expect("fixed H4096 dispatch");
    let generic = DISPATCH[fixed..]
        .find("h128_post_unpermute_k")
        .map(|offset| fixed + offset)
        .expect("generic mismatch fallback");
    assert!(fixed < generic);
}
