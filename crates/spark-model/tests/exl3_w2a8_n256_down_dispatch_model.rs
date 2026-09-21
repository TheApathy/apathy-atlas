// SPDX-License-Identifier: AGPL-3.0-only

//! Source contract for the subordinate DeepSeek W2A8 N256-down selector.

use std::fs;
use std::path::PathBuf;

const FLAG: &str = "ATLAS_EXL3_PREFILL_W2A8_N256_DOWN";
const REQUESTED: &str = "w2a8_n256_down_requested";
const HANDLE: &str = "w2a8_grouped_n256_down_k";
const MODULE_SYMBOL: &str = "exl3_w2a8_grouped_prefill_n256_k2_down";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative))
        .unwrap_or_else(|error| panic!("missing {relative}: {error}"))
}

fn compact(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

#[test]
fn exact_deepseek_wrapper_is_include_only_and_binds_the_n256_down_abi() {
    let wrapper = source(
        "kernels/gb10/deepseek-v4-flash/nvfp4/\
         exl3_w2a8_grouped_prefill_n256_k2_down.cu",
    );
    assert_eq!(wrapper.matches("#include").count(), 1);
    assert_eq!(wrapper.matches("#define").count(), 6);
    assert!(wrapper.contains("#define W2A8_FIXED_N 4096"));
    assert!(wrapper.contains("#define W2A8_FIXED_K 2048"));
    assert!(wrapper.contains(&format!("#define W2A8_KERNEL_NAME {MODULE_SYMBOL}")));
    assert!(wrapper.contains("#define W2A8_PACKED_E4M3_CANDIDATE 1"));
    assert!(wrapper.contains("#define W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE 1"));
    assert!(wrapper.contains("#define W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE 1"));
    assert!(wrapper.contains("#include \"../../experiments/exl3_w2a8_grouped_prefill_n256.cu\""));

    let component = source("kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu");
    let flat = compact(&component);
    assert!(component.contains("#define W2A8_N_TILE 256"));
    assert!(component.contains("#define W2A8_THREADS 512"));
    assert!(component.contains("__launch_bounds__(W2A8_THREADS)"));
    assert!(
        flat.contains("unsignedintnum_experts,unsignedinttotal_rows,unsignedintN,unsignedintK,")
    );
    assert!(flat.contains("constexprunsignedintn_tiles=W2A8_FIXED_N/W2A8_N_TILE"));
    assert!(flat.contains("gridDim.x!=(unsignedlonglong)num_experts*n_tiles"));
}

#[test]
fn state_requires_a_separate_explicit_subordinate_selector_and_handle() {
    let state = source("crates/spark-model/src/layers/moe/exl3_decode.rs");
    assert!(state.contains(&format!("pub(crate) {REQUESTED}: bool")));
    assert!(state.contains(&format!("pub(crate) {HANDLE}: KernelHandle")));
    assert!(state.contains(FLAG));
    assert!(state.contains("as_deref() == Ok(\"1\")"));
    assert!(state.contains(MODULE_SYMBOL));
}

#[test]
fn dispatch_selects_n256_down_only_beneath_w2a8_and_keeps_n64_fallback() {
    let dispatch = source("crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs");
    let flat = compact(&dispatch);

    assert!(dispatch.contains(REQUESTED));
    assert!(dispatch.contains(HANDLE));
    assert!(dispatch.contains("w2a8_grouped_down_k"));
    assert!(flat.contains("W2A8_DOWN_N/256"));
    assert!(flat.contains(".block([512,1,1])"));
    assert!(flat.contains(".arg_u32(total_expanded)"));

    let mutation = dispatch.find("// MUTATION START").expect("mutation marker");
    let before = &dispatch[..mutation];
    assert!(before.contains("pf.w2a8_n256_down_requested"));
    assert!(before.contains("pf.w2a8_grouped_n256_down_k.0 != 0"));
    assert!(before.contains("let use_n256_down"));
    assert!(before.contains("n256_ranges_disjoint"));
    assert!(before.contains("n256_pointers_aligned"));
    assert!(before.contains("checked_range(down_fp8, down.total_bytes)"));
    assert!(before.contains("checked_range(expert_down_out, down_output_bytes)"));
    let after = &dispatch[mutation..];
    let selector = after
        .find("plan.use_n256_down")
        .expect("prevalidated N256 subordinate selector");
    let n256 = after[selector..].find(HANDLE).expect("N256 down handle") + selector;
    let fallback = after[n256..]
        .find("w2a8_grouped_down_k")
        .expect("N64 down fallback")
        + n256;
    assert!(selector < n256 && n256 < fallback);
}
