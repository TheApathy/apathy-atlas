// SPDX-License-Identifier: AGPL-3.0-only

//! Offline source/resource contracts for DeepSeek-V4 TC2 prefill attention.

const CUDA: &str =
    include_str!("../../../kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu");
const DISPATCH: &str = include_str!("../src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
const SASS_GATE: &str = include_str!("../../../scripts/check-v4-prefill-attn-tc2-sass.sh");

#[test]
fn source_keeps_three_independent_attention_entries() {
    for symbol in [
        "prefill_attn_compressed",
        "prefill_attn_compressed_tc",
        "prefill_attn_compressed_tc2",
    ] {
        assert_eq!(CUDA.matches(&format!("void {symbol}(")).count(), 1);
        assert!(SASS_GATE.contains(symbol));
    }
}

#[test]
fn tc2_dispatch_remains_default_with_explicit_fallbacks() {
    assert!(DISPATCH.contains("std::env::var(\"ATLAS_V4_PREFILL_TC2\").as_deref() != Ok(\"0\")"));
    assert!(DISPATCH.contains("std::env::var(\"ATLAS_V4_PREFILL_TC\").as_deref() != Ok(\"0\")"));
    assert!(DISPATCH.contains("self.prefill_attn_compressed_tc2_k"));
    assert!(DISPATCH.contains("self.prefill_attn_compressed_tc_k"));
    assert!(DISPATCH.contains("self.prefill_attn_compressed_k"));
}

#[test]
fn sass_gate_pins_sm121a_resources_and_tc2_topology() {
    for contract in [
        "-arch=sm_121a",
        "REG:167",
        "SHARED:22016",
        "REG:150",
        "SHARED:39744",
        "REG:166",
        "SHARED:33792",
        "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads",
        "HMMA.16816.F32.BF16",
        "LDSM.16.M88.4",
        "LDSM.16.MT88.4",
        "MUFU.EX2",
        "BAR.SYNC",
        "[[ $instruction_count == 1776 ]]",
        "allocated_registers_per_block",
    ] {
        assert!(SASS_GATE.contains(contract), "SASS gate omits `{contract}`");
    }
}
