// SPDX-License-Identifier: AGPL-3.0-only

//! CPU source/ABI gates; numerical equivalence still requires the GPU probe.

use std::path::PathBuf;

fn source(relative: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join(relative)).expect("production source must exist")
}

fn compact(value: &str) -> String {
    value.split_whitespace().collect()
}

#[test]
fn sigmoid_model_resolves_sigmoid_multisequence_not_silu() {
    let init = source("crates/spark-model/src/layers/qwen3_ssm/init.rs");
    let block = init
        .split("gated_rms_norm_f32_multi_seq_k:")
        .nth(1)
        .unwrap();
    let block = compact(block.split("conv1d_l2norm_chunk3_k:").next().unwrap());
    assert_eq!(
        block,
        concat!(
            "ifconfig.output_gate_type==\"sigmoid\"{",
            "super::super::try_kernel(gpu,\"qwen4_hyper\",",
            "\"qwen4_gated_rms_norm_sigmoid_f32_multi_seq\",)",
            "}else{super::super::try_kernel(gpu,\"gated_rms_norm_f32_multi_seq\",",
            "\"gated_rms_norm_f32_multi_seq\",)},"
        )
    );
}

#[test]
fn wrapper_abi_matches_existing_four_pointer_five_scalar_launcher() {
    let wrapper = source("kernels/gb10/common/qwen4_sigmoid_multi_seq.cuh");
    let signature = wrapper
        .split("qwen4_gated_rms_norm_sigmoid_f32_multi_seq(")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    assert_eq!(
        compact(signature),
        concat!(
            "constfloat*input,const__nv_bfloat16*gate,const__nv_bfloat16*weight,",
            "__nv_bfloat16*output,unsignedinthead_dim,floateps,",
            "unsignedintinput_stride,unsignedintgate_stride,unsignedintoutput_stride"
        )
    );
    let ops = compact(&source("crates/spark-model/src/layers/ops/norm.rs"));
    assert!(ops.contains(".grid([num_v_heads,num_seqs,1]).block([head_dim.min(1024),1,1])"));
    assert!(ops.contains(concat!(
        ".arg_ptr(input).arg_ptr(gate).arg_ptr(weight.weight).arg_ptr(output)",
        ".arg_u32(head_dim).arg_f32(eps).arg_u32(input_stride)",
        ".arg_u32(gate_stride).arg_u32(output_stride).launch(stream)"
    )));
}

#[test]
fn wrapper_only_offsets_rows_then_calls_the_shipping_sigmoid_body() {
    let wrapper = source("kernels/gb10/common/qwen4_sigmoid_multi_seq.cuh");
    let body = wrapper
        .split("unsigned int output_stride) {")
        .nth(1)
        .unwrap();
    assert_eq!(
        compact(body.split("#endif").next().unwrap()),
        concat!(
            "constunsignedlonglongseq=blockIdx.y;",
            "qwen4_gated_rms_sigmoid_body(input+seq*input_stride,",
            "gate+seq*gate_stride,weight,output+seq*output_stride,",
            "head_dim,eps,head_dim);}",
        )
    );
}

#[test]
fn wrapper_is_included_after_scalar_body_without_replacing_generic_silu() {
    let source = source("kernels/gb10/common/qwen4_hyper.cu");
    let scalar = source
        .find("void qwen4_gated_rms_norm_sigmoid_f32(")
        .unwrap();
    let include = source
        .find("#include \"qwen4_sigmoid_multi_seq.cuh\"")
        .unwrap();
    assert!(include > scalar);
    assert_eq!(
        source
            .matches("#include \"qwen4_sigmoid_multi_seq.cuh\"")
            .count(),
        1
    );
    assert!(source.contains("v * inv * w / (1.0f + expf(-g))"));
    let generic = self::source("kernels/gb10/common/gated_rms_norm_f32_multi_seq.cu");
    assert!(generic.contains("float s  = gv / (1.0f + __expf(-gv));"));
    assert!(!generic.contains("qwen4_gated_rms_sigmoid_body"));
}
