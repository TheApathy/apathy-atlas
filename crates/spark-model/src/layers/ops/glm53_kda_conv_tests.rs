// SPDX-License-Identifier: AGPL-3.0-only

use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};

use super::*;

const HOST: &str = include_str!("glm53_kda_conv.rs");
const CUDA: &str = include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/glm53_kda_conv.cu");

fn ordered(source: &str, terms: &[&str]) -> bool {
    let mut tail = source;
    for term in terms {
        let Some(index) = tail.find(term) else {
            return false;
        };
        tail = &tail[index + term.len()..];
    }
    true
}

fn section<'a>(source: &'a str, begin: &str, end: &str) -> &'a str {
    let start = source.find(begin).unwrap();
    let tail = &source[start..];
    let finish = tail.find(end).unwrap();
    &tail[..finish]
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn signature(source: &str, name: &str) -> Option<String> {
    let start = source.find(name)?;
    let tail = &source[start..];
    Some(compact(&tail[..tail.find('{')?]))
}

fn has_compact(source: &str, exact: &str) -> bool {
    compact(source).contains(&compact(exact))
}

fn reject_mutation(contract: fn(&str) -> bool, source: &str, from: &str, to: &str) {
    let mutant = source.replacen(from, to, 1);
    assert_ne!(mutant, source, "mutation seam must exist: {from}");
    assert!(!contract(&mutant), "contract accepted mutation: {from}");
}

#[test]
fn plan_pins_exact_geometry_full_context_and_checked_boundaries() {
    let plan = Glm53KdaConvPlan::new(1, 65_520, 1_048_576, 983_056, 1_048_576, 64, 128, 4, 0x1234)
        .unwrap();
    assert_eq!(plan.stream_bytes, 1_073_479_680);
    assert_eq!(plan.weight_bytes, 131_072);
    assert_eq!(plan.state_bytes, 393_216);
    assert_eq!(
        (plan.end_bytes, plan.nonce_bytes, plan.length_bytes),
        (4, 8, 4)
    );

    for hostile in [
        Glm53KdaConvPlan::new(0, 1, 1, 0, 1, 64, 128, 4, 1),
        Glm53KdaConvPlan::new(65_536, 1, 1, 0, 1, 64, 128, 4, 1),
        Glm53KdaConvPlan::new(1, 0, 1, 0, 0, 64, 128, 4, 1),
        Glm53KdaConvPlan::new(1, 65_521, 65_521, 0, 65_521, 64, 128, 4, 1),
        Glm53KdaConvPlan::new(1, 1, 1_048_577, 0, 1, 64, 128, 4, 1),
        Glm53KdaConvPlan::new(1, 1, 1, 0, 1, 63, 128, 4, 1),
        Glm53KdaConvPlan::new(1, 1, 1, 0, 1, 64, 127, 4, 1),
        Glm53KdaConvPlan::new(1, 1, 1, 0, 1, 64, 128, 3, 1),
        Glm53KdaConvPlan::new(1, 1, 1, 0, 1, 64, 128, 4, 0),
        Glm53KdaConvPlan::new(1, 2, 2, 0, 1, 64, 128, 4, 1),
        Glm53KdaConvPlan::new(1, 1, 1, u32::MAX, 0, 64, 128, 4, 1),
    ] {
        assert!(hostile.is_err());
    }
    Glm53KdaConvPlan::new(1, 1, 1_048_576, 1_048_575, 1_048_576, 64, 128, 4, 1).unwrap();
}

#[derive(Clone, Copy)]
enum Mutation {
    None,
    ReverseWeights,
    ResetEachToken,
    ActivateTerms,
    SwapKeyValue,
}

fn oracle(
    inputs: &[Vec<f32>],
    weights: &[Vec<f32>],
    persistent: &[f32],
    channels: usize,
    mutation: Mutation,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let queries = inputs[0].len() / channels;
    let mut outputs = vec![vec![0.0; queries * channels]; 3];
    let mut staged = persistent.to_vec();
    for stream in 0..3 {
        let input_stream = match (mutation, stream) {
            (Mutation::SwapKeyValue, 1) => 2,
            (Mutation::SwapKeyValue, 2) => 1,
            _ => stream,
        };
        for channel in 0..channels {
            let base = (stream * channels + channel) * 4;
            let initial = &persistent[base..base + 4];
            let mut state = [initial[0], initial[1], initial[2], initial[3]];
            for token in 0..queries {
                if matches!(mutation, Mutation::ResetEachToken) {
                    state.copy_from_slice(initial);
                }
                state.rotate_left(1);
                state[3] =
                    bf16::from_f32(inputs[input_stream][token * channels + channel]).to_f32();
                let mut terms = [0.0; 4];
                for tap in 0..4 {
                    let weight_tap = if matches!(mutation, Mutation::ReverseWeights) {
                        3 - tap
                    } else {
                        tap
                    };
                    terms[tap] = state[tap] * weights[stream][channel * 4 + weight_tap];
                }
                let value = if matches!(mutation, Mutation::ActivateTerms) {
                    terms.iter().map(|x| x / (1.0 + (-x).exp())).sum()
                } else {
                    let sum: f32 = terms.iter().sum();
                    sum / (1.0 + (-sum).exp())
                };
                outputs[stream][token * channels + channel] = bf16::from_f32(value).to_f32();
            }
            staged[base..base + 4].copy_from_slice(&state);
        }
    }
    (outputs, staged)
}

#[test]
fn cpu_oracle_discriminates_three_stream_order_carry_weights_and_activation() {
    let inputs = vec![
        vec![5.0, -2.0, 6.0, -3.0, 7.0, -4.0, 8.0, -5.0, 9.0, -6.0],
        vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        vec![
            -1.25, 0.75, -2.25, 1.75, -3.25, 2.75, -4.25, 3.75, -5.25, 4.75,
        ],
    ];
    let weights = vec![
        vec![0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, -0.9],
        vec![0.9, -0.3, 0.6, 0.2, 0.4, 0.1, -0.8, 0.5],
        vec![-0.7, 0.2, 0.8, 0.3, 0.6, -0.4, 0.2, 0.9],
    ];
    let persistent: Vec<f32> = (0..24).map(|i| i as f32 * 0.125 - 1.0).collect();
    let exact = oracle(&inputs, &weights, &persistent, 2, Mutation::None);
    for mutation in [
        Mutation::ReverseWeights,
        Mutation::ResetEachToken,
        Mutation::ActivateTerms,
        Mutation::SwapKeyValue,
    ] {
        assert_ne!(oracle(&inputs, &weights, &persistent, 2, mutation), exact);
    }
    assert_eq!(&exact.1[0..4], &[6.0, 7.0, 8.0, 9.0]);
    assert_eq!(&exact.1[8..12], &[3.5, 5.5, 7.5, 9.5]);
    let prefix_inputs: Vec<Vec<f32>> = inputs.iter().map(|stream| stream[..4].to_vec()).collect();
    let accepted_two = oracle(&prefix_inputs, &weights, &persistent, 2, Mutation::None);
    assert_ne!(accepted_two.1, exact.1);
    assert_eq!(&accepted_two.1[0..4], &[-0.75, -0.625, 5.0, 6.0]);
}

fn commit_reference(
    staged: &[f32],
    persistent: &mut [f32],
    ends: &mut [u32],
    nonces: &mut [u64],
    lengths: &mut [u32],
    start: u32,
    end: u32,
    nonce: u64,
) -> bool {
    if ends.iter().any(|&x| x != end)
        || nonces.iter().any(|&x| x != nonce)
        || lengths.iter().any(|&x| x != start)
    {
        return false;
    }
    persistent.copy_from_slice(staged);
    lengths.fill(end);
    ends.fill(0);
    nonces.fill(0);
    true
}

#[test]
fn transaction_prevalidates_every_sequence_before_any_state_publication() {
    let staged: Vec<f32> = (0..96).map(|x| x as f32).collect();
    let original = vec![-1.0; staged.len()];
    for fault in 0..3 {
        let mut persistent = original.clone();
        let mut ends = vec![8, 8];
        let mut nonces = vec![44, 44];
        let mut lengths = vec![3, 3];
        match fault {
            0 => ends[1] = 7,
            1 => nonces[1] = 45,
            _ => lengths[1] = 4,
        }
        let before = (
            persistent.clone(),
            ends.clone(),
            nonces.clone(),
            lengths.clone(),
        );
        assert!(!commit_reference(
            &staged,
            &mut persistent,
            &mut ends,
            &mut nonces,
            &mut lengths,
            3,
            8,
            44,
        ));
        assert_eq!((persistent, ends, nonces, lengths), before);
    }
    let mut persistent = original;
    let mut ends = vec![8, 8];
    let mut nonces = vec![44, 44];
    let mut lengths = vec![3, 3];
    assert!(commit_reference(
        &staged,
        &mut persistent,
        &mut ends,
        &mut nonces,
        &mut lengths,
        3,
        8,
        44,
    ));
    assert_eq!(persistent, staged);
    assert_eq!(
        (ends, nonces, lengths),
        (vec![0, 0], vec![0, 0], vec![8, 8])
    );
}

fn buffer(gpu: &MockGpuBackend, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: gpu.alloc(bytes).unwrap(),
        bytes,
    }
}

fn buffers(gpu: &MockGpuBackend, plan: Glm53KdaConvPlan) -> Glm53KdaConvBuffers {
    Glm53KdaConvBuffers {
        q_input_bf16: buffer(gpu, plan.stream_bytes),
        k_input_bf16: buffer(gpu, plan.stream_bytes),
        v_input_bf16: buffer(gpu, plan.stream_bytes),
        q_weight_f32: buffer(gpu, plan.weight_bytes),
        k_weight_f32: buffer(gpu, plan.weight_bytes),
        v_weight_f32: buffer(gpu, plan.weight_bytes),
        persistent_state_f32: buffer(gpu, plan.state_bytes),
        staged_state_f32: buffer(gpu, plan.state_bytes),
        q_output_bf16: buffer(gpu, plan.stream_bytes),
        k_output_bf16: buffer(gpu, plan.stream_bytes),
        v_output_bf16: buffer(gpu, plan.stream_bytes),
        published_ends_u32: buffer(gpu, plan.end_bytes),
        published_nonces_u64: buffer(gpu, plan.nonce_bytes),
        logical_lengths_u32: buffer(gpu, plan.length_bytes),
    }
}

#[test]
fn host_rejects_forgery_alias_alignment_null_and_overflow_before_effects() {
    let gpu = MockGpuBackend::new();
    let kernel = Glm53KdaConvKernel::load(&gpu).unwrap();
    let plan = Glm53KdaConvPlan::new(1, 1, 8, 0, 1, 64, 128, 4, 9).unwrap();
    let valid = buffers(&gpu, plan);
    let mut forged = plan;
    forged.state_bytes += 4;
    assert!(kernel.launch_stage(&gpu, forged, valid, 0).is_err());
    assert!(kernel.launch_commit(&gpu, plan, 2, valid, 0).is_err());
    for hostile in [
        Glm53KdaConvBuffers {
            k_output_bf16: GgmlIqBuffer {
                ptr: valid.q_output_bf16.ptr,
                ..valid.k_output_bf16
            },
            ..valid
        },
        Glm53KdaConvBuffers {
            q_weight_f32: GgmlIqBuffer {
                ptr: DevicePtr(valid.q_weight_f32.ptr.0 + 2),
                ..valid.q_weight_f32
            },
            ..valid
        },
        Glm53KdaConvBuffers {
            v_input_bf16: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                ..valid.v_input_bf16
            },
            ..valid
        },
        Glm53KdaConvBuffers {
            q_input_bf16: GgmlIqBuffer {
                ptr: DevicePtr(u64::MAX - 1),
                ..valid.q_input_bf16
            },
            ..valid
        },
    ] {
        assert!(kernel.launch_stage(&gpu, plan, hostile, 0).is_err());
    }
    assert_eq!(gpu.launch_count(), 0);
    kernel.launch_stage(&gpu, plan, valid, 0).unwrap();
    assert_eq!(gpu.launch_count(), 2);
    kernel.launch_commit(&gpu, plan, 1, valid, 0).unwrap();
    assert_eq!(gpu.launch_count(), 3);
}

#[rustfmt::skip]
fn host_contract(source: &str) -> bool {
    let stage = section(source, "pub fn launch_stage", "pub fn launch_commit");
    stage.matches(".arg_").count() == 22 && source[source.find("pub fn launch_commit").unwrap()..].matches(".arg_").count() == 15 && ordered(stage, &[".arg_ptr(buffers.q_input_bf16.ptr)", ".arg_ptr(buffers.k_input_bf16.ptr)", ".arg_ptr(buffers.v_input_bf16.ptr)", ".arg_ptr(buffers.q_weight_f32.ptr)", ".arg_ptr(buffers.k_weight_f32.ptr)", ".arg_ptr(buffers.v_weight_f32.ptr)", ".arg_ptr(buffers.persistent_state_f32.ptr)", ".arg_ptr(buffers.staged_state_f32.ptr)", ".arg_ptr(buffers.q_output_bf16.ptr)", ".arg_ptr(buffers.k_output_bf16.ptr)", ".arg_ptr(buffers.v_output_bf16.ptr)", ".arg_u32(plan.batch)", ".arg_u32(plan.queries)", ".launch(stream)?;", ".arg_ptr(buffers.published_ends_u32.ptr)", ".arg_ptr(buffers.published_nonces_u64.ptr)", ".arg_ptr(buffers.logical_lengths_u32.ptr)", ".arg_u32(plan.batch)", ".arg_u32(plan.queries)", ".arg_u32(plan.capacity)", ".arg_u32(plan.start_position)", ".arg_u32(plan.end_position)", ".arg_u64(plan.transaction_nonce)"])
        && ordered(&source[source.find("pub fn launch_commit").unwrap()..], &[".arg_ptr(buffers.q_input_bf16.ptr)", ".arg_ptr(buffers.k_input_bf16.ptr)", ".arg_ptr(buffers.v_input_bf16.ptr)", ".arg_ptr(buffers.staged_state_f32.ptr)", ".arg_ptr(buffers.persistent_state_f32.ptr)", ".arg_ptr(buffers.published_ends_u32.ptr)", ".arg_ptr(buffers.published_nonces_u64.ptr)", ".arg_ptr(buffers.logical_lengths_u32.ptr)", ".arg_u32(plan.batch)", ".arg_u32(plan.queries)", ".arg_u32(accepted)", ".arg_u32(plan.capacity)", ".arg_u32(plan.start_position)", ".arg_u32(plan.end_position)", ".arg_u64(plan.transaction_nonce)"])
}

fn cuda_contract(source: &str) -> bool {
    const STAGE_SIG: &str = r#"atlas_glm53_kda_conv_f32_stage(const __nv_bfloat16 *__restrict__ q_input,const __nv_bfloat16 *__restrict__ k_input,const __nv_bfloat16 *__restrict__ v_input,const float *__restrict__ q_weight,const float *__restrict__ k_weight,const float *__restrict__ v_weight,const float *__restrict__ persistent_state,float *__restrict__ staged_state,__nv_bfloat16 *__restrict__ q_output,__nv_bfloat16 *__restrict__ k_output,__nv_bfloat16 *__restrict__ v_output,unsigned int batch,unsigned int query_count)"#;
    const FINALIZE_SIG: &str = r#"atlas_glm53_kda_conv_finalize(unsigned int *__restrict__ published_ends,unsigned long long *__restrict__ published_nonces,const unsigned int *__restrict__ logical_lengths,unsigned int batch,unsigned int query_count,unsigned int capacity,unsigned int start_position,unsigned int end_position,unsigned long long transaction_nonce)"#;
    const COMMIT_SIG: &str = r#"atlas_glm53_kda_conv_commit(const __nv_bfloat16 *__restrict__ q_input,const __nv_bfloat16 *__restrict__ k_input,const __nv_bfloat16 *__restrict__ v_input,const float *__restrict__ staged_state,float *__restrict__ persistent_state,unsigned int *__restrict__ published_ends,unsigned long long *__restrict__ published_nonces,unsigned int *__restrict__ logical_lengths,unsigned int batch,unsigned int query_count,unsigned int accepted_count,unsigned int capacity,unsigned int start_position,unsigned int end_position,unsigned long long transaction_nonce)"#;
    let stage = section(
        source,
        "atlas_glm53_kda_conv_f32_stage",
        "atlas_glm53_kda_conv_finalize",
    );
    let finalize = section(
        source,
        "atlas_glm53_kda_conv_finalize",
        "atlas_glm53_kda_conv_commit",
    );
    let commit = &source[source.find("atlas_glm53_kda_conv_commit").unwrap()..];
    signature(source, "atlas_glm53_kda_conv_f32_stage") == Some(compact(STAGE_SIG))
        && signature(source, "atlas_glm53_kda_conv_finalize") == Some(compact(FINALIZE_SIG))
        && signature(source, "atlas_glm53_kda_conv_commit") == Some(compact(COMMIT_SIG))
        && has_compact(
            stage,
            "const __nv_bfloat16 *input=stream_index==0U?q_input:(stream_index==1U?k_input:v_input);const float *weight=stream_index==0U?q_weight:(stream_index==1U?k_weight:v_weight);__nv_bfloat16 *output=stream_index==0U?q_output:(stream_index==1U?k_output:v_output);",
        )
        && has_compact(
            stage,
            "(((unsigned long long)batch_index*GLM53_KDA_CONV_STREAMS+stream_index)*GLM53_KDA_CONV_CHANNELS+channel)*GLM53_KDA_CONV_KERNEL",
        )
        && has_compact(stage, "(unsigned long long)channel*GLM53_KDA_CONV_KERNEL")
        && has_compact(
            stage,
            "((unsigned long long)batch_index*query_count+token)*GLM53_KDA_CONV_CHANNELS+channel",
        )
        && ordered(
            stage,
            &[
                "s0 = s1;",
                "s1 = s2;",
                "s2 = s3;",
                "s3 = newest;",
                "s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3",
                "output[token_index]",
                "staged_state[state_base + 0ULL]",
            ],
        )
        && finalize.matches("__syncthreads();").count() == 4
        && finalize.matches("__threadfence();").count() == 1
        && has_compact(
            finalize,
            "logical_lengths[b]!=start_position){atomicExch(&invalid_length,1U);}}__syncthreads();if(invalid_length!=0U){return;}",
        )
        && has_compact(
            finalize,
            "published_ends[b]=end_position;}__syncthreads();__threadfence();__syncthreads();for(unsigned long long b=lane;b<batch;b+=GLM53_KDA_CONV_THREADS){atomicExch(published_nonces+b,transaction_nonce);",
        )
        && has_compact(
            commit,
            "const unsigned long long stream_batch=item/GLM53_KDA_CONV_CHANNELS;const unsigned int stream_index=stream_batch%GLM53_KDA_CONV_STREAMS;const unsigned long long batch_index=stream_batch/GLM53_KDA_CONV_STREAMS;const __nv_bfloat16 *input=stream_index==0U?q_input:(stream_index==1U?k_input:v_input);const unsigned long long state_base=item*GLM53_KDA_CONV_KERNEL;",
        )
        && has_compact(
            commit,
            "(batch_index*query_count+token)*GLM53_KDA_CONV_CHANNELS+channel",
        )
        && commit.matches("__syncthreads();").count() == 8
        && commit.matches("__threadfence();").count() == 3
        && has_compact(
            commit,
            "logical_lengths[b]!=start_position){atomicExch(&invalid_transaction,1U);}}__syncthreads();if(invalid_transaction!=0U){return;}",
        )
        && has_compact(
            commit,
            "}__syncthreads();__threadfence();__syncthreads();for(unsigned long long b=lane;b<batch;b+=GLM53_KDA_CONV_THREADS){logical_lengths[b]=start_position+accepted_count;}__syncthreads();__threadfence();__syncthreads();for(unsigned long long b=lane;b<batch;b+=GLM53_KDA_CONV_THREADS){published_ends[b]=0U;}__syncthreads();__threadfence();__syncthreads();for(unsigned long long b=lane;b<batch;b+=GLM53_KDA_CONV_THREADS){atomicExch(published_nonces+b,0ULL);",
        )
}

#[test]
#[rustfmt::skip]
fn exact_abi_address_and_barrier_contract_rejects_compile_plausible_mutants() {
    assert!(host_contract(HOST));
    assert!(cuda_contract(CUDA));
    for (from, to) in [
        (
            ".arg_u32(plan.batch)\n            .arg_u32(plan.queries)",
            ".arg_u32(plan.queries)\n            .arg_u32(plan.batch)",
        ),
        (
            ".arg_ptr(buffers.published_ends_u32.ptr)\n            .arg_ptr(buffers.published_nonces_u64.ptr)",
            ".arg_ptr(buffers.published_nonces_u64.ptr)\n            .arg_ptr(buffers.published_ends_u32.ptr)",
        ),
        (
            ".arg_u32(plan.capacity)\n            .arg_u32(plan.start_position)",
            ".arg_u32(plan.start_position)\n            .arg_u32(plan.capacity)",
        ),
        (".arg_ptr(buffers.q_input_bf16.ptr)\n            .arg_ptr(buffers.k_input_bf16.ptr)", ".arg_ptr(buffers.q_input_bf16.ptr)\n            .arg_ptr(buffers.q_input_bf16.ptr)\n            .arg_ptr(buffers.k_input_bf16.ptr)"),
    ] {
        reject_mutation(host_contract, HOST, from, to);
    }
    reject_mutation(
        host_contract,
        HOST,
        ".arg_u32(accepted)\n            .arg_u32(plan.capacity)\n            .arg_u32(plan.start_position)",
        ".arg_u32(accepted)\n            .arg_u32(plan.start_position)\n            .arg_u32(plan.capacity)",
    );
    for (from, to) in [
        (
            "unsigned int batch, unsigned int query_count)",
            "unsigned int query_count, unsigned int batch)",
        ),
        (
            "unsigned int capacity, unsigned int start_position,",
            "unsigned int start_position, unsigned int capacity,",
        ),
        (
            "stream_index == 0U ? q_input",
            "stream_index == 0U ? k_input",
        ),
        (
            "stream_index == 0U ? q_weight",
            "stream_index == 0U ? k_weight",
        ),
        (
            "batch_index * query_count + token",
            "token * query_count + batch_index",
        ),
        (
            "__syncthreads();\n    if (invalid_length != 0U)",
            "if (invalid_length != 0U)",
        ),
        (
            "__syncthreads();\n    if (invalid_transaction != 0U)",
            "if (invalid_transaction != 0U)",
        ),
        (
            "__syncthreads();\n    __threadfence();\n    __syncthreads();",
            "__threadfence();\n    __syncthreads();",
        ),
        ("item * GLM53_KDA_CONV_KERNEL", "stream_batch * GLM53_KDA_CONV_KERNEL"),
        ("stream_batch % GLM53_KDA_CONV_STREAMS", "stream_batch % 2U"),
        ("stream_batch / GLM53_KDA_CONV_STREAMS", "stream_batch / GLM53_KDA_CONV_CHANNELS"),
    ] {
        reject_mutation(cuda_contract, CUDA, from, to);
    }
    reject_mutation(
        cuda_contract,
        CUDA,
        "channel * GLM53_KDA_CONV_KERNEL",
        "stream_index * GLM53_KDA_CONV_KERNEL",
    );
    reject_mutation(
        cuda_contract,
        CUDA,
        "unsigned int accepted_count,\n        unsigned int capacity, unsigned int start_position,",
        "unsigned int accepted_count,\n        unsigned int start_position, unsigned int capacity,",
    );
    reject_mutation(
        cuda_contract,
        CUDA,
        "persistent_state[state_base + 3ULL] = s3;\n        }\n    }\n    __syncthreads();",
        "persistent_state[state_base + 3ULL] = s3;\n        }\n    }",
    );
    reject_mutation(
        cuda_contract,
        CUDA,
        "__threadfence();",
        "/* missing fence */",
    );
}
