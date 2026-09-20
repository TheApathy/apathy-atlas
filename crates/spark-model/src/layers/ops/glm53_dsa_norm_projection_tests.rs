// SPDX-License-Identifier: AGPL-3.0-only

use half::bf16;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

const HOST: &str = include_str!("glm53_dsa_norm_projection.rs");
const CUDA: &str =
    include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/glm53_dsa_norm_projection.cu");

fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn absolute_rms(input: &[f32], weights: &[f32], eps: f32) -> Vec<f32> {
    let input = input.iter().copied().map(round).collect::<Vec<_>>();
    let inverse = (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32 + eps)
        .sqrt()
        .recip();
    input
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| round(round(value * inverse) * round(weight)))
        .collect()
}

fn biased_layer_norm(input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
    let input = input.iter().copied().map(round).collect::<Vec<_>>();
    let mean = input.iter().sum::<f32>() / input.len() as f32;
    let variance = input
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / input.len() as f32;
    let inverse = (variance + 1.0e-6).sqrt().recip();
    input
        .iter()
        .zip(weights)
        .zip(bias)
        .map(|((&value, &weight), &bias)| round((value - mean) * inverse * weight + bias))
        .collect()
}

fn index_projection(input: &[f32], weights: &[f32]) -> Vec<f32> {
    (0..32)
        .map(|head| {
            let mut sum = 0.0f32;
            for column in 0..4096 {
                sum = weights[head * 4096 + column].mul_add(round(input[column]), sum);
            }
            round(sum)
        })
        .collect()
}

fn at(ptr: u64, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer {
        ptr: DevicePtr(ptr),
        bytes,
    }
}

fn buffers(plan: Glm53DsaNormProjectionPlan) -> Glm53DsaNormProjectionBuffers {
    Glm53DsaNormProjectionBuffers {
        input_bf16: at(0x10_0000, plan.input_bytes),
        weight_f32: at(0x3000_0000, plan.weight_bytes),
        bias_f32: if plan.bias_bytes == 0 {
            at(0, 0)
        } else {
            at(0x4000_0000, plan.bias_bytes)
        },
        output_bf16: at(0x5000_0000, plan.output_bytes),
    }
}

#[test]
fn exact_variants_extents_limits_and_forgery_are_closed() {
    let r512 = Glm53DsaNormProjectionPlan::new(
        Glm53DsaNormProjectionKind::AbsoluteRms512,
        65_520,
        512,
        512,
        1.0e-5,
    )
    .unwrap();
    assert_eq!(r512.input_bytes, 65_520 * 512 * 2);
    assert_eq!(r512.weight_bytes, 512 * 4);
    assert_eq!(r512.bias_bytes, 0);
    let r1536 = Glm53DsaNormProjectionPlan::new(
        Glm53DsaNormProjectionKind::AbsoluteRms1536,
        1,
        1536,
        1536,
        1.0e-5,
    )
    .unwrap();
    assert_eq!((r1536.input_bytes, r1536.output_bytes), (3072, 3072));
    let layer = Glm53DsaNormProjectionPlan::new(
        Glm53DsaNormProjectionKind::BiasedLayerNorm128,
        1,
        128,
        128,
        1.0e-6,
    )
    .unwrap();
    assert_eq!((layer.weight_bytes, layer.bias_bytes), (512, 512));
    let projection = Glm53DsaNormProjectionPlan::new(
        Glm53DsaNormProjectionKind::F32IndexProjection,
        65_520,
        4096,
        32,
        0.0,
    )
    .unwrap();
    assert_eq!(projection.input_bytes, 536_739_840);
    assert_eq!(projection.weight_bytes, 524_288);
    assert_eq!(projection.output_bytes, 4_193_280);

    assert!(Glm53DsaNormProjectionPlan::new(r512.kind, 0, 512, 512, 1.0e-5).is_err());
    assert!(Glm53DsaNormProjectionPlan::new(r512.kind, 65_521, 512, 512, 1.0e-5).is_err());
    assert!(Glm53DsaNormProjectionPlan::new(r512.kind, 1, 513, 512, 1.0e-5).is_err());
    assert!(Glm53DsaNormProjectionPlan::new(r512.kind, 1, 512, 512, 1.0e-6).is_err());
    assert!(Glm53DsaNormProjectionPlan::new(projection.kind, 1, 4096, 31, 0.0).is_err());
    let mut forged = layer;
    forged.threads = 256;
    assert!(forged.validate().is_err());
}

#[test]
fn cpu_oracles_pin_absolute_rms_bias_and_output_major_projection() {
    let rms = absolute_rms(&[1.0, -2.0, 3.0, -4.0], &[2.0, 2.0, 2.0, 2.0], 1.0e-5);
    let shifted = absolute_rms(&[1.0, -2.0, 3.0, -4.0], &[3.0, 3.0, 3.0, 3.0], 1.0e-5);
    assert_ne!(rms, shifted, "absolute scale must not become 1+weight");

    let layer = biased_layer_norm(
        &[1.0, 2.0, 5.0, 8.0],
        &[1.0, 2.0, -1.0, 0.5],
        &[0.25, -0.5, 1.0, 2.0],
    );
    let no_bias = biased_layer_norm(&[1.0, 2.0, 5.0, 8.0], &[1.0, 2.0, -1.0, 0.5], &[0.0; 4]);
    assert_ne!(layer, no_bias);

    let mut input = vec![0.0; 4096];
    input[0] = 1.0;
    input[4095] = 2.0;
    let mut weight = vec![0.0; 4096 * 32];
    weight[0] = 3.0;
    weight[4095] = 5.0;
    weight[4096] = 7.0;
    let projected = index_projection(&input, &weight);
    assert_eq!((projected[0], projected[1]), (13.0, 7.0));
    assert!(projected[2..].iter().all(|&value| value == 0.0));
}

#[test]
fn every_pointer_is_checked_before_any_enqueue() {
    let gpu = MockGpuBackend::new();
    let kernels = Glm53DsaNormProjectionKernels::load(&gpu).unwrap();
    let plans = [
        Glm53DsaNormProjectionPlan::new(
            Glm53DsaNormProjectionKind::AbsoluteRms512,
            1,
            512,
            512,
            1.0e-5,
        )
        .unwrap(),
        Glm53DsaNormProjectionPlan::new(
            Glm53DsaNormProjectionKind::AbsoluteRms1536,
            1,
            1536,
            1536,
            1.0e-5,
        )
        .unwrap(),
        Glm53DsaNormProjectionPlan::new(
            Glm53DsaNormProjectionKind::BiasedLayerNorm128,
            1,
            128,
            128,
            1.0e-6,
        )
        .unwrap(),
        Glm53DsaNormProjectionPlan::new(
            Glm53DsaNormProjectionKind::F32IndexProjection,
            1,
            4096,
            32,
            0.0,
        )
        .unwrap(),
    ];
    let plan = plans[2];
    let valid = buffers(plan);
    for invalid in [
        Glm53DsaNormProjectionBuffers {
            input_bf16: at(0, plan.input_bytes),
            ..valid
        },
        Glm53DsaNormProjectionBuffers {
            weight_f32: at(valid.weight_f32.ptr.0 + 2, plan.weight_bytes),
            ..valid
        },
        Glm53DsaNormProjectionBuffers {
            bias_f32: at(valid.bias_f32.ptr.0, plan.bias_bytes - 1),
            ..valid
        },
        Glm53DsaNormProjectionBuffers {
            output_bf16: at(valid.input_bf16.ptr.0, plan.output_bytes),
            ..valid
        },
        Glm53DsaNormProjectionBuffers {
            input_bf16: at(u64::MAX - 1, plan.input_bytes),
            ..valid
        },
    ] {
        assert!(kernels.launch(&gpu, plan, invalid, 0).is_err());
        assert_eq!(gpu.launch_count(), 0);
    }
    for plan in plans {
        kernels.launch(&gpu, plan, buffers(plan), 0).unwrap();
    }
    assert_eq!(gpu.launch_count(), 4);
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|byte| !byte.is_whitespace())
        .collect()
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source.find(start).unwrap();
    let end = source[start..].find(end).unwrap() + start;
    &source[start..end]
}

fn exact_launch(section: &str, chain: &str, arguments: usize) -> bool {
    let section = compact(section);
    section.contains(chain)
        && section.matches(".grid(").count() == 1
        && section.matches(".block(").count() == 1
        && section.matches(".arg_").count() == arguments
}

fn host_contract(source: &str) -> bool {
    let launches = &source[source.find("let launch = match plan.kind").unwrap()..];
    let rms = section(
        launches,
        "Glm53DsaNormProjectionKind::AbsoluteRms512",
        "Glm53DsaNormProjectionKind::BiasedLayerNorm128",
    );
    let layer = section(
        launches,
        "Glm53DsaNormProjectionKind::BiasedLayerNorm128",
        "Glm53DsaNormProjectionKind::F32IndexProjection",
    );
    let projection = section(
        launches,
        "Glm53DsaNormProjectionKind::F32IndexProjection",
        "launch.launch(stream)",
    );
    source.contains("plan.validate()?;\n        validate_buffers(plan, buffers)?;")
        && exact_launch(
            rms,
            "KernelLaunch::new(gpu,self.rms_norm).grid([plan.rows,1,1]).block([plan.threads,1,1]).arg_ptr(buffers.input_bf16.ptr).arg_ptr(buffers.weight_f32.ptr).arg_ptr(buffers.output_bf16.ptr).arg_u32(plan.rows).arg_u32(plan.input_width).arg_f32(f32::from_bits(plan.epsilon_bits))",
            6,
        )
        && exact_launch(
            layer,
            "KernelLaunch::new(gpu,self.layer_norm).grid([plan.rows,1,1]).block([plan.threads,1,1]).arg_ptr(buffers.input_bf16.ptr).arg_ptr(buffers.weight_f32.ptr).arg_ptr(buffers.bias_f32.ptr).arg_ptr(buffers.output_bf16.ptr).arg_u32(plan.rows).arg_u32(plan.input_width).arg_f32(f32::from_bits(plan.epsilon_bits))",
            7,
        )
        && exact_launch(
            projection,
            "KernelLaunch::new(gpu,kernel).grid([plan.rows,1,1]).block([plan.threads,1,1]).arg_ptr(buffers.input_bf16.ptr).arg_ptr(buffers.weight_f32.ptr).arg_ptr(buffers.output_bf16.ptr).arg_u32(plan.rows).arg_u32(plan.input_width).arg_u32(plan.output_width)",
            6,
        )
        // The projection arm no longer names its kernel inline: the
        // `ATLAS_GLM53_DSA_INDEX_PROJ_TILED` lever picks between the donor
        // kernel and the shared-memory-staged one. Same ABI either way, so the
        // pin moves to the selection: exactly one alternative, the donor as the
        // default/else arm, and no second launch built from it.
        && compact(projection).contains("}else{self.index_projection};")
        && compact(projection).matches("gpu.kernel(").count() == 1
        && compact(projection)
            .contains("gpu.kernel(\"glm53_prompt_glue\",\"atlas_glm53_dsa_index_projection_tiled\",)?")
}

fn cuda_contract(source: &str) -> bool {
    let rms = compact(section(
        source,
        "atlas_glm53_dsa_absolute_rms_norm_bf16(",
        "atlas_glm53_dsa_biased_layer_norm_bf16(",
    ));
    let layer = compact(section(
        source,
        "atlas_glm53_dsa_biased_layer_norm_bf16(",
        "atlas_glm53_dsa_index_projection_f32_bf16(",
    ));
    let projection = compact(
        &source[source
            .find("atlas_glm53_dsa_index_projection_f32_bf16(")
            .unwrap()..],
    );
    let rms_guard = "if(blockDim.x!=256U||blockDim.y!=1U||blockDim.z!=1U||gridDim.x!=rows||gridDim.y!=1U||gridDim.z!=1U||rows==0U||rows>GLM53_MAX_ROWS||(width!=GLM53_KV_RANK&&width!=GLM53_Q_RANK)||eps!=1.0e-5f){return;}";
    let layer_guard = "if(blockDim.x!=GLM53_INDEX_DIM||blockDim.y!=1U||blockDim.z!=1U||gridDim.x!=rows||gridDim.y!=1U||gridDim.z!=1U||rows==0U||rows>GLM53_MAX_ROWS||width!=GLM53_INDEX_DIM||eps!=1.0e-6f){return;}";
    let projection_guard = "if(blockDim.x!=GLM53_INDEX_HEADS||blockDim.y!=1U||blockDim.z!=1U||gridDim.x!=rows||gridDim.y!=1U||gridDim.z!=1U||rows==0U||rows>GLM53_MAX_ROWS||inner!=GLM53_HIDDEN||heads!=GLM53_INDEX_HEADS){return;}";
    rms.contains("const__nv_bfloat16*__restrict__input,constfloat*__restrict__weight,__nv_bfloat16*__restrict__output,unsignedintrows,unsignedintwidth,floateps)")
        && rms.contains(rms_guard)
        && rms.find(rms_guard) < rms.find("constunsignedintrow=blockIdx.x")
        && rms.contains("constunsignedlonglongbase=(unsignedlonglong)row*width;")
        && rms.contains("constunsignedinttid=threadIdx.x;")
        && rms.contains("input[base+column]")
        && rms.contains("output[base+column]")
        && rms.contains("square_sum=fmaf(value,value,square_sum);")
        && layer.contains("const__nv_bfloat16*__restrict__input,constfloat*__restrict__weight,constfloat*__restrict__bias,__nv_bfloat16*__restrict__output,unsignedintrows,unsignedintwidth,floateps)")
        && layer.contains(layer_guard)
        && layer.find(layer_guard) < layer.find("constunsignedintrow=blockIdx.x")
        && layer.contains("constunsignedlonglongbase=(unsignedlonglong)row*GLM53_INDEX_DIM;")
        && layer.contains("constunsignedintcolumn=threadIdx.x;")
        && layer.contains("input[base+column]")
        && layer.contains("output[base+column]")
        && layer.contains("fmaf(normalized,weight[column],bias[column])")
        && projection.contains("const__nv_bfloat16*__restrict__input,constfloat*__restrict__weight,__nv_bfloat16*__restrict__output,unsignedintrows,unsignedintinner,unsignedintheads)")
        && projection.contains(projection_guard)
        && projection.find(projection_guard) < projection.find("constunsignedintrow=blockIdx.x")
        && projection.contains("constunsignedlonglonginput_base=(unsignedlonglong)row*GLM53_HIDDEN;")
        && projection.contains("constunsignedinthead=threadIdx.x;")
        && projection.contains("input[input_base+column]")
        && projection.contains("weight[(unsignedlonglong)head*GLM53_HIDDEN+column]")
        && projection.contains("output[(unsignedlonglong)row*GLM53_INDEX_HEADS+head]")
        && projection.contains("sum=fmaf(weight_value,input_value,sum);")
}

fn mutation(source: &str, from: &str, to: &str) -> String {
    assert_eq!(
        source.matches(from).count(),
        1,
        "ambiguous mutation: {from}"
    );
    source.replacen(from, to, 1)
}

#[test]
fn source_contract_rejects_full_host_abi_swaps_and_duplicates() {
    assert!(host_contract(HOST));
    for (from, to) in [
        (
            ".grid([plan.rows, 1, 1])\n                .block([plan.threads, 1, 1])",
            ".block([plan.threads, 1, 1])\n                .grid([plan.rows, 1, 1])",
        ),
        (
            ".arg_ptr(buffers.bias_f32.ptr)\n                    .arg_ptr(buffers.output_bf16.ptr)",
            ".arg_ptr(buffers.output_bf16.ptr)\n                    .arg_ptr(buffers.bias_f32.ptr)",
        ),
        (
            ".arg_u32(plan.rows)\n                    .arg_u32(plan.input_width)\n                    .arg_f32",
            ".arg_u32(plan.input_width)\n                    .arg_u32(plan.rows)\n                    .arg_f32",
        ),
        (
            ".arg_u32(plan.rows)\n                    .arg_u32(plan.input_width)\n                    .arg_u32(plan.output_width)",
            ".arg_u32(plan.rows)\n                    .arg_u32(plan.output_width)\n                    .arg_u32(plan.input_width)",
        ),
        (
            ".arg_ptr(buffers.bias_f32.ptr)\n                    .arg_ptr(buffers.output_bf16.ptr)",
            ".arg_ptr(buffers.bias_f32.ptr)\n                    .arg_ptr(buffers.bias_f32.ptr)\n                    .arg_ptr(buffers.output_bf16.ptr)",
        ),
        (".arg_ptr(buffers.bias_f32.ptr)\n", ""),
        (
            ".arg_u32(plan.rows)\n                .arg_u32(plan.input_width)\n                .arg_f32",
            ".arg_u32(plan.input_width)\n                .arg_u32(plan.rows)\n                .arg_f32",
        ),
        (
            "KernelLaunch::new(gpu, self.rms_norm)",
            "KernelLaunch::new(gpu, self.layer_norm)",
        ),
        (
            "KernelLaunch::new(gpu, self.layer_norm)",
            "KernelLaunch::new(gpu, kernel)",
        ),
        // The tiled lever must not be able to swap in another kind's kernel,
        // and the donor must stay the default arm.
        (
            "} else {\n                    self.index_projection\n                };",
            "} else {\n                    self.layer_norm\n                };",
        ),
        (
            "\"atlas_glm53_dsa_index_projection_tiled\",",
            "\"atlas_glm53_dsa_biased_layer_norm_bf16\",",
        ),
    ] {
        assert!(!host_contract(&mutation(HOST, from, to)));
    }
    let rms_pointer_swap = HOST.replacen(
        ".arg_ptr(buffers.input_bf16.ptr)\n                .arg_ptr(buffers.weight_f32.ptr)\n                .arg_ptr(buffers.output_bf16.ptr)",
        ".arg_ptr(buffers.weight_f32.ptr)\n                .arg_ptr(buffers.input_bf16.ptr)\n                .arg_ptr(buffers.output_bf16.ptr)",
        1,
    );
    assert!(!host_contract(&rms_pointer_swap));
    for (kernel, indentation) in [
        ("self.layer_norm", "                    "),
        // The projection arm launches through the `kernel` binding the tiled
        // lever selects, not the field directly.
        ("kernel", "                    "),
    ] {
        let from = format!(
            "KernelLaunch::new(gpu, {kernel})\n{indentation}.grid([plan.rows, 1, 1])\n{indentation}.block([plan.threads, 1, 1])"
        );
        let to = format!(
            "KernelLaunch::new(gpu, {kernel})\n{indentation}.block([plan.threads, 1, 1])\n{indentation}.grid([plan.rows, 1, 1])"
        );
        assert!(!host_contract(&mutation(HOST, &from, &to)));
    }
    assert!(!host_contract(&mutation(
        HOST,
        ".arg_ptr(buffers.input_bf16.ptr)\n                    .arg_ptr(buffers.weight_f32.ptr)\n                    .arg_ptr(buffers.output_bf16.ptr)\n                    .arg_u32(plan.rows)\n                    .arg_u32(plan.input_width)\n                    .arg_u32(plan.output_width)",
        ".arg_ptr(buffers.output_bf16.ptr)\n                    .arg_ptr(buffers.weight_f32.ptr)\n                    .arg_ptr(buffers.input_bf16.ptr)\n                    .arg_u32(plan.rows)\n                    .arg_u32(plan.input_width)\n                    .arg_u32(plan.output_width)",
    )));
}

#[test]
fn source_contract_rejects_launch_guard_and_address_mutants() {
    assert!(cuda_contract(CUDA));
    for (from, to) in [
        ("blockDim.x != 256U", "blockDim.x != 128U"),
        ("blockDim.x != GLM53_INDEX_DIM", "blockDim.x != 256U"),
        ("blockDim.x != GLM53_INDEX_HEADS", "blockDim.x != 64U"),
        ("gridDim.x != rows", "gridDim.x == rows"),
        (
            "gridDim.y != 1U || gridDim.z != 1U",
            "gridDim.z != 1U || gridDim.y != 1U",
        ),
        ("gridDim.y != 1U || ", ""),
        ("row * width", "row * GLM53_KV_RANK"),
        ("row * GLM53_INDEX_DIM", "column * GLM53_INDEX_DIM"),
        ("row * GLM53_HIDDEN", "head * GLM53_HIDDEN"),
        (
            "head * GLM53_HIDDEN + column",
            "column * GLM53_INDEX_HEADS + head",
        ),
        ("row * GLM53_INDEX_HEADS + head", "head * rows + row"),
        (
            "const unsigned int tid = threadIdx.x;",
            "const unsigned int tid = threadIdx.y;",
        ),
        (
            "const unsigned int tid = threadIdx.x;",
            "const unsigned int tid = threadIdx.z;",
        ),
        (
            "const unsigned int column = threadIdx.x;",
            "const unsigned int column = threadIdx.y;",
        ),
        (
            "const unsigned int column = threadIdx.x;",
            "const unsigned int column = threadIdx.z;",
        ),
        (
            "const unsigned int head = threadIdx.x;",
            "const unsigned int head = threadIdx.y;",
        ),
        (
            "const unsigned int head = threadIdx.x;",
            "const unsigned int head = threadIdx.z;",
        ),
    ] {
        let mutated = CUDA.replacen(from, to, 1);
        assert_ne!(mutated, CUDA, "missing mutation source: {from}");
        assert!(!cuda_contract(&mutated), "false accept: {from}");
    }
}
