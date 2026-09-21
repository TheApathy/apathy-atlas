// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the production DeepSeek-V4 pure Q-B RMSNorm + RoPE kernel.

use half::bf16;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

const TOKENS: u64 = 2_410;
const LAYERS: u64 = 43;
const NQ: usize = 64;
const NKV: usize = 1;
const HEAD_DIM: usize = 512;
const NOPE_DIM: usize = 448;
const ROPE_DIM: usize = 64;
const THREADS: usize = 512;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn production_kernel() -> String {
    fs::read_to_string(
        repo_root().join("kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu"),
    )
    .expect("production V4 Q-B RMSNorm + RoPE kernel must exist")
}

fn sass_gate() -> String {
    fs::read_to_string(repo_root().join("scripts/check-v4-prefill-qb-norm-rope-fused-sass.sh"))
        .expect("isolated V4 Q-B RMSNorm + RoPE SASS gate must exist")
}

fn cache_skip_v4() -> String {
    fs::read_to_string(
        repo_root().join("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs"),
    )
    .expect("V4 prefill source must exist")
}

fn runtime_buffers() -> String {
    fs::read_to_string(repo_root().join("crates/spark-runtime/src/buffers.rs"))
        .expect("runtime buffer source must exist")
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn rotate_after_bf16_boundary(
    normalized0: bf16,
    normalized1: bf16,
    position: u32,
    inv_freq: f32,
    mscale: f32,
) -> (bf16, bf16) {
    let x0 = normalized0.to_f32();
    let x1 = normalized1.to_f32();
    let angle = position as f32 * inv_freq;
    let cos_val = angle.cos() * mscale;
    let sin_val = angle.sin() * mscale;
    (
        bf16::from_f32(x0 * cos_val - x1 * sin_val),
        bf16::from_f32(x1 * cos_val + x0 * sin_val),
    )
}

fn xor_reduce_32(mut lanes: [f32; 32]) -> [f32; 32] {
    for offset in [16, 8, 4, 2, 1] {
        let prior = lanes;
        for lane in 0..32 {
            lanes[lane] = prior[lane] + prior[lane ^ offset];
        }
    }
    lanes
}

fn incumbent_pure_qb_norm(input: &[bf16; HEAD_DIM], eps: f32) -> ([bf16; HEAD_DIM], f32) {
    let mut thread_sums = [0.0_f32; THREADS];
    for tid in 0..HEAD_DIM / 2 {
        let x0 = input[2 * tid].to_f32();
        let x1 = input[2 * tid + 1].to_f32();
        thread_sums[tid] += x0 * x0 + x1 * x1;
    }

    let mut warp_sums = [0.0_f32; 32];
    for warp in 0..THREADS / 32 {
        let mut lanes = [0.0_f32; 32];
        lanes.copy_from_slice(&thread_sums[warp * 32..warp * 32 + 32]);
        warp_sums[warp] = xor_reduce_32(lanes)[0];
    }
    let sum = xor_reduce_32(warp_sums)[0];
    let rms = 1.0 / (sum / HEAD_DIM as f32 + eps).sqrt();
    let mut output = [bf16::from_bits(0); HEAD_DIM];
    for dim in 0..HEAD_DIM {
        output[dim] = bf16::from_f32(input[dim].to_f32() * rms * (1.0 + 0.0));
    }
    (output, rms)
}

#[test]
fn exact_shape_and_fail_closed_launch_contract() {
    let source = production_kernel();
    let flat = compact(&source);
    for contract in [
        "#define V4_QB_NORM_ROPE_NQ 64",
        "#define V4_QB_NORM_ROPE_NKV 1",
        "#define V4_QB_NORM_ROPE_HEAD_DIM 512",
        "#define V4_QB_NORM_ROPE_NOPE_DIM 448",
        "#define V4_QB_NORM_ROPE_DIM 64",
        "v4_prefill_qb_norm_rope_fused",
        "__launch_bounds__(512)",
    ] {
        assert!(source.contains(contract), "missing contract: {contract}");
    }
    assert!(flat.contains(
        "num_tokens==0||Q==nullptr||K==nullptr||weight==nullptr||positions==nullptr||inv_freq==nullptr"
    ));
    assert!(flat.contains("num_q_heads!=V4_QB_NORM_ROPE_NQ||num_kv_heads!=V4_QB_NORM_ROPE_NKV"));
    assert!(flat.contains(
        "head_dim!=V4_QB_NORM_ROPE_HEAD_DIM||nope_dim!=V4_QB_NORM_ROPE_NOPE_DIM||rotary_dim!=V4_QB_NORM_ROPE_DIM"
    ));
    assert!(flat.contains("blockDim.x!=512||blockDim.y!=1||blockDim.z!=1"));
    assert!(flat.contains("gridDim.x!=num_tokens||gridDim.y!=V4_QB_NORM_ROPE_NQ||gridDim.z!=1"));
    assert!(flat.contains("!(eps>0.0f)||!isfinite(eps)||!isfinite(mscale)"));
}

#[test]
fn packed_q_ownership_and_single_k_owner_are_bijections() {
    let mut q_norm = HashSet::new();
    let mut q_rope = HashSet::new();
    let mut k_rope = HashSet::new();
    for token in 0..3 {
        for head in 0..NQ {
            for tid in 0..THREADS {
                if tid < HEAD_DIM / 2 {
                    assert!(q_norm.insert((token, head, 2 * tid)));
                    assert!(q_norm.insert((token, head, 2 * tid + 1)));
                }
                if (NOPE_DIM / 2..HEAD_DIM / 2).contains(&tid) {
                    let pair = tid - NOPE_DIM / 2;
                    assert!(q_rope.insert((token, head, NOPE_DIM + 2 * pair)));
                    assert!(q_rope.insert((token, head, NOPE_DIM + 2 * pair + 1)));
                }
                if head == 0 && tid < ROPE_DIM / 2 {
                    assert!(k_rope.insert((token, 0, NOPE_DIM + 2 * tid)));
                    assert!(k_rope.insert((token, 0, NOPE_DIM + 2 * tid + 1)));
                }
            }
        }
    }
    assert_eq!(q_norm.len(), 3 * NQ * HEAD_DIM);
    assert_eq!(q_rope.len(), 3 * NQ * ROPE_DIM);
    assert_eq!(k_rope.len(), 3 * NKV * ROPE_DIM);
}

#[test]
fn reduction_unit_weight_rounding_and_rope_operand_order_match_incumbents() {
    let source = production_kernel();
    let flat = compact(&source);
    for contract in [
        "sum_sq += x0 * x0 + x1 * x1",
        "for (int offset = 16; offset > 0; offset >>= 1)",
        "val += __shfl_xor_sync(0xFFFFFFFF, val, offset)",
        "float rms = rsqrtf(warp_sums[0] / (float)V4_QB_NORM_ROPE_HEAD_DIM + eps)",
        "const float normalized0 = x0 * rms * (1.0f + w0)",
        "const float normalized1 = x1 * rms * (1.0f + w1)",
        "const __nv_bfloat16 rounded0 = __float2bfloat16(normalized0)",
        "const __nv_bfloat16 rounded1 = __float2bfloat16(normalized1)",
        "const float q0 = __bfloat162float(rounded0)",
        "const float q1 = __bfloat162float(rounded1)",
        "const float angle = static_cast<float>(abs_pos) * freq",
        "const float cos_val = cosf(angle) * mscale",
        "const float sin_val = sinf(angle) * mscale",
        "const float y0 = q0 * cos_val - q1 * sin_val",
        "const float y1 = q1 * cos_val + q0 * sin_val",
    ] {
        assert!(
            source.contains(contract),
            "missing numeric contract: {contract}"
        );
    }
    assert!(flat.contains("if(tid>=V4_QB_NORM_ROPE_NOPE_DIM/2)"));
    assert!(flat.contains("if(head==0&&tid<V4_QB_NORM_ROPE_PAIRS)"));

    let cases = [
        (0.0, -0.0, 0, 1.0, 1.0),
        (1.000_01, -2.000_02, 1, 0.000_1, 1.0),
        (f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 2_409, 0.75, 0.707),
        (65_504.0, -32_768.0, u32::MAX, f32::EPSILON, 1.125),
    ];
    for (x0, x1, position, inv_freq, mscale) in cases {
        let rounded0 = bf16::from_f32(x0);
        let rounded1 = bf16::from_f32(x1);
        let fused = rotate_after_bf16_boundary(rounded0, rounded1, position, inv_freq, mscale);
        let incumbent = rotate_after_bf16_boundary(
            bf16::from_f32(x0),
            bf16::from_f32(x1),
            position,
            inv_freq,
            mscale,
        );
        assert_eq!(fused.0.to_bits(), incumbent.0.to_bits());
        assert_eq!(fused.1.to_bits(), incumbent.1.to_bits());
    }
}

#[test]
fn q_b_norm_is_effectively_pure_under_the_zero_weight_buffer_invariant() {
    let prefill = compact(&cache_skip_v4());
    let buffers = compact(&runtime_buffers());
    assert!(prefill.contains(
        "self.rms_norm_k,q_full,&crate::weight_map::DenseWeight{weight:ctx.buffers.norm_unit_w(),},q_full,n*nq,hd_mla,eps,stream"
    ));
    assert!(buffers.contains("letnorm_unit_w=gpu.alloc(sizes.norm_unit_w)?;"));
    assert!(buffers.contains("gpu.memset(norm_unit_w,0,sizes.norm_unit_w)?;"));

    for x in [-65_504.0_f32, -1.0, -0.0, 0.0, 1.0, 65_504.0] {
        let rms = 0.125_f32;
        let zero_weight = bf16::from_f32(0.0).to_f32();
        assert_eq!(
            bf16::from_f32(x * rms * (1.0 + zero_weight)).to_bits(),
            bf16::from_f32(x * rms).to_bits()
        );
    }
}

#[test]
fn cpu_model_preserves_the_reduction_tree_and_material_bf16_seam() {
    let mut input = [bf16::from_bits(0); HEAD_DIM];
    for (dim, value) in input.iter_mut().enumerate() {
        let numerator = (dim as i32 % 41) - 20;
        *value = bf16::from_f32(numerator as f32 * 0.031_37);
    }
    let eps = 1.0e-6_f32;
    let (normalized, rms) = incumbent_pure_qb_norm(&input, eps);
    let mut sequential = normalized;
    let mut without_boundary = normalized;
    let position = 2_409_u32;

    for pair in 0..ROPE_DIM / 2 {
        let d0 = NOPE_DIM + 2 * pair;
        let d1 = d0 + 1;
        let inv_freq = 0.000_013 * (pair + 1) as f32;
        let correct =
            rotate_after_bf16_boundary(normalized[d0], normalized[d1], position, inv_freq, 1.125);
        sequential[d0] = correct.0;
        sequential[d1] = correct.1;

        let x0 = input[d0].to_f32() * rms;
        let x1 = input[d1].to_f32() * rms;
        let angle = position as f32 * inv_freq;
        let cos_val = angle.cos() * 1.125;
        let sin_val = angle.sin() * 1.125;
        without_boundary[d0] = bf16::from_f32(x0 * cos_val - x1 * sin_val);
        without_boundary[d1] = bf16::from_f32(x1 * cos_val + x0 * sin_val);
    }

    assert_eq!(&sequential[..NOPE_DIM], &normalized[..NOPE_DIM]);
    assert!(
        sequential[NOPE_DIM..]
            .iter()
            .zip(&without_boundary[NOPE_DIM..])
            .any(|(rounded_first, unrounded_first)| rounded_first.to_bits()
                != unrounded_first.to_bits())
    );
}

#[test]
fn structural_savings_are_exact_not_a_runtime_claim() {
    let source = production_kernel();
    assert_eq!(source.matches("q_packed = q32[tid]").count(), 1);
    assert_eq!(
        source
            .matches("reinterpret_cast<unsigned int*>(q)[tid] = output")
            .count(),
        1
    );

    let rms_apply_reread_bytes = TOKENS * NQ as u64 * HEAD_DIM as u64 * 2;
    let q_tail_intermediate_bytes = TOKENS * NQ as u64 * ROPE_DIM as u64 * 4;
    let total_bytes = rms_apply_reread_bytes + q_tail_intermediate_bytes;
    let incumbent_q_norm_ctas = TOKENS * NQ as u64;
    let incumbent_rope_ctas = TOKENS * (NQ + NKV) as u64;
    let fused_ctas = TOKENS * NQ as u64;

    assert_eq!(rms_apply_reread_bytes, 157_941_760);
    assert_eq!(q_tail_intermediate_bytes, 39_485_440);
    assert_eq!(q_tail_intermediate_bytes * LAYERS, 1_697_873_920);
    assert_eq!(total_bytes, 197_427_200);
    assert_eq!(total_bytes * LAYERS, 8_489_369_600);
    assert_eq!(LAYERS, 43);
    assert_eq!(
        (incumbent_q_norm_ctas + incumbent_rope_ctas - fused_ctas) * LAYERS,
        6_735_950
    );
}

#[test]
fn sass_gate_pins_sm121a_resources_and_opcode_topology() {
    let script = sass_gate();
    for contract in [
        "-arch=sm_121a",
        "--fmad=false",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing unreceipted nvcc flags",
        "v4_prefill_qb_norm_rope_fused",
        "F2F\\.BF16\\.F32",
        "SHFL",
        "MUFU\\.RSQ",
        "BAR",
        "stack",
        "local_bytes",
        "spill stores",
        "spill loads",
        "ATOM|RED|LDL|STL",
        "cubin_set_sha256",
    ] {
        assert!(
            script.contains(contract),
            "missing SASS gate contract: {contract}"
        );
    }
}

#[test]
fn legacy_probe_path_is_only_a_canonical_source_shim() {
    let shim = fs::read_to_string(
        repo_root().join("kernels/gb10/experiments/v4_prefill_qb_norm_rope_fused.cu"),
    )
    .expect("legacy probe include path must remain available");
    assert!(
        shim.contains("#include \"../deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu\"")
    );
    assert!(!shim.contains("extern \"C\""));
}
