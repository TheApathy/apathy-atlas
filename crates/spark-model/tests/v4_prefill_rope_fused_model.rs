// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the isolated DeepSeek-V4 in-place prefill RoPE experiment.

use half::bf16;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const TOKENS: u64 = 2_410;
const NQ: usize = 64;
const NKV: usize = 1;
const HEAD_DIM: usize = 512;
const NOPE_DIM: usize = 448;
const ROPE_DIM: usize = 64;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn experiment() -> String {
    fs::read_to_string(repo_root().join("kernels/gb10/experiments/v4_prefill_rope_fused.cu"))
        .expect("isolated V4 fused RoPE experiment must exist")
}

fn sass_gate() -> String {
    fs::read_to_string(repo_root().join("scripts/check-v4-prefill-rope-fused-sass.sh"))
        .expect("isolated V4 fused RoPE SASS gate must exist")
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn incumbent_forward(x0: f32, x1: f32, angle: f32, mscale: f32) -> (bf16, bf16) {
    let cos_val = angle.cos() * mscale;
    let sin_val = angle.sin() * mscale;
    (
        bf16::from_f32(x0 * cos_val - x1 * sin_val),
        bf16::from_f32(x1 * cos_val + x0 * sin_val),
    )
}

fn incumbent_inverse(x0: f32, x1: f32, angle: f32, mscale: f32) -> (bf16, bf16) {
    let cos_val = angle.cos() * mscale;
    let sin_val = angle.sin() * mscale;
    (
        bf16::from_f32(x0 * cos_val + x1 * sin_val),
        bf16::from_f32(x1 * cos_val - x0 * sin_val),
    )
}

fn fused_pair(
    inverse: bool,
    x0: f32,
    x1: f32,
    position: u32,
    inv_freq: f32,
    mscale: f32,
) -> (bf16, bf16) {
    let angle = position as f32 * inv_freq;
    let cos_val = angle.cos() * mscale;
    let sin_val = angle.sin() * mscale;
    if inverse {
        (
            bf16::from_f32(x0 * cos_val + x1 * sin_val),
            bf16::from_f32(x1 * cos_val - x0 * sin_val),
        )
    } else {
        (
            bf16::from_f32(x0 * cos_val - x1 * sin_val),
            bf16::from_f32(x1 * cos_val + x0 * sin_val),
        )
    }
}

fn assert_tree_excludes_forward(path: &Path) {
    for entry in fs::read_dir(path).expect("production tree must exist") {
        let path = entry.expect("production entry").path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) == Some("experiments") {
                continue;
            }
            assert_tree_excludes_forward(&path);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("rs" | "toml" | "cu" | "cuh")
        ) {
            let source = fs::read_to_string(&path).expect("production source must be UTF-8");
            assert!(
                !source.contains("v4_prefill_rope_fused_forward"),
                "{}",
                path.display()
            );
        }
    }
}

#[test]
fn exact_shape_and_fail_closed_launch_contract() {
    let source = experiment();
    let flat = compact(&source);
    for contract in [
        "#define V4_ROPE_NQ 64",
        "#define V4_ROPE_NKV 1",
        "#define V4_ROPE_HEAD_DIM 512",
        "#define V4_ROPE_NOPE_DIM 448",
        "#define V4_ROPE_DIM 64",
        "v4_prefill_rope_fused_forward",
        "v4_prefill_rope_fused_inverse",
        "__launch_bounds__(32)",
    ] {
        assert!(source.contains(contract), "missing contract: {contract}");
    }
    assert!(
        flat.contains(
            "num_tokens==0||Q==nullptr||K==nullptr||positions==nullptr||inv_freq==nullptr"
        )
    );
    assert!(flat.contains("num_tokens==0||Q==nullptr||positions==nullptr||inv_freq==nullptr"));
    assert_eq!(flat.matches("!isfinite(mscale)||mscale<=0.0f").count(), 2);
    assert!(flat.contains("blockDim.x!=32||blockDim.y!=1||blockDim.z!=1"));
    assert!(
        flat.contains("gridDim.x!=num_tokens||gridDim.y!=V4_ROPE_NQ+V4_ROPE_NKV||gridDim.z!=1")
    );
    assert!(flat.contains("gridDim.x!=num_tokens||gridDim.y!=V4_ROPE_NQ||gridDim.z!=1"));
    assert!(flat.contains("num_q_heads!=V4_ROPE_NQ||num_kv_heads!=V4_ROPE_NKV"));
    assert!(flat.contains(
        "head_dim!=V4_ROPE_HEAD_DIM||nope_dim!=V4_ROPE_NOPE_DIM||rotary_dim!=V4_ROPE_DIM"
    ));
    assert!(flat.contains("constfloatangle=static_cast<float>(abs_pos)*freq"));
    assert!(flat.contains("constfloatcos_val=cosf(angle)*mscale"));
    assert!(flat.contains("constfloatsin_val=sinf(angle)*mscale"));
    assert!(flat.contains("constfloaty0=x0*cos_val-x1*sin_val"));
    assert!(flat.contains("constfloaty1=x1*cos_val+x0*sin_val"));
    assert!(flat.contains("constfloaty0=x0*cos_val+x1*sin_val"));
    assert!(flat.contains("constfloaty1=x1*cos_val-x0*sin_val"));
    assert!(source.contains("ptr[d0] = __float2bfloat16(y0)"));
    assert!(source.contains("ptr[d1] = __float2bfloat16(y1)"));
}

#[test]
fn thread_mapping_is_a_bijection_over_only_trailing_rope_pairs() {
    let mut forward = HashSet::new();
    for token in 0..3 {
        for head_slot in 0..(NQ + NKV) {
            for pair in 0..(ROPE_DIM / 2) {
                let is_q = head_slot < NQ;
                let head = if is_q { head_slot } else { head_slot - NQ };
                let base = token * (if is_q { NQ } else { NKV }) * HEAD_DIM;
                for dim in [NOPE_DIM + 2 * pair, NOPE_DIM + 2 * pair + 1] {
                    assert!(forward.insert((is_q, base + head * HEAD_DIM + dim)));
                    assert!((NOPE_DIM..HEAD_DIM).contains(&dim));
                }
            }
        }
    }
    assert_eq!(forward.len(), 3 * (NQ + NKV) * ROPE_DIM);

    let inverse: HashSet<_> = (0..3)
        .flat_map(|token| {
            (0..NQ).flat_map(move |head| {
                (NOPE_DIM..HEAD_DIM).map(move |dim| token * NQ * HEAD_DIM + head * HEAD_DIM + dim)
            })
        })
        .collect();
    assert_eq!(inverse.len(), 3 * NQ * ROPE_DIM);
}

#[test]
fn formulas_and_bf16_boundary_match_incumbent_operand_order() {
    let cases = [
        (0.0, -0.0, 0, 1.0, 1.0),
        (1.0, -2.0, 1, 0.000_1, 1.0),
        (f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 2_409, 0.75, 0.707),
        (65_504.0, -32_768.0, u32::MAX, f32::EPSILON, 1.125),
    ];
    for (x0, x1, position, inv_freq, mscale) in cases {
        let angle = position as f32 * inv_freq;
        let forward = fused_pair(false, x0, x1, position, inv_freq, mscale);
        let inverse = fused_pair(true, x0, x1, position, inv_freq, mscale);
        assert_eq!(
            forward.0.to_bits(),
            incumbent_forward(x0, x1, angle, mscale).0.to_bits()
        );
        assert_eq!(
            forward.1.to_bits(),
            incumbent_forward(x0, x1, angle, mscale).1.to_bits()
        );
        assert_eq!(
            inverse.0.to_bits(),
            incumbent_inverse(x0, x1, angle, mscale).0.to_bits()
        );
        assert_eq!(
            inverse.1.to_bits(),
            incumbent_inverse(x0, x1, angle, mscale).1.to_bits()
        );
    }
}

#[test]
fn structural_savings_are_exact_not_a_runtime_claim() {
    let forward_elements = TOKENS * (NQ + NKV) as u64 * ROPE_DIM as u64;
    let inverse_elements = TOKENS * NQ as u64 * ROPE_DIM as u64;
    assert_eq!(forward_elements * 8, 80_204_800);
    assert_eq!(inverse_elements * 8, 78_970_880);
    assert_eq!((forward_elements + inverse_elements) * 8, 159_175_680);
    assert_eq!(159_175_680_u64 * 43, 6_844_554_240);
    assert_eq!(6_u64 * 43, 258);
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
        "v4_prefill_rope_fused_forward",
        "v4_prefill_rope_fused_inverse",
        "F2FP\\.BF16\\.F32\\.PACK_AB",
        "MUFU\\.(COS|SIN)",
        "== 13",
        "== 12",
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
fn forward_experiment_has_no_registry_or_serving_reachability() {
    let root = repo_root();
    for path in [
        root.join("crates/spark-model/src"),
        root.join("crates/atlas-kernels"),
        root.join("kernels/gb10/common"),
        root.join("kernels/gb10/deepseek-v4-flash"),
    ] {
        assert_tree_excludes_forward(&path);
    }
}
