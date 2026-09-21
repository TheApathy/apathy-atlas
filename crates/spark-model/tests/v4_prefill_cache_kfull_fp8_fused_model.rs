// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the production strided-K-full V4 prefill cache fusion.

use half::bf16;
use std::fs;
use std::path::PathBuf;

const TOKENS: u64 = 2_410;
const KV_LORA: usize = 512;
const K_HEAD_DIM: usize = 512;
const K_NOPE: usize = 448;
const ROPE: usize = 64;
const CACHE_DIM: usize = KV_LORA + ROPE;
const BLOCK_SIZE: usize = 16;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required strided-K cache input {relative}: {error}"))
}

fn compact(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    if exponent == 0 {
        return sign * mantissa as f32 * 2.0f32.powi(-9);
    }
    sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
}

fn quantize_finite(value: bf16, scale: f32) -> u8 {
    let input = value.to_f32() * scale.recip();
    let sign = if input.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = input.abs();
    if magnitude >= 448.0 {
        return sign | 0x7e;
    }
    let mut best = 0u8;
    let mut best_distance = f64::INFINITY;
    for code in 0u8..=0x7e {
        let distance = (e4m3_to_f32(code) as f64 - magnitude as f64).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0 && best & 1 != 0)
        {
            best = code;
            best_distance = distance;
        }
    }
    sign | best
}

fn incumbent_contiguous(latent: &[bf16], rope: &[bf16], scale: f32) -> Vec<u8> {
    latent
        .iter()
        .chain(rope)
        .map(|&value| quantize_finite(value, scale))
        .collect()
}

fn candidate_strided(latent: &[bf16], k_full: &[bf16], scale: f32) -> Vec<u8> {
    (0..CACHE_DIM)
        .map(|dim| {
            let value = if dim < KV_LORA {
                latent[dim]
            } else {
                k_full[K_NOPE + dim - KV_LORA]
            };
            quantize_finite(value, scale)
        })
        .collect()
}

#[test]
fn exact_shape_pointer_alignment_and_launch_contract_fail_closed() {
    let source = read("kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_cache_kfull_fp8_fused.cu");
    let flat = compact(&source);
    for contract in [
        "#define V4_CACHE_KFULL_TOKENS 2410u",
        "#define V4_CACHE_KFULL_KV_LORA 512u",
        "#define V4_CACHE_KFULL_K_HEAD_DIM 512u",
        "#define V4_CACHE_KFULL_K_NOPE 448u",
        "#define V4_CACHE_KFULL_ROPE 64u",
        "#define V4_CACHE_KFULL_DIM 576u",
        "#define V4_CACHE_KFULL_BLOCK_SIZE 16u",
        "#define V4_CACHE_KFULL_THREADS 256u",
        "__launch_bounds__(V4_CACHE_KFULL_THREADS)",
        "v4_prefill_cache_kfull_fp8_fused",
    ] {
        assert!(
            source.contains(contract),
            "missing exact contract: {contract}"
        );
    }
    for guard in [
        "kv_latent==nullptr||k_full==nullptr||k_cache==nullptr",
        "v_cache==nullptr||slot_mapping==nullptr||k_cache==v_cache",
        "num_tokens!=V4_CACHE_KFULL_TOKENS||num_blocks==0",
        "block_size!=V4_CACHE_KFULL_BLOCK_SIZE",
        "cache_stride!=V4_CACHE_KFULL_BLOCK_SIZE*V4_CACHE_KFULL_DIM",
        "!isfinite(k_scale)||!isfinite(v_scale)||k_scale<=0.0f||v_scale<=0.0f",
        "blockDim.x!=V4_CACHE_KFULL_THREADS||blockDim.y!=1||blockDim.z!=1",
        "gridDim.x!=V4_CACHE_KFULL_TOKENS||gridDim.y!=1||gridDim.z!=1",
        "(reinterpret_cast<unsignedlonglong>(kv_latent)&3ull)!=0",
        "(reinterpret_cast<unsignedlonglong>(k_full)&3ull)!=0",
        "(reinterpret_cast<unsignedlonglong>(k_cache)&1ull)!=0",
        "(reinterpret_cast<unsignedlonglong>(v_cache)&1ull)!=0",
        "(reinterpret_cast<unsignedlonglong>(slot_mapping)&7ull)!=0",
        "slot<0||static_cast<unsignedlonglong>(slot)>=max_slots",
    ] {
        assert!(flat.contains(guard), "missing fail-closed guard: {guard}");
    }
}

#[test]
fn strided_k_full_tail_matches_current_contiguous_incumbent_bits() {
    let latent: Vec<_> = (0..KV_LORA)
        .map(|index| bf16::from_f32((index as i32 - 263) as f32 * 0.031_25))
        .collect();
    let k_full: Vec<_> = (0..K_HEAD_DIM)
        .map(|index| bf16::from_f32((241 - index as i32) as f32 * 0.046_875))
        .collect();
    let contiguous_rope = &k_full[K_NOPE..K_NOPE + ROPE];
    for scale in [0.0625, 0.5, 1.0, 3.25] {
        assert_eq!(
            candidate_strided(&latent, &k_full, scale),
            incumbent_contiguous(&latent, contiguous_rope, scale)
        );
    }
    let k_bytes = candidate_strided(&latent, &k_full, 0.5);
    let v_bytes = candidate_strided(&latent, &k_full, 2.0);
    assert_ne!(
        k_bytes, v_bytes,
        "K/V scales must remain independently owned"
    );
    let same_scale_v = incumbent_contiguous(&latent, contiguous_rope, 0.5);
    assert_eq!(
        k_bytes, same_scale_v,
        "equal scales preserve V4 K=V cache identity"
    );
}

#[test]
fn pair_owners_cover_cache_once_and_no_pair_crosses_either_boundary() {
    let mut owners = [None; CACHE_DIM / 2];
    for thread in 0..256 {
        for pair in (thread..CACHE_DIM / 2).step_by(256) {
            assert!(owners[pair].replace(thread).is_none());
            let first = 2 * pair;
            let second = first + 1;
            assert_ne!(first < KV_LORA, second >= KV_LORA);
            if first >= KV_LORA {
                let k_first = K_NOPE + first - KV_LORA;
                let k_second = K_NOPE + second - KV_LORA;
                assert!(k_first >= K_NOPE && k_second < K_HEAD_DIM);
            }
        }
    }
    assert!(owners.iter().all(Option::is_some));
    assert_eq!(K_NOPE % 2, 0);
    assert_eq!(KV_LORA % 2, 0);
}

#[test]
fn paged_mapping_accepts_unique_nonmonotonic_and_negative_slots_only() {
    let num_blocks = 3usize;
    let slots = [31i64, -1, 0, 18, 47];
    let mut cache = vec![0xa5; num_blocks * BLOCK_SIZE * CACHE_DIM];
    let latent = vec![bf16::from_f32(0.5); KV_LORA];
    let k_full = vec![bf16::from_f32(-0.75); K_HEAD_DIM];
    let row = candidate_strided(&latent, &k_full, 0.5);
    for slot in slots {
        if slot < 0 {
            continue;
        }
        let block = slot as usize / BLOCK_SIZE;
        let offset = slot as usize % BLOCK_SIZE;
        let destination = (block * BLOCK_SIZE + offset) * CACHE_DIM;
        cache[destination..destination + CACHE_DIM].copy_from_slice(&row);
    }
    for slot in 0..num_blocks * BLOCK_SIZE {
        let destination = slot * CACHE_DIM;
        if [31usize, 0, 18, 47].contains(&slot) {
            assert_eq!(&cache[destination..destination + CACHE_DIM], row.as_slice());
        } else {
            assert!(
                cache[destination..destination + CACHE_DIM]
                    .iter()
                    .all(|&byte| byte == 0xa5)
            );
        }
    }
}

#[test]
fn source_audit_locks_direct_k_rotation_and_current_contiguous_dependency() {
    let direct = read("kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu");
    let prefill = read("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
    let candidate =
        read("kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_cache_kfull_fp8_fused.cu");
    assert!(
        direct.contains("K + static_cast<unsigned long long>(token) * V4_QB_NORM_ROPE_HEAD_DIM")
    );
    assert!(
        direct.contains(
            "reinterpret_cast<unsigned int*>(k + V4_QB_NORM_ROPE_NOPE_DIM)[tid] = output"
        )
    );
    assert!(prefill.contains("let k_rope_tmp = q_latent;"));
    assert!(prefill.contains("k_rope_tmp, // 64-dim RoPE from K (reused from step 3)"));
    assert!(
        candidate.contains(
            "k_full + static_cast<unsigned long long>(token) * V4_CACHE_KFULL_K_HEAD_DIM +"
        )
    );
    assert!(candidate.contains("V4_CACHE_KFULL_K_NOPE"));
}

#[test]
fn composability_saves_recreating_contiguous_rope_not_a_runtime_claim() {
    let scratch_roundtrip_per_token = (ROPE * 2 * 2) as u64;
    let extract_ctas_per_layer = (TOKENS * ROPE as u64).div_ceil(256);
    assert_eq!(scratch_roundtrip_per_token, 256);
    assert_eq!(scratch_roundtrip_per_token * TOKENS, 616_960);
    assert_eq!(scratch_roundtrip_per_token * TOKENS * 43, 26_529_280);
    let avoided_launches = 1u64 * 43;
    assert_eq!(avoided_launches, 43);
    assert_eq!(extract_ctas_per_layer, 603);
    assert_eq!(extract_ctas_per_layer * 43, 25_929);
}

#[test]
fn sass_gate_pins_sm121a_resources_pair_conversion_and_no_spills() {
    let script = read("scripts/check-v4-prefill-cache-kfull-fp8-fused-sass.sh");
    for contract in [
        "-arch=sm_121a",
        "--fmad=false",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing unreceipted nvcc flags",
        "v4_prefill_cache_kfull_fp8_fused",
        "F2FP\\.SATFINITE\\.E4M3",
        "STACK:0",
        "LOCAL:0",
        "spill stores",
        "spill loads",
        "ATOM|RED|LDL|STL|BAR",
        "source_sha256",
        "cubin_sha256",
    ] {
        assert!(
            script.contains(contract),
            "missing SASS contract: {contract}"
        );
    }
}

#[test]
fn legacy_probe_path_is_only_a_canonical_source_shim() {
    let shim = read("kernels/gb10/experiments/v4_prefill_cache_kfull_fp8_fused.cu");
    assert!(
        shim.contains(
            "#include \"../deepseek-v4-flash/nvfp4/v4_prefill_cache_kfull_fp8_fused.cu\""
        )
    );
    assert!(!shim.contains("extern \"C\""));
}
