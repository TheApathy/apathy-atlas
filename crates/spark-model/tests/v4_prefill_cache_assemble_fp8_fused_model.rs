// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the isolated V4 prefill cache-assemble + FP8 write experiment.

use half::bf16;
use std::fs;
use std::path::{Path, PathBuf};

const TOKENS: u64 = 2_410;
const KV_LORA: usize = 512;
const ROPE: usize = 64;
const CACHE_DIM: usize = KV_LORA + ROPE;
const ATTENTION_HEAD_DIM: usize = 512;
const BLOCK_SIZE: usize = 16;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn experiment() -> String {
    fs::read_to_string(
        repo_root().join("kernels/gb10/experiments/v4_prefill_cache_assemble_fp8_fused.cu"),
    )
    .expect("isolated V4 cache-assemble/FP8 experiment must exist")
}

fn sass_gate() -> String {
    fs::read_to_string(
        repo_root().join("scripts/check-v4-prefill-cache-assemble-fp8-fused-sass.sh"),
    )
    .expect("isolated V4 cache-assemble/FP8 SASS gate must exist")
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
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

fn f32_to_e4m3_rne_satfinite(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return sign | 0x7f;
    }
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

fn quantize(value: bf16, scale: f32) -> u8 {
    f32_to_e4m3_rne_satfinite(value.to_f32() * scale.recip())
}

fn assembled_row(latent: &[bf16], rope: &[bf16]) -> Vec<bf16> {
    latent.iter().chain(rope).copied().collect()
}

fn incumbent_row(latent: &[bf16], rope: &[bf16], scale: f32) -> Vec<u8> {
    assembled_row(latent, rope)
        .into_iter()
        .map(|value| quantize(value, scale))
        .collect()
}

fn fused_row(latent: &[bf16], rope: &[bf16], scale: f32) -> Vec<u8> {
    (0..CACHE_DIM)
        .map(|dim| {
            let value = if dim < KV_LORA {
                latent[dim]
            } else {
                rope[dim - KV_LORA]
            };
            quantize(value, scale)
        })
        .collect()
}

fn write_model(
    slots: &[i64],
    rows: &[(Vec<bf16>, Vec<bf16>)],
    num_blocks: usize,
    k_scale: f32,
    v_scale: f32,
) -> (Vec<u8>, Vec<u8>) {
    let mut k_cache = vec![0xa5; num_blocks * BLOCK_SIZE * CACHE_DIM];
    let mut v_cache = vec![0x5a; num_blocks * BLOCK_SIZE * CACHE_DIM];
    let max_slots = num_blocks * BLOCK_SIZE;
    for (&slot, (latent, rope)) in slots.iter().zip(rows) {
        if slot < 0 || slot as usize >= max_slots {
            continue;
        }
        let block = slot as usize / BLOCK_SIZE;
        let offset = slot as usize % BLOCK_SIZE;
        let dst = (block * BLOCK_SIZE + offset) * CACHE_DIM;
        k_cache[dst..dst + CACHE_DIM].copy_from_slice(&fused_row(latent, rope, k_scale));
        v_cache[dst..dst + CACHE_DIM].copy_from_slice(&fused_row(latent, rope, v_scale));
    }
    (k_cache, v_cache)
}

fn assert_tree_excludes(path: &Path) {
    for entry in fs::read_dir(path).expect("production tree must exist") {
        let path = entry.expect("production entry").path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) == Some("experiments") {
                continue;
            }
            assert_tree_excludes(&path);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("rs" | "toml" | "cu" | "cuh")
        ) {
            let source = fs::read_to_string(&path).expect("production source must be UTF-8");
            assert!(
                !source.contains("v4_prefill_cache_assemble_fp8_fused"),
                "{}",
                path.display()
            );
        }
    }
}

#[test]
fn exact_shape_and_abi_fail_closed() {
    let source = experiment();
    let flat = compact(&source);
    for contract in [
        "#define V4_CACHE_TOKENS 2410u",
        "#define V4_CACHE_KV_LORA 512u",
        "#define V4_CACHE_ROPE 64u",
        "#define V4_CACHE_DIM 576u",
        "#define V4_CACHE_BLOCK_SIZE 16u",
        "#define V4_CACHE_THREADS 256u",
        "__launch_bounds__(V4_CACHE_THREADS)",
        "v4_prefill_cache_assemble_fp8_fused",
    ] {
        assert!(source.contains(contract), "missing contract: {contract}");
    }
    assert!(flat.contains("kv_latent==nullptr||k_rope==nullptr||k_cache==nullptr"));
    assert!(flat.contains("v_cache==nullptr||slot_mapping==nullptr||k_cache==v_cache"));
    assert!(flat.contains("num_tokens!=V4_CACHE_TOKENS||num_blocks==0"));
    assert!(flat.contains("block_size!=V4_CACHE_BLOCK_SIZE"));
    assert!(flat.contains("cache_stride!=V4_CACHE_BLOCK_SIZE*V4_CACHE_DIM"));
    assert!(flat.contains("!isfinite(k_scale)||!isfinite(v_scale)||k_scale<=0.0f||v_scale<=0.0f"));
    assert!(flat.contains("blockDim.x!=V4_CACHE_THREADS||blockDim.y!=1||blockDim.z!=1"));
    assert!(flat.contains("gridDim.x!=V4_CACHE_TOKENS||gridDim.y!=1||gridDim.z!=1"));
    assert!(flat.contains("slot<0||static_cast<unsignedlonglong>(slot)>=max_slots"));
}

#[test]
fn pair_conversion_matches_incumbent_bits_and_scale_ownership() {
    let latent: Vec<_> = (0..KV_LORA)
        .map(|index| bf16::from_f32((index as i32 - 257) as f32 * 0.031_25))
        .collect();
    let rope: Vec<_> = (0..ROPE)
        .map(|index| bf16::from_f32((31 - index as i32) as f32 * 0.093_75))
        .collect();
    for scale in [0.0625, 0.5, 1.0, 3.25] {
        assert_eq!(
            fused_row(&latent, &rope, scale),
            incumbent_row(&latent, &rope, scale)
        );
    }
    let k = fused_row(&latent, &rope, 0.5);
    let v = fused_row(&latent, &rope, 2.0);
    assert_ne!(k, v, "K and V must retain their independent tensor scales");
    assert_eq!(k, incumbent_row(&latent, &rope, 0.5));
    assert_eq!(v, incumbent_row(&latent, &rope, 2.0));
    let rows = [(latent, rope)];
    let (same_k, same_v) = write_model(&[0], &rows, 1, 0.5, 0.5);
    assert_eq!(
        &same_k[..CACHE_DIM],
        &same_v[..CACHE_DIM],
        "V4 K=V makes cache bytes identical when the tensor scales match"
    );
}

#[test]
fn nonmonotonic_slots_map_to_exact_pages_and_invalid_slots_do_not_write() {
    let slots = [31, -1, 0, 18, 64];
    let rows: Vec<_> = (0..slots.len())
        .map(|token| {
            let latent = (0..KV_LORA)
                .map(|dim| bf16::from_f32(token as f32 + dim as f32 / 256.0))
                .collect();
            let rope = (0..ROPE)
                .map(|dim| bf16::from_f32(-(token as f32) - dim as f32 / 64.0))
                .collect();
            (latent, rope)
        })
        .collect();
    let (k, v) = write_model(&slots, &rows, 4, 0.5, 0.5);
    for (token, &slot) in slots.iter().enumerate() {
        if slot < 0 || slot >= 64 {
            continue;
        }
        let dst = slot as usize * CACHE_DIM;
        let expected = incumbent_row(&rows[token].0, &rows[token].1, 0.5);
        assert_eq!(&k[dst..dst + CACHE_DIM], expected.as_slice());
        assert_eq!(&v[dst..dst + CACHE_DIM], expected.as_slice());
    }
    for slot in 0..64 {
        if ![31usize, 0, 18].contains(&slot) {
            let offset = slot * CACHE_DIM;
            assert!(
                k[offset..offset + CACHE_DIM]
                    .iter()
                    .all(|&byte| byte == 0xa5)
            );
            assert!(
                v[offset..offset + CACHE_DIM]
                    .iter()
                    .all(|&byte| byte == 0x5a)
            );
        }
    }
}

#[test]
fn every_output_pair_has_one_thread_owner_and_never_crosses_source_boundary() {
    let mut owners = [None; CACHE_DIM / 2];
    for thread in 0..256 {
        for pair in (thread..CACHE_DIM / 2).step_by(256) {
            assert!(owners[pair].replace(thread).is_none());
            assert_ne!(2 * pair < KV_LORA, 2 * pair + 1 >= KV_LORA);
        }
    }
    assert!(owners.iter().all(Option::is_some));
}

#[test]
fn structural_savings_match_current_prefill_source_not_a_runtime_claim() {
    let source_bf16_read = (CACHE_DIM * 2) as u64;
    let scratch_kv_bf16 = (2 * CACHE_DIM * 2) as u64;
    let paged_kv_fp8 = (2 * CACHE_DIM) as u64;
    let incumbent_cache_traffic = source_bf16_read + scratch_kv_bf16 * 2 + paged_kv_fp8;
    let fused_cache_traffic = source_bf16_read + paged_kv_fp8;
    assert_eq!(incumbent_cache_traffic, 6_912);
    assert_eq!(fused_cache_traffic, 2_304);
    let fused_cache_bytes_saved_per_token = incumbent_cache_traffic - fused_cache_traffic;
    let dead_k_to_v_copy_bytes_saved_per_token = (ATTENTION_HEAD_DIM * 2 * 2) as u64;
    assert_eq!(fused_cache_bytes_saved_per_token, 4_608);
    assert_eq!(dead_k_to_v_copy_bytes_saved_per_token, 2_048);
    assert_eq!(
        (fused_cache_bytes_saved_per_token + dead_k_to_v_copy_bytes_saved_per_token) * TOKENS,
        16_040_960
    );
    assert_eq!(16_040_960u64 * 43, 689_761_280);
    assert_eq!(689_761_280f64 / 1_048_576.0, 657.807_617_187_5);
    assert_eq!(1u64 * 43, 43, "one cache-kernel launch removed per layer");
    assert_eq!(1u64 * 43, 43, "one dead D2D enqueue removed per layer");
    assert_eq!(TOKENS * 43, 103_630, "one CTA removed per token and layer");
}

#[test]
fn source_audit_locks_incumbent_mapping_and_dead_copy() {
    let root = repo_root();
    let prefill = fs::read_to_string(
        root.join("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs"),
    )
    .expect("V4 prefill source must exist");
    let assemble =
        fs::read_to_string(root.join("kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu"))
            .expect("MLA assembly source must exist");
    let reshape = fs::read_to_string(root.join("kernels/gb10/common/reshape_and_cache.cu"))
        .expect("FP8 paged-write source must exist");
    assert!(prefill.contains("copy_d2d_async(k_out, v_out, (n * kv_dim) as usize * 2, stream)"));
    assert_eq!(prefill.matches(".arg_ptr(k_out)").count(), 6);
    assert!(prefill.contains("ops::mla_cache_assemble_batched("));
    assert!(prefill.contains("self.write_kv_cache("));
    assert!(assemble.contains("v_cache[k_off + idx] = val;"));
    assert!(assemble.contains("v_cache[k_off + idx] = rope_val;"));
    assert!(reshape.contains("const float inv_k_scale = 1.0f / k_scale;"));
    assert!(reshape.contains("const float inv_v_scale = 1.0f / v_scale;"));
    assert!(reshape.contains("block_idx * cache_stride"));
    assert!(reshape.contains("block_offset * n_elems"));
}

#[test]
fn sass_gate_pins_sm121a_resources_conversion_and_no_spills() {
    let script = sass_gate();
    for contract in [
        "-arch=sm_121a",
        "--fmad=false",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing unreceipted nvcc flags",
        "v4_prefill_cache_assemble_fp8_fused",
        "F2FP\\.SATFINITE\\.E4M3",
        "stack",
        "LOCAL:0",
        "spill stores",
        "spill loads",
        "ATOM|RED|LDL|STL|BAR",
        "source_sha256",
        "cubin_sha256",
    ] {
        assert!(
            script.contains(contract),
            "missing SASS gate contract: {contract}"
        );
    }
}

#[test]
fn experiment_has_no_registry_or_serving_reachability() {
    let root = repo_root();
    for path in [
        root.join("crates/spark-model/src"),
        root.join("crates/atlas-kernels"),
        root.join("kernels/gb10/common"),
        root.join("kernels/gb10/deepseek-v4-flash"),
    ] {
        assert_tree_excludes(&path);
    }
}
