// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the isolated V4 hc_pre_finish + RMS experiment.

use half::bf16;
use std::fs;
use std::path::PathBuf;

const TOKENS: u64 = 2_410;
const H: usize = 4_096;
const HC: usize = 4;
const BLOCK: usize = 1_024;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn experiment() -> String {
    fs::read_to_string(repo_root().join("kernels/gb10/experiments/v4_hc_pre_finish_rms_fused.cu"))
        .expect("isolated V4 HC finish/RMS experiment must exist")
}

fn sass_gate() -> String {
    fs::read_to_string(repo_root().join("scripts/check-v4-hc-pre-finish-rms-fused-sass.sh"))
        .expect("isolated V4 HC finish/RMS SASS gate must exist")
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn collapse_incumbent(streams: &[f32], pre: &[f32; HC]) -> Vec<bf16> {
    (0..H)
        .map(|dim| {
            let mut value = 0.0f32;
            for stream in 0..HC {
                value += pre[stream] * streams[stream * H + dim];
            }
            bf16::from_f32(value)
        })
        .collect()
}

fn xor_reduce(values: &mut [f32; 32]) {
    for offset in [16, 8, 4, 2, 1] {
        let before = *values;
        for lane in 0..32 {
            values[lane] = before[lane] + before[lane ^ offset];
        }
    }
}

fn incumbent_rms_sum(collapsed: &[bf16]) -> f32 {
    let mut partials = [0.0f32; BLOCK];
    for tid in 0..BLOCK {
        for pair in [tid, tid + BLOCK] {
            let x0 = collapsed[2 * pair].to_f32();
            let x1 = collapsed[2 * pair + 1].to_f32();
            partials[tid] += x0 * x0 + x1 * x1;
        }
    }
    let mut warp_sums = [0.0f32; 32];
    for (warp, chunk) in partials.chunks_exact(32).enumerate() {
        let mut lanes = [0.0f32; 32];
        lanes.copy_from_slice(chunk);
        xor_reduce(&mut lanes);
        warp_sums[warp] = lanes[0];
    }
    xor_reduce(&mut warp_sums);
    warp_sums[0]
}

fn fused_pairs_and_sum(streams: &[f32], pre: &[f32; HC]) -> (Vec<bf16>, f32) {
    let mut collapsed = vec![bf16::ZERO; H];
    let mut partials = [0.0f32; BLOCK];
    for tid in 0..BLOCK {
        for pair in [tid, tid + BLOCK] {
            for lane in 0..2 {
                let dim = 2 * pair + lane;
                let mut value = 0.0f32;
                for stream in 0..HC {
                    value += pre[stream] * streams[stream * H + dim];
                }
                collapsed[dim] = bf16::from_f32(value);
            }
            let x0 = collapsed[2 * pair].to_f32();
            let x1 = collapsed[2 * pair + 1].to_f32();
            partials[tid] += x0 * x0 + x1 * x1;
        }
    }
    let mut warp_sums = [0.0f32; 32];
    for (warp, chunk) in partials.chunks_exact(32).enumerate() {
        let mut lanes = [0.0f32; 32];
        lanes.copy_from_slice(chunk);
        xor_reduce(&mut lanes);
        warp_sums[warp] = lanes[0];
    }
    xor_reduce(&mut warp_sums);
    (collapsed, warp_sums[0])
}

fn normalize(collapsed: &[bf16], weights: &[bf16], sum: f32, eps: f32) -> Vec<bf16> {
    let rms = 1.0f32 / (sum / H as f32 + eps).sqrt();
    collapsed
        .iter()
        .zip(weights)
        .map(|(value, weight)| bf16::from_f32(value.to_f32() * rms * weight.to_f32()))
        .collect()
}

#[test]
fn exact_shape_and_launch_contract_fail_closed() {
    let source = experiment();
    let flat = compact(&source);
    for contract in [
        "#define V4_HC_H 4096u",
        "#define V4_HC_MULT 4u",
        "#define V4_HC_BLOCK 1024u",
        "__launch_bounds__(1024, 1)",
        "v4_hc_pre_finish_rms_fused",
    ] {
        assert!(source.contains(contract), "missing contract: {contract}");
    }
    assert!(flat.contains("num_tokens==0||streams==nullptr||mix_in==nullptr"));
    assert!(flat.contains("hc_scale==nullptr||hc_base==nullptr||norm_weight==nullptr"));
    assert!(flat.contains(
        "hidden_out==nullptr||normed_out==nullptr||post_out==nullptr||comb_out==nullptr"
    ));
    assert!(flat.contains("hidden_size!=V4_HC_H||hc_mult!=V4_HC_MULT"));
    assert!(flat.contains("blockDim.x!=V4_HC_BLOCK||blockDim.y!=1||blockDim.z!=1"));
    assert!(flat.contains("gridDim.x!=num_tokens||gridDim.y!=1||gridDim.z!=1"));
}

#[test]
fn deepseek_v4_uses_vanilla_rms_weights_without_an_offset() {
    let source = experiment();
    let flat = compact(&source);
    assert!(source.contains("DeepSeek-V4 vanilla RMSNorm weights"));
    assert!(flat.contains("x00*rms*w00,x01*rms*w01"));
    assert!(flat.contains("x10*rms*w10,x11*rms*w11"));
    assert!(!flat.contains("1.0f+w00"));
    assert!(!flat.contains("1.0f+w01"));
    assert!(!flat.contains("1.0f+w10"));
    assert!(!flat.contains("1.0f+w11"));

    let values = [bf16::from_f32(2.0), bf16::from_f32(-4.0)];
    let weights = [bf16::ZERO, bf16::from_f32(1.0)];
    let normalized = normalize(&values, &weights, 20.0, 0.0);
    assert_eq!(normalized[0], bf16::ZERO);
    assert_ne!(normalized[1], bf16::ZERO);
}

#[test]
fn fused_mapping_preserves_bf16_boundary_and_rms_reduction_tree() {
    let streams: Vec<f32> = (0..HC * H)
        .map(|index| {
            let signed = (index as i32 % 257) - 128;
            signed as f32 * 0.003_906_25 + (index % 7) as f32 * 0.000_31
        })
        .collect();
    let pre = [0.125f32, -0.375, 0.75, 1.031_25];
    let weights: Vec<bf16> = (0..H)
        .map(|index| bf16::from_f32(1.0 + (index as i32 % 31 - 15) as f32 * 0.001_25))
        .collect();
    let incumbent = collapse_incumbent(&streams, &pre);
    let incumbent_sum = incumbent_rms_sum(&incumbent);
    let (fused, fused_sum) = fused_pairs_and_sum(&streams, &pre);
    assert_eq!(
        incumbent
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        fused
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(incumbent_sum.to_bits(), fused_sum.to_bits());
    assert_eq!(
        normalize(&incumbent, &weights, incumbent_sum, 1.0e-6)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        normalize(&fused, &weights, fused_sum, 1.0e-6)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn materialized_hidden_output_keeps_the_incumbent_bf16_boundary() {
    let source = compact(&experiment());
    assert!(source.contains("unsignedint*consthidden=reinterpret_cast<unsignedint*>("));
    assert!(source.contains("hidden[pair0]=collapsed0"));
    assert!(source.contains("hidden[pair1]=collapsed1"));
    assert!(
        source.find("hidden[pair1]=collapsed1").unwrap()
            < source.find("v4_hc_unpack_bf16(collapsed0").unwrap()
    );
}

#[test]
fn every_packed_word_is_owned_once_with_incumbent_lane_order() {
    let mut owners = vec![None; H / 2];
    for tid in 0..BLOCK {
        for pair in [tid, tid + BLOCK] {
            assert!(owners[pair].replace(tid).is_none());
        }
    }
    assert!(owners.iter().all(Option::is_some));
}

#[test]
fn structural_savings_are_exact_not_a_timing_claim() {
    // The fused arm still writes `hidden` for exact observable semantics. It
    // removes only rms_norm_vanilla's immediate BF16 reread of that tensor.
    let bytes_per_site = TOKENS * H as u64 * 2;
    assert_eq!(bytes_per_site, 19_742_720);
    assert_eq!(bytes_per_site * 2, 39_485_440);
    assert_eq!(bytes_per_site * 2 * 43, 1_697_873_920);
    assert_eq!(2_u64 * 43, 86);
}

#[test]
fn sass_gate_pins_sm121a_resources_and_reduction_topology() {
    let script = sass_gate();
    for contract in [
        "-arch=sm_121a",
        "--fmad=false",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "v4_hc_pre_finish_rms_fused",
        "REG:48 STACK:0 SHARED:1168 LOCAL:0",
        "Used 48 registers, used 1 barriers, 144 bytes smem",
        "F2F\\.BF16\\.F32",
        "MUFU\\.RSQ",
        "SHFL\\.BFLY",
        "BAR\\.SYNC",
        "ATOM|RED",
        "LDL|STL",
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
fn production_registration_is_deepseek_target_scoped() {
    let root = repo_root();
    let wrapper = fs::read_to_string(
        root.join("kernels/gb10/deepseek-v4-flash/nvfp4/v4_hc_pre_finish_rms_fused.cu"),
    )
    .expect("production DeepSeek-V4 HC/RMS wrapper");
    assert!(wrapper.contains("../../experiments/v4_hc_pre_finish_rms_fused.cu"));
    assert!(
        !root
            .join("kernels/gb10/common/v4_hc_pre_finish_rms_fused.cu")
            .exists()
    );
}
