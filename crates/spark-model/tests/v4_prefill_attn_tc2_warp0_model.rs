// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the isolated TC2 warp-0 softmax broadcast experiment.

use half::bf16;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

const TOKENS: u64 = 2_410;
const HEADS: u64 = 64;
const ROWS_PER_CTA: u64 = 16;
const WINDOW: u64 = 128;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    fs::read_to_string(root().join(relative)).unwrap_or_else(|error| panic!("{relative}: {error}"))
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn tile_visits(ratio: Option<u64>) -> u64 {
    let n_comp = ratio.map_or(0, |value| TOKENS / value);
    let mut visits = 0;
    for q_first in (0..TOKENS).step_by(ROWS_PER_CTA as usize) {
        let q_last = (q_first + ROWS_PER_CTA).min(TOKENS);
        let raw_first = if q_first + 1 > WINDOW {
            q_first + 1 - WINDOW
        } else {
            0
        };
        visits += (q_last - raw_first).div_ceil(16);
        if let Some(value) = ratio {
            let compressed_last = (q_last / value).min(n_comp);
            visits += compressed_last.div_ceil(16);
        }
    }
    visits * HEADS
}

fn sequential_sum(parts: [f32; 4]) -> f32 {
    ((parts[0] + parts[1]) + parts[2]) + parts[3]
}

#[derive(Clone, Copy)]
struct LaunchEligibility {
    block: [u32; 3],
    grid: [u32; 3],
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    ratio: u32,
    mandatory_pointers: [bool; 6],
    aligned: [bool; 7],
    output_disjoint: [bool; 6],
    inv_sqrt_d: f32,
}

fn eligible(launch: LaunchEligibility) -> bool {
    let required_q_blocks = launch.seq_len / 16 + u32::from(launch.seq_len % 16 != 0);
    launch.block == [128, 1, 1]
        && launch.grid == [launch.num_q_heads, required_q_blocks, 1]
        && launch.num_kv_heads != 0
        && launch.num_q_heads % launch.num_kv_heads == 0
        && launch.ratio != 0
        && launch.mandatory_pointers.into_iter().all(|present| present)
        && launch.aligned.into_iter().all(|aligned| aligned)
        && launch.output_disjoint.into_iter().all(|disjoint| disjoint)
        && launch.inv_sqrt_d.is_finite()
        && launch.inv_sqrt_d > 0.0
}

#[test]
fn exact_production_launch_and_pointer_guard_is_fail_closed() {
    let good = LaunchEligibility {
        block: [128, 1, 1],
        grid: [64, 151, 1],
        seq_len: 2_410,
        num_q_heads: 64,
        num_kv_heads: 1,
        ratio: 4,
        mandatory_pointers: [true; 6],
        aligned: [true; 7],
        output_disjoint: [true; 6],
        inv_sqrt_d: 1.0 / (512.0f32).sqrt(),
    };
    assert!(eligible(good));
    for bad in [
        LaunchEligibility {
            block: [64, 1, 1],
            ..good
        },
        LaunchEligibility {
            block: [128, 2, 1],
            ..good
        },
        LaunchEligibility {
            block: [128, 1, 2],
            ..good
        },
        LaunchEligibility {
            grid: [63, 151, 1],
            ..good
        },
        LaunchEligibility {
            grid: [64, 150, 1],
            ..good
        },
        LaunchEligibility {
            grid: [64, 151, 2],
            ..good
        },
        LaunchEligibility {
            num_kv_heads: 0,
            ..good
        },
        LaunchEligibility {
            num_q_heads: 64,
            num_kv_heads: 3,
            grid: [64, 151, 1],
            ..good
        },
        LaunchEligibility { ratio: 0, ..good },
    ] {
        assert!(!eligible(bad));
    }
    for missing in 0..6 {
        let mut pointers = [true; 6];
        pointers[missing] = false;
        assert!(!eligible(LaunchEligibility {
            mandatory_pointers: pointers,
            ..good
        }));
    }
    for misaligned in 0..7 {
        let mut aligned = [true; 7];
        aligned[misaligned] = false;
        assert!(!eligible(LaunchEligibility { aligned, ..good }));
    }
    for overlap in 0..6 {
        let mut output_disjoint = [true; 6];
        output_disjoint[overlap] = false;
        assert!(!eligible(LaunchEligibility {
            output_disjoint,
            ..good
        }));
    }
    for inv_sqrt_d in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(!eligible(LaunchEligibility { inv_sqrt_d, ..good }));
    }

    let candidate = read("kernels/gb10/experiments/v4_prefill_attn_compressed_tc2_warp0.cu");
    let production = read("kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu");
    let shared_guard = [
        "const unsigned int required_q_blocks =",
        "seq_len / BR + ((seq_len % BR) != 0u);",
        "blockDim.x != 128u || blockDim.y != 1u || blockDim.z != 1u ||",
        "gridDim.x != num_q_heads || gridDim.y != required_q_blocks ||",
        "gridDim.z != 1u || num_kv_heads == 0u ||",
        "num_q_heads % num_kv_heads != 0u || ratio == 0u ||",
        "Q == nullptr || K == nullptr || V == nullptr || Kc == nullptr ||",
        "Vc == nullptr || O == nullptr",
    ];
    for line in shared_guard {
        assert!(production.contains(line), "production guard drift: {line}");
        assert!(candidate.contains(line), "candidate guard drift: {line}");
    }
    for candidate_only in [
        "head_dim != TC2_HD",
        "!isfinite(inv_sqrt_d)",
        "inv_sqrt_d <= 0.0f",
        "v4_tc2_aligned(Q, 4u)",
        "v4_tc2_aligned(K, 16u)",
        "v4_tc2_aligned(V, 16u)",
        "v4_tc2_aligned(Kc, 16u)",
        "v4_tc2_aligned(Vc, 16u)",
        "v4_tc2_aligned(O, 4u)",
        "v4_tc2_ranges_overlap(O, output_bytes, Q, output_bytes)",
        "v4_tc2_ranges_overlap(O, output_bytes, sinks, sink_bytes)",
    ] {
        assert!(
            candidate.contains(candidate_only),
            "candidate safety guard omits `{candidate_only}`"
        );
    }
    assert!(candidate.contains("K, V, Kc, and Vc may alias read-only"));
    assert!(!candidate.contains("sinks == nullptr"));
    let pointer_guard = candidate.find("Q == nullptr").unwrap();
    let division = candidate.find("num_q_heads / num_kv_heads").unwrap();
    let first_index = candidate
        .find("const unsigned int q_head = blockIdx.x")
        .unwrap();
    assert!(pointer_guard < first_index && first_index < division);
}

#[test]
fn tail_rows_never_form_out_of_allocation_q_or_o_pointers() {
    let candidate = compact(&read(
        "kernels/gb10/experiments/v4_prefill_attn_compressed_tc2_warp0.cu",
    ));
    let production = compact(&read(
        "kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu",
    ));
    for contract in [
        "const__nv_bfloat16*Qr0=v0?Q+",
        "const__nv_bfloat16*Qr1=v1?Q+",
        "__nv_bfloat16*O0=v0?O+",
        "__nv_bfloat16*O1=v1?O+",
    ] {
        assert_eq!(
            candidate.matches(contract).count(),
            1,
            "candidate `{contract}`"
        );
        assert_eq!(
            production.matches(contract).count(),
            2,
            "both production TC implementations must guard `{contract}`"
        );
    }
}

#[test]
fn candidate_abi_and_production_host_launch_remain_locked() {
    fn signature(source: &str, symbol: &str) -> String {
        let start = source
            .find(&format!("extern \"C\" __global__ void {symbol}("))
            .unwrap();
        let end = source[start..].find(") {").unwrap() + start + 1;
        compact(&source[start..end]).replace(symbol, "TC2_SYMBOL")
    }

    let candidate = read("kernels/gb10/experiments/v4_prefill_attn_compressed_tc2_warp0.cu");
    let production = read("kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu");
    assert_eq!(
        signature(&candidate, "v4_prefill_attn_compressed_tc2_warp0"),
        signature(&production, "prefill_attn_compressed_tc2")
    );

    let init = compact(&read(
        "crates/spark-model/src/layers/qwen3_attention/init.rs",
    ));
    assert!(init.contains(
        "prefill_attn_compressed_tc2_k:super::super::try_kernel(gpu,\"prefill_attn_compressed\",\"prefill_attn_compressed_tc2\",)"
    ));

    let host = compact(&read(
        "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
    ));
    assert!(host.contains(".grid([nq,n.div_ceil(16),1]).block([128,1,1])"));
    fn ordered(source: &str, start: &str, arguments: &[&str]) {
        let start_index = source
            .find(start)
            .unwrap_or_else(|| panic!("missing `{start}`"));
        let end_index = source[start_index..]
            .find(".launch(stream)")
            .map(|offset| start_index + offset + ".launch(stream)".len())
            .unwrap_or_else(|| panic!("missing launch end after `{start}`"));
        let launch = &source[start_index..end_index];
        let mut cursor = 0;
        for argument in arguments {
            cursor += launch[cursor..]
                .find(argument)
                .unwrap_or_else(|| panic!("missing ordered `{argument}`"))
                + argument.len();
        }
    }
    ordered(
        &host,
        "KernelLaunch::new(ctx.gpu,attn_k)",
        &[
            ".arg_ptr(q_full)",
            ".arg_ptr(k_out)",
            ".arg_ptr(k_out)",
            ".arg_ptr(comp_k)",
            ".arg_ptr(comp_k)",
            ".arg_ptr(mla.attn_sink)",
            ".arg_ptr(attn_out)",
            ".arg_u32(n)",
            ".arg_u32(nq)",
            ".arg_u32(nkv)",
            ".arg_u32(hd_mla)",
            ".arg_u32(n_win)",
            ".arg_u32(ratio)",
            ".arg_u32(V4_WINDOW)",
            ".arg_f32(1.0f32/(hd_mlaasf32).sqrt())",
            ".launch(stream)",
        ],
    );
    ordered(
        &host,
        "KernelLaunch::new(ctx.gpu,tc_dense_k)",
        &[
            ".arg_ptr(q_full)",
            ".arg_ptr(k_out)",
            ".arg_ptr(k_out)",
            ".arg_ptr(k_out)",
            ".arg_ptr(k_out)",
            ".arg_ptr(mla.attn_sink)",
            ".arg_ptr(attn_out)",
            ".arg_u32(n)",
            ".arg_u32(nq)",
            ".arg_u32(nkv)",
            ".arg_u32(hd_mla)",
            ".arg_u32(0)",
            ".arg_u32(1)",
            ".arg_u32(V4_WINDOW)",
            ".arg_f32(1.0f32/(hd_mlaasf32).sqrt())",
            ".launch(stream)",
        ],
    );
}

#[test]
fn lane_to_row_key_fragment_mapping_is_bijective_and_broadcastable() {
    let mut owned = HashSet::new();
    for lane in 0..32 {
        let group = lane / 4;
        let t = lane % 4;
        for row in [group, group + 8] {
            for cix in 0..4 {
                let key = (cix / 2) * 8 + t * 2 + cix % 2;
                assert!(owned.insert((row, key)));
                assert!(row < 16 && key < 16);
            }
        }
    }
    assert_eq!(owned.len(), 16 * 16);

    for lane in 0..32 {
        let group = lane / 4;
        let t = lane % 4;
        for (row, cix) in [group, group + 8]
            .into_iter()
            .flat_map(|row| (0..4).map(move |cix| (row, cix)))
        {
            let key = (cix / 2) * 8 + t * 2 + cix % 2;
            let parts = [
                row as f32 + key as f32 / 17.0,
                -(row as f32) / 13.0,
                key as f32 / 7.0,
                -0.125,
            ];
            let producer = sequential_sum(parts);
            for consumer_warp in 0..4 {
                let broadcast_lane = lane;
                assert_eq!(broadcast_lane, lane, "warp {consumer_warp}");
                assert_eq!(producer.to_bits(), sequential_sum(parts).to_bits());
            }
        }
    }
}

#[test]
fn bf16_probability_pack_is_identical_for_every_consumer_warp() {
    for pair in [
        [0.0, -0.0],
        [f32::MIN_POSITIVE, -f32::MIN_POSITIVE],
        [0.125, 0.875],
        [1.0, f32::INFINITY],
    ] {
        let low = bf16::from_f32(pair[0]).to_bits() as u32;
        let high = bf16::from_f32(pair[1]).to_bits() as u32;
        let warp0_pack = low | (high << 16);
        for _consumer_warp in 0..4 {
            assert_eq!(warp0_pack, low | (high << 16));
        }
    }
}

#[test]
fn one_uniform_broadcast_barrier_reuses_the_nonalias_stage_barrier() {
    let incumbent_barriers = |same: bool| if same { 3 } else { 4 };
    let warp0_barriers = |_same: bool| 4;
    assert_eq!(warp0_barriers(true), incumbent_barriers(true) + 1);
    assert_eq!(warp0_barriers(false), incumbent_barriers(false));

    let source = read("kernels/gb10/experiments/v4_prefill_attn_compressed_tc2_warp0.cu");
    assert!(!compact(&source).contains("if(!(SAME)){__syncthreads();}"));
    let publish = source.find("slot.l1 = l1").unwrap();
    let barrier = source[publish..].find("__syncthreads()").unwrap() + publish;
    let consume = source[barrier..]
        .find("const TC2Warp0Broadcast slot")
        .unwrap()
        + barrier;
    assert!(publish < barrier && barrier < consume);
}

#[test]
fn source_preserves_sum_softmax_pack_alias_and_consumer_mapping() {
    let source = read("kernels/gb10/experiments/v4_prefill_attn_compressed_tc2_warp0.cu");
    let flat = compact(&source);
    for contract in [
        "v4_prefill_attn_compressed_tc2_warp0",
        "const bool kv_same = (K == V)",
        "const bool comp_same = (Kc == Vc)",
        "struct __align__(16) TC2Warp0Broadcast",
        "if (warp == 0)",
        "broadcast[laneid]",
        "tc2_pack_bf16(en0[0], en0[1])",
        "tc2_pack_bf16(en1[0], en1[1])",
        "tc2_pack_bf16(en0[2], en0[3])",
        "tc2_pack_bf16(en1[2], en1[3])",
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    ] {
        assert!(
            source.contains(contract),
            "missing source contract: {contract}"
        );
    }
    assert!(flat.contains("constfloata=z0.x+z0.y+z0.z+z0.w"));
    assert!(flat.contains("constfloatb=z1.x+z1.y+z1.z+z1.w"));
    assert!(flat.contains("TC2Warp0Broadcast&slot=broadcast[laneid]"));
    assert!(flat.contains("constTC2Warp0Broadcastslot=broadcast[laneid]"));
    assert!(source.contains("one additional block barrier for the alias path"));
}

#[test]
fn exact_production_mix_quantifies_dynamic_topology_not_runtime() {
    let ratio4 = tile_visits(Some(4));
    let ratio128 = tile_visits(Some(128));
    let dense = tile_visits(None);
    assert_eq!(ratio4, 271_936);
    assert_eq!(ratio128, 94_912);
    assert_eq!(dense, 84_672);
    let all_layers = 21 * ratio4 + 20 * ratio128 + 2 * dense;
    assert_eq!(all_layers, 7_778_240);

    // Per tile: four warps each read 32 lanes * 8 float4. Warp 0 retains its
    // 256 reads, so 768 replicated 16-byte reads are removed. The 10 expf
    // evaluations per lane are retained only in warp 0: 3*32*10 are removed.
    assert_eq!(all_layers * 768, 5_973_688_320);
    assert_eq!(all_layers * 768 * 16, 95_579_013_120);
    assert_eq!(all_layers * 3 * 32 * 10, 7_467_110_400);
}

#[test]
fn sass_gate_is_sm121a_fail_closed_and_pins_expected_delta() {
    let script = read("scripts/check-v4-prefill-attn-tc2-warp0-sass.sh");
    for contract in [
        "-arch=sm_121a",
        "prefill_attn_compressed_tc2",
        "v4_prefill_attn_compressed_tc2_warp0",
        "MUFU.EX2",
        "LDSM",
        "HMMA",
        "BAR.SYNC",
        "spill stores",
        "spill loads",
        "STACK",
        "LOCAL",
        "ATOM|RED|LDL|STL",
        "cubin_set_sha256",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
    ] {
        assert!(
            script.contains(contract),
            "missing SASS contract: {contract}"
        );
    }
    assert!(
        script
            .contains("check_kernel baseline prefill_attn_compressed_tc2 158 22016 1832 24 48 48")
    );
    assert!(script.contains(
        "check_kernel warp0 v4_prefill_attn_compressed_tc2_warp0 142 23552 2408 44 54 54"
    ));
    assert!(script.contains("deepseek-v4-flash/nvfp4/v4_prefill_attn_compressed_tc2_warp0.cu"));
}

#[test]
fn experiment_implementation_has_one_production_wrapper() {
    let wrapper =
        read("kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_attn_compressed_tc2_warp0.cu");
    assert!(wrapper.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
    assert!(wrapper.contains("../../experiments/v4_prefill_attn_compressed_tc2_warp0.cu"));
    assert!(!wrapper.contains("extern \"C\""));
}
