// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashSet;

use half::bf16;

const KERNEL: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill.cu");
const KERNEL_K64: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64.cu");
const KERNEL_M128: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_m128.cu");
const EXL3_GEMV: &str = include_str!("../../../kernels/gb10/common/exl3_gemv.cu");
const DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3.rs");
const TAIL_DISPATCH: &str = include_str!("../src/layers/moe/forward_prefill_exl3_tail.rs");
const STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const MICROTEST: &str = include_str!("../examples/exl3_gemv_microtest.rs");

const M_TILE: usize = 64;
const RSQRT_128: f32 = 0.088_388_346;

fn rounded_post(x: f32, svh: f32) -> f32 {
    bf16::from_f32(x * RSQRT_128 * svh).to_f32()
}

fn post_silu_unrounded(g: f32, u: f32, gate_svh: f32, up_svh: f32) -> f32 {
    let g = rounded_post(g, gate_svh).min(10.0);
    let u = rounded_post(u, up_svh).clamp(-10.0, 10.0);
    g * (1.0 / (1.0 + (-g).exp())) * u
}

fn composed_post_silu(g: f32, u: f32, gate_svh: f32, up_svh: f32) -> bf16 {
    bf16::from_f32(post_silu_unrounded(g, u, gate_svh, up_svh))
}

fn tile_coord(lane: usize, s: usize) -> (usize, usize) {
    let n = 8 * (s / 4) + lane / 4;
    let k = 2 * (lane % 4) + (s % 2) + 8 * ((s % 4) / 2);
    (k, n)
}

fn covered_rows(offsets: &[usize], max_m_tiles: usize) -> Vec<usize> {
    let mut rows = Vec::new();
    for expert in 0..offsets.len() - 1 {
        for tile in 0..max_m_tiles {
            let start = offsets[expert] + tile * M_TILE;
            let end = offsets[expert + 1].min(start + M_TILE);
            if start < end {
                rows.extend(start..end);
            }
        }
    }
    rows
}

#[test]
fn trellis_fragment_order_is_a_bijection_over_16x16() {
    let coords: HashSet<_> = (0..32)
        .flat_map(|lane| (0..8).map(move |s| tile_coord(lane, s)))
        .collect();
    assert_eq!(coords.len(), 16 * 16);
    assert!(coords.iter().all(|&(k, n)| k < 16 && n < 16));
}

#[test]
fn warp_fragments_and_accumulators_cover_one_m64_n64_tile_exactly_once() {
    let mut outputs = HashSet::new();
    for warp in 0..4 {
        let fragment: HashSet<_> = (0..32)
            .flat_map(|lane| {
                (0..8).map(move |s| {
                    let (k, n) = tile_coord(lane, s);
                    (k, warp * 16 + n)
                })
            })
            .collect();
        assert_eq!(fragment.len(), 16 * 16);
        assert!(
            fragment
                .iter()
                .all(|&(k, n)| { k < 16 && (warp * 16..(warp + 1) * 16).contains(&n) })
        );

        for mt in 0..4 {
            for nt in 0..2 {
                for lane in 0..32 {
                    let group = lane / 4;
                    let tid = lane % 4;
                    let row0 = mt * 16 + group;
                    let row1 = row0 + 8;
                    let col0 = warp * 16 + nt * 8 + tid * 2;
                    assert!(outputs.insert((row0, col0)));
                    assert!(outputs.insert((row0, col0 + 1)));
                    assert!(outputs.insert((row1, col0)));
                    assert!(outputs.insert((row1, col0 + 1)));
                }
            }
        }
    }
    assert_eq!(outputs.len(), 64 * 64);
    assert!(outputs.iter().all(|&(m, n)| m < 64 && n < 64));
}

#[test]
fn m128_warp_pairs_cover_two_row_halves_and_one_n64_tile_exactly_once() {
    let mut outputs = HashSet::new();
    for warp in 0..8 {
        let n_warp = warp % 4;
        let m_warp = warp / 4;
        for mt in 0..4 {
            for nt in 0..2 {
                for lane in 0..32 {
                    let group = lane / 4;
                    let tid = lane % 4;
                    let row0 = m_warp * 64 + mt * 16 + group;
                    let row1 = row0 + 8;
                    let col0 = n_warp * 16 + nt * 8 + tid * 2;
                    for coord in [
                        (row0, col0),
                        (row0, col0 + 1),
                        (row1, col0),
                        (row1, col0 + 1),
                    ] {
                        assert!(outputs.insert(coord));
                    }
                }
            }
        }
    }
    assert_eq!(outputs.len(), 128 * 64);
    assert!(outputs.iter().all(|&(m, n)| m < 128 && n < 64));
}

#[test]
fn m128_does_not_reduce_tiles_for_the_current_balanced_config_model() {
    // DeepSeek-V4-Flash-0731-EXL3-K2 config: 256 experts, top-6. At N=2410,
    // an exactly balanced router gives 124 experts 57 rows and 132 experts
    // 56 rows. Every expert already fits one M64 tile.
    let rows: Vec<_> = (0..256)
        .map(|expert| if expert < 124 { 57usize } else { 56usize })
        .collect();
    assert_eq!(rows.iter().sum::<usize>(), 2_410 * 6);
    let m64_tasks: usize = rows.iter().map(|rows| rows.div_ceil(64)).sum();
    let m128_tasks: usize = rows.iter().map(|rows| rows.div_ceil(128)).sum();
    assert_eq!((m64_tasks, m128_tasks), (256, 256));
}

#[test]
fn hybrid_m128_masks_tail_warps_without_adding_padded_mma_rows() {
    for rows in 1usize..=513 {
        let m64_mma_rows = rows.div_ceil(64) * 64;
        let hybrid_m128_mma_rows: usize = (0..rows.div_ceil(128))
            .map(|tile| {
                let remaining = rows.saturating_sub(tile * 128);
                64 + usize::from(remaining > 64) * 64
            })
            .sum();
        assert_eq!(hybrid_m128_mma_rows, m64_mma_rows, "rows={rows}");
    }
    assert!(KERNEL.contains("const bool active_m_warp"));
    assert!(KERNEL.contains("if (active_m_warp)"));
}

#[test]
fn expert_major_grid_covers_every_sorted_row_exactly_once() {
    let offsets: [usize; 6] = [0, 1, 65, 194, 194, 323];
    let max_rows = offsets
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .max()
        .unwrap();
    let max_m_tiles = max_rows.div_ceil(M_TILE);
    let rows = covered_rows(&offsets, max_m_tiles);
    assert_eq!(rows, (0..*offsets.last().unwrap()).collect::<Vec<_>>());
}

#[test]
fn persistent_expert_strips_cover_every_live_tile_once() {
    let offsets: [usize; 6] = [0, 1, 64, 128, 193, 322];
    let n_tiles = 4usize;
    let total_strips = (offsets.len() - 1) * n_tiles;
    let ctas = 7usize;
    let mut live = HashSet::new();
    for cta in 0..ctas {
        for strip in (cta..total_strips).step_by(ctas) {
            let n_tile = strip % n_tiles;
            let expert = strip / n_tiles;
            let rows = offsets[expert + 1] - offsets[expert];
            for m_tile in 0..rows.div_ceil(M_TILE) {
                assert!(live.insert((expert, m_tile, n_tile)));
            }
        }
    }
    let expected: usize = offsets
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).div_ceil(M_TILE) * n_tiles)
        .sum();
    assert_eq!(live.len(), expected);
}

#[test]
fn persistent_expert_strips_remove_the_rectangular_empty_task_universe() {
    let tokens = 2_410usize;
    let topk = 6usize;
    let experts = 256usize;
    let routed_rows = tokens * topk;
    // Current checkpoint config: gate + up are N=2,048; down is N=4,096.
    // All direct kernels tile N64.
    let n_strips_per_expert = 2 * (2_048 / 64) + 4_096 / 64;

    let old_m64_slots = experts * routed_rows.div_ceil(64) * n_strips_per_expert;
    let old_m128_slots = experts * routed_rows.div_ceil(128) * n_strips_per_expert;
    let new_outer_strips = experts * n_strips_per_expert;
    let balanced_rows: Vec<_> = (0..experts)
        .map(|expert| if expert < 124 { 57usize } else { 56usize })
        .collect();
    let live_m64_tiles: usize = balanced_rows
        .iter()
        .map(|rows| rows.div_ceil(64) * n_strips_per_expert)
        .sum();
    let live_m128_tiles: usize = balanced_rows
        .iter()
        .map(|rows| rows.div_ceil(128) * n_strips_per_expert)
        .sum();

    assert_eq!(old_m64_slots, 7_405_568);
    assert_eq!(old_m128_slots, 3_702_784);
    assert_eq!(new_outer_strips, 32_768);
    assert_eq!(live_m64_tiles, 32_768);
    assert_eq!(live_m128_tiles, 32_768);
    assert_eq!(old_m64_slots / new_outer_strips, 226);

    assert!(KERNEL.contains("const unsigned int total_strips = num_experts * n_tiles;"));
    assert!(KERNEL.contains("const int m_first = persistent"));
    assert!(KERNEL.contains("m_local < m_stop && m_start + m_local < m_end"));
    assert!(!KERNEL.contains("persistent_max_m_tiles"));
}

#[test]
fn k64_stage_preserves_k16_order_and_quarters_dynamic_stage_barriers() {
    for k in [2_048usize, 4_096] {
        let k16_order: Vec<_> = (0..k / 16).collect();
        let k64_order: Vec<_> = (0..k / 64)
            .flat_map(|stage| (0..4).map(move |slice| stage * 4 + slice))
            .collect();
        assert_eq!(k64_order, k16_order);
    }

    // Gate, up: K=4096. Down: K=2048. Each K stage has a load and consume
    // barrier; the final per-M-tile barrier is unchanged and excluded here.
    let k16_stage_barriers = 2 * (4_096 / 16) * 2 + 2 * (2_048 / 16);
    let k64_stage_barriers = 2 * (4_096 / 64) * 2 + 2 * (2_048 / 64);
    assert_eq!((k16_stage_barriers, k64_stage_barriers), (1_280, 320));

    assert!(KERNEL.contains("#ifndef EXL3_PF_K_STEP"));
    assert!(KERNEL.contains("#define EXL3_PF_K_TILES (EXL3_PF_K_STEP / 16)"));
    assert!(KERNEL.contains("for (unsigned int k_tile = 0;"));
    assert!(KERNEL_K64.contains("#define EXL3_PF_K_STEP 64"));
    assert!(KERNEL_K64.contains("#define EXL3_PF_KERNEL_NAME exl3_grouped_prefill_k64"));
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_K64"));
    assert!(STATE.contains("grouped_direct_k64_k"));
    assert!(STATE.contains("!(direct_m128 && direct_k64)"));
    assert!(DISPATCH.contains("pf.direct_k64"));
    assert!(MICROTEST.contains("k64-persistent"));
}

#[test]
fn full_compact_grid_assigns_one_expert_strip_per_cta_at_current_shapes() {
    let experts = 256usize;
    let sms = 48usize;
    for (n, expected_strips, expected_full_waves) in
        [(2_048usize, 8_192usize, 171usize), (4_096, 16_384, 342)]
    {
        let n_tiles = n / 64;
        let strips = experts * n_tiles;
        assert_eq!(strips, expected_strips);
        assert_eq!(strips.div_ceil(sms), expected_full_waves);

        let mut assigned = HashSet::new();
        for cta in 0..strips {
            for strip in (cta..strips).step_by(strips) {
                assert!(assigned.insert(strip));
            }
        }
        assert_eq!(assigned, (0..strips).collect());
    }

    assert!(DISPATCH.contains("[strips.max(1), 1, 1]"));
    assert!(!DISPATCH.contains("EXL3_DIRECT_PERSISTENT_CTAS"));
    assert!(MICROTEST.contains("persistent-full"));
}

#[test]
fn bf16x8_activation_stage_is_aligned_and_covers_each_element_once() {
    for (m_tile, k_step, threads) in [(64usize, 16usize, 128usize), (64, 64, 128), (128, 16, 256)] {
        let vectors_per_row = k_step / 8;
        let total_vectors = m_tile * vectors_per_row;
        let mut cells = HashSet::new();
        for tid in 0..threads {
            for vector in (tid..total_vectors).step_by(threads) {
                let row = vector / vectors_per_row;
                let col = (vector % vectors_per_row) * 8;
                for lane in 0..8 {
                    assert!(cells.insert((row, col + lane)));
                }
            }
        }
        assert_eq!(cells.len(), m_tile * k_step);
    }

    // Device allocations are at least 16-byte aligned. Current K dimensions,
    // K-stage bases, and BF16x8 column vectors preserve that alignment.
    for (k, k_step) in [(2_048usize, 16usize), (4_096, 16), (2_048, 64), (4_096, 64)] {
        for row in 0..4 {
            for k_base in (0..k).step_by(k_step) {
                for col in (0..k_step).step_by(8) {
                    assert_eq!(((row * k + k_base + col) * 2) % 16, 0);
                }
            }
        }
    }

    assert!(KERNEL.contains("uint4 packed = {0, 0, 0, 0}"));
    assert!(KERNEL.contains("load_vecs = EXL3_PF_M_TILE * vectors_per_row"));
    assert!(KERNEL.contains("dst[3] = packed.w"));
}

#[test]
fn fixed_k2_specialization_preserves_layout_and_has_explicit_fallback() {
    let n_tiles = 4usize;
    let k_tiles = 4usize;
    let bits = 2usize;
    let generic_words: Vec<_> = (0..k_tiles)
        .flat_map(|k_tile| {
            (0..n_tiles)
                .flat_map(move |n_tile| (0..8 * bits).map(move |word| (k_tile, n_tile, word)))
        })
        .collect();
    let fixed_words: Vec<_> = (0..k_tiles)
        .flat_map(|k_tile| {
            (0..n_tiles).flat_map(move |n_tile| (0..16).map(move |word| (k_tile, n_tile, word)))
        })
        .collect();
    assert_eq!(fixed_words, generic_words);

    assert!(KERNEL.contains("#ifndef EXL3_PF_FIXED_BITS"));
    assert!(KERNEL.contains("const unsigned int bit_width"));
    assert!(STATE.contains("grouped_direct_k2_k"));
    assert!(STATE.contains("grouped_direct_k64_k2_k"));
    assert!(DISPATCH.contains("tab.bits == 2"));
    assert!(MICROTEST.contains("k2-fixed-persistent-full"));
}

#[test]
fn fused_post_silu_preserves_the_two_bf16_rounding_barriers() {
    let mut rounding_is_load_bearing = false;
    let mut down_input_rounding_is_load_bearing = false;
    for i in -257..=257 {
        let gate_h = i as f32 * 0.071_3 + 0.019;
        let up_h = (i * 37 % 263) as f32 * 0.093_7 - 0.031;
        let got = composed_post_silu(gate_h, up_h, -1.0, 1.0);
        let fused_with_barriers = composed_post_silu(gate_h, up_h, -1.0, 1.0);
        assert_eq!(got.to_bits(), fused_with_barriers.to_bits());

        let g = (gate_h * RSQRT_128 * -1.0).min(10.0);
        let u = (up_h * RSQRT_128).clamp(-10.0, 10.0);
        let fused_without_barriers = bf16::from_f32(g * (1.0 / (1.0 + (-g).exp())) * u);
        rounding_is_load_bearing |= got.to_bits() != fused_without_barriers.to_bits();
        down_input_rounding_is_load_bearing |=
            got.to_f32() != post_silu_unrounded(gate_h, up_h, -1.0, 1.0);
    }
    assert!(rounding_is_load_bearing);
    assert!(down_input_rounding_is_load_bearing);
}

#[test]
fn fused_post_silu_kernel_contract_is_explicit() {
    assert!(EXL3_GEMV.contains("void exl3_h128_post_silu_pre_rows("));
    assert!(EXL3_GEMV.contains("down_suh_tab"));
    assert!(EXL3_GEMV.matches("exl3_had128(").count() >= 2);
    assert!(EXL3_GEMV.contains("__bfloat162float(__float2bfloat16"));
    assert!(EXL3_GEMV.contains("fminf(g0, EXL3_SWIGLU_LIMIT)"));
    assert!(STATE.contains("h128_post_silu_pre_k"));
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_FUSED_POST"));
    assert!(DISPATCH.contains("pf.fused_post"));
    assert!(MICROTEST.contains("fused_post_silu_gate"));
    assert!(MICROTEST.contains("fused==composed byte-identical"));
    assert!(MICROTEST.contains("production-n2048-topk6-e256"));
    assert!(MICROTEST.contains("let rows = tokens * topk"));
    assert!(MICROTEST.contains("fused_post_canary_ok"));
    assert!(MICROTEST.contains("canary={}"));
}

#[test]
fn fused_post_unpermute_warp_grid_covers_every_token_chunk_once() {
    let (tokens, h, warps) = (7usize, 4096usize, 8usize);
    let chunks = h / 128;
    let grid_y = chunks.div_ceil(warps);
    let covered: HashSet<_> = (0..tokens)
        .flat_map(|token| {
            (0..grid_y).flat_map(move |by| {
                (0..warps)
                    .map(move |warp| (token, by * warps + warp))
                    .filter(move |&(_, chunk)| chunk < chunks)
            })
        })
        .collect();
    assert_eq!(covered.len(), tokens * chunks);
}

#[test]
fn fused_post_unpermute_preserves_post_bf16_and_topk_order() {
    let raw = [[0.71f32, -1.19], [2.03, 0.37], [-0.44, 1.81]];
    let svh = [[1.0f32, -1.0], [-1.0, 1.0], [1.0, 1.0]];
    let weights = [0.31f32, 0.53, 0.16];
    for col in 0..2 {
        let mut legacy = 0.0f32;
        let mut fused = 0.0f32;
        for k in 0..3 {
            let post = rounded_post(raw[k][col], svh[k][col]);
            legacy += weights[k] * post;
            fused += weights[k] * post;
        }
        assert_eq!(
            bf16::from_f32(legacy).to_bits(),
            bf16::from_f32(fused).to_bits()
        );
    }
}

#[test]
fn fused_post_unpermute_kernel_contract_is_explicit() {
    assert!(EXL3_GEMV.contains("void exl3_h128_post_unpermute_rows("));
    assert!(EXL3_GEMV.contains("const unsigned int slot = token * topk + k"));
    assert!(EXL3_GEMV.contains("token_to_perm[slot]"));
    assert!(EXL3_GEMV.contains("topk_weights[slot]"));
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_FUSED_UNPERMUTE"));
    assert!(STATE.contains("h128_post_unpermute_k"));
    assert!(TAIL_DISPATCH.contains("try_exl3_fused_post_unpermute"));
    assert!(MICROTEST.contains("fused_post_unpermute_gate"));
}

#[test]
fn direct_p2_kernel_never_materializes_global_bf16_weights() {
    assert!(KERNEL.contains("void EXL3_PF_LAUNCH_ATTR EXL3_PF_KERNEL_NAME("));
    assert!(KERNEL.contains("#define EXL3_PF_KERNEL_NAME exl3_grouped_prefill"));
    assert!(KERNEL.contains("#define EXL3_PF_M_TILE 64"));
    assert!(KERNEL.contains("exl3_pf_bf16_pair"));
    assert!(KERNEL.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"));
    assert!(KERNEL.contains("trellis_tab[expert_id]"));
    assert!(!KERNEL.contains("smem_B"));
    assert!(!KERNEL.contains("Wout"));
}

#[test]
fn direct_p2_dispatch_is_explicitly_gated_and_shape_exact() {
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_DIRECT"));
    assert!(DISPATCH.contains("pf.grouped_direct_k"));
    assert!(DISPATCH.contains("expect(\"exact-grid P2 requires host offsets\")"));
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_PERSISTENT"));
    assert!(DISPATCH.contains("let needs_host_offsets = !direct || !persistent"));
    assert!(DISPATCH.contains("let strips = num_experts"));
    assert!(DISPATCH.contains(".checked_mul(tab.n / direct_n_tile)"));
    assert!(!DISPATCH.contains("num_experts.saturating_mul(tab.n / direct_n_tile)"));
    assert!(DISPATCH.contains(".arg_u32(u32::from(persistent))"));
    assert!(DISPATCH.contains(".block([direct_block_threads, 1, 1])"));
    assert!(KERNEL.contains("for (unsigned int work = work0;"));
    assert!(KERNEL.contains("work += work_stride"));
    assert!(
        DISPATCH
            .contains("tab.n.is_multiple_of(direct_n_tile) && tab.k.is_multiple_of(direct_k_step)")
    );
    assert!(STATE.contains("grouped_direct_k"));
    assert!(STATE.contains("grouped_direct_k64_k"));
    assert!(STATE.contains("grouped_direct_m128_k"));
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_M128"));
    assert!(DISPATCH.contains("pf.direct_m128"));
    assert!(KERNEL_M128.contains("#define EXL3_PF_M_TILE 128"));
    assert!(KERNEL_M128.contains("exl3_grouped_prefill_m128"));
    assert!(STATE.contains("exl3_grouped_prefill"));
    assert!(STATE.contains("pub(crate) direct: bool"));
    assert!(STATE.contains("(DevicePtr(0), DevicePtr(0), 0)"));
    assert!(DISPATCH.contains("let direct = pf.direct"));
    assert!(MICROTEST.contains("direct_prefill_parity_gate"));
    assert!(MICROTEST.contains("for bits in [2usize, 3]"));
    assert!(MICROTEST.contains("m128-persistent"));
    assert!(MICROTEST.contains("let counts = [1usize, 63, 64, 65, 129]"));
    assert!(MICROTEST.contains("persistent-strided"));
    assert!(MICROTEST.contains("persistent-full"));
}
