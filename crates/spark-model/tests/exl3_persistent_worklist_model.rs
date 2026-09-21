// SPDX-License-Identifier: AGPL-3.0-only

use spark_model::layers::moe::persistent_work::{
    EXL3_M16_MAX_ROWS, EXL3_M16_WORK_CAPACITY, EXL3_M16_WORK_RECORD_BYTES, EXL3_WORK_RECORD_BYTES,
    Exl3PersistentM16Work, Exl3PersistentWork, PersistentWorkError, try_exl3_worklist,
    try_exl3_worklist_m16,
};

const KERNEL: &str = include_str!("../../../kernels/gb10/common/exl3_gemv.cu");
const DISPATCH: &str = include_str!("../src/layers/moe/exl3_decode.rs");

#[test]
fn exl3_wire_layout_is_exactly_32_bytes_and_16_aligned() {
    assert_eq!(EXL3_WORK_RECORD_BYTES, 32);
    assert_eq!(std::mem::size_of::<Exl3PersistentWork>(), 32);
    assert_eq!(std::mem::align_of::<Exl3PersistentWork>(), 16);
    let work = Exl3PersistentWork::new(7, &[3, 11, 19]);
    assert_eq!(work.expert, 7);
    assert_eq!(work.count, 3);
    assert_eq!(work.slots, [3, 11, 19, 3, 3, 3]);
    assert_eq!(
        Exl3PersistentWork::from_bytes(work.to_bytes()).unwrap(),
        work
    );
}

#[test]
fn exl3_wire_decoder_rejects_zero_and_overwide_counts() {
    let mut bytes = Exl3PersistentWork::new(7, &[3]).to_bytes();
    bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        Exl3PersistentWork::from_bytes(bytes),
        Err(PersistentWorkError::InvalidCount(0))
    );
    bytes[4..8].copy_from_slice(&7u32.to_le_bytes());
    assert_eq!(
        Exl3PersistentWork::from_bytes(bytes),
        Err(PersistentWorkError::InvalidCount(7))
    );
}

#[test]
fn exl3_m16_wire_layout_is_exactly_80_bytes_with_zero_padding() {
    assert_eq!(EXL3_M16_WORK_RECORD_BYTES, 80);
    assert_eq!(std::mem::size_of::<Exl3PersistentM16Work>(), 80);
    assert_eq!(std::mem::align_of::<Exl3PersistentM16Work>(), 16);
    let work = Exl3PersistentM16Work::new(7, &[3, 11, 19]);
    assert_eq!(work.expert, 7);
    assert_eq!(work.count, 3);
    assert_eq!(work.active_slots(), &[3, 11, 19]);
    assert!(work.slots[3..].iter().all(|&slot| slot == 3));
    let bytes = work.to_bytes();
    assert_eq!(&bytes[72..80], &[0; 8]);
    assert_eq!(Exl3PersistentM16Work::from_bytes(bytes).unwrap(), work);
}

#[test]
fn exl3_m16_wire_decoder_rejects_invalid_counts_and_nonzero_padding() {
    assert_eq!(
        Exl3PersistentM16Work::try_new(7, &[]),
        Err(PersistentWorkError::InvalidCount(0))
    );
    assert_eq!(
        Exl3PersistentM16Work::try_new(7, &[3; EXL3_M16_MAX_ROWS + 1]),
        Err(PersistentWorkError::InvalidCount(EXL3_M16_MAX_ROWS + 1))
    );

    let mut bytes = Exl3PersistentM16Work::new(7, &[3]).to_bytes();
    bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        Exl3PersistentM16Work::from_bytes(bytes),
        Err(PersistentWorkError::InvalidCount(0))
    );
    bytes[4..8].copy_from_slice(&17u32.to_le_bytes());
    assert_eq!(
        Exl3PersistentM16Work::from_bytes(bytes),
        Err(PersistentWorkError::InvalidCount(17))
    );
    bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    bytes[79] = 1;
    assert_eq!(
        Exl3PersistentM16Work::from_bytes(bytes),
        Err(PersistentWorkError::InvalidPadding)
    );
}

#[test]
fn cpu_reference_compacts_first_seen_experts_and_slots() {
    let routing = [vec![7, 2, 9, 5], vec![2, 9, 7, 8], vec![7, 4, 8, 2]];
    let work = try_exl3_worklist(&routing, 4, 16, 12).unwrap();
    assert_eq!(
        work.iter().map(|w| w.expert).collect::<Vec<_>>(),
        [7, 2, 9, 5, 8, 4]
    );
    assert_eq!(work[0].active_slots(), &[0, 6, 8]);
    assert_eq!(work[1].active_slots(), &[1, 4, 11]);
    assert_eq!(work[2].active_slots(), &[2, 5]);
    assert_eq!(work[3].active_slots(), &[3]);
    assert_eq!(work[4].active_slots(), &[7, 10]);
    assert_eq!(work[5].active_slots(), &[9]);
}

#[test]
fn cpu_reference_rejects_capacity_before_truncation() {
    let routing = [vec![0, 1, 2], vec![3, 4, 5]];
    assert_eq!(
        try_exl3_worklist(&routing, 3, 8, 5),
        Err(PersistentWorkError::CapacityExceeded {
            required: 6,
            capacity: 5,
        })
    );
}

#[test]
fn cpu_reference_rejects_a_duplicate_in_the_last_row_before_emission() {
    let routing = vec![
        vec![0, 1, 2, 3, 4, 5],
        vec![0, 6, 7, 8, 9, 10],
        vec![0, 11, 12, 13, 14, 15],
        vec![0, 16, 17, 18, 19, 20],
        vec![0, 21, 22, 23, 24, 25],
        vec![0, 0, 26, 27, 28, 29],
    ];
    assert_eq!(
        try_exl3_worklist(&routing, 6, 64, 36),
        Err(PersistentWorkError::DuplicateExpert { row: 5, expert: 0 })
    );
}

#[test]
fn cpu_reference_covers_every_routed_slot_once_on_adversarial_routes() {
    let mut seed = 0xD54A_71A5_u64;
    for _ in 0..512 {
        let mut routing = vec![Vec::with_capacity(6); 6];
        for row in &mut routing {
            while row.len() < 6 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let expert = (seed % 64) as u32;
                if !row.contains(&expert) {
                    row.push(expert);
                }
            }
        }
        let work = try_exl3_worklist(&routing, 6, 64, 36).unwrap();
        let mut covered = [0u8; 36];
        for record in &work {
            assert!((1..=6).contains(&record.count));
            for &slot in record.active_slots() {
                assert_eq!(routing[slot as usize / 6][slot as usize % 6], record.expert);
                covered[slot as usize] += 1;
            }
            assert!(
                record.slots[record.count as usize..]
                    .iter()
                    .all(|&slot| slot == record.slots[0])
            );
        }
        assert_eq!(covered, [1; 36]);
    }
}

#[test]
fn m16_cpu_reference_covers_overlap_no_overlap_and_random_routes() {
    let overlap = vec![(0..6).collect::<Vec<u32>>(); 16];
    let overlap_work = try_exl3_worklist_m16(&overlap, 6, 256, 96).unwrap();
    assert_eq!(overlap_work.len(), 6);
    assert!(overlap_work.iter().all(|record| record.count == 16));

    let no_overlap: Vec<Vec<u32>> = (0..16)
        .map(|row| (0..6).map(|column| row * 6 + column).collect())
        .collect();
    let no_overlap_work = try_exl3_worklist_m16(&no_overlap, 6, 256, 96).unwrap();
    assert_eq!(no_overlap_work.len(), 96);
    assert_eq!(
        try_exl3_worklist_m16(&no_overlap, 6, 256, 95),
        Err(PersistentWorkError::CapacityExceeded {
            required: 96,
            capacity: 95,
        })
    );

    let mut seed = 0xD54A_71A5_u64;
    for _ in 0..128 {
        let mut routing = vec![Vec::with_capacity(6); 16];
        for row in &mut routing {
            while row.len() < 6 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let expert = (seed % 96) as u32;
                if !row.contains(&expert) {
                    row.push(expert);
                }
            }
        }
        let work = try_exl3_worklist_m16(&routing, 6, 96, 96).unwrap();
        let mut covered = [0u8; 96];
        for record in work {
            for slot in record.active_slots() {
                assert_eq!(
                    routing[*slot as usize / 6][*slot as usize % 6],
                    record.expert
                );
                covered[*slot as usize] += 1;
            }
        }
        assert_eq!(covered, [1; 96]);
    }
}

#[test]
fn m16_cpu_reference_rejects_last_row_duplicate_and_out_of_range() {
    let mut duplicate = vec![(0..6).collect::<Vec<u32>>(); 16];
    duplicate[15] = vec![0, 1, 2, 3, 4, 0];
    assert_eq!(
        try_exl3_worklist_m16(&duplicate, 6, 256, 96),
        Err(PersistentWorkError::DuplicateExpert { row: 15, expert: 0 })
    );
    let mut out_of_range = vec![(0..6).collect::<Vec<u32>>(); 16];
    out_of_range[15][5] = 256;
    assert_eq!(
        try_exl3_worklist_m16(&out_of_range, 6, 256, 96),
        Err(PersistentWorkError::ExpertOutOfRange {
            row: 15,
            expert: 256,
            num_experts: 256,
        })
    );

    let too_many_rows = vec![vec![0]; EXL3_M16_MAX_ROWS + 1];
    assert_eq!(
        try_exl3_worklist_m16(&too_many_rows, 1, 256, EXL3_M16_WORK_CAPACITY),
        Err(PersistentWorkError::InvalidRowCount(EXL3_M16_MAX_ROWS + 1))
    );

    let over_capacity: Vec<Vec<u32>> = (0..EXL3_M16_MAX_ROWS)
        .map(|row| (0..7).map(|column| (row * 7 + column) as u32).collect())
        .collect();
    assert_eq!(
        try_exl3_worklist_m16(&over_capacity, 7, 256, usize::MAX),
        Err(PersistentWorkError::CapacityExceeded {
            required: 112,
            capacity: EXL3_M16_WORK_CAPACITY,
        })
    );
}

#[test]
fn fixed_96_cta_task_mapping_is_a_bijection_and_split_balanced() {
    const CTAS: usize = 96;
    for capacity in [36usize, 96] {
        for records in 0..=capacity {
            for splits in 1..=12usize {
                let gu_total = records * 2 * splits * 16;
                let mut gu_seen = vec![0u8; gu_total];
                let mut gu_counter = vec![0u8; records * 2 * 16];
                for physical in 0..CTAS {
                    for logical in (physical..gu_total).step_by(CTAS) {
                        let mut q = logical;
                        let strip = q % 16;
                        q /= 16;
                        let split = q % splits;
                        q /= splits;
                        let projection = q & 1;
                        let record = q >> 1;
                        assert!(record < records && projection < 2 && split < splits && strip < 16);
                        gu_seen[logical] += 1;
                        gu_counter[(record * 2 + projection) * 16 + strip] += 1;
                    }
                }
                assert!(gu_seen.iter().all(|&count| count == 1));
                assert!(gu_counter.iter().all(|&count| count == splits as u8));

                let down_total = records * splits * 32;
                let mut down_seen = vec![0u8; down_total];
                let mut down_counter = vec![0u8; records * 32];
                for physical in 0..CTAS {
                    for logical in (physical..down_total).step_by(CTAS) {
                        let mut q = logical;
                        let strip = q % 32;
                        q /= 32;
                        let split = q % splits;
                        let record = q / splits;
                        assert!(record < records && split < splits && strip < 32);
                        down_seen[logical] += 1;
                        down_counter[record * 32 + strip] += 1;
                    }
                }
                assert!(down_seen.iter().all(|&count| count == 1));
                assert!(down_counter.iter().all(|&count| count == splits as u8));
            }
        }
    }
}

#[test]
fn cuda_builder_and_fixed_grid_consumers_share_bounded_state() {
    for needle in [
        "struct __align__(16) Exl3PersistentM6Work",
        "static_assert(sizeof(Exl3PersistentM6Work) == 32",
        "exl3_build_m6_worklist",
        "exl3_gemv_mrow_persistent_gate_up_m6",
        "exl3_gemv_mrow_persistent_down_m6",
        "struct __align__(16) Exl3PersistentM16Work",
        "static_assert(sizeof(Exl3PersistentM16Work) == 80",
        "exl3_build_m16_worklist",
        "exl3_gemv_mrow_persistent_gate_up_m16",
        "exl3_gemv_mrow_persistent_down_m16",
        "for (unsigned int logical = blockIdx.x; logical < total_tasks; logical += gridDim.x)",
        "counters + group * n_blocks",
        "counters + work_index * n_blocks",
        "gridDim.x == 96",
        "blockDim.x == 256",
        "state->count <= 36",
        "Validate the complete route before writing any record",
        "if (count == 6)",
        "exl3_zero_m6_routed<2048>(gate_out)",
        "exl3_zero_m6_routed<2048>(up_out)",
        "exl3_zero_m6_routed<4096>(down_out)",
        "exl3_zero_routed<16, 2048>(gate_out)",
        "exl3_zero_routed<16, 2048>(up_out)",
        "exl3_zero_routed<16, 4096>(down_out)",
    ] {
        assert!(
            KERNEL.contains(needle),
            "missing CUDA worklist contract: {needle}"
        );
    }
}

#[test]
fn reusable_mrow_body_uses_only_explicit_logical_coordinates() {
    let start = KERNEL.find("void exl3_gemv_mrow_body(").unwrap();
    let end = KERNEL[start..]
        .find("// Register-budget hint")
        .map(|offset| start + offset)
        .unwrap();
    let body = &KERNEL[start..end];
    assert!(!body.contains("blockIdx"));
    assert!(!body.contains("gridDim"));
    for needle in [
        "unsigned int n_block, unsigned int split, unsigned int splits",
        "const int n0 = n_block * EXL3_NSTRIP",
        "atomicAdd(&counters[n_block], 1)",
    ] {
        assert!(
            body.contains(needle),
            "missing logical-coordinate contract: {needle}"
        );
    }
}

#[test]
fn persistent_dispatch_stays_same_stream_and_preserves_stage_order() {
    let start = DISPATCH
        .find("fn dispatch_exl3_persistent_verify(")
        .unwrap();
    let end = DISPATCH[start..]
        .find("/// `num_tokens`-row speculative-verify")
        .map(|offset| start + offset)
        .unwrap();
    let body = &DISPATCH[start..end];
    let build = body
        .find("KernelLaunch::new(gpu, st.persistent_build_k[arm])")
        .unwrap();
    let gate_up = body
        .find("KernelLaunch::new(gpu, st.persistent_gate_up_k[arm])")
        .unwrap();
    let silu = body.find("ops::moe_silu_mul(").unwrap();
    let down = body
        .find("KernelLaunch::new(gpu, st.persistent_down_k[arm])")
        .unwrap();
    assert!(build < gate_up && gate_up < silu && silu < down);
    assert_eq!(body.matches(".launch(stream)").count(), 3);
    assert!(KERNEL.matches("exl3_gemv_mrow_body<6").count() >= 2);
    assert!(KERNEL.matches("exl3_gemv_mrow_body<16").count() >= 2);
    assert!(KERNEL.matches("__syncthreads();\n    }").count() >= 2);
}

#[test]
fn production_dispatch_is_explicit_opt_in_with_shape_guard_and_fallback() {
    for needle in [
        "ATLAS_EXL3_VERIFY_WORKLIST",
        "persistent_verify_enabled",
        "EXL3_PERSISTENT_ROWS.contains(&num_tokens)",
        "dispatch_exl3_persistent_verify",
        "st.persistent_build_k[arm]",
        "EXL3_PERSISTENT_WORK_CAPACITIES[arm]",
        "st.mrow_gate_up_k[arm]",
    ] {
        assert!(
            DISPATCH.contains(needle),
            "missing dispatch contract: {needle}"
        );
    }
    let harness = include_str!("../examples/exl3_gemv_microtest.rs");
    for needle in [
        "m == 6 || m == 16",
        "exl3_build_m16_worklist",
        "96 * 80",
        "malformed_zeroed",
        "work_status == 3",
        "out_of_range_zeroed",
        "range_status == 2",
        "work_count == 0",
    ] {
        assert!(
            harness.contains(needle),
            "missing malformed-route GPU promotion gate: {needle}"
        );
    }
    for needle in [
        "(\"exl3_gemv\", 3usize, \"K3\")",
        "(\"exl3_gemv_k2\", 2usize, \"K2\")",
        "16 * bits",
        "g.kernel(module, \"exl3_build_m6_worklist\")",
        ".arg_u32(bits as u32)",
        "MROW GATE9 {quant_label}",
        "MROW GATE9d {quant_label}",
    ] {
        assert!(
            harness.contains(needle),
            "missing generic-K3/fixed-K2 promotion matrix contract: {needle}"
        );
    }
}
