// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for direct EXL3 global-to-shared staging.

const KERNEL: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill.cu");
const K16_K2: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2.cu");
const K64_K2: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_k2.cu");
const K64_N128_K2: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_n128_k2_gu.cu");

fn owners(items: usize, threads: usize) -> Vec<usize> {
    let mut hits = vec![0; items];
    for thread in 0..threads {
        for item in (thread..items).step_by(threads) {
            hits[item] += 1;
        }
    }
    hits
}

#[test]
fn activation_vectors_have_one_copy_owner_for_every_stage_shape() {
    for (m_tile, k_step, threads) in [(64, 16, 128), (64, 64, 128), (64, 64, 256), (128, 16, 256)] {
        let vectors = m_tile * (k_step / 8);
        assert!(owners(vectors, threads).into_iter().all(|count| count == 1));
    }
}

#[test]
fn trellis_vectors_have_one_copy_owner_for_k2_and_k3() {
    for bits in [2, 3] {
        for (k_step, n_warps, threads) in [(16, 4, 128), (64, 4, 128), (64, 8, 256)] {
            let k_tiles = k_step / 16;
            let vectors_per_strip = n_warps * 2 * bits;
            for _ in 0..k_tiles {
                assert!(
                    owners(vectors_per_strip, threads)
                        .into_iter()
                        .all(|count| count == 1)
                );
            }
        }
    }
}

#[test]
fn source_uses_cache_appropriate_async_copies_and_zero_fills_m_tails() {
    // The +2 BF16 row pad is load-bearing for bank dispersion, so row starts
    // are only four-byte aligned. Activation copies must therefore stay 4 B;
    // the naturally aligned trellis vectors can use 16 B.
    assert!(KERNEL.contains("#define EXL3_PF_PAD 2"));
    assert!(KERNEL.contains("exl3_pf_cp_async_ca_4"));
    assert!(KERNEL.contains("cp.async.ca.shared.global"));
    assert!(KERNEL.contains("exl3_pf_cp_async_cg_16"));
    assert!(KERNEL.contains("cp.async.cg.shared.global"));
    assert!(KERNEL.contains("const uint4 zero = {0, 0, 0, 0}"));
    assert!(KERNEL.contains("dst[0] = zero.x"));
    assert!(KERNEL.contains("dst[3] = zero.w"));
}

#[test]
fn async_staging_is_confined_to_k64_where_registers_do_not_regress() {
    assert!(KERNEL.contains("#ifndef EXL3_PF_ASYNC_STAGE"));
    assert!(!K16_K2.contains("#define EXL3_PF_ASYNC_STAGE 1"));
    assert!(K64_K2.contains("#define EXL3_PF_ASYNC_STAGE 1"));
    assert!(K64_N128_K2.contains("#define EXL3_PF_ASYNC_STAGE 1"));
}

#[test]
fn every_async_group_is_committed_and_waited_before_cta_consumption() {
    let activation = KERNEL
        .find("exl3_pf_cp_async_ca_4(dst + word, src + word)")
        .unwrap();
    let trellis = KERNEL
        .find("exl3_pf_cp_async_cg_16(&smem_T[k_tile][threadIdx.x], src)")
        .unwrap();
    let commit = KERNEL[trellis..]
        .find("cp.async.commit_group")
        .map(|offset| trellis + offset)
        .unwrap();
    let wait = KERNEL[commit..]
        .find("cp.async.wait_group 0")
        .map(|offset| commit + offset)
        .unwrap();
    let barrier = KERNEL[wait..]
        .find("__syncthreads()")
        .map(|offset| wait + offset)
        .unwrap();
    assert!(activation < trellis && trellis < commit && commit < wait && wait < barrier);
}
