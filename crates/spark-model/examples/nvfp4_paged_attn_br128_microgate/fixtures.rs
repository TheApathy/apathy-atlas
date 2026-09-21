// SPDX-License-Identifier: AGPL-3.0-only

use half::bf16;

use super::contract::{CACHE_BLOCK, Case, HD, NKV, NQ};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.0 = value;
        value.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

pub(super) fn as_bytes_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub(super) fn q_fixture(case: Case) -> Vec<u8> {
    const VALUES: [f32; 16] = [
        -8.0,
        -3.0,
        -1.0,
        -0.5,
        -f32::MIN_POSITIVE,
        -0.0,
        0.0,
        f32::MIN_POSITIVE,
        0.03125,
        0.5,
        1.0,
        1.5,
        3.0,
        6.0,
        8.0,
        16.0,
    ];
    let words = case.q_len as usize * NQ as usize * HD as usize;
    let mut bytes = Vec::with_capacity(words * 2);
    for index in 0..words {
        let salt = case.q_offset as usize + index.wrapping_mul(17);
        bytes.extend_from_slice(
            &bf16::from_f32(VALUES[salt % VALUES.len()])
                .to_bits()
                .to_le_bytes(),
        );
    }
    bytes
}

pub(super) fn cache_fixture(case: Case, seed: u64) -> (Vec<u8>, Vec<u32>, u64, u64) {
    let logical_blocks = case.kv_len().div_ceil(CACHE_BLOCK) as usize;
    let physical_blocks = logical_blocks + 3;
    let data_bytes = CACHE_BLOCK as usize * NKV as usize * HD as usize / 2;
    let scale_bytes = CACHE_BLOCK as usize * NKV as usize * (HD as usize / 16);
    let block_stride = data_bytes + scale_bytes;
    let mut cache = vec![0u8; physical_blocks * block_stride];
    let mut rng = Rng(seed ^ case.q_len as u64 ^ ((case.q_offset as u64) << 32));
    for physical in 0..physical_blocks {
        let base = physical * block_stride;
        for byte in &mut cache[base..base + data_bytes] {
            *byte = (rng.next() >> 40) as u8;
        }
        const SCALES: [u8; 4] = [0x30, 0x38, 0x3c, 0x40];
        for (index, byte) in cache[base + data_bytes..base + block_stride]
            .iter_mut()
            .enumerate()
        {
            *byte = SCALES[(index + physical) & 3];
        }
    }
    let table = (0..logical_blocks)
        .map(|logical| (physical_blocks - 1 - logical) as u32)
        .collect();
    (cache, table, block_stride as u64, data_bytes as u64)
}
