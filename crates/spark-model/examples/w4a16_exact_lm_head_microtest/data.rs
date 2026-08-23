// SPDX-License-Identifier: AGPL-3.0-only

use half::bf16;

pub struct Fixture {
    pub rows: usize,
    pub logical_n: usize,
    pub physical_n: usize,
    pub k: usize,
    pub activations: Vec<u16>,
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
}

struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn index(&mut self, len: usize) -> usize {
        (self.next() as usize) % len
    }
}

fn bf16_bits(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}

pub fn as_le_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn from_le_bytes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

pub fn random_fixture(rows: usize, logical_n: usize, k: usize, seed: u64) -> Fixture {
    assert!(rows > 0 && logical_n > 0 && k > 0 && k.is_multiple_of(16));
    let physical_n = logical_n.div_ceil(4) * 4;
    let activation_values = [
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    ];
    let fp4_nibbles = [1u8, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15];
    // E4M3 encodings for finite, nonzero positive scales around [0.5, 2.0].
    let fp8_scales = [0x30u8, 0x34, 0x38, 0x3c, 0x40];
    let mut rng = XorShift64(seed);

    let activations = (0..rows * k)
        .map(|_| bf16_bits(activation_values[rng.index(activation_values.len())]))
        .collect();
    let packed = (0..physical_n * (k / 2))
        .map(|_| {
            let lo = fp4_nibbles[rng.index(fp4_nibbles.len())];
            let hi = fp4_nibbles[rng.index(fp4_nibbles.len())];
            lo | (hi << 4)
        })
        .collect();
    let scales = (0..physical_n * (k / 16))
        .map(|_| fp8_scales[rng.index(fp8_scales.len())])
        .collect();

    Fixture {
        rows,
        logical_n,
        physical_n,
        k,
        activations,
        packed,
        scales,
    }
}

pub fn cancellation_fixture(rows: usize, logical_n: usize, k: usize) -> Fixture {
    assert!(rows > 0 && logical_n > 0 && k > 0 && k.is_multiple_of(16));
    let physical_n = logical_n.div_ceil(4) * 4;
    let cancellation = [
        64.0, -64.0, 32.0, -32.0, 8.0, -8.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 2.0, -2.0, 4.0,
        -4.0,
    ];
    let activations = (0..rows * k)
        .map(|index| {
            let row = index / k;
            let feature = index % k;
            bf16_bits(cancellation[(feature + row * 3) % cancellation.len()])
        })
        .collect();
    let packed = (0..physical_n * (k / 2))
        .map(|index| {
            let byte = index % (k / 2);
            let output = index / (k / 2);
            let lo = [7u8, 15, 6, 14, 5, 13, 4, 12][(byte + output) % 8];
            let hi = [15u8, 7, 14, 6, 13, 5, 12, 4][(byte + output * 3) % 8];
            lo | (hi << 4)
        })
        .collect();
    let scales = (0..physical_n * (k / 16))
        .map(|index| [0x30u8, 0x38, 0x40][index % 3])
        .collect();
    Fixture {
        rows,
        logical_n,
        physical_n,
        k,
        activations,
        packed,
        scales,
    }
}

/// Exact reassociation witness for the legacy K8 batch3 path.
///
/// K1 assigns features 0 and 8 to one K16 lane, and feature 512 to the other
/// warp. It computes `(2^24 + 1) + (-2^24) == 0` in FP32. Legacy batch3 puts
/// features 0 and 512 in one K8 lane and feature 8 in the next lane, computing
/// `(2^24 - 2^24) + 1 == 1`. The BF16 outputs must therefore differ.
pub fn association_negative_fixture() -> Fixture {
    let rows = 3;
    let logical_n = 4;
    let physical_n = 4;
    let k = 1_024;
    let mut activations = vec![bf16_bits(0.0); rows * k];
    for row in 0..rows {
        let base = row * k;
        activations[base] = bf16_bits(16_777_216.0);
        activations[base + 8] = bf16_bits(1.0);
        activations[base + 512] = bf16_bits(-16_777_216.0);
    }
    let mut packed = vec![0u8; physical_n * (k / 2)];
    // FP4 E2M1 nibble 2 is +1.0; all three features are even/low nibbles.
    for feature in [0usize, 8, 512] {
        packed[feature / 2] = 2;
    }
    let scales = vec![0x38u8; physical_n * (k / 16)]; // E4M3 +1.0
    Fixture {
        rows,
        logical_n,
        physical_n,
        k,
        activations,
        packed,
        scales,
    }
}

pub fn fnv1a64(values: &[u16]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        value.to_le_bytes().into_iter().fold(hash, |inner, byte| {
            (inner ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}
