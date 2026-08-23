// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};
use half::bf16;
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub struct WeightFixture {
    pub logical_n: usize,
    pub k: usize,
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
}

pub struct DualFixture {
    pub rows: usize,
    pub activations: Vec<u16>,
    pub gate: WeightFixture,
    pub up: WeightFixture,
}

pub struct SiluFixture {
    pub rows: usize,
    pub gate: Vec<u16>,
    pub up: Vec<u16>,
    pub down: WeightFixture,
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

fn random_weight(logical_n: usize, k: usize, rng: &mut XorShift64) -> WeightFixture {
    assert!(logical_n > 0 && k > 0 && k.is_multiple_of(16));
    let physical_n = logical_n.div_ceil(4) * 4;
    let fp4_nibbles = [1u8, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15];
    let fp8_scales = [0x30u8, 0x34, 0x38, 0x3c, 0x40];
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
    WeightFixture {
        logical_n,
        k,
        packed,
        scales,
    }
}

pub fn random_dual(rows: usize, n: usize, k: usize, seed: u64) -> DualFixture {
    assert!(rows > 0);
    let values = [
        -6.0, -4.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0,
    ];
    let mut rng = XorShift64(seed);
    let activations = (0..rows * k)
        .map(|_| bf16_bits(values[rng.index(values.len())]))
        .collect();
    let gate = random_weight(n, k, &mut rng);
    let up = random_weight(n, k, &mut rng);
    DualFixture {
        rows,
        activations,
        gate,
        up,
    }
}

pub fn random_silu(rows: usize, n: usize, k: usize, seed: u64) -> SiluFixture {
    assert!(rows > 0);
    let gate_values = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0];
    let up_values = [-6.0, -3.0, -1.5, -0.5, 0.5, 1.5, 3.0, 6.0];
    let mut rng = XorShift64(seed);
    let gate = (0..rows * k)
        .map(|_| bf16_bits(gate_values[rng.index(gate_values.len())]))
        .collect();
    let up = (0..rows * k)
        .map(|_| bf16_bits(up_values[rng.index(up_values.len())]))
        .collect();
    let down = random_weight(n, k, &mut rng);
    SiluFixture {
        rows,
        gate,
        up,
        down,
    }
}

fn association_weight(logical_n: usize, k: usize) -> WeightFixture {
    assert!(k >= 512 && k.is_multiple_of(16));
    let physical_n = logical_n.div_ceil(4) * 4;
    let mut packed = vec![0u8; physical_n * (k / 2)];
    for output in 0..physical_n {
        for feature in [0usize, 8, 256] {
            packed[output * (k / 2) + feature / 2] = 2; // NVFP4 +1.0
        }
    }
    WeightFixture {
        logical_n,
        k,
        packed,
        scales: vec![0x38u8; physical_n * (k / 16)], // FP8 E4M3 +1.0
    }
}

/// Witness for preserving the K1 K8-lane reduction association.
///
/// The ordinary kernel reduces `(2^24 + 1) + (-2^24)` to zero. Pairing the
/// two lane-0 warp partials first instead produces one.
pub fn cancellation_dual(rows: usize, n: usize) -> DualFixture {
    let k = 512;
    let mut activations = vec![bf16_bits(0.0); rows * k];
    for row in 0..rows {
        let base = row * k;
        activations[base] = bf16_bits(16_777_216.0);
        activations[base + 8] = bf16_bits(1.0);
        activations[base + 256] = bf16_bits(-16_777_216.0);
    }
    DualFixture {
        rows,
        activations,
        gate: association_weight(n, k),
        up: association_weight(n, k),
    }
}

/// The same association witness after inline `SiLU(gate) * up`.
///
/// `gate=128` makes the inline sigmoid exactly one on CUDA; the BF16 power-of-
/// two up values therefore generate `2^24`, `1`, and `-2^24` exactly.
pub fn cancellation_silu(rows: usize, n: usize) -> SiluFixture {
    let k = 512;
    let mut gate = vec![bf16_bits(0.0); rows * k];
    let mut up = vec![bf16_bits(0.0); rows * k];
    for row in 0..rows {
        let base = row * k;
        for feature in [0usize, 8, 256] {
            gate[base + feature] = bf16_bits(128.0);
        }
        up[base] = bf16_bits(131_072.0);
        up[base + 8] = bf16_bits(0.007_812_5);
        up[base + 256] = bf16_bits(-131_072.0);
    }
    SiluFixture {
        rows,
        gate,
        up,
        down: association_weight(n, k),
    }
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

pub fn fnv1a64(values: &[u16]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        value.to_le_bytes().into_iter().fold(hash, |inner, byte| {
            (inner ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub fn upload_weight(
    gpu: &dyn GpuBackend,
    fixture: &WeightFixture,
    scale2: f32,
) -> Result<QuantizedWeight> {
    Ok(QuantizedWeight {
        weight: upload(gpu, &fixture.packed)?,
        weight_scale: upload(gpu, &fixture.scales)?,
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
    })
}

pub fn read_bf16(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    elements: usize,
    stream: u64,
) -> Result<Vec<u16>> {
    let mut bytes = vec![0u8; elements * size_of::<u16>()];
    gpu.copy_d2h_on_stream(ptr, &mut bytes, stream)?;
    Ok(from_le_bytes(&bytes))
}

pub fn raw_bf16_equal(label: &str, exact: &[u16], serial: &[u16], width: usize) -> Result<()> {
    if exact != serial {
        let first = exact
            .iter()
            .zip(serial)
            .position(|(actual, oracle)| actual != oracle)
            .expect("different vectors have a differing element");
        bail!(
            "{label}: raw BF16 mismatch flat={first} row={} n={} exact=0x{:04x} K1=0x{:04x}",
            first / width,
            first % width,
            exact[first],
            serial[first]
        );
    }
    Ok(())
}
