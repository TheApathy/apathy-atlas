// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail, ensure};
use half::bf16;
use spark_runtime::gpu::GpuBackend;

use super::contract::Fixture;
use super::guarded::Guarded;

pub(super) fn input_bytes(rows: u32, cols: u32, fixture: Fixture) -> Vec<u8> {
    const PRODUCTION: [f32; 16] = [
        0.0, 0.1, -0.1, 0.4, -0.4, 0.9, -0.9, 1.4, -1.4, 1.9, -1.9, 2.9, -2.9, 3.0, -3.0, 6.0,
    ];
    const CANCELLATION: [f32; 16] = [
        64.0, -64.0, 32.0, -32.0, 8.0, -8.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 2.0, -2.0, 4.0,
        -4.0,
    ];
    const HALFWAY: [f32; 16] = [
        6.0, -6.0, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5.0,
        -5.0,
    ];
    const SIGNED_ZERO: [f32; 16] = [
        0.0, -0.0, 6.0, -6.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0,
    ];
    let elements = usize::try_from(rows).unwrap() * usize::try_from(cols).unwrap();
    let mut bytes = Vec::with_capacity(elements * 2);
    for index in 0..elements {
        let bits = match fixture {
            Fixture::Production => {
                let mixed = index.wrapping_mul(1_103_515_245).wrapping_add(index >> 7);
                bf16::from_f32(PRODUCTION[mixed % PRODUCTION.len()]).to_bits()
            }
            Fixture::Cancellation => bf16::from_f32(CANCELLATION[index % 16]).to_bits(),
            Fixture::Halfway => bf16::from_f32(HALFWAY[index % 16]).to_bits(),
            Fixture::SignedZero => bf16::from_f32(SIGNED_ZERO[index % 16]).to_bits(),
            Fixture::AllZero => 0,
            Fixture::Subnormal => {
                if index.is_multiple_of(2) {
                    0x0001
                } else {
                    0x8001
                }
            }
            Fixture::MaxFinite => {
                if index.is_multiple_of(2) {
                    0x7f7f
                } else {
                    0xff7f
                }
            }
        };
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

pub(super) fn invalid_input(rows: u32, cols: u32, kind: &str) -> Result<Vec<u8>> {
    let mut bytes = input_bytes(rows, cols, Fixture::AllZero);
    let bits = match kind {
        "nan" => 0x7fc1u16,
        "posinf" => 0x7f80u16,
        "neginf" => 0xff80u16,
        _ => bail!("unknown invalid child kind {kind}"),
    };
    bytes[..2].copy_from_slice(&bits.to_le_bytes());
    Ok(bytes)
}

pub(super) fn f32_payload(buffer: &Guarded, gpu: &dyn GpuBackend, label: &str) -> Result<f32> {
    let bytes = buffer.payload(gpu, label)?;
    ensure!(bytes.len() == 4, "{label}: expected scalar");
    Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
}

pub(super) fn u32_payload(buffer: &Guarded, gpu: &dyn GpuBackend, label: &str) -> Result<u32> {
    let bytes = buffer.payload(gpu, label)?;
    ensure!(bytes.len() == 4, "{label}: expected scalar");
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

pub(super) fn require_exact(label: &str, reference: &[u8], candidate: &[u8]) -> Result<()> {
    ensure!(
        reference.len() == candidate.len(),
        "{label}: length mismatch"
    );
    if let Some(index) = reference
        .iter()
        .zip(candidate)
        .position(|(lhs, rhs)| lhs != rhs)
    {
        bail!(
            "{label}: byte {index} differs reference=0x{:02x} candidate=0x{:02x}",
            reference[index],
            candidate[index]
        );
    }
    Ok(())
}

pub(super) fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
