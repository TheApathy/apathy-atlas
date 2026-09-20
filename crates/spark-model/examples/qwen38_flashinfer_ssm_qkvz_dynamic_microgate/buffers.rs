// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gate::evidence::{K, MAX_RELATIVE_RMS, MIN_COSINE, N};
use spark_model::layers::ops::nvfp4_dynamic_scale::Nvfp4DynamicScalePlan;

pub(super) struct CaseBuffers {
    pub input: Guarded,
    pub weight: Guarded,
    pub weight_scales: Guarded,
    pub weight_t: Guarded,
    pub weight_scales_t: Guarded,
    pub parent: Guarded,
    pub candidate: Guarded,
    pub replay: Guarded,
    pub packed: Guarded,
    pub scales: Guarded,
    pub maximum: Guarded,
    pub scale2: Guarded,
    pub status: Guarded,
    pub alpha: Guarded,
    pub replay_packed: Guarded,
    pub replay_scales: Guarded,
    pub replay_maximum: Guarded,
    pub replay_scale2: Guarded,
    pub replay_status: Guarded,
    pub replay_alpha: Guarded,
    pub reference_packed: Guarded,
    pub reference_scales: Guarded,
}

impl CaseBuffers {
    pub fn new(
        gpu: &dyn GpuBackend,
        stream: u64,
        data: &ProjectionData,
        physical_scales: &[u8],
        plan: Nvfp4DynamicScalePlan,
    ) -> Result<Self> {
        let rows = plan.rows as usize;
        let output_bytes = rows
            .checked_mul(N)
            .and_then(|v| v.checked_mul(2))
            .context("output bytes overflow")?;
        let weight_t = transpose(&data.packed.bytes, N, K / 2);
        let weight_scales_t = transpose(&data.scales.bytes, N, K / 16);
        Ok(Self {
            input: Guarded::input(gpu, stream, input_bytes(rows), 0x01)?,
            weight: Guarded::input(gpu, stream, data.packed.bytes.clone(), 0x02)?,
            weight_scales: Guarded::input(gpu, stream, physical_scales.to_vec(), 0x03)?,
            weight_t: Guarded::input(gpu, stream, weight_t, 0x04)?,
            weight_scales_t: Guarded::input(gpu, stream, weight_scales_t, 0x05)?,
            parent: Guarded::output(gpu, stream, output_bytes, 0x11)?,
            candidate: Guarded::output(gpu, stream, output_bytes, 0x12)?,
            replay: Guarded::output(gpu, stream, output_bytes, 0x13)?,
            packed: Guarded::output(gpu, stream, plan.packed_bytes, 0x21)?,
            scales: Guarded::output(gpu, stream, plan.physical_scale_bytes, 0x22)?,
            maximum: Guarded::output(gpu, stream, 4, 0x23)?,
            scale2: Guarded::output(gpu, stream, 4, 0x24)?,
            status: Guarded::output(gpu, stream, 4, 0x25)?,
            alpha: Guarded::output(gpu, stream, 4, 0x26)?,
            replay_packed: Guarded::output(gpu, stream, plan.packed_bytes, 0x31)?,
            replay_scales: Guarded::output(gpu, stream, plan.physical_scale_bytes, 0x32)?,
            replay_maximum: Guarded::output(gpu, stream, 4, 0x33)?,
            replay_scale2: Guarded::output(gpu, stream, 4, 0x34)?,
            replay_status: Guarded::output(gpu, stream, 4, 0x35)?,
            replay_alpha: Guarded::output(gpu, stream, 4, 0x36)?,
            reference_packed: Guarded::output(gpu, stream, plan.packed_bytes, 0x41)?,
            reference_scales: Guarded::output(gpu, stream, plan.physical_scale_bytes, 0x42)?,
        })
    }

    pub fn check_inputs(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for (buffer, label) in [
            (&self.input, "input"),
            (&self.weight, "weight"),
            (&self.weight_scales, "weight-scales"),
            (&self.weight_t, "weight-t"),
            (&self.weight_scales_t, "weight-scales-t"),
        ] {
            buffer.check_immutable(gpu, label)?;
        }
        Ok(())
    }

    pub fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for buffer in [
            &self.input,
            &self.weight,
            &self.weight_scales,
            &self.weight_t,
            &self.weight_scales_t,
            &self.parent,
            &self.candidate,
            &self.replay,
            &self.packed,
            &self.scales,
            &self.maximum,
            &self.scale2,
            &self.status,
            &self.alpha,
            &self.replay_packed,
            &self.replay_scales,
            &self.replay_maximum,
            &self.replay_scale2,
            &self.replay_status,
            &self.replay_alpha,
            &self.reference_packed,
            &self.reference_scales,
        ] {
            buffer.free(gpu)?;
        }
        Ok(())
    }
}

pub(super) fn input_bytes(rows: usize) -> Vec<u8> {
    const VALUES: [f32; 16] = [
        -6.0, -3.0, -1.9, -1.4, -0.9, -0.4, -0.1, 0.0, 0.1, 0.4, 0.9, 1.4, 1.9, 2.9, 3.0, 6.0,
    ];
    let mut bytes = Vec::with_capacity(rows * K * 2);
    for index in 0..rows * K {
        let mixed = index.wrapping_mul(1_103_515_245).wrapping_add(index >> 7);
        bytes.extend_from_slice(
            &bf16::from_f32(VALUES[mixed % VALUES.len()])
                .to_bits()
                .to_le_bytes(),
        );
    }
    bytes
}

pub(super) fn scalar_f32(buffer: &Guarded, gpu: &dyn GpuBackend, label: &str) -> Result<f32> {
    Ok(f32::from_le_bytes(
        buffer
            .payload(gpu, label)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("{label}: not f32"))?,
    ))
}

pub(super) fn scalar_u32(buffer: &Guarded, gpu: &dyn GpuBackend, label: &str) -> Result<u32> {
    Ok(u32::from_le_bytes(
        buffer
            .payload(gpu, label)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("{label}: not u32"))?,
    ))
}

#[derive(Clone, Copy)]
pub(super) struct Metrics {
    pub cosine: f64,
    pub relative_rms: f64,
    pub differing: usize,
}

pub(super) fn metrics(parent: &[u8], candidate: &[u8]) -> Result<Metrics> {
    ensure!(
        parent.len() == candidate.len() && parent.len().is_multiple_of(2),
        "output length mismatch"
    );
    let (mut dot, mut pp, mut cc, mut error) = (0.0, 0.0, 0.0, 0.0);
    let mut differing = 0;
    for (lhs, rhs) in parent.chunks_exact(2).zip(candidate.chunks_exact(2)) {
        let p = bf16::from_bits(u16::from_le_bytes(lhs.try_into().unwrap())).to_f32() as f64;
        let c = bf16::from_bits(u16::from_le_bytes(rhs.try_into().unwrap())).to_f32() as f64;
        ensure!(
            p.is_finite() && c.is_finite(),
            "nonfinite projection output"
        );
        dot += p * c;
        pp += p * p;
        cc += c * c;
        error += (p - c) * (p - c);
        differing += usize::from(lhs != rhs);
    }
    ensure!(pp > 0.0 && cc > 0.0, "degenerate projection output");
    let result = Metrics {
        cosine: dot / (pp * cc).sqrt(),
        relative_rms: (error / pp).sqrt(),
        differing,
    };
    ensure!(
        result.cosine >= MIN_COSINE && result.relative_rms <= MAX_RELATIVE_RMS,
        "numerical contract failed cosine={} relative_rms={}",
        result.cosine,
        result.relative_rms
    );
    Ok(result)
}

pub(super) fn p90(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[(values.len() * 9).div_ceil(10).saturating_sub(1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_qkvz_extents_and_tail() {
        let short = Nvfp4DynamicScalePlan::qwen38_ssm_qkvz(2_079, K as u32).unwrap();
        assert_eq!(
            (
                short.padded_rows,
                short.packed_bytes,
                short.physical_scale_bytes
            ),
            (2_176, 5_322_240, 696_320)
        );
        assert_eq!((short.padded_rows - short.rows) as usize * K / 16, 31_040);
        let long = Nvfp4DynamicScalePlan::qwen38_ssm_qkvz(8_192, K as u32).unwrap();
        assert_eq!(
            (
                long.padded_rows,
                long.packed_bytes,
                long.physical_scale_bytes
            ),
            (8_192, 20_971_520, 2_621_440)
        );
    }
}
