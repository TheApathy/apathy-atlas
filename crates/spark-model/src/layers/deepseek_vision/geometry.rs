// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};

pub(super) struct Geometry {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub patch_dim: usize,
    pub ratio: usize,
    pub max_rows: usize,
    pub max_patches: usize,
    pub text_hidden: usize,
}

impl Geometry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        intermediate: usize,
        heads: usize,
        patch: usize,
        ratio: usize,
        max_rows: usize,
        text_hidden: usize,
    ) -> Result<Self> {
        // The tower is identical in V4-Flash-Vision (text 4096, <=384 image
        // tokens) and V4.1 (text 5120, <=1024 image tokens incl. newlines).
        ensure!(
            (hidden, intermediate, heads, patch, ratio) == (1024, 2816, 16, 14, 3)
                && matches!(text_hidden, 4096 | 5120),
            "unsupported DeepSeek vision encoder geometry"
        );
        let cap = if text_hidden == 5120 { 1024 } else { 384 };
        ensure!(
            (1..=cap).contains(&max_rows),
            "DeepSeek vision image token capacity must be 1..{cap}"
        );
        Ok(Self {
            hidden,
            intermediate,
            heads,
            head_dim: hidden / heads,
            patch_dim: 3 * patch * patch,
            ratio,
            max_rows,
            max_patches: max_rows * ratio * ratio,
            text_hidden,
        })
    }

    pub fn output_rows(&self, gh: usize, gw: usize) -> Result<usize> {
        ensure!(
            gh > 0 && gw > 0,
            "DeepSeek vision patch grid cannot be empty"
        );
        let patches = gh
            .checked_mul(gw)
            .ok_or_else(|| anyhow::anyhow!("vision grid overflow"))?;
        ensure!(
            patches <= self.max_patches,
            "DeepSeek vision patch grid exceeds scratch capacity"
        );
        let rows = gh
            .div_ceil(self.ratio)
            .checked_mul(gw.div_ceil(self.ratio))
            .ok_or_else(|| anyhow::anyhow!("vision aligner grid overflow"))?;
        ensure!(
            rows <= self.max_rows,
            "DeepSeek vision aligned rows exceed image token capacity"
        );
        Ok(rows)
    }

    pub fn grid(&self, gh: usize, gw: usize, input_len: usize) -> Result<(usize, usize)> {
        let rows = self.output_rows(gh, gw)?;
        let patches = gh * gw;
        ensure!(
            input_len == patches * self.patch_dim,
            "DeepSeek vision pixels: expected {} values, got {input_len}",
            patches * self.patch_dim
        );
        Ok((patches, rows))
    }

    pub fn scratch_layout(&self) -> Result<(Vec<usize>, usize)> {
        let p = self.max_patches;
        let ph = p * self.hidden;
        let sizes = [
            p * self.patch_dim * 2,
            ph * 2,
            (ph.max(self.max_rows * self.text_hidden)) * 2,
            ph * 6,
            ph * 2,
            ph * 2,
            ph * 2,
            ph * 2,
            (p * 2 * self.intermediate).max(self.max_rows * self.hidden * self.ratio * self.ratio)
                * 2,
            p * p * 4,
            p * p * 4,
            p * self.head_dim * 4,
            self.max_rows * self.text_hidden * 2,
        ];
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut total = 0usize;
        for bytes in sizes {
            // Every scratch suballocation begins at a 256-byte aligned address.
            total = total
                .checked_add(255)
                .ok_or_else(|| anyhow::anyhow!("vision arena overflow"))?
                & !255;
            offsets.push(total);
            total = total
                .checked_add(bytes)
                .ok_or_else(|| anyhow::anyhow!("vision arena overflow"))?;
        }
        Ok((offsets, total))
    }

    pub fn scratch_bytes(&self) -> Result<usize> {
        Ok(self.scratch_layout()?.1)
    }
}

/// Original split-half 2D rotation: height frequencies, then width frequencies.
/// Per patch: cos[head_dim/2], sin[head_dim/2], both FP32.
/// CPU geometry fixtures only; production angles use the CUDA expression.
#[cfg(test)]
pub(super) fn rope_angles(gh: usize, gw: usize, head_dim: usize, theta: f64) -> Result<Vec<f32>> {
    ensure!(
        gh > 0 && gw > 0 && head_dim > 0 && head_dim.is_multiple_of(4),
        "invalid vision rotary grid"
    );
    ensure!(
        theta.is_finite() && (theta as f32).is_finite() && (theta as f32) > 0.0,
        "invalid vision rotary theta"
    );
    let n = gh
        .checked_mul(gw)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| anyhow::anyhow!("vision rotary table overflow"))?;
    ensure!(
        n <= 3456 * 64,
        "vision rotary table exceeds supported capacity"
    );
    let quarter = head_dim / 4;
    let mut out = Vec::with_capacity(n);
    let freq: Vec<f32> = (0..quarter)
        .map(|i| (theta as f32).powf(i as f32 / quarter as f32).recip())
        .collect();
    ensure!(
        freq.iter().all(|v| v.is_finite()),
        "nonfinite vision rotary frequency"
    );
    for h in 0..gh {
        for w in 0..gw {
            let angles: Vec<f32> = [h, w]
                .into_iter()
                .flat_map(|pos| freq.iter().map(move |inv| pos as f32 * inv))
                .collect();
            out.extend(angles.iter().map(|a| a.cos()));
            out.extend(angles.iter().map(|a| a.sin()));
        }
    }
    Ok(out)
}

pub(super) fn bf16_bytes(values: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &value in values {
        ensure!(value.is_finite(), "DeepSeek vision pixels must be finite");
        let bits = value.to_bits();
        let bf = (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
        ensure!(
            bf & 0x7f80 != 0x7f80,
            "DeepSeek vision pixel exceeds BF16 range"
        );
        out.extend_from_slice(&bf.to_le_bytes());
    }
    Ok(out)
}
