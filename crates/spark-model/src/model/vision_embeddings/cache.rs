// SPDX-License-Identifier: AGPL-3.0-only

//! Hashes only reject cache misses. A hit requires exact retained input bits.

pub(super) const MAX_CACHE_INPUT_BYTES: usize = 256 * 1024 * 1024;

pub(super) struct ExactCacheKey {
    pub fingerprint: u64,
    variant: u8,
    bits: Vec<u32>,
}

impl ExactCacheKey {
    /// Oversized input or reservation failure disables reuse, not image serving.
    /// Geometry/image boundaries are compared separately in the published layout.
    pub fn capture(
        images: &[(Vec<f32>, usize, usize)],
        variant: u8,
        fingerprint: u64,
        limit_bytes: usize,
    ) -> Option<Self> {
        let values = images
            .iter()
            .try_fold(0usize, |n, (pixels, _, _)| n.checked_add(pixels.len()))?;
        if values.checked_mul(4)? > limit_bytes {
            return None;
        }
        let mut bits = Vec::new();
        bits.try_reserve_exact(values).ok()?;
        bits.extend(
            images
                .iter()
                .flat_map(|(pixels, _, _)| pixels.iter().map(|v| v.to_bits())),
        );
        Some(Self {
            fingerprint,
            variant,
            bits,
        })
    }

    pub fn matches(
        &self,
        images: &[(Vec<f32>, usize, usize)],
        variant: u8,
        fingerprint: u64,
    ) -> bool {
        self.fingerprint == fingerprint
            && self.variant == variant
            && self.bits.iter().copied().eq(images
                .iter()
                .flat_map(|(pixels, _, _)| pixels.iter().map(|v| v.to_bits())))
    }
}
