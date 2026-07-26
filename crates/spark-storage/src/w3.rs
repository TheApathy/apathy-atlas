// SPDX-License-Identifier: AGPL-3.0-only

//! W3 Lloyd-Max expert-weight format (3-bit codebook indices, Turbo3 packing).
//!
//! Offline requant of Laguna's NVFP4 routed experts (E2M1 nibbles + FP8-E4M3
//! per-16 group scales + per-tensor f32 scale2) down to 3 bits/param:
//!
//!   * The FP8 group scales and the per-tensor scale2 are kept UNCHANGED —
//!     only the 16-entry E2M1 value set is collapsed onto a symmetric 8-entry
//!     Lloyd-Max codebook (4 magnitudes × sign), fitted globally on the
//!     empirical, scale²-weighted magnitude distribution of the shipped
//!     checkpoint. Kernel dequant stays `LUT[idx] * e4m3(scale) * scale2`,
//!     with `LUT` now the 8-entry codebook in E2M1 units.
//!   * Packing is the Turbo3 idiom (paged_decode_attn_turbo3.cu): 8 × 3-bit
//!     values per 3 bytes, little-endian bit stream — index j occupies bits
//!     [3j, 3j+3) of the 24-bit trio.
//!
//! One file per MoE layer (`layer_{L:03}.w3x`), fixed strides:
//!
//! ```text
//!   [ header 64 B: magic, version, layer, num_experts, hidden, inter,
//!                  group_size, flags, lut[8] f32 ]
//!   [ scale2 f32[num_experts][3] (gate, up, down) ]
//!   [ pad to 4096 ]
//!   per expert e (stride = 3*(packed3+scale) for gate/up/down):
//!     gate.packed3 [inter, hidden*3/8]  gate.scale [inter, hidden/16]
//!     up.packed3   [inter, hidden*3/8]  up.scale   [inter, hidden/16]
//!     down.packed3 [hidden, inter*3/8]  down.scale [hidden, inter/16]
//! ```
//!
//! Consumed by `spark-model`'s `weight_map::w3cache` (ATLAS_MOE_W3=1). The
//! header layout is duplicated there — bump [`W3_VERSION`] on ANY change.

/// Magic: "W3LM" little-endian.
pub const W3_MAGIC: u32 = 0x4D4C_3357;
pub const W3_VERSION: u32 = 1;
pub const W3_HEADER_BYTES: usize = 64;
/// Payload (per-expert records) starts at a 4 KiB boundary.
pub const W3_PAYLOAD_ALIGN: usize = 4096;
pub const GROUP_SIZE: usize = 16;

/// E2M1 magnitude table (low 3 bits of an NVFP4 nibble; bit 3 = sign).
pub const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decode an FP8-E4M3 byte (e4m3fn: no inf, 0x7F/0xFF = NaN).
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

/// Exact 1-D Lloyd-Max (weighted k-means, k=4) over the 8 discrete E2M1
/// magnitudes with the given non-negative masses. Optimal clusters of ordered
/// scalars are contiguous, so brute-force the C(7,3)=35 contiguous partitions
/// and return the 4 mass-weighted centroids (ascending).
pub fn fit_codebook4(mass: &[f64; 8]) -> [f32; 4] {
    let pts = E2M1_MAG.map(|v| v as f64);
    let seg_cost = |a: usize, b: usize| -> (f64, f64) {
        // inclusive range [a, b] → (centroid, weighted SSE)
        let mut m = 0.0f64;
        let mut s = 0.0f64;
        for i in a..=b {
            m += mass[i];
            s += mass[i] * pts[i];
        }
        let c = if m > 0.0 {
            s / m
        } else {
            (pts[a] + pts[b]) * 0.5
        };
        let mut e = 0.0f64;
        for i in a..=b {
            e += mass[i] * (pts[i] - c) * (pts[i] - c);
        }
        (c, e)
    };
    let mut best_cost = f64::INFINITY;
    let mut best = [0.0f32; 4];
    // Cut points: 0 <= c1 < c2 < c3 < 7 partition [0,c1][c1+1,c2][c2+1,c3][c3+1,7].
    for c1 in 0..=4usize {
        for c2 in (c1 + 1)..=5 {
            for c3 in (c2 + 1)..=6 {
                let segs = [(0, c1), (c1 + 1, c2), (c2 + 1, c3), (c3 + 1, 7)];
                let mut cost = 0.0;
                let mut cents = [0.0f32; 4];
                for (i, (a, b)) in segs.iter().enumerate() {
                    let (c, e) = seg_cost(*a, *b);
                    cost += e;
                    cents[i] = c as f32;
                }
                if cost < best_cost {
                    best_cost = cost;
                    best = cents;
                }
            }
        }
    }
    best
}

/// Symmetric 8-entry codebook + the 16→8 nibble-code remap table.
///
/// 3-bit index layout mirrors E2M1: bit 2 = sign, bits 0-1 = magnitude level.
/// `lut[idx]` is in E2M1 units, so kernel dequant is
/// `lut[idx] * e4m3(group_scale) * scale2` — identical shape to NVFP4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Codebook {
    pub lut: [f32; 8],
    pub map16: [u8; 16],
}

impl Codebook {
    pub fn from_centroids(cents: [f32; 4]) -> Self {
        let mut lut = [0.0f32; 8];
        for c in 0..4 {
            lut[c] = cents[c];
            lut[4 + c] = -cents[c];
        }
        let mut map16 = [0u8; 16];
        for code in 0..16u8 {
            let mag = E2M1_MAG[(code & 7) as usize];
            // Nearest centroid (ties → lower index).
            let mut bestc = 0usize;
            let mut bestd = f32::INFINITY;
            for (c, &cent) in cents.iter().enumerate() {
                let d = (mag - cent).abs();
                if d < bestd {
                    bestd = d;
                    bestc = c;
                }
            }
            let sign = (code >> 3) & 1;
            map16[code as usize] = (sign << 2) | bestc as u8;
        }
        Self { lut, map16 }
    }

    pub fn fit(mass: &[f64; 8]) -> Self {
        Self::from_centroids(fit_codebook4(mass))
    }
}

/// Pack 8 3-bit indices into 3 bytes (Turbo3 little-endian bit stream).
#[inline]
pub fn pack8(idx: &[u8; 8]) -> [u8; 3] {
    let mut bits: u32 = 0;
    for (j, &v) in idx.iter().enumerate() {
        debug_assert!(v < 8);
        bits |= (v as u32 & 7) << (3 * j);
    }
    [bits as u8, (bits >> 8) as u8, (bits >> 16) as u8]
}

/// Unpack 3 bytes into 8 3-bit indices.
#[inline]
pub fn unpack8(b: &[u8; 3]) -> [u8; 8] {
    let bits = b[0] as u32 | ((b[1] as u32) << 8) | ((b[2] as u32) << 16);
    let mut out = [0u8; 8];
    for (j, o) in out.iter_mut().enumerate() {
        *o = ((bits >> (3 * j)) & 7) as u8;
    }
    out
}

/// Remap + repack one NVFP4 nibble row (`[k/2]` bytes, element k's nibble is
/// high when k is odd) into a W3 row (`[k*3/8]` bytes). `k % 8 == 0` required.
pub fn repack_row(nibble_row: &[u8], map16: &[u8; 16], out: &mut [u8]) {
    let k = nibble_row.len() * 2;
    debug_assert_eq!(k % 8, 0);
    debug_assert_eq!(out.len(), k * 3 / 8);
    for (t, chunk) in nibble_row.chunks_exact(4).enumerate() {
        let mut idx = [0u8; 8];
        for (b, &byte) in chunk.iter().enumerate() {
            idx[b * 2] = map16[(byte & 0x0F) as usize];
            idx[b * 2 + 1] = map16[(byte >> 4) as usize];
        }
        out[t * 3..t * 3 + 3].copy_from_slice(&pack8(&idx));
    }
}

/// Fixed byte geometry of one layer file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct W3LayerGeom {
    pub num_experts: usize,
    pub hidden: usize,
    pub inter: usize,
}

impl W3LayerGeom {
    pub fn gate_up_packed3(&self) -> usize {
        self.inter * self.hidden * 3 / 8
    }
    pub fn gate_up_scale(&self) -> usize {
        self.inter * self.hidden / GROUP_SIZE
    }
    pub fn down_packed3(&self) -> usize {
        self.hidden * self.inter * 3 / 8
    }
    pub fn down_scale(&self) -> usize {
        self.hidden * self.inter / GROUP_SIZE
    }
    /// Per-expert record stride (gate+up+down, packed3 + scale each).
    pub fn expert_stride(&self) -> usize {
        2 * (self.gate_up_packed3() + self.gate_up_scale())
            + self.down_packed3()
            + self.down_scale()
    }
    /// Sub-buffer offsets within one expert record:
    /// (gate_p, gate_s, up_p, up_s, down_p, down_s).
    pub fn expert_offsets(&self) -> [usize; 6] {
        let gp = 0;
        let gs = gp + self.gate_up_packed3();
        let up = gs + self.gate_up_scale();
        let us = up + self.gate_up_packed3();
        let dp = us + self.gate_up_scale();
        let ds = dp + self.down_packed3();
        [gp, gs, up, us, dp, ds]
    }
    pub fn scale2_off(&self) -> usize {
        W3_HEADER_BYTES
    }
    pub fn scale2_bytes(&self) -> usize {
        self.num_experts * 3 * 4
    }
    pub fn payload_off(&self) -> usize {
        (W3_HEADER_BYTES + self.scale2_bytes()).div_ceil(W3_PAYLOAD_ALIGN) * W3_PAYLOAD_ALIGN
    }
    pub fn file_bytes(&self) -> usize {
        self.payload_off() + self.num_experts * self.expert_stride()
    }
}

/// Header of one `layer_{L:03}.w3x` file.
#[derive(Debug, Clone, PartialEq)]
pub struct W3LayerHeader {
    pub layer: u32,
    pub num_experts: u32,
    pub hidden: u32,
    pub inter: u32,
    pub lut: [f32; 8],
}

impl W3LayerHeader {
    pub fn to_bytes(&self) -> [u8; W3_HEADER_BYTES] {
        let mut out = [0u8; W3_HEADER_BYTES];
        let mut w = |off: usize, v: u32| out[off..off + 4].copy_from_slice(&v.to_le_bytes());
        w(0, W3_MAGIC);
        w(4, W3_VERSION);
        w(8, self.layer);
        w(12, self.num_experts);
        w(16, self.hidden);
        w(20, self.inter);
        w(24, GROUP_SIZE as u32);
        w(28, 0); // flags
        for (i, v) in self.lut.iter().enumerate() {
            out[32 + i * 4..36 + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < W3_HEADER_BYTES {
            return None;
        }
        let r =
            |off: usize| u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        if r(0) != W3_MAGIC || r(4) != W3_VERSION || r(24) != GROUP_SIZE as u32 {
            return None;
        }
        let mut lut = [0.0f32; 8];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = f32::from_le_bytes([
                buf[32 + i * 4],
                buf[33 + i * 4],
                buf[34 + i * 4],
                buf[35 + i * 4],
            ]);
        }
        Some(Self {
            layer: r(8),
            num_experts: r(12),
            hidden: r(16),
            inter: r(20),
            lut,
        })
    }
}

/// Per-layer quality accumulator: scale²-weighted 16-bin code histogram.
/// All quality stats derive from it exactly (the reference and the W3
/// reconstruction differ only in the code→value map; the scale chain is
/// shared), so no second dequant pass is needed:
///   x  = E2M1[code] * s,  x̂ = lut[map16[code]] * s,  M[code] = Σ s².
#[derive(Debug, Clone, Default)]
pub struct CodeHist {
    pub mass: [f64; 16],
    /// Raw (unweighted) value count, for RMSE normalization.
    pub count: u64,
}

impl CodeHist {
    pub fn add(&mut self, other: &CodeHist) {
        for i in 0..16 {
            self.mass[i] += other.mass[i];
        }
        self.count += other.count;
    }

    /// Accumulate one NVFP4 tensor: nibbles `[n, k/2]`, e4m3 scales
    /// `[n, k/16]`, per-tensor scale2.
    pub fn accum_tensor(&mut self, packed: &[u8], scales: &[u8], scale2: f32, n: usize, k: usize) {
        debug_assert_eq!(packed.len(), n * k / 2);
        debug_assert_eq!(scales.len(), n * k / GROUP_SIZE);
        let groups = k / GROUP_SIZE;
        for row in 0..n {
            let prow = &packed[row * k / 2..(row + 1) * (k / 2)];
            let srow = &scales[row * groups..(row + 1) * groups];
            for (g, &sb) in srow.iter().enumerate() {
                let s = e4m3_to_f32(sb) * scale2;
                let w = (s as f64) * (s as f64);
                // 16 elements = 8 packed bytes per group.
                for &byte in &prow[g * 8..(g + 1) * 8] {
                    self.mass[(byte & 0x0F) as usize] += w;
                    self.mass[(byte >> 4) as usize] += w;
                }
            }
        }
        self.count += (n * k) as u64;
    }

    /// Fold signed 16-bin mass into 8 magnitude masses (for codebook fit).
    pub fn magnitude_mass(&self) -> [f64; 8] {
        let mut m = [0.0f64; 8];
        for code in 0..16 {
            m[code & 7] += self.mass[code];
        }
        m
    }

    /// (rmse, cosine, ref_rms) of the W3 reconstruction vs the NVFP4-dequant
    /// reference under `cb`, in absolute weight units.
    pub fn quality(&self, cb: &Codebook) -> (f64, f64, f64) {
        let mut se = 0.0f64; // Σ m (x - x̂)²
        let mut xx = 0.0f64;
        let mut yy = 0.0f64;
        let mut xy = 0.0f64;
        for code in 0..16 {
            let m = self.mass[code];
            if m == 0.0 {
                continue;
            }
            let sign = if code & 8 != 0 { -1.0f64 } else { 1.0 };
            let x = sign * E2M1_MAG[code & 7] as f64;
            let y = cb.lut[cb.map16[code] as usize] as f64;
            se += m * (x - y) * (x - y);
            xx += m * x * x;
            yy += m * y * y;
            xy += m * x * y;
        }
        let n = self.count.max(1) as f64;
        let rmse = (se / n).sqrt();
        let cos = if xx > 0.0 && yy > 0.0 {
            xy / (xx.sqrt() * yy.sqrt())
        } else {
            1.0
        };
        (rmse, cos, (xx / n).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trips() {
        let idx = [0u8, 7, 3, 5, 1, 6, 2, 4];
        assert_eq!(unpack8(&pack8(&idx)), idx);
        // Exhaustive over byte-boundary-crossing positions.
        for v in 0..8u8 {
            for pos in 0..8usize {
                let mut idx = [0u8; 8];
                idx[pos] = v;
                assert_eq!(unpack8(&pack8(&idx)), idx, "v={v} pos={pos}");
            }
        }
    }

    #[test]
    fn pack8_matches_turbo3_kernel_layout() {
        // paged_decode_attn_turbo3.cu: b0 = v0|(v1<<3)|(v2<<6),
        // b1 = (v2>>2)|(v3<<1)|(v4<<4)|(v5<<7), b2 = (v5>>1)|(v6<<2)|(v7<<5).
        let v = [1u8, 2, 5, 3, 7, 6, 4, 2];
        let b = pack8(&v);
        assert_eq!(b[0], v[0] | (v[1] << 3) | (v[2] << 6));
        assert_eq!(b[1], (v[2] >> 2) | (v[3] << 1) | (v[4] << 4) | (v[5] << 7));
        assert_eq!(b[2], (v[5] >> 1) | (v[6] << 2) | (v[7] << 5));
    }

    #[test]
    fn repack_row_maps_nibbles() {
        // Identity-ish map: mag code m → cluster m%4, sign preserved.
        let mut map16 = [0u8; 16];
        for c in 0..16u8 {
            map16[c as usize] = ((c >> 3) << 2) | (c & 3);
        }
        // Row of 8 elements: nibbles [0x21, 0x43, 0x65, 0x87] → elems 1,2,3,4,5,6,7,8.
        let row = [0x21u8, 0x43, 0x65, 0x87];
        let mut out = [0u8; 3];
        repack_row(&row, &map16, &mut out);
        let idx = unpack8(&out);
        let expect: Vec<u8> = (1..=8u8).map(|e| map16[(e & 0x0F) as usize]).collect();
        assert_eq!(&idx[..], &expect[..]);
    }

    #[test]
    fn codebook_fit_recovers_synthetic_clusters() {
        // All mass on magnitudes {0, 1.5, 3, 6} → centroids land exactly there.
        let mut mass = [0.0f64; 8];
        mass[0] = 10.0; // 0.0
        mass[3] = 5.0; // 1.5
        mass[5] = 2.0; // 3.0
        mass[7] = 1.0; // 6.0
        let cents = fit_codebook4(&mass);
        assert_eq!(cents, [0.0, 1.5, 3.0, 6.0]);
        let cb = Codebook::from_centroids(cents);
        // Those four codes must round-trip exactly, both signs.
        for (code, cent) in [(0usize, 0.0f32), (3, 1.5), (5, 3.0), (7, 6.0)] {
            assert_eq!(cb.lut[cb.map16[code] as usize], cent);
            assert_eq!(cb.lut[cb.map16[code | 8] as usize], -cent);
        }
    }

    #[test]
    fn codebook_fit_merges_neighbors_by_mass() {
        // Heavy mass at low codes: expect fine resolution near 0 and one
        // coarse top cluster. With uniform mass the 4 clusters must cover
        // all 8 magnitudes contiguously.
        let mass = [1.0f64; 8];
        let cents = fit_codebook4(&mass);
        assert!(
            cents.windows(2).all(|w| w[0] < w[1]),
            "ascending: {cents:?}"
        );
        let cb = Codebook::from_centroids(cents);
        // Every magnitude maps to the centroid nearest to it.
        for code in 0..8usize {
            let mag = E2M1_MAG[code];
            let assigned = cb.lut[cb.map16[code] as usize];
            for &c in &cents {
                assert!((mag - assigned).abs() <= (mag - c).abs() + 1e-6);
            }
        }
    }

    /// Per-layer codebooks (w3-requant `--codebook per-layer`, the default):
    /// fitting each histogram separately can never lose to one pooled fit on
    /// the fit metric (weighted SSE == RMSE² here), and strictly wins when
    /// the two distributions diverge. This is the exact quantity `quality()`
    /// reports per layer, so per-layer RMSE ≤ global-codebook RMSE, layer by
    /// layer.
    #[test]
    fn per_hist_fit_never_loses_to_pooled_fit() {
        // Layer A: mass concentrated on small magnitudes; layer B: on large.
        let hist = |mass8: [f64; 8]| {
            let mut h = CodeHist::default();
            for (i, m) in mass8.iter().enumerate() {
                h.mass[i] = *m; // positive codes only — magnitudes suffice
            }
            h.count = 16;
            h
        };
        let a = hist([4.0, 10.0, 6.0, 2.0, 0.5, 0.1, 0.0, 0.0]);
        let b = hist([0.0, 0.1, 0.5, 2.0, 6.0, 10.0, 4.0, 1.0]);
        let mut pooled = CodeHist::default();
        pooled.add(&a);
        pooled.add(&b);

        let cb_a = Codebook::fit(&a.magnitude_mass());
        let cb_b = Codebook::fit(&b.magnitude_mass());
        let cb_p = Codebook::fit(&pooled.magnitude_mass());

        let rmse = |h: &CodeHist, cb: &Codebook| h.quality(cb).0;
        assert!(rmse(&a, &cb_a) <= rmse(&a, &cb_p) + 1e-15);
        assert!(rmse(&b, &cb_b) <= rmse(&b, &cb_p) + 1e-15);
        // Divergent enough that at least one side strictly improves.
        assert!(
            rmse(&a, &cb_a) < rmse(&a, &cb_p) - 1e-9 || rmse(&b, &cb_b) < rmse(&b, &cb_p) - 1e-9,
            "expected a strict win on divergent histograms: a {} vs {}, b {} vs {}",
            rmse(&a, &cb_a),
            rmse(&a, &cb_p),
            rmse(&b, &cb_b),
            rmse(&b, &cb_p),
        );
    }

    #[test]
    fn e4m3_decodes_known_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0); // exp=7 mant=0
        assert_eq!(e4m3_to_f32(0x40), 2.0); // exp=8
        assert_eq!(e4m3_to_f32(0xC0), -2.0);
        assert_eq!(e4m3_to_f32(0x08), 2f32.powi(-6)); // smallest normal
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9)); // subnormal
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // max finite e4m3fn
        assert!(e4m3_to_f32(0x7F).is_nan());
    }

    #[test]
    fn header_round_trips_and_rejects_bad_magic() {
        let h = W3LayerHeader {
            layer: 7,
            num_experts: 256,
            hidden: 3072,
            inter: 1024,
            lut: [0.0, 0.9, 2.1, 4.8, -0.0, -0.9, -2.1, -4.8],
        };
        let bytes = h.to_bytes();
        assert_eq!(W3LayerHeader::from_bytes(&bytes), Some(h.clone()));
        let mut bad = bytes;
        bad[0] ^= 0xFF;
        assert_eq!(W3LayerHeader::from_bytes(&bad), None);
    }

    #[test]
    fn geometry_is_consistent() {
        let g = W3LayerGeom {
            num_experts: 256,
            hidden: 3072,
            inter: 1024,
        };
        assert_eq!(g.gate_up_packed3(), 1024 * 3072 * 3 / 8);
        assert_eq!(g.expert_offsets()[5] + g.down_scale(), g.expert_stride());
        assert_eq!(g.payload_off() % W3_PAYLOAD_ALIGN, 0);
        assert!(g.payload_off() >= W3_HEADER_BYTES + g.scale2_bytes());
        // W3 row stride must stay 4-byte aligned for the u32-vectorized
        // kernel loads (K % 32 == 0 for both projections).
        assert_eq!((g.hidden * 3 / 8) % 4, 0);
        assert_eq!((g.inter * 3 / 8) % 4, 0);
    }

    #[test]
    fn hist_quality_exact_when_codebook_covers_used_codes() {
        // Tensor using only codes whose magnitudes are centroids → RMSE 0.
        let mut h = CodeHist::default();
        // one group of 16 elems, scale byte 0x38 (=1.0), scale2 1.0
        // codes: 0 and 3 (mag 0.0 / 1.5) alternating, packed 0x30 nibbles.
        let packed = vec![0x30u8; 8];
        let scales = vec![0x38u8; 1];
        h.accum_tensor(&packed, &scales, 1.0, 1, 16);
        let cb = Codebook::from_centroids([0.0, 1.5, 3.0, 6.0]);
        let (rmse, cos, _) = h.quality(&cb);
        assert!(rmse < 1e-12, "rmse={rmse}");
        assert!((cos - 1.0).abs() < 1e-12);
    }
}
