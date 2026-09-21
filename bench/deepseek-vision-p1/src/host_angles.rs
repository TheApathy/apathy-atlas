// SPDX-License-Identifier: AGPL-3.0-only
//! Historical CPU/libm control only, not the current production angle producer.
//!
//! Reimplements the FP32 expression from the retained V7 geometry helper
//! (geometry.rs SHA faa3e794ea41e82ec4a93427390c4dc68ed905f2ec3aa57c24ec157591339a69).
//! The controlled corpus fixes head_dim=64 and theta=10000; it must not be
//! substituted for the CUDA expression or labeled an official angle oracle.

pub(crate) fn legacy_angles(gh: usize, gw: usize) -> Option<Vec<f32>> {
    if !matches!((gh, gw), (3, 3) | (4, 5) | (54, 54)) {
        return None;
    }
    let head_dim = 64usize;
    let theta = 10000.0f64;
    let n = gh.checked_mul(gw)?.checked_mul(head_dim)?;
    if n > 3456 * 64 {
        return None;
    }
    let quarter = head_dim / 4;
    let mut out = Vec::with_capacity(n);
    let freq: Vec<f32> = (0..quarter)
        .map(|i| (theta as f32).powf(i as f32 / quarter as f32).recip())
        .collect();
    if !freq.iter().all(|value| value.is_finite()) {
        return None;
    }
    for h in 0..gh {
        for w in 0..gw {
            let angles: Vec<f32> = [h, w]
                .into_iter()
                .flat_map(|pos| freq.iter().map(move |inv| pos as f32 * inv))
                .collect();
            out.extend(angles.iter().map(|angle| angle.cos()));
            out.extend(angles.iter().map(|angle| angle.sin()));
        }
    }
    Some(out)
}
