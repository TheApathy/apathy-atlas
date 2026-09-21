// SPDX-License-Identifier: AGPL-3.0-only

//! Effect-free extent checks for the existing exact tiled HC kernels.

#[derive(Debug)]
pub(crate) struct Layout {
    pub row_bytes: usize,
    pub input_bytes: usize,
    pub residual_bytes: usize,
    pub tile: usize,
}

impl Layout {
    pub(crate) fn new(
        rows: usize,
        hidden: usize,
        hc: usize,
        rank: usize,
        capacity: usize,
        bytes: [usize; 5], // BA, QKV, norm, hidden, residual
    ) -> Result<Self, &'static str> {
        if rows == 0 || rows > capacity || rows > u32::MAX as usize {
            return Err("exact HC rows exceed arena or launch capacity");
        }
        if hidden == 0
            || !hidden.is_multiple_of(16)
            || hc <= 1
            || rank == 0
            || !rank.is_multiple_of(16)
        {
            return Err("invalid exact HC projection geometry");
        }
        let width = hidden.checked_mul(hc).ok_or("HC width overflow")?;
        if width > u32::MAX as usize
            || rank > u32::MAX as usize
            || width.checked_sub(hc).is_none_or(|tail| tail < hidden)
        {
            return Err("HC staging overlaps saved injection or exceeds launch width");
        }
        let row_bytes = width.checked_mul(2).ok_or("HC row overflow")?;
        let residual_bytes = rows.checked_mul(row_bytes).ok_or("HC rows overflow")?;
        let elements = rows.checked_mul(hidden).ok_or("HC input overflow")?;
        if elements > u32::MAX as usize {
            return Err("HC pack launch exceeds u32 element capacity");
        }
        let input_bytes = elements.checked_mul(2).ok_or("HC input bytes overflow")?;
        let tile = rows.min(32);
        let projection_bytes = rank
            .checked_add(hc)
            .and_then(|width| width.checked_mul(tile))
            .and_then(|count| count.checked_mul(2))
            .ok_or("HC projection scratch overflow")?;
        let singleton_bytes = if rows % 32 == 1 {
            // prepare_decode uses fixed-offset injection scratch.
            if rank.checked_mul(2).is_none_or(|down| down > 2048) {
                return Err("HC singleton rank overlaps fixed injection offset");
            }
            2048usize
                .checked_add(hc.checked_mul(2).ok_or("HC injection overflow")?)
                .ok_or("HC singleton scratch overflow")?
        } else {
            0
        };
        let required = [
            projection_bytes.max(singleton_bytes),
            tile * row_bytes,
            input_bytes,
            residual_bytes,
            residual_bytes,
        ];
        if required
            .iter()
            .zip(bytes)
            .any(|(&need, available)| need > available)
        {
            return Err("exact HC scratch exceeds arena");
        }
        Ok(Self {
            row_bytes,
            input_bytes,
            residual_bytes,
            tile,
        })
    }
}
