// SPDX-License-Identifier: AGPL-3.0-only

//! Wire contract for the DeepSeek six-row persistent expert-major worklist.
//!
//! The GPU kernels consume this 48-byte record directly. Keeping construction,
//! bucketing, and byte serialization here prevents the microbenchmark and the
//! forthcoming device builder from silently adopting different layouts.

pub const PERSISTENT_MAX_ROWS: usize = 6;
pub const PERSISTENT_RECORD_BYTES: usize = 48;
pub const WORK_COUNT_MASK: u32 = 0x7;
pub const WORK_SHARED: u32 = 1 << 8;
pub const WORK_UP: u32 = 1 << 9;
const WORK_VALID_META: u32 = WORK_COUNT_MASK | WORK_SHARED | WORK_UP;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PersistentWorkError {
    InvalidCount(usize),
    SlotCount {
        count: usize,
        slots: usize,
    },
    InvalidMeta(u32),
    InvalidRowCount(usize),
    InvalidTopK {
        row: usize,
        expected: usize,
        actual: usize,
    },
    DuplicateExpert {
        row: usize,
        expert: u32,
    },
    ExpertOutOfRange {
        row: usize,
        expert: u32,
        num_experts: u32,
    },
}

impl std::fmt::Display for PersistentWorkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid persistent work contract: {self:?}")
    }
}

impl std::error::Error for PersistentWorkError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C, align(16))]
pub struct PersistentWork {
    pub packed: u64,
    pub scale: u64,
    pub scale2_bits: u32,
    pub meta: u32,
    pub slots: [u32; PERSISTENT_MAX_ROWS],
}

impl PersistentWork {
    pub fn new(
        packed: u64,
        scale: u64,
        scale2: f32,
        count: usize,
        shared: bool,
        up: bool,
        gathered: &[u32],
    ) -> Self {
        Self::try_new(packed, scale, scale2, count, shared, up, gathered)
            .expect("valid persistent work record")
    }

    pub fn try_new(
        packed: u64,
        scale: u64,
        scale2: f32,
        count: usize,
        shared: bool,
        up: bool,
        gathered: &[u32],
    ) -> Result<Self, PersistentWorkError> {
        if !(1..=PERSISTENT_MAX_ROWS).contains(&count) {
            return Err(PersistentWorkError::InvalidCount(count));
        }
        if count != gathered.len() {
            return Err(PersistentWorkError::SlotCount {
                count,
                slots: gathered.len(),
            });
        }
        let mut slots = [gathered[0]; PERSISTENT_MAX_ROWS];
        slots[..count].copy_from_slice(gathered);
        Ok(Self {
            packed,
            scale,
            scale2_bits: scale2.to_bits(),
            meta: count as u32
                | if shared { WORK_SHARED } else { 0 }
                | if up { WORK_UP } else { 0 },
            slots,
        })
    }

    pub fn count(self) -> usize {
        (self.meta & WORK_COUNT_MASK) as usize
    }

    pub fn to_bytes(self) -> [u8; PERSISTENT_RECORD_BYTES] {
        let mut out = [0u8; PERSISTENT_RECORD_BYTES];
        out[0..8].copy_from_slice(&self.packed.to_le_bytes());
        out[8..16].copy_from_slice(&self.scale.to_le_bytes());
        out[16..20].copy_from_slice(&self.scale2_bits.to_le_bytes());
        out[20..24].copy_from_slice(&self.meta.to_le_bytes());
        for (index, slot) in self.slots.iter().enumerate() {
            out[24 + index * 4..28 + index * 4].copy_from_slice(&slot.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: [u8; PERSISTENT_RECORD_BYTES]) -> Result<Self, PersistentWorkError> {
        let meta = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        if meta & !WORK_VALID_META != 0 {
            return Err(PersistentWorkError::InvalidMeta(meta));
        }
        let count = (meta & WORK_COUNT_MASK) as usize;
        if !(1..=PERSISTENT_MAX_ROWS).contains(&count) {
            return Err(PersistentWorkError::InvalidCount(count));
        }
        let mut slots = [0; PERSISTENT_MAX_ROWS];
        for (index, slot) in slots.iter_mut().enumerate() {
            *slot = u32::from_le_bytes(bytes[24 + index * 4..28 + index * 4].try_into().unwrap());
        }
        Ok(Self {
            packed: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            scale: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            scale2_bits: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            meta,
            slots,
        })
    }
}

pub fn work_bucket(count: usize) -> usize {
    match count {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        5 | 6 => 3,
        _ => panic!("invalid persistent row count {count}"),
    }
}

pub fn gathered_experts(routing: &[Vec<u32>]) -> Vec<(u32, Vec<u32>)> {
    let top_k = routing.first().map_or(0, Vec::len);
    let num_experts = routing.iter().flatten().copied().max().map_or(0, |v| v + 1);
    try_gathered_experts(routing, top_k, num_experts).expect("valid persistent expert routing")
}

pub fn try_gathered_experts(
    routing: &[Vec<u32>],
    top_k: usize,
    num_experts: u32,
) -> Result<Vec<(u32, Vec<u32>)>, PersistentWorkError> {
    if !(1..=PERSISTENT_MAX_ROWS).contains(&routing.len()) {
        return Err(PersistentWorkError::InvalidRowCount(routing.len()));
    }
    if top_k == 0 {
        return Err(PersistentWorkError::InvalidTopK {
            row: 0,
            expected: 1,
            actual: 0,
        });
    }
    let mut groups: Vec<(u32, Vec<u32>)> = Vec::new();
    for (row_index, row) in routing.iter().enumerate() {
        if row.len() != top_k {
            return Err(PersistentWorkError::InvalidTopK {
                row: row_index,
                expected: top_k,
                actual: row.len(),
            });
        }
        for (column, expert) in row.iter().copied().enumerate() {
            if expert >= num_experts {
                return Err(PersistentWorkError::ExpertOutOfRange {
                    row: row_index,
                    expert,
                    num_experts,
                });
            }
            if row[..column].contains(&expert) {
                return Err(PersistentWorkError::DuplicateExpert {
                    row: row_index,
                    expert,
                });
            }
            let slot = (row_index * top_k + column) as u32;
            if let Some((_, slots)) = groups.iter_mut().find(|(value, _)| *value == expert) {
                slots.push(slot);
            } else {
                groups.push((expert, vec![slot]));
            }
        }
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_layout_is_exactly_48_bytes_and_16_aligned() {
        assert_eq!(std::mem::size_of::<PersistentWork>(), 48);
        assert_eq!(std::mem::align_of::<PersistentWork>(), 16);
        let record = PersistentWork::new(1, 2, 0.5, 2, true, true, &[9, 11]);
        let bytes = record.to_bytes();
        assert_eq!(PersistentWork::from_bytes(bytes).unwrap(), record);
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            2 | WORK_SHARED | WORK_UP
        );
        assert_eq!(record.slots, [9, 11, 9, 9, 9, 9]);
    }

    #[test]
    fn routing_is_grouped_in_first_seen_order() {
        let grouped = gathered_experts(&[vec![7, 2, 9], vec![2, 9, 7]]);
        assert_eq!(
            grouped,
            vec![(7, vec![0, 5]), (2, vec![1, 3]), (9, vec![2, 4])]
        );
        assert_eq!(
            grouped
                .iter()
                .map(|(_, rows)| work_bucket(rows.len()))
                .collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
    }

    #[test]
    fn production_routing_validation_rejects_unsafe_shapes() {
        assert_eq!(
            try_gathered_experts(&[], 6, 256),
            Err(PersistentWorkError::InvalidRowCount(0))
        );
        assert_eq!(
            try_gathered_experts(&[vec![1, 1]], 2, 256),
            Err(PersistentWorkError::DuplicateExpert { row: 0, expert: 1 })
        );
        assert_eq!(
            try_gathered_experts(&[vec![256]], 1, 256),
            Err(PersistentWorkError::ExpertOutOfRange {
                row: 0,
                expert: 256,
                num_experts: 256,
            })
        );
    }

    #[test]
    fn wire_decoder_rejects_reserved_meta_bits_and_zero_count() {
        let mut bytes = PersistentWork::new(1, 2, 0.5, 1, false, false, &[0]).to_bytes();
        bytes[20..24].copy_from_slice(&(1u32 << 31 | 1).to_le_bytes());
        assert!(matches!(
            PersistentWork::from_bytes(bytes),
            Err(PersistentWorkError::InvalidMeta(_))
        ));
        bytes[20..24].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            PersistentWork::from_bytes(bytes),
            Err(PersistentWorkError::InvalidCount(0))
        );
    }
}
