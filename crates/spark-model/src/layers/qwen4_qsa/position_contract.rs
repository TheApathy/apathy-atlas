// SPDX-License-Identifier: AGPL-3.0-only

//! Physical metadata admission, independent of rotary coordinates.

pub(super) fn valid_physical_query(position: usize, sequence_length: usize) -> bool {
    sequence_length <= u32::MAX as usize && position.checked_add(1) == Some(sequence_length)
}

/// The current batched staging kernel has only four raw rows per page.
/// Admitting one compression group prevents both inter-group overwrites and
/// overwriting the old tail required by a group that crosses a chunk boundary.
/// Wider batches need a different pooling/staging transaction, not a fallback.
pub(super) fn prefill_index_batch_is_safe(num_tokens: usize, seq_len_start: usize) -> bool {
    num_tokens > 0
        && num_tokens <= 4
        && seq_len_start
            .checked_add(num_tokens)
            .is_some_and(|end| end <= u32::MAX as usize)
        && seq_len_start % 4 + num_tokens <= 4
}
