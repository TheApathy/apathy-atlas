// SPDX-License-Identifier: AGPL-3.0-only

//! Model-aware prefix-cache lookup seam.

use spark_runtime::prefix_cache::PrefixMatch;

use super::super::types::TransformerModel;

impl TransformerModel {
    /// Hybrid models require a restorable SSM checkpoint in the same atomic
    /// lookup which acquires KV refs. Pure-attention models retain KV-only
    /// prefix reuse.
    pub(in crate::model) fn lookup_prefill_prefix(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
    ) -> PrefixMatch {
        if self.config.num_ssm_layers() > 0 {
            self.prefix_cache
                .lookup_paired(tokens, block_size, session_hash)
        } else {
            self.prefix_cache.lookup(tokens, block_size, session_hash)
        }
    }
}

#[cfg(test)]
mod tests {
    use spark_runtime::prefix_cache::PrefixMatch;

    #[test]
    fn raw_kv_without_ssm_is_not_a_hybrid_hit() {
        let raw = PrefixMatch {
            matched_blocks: vec![1, 2],
            matched_disk_block_ids: Vec::new(),
            matched_tokens: 32,
            ssm_snapshot: None,
            ssm_snapshot_tokens: 0,
        };
        assert_eq!(raw.paired_ssm_tokens(), None);
    }

    #[test]
    fn direct_and_two_phase_prefill_entries_use_the_model_aware_lookup() {
        for (name, source) in [
            ("prefill_a", include_str!("prefill_a.rs")),
            ("prefill_b", include_str!("prefill_b/prefix_lookup.rs")),
            ("prefill_c", include_str!("prefill_c.rs")),
        ] {
            assert!(
                source.contains("lookup_prefill_prefix"),
                "{name} bypasses hybrid prefix pairing"
            );
        }

        let two_phase = include_str!("prefill_c.rs");
        assert!(
            two_phase.contains("should_restore_ssm_snapshot"),
            "two-phase prefill bypasses the exact-snapshot safety policy"
        );
        assert!(
            two_phase.contains("restored_prefix_skip_tokens"),
            "two-phase prefill advances beyond its restored recurrent checkpoint"
        );
        assert!(
            two_phase.contains("cached_kv_rows_in_slice"),
            "two-phase recurrent replay overwrites its deeper paired KV hit"
        );
    }
}
