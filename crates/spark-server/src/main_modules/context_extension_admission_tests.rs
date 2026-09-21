// SPDX-License-Identifier: AGPL-3.0-only

use super::{ContextAdmissionReceipt, ContextRuntimeMode};

fn mode() -> ContextRuntimeMode {
    ContextRuntimeMode {
        speculative: false,
        dflash: false,
        self_speculative: false,
        ngram_speculative: false,
        high_speed_swap: true,
        hss_cache_blocks_per_seq: 62_500,
        block_size: 16,
        max_batch_size: 1,
        max_seq_len: 1_000_000,
        config_capacity: 1_000_000,
    }
}

#[test]
fn consuming_receipt_supplies_every_factory_value() {
    let values = ContextAdmissionReceipt::mint(mode(), true)
        .consume(1_000_000)
        .unwrap();
    assert_eq!(values.block_size(), 16);
    assert_eq!(values.max_seq_len(), 1_000_000);
    assert_eq!(values.max_batch_size(), 1);
    assert!(!values.speculative());
    assert!(!values.self_speculative());
    assert!(!values.dflash());
    assert_eq!(values.hss_cache_blocks_per_seq(), Some(62_500));
}

#[test]
fn stale_config_capacity_rejects_before_values_are_released() {
    assert!(
        ContextAdmissionReceipt::mint(mode(), true)
            .consume(1_048_576)
            .is_err()
    );
}

#[test]
fn native_receipt_preserves_existing_factory_values() {
    let native = ContextRuntimeMode {
        speculative: true,
        dflash: true,
        self_speculative: true,
        ngram_speculative: true,
        high_speed_swap: false,
        hss_cache_blocks_per_seq: 99,
        block_size: 32,
        max_batch_size: 8,
        max_seq_len: 262_144,
        config_capacity: 262_144,
    };
    let values = ContextAdmissionReceipt::mint(native, false)
        .consume(262_144)
        .unwrap();
    assert_eq!(values.block_size(), 32);
    assert_eq!(values.max_seq_len(), 262_144);
    assert_eq!(values.max_batch_size(), 8);
    assert!(values.speculative());
    assert!(values.self_speculative());
    assert!(values.dflash());
    assert_eq!(values.hss_cache_blocks_per_seq(), None);
}
