// SPDX-License-Identifier: AGPL-3.0-only

//! Hybrid prefix-cache invariant: KV without restorable SSM is a miss.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

#[test]
fn raw_kv_without_restorable_ssm_is_a_paired_miss() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0);

    let raw = tree.lookup(&tokens, 16, 7);
    assert_eq!(raw.matched_tokens, 64);
    assert!(raw.ssm_snapshot.is_none());
    tree.release(&tokens, 16);

    let paired = tree.lookup_paired(&tokens, 16, 7);
    assert!(paired.is_empty(), "KV-only hybrid lookup lied: {paired:?}");
}

#[test]
fn paired_lookup_keeps_deeper_kv_for_recurrent_replay() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0);
    let checkpoint: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&checkpoint, &[10, 20], &[], 16, 42, 7, 0);

    let paired = tree.lookup_paired(&tokens, 16, 7);
    assert_eq!(paired.matched_tokens, 64);
    assert_eq!(paired.ssm_snapshot, Some(42));
    assert_eq!(paired.ssm_snapshot_tokens, 32);
    assert_eq!(paired.paired_ssm_tokens(), Some(32));
    tree.release(&tokens, 16);
}

#[test]
fn evicted_ssm_snapshot_turns_a_raw_kv_hit_into_a_paired_miss() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0);
    tree.insert_intermediate_snapshot(&tokens, &[10, 20], &[], 16, 42, 7, 0);
    tree.release(&tokens, 16);

    assert_eq!(tree.evict_snapshot_lru(), Some(42));
    assert_eq!(tree.lookup(&tokens, 16, 7).matched_tokens, 32);
    tree.release(&tokens, 16);
    assert!(tree.lookup_paired(&tokens, 16, 7).is_empty());
}

#[test]
fn paired_miss_releases_the_kv_walk_refs() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    tree.insert(&tokens, &[10], &[], 16, 0);
    tree.release(&tokens, 16);

    assert!(tree.lookup_paired(&tokens, 16, 7).is_empty());
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);
}
