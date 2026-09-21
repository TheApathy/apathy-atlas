// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only tests of the exact committed/noise token plan used by draft MoEs.

#[path = "../src/layers/dspark_head/input_plan.rs"]
mod input_plan;

use input_plan::DraftInputPlan;

#[test]
fn actual_vision_block_contains_only_committed_and_noise_text_ids() {
    let plan = DraftInputPlan::new(5, 42, 128799, 129280).unwrap();
    assert_eq!(plan.token_ids(), &[42, 128799, 128799, 128799, 128799]);
    assert_eq!(plan.bytes().len(), 20);
    assert_eq!(&plan.bytes()[..8], &[42, 0, 0, 0, 31, 247, 1, 0]);
    let decoded: Vec<_> = plan
        .bytes()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(decoded, plan.token_ids());
}

#[test]
fn every_supported_width_fits_the_existing_markov_token_allocation() {
    for block in 1..=8 {
        let plan = DraftInputPlan::new(block, 2, 3, 4).unwrap();
        assert_eq!(plan.token_ids().len(), block);
        assert_eq!(plan.token_ids()[0], 2);
        assert!(plan.token_ids()[1..].iter().all(|&id| id == 3));
        assert_eq!(plan.bytes().len(), block * 4);
        assert!(plan.bytes().len() < (block + 1) * 4);
    }
}

#[test]
fn sentinel_and_out_of_vocabulary_ids_are_rejected() {
    for bad in [129280, 129284, u32::MAX] {
        assert!(DraftInputPlan::new(5, bad, 128799, 129280).is_err());
        assert!(DraftInputPlan::new(5, 42, bad, 129280).is_err());
    }
    assert!(DraftInputPlan::new(5, 0, 0, 0).is_err());
}

#[test]
fn invalid_block_widths_cannot_reach_embedding_or_ring_updates() {
    for block in [0, 9, usize::MAX] {
        assert!(DraftInputPlan::new(block, 1, 2, 3).is_err());
    }
}

#[test]
fn production_stages_same_ids_before_ring_updates_and_uses_them_for_moe() {
    let source = include_str!("../src/layers/dspark_head.rs");
    let propose = source.split("pub fn propose_block(").nth(1).unwrap();
    let validation = propose.find("let input = DraftInputPlan::new(").unwrap();
    let upload = propose
        .find("gpu.copy_h2d(input.bytes(), self.tok_dev)?")
        .unwrap();
    let seed = propose.find("self.seed_position(").unwrap();
    assert!(validation < upload && upload < seed);
    assert!(propose.contains("for (r, &token) in input.token_ids().iter().enumerate()"));
    let context = propose
        .split("let moe_ctx = ForwardContext {")
        .nth(1)
        .unwrap()
        .split("};")
        .next()
        .unwrap();
    assert!(context.contains("token_ids: Some(self.tok_dev)"));
    assert!(!context.contains("token_ids: None"));
    assert!(
        propose.find(".forward_kn(").unwrap()
            < propose
                .find("gpu.memset_u32_async(self.tok_dev, committed")
                .unwrap()
    );
}
