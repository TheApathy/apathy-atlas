// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn lm_head_proof_requires_bf16_full_vocab_and_exact_rows() {
    assert!(DflashK1LmHeadProof::begin(0, 10, 10, false).is_err());
    assert!(DflashK1LmHeadProof::begin(5, 9, 10, false).is_err());
    assert!(DflashK1LmHeadProof::begin(5, 10, 10, true).is_err());

    let mut proof = DflashK1LmHeadProof::begin(5, 10, 10, false).unwrap();
    for _ in 0..5 {
        proof.engage().unwrap();
    }
    assert!(proof.engage().is_err());
    proof.finish().unwrap();

    let mut incomplete = DflashK1LmHeadProof::begin(5, 10, 10, false).unwrap();
    incomplete.engage().unwrap();
    assert!(incomplete.finish().is_err());
}

#[test]
fn lm_head_proof_line_binds_family_frame_and_contract() {
    let mut proof = DflashK1LmHeadProof::begin(2, 10, 10, false).unwrap();
    proof.engage().unwrap();
    assert!(proof.clone().finish().is_err());
    proof.engage().unwrap();
    let proof = proof.finish().unwrap();
    assert!(proof.proof_line("family", 7, &[3]).is_err());
    assert_eq!(
        proof.proof_line("family", 7, &[3, 4]).unwrap(),
        "DFLASH_K1_LM_HEAD_PATH_PROOF family=\"family\" pre_verify_len=7 \
         tokens=[3, 4] requested=true engaged=true requested_rows=2 \
         engaged_rows=2 full_vocab=true vocab=10 dtype=\"bf16\""
    );
}
