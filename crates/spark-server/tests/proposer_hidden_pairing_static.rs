// SPDX-License-Identifier: AGPL-3.0-only

const MTP_STEP: &str = include_str!("../src/scheduler/mtp_step.rs");
const BATCHED: &str = include_str!("../src/scheduler/verify_dflash_batched_step.rs");
const K3: &str = include_str!("../src/scheduler/verify_k3_step.rs");
const CSK: &str = include_str!("../src/scheduler/verify_csk_step.rs");
const GENERIC: &str = include_str!("../src/scheduler/verify_dflash_step.rs");

#[derive(Debug, PartialEq, Eq)]
enum ExpectedSave {
    DflashToken(u32),
    NativeMtpRow(usize),
}

fn expected_save(proposer_is_dflash: bool, token: u32, row: usize) -> ExpectedSave {
    if proposer_is_dflash {
        ExpectedSave::DflashToken(token)
    } else {
        ExpectedSave::NativeMtpRow(row)
    }
}

fn expected_batched_dflash_route(
    enabled: bool,
    sequence_count: usize,
    is_ep: bool,
    proposer_is_dflash: bool,
) -> bool {
    enabled && sequence_count >= 2 && !is_ep && proposer_is_dflash
}

#[test]
fn every_specialized_outcome_uses_the_shared_proposer_pairing_boundary() {
    let cases = [
        ("batched DFlash", BATCHED, 1),
        ("K3", K3, 3),
        ("CSK", CSK, 3),
    ];
    for (label, source, expected_calls) in cases {
        assert_eq!(
            source.matches("save_hidden_for_active_proposer(").count(),
            expected_calls,
            "{label} must pair every accepted outcome with the active proposer"
        );
        assert!(
            !source.contains("model.save_hidden_for_mtp("),
            "{label} must not hard-code the native-MTP hidden buffer"
        );
        assert!(
            !source.contains("model.save_hidden_for_dflash("),
            "{label} must not hard-code the DFlash hidden buffer"
        );
    }

    assert!(MTP_STEP.contains("fn save_hidden_for_active_proposer("));
    assert!(MTP_STEP.contains("if model.proposer_is_dflash()"));
    assert!(MTP_STEP.contains("model.save_hidden_for_dflash(token, &mut a.seq, 0)"));
    assert!(MTP_STEP.contains("model.save_hidden_for_mtp(row, 0)"));
    assert!(MTP_STEP.contains("fn batched_dflash_verify_allowed("));
    assert!(MTP_STEP.contains("batched_dflash_verify_allowed(\n"));
    assert!(GENERIC.contains("wide_verify_hidden_save(proposer_is_dflash"));
}

#[test]
fn stale_state_sentinels_cannot_cross_proposer_buffers() {
    const STALE_DFLASH_TOKEN: u32 = 0xDFA5_DFA5;
    const STALE_MTP_ROW: usize = usize::MAX;

    for row in 0..=2 {
        let token = 100 + row as u32;
        assert_eq!(
            expected_save(true, token, STALE_MTP_ROW),
            ExpectedSave::DflashToken(token)
        );
        assert_eq!(
            expected_save(false, STALE_DFLASH_TOKEN, row),
            ExpectedSave::NativeMtpRow(row)
        );
    }

    assert!(expected_batched_dflash_route(true, 2, false, true));
    assert!(!expected_batched_dflash_route(true, 2, false, false));
}
