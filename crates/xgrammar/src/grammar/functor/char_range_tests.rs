// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn ascii_codepoint_packs_to_itself() {
    assert_eq!(codepoint_to_packed_utf8(b'a' as u32), b'a' as u32);
    assert_eq!(codepoint_to_packed_utf8(0x7F), 0x7F);
}

#[test]
fn two_byte_codepoint() {
    // U+00E9 'é' -> 0xC3 0xA9
    assert_eq!(codepoint_to_packed_utf8(0xE9), 0xC3A9);
}

#[test]
fn three_byte_codepoint() {
    // U+20AC '€' -> 0xE2 0x82 0xAC
    assert_eq!(codepoint_to_packed_utf8(0x20AC), 0xE282AC);
}

#[test]
fn ascii_range_single_edge() {
    let mut fsm = FsmWithStartEnd::default();
    let s = fsm.add_state();
    let e = fsm.add_state();
    fsm.set_start_state(s);
    fsm.add_end_state(e);
    add_character_range(&mut fsm, s, e, b'a' as u32, b'z' as u32);
    assert!(fsm.accept_string(b"m"));
    assert!(!fsm.accept_string(b"A"));
}

/// Build the FSM for codepoint range [lo, hi] and check that EVERY probed
/// codepoint is accepted iff it lies in the range. Probes cover all of the
/// BMP (minus surrogates) plus a stride through the supplementary planes and
/// every UTF-8 length boundary.
fn assert_range_exact(lo: u32, hi: u32) {
    let mut fsm = FsmWithStartEnd::default();
    let s = fsm.add_state();
    let e = fsm.add_state();
    fsm.set_start_state(s);
    fsm.add_end_state(e);
    add_character_range(
        &mut fsm,
        s,
        e,
        codepoint_to_packed_utf8(lo),
        codepoint_to_packed_utf8(hi),
    );
    let probes = (0u32..0x1_0000)
        .chain((0x1_0000..=0x10_FFFF).step_by(97))
        .chain([
            0x7F,
            0x80,
            0x7FF,
            0x800,
            0xFFFF,
            0x1_0000,
            0x10_FFFF,
            lo,
            hi,
            lo.saturating_sub(1),
            hi + 1,
        ]);
    let mut wrong = Vec::new();
    for cp in probes {
        let Some(ch) = char::from_u32(cp) else {
            continue;
        };
        let mut buf = [0u8; 4];
        let got = fsm.accept_string(ch.encode_utf8(&mut buf).as_bytes());
        if got != (lo..=hi).contains(&cp) {
            wrong.push(cp);
        }
    }
    assert!(
        wrong.is_empty(),
        "[U+{lo:04X}-U+{hi:04X}]: {} codepoints misclassified, first U+{:04X}",
        wrong.len(),
        wrong[0]
    );
}

/// Regression: the 3-byte `tmax` recursion passed 0x0080 for 0x8080 (a bug
/// inherited from upstream C++ xgrammar), so `[\u0000-｛]` lost
/// U+F000..U+FF3F (fullwidth punctuation such as "，" U+FF0C and "＂" U+FF02).
#[test]
fn codepoint_ranges_are_exact_across_utf8_lengths() {
    for (lo, hi) in [
        (0x0000, 0xFF5B),    // the DSML value class, lower half
        (0xFF5D, 0x10_FFFF), // ... and upper half
        (0x0020, 0xFF5B),
        (0x0800, 0xFFFF),      // exactly the 3-byte block
        (0x0901, 0xE0C2),      // 3-byte, mid-byte bounds on both ends
        (0x00A0, 0x07BF),      // 2-byte, mid bounds
        (0x0081, 0x0800),      // 2-byte into 3-byte
        (0x1_2345, 0x10_ABCD), // 4-byte, mid bounds
        (0x4E00, 0x9FFF),      // CJK
    ] {
        assert_range_exact(lo, hi);
    }
}
