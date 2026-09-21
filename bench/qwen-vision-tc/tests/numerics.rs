// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/contract.rs"]
mod contract;
#[path = "../src/numerics.rs"]
mod numerics;

use contract::Mode;
use numerics::{bias_epilogue, narrow_bf16, scalar_dot, validate_output};

#[test]
fn exact_bf16_rounding_witness_separates_the_two_operators() {
    // All input values are exactly representable BF16. The F32 dot is
    // exactly 1 + 2^-8: halfway between BF16 1 and 1+2^-7.
    let a = [0x3f80, 0x3b80]; // 1, 2^-8
    let b = [0x3f80, 0x3f80];
    let bias = 0x3b00; // 2^-9
    assert_eq!(scalar_dot(&a, &b, bias).unwrap(), 0x3f81);
    let sum = 1.0 + 1.0 / 256.0;
    assert_eq!(bias_epilogue(sum, bias, Mode::Scalar).unwrap(), 0x3f81);
    assert_eq!(bias_epilogue(sum, bias, Mode::FusedBias).unwrap(), 0x3f81);
    assert_eq!(
        bias_epilogue(sum, bias, Mode::UpstreamSeparate).unwrap(),
        0x3f80
    );
    // These are epilogue facts, NOT a CPU emulation of tensor-core reduction.
}

#[test]
fn rne_ties_signs_and_bias_are_explicit() {
    assert_eq!(narrow_bf16(1.0 + 1.0 / 256.0).unwrap(), 0x3f80);
    assert_eq!(narrow_bf16(1.0 + 3.0 / 256.0).unwrap(), 0x3f82);
    assert_eq!(narrow_bf16(-1.0 - 1.0 / 256.0).unwrap(), 0xbf80);
    assert_eq!(narrow_bf16(-0.0).unwrap(), 0x8000);
    assert_eq!(scalar_dot(&[0x3f80], &[0x3f80], 0xbf80).unwrap(), 0);
    assert_eq!(
        bias_epilogue(1.0, 0, Mode::UpstreamSeparate).unwrap(),
        0x3f80
    );
}

#[test]
fn hostile_or_nonfinite_numerics_are_never_a_pass() {
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        assert!(narrow_bf16(x).is_err());
    }
    assert!(scalar_dot(&[], &[], 0).is_err());
    assert!(scalar_dot(&[0x3f80], &[], 0).is_err());
    assert!(scalar_dot(&[0x7f80], &[0], 0).is_err());
    assert!(bias_epilogue(0.0, 0xffff, Mode::Scalar).is_err());
    assert!(validate_output(&[], 0).is_err());
    assert!(validate_output(&[0x3f80], 2).is_err());
    for bad in [0x7f80, 0xff80, 0x7fc0, 0xffff] {
        assert!(validate_output(&[0x3f80, bad], 2).is_err());
    }
    assert!(validate_output(&[0, 0x8000, 0x3f80, 0xbf80], 4).is_ok());
}
