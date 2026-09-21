// SPDX-License-Identifier: AGPL-3.0-only
//! RED against the intended pure production contract, not a CUDA emulator.
#[path = "../src/layers/deepseek_vision/linear_diagnostic_plan.rs"]
mod plan;

use plan::{ActualLinear, BackendMode, Buffers, Dimensions, EncoderPlan, Family, Span};
use std::ffi::OsStr;

fn dimensions() -> Dimensions {
    Dimensions {
        hidden: 1024,
        intermediate: 2816,
        patch_dim: 588,
        ratio: 3,
        text_hidden: 4096,
        depth: 32,
        max_patches: 3456,
        max_rows: 384,
    }
}

fn buffers(p: &EncoderPlan, slot: usize) -> Buffers {
    let s = &p.calls()[slot];
    Buffers {
        input: Span::new(0x1000_0000, s.input_bytes()).unwrap(),
        weight: Span::new(0x2000_0000, s.weight_bytes()).unwrap(),
        bias: s
            .has_bias()
            .then(|| Span::new(0x3000_0000, s.shape().1 * 2).unwrap()),
        output: Span::new(0x4000_0000, s.output_bytes()).unwrap(),
        workspace: Some(Span::new(0x5000_0000, 8_519_680).unwrap()),
    }
}

fn actual(p: &EncoderPlan, slot: usize) -> ActualLinear {
    let s = &p.calls()[slot];
    let (m, n, k) = s.shape();
    ActualLinear {
        m,
        n,
        k,
        ldc: n,
        has_bias: s.has_bias(),
    }
}

#[test]
fn seven_families_follow_the_real_32_block_131_call_order() {
    let p = EncoderPlan::new(dimensions(), 20, 4, BackendMode::Default).unwrap();
    assert_eq!(p.calls().len(), 131);
    assert_eq!(p.calls()[0].family(), Family::Patch);
    assert_eq!(p.calls()[0].shape(), (20, 1024, 588));
    assert_eq!(p.calls()[0].layer(), None);
    for layer in 0..32 {
        for (offset, family, n, k, biased) in [
            (0, Family::Qkv, 3072, 1024, true),
            (1, Family::Projection, 1024, 1024, true),
            (2, Family::Fc1, 5632, 1024, false),
            (3, Family::Fc2, 1024, 2816, false),
        ] {
            let s = &p.calls()[1 + layer * 4 + offset];
            assert_eq!(s.family(), family);
            assert_eq!(s.layer(), Some(layer));
            assert_eq!(s.shape(), (20, n, k));
            assert_eq!(s.has_bias(), biased);
        }
    }
    assert_eq!(p.calls()[129].family(), Family::Align1);
    assert_eq!(p.calls()[129].shape(), (4, 4096, 9216));
    assert_eq!(p.calls()[130].family(), Family::Align2);
    assert_eq!(p.calls()[130].shape(), (4, 4096, 4096));
    assert_eq!(p.calls().iter().filter(|s| s.has_bias()).count(), 67);
    for slot in 0..131 {
        p.bind(slot, actual(&p, slot), buffers(&p, slot)).unwrap();
    }
    assert!(p.bind(131, actual(&p, 0), buffers(&p, 0)).is_err());
}

#[test]
fn modes_are_explicit_and_native_receipt_does_not_claim_scalar_math() {
    for (text, mode, math, label) in [
        ("scalar", BackendMode::Scalar, None, "native-wmma"),
        ("default", BackendMode::Default, Some(0), "gemmex-default"),
        ("full", BackendMode::Full, Some(16), "gemmex-full"),
    ] {
        assert_eq!(BackendMode::parse(Some(OsStr::new(text))).unwrap(), mode);
        assert_eq!(mode.math_mode(), math);
        assert_eq!(mode.receipt_label(), label);
    }
    assert!(BackendMode::parse(None).is_err());
    for bad in [
        "", "0", "1", "auto", "native", "DEFAULT", "default ", "full2",
    ] {
        assert!(BackendMode::parse(Some(OsStr::new(bad))).is_err());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(BackendMode::parse(Some(OsStr::from_bytes(&[0xff]))).is_err());
    }
}

#[test]
fn gemmex_bias_is_beta_one_with_precise_prefill_not_a_second_rounding() {
    for mode in [BackendMode::Default, BackendMode::Full] {
        let p = EncoderPlan::new(dimensions(), 20, 4, mode).unwrap();
        assert_eq!(p.mode(), mode);
        for slot in [0, 1, 2, 3, 4, 129, 130] {
            let bound = p.bind(slot, actual(&p, slot), buffers(&p, slot)).unwrap();
            let c = bound.gemm_call().unwrap();
            let s = &p.calls()[slot];
            let (m, n, k) = s.shape();
            assert_eq!((c.m, c.n, c.k), (n as i32, m as i32, k as i32));
            assert_eq!((c.transa, c.transb), (1, 0));
            assert_eq!((c.lda, c.ldb, c.ldc), (k as i32, k as i32, n as i32));
            assert_eq!((c.a, c.b, c.c), (0x2000_0000, 0x1000_0000, 0x4000_0000));
            assert_eq!(
                (c.a_type, c.b_type, c.c_type, c.compute_type, c.algorithm),
                (14, 14, 14, 68, 99)
            );
            assert_eq!(c.alpha, 1.0);
            assert_eq!(c.beta, if s.has_bias() { 1.0 } else { 0.0 });
            let copies = bound.bias_copies();
            assert_eq!(copies.len(), if s.has_bias() { m } else { 0 });
            for (row, copy) in copies.iter().enumerate() {
                assert_eq!(copy.src, 0x3000_0000);
                assert_eq!(copy.dst, 0x4000_0000 + (row * n * 2) as u64);
                assert_eq!(copy.bytes, n * 2);
            }
        }
    }
    let p = EncoderPlan::new(dimensions(), 20, 4, BackendMode::Scalar).unwrap();
    let mut b = buffers(&p, 0);
    b.workspace = None;
    let native = p.bind(0, actual(&p, 0), b).unwrap();
    assert!(native.gemm_call().is_none());
    assert!(native.bias_copies().is_empty());
}

#[test]
fn k588_and_real_row_tails_are_not_rejected_as_tc_aligned_shapes() {
    for (patches, rows) in [(9, 1), (20, 4), (2916, 324), (3456, 384)] {
        let p = EncoderPlan::new(dimensions(), patches, rows, BackendMode::Default).unwrap();
        assert_eq!(p.calls()[0].shape().2 % 16, 12);
        assert_eq!(p.calls()[0].input_bytes(), patches * 588 * 2);
        assert_eq!(p.calls()[129].weight_bytes(), 4096 * 9216 * 2);
        for slot in [0, 1, 3, 4, 129, 130] {
            p.bind(slot, actual(&p, slot), buffers(&p, slot)).unwrap();
        }
    }
    for (p, r) in [(0, 1), (1, 0), (3457, 384), (3456, 385), (usize::MAX, 1)] {
        assert!(EncoderPlan::new(dimensions(), p, r, BackendMode::Full).is_err());
    }
    // The loader owns the checkpoint's depth32 restriction. This pure planner
    // derives call count from admitted depth rather than duplicating that guard.
    let mut d = dimensions();
    d.depth = 31;
    assert_eq!(
        EncoderPlan::new(d, 20, 4, BackendMode::Default)
            .unwrap()
            .calls()
            .len(),
        127
    );
    for depth in [0, usize::MAX] {
        let mut d = dimensions();
        d.depth = depth;
        assert!(EncoderPlan::new(d, 20, 4, BackendMode::Default).is_err());
    }
}

#[test]
fn actual_call_mismatch_capacity_alias_alignment_and_overflow_fail_before_io() {
    let p = EncoderPlan::new(dimensions(), 20, 4, BackendMode::Default).unwrap();
    let mut a = actual(&p, 0);
    a.k = 592;
    assert!(p.bind(0, a, buffers(&p, 0)).is_err());
    let mut a = actual(&p, 0);
    a.ldc += 1;
    assert!(p.bind(0, a, buffers(&p, 0)).is_err());
    assert!(p.bind(0, actual(&p, 1), buffers(&p, 0)).is_err());
    for fault in 0..7 {
        let mut b = buffers(&p, 0);
        match fault {
            0 => b.input = Span::new(0x1000_0000, 20 * 588 * 2 - 2).unwrap(),
            1 => b.weight = Span::new(0x2000_0001, 1024 * 588 * 2).unwrap(),
            2 => b.output = Span::new(0x1000_0000, 20 * 1024 * 2).unwrap(),
            3 => b.bias = None,
            4 => b.workspace = None,
            5 => b.workspace = Some(Span::new(0x5000_0002, 8_519_680).unwrap()),
            _ => b.output = Span::new(0x4000_0000, 20 * 1024 * 2 - 2).unwrap(),
        }
        assert!(p.bind(0, actual(&p, 0), b).is_err(), "fault {fault}");
    }
    let mut b = buffers(&p, 3);
    b.bias = Some(Span::new(0x3000_0000, 5632 * 2).unwrap());
    assert!(p.bind(3, actual(&p, 3), b).is_err());
    assert!(Span::new(0, 4).is_err());
    assert!(Span::new(256, 0).is_err());
    assert!(Span::new(u64::MAX - 1, 4).is_err());
}
