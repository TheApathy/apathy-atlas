// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/contract.rs"]
mod contract;

use contract::{Arg, Buffers, Family, Handles, Mode, Plan, Region};
use std::ffi::OsStr;

fn handles() -> Handles {
    Handles {
        scalar: 11,
        pipelined: 22,
        add_bias: 33,
        fused_bias: 44,
    }
}

fn buffers(p: &Plan) -> Buffers {
    Buffers {
        a: Region::new(0x1000, p.input_bytes()).unwrap(),
        b: Region::new(0x1000_0000, p.weight_bytes()).unwrap(),
        bias: Region::new(0x2000_0000, p.bias_bytes()).unwrap(),
        c: Region::new(0x3000_0000, p.output_bytes()).unwrap(),
    }
}

#[test]
fn seven_real_families_keep_original_weight_layout() {
    let cases = [
        (Family::Patch, 1152, 1536),
        (Family::Qkv, 3456, 1152),
        (Family::AttentionOutput, 1152, 1152),
        (Family::Fc1, 4304, 1152),
        (Family::Fc2, 1152, 4304),
        (Family::MergerFc1, 4608, 4608),
        (Family::MergerFc2, 5120, 4608),
    ];
    for (family, n, k) in cases {
        let p = Plan::new(family, 36, 5120, Mode::Scalar).unwrap();
        assert_eq!(p.shape(), (36, n, k));
        assert_eq!(p.input_bytes(), 36 * k as usize * 2);
        assert_eq!(p.weight_bytes(), n as usize * k as usize * 2);
        assert_eq!(p.output_bytes(), 36 * n as usize * 2);
        assert_eq!(p.bias_bytes(), n as usize * 2);
        assert!(p.total_bytes() < 96 * 1024 * 1024);
    }
    let p = Plan::new(Family::MergerFc2, 9, 2560, Mode::Scalar).unwrap();
    assert_eq!(p.shape(), (9, 2560, 4608));
}

#[test]
fn benchmark_requires_an_explicit_exact_mode() {
    assert_eq!(
        Mode::parse(Some(OsStr::new("scalar"))).unwrap(),
        Mode::Scalar
    );
    assert_eq!(
        Mode::parse(Some(OsStr::new("upstream-separate"))).unwrap(),
        Mode::UpstreamSeparate
    );
    assert_eq!(
        Mode::parse(Some(OsStr::new("fused-bias"))).unwrap(),
        Mode::FusedBias
    );
    assert!(Mode::parse(None).is_err());
    for bad in [
        "", "1", "0", "auto", "SCALAR", "scalar ", "fused", "upstream",
    ] {
        assert!(Mode::parse(Some(OsStr::new(bad))).is_err(), "{bad:?}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(Mode::parse(Some(OsStr::from_bytes(&[0xff]))).is_err());
    }
}

#[test]
fn row_caps_are_projection_caps_not_a_duplicate_image_validator() {
    for family in [
        Family::Patch,
        Family::Qkv,
        Family::AttentionOutput,
        Family::Fc1,
        Family::Fc2,
    ] {
        assert!(Plan::new(family, 6400, 5120, Mode::Scalar).is_ok());
        assert!(Plan::new(family, 6401, 5120, Mode::Scalar).is_err());
    }
    for family in [Family::MergerFc1, Family::MergerFc2] {
        assert!(Plan::new(family, 1600, 2560, Mode::Scalar).is_ok());
        assert!(Plan::new(family, 1601, 2560, Mode::Scalar).is_err());
    }
    for rows in [0, usize::MAX] {
        assert!(Plan::new(Family::Fc2, rows, 5120, Mode::Scalar).is_err());
    }
    for width in [0, 2048, 5119, usize::MAX] {
        assert!(Plan::new(Family::Patch, 1, width, Mode::Scalar).is_err());
    }
    // A microkernel row can be odd. Pixel/grid admission stays in ImageLayout.
    assert!(Plan::new(Family::Qkv, 127, 5120, Mode::Scalar).is_ok());
}

#[test]
fn original_scalar_and_upstream_separate_launch_abis_are_exact() {
    let p = Plan::new(Family::Fc2, 129, 5120, Mode::Scalar).unwrap();
    let scalar = p.bind(buffers(&p), handles()).unwrap();
    let launch = &scalar.launches()[0];
    assert_eq!(scalar.launches().len(), 1);
    assert_eq!(launch.kernel(), 11);
    assert_eq!(launch.grid(), [36, 5, 1]);
    assert_eq!(launch.block(), [32, 32, 1]);
    assert_eq!(
        launch.args(),
        &[
            Arg::Ptr(0x1000),
            Arg::Ptr(0x1000_0000),
            Arg::Ptr(0x2000_0000),
            Arg::Ptr(0x3000_0000),
            Arg::U32(129),
            Arg::U32(1152),
            Arg::U32(4304),
        ]
    );
    let p = Plan::new(Family::Fc1, 129, 2560, Mode::UpstreamSeparate).unwrap();
    let bound = p.bind(buffers(&p), handles()).unwrap();
    assert_eq!(bound.launches().len(), 2);
    assert_eq!(bound.launches()[0].kernel(), 22);
    assert_eq!(bound.launches()[0].grid(), [34, 2, 1]);
    assert_eq!(bound.launches()[0].block(), [256, 1, 1]);
    assert_eq!(
        bound.launches()[0].args(),
        &[
            Arg::Ptr(0x1000),
            Arg::Ptr(0x1000_0000),
            Arg::Ptr(0x3000_0000),
            Arg::U32(129),
            Arg::U32(4304),
            Arg::U32(1152),
        ]
    );
    assert_eq!(bound.launches()[1].kernel(), 33);
    assert_eq!(bound.launches()[1].grid(), [2169, 1, 1]);
    assert_eq!(
        bound.launches()[1].args(),
        &[
            Arg::Ptr(0x3000_0000),
            Arg::Ptr(0x2000_0000),
            Arg::U32(129),
            Arg::U32(4304),
        ]
    );
}

#[test]
fn selected_handles_are_required_no_silent_fallback() {
    for mode in [Mode::Scalar, Mode::UpstreamSeparate, Mode::FusedBias] {
        let p = Plan::new(Family::Patch, 4, 5120, mode).unwrap();
        assert!(p.bind(buffers(&p), Handles::default()).is_err());
        let mut h = handles();
        match mode {
            Mode::Scalar => h.scalar = 0,
            Mode::UpstreamSeparate => h.add_bias = 0,
            Mode::FusedBias => h.fused_bias = 0,
        }
        assert!(p.bind(buffers(&p), h).is_err());
    }
    let p = Plan::new(Family::Patch, 4, 5120, Mode::UpstreamSeparate).unwrap();
    let mut h = handles();
    h.pipelined = 0;
    assert!(p.bind(buffers(&p), h).is_err());
    let p = Plan::new(Family::Patch, 4, 5120, Mode::Scalar).unwrap();
    assert!(
        p.bind(
            buffers(&p),
            Handles {
                scalar: 11,
                ..Handles::default()
            }
        )
        .is_ok()
    );
}

#[test]
fn capacities_alignment_overflow_and_output_alias_fail_closed() {
    let p = Plan::new(Family::Fc2, 4, 5120, Mode::UpstreamSeparate).unwrap();
    let mut b = buffers(&p);
    b.a = Region::new(0x1002, p.input_bytes()).unwrap();
    assert!(p.bind(b, handles()).is_err());
    let mut b = buffers(&p);
    b.b = Region::new(0x1000_0002, p.weight_bytes()).unwrap();
    assert!(p.bind(b, handles()).is_err());
    let mut b = buffers(&p);
    b.c = Region::new(0x3000_0000, p.output_bytes() - 2).unwrap();
    assert!(p.bind(b, handles()).is_err());
    let mut b = buffers(&p);
    b.c = Region::new(0x1000, p.output_bytes()).unwrap();
    assert!(p.bind(b, handles()).is_err());
    let mut b = buffers(&p);
    b.bias = Region::new(0x2000_0001, p.bias_bytes()).unwrap();
    assert!(p.bind(b, handles()).is_err());
    assert!(Region::new(0, 2).is_err());
    assert!(Region::new(0x1000, 0).is_err());
    assert!(Region::new(u64::MAX - 1, 4).is_err());
    assert!(Region::new(0x1000, usize::MAX).is_err());
}

#[test]
fn tc_tails_and_fused_bias_abi_do_not_round_down_launches() {
    for m in [1, 4, 36, 127, 128, 129, 1024] {
        let p = Plan::new(Family::Fc2, m, 2560, Mode::FusedBias).unwrap();
        let b = p.bind(buffers(&p), handles()).unwrap();
        assert_eq!(p.shape().2 % 32, 16); // actual K4304 tail
        assert_eq!(b.launches().len(), 1);
        assert_eq!(b.launches()[0].kernel(), 44);
        assert_eq!(b.launches()[0].grid(), [9, (m as u32).div_ceil(128), 1]);
        assert_eq!(b.launches()[0].block(), [256, 1, 1]);
        assert_eq!(b.launches()[0].args().len(), 7);
        assert_eq!(b.launches()[0].args()[2], Arg::Ptr(0x2000_0000));
        assert_eq!(b.launches()[0].args()[3], Arg::Ptr(0x3000_0000));
    }
}
