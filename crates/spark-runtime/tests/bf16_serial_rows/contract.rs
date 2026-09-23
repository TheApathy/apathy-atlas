// SPDX-License-Identifier: AGPL-3.0-only
use super::recording::{WORKSPACE_BYTES, request, workspace};
use super::serial_rows_contract::{ByteSpan, MatrixLayout, Orientation, SerialRowsPlan};

#[test]
fn all_widths_and_orientations_use_identical_m1_layouts_and_exact_row_pointers() {
    for rows in 2..=8 {
        for orientation in [Orientation::Nk, Orientation::Kn] {
            let req = request(rows, orientation);
            let plan = SerialRowsPlan::new(req).unwrap();
            assert_eq!(plan.rows(), rows);
            assert_eq!(
                plan.trans_a(),
                if orientation == Orientation::Nk { 1 } else { 0 }
            );
            let weight = match orientation {
                Orientation::Nk => MatrixLayout {
                    dtype: 14,
                    rows: 128,
                    cols: 64,
                    ld: 128,
                },
                Orientation::Kn => MatrixLayout {
                    dtype: 14,
                    rows: 64,
                    cols: 128,
                    ld: 64,
                },
            };
            assert_eq!(
                plan.layouts(),
                [
                    weight,
                    MatrixLayout {
                        dtype: 14,
                        rows: 128,
                        cols: 1,
                        ld: 128
                    },
                    MatrixLayout {
                        dtype: 14,
                        rows: 64,
                        cols: 1,
                        ld: 64
                    },
                ]
            );
            let bound = plan.bind_workspace(workspace()).unwrap();
            for row in 0..rows {
                let call = bound.row(row).unwrap();
                assert_eq!(call.act, req.act.address + u64::from(row) * 256);
                assert_eq!(call.weight, req.weight.address);
                assert_eq!(call.out, req.out.address + u64::from(row) * 128);
                assert_eq!(call.workspace, workspace().address);
                assert_eq!(call.workspace_bytes, WORKSPACE_BYTES);
                assert_eq!(call.stream, 17);
                assert_eq!(call.alpha_bits, 1.0f32.to_bits());
                assert_eq!(call.beta_bits, 0.0f32.to_bits());
            }
            assert!(bound.row(rows).is_err());
            assert!(bound.row(u32::MAX).is_err());
        }
    }
}

#[test]
fn native_bf16_requires_two_byte_not_256_byte_alignment() {
    let mut req = request(8, Orientation::Nk);
    req.act.address += 2;
    req.weight.address += 2;
    req.out.address += 2;
    let bound = SerialRowsPlan::new(req)
        .unwrap()
        .bind_workspace(workspace())
        .unwrap();
    assert_eq!(bound.row(1).unwrap().out, 0x30_0082);
}

#[test]
fn zero_outside_widths_and_non_i32_dimensions_are_rejected() {
    for rows in [0, 1, 9, u32::MAX] {
        assert!(SerialRowsPlan::new(request(rows, Orientation::Nk)).is_err());
    }
    for dimension in [0, i32::MAX as u32 + 1, u32::MAX] {
        let mut req = request(2, Orientation::Nk);
        req.n = dimension;
        assert!(SerialRowsPlan::new(req).is_err());
        req = request(2, Orientation::Nk);
        req.k = dimension;
        assert!(SerialRowsPlan::new(req).is_err());
    }
}

#[test]
fn null_odd_short_and_wrapping_entire_spans_are_rejected() {
    for which in 0..3 {
        for fault in 0..4 {
            let mut req = request(8, Orientation::Kn);
            let span = match which {
                0 => &mut req.act,
                1 => &mut req.weight,
                _ => &mut req.out,
            };
            match fault {
                0 => span.address = 0,
                1 => span.address += 1,
                2 => span.bytes -= 1,
                _ => span.address = u64::MAX - 1,
            }
            assert!(
                SerialRowsPlan::new(req).is_err(),
                "which={which} fault={fault}"
            );
        }
    }
}

#[test]
fn overlapping_input_weight_output_are_rejected_but_adjacent_spans_are_valid() {
    for pair in 0..3 {
        let mut req = request(8, Orientation::Nk);
        match pair {
            0 => req.weight.address = req.act.address + 2,
            1 => req.out.address = req.act.address + 2,
            _ => req.out.address = req.weight.address + 2,
        }
        assert!(SerialRowsPlan::new(req).is_err());
    }
    let mut req = request(8, Orientation::Nk);
    req.weight.address = req.act.address + req.act.bytes as u64;
    req.out.address = req.weight.address + req.weight.bytes as u64;
    assert!(SerialRowsPlan::new(req).is_ok());
}

#[test]
fn input_admission_is_independent_of_context_workspace_initialization() {
    let req = request(8, Orientation::Nk);
    let plan = SerialRowsPlan::new(req).unwrap();
    assert_eq!(plan.rows(), 8);
    assert!(
        plan.bind_workspace(ByteSpan {
            address: 0,
            bytes: WORKSPACE_BYTES
        })
        .is_err()
    );
    assert!(plan.bind_workspace(workspace()).is_ok());
}

#[test]
fn validation_covers_declared_capacity_not_only_the_accessed_prefix() {
    let mut req = request(8, Orientation::Nk);
    req.act.address = u64::MAX - 4095;
    req.act.bytes = 8192;
    assert!(SerialRowsPlan::new(req).is_err());
    let mut req = request(8, Orientation::Nk);
    req.act.bytes = (req.weight.address - req.act.address) as usize + 2;
    assert!(SerialRowsPlan::new(req).is_err());
    let mut req = request(8, Orientation::Nk);
    req.act.bytes += 256;
    assert!(SerialRowsPlan::new(req).is_ok());
}

#[test]
fn default_stream_zero_is_valid_without_becoming_a_buffer_null_error() {
    let mut req = request(2, Orientation::Kn);
    req.stream = 0;
    let bound = SerialRowsPlan::new(req)
        .unwrap()
        .bind_workspace(workspace())
        .unwrap();
    assert_eq!(bound.row(0).unwrap().stream, 0);
}

#[test]
fn workspace_binding_rejects_wrong_extent_alignment_wrap_and_all_operand_aliases() {
    let req = request(8, Orientation::Nk);
    let plan = SerialRowsPlan::new(req).unwrap();
    for span in [
        ByteSpan {
            address: 0,
            bytes: WORKSPACE_BYTES,
        },
        ByteSpan {
            address: workspace().address + 2,
            bytes: WORKSPACE_BYTES,
        },
        ByteSpan {
            address: workspace().address,
            bytes: WORKSPACE_BYTES - 1,
        },
        ByteSpan {
            address: workspace().address,
            bytes: WORKSPACE_BYTES + 1,
        },
        ByteSpan {
            address: u64::MAX - 255,
            bytes: WORKSPACE_BYTES,
        },
        ByteSpan {
            address: req.act.address,
            bytes: WORKSPACE_BYTES,
        },
        ByteSpan {
            address: req.weight.address,
            bytes: WORKSPACE_BYTES,
        },
        ByteSpan {
            address: req.out.address,
            bytes: WORKSPACE_BYTES,
        },
    ] {
        assert!(plan.bind_workspace(span).is_err(), "{span:?}");
    }
}
