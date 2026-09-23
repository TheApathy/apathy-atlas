// SPDX-License-Identifier: AGPL-3.0-only

mod layers {
    pub use spark_model::layers::glm53_dsa_t1_transaction;
    pub use spark_model::layers::ops;
}
#[allow(dead_code)] // This target exercises planning; the other target exercises binding.
#[path = "../src/model/glm53/dsa_verify_plan.rs"]
mod dsa_verify_plan;

use dsa_verify_plan::{DsaVerifyPlan, MAX_BACKUP_BYTES, validate_capture_advance};
use spark_model::layers::ops::Glm53DsaPoolPlan;

#[test]
fn all_widths_and_start_residues_use_the_production_pool_contract() {
    for position in 0..16 {
        for rows in 1..=8 {
            let plan = DsaVerifyPlan::new(position, rows, 64).unwrap();
            let pool = Glm53DsaPoolPlan::new(1, rows, position, 128, 4, 1_048_576).unwrap();
            let spans = plan.source_regions();
            assert_eq!(
                (spans[0].offset, spans[0].bytes),
                (position as usize * 1024, rows as usize * 1024)
            );
            assert_eq!(
                (spans[1].offset, spans[1].bytes),
                (position as usize / 4 * 256, pool.pool_vector_bytes)
            );
            assert_eq!(
                (spans[2].offset, spans[2].bytes),
                (position as usize / 4, pool.pool_validity_bytes)
            );
            assert_eq!((spans[3].offset, spans[3].bytes), (0, 768));
            assert_eq!((spans[4].offset, spans[4].bytes), (0, 768));
            assert_eq!((spans[5].offset, spans[5].bytes), (0, 3));
            assert!(plan.backup_bytes() <= MAX_BACKUP_BYTES);
            assert_eq!(
                plan.payload_bytes(),
                spans.iter().map(|s| s.bytes).sum::<usize>() * 11
            );
        }
    }
}

#[test]
fn maximum_backup_is_small_and_not_the_recurrent_arena() {
    let plan = DsaVerifyPlan::new(3, 8, 64).unwrap();
    assert_eq!(plan.payload_bytes(), 112_695);
    assert_eq!(plan.backup_bytes(), 118_272);
    assert_eq!(MAX_BACKUP_BYTES, 118_272);
}

#[test]
fn empty_pool_spans_and_absolute_context_boundary_are_checked() {
    let plan = DsaVerifyPlan::new(0, 2, 2).unwrap();
    assert_eq!(plan.source_regions()[1].bytes, 0);
    assert_eq!(plan.source_regions()[2].bytes, 0);
    assert!(DsaVerifyPlan::new(1_048_568, 8, 1_048_576).is_ok());
    for (position, rows, capacity) in [
        (0, 0, 64),
        (0, 9, 64),
        (0, 1, 0),
        (63, 2, 64),
        (u32::MAX, 2, u32::MAX),
        (0, 2, 1_048_577),
    ] {
        assert!(DsaVerifyPlan::new(position, rows, capacity).is_err());
    }
}

#[test]
fn capture_advance_rejects_order_and_capacity_before_target_effects() {
    assert!(validate_capture_advance(2039, 2039, 8, 2047).is_ok());
    assert!(validate_capture_advance(0, 0, 1, 2047).is_ok());
    for (position, context, rows) in [
        (1, 0, 2),
        (0, 1, 2),
        (2040, 2040, 8),
        (0, 0, 0),
        (0, 0, 9),
        (u32::MAX, u32::MAX, 2),
    ] {
        assert!(validate_capture_advance(position, context, rows, 2047).is_err());
    }
}
