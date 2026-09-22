// SPDX-License-Identifier: AGPL-3.0-only
//! Actual metadata-free projection planner and ordered submission boundary.
#[path = "../src/model/glm53/dflash2_gemv_plan.rs"]
#[allow(dead_code)]
mod gemv_plan;
#[path = "../src/model/glm53/dflash2_kv_prefix.rs"]
#[allow(dead_code)]
mod kv_prefix;
#[path = "../src/model/glm53/dflash2_projection_contract.rs"]
#[allow(dead_code)]
mod projection_contract;
use anyhow::{Result, bail};
use gemv_plan::{GemvChunk, GemvIo, GemvPlan};

fn buffers(rows: u32, source_row: u32) -> [(u64, usize); 3] {
    [
        (
            0x1000000 + u64::from(source_row) * 8192,
            rows as usize * 8192,
        ),
        (0x4000000, 1024 * 4096 * 2),
        (0x6000000, rows as usize * 2048),
    ]
}
fn plan(rows: u32, b: [(u64, usize); 3], handles: [u64; 2]) -> Result<GemvPlan> {
    GemvPlan::new(rows, 2047, 4096, 1024, b[0], b[1], b[2], handles)
}

#[derive(Default)]
struct Recorder {
    calls: Vec<GemvChunk>,
    fail_at: Option<usize>,
    panic_at: Option<usize>,
}
impl GemvIo for Recorder {
    fn launch(&mut self, chunk: GemvChunk) -> Result<()> {
        let index = self.calls.len();
        self.calls.push(chunk);
        assert_ne!(self.panic_at, Some(index), "injected submitted-call panic");
        if self.fail_at == Some(index) {
            bail!("injected submitted-call failure");
        }
        Ok(())
    }
}

#[test]
fn pairs_cover_every_row_once_for_actual_odd_even_and_tail_shapes() {
    for rows in [1, 2, 3, 7, 8, 15, 16, 17, 2047] {
        for source_row in [0, 1, 2, 15, 16, 2039, 2046] {
            if source_row + rows > 2047 {
                continue;
            }
            let b = buffers(rows, source_row);
            let p = plan(rows, b, [11, 22]).unwrap();
            assert_eq!(p.rows(), rows);
            assert_eq!(p.hidden(), 4096);
            assert_eq!(p.width(), 1024);
            assert_eq!(p.weight(), b[1].0);
            assert_eq!(p.handles(), [11, 22]);
            let chunks = p.chunks().collect::<Vec<_>>();
            assert_eq!(chunks.len(), rows.div_ceil(2) as usize);
            let mut covered = 0;
            for chunk in &chunks {
                assert_eq!(chunk.row, covered);
                assert_eq!(chunk.rows, (rows - covered).min(2));
                assert_eq!(chunk.input, b[0].0 + u64::from(covered) * 8192);
                assert_eq!(chunk.output, b[2].0 + u64::from(covered) * 2048);
                covered += chunk.rows;
            }
            assert_eq!(covered, rows);
            let mut io = Recorder::default();
            p.execute(&mut io).unwrap();
            assert_eq!(io.calls, chunks);
        }
    }
}

#[test]
fn invalid_rows_bounds_and_handles_fail_before_any_submission() {
    let mut io = Recorder::default();
    for rows in [0, 2048, u32::MAX] {
        assert!(
            plan(rows, buffers(1, 0), [11, 22])
                .and_then(|p| p.execute(&mut io))
                .is_err()
        );
    }
    for handles in [[0, 22], [11, 0], [0, 0]] {
        assert!(
            plan(2, buffers(2, 0), handles)
                .and_then(|p| p.execute(&mut io))
                .is_err()
        );
    }
    for index in 0..3 {
        let mut b = buffers(8, 0);
        b[index].1 -= 2;
        assert!(
            plan(8, b, [11, 22])
                .and_then(|p| p.execute(&mut io))
                .is_err()
        );
    }
    assert!(io.calls.is_empty());
}

#[test]
fn vector_alignment_is_gemv_only_and_outputs_remain_bf16_aligned() {
    for index in [0, 1] {
        let mut b = buffers(2, 1);
        b[index].0 += 2;
        projection_contract::checked_spans(2, 2047, 4096, 1024, b[0], b[1], b[2]).unwrap();
        assert!(plan(2, b, [11, 22]).is_err());
    }
    let mut b = buffers(2, 0);
    b[2].0 += 2;
    assert!(plan(2, b, [11, 22]).is_ok());
    b[2].0 += 1;
    assert!(plan(2, b, [11, 22]).is_err());
    let b = buffers(1, 0);
    assert!(GemvPlan::new(1, 2047, 4097, 1024, b[0], b[1], b[2], [11, 22]).is_err());
    assert!(GemvPlan::new(1, 2047, 4096, 1025, b[0], b[1], b[2], [11, 22]).is_err());
}

#[test]
fn null_pointer_overflow_and_every_operand_alias_are_rejected() {
    for index in 0..3 {
        for address in [0, u64::MAX - 15] {
            let mut b = buffers(2, 0);
            b[index].0 = address;
            assert!(plan(2, b, [11, 22]).is_err());
        }
    }
    for left in 0..3 {
        for right in left + 1..3 {
            let mut b = buffers(2, 0);
            b[right].0 = b[left].0;
            assert!(plan(2, b, [11, 22]).is_err());
        }
    }
    let b = buffers(2, 0);
    assert!(
        GemvPlan::new(
            u32::MAX,
            u32::MAX,
            u32::MAX - 7,
            1024,
            (b[0].0, usize::MAX - 1),
            b[1],
            b[2],
            [11, 22]
        )
        .is_err()
    );
}

#[test]
fn error_or_panic_after_a_pair_never_submits_later_rows() {
    let p = plan(7, buffers(7, 1), [11, 22]).unwrap();
    for fail_at in 0..4 {
        let mut io = Recorder {
            fail_at: Some(fail_at),
            ..Recorder::default()
        };
        assert!(p.execute(&mut io).is_err());
        assert_eq!(io.calls.len(), fail_at + 1);
        assert_eq!(io.calls.last().unwrap().row, (fail_at * 2) as u32);
    }
    let mut io = Recorder {
        panic_at: Some(1),
        ..Recorder::default()
    };
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.execute(&mut io))).is_err());
    assert_eq!(io.calls.len(), 2);
}
