// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::collections::BTreeMap;

const ROW_BYTES: usize = 4096 * 2;
const VOCAB: usize = 154_880;
const PAD: u32 = 154_854;
const TABLE: u64 = 0x1_0000_0000;
const VISION: u64 = 0x3_0000_0000;
const DEST: u64 = 0x5_0000_0000;

fn table() -> InputRegion {
    InputRegion {
        address: TABLE,
        bytes: VOCAB * ROW_BYTES,
    }
}
fn prepared(rows: usize) -> PreparedRows {
    PreparedRows {
        region: InputRegion {
            address: VISION,
            bytes: rows * ROW_BYTES,
        },
        rows,
    }
}
fn destination(rows: usize) -> InputRegion {
    InputRegion {
        address: DEST,
        bytes: rows * ROW_BYTES,
    }
}
fn row(tag: u16) -> Vec<u8> {
    tag.to_le_bytes().repeat(4096)
}

#[derive(Default)]
struct ByteIo {
    memory: BTreeMap<u64, Vec<u8>>,
    calls: Vec<(u64, u64, usize, u64)>,
    completed: usize,
    fail_at: Option<usize>,
}
impl InputCopyIo for ByteIo {
    fn copy(&mut self, source: u64, dest: u64, bytes: usize, stream: u64) -> anyhow::Result<()> {
        let call = self.calls.len();
        self.calls.push((source, dest, bytes, stream));
        anyhow::ensure!(self.fail_at != Some(call), "injected copy failure");
        let (&base, data) = self.memory.range(..=source).next_back().unwrap();
        let offset = usize::try_from(source - base)?;
        let payload = data[offset..offset + bytes].to_vec();
        let (&base, data) = self.memory.range_mut(..=dest).next_back().unwrap();
        let offset = usize::try_from(dest - base)?;
        data[offset..offset + bytes].copy_from_slice(&payload);
        self.completed += 1;
        Ok(())
    }
}
impl ByteIo {
    fn fixture(vision_rows: usize, dest_rows: usize) -> Self {
        let mut io = Self::default();
        for token in 1..=3u32 {
            io.memory.insert(
                TABLE + u64::from(token) * ROW_BYTES as u64,
                row(token as u16),
            );
        }
        let vision = (0..vision_rows).flat_map(|r| row(100 + r as u16)).collect();
        io.memory.insert(VISION, vision);
        io.memory
            .insert(DEST - 32, vec![0xa5; dest_rows * ROW_BYTES + 64]);
        io
    }
    fn output(&self, rows: usize) -> &[u8] {
        &self.memory[&(DEST - 32)][32..32 + rows * ROW_BYTES]
    }
    fn assert_redzones(&self, rows: usize) {
        let bytes = &self.memory[&(DEST - 32)];
        assert_eq!(&bytes[..32], &[0xa5; 32]);
        assert_eq!(&bytes[32 + rows * ROW_BYTES..], &[0xa5; 32]);
    }
}

#[test]
fn mixed_chunks_preserve_every_text_and_image_byte_at_seven_eight_nine_boundaries() {
    // Two image runs: six rows then three rows, with text between and after.
    let tokens = [1, PAD, PAD, PAD, PAD, PAD, PAD, 2, PAD, PAD, PAD, 3];
    let plan = InputPlan::new(
        &tokens,
        Some(prepared(9)),
        table(),
        7,
        64,
        Some(DraftContext {
            position: 7,
            capacity: 2047,
        }),
    )
    .unwrap();
    let mut expected = Vec::new();
    let mut image = 0;
    for token in tokens {
        expected.extend(row(if token == PAD {
            image += 1;
            99 + image
        } else {
            token as u16
        }));
    }
    for chunk_rows in [1, 2, 4, 7, 8] {
        let mut combined = Vec::new();
        for start in (0..tokens.len()).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(tokens.len());
            let mut io = ByteIo::fixture(9, end - start);
            let before = io.memory[&VISION].clone();
            let batch = plan.chunk(start..end, destination(end - start)).unwrap();
            assert_eq!(batch.start_position(), 7 + start as u32);
            assert_eq!(batch.rows(), end - start);
            batch.enqueue(&mut io, 17).unwrap();
            assert!(io.calls.iter().all(|call| call.3 == 17));
            combined.extend_from_slice(io.output(end - start));
            io.assert_redzones(end - start);
            assert_eq!(io.memory[&VISION], before);
        }
        assert_eq!(combined, expected, "chunk_rows={chunk_rows}");
    }
    assert!(plan.chunk(0..9, destination(9)).is_err());
}

#[test]
fn contiguous_external_rows_are_one_copy_and_tokens_are_not_reembedded_over_them() {
    let tokens = [1, PAD, PAD, 2];
    let plan = InputPlan::new(&tokens, Some(prepared(2)), table(), 0, 32, None).unwrap();
    let mut io = ByteIo::fixture(2, 4);
    plan.chunk(0..4, destination(4))
        .unwrap()
        .enqueue(&mut io, 23)
        .unwrap();
    assert_eq!(io.calls.len(), 3);
    assert_eq!(
        io.calls[1],
        (VISION, DEST + ROW_BYTES as u64, 2 * ROW_BYTES, 23)
    );
    assert_eq!(io.output(4), [row(1), row(100), row(101), row(2)].concat());
}

#[test]
fn layer_major_overwrite_accepts_more_than_eight_rows_and_only_replaces_image_spans() {
    let tokens = [1, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD, 2];
    let plan = InputPlan::new(&tokens, Some(prepared(9)), table(), 0, 64, None).unwrap();
    let mut io = ByteIo::fixture(9, tokens.len());
    plan.overwrite_vision(0..tokens.len(), destination(tokens.len()), &mut io, 29)
        .unwrap();
    assert_eq!(
        io.calls,
        vec![(VISION, DEST + ROW_BYTES as u64, 9 * ROW_BYTES, 29)]
    );
    assert!(
        io.output(tokens.len())[..ROW_BYTES]
            .iter()
            .all(|byte| *byte == 0xa5)
    );
    assert_eq!(
        &io.output(tokens.len())[ROW_BYTES..10 * ROW_BYTES],
        (0..9)
            .flat_map(|row_index| row(100 + row_index))
            .collect::<Vec<_>>()
    );
    assert!(
        io.output(tokens.len())[10 * ROW_BYTES..]
            .iter()
            .all(|byte| *byte == 0xa5)
    );
    io.assert_redzones(tokens.len());
}

#[test]
fn whole_prompt_vocabulary_pad_count_and_context_are_validated_before_any_copy() {
    for (tokens, vision) in [
        (vec![], None),
        (vec![1, VOCAB as u32], None),
        (vec![1, PAD], None),
        (vec![1, PAD], Some(prepared(2))),
        (vec![1], Some(prepared(1))),
        (vec![PAD], Some(prepared(0))),
    ] {
        assert!(InputPlan::new(&tokens, vision, table(), 0, 64, None).is_err());
    }
    let tokens = [1, 2];
    assert!(InputPlan::new(&tokens, None, table(), 63, 64, None).is_err());
    assert!(InputPlan::new(&tokens, None, table(), u32::MAX, u32::MAX, None).is_err());
    assert!(
        InputPlan::new(
            &tokens,
            None,
            table(),
            10,
            64,
            Some(DraftContext {
                position: 9,
                capacity: 2047
            })
        )
        .is_err()
    );
    assert!(
        InputPlan::new(
            &tokens,
            None,
            table(),
            2046,
            2048,
            Some(DraftContext {
                position: 2046,
                capacity: 2047
            })
        )
        .is_err()
    );
    assert!(
        InputPlan::new(
            &[1],
            None,
            table(),
            2046,
            2048,
            Some(DraftContext {
                position: 2046,
                capacity: 2047
            })
        )
        .is_ok()
    );
    assert!(InputPlan::new(&tokens, None, table(), 2046, 2048, None).is_ok());
}

#[test]
fn prepared_and_destination_extents_overflow_alignment_and_alias_reject_without_io() {
    let tokens = [1, PAD, 2];
    for bad in [
        InputRegion {
            address: TABLE,
            bytes: ROW_BYTES,
        },
        InputRegion {
            address: 0,
            bytes: VOCAB * ROW_BYTES,
        },
        InputRegion {
            address: u64::MAX - 1,
            bytes: VOCAB * ROW_BYTES,
        },
    ] {
        assert!(InputPlan::new(&tokens, Some(prepared(1)), bad, 0, 64, None).is_err());
    }
    assert!(
        InputPlan::new(
            &tokens,
            Some(PreparedRows {
                region: InputRegion {
                    address: VISION,
                    bytes: usize::MAX
                },
                rows: usize::MAX,
            }),
            table(),
            0,
            64,
            None
        )
        .is_err()
    );
    for region in [
        InputRegion {
            address: 0,
            bytes: ROW_BYTES,
        },
        InputRegion {
            address: VISION,
            bytes: ROW_BYTES - 2,
        },
        InputRegion {
            address: u64::MAX - 1,
            bytes: ROW_BYTES,
        },
        InputRegion {
            address: VISION + 1,
            bytes: ROW_BYTES,
        },
    ] {
        assert!(
            InputPlan::new(
                &tokens,
                Some(PreparedRows { region, rows: 1 }),
                table(),
                0,
                64,
                None
            )
            .is_err()
        );
    }
    let plan = InputPlan::new(&tokens, Some(prepared(1)), table(), 0, 64, None).unwrap();
    for region in [
        InputRegion {
            address: 0,
            bytes: 3 * ROW_BYTES,
        },
        InputRegion {
            address: DEST,
            bytes: 3 * ROW_BYTES - 2,
        },
        InputRegion {
            address: u64::MAX - 1,
            bytes: 3 * ROW_BYTES,
        },
        InputRegion {
            address: VISION - ROW_BYTES as u64,
            bytes: 3 * ROW_BYTES,
        },
        InputRegion {
            address: TABLE + 100 * ROW_BYTES as u64,
            bytes: 3 * ROW_BYTES,
        },
        InputRegion {
            address: DEST + 1,
            bytes: 3 * ROW_BYTES,
        },
    ] {
        assert!(plan.chunk(0..3, region).is_err());
    }
    assert!(plan.chunk(0..0, destination(1)).is_err());
    assert!(plan.chunk(2..4, destination(2)).is_err());
    assert!(plan.chunk(0..usize::MAX, destination(1)).is_err());
}

#[test]
fn copy_failure_preserves_exact_completed_prefix_and_borrowed_sources_for_owner_cleanup() {
    let tokens = [1, PAD, PAD, 2];
    let plan = InputPlan::new(&tokens, Some(prepared(2)), table(), 0, 64, None).unwrap();
    let mut io = ByteIo::fixture(2, 4);
    let source = io.memory[&VISION].clone();
    io.fail_at = Some(1);
    assert!(
        plan.chunk(0..4, destination(4))
            .unwrap()
            .enqueue(&mut io, 11)
            .is_err()
    );
    assert_eq!((io.calls.len(), io.completed), (2, 1));
    assert_eq!(&io.output(4)[..ROW_BYTES], row(1));
    assert!(io.output(4)[ROW_BYTES..].iter().all(|byte| *byte == 0xa5));
    assert_eq!(io.memory[&VISION], source);
    io.assert_redzones(4);
    // The adapter has no free/commit/observe capability. Caller owns drain,
    // poison and cursor publication after a failed asynchronous prefix.
    assert_eq!(
        plan.chunk(0..1, destination(1)).unwrap().start_position(),
        0
    );
}

#[test]
fn shipping_prefill_must_route_owned_vision_through_the_validated_wide_input_seam() {
    let source: String = [
        include_str!("target_model_exl3.rs"),
        include_str!("target_prefill_exl3.rs"),
        include_str!("target_staged_exl3.rs"),
    ]
    .join("\n")
    .split_whitespace()
    .collect();
    assert!(
        source.contains("InputPlan::new("),
        "whole-prompt admission is not wired"
    );
    assert!(
        source.contains("verify_inputs_staged("),
        "shared wide input seam is missing"
    );
    assert!(
        !source.contains("letwide_prefill=ifexpected_rows==0{"),
        "vision still disables every wide text/embedding chunk"
    );
}

#[test]
fn single_source_keeps_t1_token_and_global_image_row_sources_exact() {
    let tokens = [1, PAD, PAD, 2];
    let plan = InputPlan::new(&tokens, Some(prepared(2)), table(), 5, 64, None).unwrap();
    let expected = [
        TABLE + ROW_BYTES as u64,
        VISION,
        VISION + ROW_BYTES as u64,
        TABLE + 2 * ROW_BYTES as u64,
    ];
    for (index, address) in expected.into_iter().enumerate() {
        let batch = plan.chunk(index..index + 1, destination(1)).unwrap();
        let source = batch.single_source().unwrap();
        assert_eq!((source.address, source.bytes), (address, ROW_BYTES));
        assert_eq!(batch.start_position(), 5 + index as u32);
    }
}

#[test]
fn single_source_rejects_multiple_rows_even_one_coalesced_image_copy() {
    let tokens = [1, PAD, PAD, 2];
    let plan = InputPlan::new(&tokens, Some(prepared(2)), table(), 0, 64, None).unwrap();
    assert!(
        plan.chunk(0..2, destination(2))
            .unwrap()
            .single_source()
            .is_err()
    );
    assert!(
        plan.chunk(1..3, destination(2))
            .unwrap()
            .single_source()
            .is_err()
    );
}
