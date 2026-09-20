// SPDX-License-Identifier: AGPL-3.0-only

#![cfg(target_os = "linux")]

use std::sync::Mutex;

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::session::{UploadTransaction, W3SidecarSession, upload_layer_plan_with};
use super::session_tests::{Fixture, TempArtifact, patterned_bytes, request};

const PREFIX_0: &str = "model.language_model.layers.0";
const PREFIX_1: &str = "model.language_model.layers.1";

#[cfg(target_os = "linux")]
#[test]
fn production_upload_transaction_has_exact_asymmetric_layout() {
    let artifact = Fixture::asymmetric().finish();
    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 2);
    let prefixes = vec![PREFIX_0.to_owned(), PREFIX_1.to_owned()];
    let mut session = W3SidecarSession::prepare(admitted_request, &prefixes, 16, 32).unwrap();
    let transaction = RecordingTransaction::new(None, None);
    let weights = session
        .upload_layer_with(0, |bytes, plan| {
            upload_layer_plan_with(&transaction, bytes, plan)
        })
        .unwrap()
        .unwrap();

    let pointers = (1..=12).map(test_pointer).collect::<Vec<_>>();
    assert_eq!(
        vec![
            weights.gate.weight,
            weights.gate.weight_scale,
            weights.gate_t.weight,
            weights.gate_t.weight_scale,
            weights.up.weight,
            weights.up.weight_scale,
            weights.up_t.weight,
            weights.up_t.weight_scale,
            weights.down.weight,
            weights.down.weight_scale,
            weights.down_t.weight,
            weights.down_t.weight_scale,
        ],
        pointers
    );
    assert_eq!(
        [
            weights.gate.weight_scale_2,
            weights.gate_t.weight_scale_2,
            weights.up.weight_scale_2,
            weights.up_t.weight_scale_2,
            weights.down.weight_scale_2,
            weights.down_t.weight_scale_2,
        ],
        [1.25, 1.25, 2.5, 2.5, 3.75, 3.75]
    );
    assert!(
        [
            weights.gate.input_scale,
            weights.gate_t.input_scale,
            weights.up.input_scale,
            weights.up_t.input_scale,
            weights.down.input_scale,
            weights.down_t.input_scale,
        ]
        .into_iter()
        .all(|pointer| pointer == DevicePtr::NULL)
    );

    let expected_payloads = expected_payloads();
    let expected_allocations = pointers
        .iter()
        .copied()
        .zip(expected_payloads.iter().map(Vec::len))
        .collect::<Vec<_>>();
    let expected_copies = pointers
        .iter()
        .copied()
        .zip(expected_payloads)
        .collect::<Vec<_>>();
    let trace = transaction.trace();
    assert_eq!(trace.alloc_calls, 12);
    assert_eq!(trace.copy_calls, 12);
    assert_eq!(trace.allocations, expected_allocations);
    assert_eq!(trace.copies, expected_copies);
    assert!(trace.frees.is_empty());

    session.mark_installed(0).unwrap();
    assert_eq!(session.finish().unwrap().installed_count, 1);
}

#[cfg(target_os = "linux")]
#[test]
fn every_allocation_and_copy_failure_rolls_back_and_preserves_original_poison() {
    for fail_alloc in 1..=12 {
        let transaction = RecordingTransaction::new(Some(fail_alloc), None);
        assert_upload_fails(
            &transaction,
            &format!("injected allocation failure {fail_alloc}"),
        );
        let trace = transaction.trace();
        assert_eq!(trace.alloc_calls, fail_alloc);
        assert_eq!(trace.copy_calls, fail_alloc - 1);
        assert_eq!(
            trace.frees,
            (1..fail_alloc).rev().map(test_pointer).collect::<Vec<_>>()
        );
    }
    for fail_copy in 1..=12 {
        let transaction = RecordingTransaction::new(None, Some(fail_copy));
        assert_upload_fails(&transaction, &format!("injected copy failure {fail_copy}"));
        let trace = transaction.trace();
        assert_eq!(trace.alloc_calls, fail_copy);
        assert_eq!(trace.copy_calls, fail_copy);
        assert_eq!(
            trace.frees,
            (1..=fail_copy).rev().map(test_pointer).collect::<Vec<_>>()
        );
    }
}

#[cfg(target_os = "linux")]
fn assert_upload_fails(transaction: &RecordingTransaction, injected: &str) {
    let artifact = Fixture::asymmetric().finish();
    let temp = TempArtifact::new(&artifact);
    let admitted_request = request(&temp.path, &artifact, "0", 2);
    let prefixes = vec![PREFIX_0.to_owned(), PREFIX_1.to_owned()];
    let mut session = W3SidecarSession::prepare(admitted_request, &prefixes, 16, 32).unwrap();
    let original = match session.upload_layer_with(0, |bytes, plan| {
        upload_layer_plan_with(transaction, bytes, plan)
    }) {
        Ok(_) => panic!("injected upload failure unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(
        format!("{original:#}"),
        format!("upload W3 layer 0: {injected}")
    );

    let poison = session
        .upload_layer_with::<()>(1, |_, _| {
            panic!("poison must fail before unrequested upload")
        })
        .unwrap_err();
    assert_eq!(
        format!("{poison:#}"),
        format!("W3 sidecar session is poisoned: W3 layer 0 upload failed: {injected}")
    );
}

fn expected_payloads() -> Vec<Vec<u8>> {
    let mut payloads = Vec::with_capacity(12);
    for (rows, columns, packed_seed, scale_columns, scale_seed) in [
        (32, 6, 0x10, 1, 0x40),
        (32, 6, 0x20, 1, 0x50),
        (16, 12, 0x30, 2, 0x60),
    ] {
        let packed = patterned_bytes(rows * columns, packed_seed);
        let scales = patterned_bytes(rows * scale_columns, scale_seed);
        payloads.push(packed.clone());
        payloads.push(scales.clone());
        payloads.push(reference_transpose(&packed, rows, columns));
        payloads.push(reference_transpose(&scales, rows, scale_columns));
    }
    payloads
}

fn reference_transpose(source: &[u8], rows: usize, columns: usize) -> Vec<u8> {
    assert_eq!(source.len(), rows * columns);
    let padded_rows = rows.div_ceil(64) * 64;
    let mut output = vec![0; columns * padded_rows];
    for column in 0..columns {
        for row in 0..rows {
            output[column * padded_rows + row] = source[row * columns + column];
        }
    }
    output
}

fn test_pointer(call: usize) -> DevicePtr {
    DevicePtr(0x1000 + call as u64 * 0x1000)
}

#[derive(Clone, Debug, Default)]
struct UploadTrace {
    alloc_calls: usize,
    copy_calls: usize,
    allocations: Vec<(DevicePtr, usize)>,
    copies: Vec<(DevicePtr, Vec<u8>)>,
    frees: Vec<DevicePtr>,
}

struct RecordingTransaction {
    fail_alloc: Option<usize>,
    fail_copy: Option<usize>,
    trace: Mutex<UploadTrace>,
}

impl RecordingTransaction {
    fn new(fail_alloc: Option<usize>, fail_copy: Option<usize>) -> Self {
        Self {
            fail_alloc,
            fail_copy,
            trace: Mutex::new(UploadTrace::default()),
        }
    }

    fn trace(&self) -> UploadTrace {
        self.trace.lock().unwrap().clone()
    }
}

impl UploadTransaction for RecordingTransaction {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let mut trace = self.trace.lock().unwrap();
        trace.alloc_calls += 1;
        if self.fail_alloc == Some(trace.alloc_calls) {
            bail!("injected allocation failure {}", trace.alloc_calls);
        }
        let pointer = test_pointer(trace.alloc_calls);
        trace.allocations.push((pointer, bytes));
        Ok(pointer)
    }

    fn copy_h2d(&self, bytes: &[u8], destination: DevicePtr) -> Result<()> {
        let mut trace = self.trace.lock().unwrap();
        trace.copy_calls += 1;
        if self.fail_copy == Some(trace.copy_calls) {
            bail!("injected copy failure {}", trace.copy_calls);
        }
        trace.copies.push((destination, bytes.to_vec()));
        Ok(())
    }

    fn free(&self, pointer: DevicePtr) -> Result<()> {
        self.trace.lock().unwrap().frees.push(pointer);
        Ok(())
    }
}
