// SPDX-License-Identifier: AGPL-3.0-only
use std::{fs, path::Path};
fn source(file: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join(file),
    )
    .unwrap()
}
#[test]
fn poison_is_owned_nonfinite_and_retained_on_failed_drain() {
    let text = source("glm53_projection_probe/session.rs")
        .split_whitespace()
        .collect::<String>();
    assert!(text.contains("poison:Vec<u8>"));
    assert!(text.contains("poison:[0xc0,0x7f].repeat(2048)"));
    assert!(text.contains("copy_h2d(&self.poison,ptr)?;"));
    assert!(text.contains("forget(std::mem::take(&mutself.poison))"));
    let method = text
        .split("pubfnpoison_output(")
        .nth(1)
        .expect("missing poison method");
    assert!(
        method
            .split("pubfnread(")
            .next()
            .unwrap()
            .contains("synchronize(self.stream)")
    );
}
#[test]
fn each_independent_schedule_poisoned_once_and_raw_saved_before_validation() {
    let text = source("glm53_projection_probe.rs");
    let case = text
        .split("fn run_case(")
        .nth(1)
        .unwrap()
        .split("fn main()")
        .next()
        .unwrap();
    assert_eq!(case.matches("poison_output(").count(), 1);
    assert!(case.find("poison_output(").unwrap() < case.find("for &(m, row)").unwrap());
    assert!(
        case.find("artifacts::write(").unwrap()
            < case.find("contract::compare(&full, &split)").unwrap()
    );
}
#[test]
fn warmup_and_every_timing_batch_are_poisoned_outside_the_clock() {
    let text = source("glm53_projection_probe/timing.rs");
    assert_eq!(text.matches("poison_output(").count(), 2);
    let warmup = text
        .split("for policy in ARMS")
        .nth(1)
        .unwrap()
        .split("let mut samples")
        .next()
        .unwrap();
    assert!(warmup.find("poison_output(").unwrap() < warmup.find("enqueue(").unwrap());
    let batch = text.split("for (position, arm)").nth(1).unwrap();
    assert!(
        batch.find("poison_output(").unwrap()
            < batch.find("let started = Instant::now();").unwrap()
    );
    let measured = batch
        .split("let started = Instant::now();")
        .nth(1)
        .unwrap()
        .split("let elapsed = started.elapsed();")
        .next()
        .unwrap();
    assert!(!measured.contains("poison_output("));
}
