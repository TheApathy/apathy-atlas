// SPDX-License-Identifier: AGPL-3.0-only
#[path = "../examples/glm53_projection_probe/timing_order.rs"]
mod timing_order;

#[test]
fn ten_orders_balance_every_arm_position_without_reusing_a_fixed_first_arm() {
    let orders = (0..10).map(timing_order::order).collect::<Vec<_>>();
    for row in &orders {
        let mut sorted = *row;
        sorted.sort();
        assert_eq!(sorted, [0, 1, 2, 3, 4]);
    }
    for arm in 0..5 {
        for position in 0..5 {
            assert_eq!(orders.iter().filter(|row| row[position] == arm).count(), 2);
        }
    }
    assert_eq!(timing_order::order(10), timing_order::order(0));
}

#[test]
fn explicit_operator_timing_does_not_change_numeric_admission_or_include_readback() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let main = std::fs::read_to_string(root.join("examples/glm53_projection_probe.rs")).unwrap();
    assert!(main.contains("--operator-timing"));
    assert!(main.contains("with_timing && family_exact"));
    let timing =
        std::fs::read_to_string(root.join("examples/glm53_projection_probe/timing.rs")).unwrap();
    let start = timing.find("let started = Instant::now();").unwrap();
    let stop = start
        + timing[start..]
            .find("let elapsed = started.elapsed();")
            .unwrap();
    let measured = &timing[start..stop];
    assert!(measured.contains("enqueue("));
    assert!(measured.contains("synchronize("));
    for bad in [
        "session.read(",
        "artifacts::",
        "gpu.kernel(",
        "copy_d2h",
        "copy_h2d",
    ] {
        assert!(!measured.contains(bad), "timed I/O/lookup {bad}");
    }
}
