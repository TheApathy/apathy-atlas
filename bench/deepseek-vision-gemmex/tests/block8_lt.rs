// SPDX-License-Identifier: AGPL-3.0-only
//! Real Lt boundary plan/protocol, exercised without loading CUDA libraries.
#[path = "../src/block8/contract.rs"]
mod contract;
#[path = "../src/contract.rs"]
mod gemm_geometry;
#[path = "../src/block8/lt_plan.rs"]
mod lt_plan;

use anyhow::{Result, bail};
use contract::DeviceSpan;
use lt_plan::{BoundLt, LtIo, LtMode, LtPlan, SelectedAlgorithm, execute};

fn spans() -> [DeviceSpan; 4] {
    [
        DeviceSpan {
            ptr: 0x1000,
            bytes: 40960,
        },
        DeviceSpan {
            ptr: 0x100000,
            bytes: 11_534_336,
        },
        DeviceSpan {
            ptr: 0x2000000,
            bytes: 225280,
        },
        DeviceSpan {
            ptr: 0x3000000,
            bytes: 64 * 1024 * 1024,
        },
    ]
}
fn bound(mode: LtMode) -> BoundLt {
    let [x, w, y, workspace] = spans();
    LtPlan::new(mode).bind(x, w, y, workspace, 0x9876).unwrap()
}
fn algorithm() -> SelectedAlgorithm {
    SelectedAlgorithm {
        returned: 1,
        state: 0,
        algorithm_id: 21,
        tile_id: 5,
        split_k: 9,
        reduction_scheme: 2,
        workspace_bytes: 4096,
        waves: 1.0,
    }
}

#[test]
fn lt_fc1_descriptors_keep_bf16_operands_fp32_compute_and_exact_strides() {
    for mode in [LtMode::Baseline, LtMode::ComputeTypeOnly] {
        let bound = bound(mode);
        let c = bound.call();
        let [x, w, y, workspace] = spans();
        assert_eq!(c.transpose, [1, 0]);
        assert_eq!(c.m_n_k, [5632, 20, 1024]);
        assert_eq!(c.weight_layout, [1024, 5632, 1024]);
        assert_eq!(c.input_layout, [1024, 20, 1024]);
        assert_eq!(c.output_layout, [5632, 20, 5632]);
        assert_eq!((c.a, c.b, c.c, c.d), (w.ptr, x.ptr, y.ptr, y.ptr));
        assert_eq!((c.a_type, c.b_type, c.c_type, c.d_type), (14, 14, 14, 14));
        assert_eq!((c.compute_type, c.scale_type), (68, 0));
        assert_eq!((c.alpha.to_bits(), c.beta.to_bits()), (1f32.to_bits(), 0));
        assert_eq!(c.workspace, workspace);
        assert_eq!(c.stream, 0x9876);
        assert_eq!(c.heuristic_count, 1);
    }
    assert_eq!(LtMode::Baseline.preference_mask(), None);
    assert_eq!(LtMode::ComputeTypeOnly.preference_mask(), Some(2));
}

#[test]
fn lt_preflight_rejects_each_extent_alignment_alias_overflow_and_null_stream() {
    for mode in [LtMode::Baseline, LtMode::ComputeTypeOnly] {
        let [x, w, y, workspace] = spans();
        assert!(LtPlan::new(mode).bind(x, w, y, workspace, 0).is_err());
        for index in 0..4 {
            for replacement in [
                DeviceSpan {
                    ptr: 0,
                    bytes: spans()[index].bytes,
                },
                DeviceSpan {
                    ptr: spans()[index].ptr + 1,
                    bytes: spans()[index].bytes,
                },
                DeviceSpan {
                    ptr: u64::MAX - 255,
                    bytes: spans()[index].bytes,
                },
                DeviceSpan {
                    ptr: spans()[index].ptr,
                    bytes: spans()[index].bytes - 2,
                },
                DeviceSpan {
                    ptr: spans()[index].ptr,
                    bytes: usize::MAX,
                },
            ] {
                let mut s = spans();
                s[index] = replacement;
                assert!(
                    LtPlan::new(mode)
                        .bind(s[0], s[1], s[2], s[3], 0x9876)
                        .is_err()
                );
            }
        }
        for left in 0..4 {
            for right in left + 1..4 {
                let mut s = spans();
                s[right].ptr = s[left].ptr + 256;
                assert!(
                    LtPlan::new(mode)
                        .bind(s[0], s[1], s[2], s[3], 0x9876)
                        .is_err()
                );
            }
        }
    }
}

struct RecordingIo {
    events: Vec<&'static str>,
    failures: Vec<&'static str>,
    selected: SelectedAlgorithm,
    expected_mask: Option<u32>,
}
impl RecordingIo {
    fn new(mode: LtMode) -> Self {
        Self {
            events: vec![],
            failures: vec![],
            selected: algorithm(),
            expected_mask: mode.preference_mask(),
        }
    }
    fn event(&mut self, name: &'static str) -> Result<()> {
        self.events.push(name);
        if self.failures.contains(&name) {
            bail!("injected {name}");
        }
        Ok(())
    }
}
impl LtIo for RecordingIo {
    fn configure(&mut self, bound: &BoundLt) -> Result<()> {
        assert_eq!(bound.mode().preference_mask(), self.expected_mask);
        self.event("configure")
    }
    fn select_first(&mut self) -> Result<SelectedAlgorithm> {
        self.event("select_first")?;
        Ok(self.selected.clone())
    }
    fn matmul(&mut self) -> Result<()> {
        self.event("matmul")
    }
    fn synchronize(&mut self) -> Result<()> {
        self.event("synchronize")
    }
    fn close(&mut self) -> Result<()> {
        self.event("close")
    }
}

#[test]
fn actual_protocol_selects_one_heuristic_then_launches_fences_and_closes() {
    for mode in [LtMode::Baseline, LtMode::ComputeTypeOnly] {
        let mut io = RecordingIo::new(mode);
        let selected = execute(&mut io, &bound(mode)).unwrap();
        assert_eq!(selected.reduction_scheme, 2);
        assert_eq!(
            io.events,
            [
                "configure",
                "select_first",
                "matmul",
                "synchronize",
                "close"
            ]
        );
    }
}

#[test]
fn compute_only_admission_rejects_output_type_reduction_before_matmul() {
    for (split, reduction) in [(1, 0), (9, 2)] {
        let mut io = RecordingIo::new(LtMode::ComputeTypeOnly);
        io.selected.split_k = split;
        io.selected.reduction_scheme = reduction;
        assert!(execute(&mut io, &bound(LtMode::ComputeTypeOnly)).is_ok());
    }
    for reduction in [1, 4] {
        let mut io = RecordingIo::new(LtMode::Baseline);
        io.selected.reduction_scheme = reduction;
        assert!(execute(&mut io, &bound(LtMode::Baseline)).is_ok());
        let mut io = RecordingIo::new(LtMode::ComputeTypeOnly);
        io.selected.reduction_scheme = reduction;
        assert!(execute(&mut io, &bound(LtMode::ComputeTypeOnly)).is_err());
        assert_eq!(
            io.events,
            ["configure", "select_first", "synchronize", "close"]
        );
    }
}

#[test]
fn malformed_heuristic_metadata_never_reaches_the_device_call() {
    let mut invalid = Vec::new();
    for count in [0, 2] {
        let mut a = algorithm();
        a.returned = count;
        invalid.push(a);
    }
    let mut a = algorithm();
    a.state = 1;
    invalid.push(a);
    let mut a = algorithm();
    a.algorithm_id = -1;
    invalid.push(a);
    let mut a = algorithm();
    a.split_k = -1;
    invalid.push(a);
    let mut a = algorithm();
    a.reduction_scheme = 3;
    invalid.push(a);
    let mut a = algorithm();
    a.reduction_scheme = 0;
    invalid.push(a); // split9 needs reduction
    let mut a = algorithm();
    a.workspace_bytes = 64 * 1024 * 1024 + 1;
    invalid.push(a);
    for waves in [-1.0, f32::INFINITY, f32::NAN] {
        let mut a = algorithm();
        a.waves = waves;
        invalid.push(a);
    }
    for selected in invalid {
        let mut io = RecordingIo::new(LtMode::Baseline);
        io.selected = selected;
        assert!(execute(&mut io, &bound(LtMode::Baseline)).is_err());
        assert_eq!(
            io.events,
            ["configure", "select_first", "synchronize", "close"]
        );
    }
}

#[test]
fn all_attempted_prefix_failures_drain_close_and_retain_cleanup_errors() {
    for (failure, prefix) in [("configure", 1), ("select_first", 2), ("matmul", 3)] {
        let mut io = RecordingIo::new(LtMode::Baseline);
        io.failures.push(failure);
        let error = execute(&mut io, &bound(LtMode::Baseline))
            .unwrap_err()
            .to_string();
        assert!(error.contains(failure));
        assert_eq!(&io.events[prefix..], ["synchronize", "close"]);
    }
    for failures in [
        vec!["synchronize"],
        vec!["close"],
        vec!["matmul", "synchronize", "close"],
    ] {
        let mut io = RecordingIo::new(LtMode::ComputeTypeOnly);
        io.failures = failures.clone();
        let error = execute(&mut io, &bound(LtMode::ComputeTypeOnly))
            .unwrap_err()
            .to_string();
        for name in failures {
            assert!(error.contains(name), "lost {name}: {error}");
        }
        assert_eq!(io.events.last(), Some(&"close"));
    }
}
