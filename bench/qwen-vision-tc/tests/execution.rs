// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/contract.rs"]
mod contract;
#[path = "../src/execution.rs"]
mod execution;
#[path = "../src/numerics.rs"]
mod numerics;

use contract::{BoundPlan, Buffers, Family, Handles, Launch, Mode, Plan, Region};
use execution::{ProbeIo, execute};

fn plan(mode: Mode) -> BoundPlan {
    let p = Plan::new(Family::AttentionOutput, 1, 5120, mode).unwrap();
    let b = Buffers {
        a: Region::new(0x1000, p.input_bytes()).unwrap(),
        b: Region::new(0x1000_0000, p.weight_bytes()).unwrap(),
        bias: Region::new(0x2000_0000, p.bias_bytes()).unwrap(),
        c: Region::new(0x3000_0000, p.output_bytes()).unwrap(),
    };
    p.bind(
        b,
        Handles {
            scalar: 11,
            pipelined: 22,
            add_bias: 33,
            fused_bias: 44,
        },
    )
    .unwrap()
}

// A real deferred-write buffer backend for execution ownership tests. This
// models stream visibility, not GEMM arithmetic (covered separately). Copies
// enter `pending`; output becomes readable only after a successful fence.
struct CpuIo {
    calls: Vec<&'static str>,
    input: Vec<u16>,
    input_before: Vec<u16>,
    output: Vec<u16>,
    pending: Option<Vec<u16>>,
    fail_submit: Option<u64>,
    fail_fence: bool,
    corrupt_guard: bool,
    corrupt_input: bool,
    nonfinite: bool,
    poisoned: bool,
}

impl CpuIo {
    fn new() -> Self {
        let input = vec![0x3f80; 1152];
        Self {
            calls: vec![],
            input_before: input.clone(),
            input,
            output: vec![0xffff; 1152],
            pending: None,
            fail_submit: None,
            fail_fence: false,
            corrupt_guard: false,
            corrupt_input: false,
            nonfinite: false,
            poisoned: false,
        }
    }
}

impl ProbeIo for CpuIo {
    fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    fn enqueue(&mut self, launch: &Launch) -> Result<(), String> {
        match launch.kernel() {
            11 | 44 => {
                self.calls.push("gemm-fused");
                self.pending = Some(self.input.clone());
            }
            22 => {
                self.calls.push("gemm");
                self.pending = Some(self.input.clone());
            }
            33 => {
                self.calls.push("bias");
                let p = self.pending.as_mut().ok_or("bias before gemm")?;
                p[0] = 0x4000;
            }
            _ => return Err("unexpected kernel".into()),
        }
        // A failed enqueue may already have submitted work: still must drain.
        if self.fail_submit == Some(launch.kernel()) {
            return Err("enqueue failed".into());
        }
        Ok(())
    }

    fn fence(&mut self) -> Result<(), String> {
        self.calls.push("fence");
        if self.fail_fence {
            return Err("cannot establish completion".into());
        }
        if let Some(p) = self.pending.take() {
            self.output.copy_from_slice(&p);
        }
        if self.nonfinite {
            self.output[1151] = 0xffff;
        }
        if self.corrupt_input {
            self.input[0] = 0;
        }
        Ok(())
    }

    fn verify_inputs_and_guards(&mut self) -> Result<(), String> {
        self.calls.push("verify");
        if self.pending.is_some() {
            return Err("verification before fence".into());
        }
        if self.corrupt_guard || self.input != self.input_before {
            return Err("guard or immutable operand changed".into());
        }
        Ok(())
    }

    fn read_output(&mut self) -> Result<Vec<u16>, String> {
        self.calls.push("read");
        if self.pending.is_some() {
            return Err("read before fence".into());
        }
        Ok(self.output.clone())
    }

    fn poison(&mut self) {
        self.poisoned = true;
        self.calls.push("poison");
    }
}

#[test]
fn separate_mode_publishes_only_after_both_launches_and_fence() {
    let mut io = CpuIo::new();
    let out = execute(&plan(Mode::UpstreamSeparate), &mut io).unwrap();
    assert_eq!(io.calls, ["gemm", "bias", "fence", "verify", "read"]);
    assert_eq!(out.len(), 1152);
    assert_eq!(out[0], 0x4000);
    assert!(out[1..].iter().all(|x| *x == 0x3f80));
    assert_eq!(io.input, io.input_before);
    assert!(io.pending.is_none());
}

#[test]
fn single_launch_modes_do_not_enqueue_a_second_bias() {
    for mode in [Mode::Scalar, Mode::FusedBias] {
        let mut io = CpuIo::new();
        assert_eq!(execute(&plan(mode), &mut io).unwrap(), io.input_before);
        assert_eq!(io.calls, ["gemm-fused", "fence", "verify", "read"]);
    }
}

#[test]
fn failed_first_or_second_enqueue_drains_but_never_publishes() {
    for kernel in [22, 33] {
        let mut io = CpuIo::new();
        io.fail_submit = Some(kernel);
        let error = execute(&plan(Mode::UpstreamSeparate), &mut io).unwrap_err();
        assert!(error.contains("enqueue failed"));
        assert!(io.pending.is_none(), "submitted bytes were not drained");
        assert!(!io.calls.contains(&"read"));
        assert!(!io.poisoned, "successful drain permits owner cleanup");
        assert_eq!(io.calls.last(), Some(&"fence"));
        if kernel == 22 {
            assert!(!io.calls.contains(&"bias"));
        }
    }
}

#[test]
fn unfenceable_work_poisoned_without_read_or_reuse() {
    for failed_submit in [None, Some(22), Some(33)] {
        let mut io = CpuIo::new();
        io.fail_submit = failed_submit;
        io.fail_fence = true;
        let error = execute(&plan(Mode::UpstreamSeparate), &mut io).unwrap_err();
        assert!(error.contains("completion"));
        if failed_submit.is_some() {
            assert!(error.contains("enqueue failed"));
        }
        assert!(io.poisoned);
        assert!(io.pending.is_some(), "owned data must remain retained");
        assert!(!io.calls.contains(&"read"));
        let before = io.calls.clone();
        assert!(execute(&plan(Mode::Scalar), &mut io).is_err());
        assert_eq!(io.calls, before, "poisoned owner must not enqueue anything");
    }
}

#[test]
fn nonfinite_or_operand_corruption_is_an_error_after_drain() {
    for fault in 0..3 {
        let mut io = CpuIo::new();
        io.nonfinite = fault == 0;
        io.corrupt_input = fault == 1;
        io.corrupt_guard = fault == 2;
        assert!(execute(&plan(Mode::Scalar), &mut io).is_err());
        assert!(io.pending.is_none());
        assert!(!io.poisoned);
        if fault != 0 {
            assert!(!io.calls.contains(&"read"));
        }
    }
}
