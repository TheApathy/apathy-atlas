// SPDX-License-Identifier: AGPL-3.0-only
// RED: no CUDA imports or calls. The real adapter must consume this same protocol.
#[path = "../src/contract.rs"]
mod contract;
#[path = "../src/protocol.rs"]
mod protocol;

use anyhow::{Result, bail};
use contract::{DeviceSpan, Fc1Plan, ReductionMode};
use protocol::{Command, GemmIo, execute_mode};

#[derive(Default)]
struct RecordingIo {
    commands: Vec<Command>,
    fail_at: Vec<usize>,
}
impl GemmIo for RecordingIo {
    fn execute(&mut self, command: Command) -> Result<()> {
        let index = self.commands.len();
        self.commands.push(command);
        if self.fail_at.contains(&index) {
            bail!("injected operation {index}");
        }
        Ok(())
    }
}
fn bound() -> contract::BoundFc1 {
    Fc1Plan::new()
        .bind(
            DeviceSpan {
                ptr: 0x1000,
                bytes: 40_960,
            },
            DeviceSpan {
                ptr: 0x100000,
                bytes: 11_534_336,
            },
            DeviceSpan {
                ptr: 0x2000000,
                bytes: 225_280,
            },
            DeviceSpan {
                ptr: 0x3000000,
                bytes: 8_519_680,
            },
        )
        .unwrap()
}

#[test]
fn stream_then_workspace_then_math_then_gemm_then_completion_then_restore() {
    let bound = bound();
    for mode in [ReductionMode::Default, ReductionMode::Full] {
        let mut io = RecordingIo::default();
        execute_mode(&mut io, &bound, 0x1234, mode).unwrap();
        assert_eq!(
            io.commands,
            vec![
                Command::SetStream(0x1234),
                Command::SetWorkspace(DeviceSpan {
                    ptr: 0x3000000,
                    bytes: 8_519_680
                }),
                Command::SetHostPointerMode,
                Command::SetMathMode(mode.math_mode()),
                Command::Gemm(bound.call()),
                Command::Synchronize,
                Command::SetMathMode(0)
            ]
        );
    }
}

#[test]
fn null_stream_fails_before_any_handle_or_device_effect() {
    let mut io = RecordingIo::default();
    assert!(execute_mode(&mut io, &bound(), 0, ReductionMode::Default).is_err());
    assert!(io.commands.is_empty());
}

#[test]
fn every_partial_effect_failure_attempts_completion_and_restores_math() {
    for failure in 0..5 {
        let mut io = RecordingIo {
            fail_at: vec![failure],
            ..RecordingIo::default()
        };
        let error = execute_mode(&mut io, &bound(), 0x1234, ReductionMode::Full).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("injected operation {failure}"))
        );
        assert_eq!(
            &io.commands[io.commands.len() - 2..],
            &[Command::Synchronize, Command::SetMathMode(0)]
        );
        assert_eq!(io.commands.len(), failure + 3);
        if failure < 4 {
            assert!(!io.commands.iter().any(|c| matches!(c, Command::Gemm(_))));
        }
    }
}

#[test]
fn completion_and_restore_failures_never_publish_success_or_hide_each_other() {
    for failure in [5, 6] {
        let mut io = RecordingIo {
            fail_at: vec![failure],
            ..RecordingIo::default()
        };
        assert!(execute_mode(&mut io, &bound(), 0x1234, ReductionMode::Full).is_err());
        assert_eq!(io.commands.last(), Some(&Command::SetMathMode(0)));
    }
    // GEMM fails at4, its mandatory fence at5 also fails; preserve both diagnostics.
    let mut io = RecordingIo {
        fail_at: vec![4, 5],
        ..RecordingIo::default()
    };
    let error = execute_mode(&mut io, &bound(), 0x1234, ReductionMode::Full).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("injected operation 4") && text.contains("injected operation 5"));
    assert_eq!(io.commands.last(), Some(&Command::SetMathMode(0)));
}
