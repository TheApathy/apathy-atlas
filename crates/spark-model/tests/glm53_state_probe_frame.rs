// SPDX-License-Identifier: AGPL-3.0-only

//! RED: value-only state-probe admission and freshness; no model, environment, or I/O.

#[path = "../src/model/glm53/state_probe_frame.rs"]
mod state_probe_frame;

use state_probe_frame::{StateProbeFrame, StateProbeStamp};

fn frame() -> StateProbeFrame {
    StateProbeFrame {
        generation: 7,
        nonce: 23,
        position: 12,
        context: 12,
        capacity: 2048,
        context_capacity: 2047,
        stream: 17,
        model_stream: 17,
        live: true,
        poisoned: false,
        capturing: false,
        prefill: false,
    }
}

fn assert_getters(stamp: &StateProbeStamp, expected: StateProbeFrame) {
    let generation: u64 = stamp.generation();
    let nonce: u64 = stamp.nonce();
    let position: u32 = stamp.position();
    let context: u32 = stamp.context();
    let stream: u64 = stamp.stream();
    assert_eq!(generation, expected.generation);
    assert_eq!(nonce, expected.nonce);
    assert_eq!(position, expected.position);
    assert_eq!(context, expected.context);
    assert_eq!(stream, expected.stream);
}

#[test]
fn frame_is_copyable_and_valid_stamp_preserves_exact_public_values() {
    fn assert_traits<T: Clone + Copy + std::fmt::Debug>() {}
    assert_traits::<StateProbeFrame>();
    let input = frame();
    let stamp = StateProbeStamp::new(input).unwrap();
    assert_getters(&stamp, input);
    stamp.check(input).unwrap();
    stamp.check(input).unwrap();
}

#[test]
fn zero_nondefault_and_maximum_streams_are_valid_only_when_model_stream_matches() {
    for stream in [0, 17, u64::MAX] {
        let input = StateProbeFrame {
            stream,
            model_stream: stream,
            ..frame()
        };
        let stamp = StateProbeStamp::new(input).unwrap();
        assert_getters(&stamp, input);
        stamp.check(input).unwrap();
        let other = if stream == 0 { 17 } else { 0 };
        assert!(
            StateProbeStamp::new(StateProbeFrame {
                stream: other,
                ..input
            })
            .is_err()
        );
        assert!(
            StateProbeStamp::new(StateProbeFrame {
                model_stream: other,
                ..input
            })
            .is_err()
        );
    }
}

#[test]
fn positive_generation_and_nonce_keep_all_u64_bits_without_an_increment() {
    for generation in [1, 7, u64::MAX] {
        for nonce in [1, 23, u64::MAX] {
            let input = StateProbeFrame {
                generation,
                nonce,
                ..frame()
            };
            let stamp = StateProbeStamp::new(input).unwrap();
            assert_getters(&stamp, input);
            stamp.check(input).unwrap();
        }
    }
}

#[test]
fn explicit_capacities_admit_both_endpoints_without_a_hidden_model_or_drafter_limit() {
    for (position, capacity, context_capacity) in [
        (1, 1, 1),
        (1, 2048, 2047),
        (2047, 2048, 2047),
        (2048, 2048, 2048),
        (3000, 3000, 4096),
        (4096, 8192, 4096),
        (1, u32::MAX, 1),
        (1, 1, u32::MAX),
        (u32::MAX, u32::MAX, u32::MAX),
    ] {
        let input = StateProbeFrame {
            position,
            context: position,
            capacity,
            context_capacity,
            ..frame()
        };
        let stamp = StateProbeStamp::new(input).unwrap();
        assert_getters(&stamp, input);
        stamp.check(input).unwrap();
    }
}

#[test]
fn all_zero_mismatch_and_noncommitted_frames_fail_new_and_check() {
    let original = frame();
    let stamp = StateProbeStamp::new(original).unwrap();
    for (name, invalid) in [
        (
            "generation",
            StateProbeFrame {
                generation: 0,
                ..original
            },
        ),
        (
            "nonce",
            StateProbeFrame {
                nonce: 0,
                ..original
            },
        ),
        (
            "capacity",
            StateProbeFrame {
                capacity: 0,
                ..original
            },
        ),
        (
            "context_capacity",
            StateProbeFrame {
                context_capacity: 0,
                ..original
            },
        ),
        (
            "position",
            StateProbeFrame {
                position: 0,
                ..original
            },
        ),
        (
            "empty committed frame",
            StateProbeFrame {
                position: 0,
                context: 0,
                ..original
            },
        ),
        (
            "context",
            StateProbeFrame {
                context: 0,
                ..original
            },
        ),
        (
            "target limit",
            StateProbeFrame {
                capacity: 11,
                ..original
            },
        ),
        (
            "drafter limit",
            StateProbeFrame {
                context_capacity: 11,
                ..original
            },
        ),
        (
            "context behind",
            StateProbeFrame {
                context: 11,
                ..original
            },
        ),
        (
            "context ahead",
            StateProbeFrame {
                context: 13,
                ..original
            },
        ),
        (
            "position beyond target",
            StateProbeFrame {
                position: 2049,
                context: 2049,
                context_capacity: 4096,
                ..original
            },
        ),
        (
            "position beyond context",
            StateProbeFrame {
                position: 2048,
                context: 2048,
                ..original
            },
        ),
        (
            "stream",
            StateProbeFrame {
                stream: 0,
                ..original
            },
        ),
        (
            "model stream",
            StateProbeFrame {
                model_stream: 0,
                ..original
            },
        ),
        (
            "not live",
            StateProbeFrame {
                live: false,
                ..original
            },
        ),
        (
            "poisoned",
            StateProbeFrame {
                poisoned: true,
                ..original
            },
        ),
        (
            "capture",
            StateProbeFrame {
                capturing: true,
                ..original
            },
        ),
        (
            "prefill",
            StateProbeFrame {
                prefill: true,
                ..original
            },
        ),
    ] {
        assert!(
            StateProbeStamp::new(invalid).is_err(),
            "new admitted {name}: {invalid:?}"
        );
        assert!(
            stamp.check(invalid).is_err(),
            "check admitted {name}: {invalid:?}"
        );
        // A failed check must not rebind or poison this immutable value stamp.
        stamp.check(original).unwrap();
        assert_getters(&stamp, original);
    }
}

#[test]
fn each_frame_field_is_part_of_the_stamp_including_nonexported_capacities_and_flags() {
    let original = frame();
    let stamp = StateProbeStamp::new(original).unwrap();
    for (name, changed) in [
        (
            "generation",
            StateProbeFrame {
                generation: 8,
                ..original
            },
        ),
        (
            "nonce",
            StateProbeFrame {
                nonce: 24,
                ..original
            },
        ),
        (
            "position",
            StateProbeFrame {
                position: 13,
                ..original
            },
        ),
        (
            "context",
            StateProbeFrame {
                context: 13,
                ..original
            },
        ),
        (
            "capacity",
            StateProbeFrame {
                capacity: 4096,
                ..original
            },
        ),
        (
            "context_capacity",
            StateProbeFrame {
                context_capacity: 4096,
                ..original
            },
        ),
        (
            "stream",
            StateProbeFrame {
                stream: 18,
                ..original
            },
        ),
        (
            "model_stream",
            StateProbeFrame {
                model_stream: 18,
                ..original
            },
        ),
        (
            "live",
            StateProbeFrame {
                live: false,
                ..original
            },
        ),
        (
            "poisoned",
            StateProbeFrame {
                poisoned: true,
                ..original
            },
        ),
        (
            "capturing",
            StateProbeFrame {
                capturing: true,
                ..original
            },
        ),
        (
            "prefill",
            StateProbeFrame {
                prefill: true,
                ..original
            },
        ),
    ] {
        assert!(
            stamp.check(changed).is_err(),
            "stamp ignored {name}: {changed:?}"
        );
    }
    stamp.check(original).unwrap();
}

#[test]
fn independently_valid_new_frames_cannot_replace_a_prior_stamp() {
    let original = frame();
    let stamp = StateProbeStamp::new(original).unwrap();
    for changed in [
        StateProbeFrame {
            generation: original.generation + 1,
            ..original
        },
        StateProbeFrame {
            nonce: original.nonce + 1,
            ..original
        },
        StateProbeFrame {
            position: 11,
            context: 11,
            ..original
        },
        StateProbeFrame {
            position: 13,
            context: 13,
            ..original
        },
        StateProbeFrame {
            capacity: original.capacity + 1,
            ..original
        },
        StateProbeFrame {
            context_capacity: original.context_capacity + 1,
            ..original
        },
        StateProbeFrame {
            stream: 0,
            model_stream: 0,
            ..original
        },
        StateProbeFrame {
            stream: u64::MAX,
            model_stream: u64::MAX,
            ..original
        },
    ] {
        let new_stamp = StateProbeStamp::new(changed).unwrap();
        new_stamp.check(changed).unwrap();
        assert!(
            stamp.check(changed).is_err(),
            "old stamp rebound: {changed:?}"
        );
        assert!(
            new_stamp.check(original).is_err(),
            "new stamp accepted old frame"
        );
        stamp.check(original).unwrap();
    }
}

#[test]
fn later_input_mutation_cannot_change_the_stored_stamp_or_its_getters() {
    let original = frame();
    let mut input = original;
    let stamp = StateProbeStamp::new(input).unwrap();
    input.generation = 1;
    input.nonce = u64::MAX;
    input.position = 2047;
    input.context = 2047;
    input.capacity = 8192;
    input.context_capacity = 4096;
    input.stream = 0;
    input.model_stream = 0;
    StateProbeStamp::new(input).unwrap();
    assert!(stamp.check(input).is_err());
    assert_getters(&stamp, original);
    stamp.check(original).unwrap();
}
