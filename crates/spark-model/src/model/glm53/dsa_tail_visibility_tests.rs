// SPDX-License-Identifier: AGPL-3.0-only

//! Actual scalar DSA call wiring, not a CUDA execution or numerical parity claim.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Pool {
        prior: u64,
        output: u64,
        stream: u64,
    },
    Selector {
        tail: u64,
        stream: u64,
    },
}

#[derive(Default)]
pub(super) struct Recorder {
    events: Mutex<Vec<Event>>,
}

impl Recorder {
    pub(super) fn record(&self, name: &str, params: &[*mut c_void], stream: u64) {
        let pointer = |index: usize| {
            assert!(params.len() > index && !params[index].is_null());
            // SAFETY: read only the named pointer arguments of the two exact
            // production symbols matched below. No device pointer is dereferenced.
            unsafe { *(params[index] as *const u64) }
        };
        let event = match name {
            "atlas_glm53_dsa_pool_k4" => Event::Pool {
                prior: pointer(5),
                output: pointer(11),
                stream,
            },
            "atlas_glm53_dsa_topk_k4" => Event::Selector {
                tail: pointer(5),
                stream,
            },
            _ => return,
        };
        self.events.lock().unwrap().push(event);
    }
}

fn observe(position: u32) -> (Vec<Event>, Glm53DsaCacheSlots) {
    let gpu = TraceGpu::new();
    let scratch = scratch(&gpu);
    // Existing cache() uses its argument only to size the pool allocation.
    // Include this step's completing pool without changing actual geometry.
    let slots = cache(&gpu, position.checked_add(1).unwrap());
    assert_ne!(
        slots.prior_tail_validity_u8.ptr,
        slots.out_tail_validity_u8.ptr
    );
    assert_eq!(slots.prior_tail_validity_u8.bytes, TAIL_CAP);
    assert_eq!(slots.out_tail_validity_u8.bytes, TAIL_CAP);
    let kernels = Glm53DsaAttentionKernels::load(&gpu).unwrap();
    let input = GgmlIqBuffer {
        ptr: gpu.alloc(8192).unwrap(),
        bytes: 8192,
    };
    let output = GgmlIqBuffer {
        ptr: gpu.alloc(8192).unwrap(),
        bytes: 8192,
    };
    kernels
        .stage(
            &gpu,
            &weights(1536),
            input,
            scratch.dsa_buffers(),
            slots,
            Glm53DsaLayerGeometry {
                position,
                capacity: CAPACITY,
                nonce: 0x1234,
            },
            output,
            7,
        )
        .unwrap();
    let events = gpu.tail_visibility.events.lock().unwrap().clone();
    (events, slots)
}

fn ordered_masks(events: &[Event], slots: Glm53DsaCacheSlots) -> u64 {
    assert_eq!(
        events.len(),
        2,
        "one pool followed by one selector: {events:?}"
    );
    match (events[0], events[1]) {
        (
            Event::Pool {
                prior,
                output,
                stream: pool_stream,
            },
            Event::Selector {
                tail,
                stream: selector_stream,
            },
        ) => {
            assert_eq!(pool_stream, 7);
            assert_eq!(selector_stream, pool_stream);
            assert_eq!(prior, slots.prior_tail_validity_u8.ptr.0);
            assert_eq!(output, slots.out_tail_validity_u8.ptr.0);
            assert_ne!(
                prior, output,
                "pool must not overwrite its prior-mask input"
            );
            tail
        }
        _ => panic!("pool must complete its enqueue before selector: {events:?}"),
    }
}

#[test]
fn scalar_q4_through_q9_selects_the_new_tail_mask_not_the_prior_mask() {
    for position in 4..=9 {
        let (events, slots) = observe(position);
        let tail = ordered_masks(&events, slots);
        assert_eq!(tail, slots.out_tail_validity_u8.ptr.0, "q={position}");
        assert_ne!(tail, slots.prior_tail_validity_u8.ptr.0, "q={position}");
    }
}

#[test]
fn pool_completing_control_preserves_distinct_prior_input_and_post_pool_output() {
    for position in [3, 7, 11] {
        let (events, slots) = observe(position);
        // Completing steps have no raw-tail entries. This control preserves
        // pool ownership/order independently of the selector-input RED above.
        let _ = ordered_masks(&events, slots);
    }
}
