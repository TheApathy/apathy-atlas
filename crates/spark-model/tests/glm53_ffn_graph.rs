// SPDX-License-Identifier: AGPL-3.0-only
#[path = "../src/model/glm53/ffn_graph.rs"]
mod ffn_graph;

use ffn_graph::{Binding, GraphCache, GraphIo, Key, Setting};
use std::ffi::OsStr;

#[derive(Default)]
struct Io {
    calls: Vec<&'static str>,
    fail: Option<&'static str>,
    panic_body: bool,
    zero: bool,
    panic_destroy: bool,
}
impl Io {
    fn call(&mut self, name: &'static str) -> Result<(), String> {
        self.calls.push(name);
        if self.fail == Some(name) {
            Err(name.into())
        } else {
            Ok(())
        }
    }
}
impl GraphIo for Io {
    fn dispatch(&mut self) -> Result<(), String> {
        self.call("body")?;
        assert!(!self.panic_body, "injected body panic");
        Ok(())
    }
    fn begin(&mut self, _: u64) -> Result<(), String> {
        self.call("begin")
    }
    fn end(&mut self, _: u64) -> Result<u64, String> {
        self.call("end")?;
        Ok(if self.zero { 0 } else { 73 })
    }
    fn launch(&mut self, graph: u64, _: u64) -> Result<(), String> {
        assert_eq!(graph, 73);
        self.call("launch")
    }
    fn synchronize(&mut self, _: u64) -> Result<(), String> {
        self.call("fence")
    }
    fn destroy(&mut self, graph: u64) -> Result<(), String> {
        assert_eq!(graph, 73);
        assert!(!self.panic_destroy, "injected indeterminate destroy panic");
        self.call("destroy")
    }
}
fn binding() -> Binding {
    Binding {
        stream: 9,
        owners: [256, 512, 768],
    }
}
fn key() -> Key {
    Key::new(4, 3).unwrap()
}
fn warm(cache: &mut GraphCache, io: &mut Io) {
    cache.execute(key(), binding(), io).unwrap();
    io.calls.clear();
}

#[test]
fn selector_is_strict_and_default_off() {
    assert!(!Setting::parse(None).unwrap().enabled());
    assert!(!Setting::parse(Some(OsStr::new("0"))).unwrap().enabled());
    assert!(Setting::parse(Some(OsStr::new("1"))).unwrap().enabled());
    for value in ["", "true", "01", " 1", "2", "1\n"] {
        assert!(Setting::parse(Some(OsStr::new(value))).is_err());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(Setting::parse(Some(OsStr::from_bytes(&[255]))).is_err());
    }
}

#[test]
fn key_space_is_exactly_294_and_excludes_attention_dense_and_prefill() {
    for rows in 2..=8 {
        for layer in 3..=44 {
            assert!(Key::new(rows, layer).is_ok());
        }
    }
    for (r, l) in [
        (0, 3),
        (1, 3),
        (9, 3),
        (2048, 3),
        (4, 0),
        (4, 2),
        (4, 45),
        (u32::MAX, u32::MAX),
    ] {
        assert!(Key::new(r, l).is_err());
    }
}

#[test]
fn cold_then_capture_then_replay_execute_exactly_once() {
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    warm(&mut cache, &mut io);
    cache.execute(key(), binding(), &mut io).unwrap();
    assert_eq!(io.calls, ["begin", "body", "end", "launch"]);
    io.calls.clear();
    cache.execute(key(), binding(), &mut io).unwrap();
    assert_eq!(io.calls, ["launch"]);
    assert_eq!(cache.counts(), [1, 1, 1]);
    assert_eq!(cache.row_counts()[2], [1, 1, 1]);
    io.calls.clear();
    cache.drain(&mut io).unwrap();
    assert_eq!(io.calls, ["fence", "destroy"]);
}

#[test]
fn different_rows_and_layers_warm_independently() {
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    for rows in 2..=8 {
        for layer in 3..=44 {
            cache
                .execute(Key::new(rows, layer).unwrap(), binding(), &mut io)
                .unwrap();
        }
    }
    assert_eq!(io.calls, vec!["body"; 294]);
    assert_eq!(cache.counts(), [294, 0, 0]);
    assert_eq!(cache.row_counts()[..7], [[42, 0, 0]; 7]);
    assert_eq!(cache.row_counts()[7], [0, 0, 0]);
}

#[test]
fn invalid_or_changed_binding_never_enqueues() {
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    let mut invalid = binding();
    invalid.stream = 0;
    assert!(cache.execute(key(), invalid, &mut io).is_err());
    assert!(io.calls.is_empty());
    warm(&mut cache, &mut io);
    for bad in [
        Binding {
            stream: 10,
            ..binding()
        },
        Binding {
            owners: [256, 512, 1024],
            ..binding()
        },
    ] {
        assert!(cache.execute(key(), bad, &mut io).is_err());
        assert!(io.calls.is_empty());
    }
}

#[test]
fn begin_failure_poisoned_without_end_or_eager_retry() {
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    warm(&mut cache, &mut io);
    io.fail = Some("begin");
    assert!(cache.execute(key(), binding(), &mut io).is_err());
    assert_eq!(io.calls, ["begin"]);
    io.calls.clear();
    io.fail = None;
    assert!(cache.execute(key(), binding(), &mut io).is_err());
    assert!(io.calls.is_empty());
}

#[test]
fn capture_body_failure_ends_and_destroys_without_launch() {
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    warm(&mut cache, &mut io);
    io.fail = Some("body");
    assert!(cache.execute(key(), binding(), &mut io).is_err());
    assert_eq!(io.calls, ["begin", "body", "end", "destroy"]);
    io.calls.clear();
    io.fail = None;
    assert!(cache.execute(key(), binding(), &mut io).is_err());
    assert!(io.calls.is_empty());
}

#[test]
fn capture_body_panic_also_ends_and_destroys_then_stays_poisoned() {
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    warm(&mut cache, &mut io);
    io.panic_body = true;
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cache.execute(
            key(),
            binding(),
            &mut io
        )))
        .is_err()
    );
    assert_eq!(io.calls, ["begin", "body", "end", "destroy"]);
    io.calls.clear();
    io.panic_body = false;
    assert!(cache.execute(key(), binding(), &mut io).is_err());
    assert!(io.calls.is_empty());
}

#[test]
fn end_failure_and_null_graph_never_launch_or_retry() {
    for zero in [false, true] {
        let mut cache = GraphCache::new();
        let mut io = Io::default();
        warm(&mut cache, &mut io);
        if zero {
            io.zero = true;
        } else {
            io.fail = Some("end");
        }
        assert!(cache.execute(key(), binding(), &mut io).is_err());
        assert_eq!(io.calls, ["begin", "body", "end"]);
    }
}

#[test]
fn launch_failure_retains_graph_until_a_successful_drain() {
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    warm(&mut cache, &mut io);
    io.fail = Some("launch");
    assert!(cache.execute(key(), binding(), &mut io).is_err());
    io.calls.clear();
    io.fail = Some("fence");
    assert!(cache.drain(&mut io).is_err());
    assert_eq!(io.calls, ["fence"]);
    io.calls.clear();
    io.fail = Some("destroy");
    assert!(cache.drain(&mut io).is_err());
    assert_eq!(io.calls, ["fence", "destroy"]);
    io.calls.clear();
    io.fail = None;
    cache.drain(&mut io).unwrap();
    assert_eq!(io.calls, ["fence", "destroy"]);
}

#[test]
fn indeterminate_destroy_panic_never_retries_the_handle_or_frees_model_owners() {
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    warm(&mut cache, &mut io);
    io.fail = Some("body");
    io.panic_destroy = true;
    assert!(cache.execute(key(), binding(), &mut io).is_err());
    io.calls.clear();
    io.fail = None;
    io.panic_destroy = false;
    assert!(cache.drain(&mut io).is_err());
    assert!(io.calls.is_empty());
}

#[test]
fn draining_is_terminal_even_for_a_only_warmed_cache() {
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    warm(&mut cache, &mut io);
    cache.drain(&mut io).unwrap();
    io.calls.clear();
    assert!(cache.execute(key(), binding(), &mut io).is_err());
    assert!(io.calls.is_empty());
}

#[test]
fn group_errors_and_panics_poison_the_model_but_success_does_not() {
    use std::cell::Cell;
    let poisoned = Cell::new(0);
    assert_eq!(
        ffn_graph::with_failure_owner(|| Ok::<_, String>(17), || poisoned.set(poisoned.get() + 1))
            .unwrap(),
        17
    );
    assert_eq!(poisoned.get(), 0);
    assert!(
        ffn_graph::with_failure_owner(
            || Err::<(), _>("injected"),
            || poisoned.set(poisoned.get() + 1)
        )
        .is_err()
    );
    assert_eq!(poisoned.get(), 1);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffn_graph::with_failure_owner(
                || -> Result<(), String> { panic!("injected") },
                || poisoned.set(poisoned.get() + 1),
            )
        }))
        .is_err()
    );
    assert_eq!(poisoned.get(), 2);
}

#[test]
fn scalar_band_is_disjoint_from_verification_keys() {
    use ffn_graph::{ENTRIES, SCALAR_ROW_SLOT, VERIFY_ENTRIES};
    assert_eq!(ENTRIES, 336);
    for layer in 3..=44 {
        let scalar = Key::new_scalar(layer).unwrap().index();
        assert!((VERIFY_ENTRIES..ENTRIES).contains(&scalar));
        assert_eq!(scalar / 42, SCALAR_ROW_SLOT);
        for rows in 2..=8 {
            assert_ne!(Key::new(rows, layer).unwrap().index(), scalar);
        }
    }
    for layer in [0, 2, 45, u32::MAX] {
        assert!(Key::new_scalar(layer).is_err());
    }
    let mut cache = GraphCache::new();
    let mut io = Io::default();
    for layer in 3..=44 {
        cache
            .execute(Key::new_scalar(layer).unwrap(), binding(), &mut io)
            .unwrap();
    }
    assert_eq!(io.calls, vec!["body"; 42]);
    assert_eq!(cache.row_counts()[SCALAR_ROW_SLOT], [42, 0, 0]);
    assert_eq!(cache.row_counts()[..7], [[0, 0, 0]; 7]);
}
