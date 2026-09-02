// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::weight_loader::{GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage};

const ARENA: u64 = 0x1_0000_0000;

fn plan() -> Glm53ContextPlan {
    Glm53ContextPlan::new(1, GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage::BF16).unwrap()
}

fn layout() -> Glm53T1StateLayout {
    Glm53T1StateLayout::exact().unwrap()
}

/// The real arena places the T1 transaction after the persistent context.
fn real_t1_offset() -> u64 {
    let arena = super::super::arena::Glm53ArenaPlan::exact_full_1m_b1().unwrap();
    arena.t1_transaction.offset_bytes
}

#[test]
fn every_ordinal_binds_a_distinct_four_mib_slot_that_tiles_the_staged_region() {
    let (context, layout, t1) = (plan(), layout(), real_t1_offset());
    let mut spans = Vec::new();
    for ordinal in 0..GLM53_KDA_RECURRENT_ORDINALS {
        let state =
            Glm53KdaScratchState::stage(DevicePtr(ARENA), t1, &layout, &context, ordinal).unwrap();
        assert_eq!(state.ordinal(), ordinal);
        let buffer = state.buffer();
        assert_eq!(buffer.bytes as u64, GLM53_KDA_RECURRENT_ORDINAL_BYTES);
        spans.push((buffer.ptr.0, buffer.ptr.0 + buffer.bytes as u64));
    }
    assert_eq!(spans.len(), 34);
    spans.sort();
    for pair in spans.windows(2) {
        assert_eq!(pair[0].1, pair[1].0, "staged slots must tile without gaps");
    }
    let total = spans[33].1 - spans[0].0;
    assert_eq!(total, 34 * GLM53_KDA_RECURRENT_ORDINAL_BYTES);
    assert_eq!(total, 142_606_336);
}

/// **The load-bearing test.** If the staged region is ever placed so that a KDA
/// slot lands on persistent state, binding must fail. A `t1_offset` of 0 puts
/// the staged recurrent region directly on top of the persistent context — the
/// exact confusion the two identically-named `kda_recurrent_f32` fields invite.
#[test]
fn a_slot_that_aliases_persistent_state_cannot_be_constructed() {
    let (context, layout) = (plan(), layout());
    let mut refused = 0usize;
    for ordinal in 0..GLM53_KDA_RECURRENT_ORDINALS {
        let error = Glm53KdaScratchState::stage(DevicePtr(ARENA), 0, &layout, &context, ordinal)
            .expect_err("a persistent alias must not be bindable");
        let message = format!("{error:#}");
        assert!(
            message.contains("aliases persistent"),
            "refusal must name the hazard: {message}"
        );
        assert!(
            message.contains("rolled back"),
            "refusal must explain why it matters: {message}"
        );
        refused += 1;
    }
    assert_eq!(refused, 34, "every ordinal must refuse, not just the first");
}

/// A staged region overlapping only the *tail* of persistent state must also be
/// refused — partial aliasing corrupts just as thoroughly as full aliasing.
#[test]
fn partial_overlap_with_persistent_state_is_refused() {
    let (context, layout) = (plan(), layout());
    // Place the staged region so its first slot straddles the end of the
    // persistent context region.
    let persistent_end = context.dsa_latent.offset_bytes + context.dsa_latent.allocation_bytes;
    let straddle = persistent_end.saturating_sub(GLM53_KDA_RECURRENT_ORDINAL_BYTES / 2);
    let result = Glm53KdaScratchState::stage(DevicePtr(ARENA), straddle, &layout, &context, 0);
    assert!(result.is_err(), "a straddling slot must be refused");
}

#[test]
fn ordinals_outside_the_kda_census_are_refused() {
    let (context, layout, t1) = (plan(), layout(), real_t1_offset());
    for ordinal in [GLM53_KDA_RECURRENT_ORDINALS, 45, 1_000] {
        assert!(
            Glm53KdaScratchState::stage(DevicePtr(ARENA), t1, &layout, &context, ordinal).is_err(),
            "ordinal {ordinal} must be refused"
        );
    }
    // 33 is the last valid ordinal.
    assert!(Glm53KdaScratchState::stage(DevicePtr(ARENA), t1, &layout, &context, 33).is_ok());
}

#[test]
fn a_null_arena_base_is_refused() {
    let (context, layout, t1) = (plan(), layout(), real_t1_offset());
    assert!(Glm53KdaScratchState::stage(DevicePtr(0), t1, &layout, &context, 0).is_err());
}

/// The guard is only worth anything if it cannot be walked around.
///
/// `Glm53KdaScratchState` makes a persistent alias unconstructible, but nothing
/// stops a future caller from building `Glm53KdaDecodeBuffers` by hand and
/// passing whatever `GgmlIqBuffer` is nearest — which is exactly how the
/// original hazard would return when the `Attention` events are wired.
///
/// This scans every production source in `model/glm53` (directory-walked, so a
/// new file is covered the moment it lands) and requires that any module
/// touching the in-place KDA state buffers also goes through the scratch
/// binding. It fails the instant someone wires KDA attention without it.
///
/// Today no production module binds KDA state, so the directory scan alone
/// would pass vacuously and prove nothing. The detection predicate is therefore
/// exercised directly against synthetic sources first — a guard that has never
/// been shown to fire is not a guard.
fn binds_kda_state_unsafely(name: &str, source: &str) -> bool {
    if name == "kda_state_binding.rs" {
        // This module names the types in its own docs to explain the hazard.
        return false;
    }
    let touches_state = source.contains("Glm53KdaDecodeBuffers")
        || source.contains("Glm53KdaPrefillBuffers")
        || source.contains("state_f32");
    touches_state && !source.contains("Glm53KdaScratchState")
}

#[test]
fn the_bypass_guard_actually_fires_on_a_violation() {
    // The shape the hazard would take when `Attention` is wired: a decode
    // buffer built straight from a raw workspace pointer.
    let offender = r#"
        let buffers = Glm53KdaDecodeBuffers {
            state_f32: self.bound.hidden_a,
            ..
        };
    "#;
    assert!(
        binds_kda_state_unsafely("attention.rs", offender),
        "the guard must flag a raw KDA state binding"
    );

    // The same code routed through the scratch binding is accepted.
    let compliant = r#"
        let scratch = Glm53KdaScratchState::stage(base, t1, &layout, &context, ordinal)?;
        let buffers = Glm53KdaDecodeBuffers {
            state_f32: scratch.buffer(),
            ..
        };
    "#;
    assert!(!binds_kda_state_unsafely("attention.rs", compliant));

    // A module that never touches recurrent state is irrelevant to the guard.
    assert!(!binds_kda_state_unsafely("dispatch.rs", "let x = 1;"));
}

#[test]
fn no_production_module_builds_kda_state_buffers_without_the_scratch_binding() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("model")
        .join("glm53");
    let mut checked = 0usize;
    let mut offenders = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        // Production only: test files legitimately fabricate raw buffers.
        if !name.ends_with(".rs") || name.ends_with("_tests.rs") || name.contains("sha256") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        checked += 1;
        if binds_kda_state_unsafely(&name, &source) {
            offenders.push(name);
        }
    }
    assert!(
        checked > 5,
        "directory walk found too few sources: {checked}"
    );
    assert!(
        offenders.is_empty(),
        "these modules bind KDA recurrent state without Glm53KdaScratchState, so a \
         persistent-state alias is reachable again: {offenders:?}"
    );
}

/// Persistent recurrent state holds the active slot **and** a padding slot that
/// `SsmStatePool` requires and that must never be written. Every ordinal must
/// address slot 0 only, and priming must read from there.
#[test]
fn persistent_slots_address_the_active_slot_never_the_padding_slot() {
    let (context, layout, t1) = (plan(), layout(), real_t1_offset());
    let slot_span = GLM53_KDA_RECURRENT_ORDINAL_BYTES * GLM53_KDA_RECURRENT_ORDINALS as u64;
    // The region really does carry more than one slot; if it ever stops doing
    // so this test is checking nothing and should fail loudly.
    assert!(
        context.kda_recurrent_f32.payload_bytes > slot_span,
        "expected a padding slot beyond the active one"
    );
    let active_end = ARENA + context.kda_recurrent_f32.offset_bytes + slot_span;
    for ordinal in 0..GLM53_KDA_RECURRENT_ORDINALS {
        let state =
            Glm53KdaScratchState::stage(DevicePtr(ARENA), t1, &layout, &context, ordinal).unwrap();
        let persistent = state.persistent();
        assert_eq!(persistent.bytes as u64, GLM53_KDA_RECURRENT_ORDINAL_BYTES);
        assert!(
            persistent.ptr.0 + persistent.bytes as u64 <= active_end,
            "ordinal {ordinal} reaches into the padding slot"
        );
        // Persistent and scratch must never be the same buffer.
        assert_ne!(persistent.ptr.0, state.buffer().ptr.0);
    }
}

/// Priming copies persistent into scratch, in that direction. The reverse would
/// publish an unaccepted step straight into persistent state.
#[test]
fn priming_copies_persistent_into_scratch_and_never_the_reverse() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use std::ffi::c_void;
    use std::sync::Mutex;

    struct CopyGpu {
        inner: MockGpuBackend,
        copies: Mutex<Vec<(u64, u64, usize)>>,
    }
    impl spark_runtime::gpu::GpuBackend for CopyGpu {
        fn alloc(&self, b: usize) -> Result<DevicePtr> {
            self.inner.alloc(b)
        }
        fn alloc_managed(&self, b: usize) -> Result<DevicePtr> {
            self.inner.alloc_managed(b)
        }
        fn free(&self, p: DevicePtr) -> Result<()> {
            self.inner.free(p)
        }
        fn copy_h2d(&self, s: &[u8], d: DevicePtr) -> Result<()> {
            self.inner.copy_h2d(s, d)
        }
        fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
            self.inner.copy_d2h(s, d)
        }
        fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, b: usize) -> Result<()> {
            self.copies.lock().unwrap().push((s.0, d.0, b));
            Ok(())
        }
        fn memset(&self, p: DevicePtr, v: u8, b: usize) -> Result<()> {
            self.inner.memset(p, v, b)
        }
        fn memset_async(&self, p: DevicePtr, v: u8, b: usize, _s: u64) -> Result<()> {
            self.inner.memset(p, v, b)
        }
        fn launch(
            &self,
            _f: spark_runtime::gpu::KernelHandle,
            _g: [u32; 3],
            _b: [u32; 3],
            _sh: u32,
            _st: u64,
            _p: &mut [*mut c_void],
        ) -> Result<()> {
            Ok(())
        }
        fn synchronize(&self, _s: u64) -> Result<()> {
            Ok(())
        }
        fn default_stream(&self) -> u64 {
            0
        }
        fn kernel(&self, _m: &str, _n: &str) -> Result<spark_runtime::gpu::KernelHandle> {
            Ok(spark_runtime::gpu::KernelHandle(1))
        }
        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }
        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
    }

    let gpu = CopyGpu {
        inner: MockGpuBackend::new(),
        copies: Mutex::new(Vec::new()),
    };
    let (context, layout, t1) = (plan(), layout(), real_t1_offset());
    let state = Glm53KdaScratchState::stage(DevicePtr(ARENA), t1, &layout, &context, 7).unwrap();
    state.prime(&gpu, 0).unwrap();

    let copies = gpu.copies.lock().unwrap().clone();
    assert_eq!(copies.len(), 1);
    let (source, destination, bytes) = copies[0];
    assert_eq!(source, state.persistent().ptr.0, "must read persistent");
    assert_eq!(destination, state.buffer().ptr.0, "must write scratch");
    assert_eq!(bytes as u64, GLM53_KDA_RECURRENT_ORDINAL_BYTES);
}
