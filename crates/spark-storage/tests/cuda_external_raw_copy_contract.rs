// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only source contract for the external pageable host-copy boundary.

const CUDA_MIN: &str = include_str!("../src/cuda_min.rs");
const CUDA_HOST_COPY: &str = include_str!("../src/cuda_host_copy.rs");
const STORAGE_HSS_OWNER: &str = include_str!("../src/high_speed_swap.rs");
const STORAGE_HSS: &str = include_str!("../src/high_speed_swap/impl_more.rs");
const STORAGE_PREDICTOR: &str = include_str!("../src/predictor.rs");
const STORAGE_BENCH: &str = include_str!("../src/bench.rs");
const STORAGE_IO_URING: &str = include_str!("../src/backend/io_uring.rs");
const STORAGE_POSIX: &str = include_str!("../src/backend/posix.rs");
const STORAGE_BACKEND: &str = include_str!("../src/backend/mod.rs");
const ATLAS_REGISTRY: &str = include_str!("../../atlas-core/src/registry.rs");
const INNERQ: &str = include_str!("../../spark-model/src/layers/qwen3_attention/innerq_driver.rs");

const STORAGE_CALLERS: &[&str] = &[
    STORAGE_HSS,
    STORAGE_PREDICTOR,
    include_str!("../src/backend/io_uring.rs"),
    include_str!("../src/backend/posix.rs"),
    include_str!("predictor_parity.rs"),
    include_str!("high_speed_swap_e2e.rs"),
    include_str!("recall_at_k.rs"),
    include_str!("tiled_attention_parity.rs"),
    include_str!("cuda_graph_capture.rs"),
    include_str!("streaming_attention_e2e.rs"),
    include_str!("../examples/long_context_bench.rs"),
];

fn function_body<'a>(source: &'a str, name: &str, next_name: &str) -> &'a str {
    let marker = format!("pub fn {name}");
    let start = source.find(&marker).expect("function start");
    let next = format!("\npub fn {next_name}");
    let relative_end = source[start..].find(&next).expect("next function");
    &source[start..start + relative_end]
}

fn unsafe_function_body<'a>(source: &'a str, name: &str, next_name: &str) -> &'a str {
    let marker = format!("pub unsafe fn {name}");
    let start = source.find(&marker).expect("unsafe function start");
    let next = format!("\npub fn {next_name}");
    let relative_end = source[start..].find(&next).expect("next function");
    &source[start..start + relative_end]
}

fn function_tail<'a>(source: &'a str, name: &str) -> &'a str {
    let marker = format!("pub fn {name}");
    let start = source.find(&marker).expect("tail function start");
    &source[start..]
}

fn method_body<'a>(source: &'a str, name: &str, next_name: &str) -> &'a str {
    let marker = format!("    pub fn {name}(");
    let start = source.find(&marker).expect("method start");
    let next = format!("\n    pub fn {next_name}(");
    let relative_end = source[start..].find(&next).expect("next method");
    &source[start..start + relative_end]
}

fn private_method_body<'a>(source: &'a str, name: &str, next_name: &str) -> &'a str {
    let marker = format!("    fn {name}(");
    let start = source.find(&marker).expect("private method start");
    let next = format!("\n    fn {next_name}(");
    let relative_end = source[start..].find(&next).expect("next private method");
    &source[start..start + relative_end]
}

#[test]
fn spark_storage_pageable_d2h_is_presynced_and_synchronous() {
    assert!(!CUDA_HOST_COPY.contains("cuMemcpyDtoHAsync_v2"));
    assert!(!CUDA_HOST_COPY.contains("copy_d_to_h_async"));
    let body = function_tail(CUDA_HOST_COPY, "copy_d_to_h");
    assert_eq!(body.matches("stream_sync(stream)?").count(), 1);
    assert_eq!(body.matches("cuMemcpyDtoH_v2(").count(), 1);
    assert!(body.find("stream_sync(stream)?") < body.find("cuMemcpyDtoH_v2("));
}

#[test]
fn spark_storage_h2d_separates_pageable_and_pinned_sources() {
    assert!(!CUDA_HOST_COPY.contains("pub fn copy_h_to_d_async"));
    assert!(!CUDA_HOST_COPY.contains("pub ptr: *mut c_void"));
    assert!(!CUDA_HOST_COPY.contains("pub bytes: usize"));
    assert!(CUDA_MIN.contains("pub use cuda_host_copy::*"));
    assert!(CUDA_HOST_COPY.contains("if bytes == 0"));
    assert!(CUDA_HOST_COPY.contains("std::ptr::NonNull::new(ptr)"));
    assert!(CUDA_HOST_COPY.contains("as_mut_ptr(&mut self)"));

    let group = function_body(CUDA_HOST_COPY, "copy_h_to_d_group", "copy_h_to_d");
    assert_eq!(group.matches("stream_sync(stream)?").count(), 1);
    assert_eq!(group.matches("cuMemcpyHtoD_v2(").count(), 1);
    assert!(group.find("stream_sync(stream)?") < group.find("for (index"));
    assert!(group.find("if status != 0") < group.rfind("Ok(())"));

    let pinned = function_body(CUDA_HOST_COPY, "copy_h_to_d_pinned", "copy_h_to_d_group");
    assert_eq!(pinned.matches("stream_sync(stream)?").count(), 1);
    assert_eq!(pinned.matches("cuMemcpyHtoD_v2(").count(), 1);
    assert!(pinned.find("stream_sync(stream)?") < pinned.find("cuMemcpyHtoD_v2("));

    let pinned_async = unsafe_function_body(
        CUDA_HOST_COPY,
        "copy_h_to_d_pinned_async",
        "copy_h_to_d_pinned",
    );
    assert_eq!(pinned_async.matches("cuMemcpyHtoDAsync_v2(").count(), 1);
    assert!(!pinned_async.contains("stream_sync("));
    assert!(pinned_async.contains("PinnedHostSlice<'_>"));
}

#[test]
fn storage_h2d_callers_preserve_grouping_and_owner_lifetimes() {
    let pageable_callers = [
        STORAGE_HSS,
        STORAGE_PREDICTOR,
        include_str!("../examples/long_context_bench.rs"),
        include_str!("predictor_parity.rs"),
        include_str!("high_speed_swap_e2e.rs"),
        include_str!("recall_at_k.rs"),
        include_str!("tiled_attention_parity.rs"),
        include_str!("cuda_graph_capture.rs"),
        include_str!("streaming_attention_e2e.rs"),
    ];
    for source in pageable_callers {
        assert!(!source.contains("copy_h_to_d_async"));
        assert!(!source.contains("copy_h_to_d_pinned_async"));
    }
    assert_eq!(STORAGE_HSS.matches("HostToDeviceCopy::new(").count(), 2);
    assert_eq!(STORAGE_HSS.matches("copy_h_to_d_group(").count(), 1);
    assert_eq!(STORAGE_PREDICTOR.matches("copy_h_to_d(").count(), 1);

    assert_eq!(
        STORAGE_IO_URING
            .matches("copy_h_to_d_pinned_async(")
            .count(),
        1
    );
    assert!(STORAGE_IO_URING.contains("impl Drop for IoUringBackend"));
    assert!(STORAGE_IO_URING.contains(".all(|transfer| transfer.wait().is_ok())"));
    assert!(STORAGE_IO_URING.contains("if !drained"));
    assert!(STORAGE_IO_URING.contains("std::process::abort()"));
    assert!(STORAGE_IO_URING.contains("poisoned: false"));
    assert_eq!(STORAGE_IO_URING.matches("self.poisoned = true").count(), 2);
    assert!(
        STORAGE_IO_URING.find("self.pending[buf_idx] = Some(PendingTransfer")
            < STORAGE_IO_URING.find("copy_h_to_d_pinned_async(")
    );
    let wait = private_method_body(STORAGE_IO_URING, "wait_buffer_free", "submit_read");
    assert!(wait.find("transfer.wait()?") < wait.find("self.pending[buf_idx] = None"));

    assert_eq!(STORAGE_POSIX.matches("copy_h_to_d_pinned(").count(), 1);
    assert!(!STORAGE_POSIX.contains("copy_h_to_d_pinned_async("));
    assert_eq!(STORAGE_BENCH.matches("copy_h_to_d_pinned(").count(), 2);
    assert!(!STORAGE_BENCH.contains("copy_h_to_d_pinned_async("));
    assert!(STORAGE_BACKEND.contains("order their H2D copies before later work"));
    assert!(STORAGE_BACKEND.contains("must keep the backend and every destination"));
    assert!(
        STORAGE_HSS_OWNER.find("\n    backend: IoUringBackend")
            < STORAGE_HSS_OWNER.find("\n    pool: ScratchPool")
    );
}

#[test]
fn atlas_registry_raw_async_host_copy_surface_is_removed() {
    for forbidden in [
        "cuMemcpyDtoHAsync_v2",
        "cuMemcpyHtoDAsync_v2",
        "copy_d2h_async",
        "copy_h2d_async",
    ] {
        assert!(!ATLAS_REGISTRY.contains(forbidden), "found {forbidden}");
        assert!(!INNERQ.contains(forbidden), "found {forbidden}");
    }

    let group = method_body(ATLAS_REGISTRY, "copy_h2d_group", "copy_d2h");
    assert_eq!(group.matches("self.stream_synchronize(stream)?").count(), 1);
    assert_eq!(group.matches("cuMemcpyHtoD_v2(").count(), 1);
    assert!(group.find("self.stream_synchronize(stream)?") < group.find("for (index"));
    assert!(group.find("if status != 0") < group.rfind("Ok(())"));

    let d2h = method_body(ATLAS_REGISTRY, "copy_d2h", "stream_synchronize");
    assert_eq!(d2h.matches("self.stream_synchronize(stream)?").count(), 1);
    assert_eq!(d2h.matches("cuMemcpyDtoH_v2(").count(), 1);
    assert!(d2h.find("self.stream_synchronize(stream)?") < d2h.find("cuMemcpyDtoH_v2("));
}

#[test]
fn all_storage_and_innerq_callers_use_the_ordered_sync_contract() {
    let storage_calls: usize = STORAGE_CALLERS
        .iter()
        .map(|source| source.matches("copy_d_to_h(").count())
        .sum();
    assert_eq!(storage_calls, 14);
    for source in STORAGE_CALLERS {
        assert!(!source.contains("copy_d_to_h_async"));
        for (offset, _) in source.match_indices("copy_d_to_h(") {
            let end = (offset + 240).min(source.len());
            assert!(
                !source[offset..end].contains("stream_sync("),
                "redundant trailing sync after copy_d_to_h"
            );
        }
    }

    assert_eq!(STORAGE_HSS.matches("copy_d_to_h(").count(), 1);
    assert_eq!(STORAGE_PREDICTOR.matches("copy_d_to_h(").count(), 1);
    assert_eq!(INNERQ.matches("copy_d2h(").count(), 2);
    assert_eq!(INNERQ.matches("copy_h2d_group(").count(), 3);
    assert!(!INNERQ.contains("stream_synchronize("));

    // The eight former raw H2D submissions remain one-for-one descriptors:
    // four start-state writes, calibration-off, two scales, and active last.
    let start = method_body(INNERQ, "start", "maybe_finalize");
    let start_group = start
        .find("reg.copy_h2d_group(")
        .map(|offset| &start[offset..])
        .expect("start group");
    for needle in ["sq_ptr", "count_ptr", "active_ptr", "calib_ptr"] {
        assert_eq!(start_group.matches(&format!("({needle},")).count(), 1);
    }
    assert!(start_group.find("(sq_ptr,") < start_group.find("(count_ptr,"));
    assert!(start_group.find("(count_ptr,") < start_group.find("(active_ptr,"));
    assert!(start_group.find("(active_ptr,") < start_group.find("(calib_ptr,"));

    assert_eq!(INNERQ.matches("let calibration_off = (").count(), 1);
    let activation = INNERQ
        .rfind("reg.copy_h2d_group(")
        .map(|offset| &INNERQ[offset..])
        .expect("activation group");
    assert_eq!(activation.matches("(scale_ptr,").count(), 1);
    assert_eq!(activation.matches("(scale_inv_ptr,").count(), 1);
    assert_eq!(activation.matches("(active_ptr,").count(), 1);
    assert!(activation.find("calibration_off") < activation.find("(scale_ptr,"));
    assert!(activation.find("(scale_ptr,") < activation.find("(scale_inv_ptr,"));
    assert!(activation.find("(scale_inv_ptr,") < activation.find("(active_ptr,"));
}
