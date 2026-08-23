// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only source contracts for the CUDA host-copy boundary.
//!
//! The production regression involved an ordinary pageable `Vec<u8>`, so the
//! generic blocking methods must never route through CUDA's page-locked-only
//! async host-copy APIs. The coalesced stream method has the same constraint.

const CUDA_BACKEND: &str = include_str!("../src/cuda_backend.rs");
const GPU_IMPL: &str = include_str!("../src/cuda_backend/gpu_impl.rs");
const MICROFIXTURE: &str = include_str!("../examples/cuda_blocking_copy_microfixture.rs");

fn method_body<'a>(source: &'a str, name: &str, next_name: &str) -> &'a str {
    let marker = format!("    fn {name}(");
    let unsafe_marker = format!("    unsafe fn {name}(");
    let start = source
        .find(&marker)
        .or_else(|| source.find(&unsafe_marker))
        .expect("method start");
    let next_marker = format!("\n    fn {next_name}(");
    let relative_end = source[start..].find(&next_marker).expect("next method");
    &source[start..start + relative_end]
}

#[test]
fn generic_blocking_host_copies_use_only_synchronous_driver_calls() {
    let h2d = method_body(GPU_IMPL, "copy_h2d", "copy_d2h");
    assert!(h2d.contains("cuMemcpyHtoD_v2("));
    assert!(!h2d.contains("cuMemcpyHtoDAsync_v2("));
    assert_eq!(h2d.matches("cuStreamSynchronize(").count(), 1);
    assert!(h2d.find("cuStreamSynchronize(") < h2d.find("cuMemcpyHtoD_v2("));

    let d2h = method_body(GPU_IMPL, "copy_d2h", "copy_d2h_on_stream");
    assert!(d2h.contains("cuMemcpyDtoH_v2("));
    assert!(!d2h.contains("cuMemcpyDtoHAsync_v2("));
    assert_eq!(d2h.matches("cuStreamSynchronize(").count(), 1);
    assert!(d2h.find("cuStreamSynchronize(") < d2h.find("cuMemcpyDtoH_v2("));
}

#[test]
fn blocking_and_coalesced_cuda_d2h_are_pageable_safe() {
    for declaration in [
        "fn cuMemcpyHtoD_v2(",
        "fn cuMemcpyDtoH_v2(",
        "fn cuMemcpyHtoDAsync_v2(",
    ] {
        assert!(CUDA_BACKEND.contains(declaration), "missing {declaration}");
    }
    assert!(!CUDA_BACKEND.contains("fn cuMemcpyDtoHAsync_v2("));

    let d2h_on_stream = method_body(GPU_IMPL, "copy_d2h_on_stream", "copy_d2h_pair_on_stream");
    assert!(d2h_on_stream.contains("cuMemcpyDtoH_v2("));
    assert!(!d2h_on_stream.contains("cuMemcpyDtoHAsync_v2("));
    assert!(d2h_on_stream.find("cuStreamSynchronize(") < d2h_on_stream.find("cuMemcpyDtoH_v2("));

    let d2h_pair = method_body(
        GPU_IMPL,
        "copy_d2h_pair_on_stream",
        "copy_h2d_group_on_stream",
    );
    assert_eq!(d2h_pair.matches("cuStreamSynchronize(").count(), 1);
    assert_eq!(d2h_pair.matches("cuMemcpyDtoH_v2(").count(), 2);
    assert!(!d2h_pair.contains("cuMemcpyDtoHAsync_v2("));
    assert!(d2h_pair.find("cuStreamSynchronize(") < d2h_pair.find("cuMemcpyDtoH_v2("));

    let h2d_group = method_body(GPU_IMPL, "copy_h2d_group_on_stream", "copy_d2d");
    assert_eq!(h2d_group.matches("cuStreamSynchronize(").count(), 1);
    assert!(h2d_group.contains("cuMemcpyHtoD_v2("));
    assert!(!h2d_group.contains("cuMemcpyHtoDAsync_v2("));

    let pinned_h2d = method_body(GPU_IMPL, "copy_h2d_pinned_async", "copy_d2d_async");
    assert!(pinned_h2d.contains("PinnedHostSlice<'_>"));
    assert!(pinned_h2d.contains("cuMemcpyHtoDAsync_v2("));
}

#[test]
fn microfixture_pins_exact_96_byte_bf16_tensor_and_repeats_d2h() {
    assert!(MICROFIXTURE.contains("const BF16_WORDS: [u16; 48]"));
    assert!(MICROFIXTURE.contains("let mut expected = [0_u8; 96]"));
    assert!(MICROFIXTURE.contains("for iteration in 0..iterations"));
    assert!(MICROFIXTURE.contains("gpu.copy_d2h(device, &mut observed)"));
    assert!(
        MICROFIXTURE.contains("e9eed511c3ae7fc96964eabb421164ad67f0fd45d3df56ef27d6dd820558de98")
    );
}
