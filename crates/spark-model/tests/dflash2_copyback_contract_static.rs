// SPDX-License-Identifier: AGPL-3.0-only

const FORWARD_LAYER: &str = include_str!("../src/layers/dflash_head/forward_block_layer.rs");
const GPU_BACKEND: &str = include_str!("../../spark-runtime/src/gpu.rs");

fn copy_call_after(marker: &str) -> &str {
    let after_marker = FORWARD_LAYER
        .split_once(marker)
        .unwrap_or_else(|| panic!("missing copy-back marker: {marker}"))
        .1;
    let after_call = after_marker
        .split_once("gpu.copy_d2d_async(")
        .unwrap_or_else(|| panic!("missing copy_d2d_async after: {marker}"))
        .1;
    after_call
        .split_once(")?;")
        .expect("unterminated copy_d2d_async call")
        .0
}

fn assert_copy_direction(marker: &str, destination: &str) {
    let call = copy_call_after(marker);
    let source_at = call
        .find("self.scratch.conv_out,")
        .expect("copy-back must read conv_out");
    let destination_at = call
        .find(destination)
        .expect("copy-back destination missing");
    assert!(
        source_at < destination_at,
        "copy_d2d_async is src,dst: conv_out must precede {destination}"
    );
}

#[test]
fn gpu_backend_copy_abi_is_source_then_destination() {
    let signature = GPU_BACKEND
        .split_once("fn copy_d2d_async(")
        .expect("GpuBackend copy ABI missing")
        .1
        .split_once(") -> Result<()>")
        .expect("GpuBackend copy ABI signature changed")
        .0;
    assert!(signature.find("src: DevicePtr").unwrap() < signature.find("dst: DevicePtr").unwrap());
}

#[test]
fn dflash2_prepare_copies_conv_output_back_to_norm_buffer() {
    assert_copy_direction(
        "// conv_out → norm_buf noise slice (stream-ordered D2D)",
        "self.scratch.norm_buf.offset(noise_byte_offset),",
    );
}

#[test]
fn dflash2_finish_copies_conv_output_back_to_stream_accumulator() {
    assert_copy_direction(
        "// conv_out → stream_acc noise slice",
        "self.scratch.stream_acc.offset(noise_byte_offset),",
    );
}
