// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// GPU-side argmax over BF16 logits.
///
/// Finds the index of the maximum value, writes a single u32 to `out`.
///
/// Kernel: `argmax_bf16(logits, out, n)`
/// Grid: (1, 1, 1)  Block: (1024, 1, 1)
pub fn argmax_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
    vocab_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(out)
        .arg_u32(vocab_size)
        .launch(stream)
}

/// GPU-side argmax over `base_logits + bias`, both BF16 full-vocabulary rows.
///
/// Writes one u32 token ID and uses the lowest ID as the exact tie-break,
/// matching the former left-to-right host scan. Used by DSpark Markov to keep
/// the sequential proposal chain device-resident.
pub fn argmax_add_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    base_logits: DevicePtr,
    bias: DevicePtr,
    out: DevicePtr,
    vocab_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(base_logits)
        .arg_ptr(bias)
        .arg_ptr(out)
        .arg_u32(vocab_size)
        .launch(stream)
}

/// GPU-side argmax + embedding lookup — eliminates D2H sync in MTP propose.
///
/// Reads the argmax result from `argmax_out`, looks up the embedding row
/// from `embed_table`, and writes it to `embed_out`. Also copies the token
/// ID to `token_id_out` for deferred CPU readback.
pub fn embed_from_argmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    argmax_out: DevicePtr,
    embed_table: DevicePtr,
    embed_out: DevicePtr,
    token_id_out: DevicePtr,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    let grid_x = hidden_size.div_ceil(256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(argmax_out)
        .arg_ptr(embed_table)
        .arg_ptr(embed_out)
        .arg_ptr(token_id_out)
        .arg_u32(hidden_size)
        .launch(stream)
}

/// GPU-side top-K over BF16 logits, batched across rows.
///
/// For each of `num_rows` rows, finds the top-`k` BF16 logits and writes
/// the resulting `(token_id, logit_value)` pairs sorted by logit descending.
///
/// Used by the DDTree (M4B v2) propose path to seed branch candidates per
/// MASK-position drafter output. K=8 is the common case; valid calls require
/// `vocab > 0` and `1 <= k <= min(16, vocab)`. Invalid shapes fail before
/// launch. A valid zero-row batch is an explicit no-launch success.
///
/// Output layout (caller-allocated, both buffers row-major):
///   - `top_indices`: `[num_rows, k]` u32  — token IDs
///   - `top_logits` : `[num_rows, k]` f32  — raw BF16-→f32 logit values
///
/// Kernel: `topk_bf16(logits, top_indices, top_logits, num_rows, vocab, k)`
/// Grid: (num_rows, 1, 1)  Block: (1024, 1, 1)
pub fn topk_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    top_indices: DevicePtr,
    top_logits: DevicePtr,
    num_rows: u32,
    vocab: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if vocab == 0 {
        anyhow::bail!("topk_bf16 requires vocab > 0");
    }
    if k == 0 {
        anyhow::bail!("topk_bf16 requires k >= 1");
    }
    if k > 16 {
        anyhow::bail!("topk_bf16 requires k <= 16, got {k}");
    }
    if k > vocab {
        anyhow::bail!("topk_bf16 requires k <= vocab ({vocab}), got {k}");
    }
    if num_rows == 0 {
        return Ok(());
    }

    KernelLaunch::new(gpu, kernel)
        .grid([num_rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(top_indices)
        .arg_ptr(top_logits)
        .arg_u32(num_rows)
        .arg_u32(vocab)
        .arg_u32(k)
        .launch(stream)
}

/// Batched embedding: gather N rows from embedding table in one launch.
///
/// Replaces N individual D2D copies with a single kernel.
/// `token_ids_dev` must point to `[num_tokens]` u32 on device.
///
/// Kernel: `batched_embed(token_ids, embed_table, output, hidden_size)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
pub fn batched_embed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    token_ids_dev: DevicePtr,
    embed_table: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(token_ids_dev)
        .arg_ptr(embed_table)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, KernelHandle};

    use super::argmax_add_bf16;

    #[test]
    fn argmax_add_launches_one_full_vocab_reduction() {
        let gpu = MockGpuBackend::new();
        argmax_add_bf16(
            &gpu,
            KernelHandle(7),
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            DevicePtr(0x3000),
            248_077,
            19,
        )
        .unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }
}

#[cfg(test)]
mod topk_tests {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, KernelHandle};

    use super::topk_bf16;

    const SOURCE: &str = include_str!("sampling.rs");
    const IMMUTABLE_PREFIX_FNV1A64: u64 = 0xf38d_9695_1329_5f41;
    const IMMUTABLE_BATCHED_EMBED_FNV1A64: u64 = 0xa384_8c11_92e6_d947;
    const IMMUTABLE_ARGMAX_ADD_TEST_FNV1A64: u64 = 0x826b_34ca_77d8_48a7;

    fn invoke(gpu: &MockGpuBackend, num_rows: u32, vocab: u32, k: u32) -> anyhow::Result<()> {
        topk_bf16(
            gpu,
            KernelHandle(11),
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            DevicePtr(0x3000),
            num_rows,
            vocab,
            k,
            23,
        )
    }

    fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source
            .find(start)
            .unwrap_or_else(|| panic!("missing section start: {start}"));
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .unwrap_or_else(|| panic!("missing section end: {end}"));
        &source[start_index..end_index]
    }

    fn compact(source: &str) -> String {
        source.chars().filter(|ch| !ch.is_whitespace()).collect()
    }

    fn fnv1a64(source: &str) -> u64 {
        source.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    #[test]
    fn invalid_scalar_shapes_error_before_launch() {
        for (num_rows, vocab, k, message) in [
            (3, 0, 0, "vocab > 0"),
            (3, 16, 0, "k >= 1"),
            (3, 248_077, 17, "k <= 16"),
            (3, 3, 4, "k <= vocab"),
            (0, 0, 1, "vocab > 0"),
        ] {
            let gpu = MockGpuBackend::new();
            let error = invoke(&gpu, num_rows, vocab, k).expect_err("invalid shape must fail");
            assert!(
                error.to_string().contains(message),
                "unexpected error for rows={num_rows}, vocab={vocab}, k={k}: {error:#}"
            );
            assert_eq!(gpu.launch_count(), 0);
        }
    }

    #[test]
    fn valid_zero_rows_is_a_no_launch_success() {
        let gpu = MockGpuBackend::new();
        invoke(&gpu, 0, 248_077, 16).unwrap();
        assert_eq!(gpu.launch_count(), 0);
    }

    #[test]
    fn valid_call_preserves_exact_abi_and_launch_geometry() {
        let gpu = MockGpuBackend::new();
        invoke(&gpu, 3, 248_077, 16).unwrap();
        assert_eq!(gpu.launch_count(), 1);

        let topk = section(
            SOURCE,
            "pub fn topk_bf16(",
            "/// Batched embedding: gather N rows",
        );
        let signature_end = topk
            .find(") -> Result<()> {")
            .map(|index| index + ") -> Result<()> {".len())
            .expect("top-K wrapper signature must remain present");
        assert_eq!(
            compact(&topk[..signature_end]),
            "pubfntopk_bf16(gpu:&dynGpuBackend,kernel:KernelHandle,logits:DevicePtr,top_indices:DevicePtr,top_logits:DevicePtr,num_rows:u32,vocab:u32,k:u32,stream:u64,)->Result<()>{"
        );
        let compact_topk = compact(topk);
        assert!(compact_topk.contains(".grid([num_rows,1,1]).block([1024,1,1])"));
        assert!(compact_topk.contains(
            ".arg_ptr(logits).arg_ptr(top_indices).arg_ptr(top_logits).arg_u32(num_rows).arg_u32(vocab).arg_u32(k).launch(stream)"
        ));
        assert!(!compact_topk.contains("num_rows.max(1)"));
    }

    #[test]
    fn every_preexisting_non_topk_region_is_byte_pinned() {
        let topk_start = SOURCE
            .find("/// GPU-side top-K over BF16 logits")
            .expect("top-K marker must remain present");
        assert_eq!(fnv1a64(&SOURCE[..topk_start]), IMMUTABLE_PREFIX_FNV1A64);

        let batched_embed = section(SOURCE, "/// Batched embedding:", "#[cfg(test)]\nmod tests");
        assert_eq!(fnv1a64(batched_embed), IMMUTABLE_BATCHED_EMBED_FNV1A64);

        let argmax_add_tests = section(
            SOURCE,
            "#[cfg(test)]\nmod tests",
            "#[cfg(test)]\nmod topk_tests",
        );
        assert_eq!(fnv1a64(argmax_add_tests), IMMUTABLE_ARGMAX_ADD_TEST_FNV1A64);
    }
}
