// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_ATTN_CORE_DUMP=<dir>` (diagnostic, default off): for the FIRST
//! full-attention layer only, write the chunk's contiguous Q/K/V inputs and
//! the attention-core output (before the sigmoid gate and O projection) as raw
//! BF16, one file set per prefill chunk, named by the chunk's absolute start.
//! Lets an offline fp32 reference check the in-chunk and paged-cache paths
//! against the same operands.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;

fn dump_dir() -> Option<&'static str> {
    static DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var("ATLAS_ATTN_CORE_DUMP").ok().filter(|d| !d.is_empty()))
        .as_deref()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dump_attention_core(
    ctx: &ForwardContext,
    attn_layer_idx: usize,
    path: &str,
    seq_len_start: usize,
    num_tokens: usize,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    attn_out: DevicePtr,
    q_dim: usize,
    kv_dim: usize,
    stream: u64,
) -> Result<()> {
    let Some(dir) = dump_dir() else {
        return Ok(());
    };
    if attn_layer_idx != 0 {
        return Ok(());
    }
    ctx.gpu.synchronize(stream)?;
    std::fs::create_dir_all(dir)?;
    for (name, ptr, width) in [
        ("q", q, q_dim),
        ("k", k, kv_dim),
        ("v", v, kv_dim),
        ("o", attn_out, q_dim),
    ] {
        let mut buf = vec![0u8; num_tokens * width * 2];
        ctx.gpu.copy_d2h(ptr, &mut buf)?;
        let file = std::path::Path::new(dir)
            .join(format!("{path}_start{seq_len_start}_n{num_tokens}_{name}.bf16"));
        std::fs::write(file, &buf)?;
    }
    tracing::info!(
        "ATLAS_ATTN_CORE_DUMP: {path} start={seq_len_start} n={num_tokens} q_dim={q_dim} kv_dim={kv_dim}"
    );
    Ok(())
}
