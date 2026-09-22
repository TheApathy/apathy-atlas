// SPDX-License-Identifier: AGPL-3.0-only

use super::super::super::GgmlType;
use super::{Schema, add};

pub(super) fn add_mhc(schema: &mut Schema, layer: usize) {
    for branch in ["attn", "ffn"] {
        add(
            schema,
            format!("blk.{layer}.hc_{branch}_fn.weight"),
            &[16384, 24],
            GgmlType::Q8_0,
        );
        add(
            schema,
            format!("blk.{layer}.hc_{branch}_base.weight"),
            &[24],
            GgmlType::F32,
        );
        add(
            schema,
            format!("blk.{layer}.hc_{branch}_scale.weight"),
            &[3],
            GgmlType::F32,
        );
    }
}

pub(super) fn add_kda(schema: &mut Schema, layer: usize) {
    for projection in ["q", "k", "v"] {
        add(
            schema,
            format!("blk.{layer}.attn_{projection}.weight"),
            &[4096, 8192],
            GgmlType::Q6_K,
        );
        add(
            schema,
            format!("blk.{layer}.ssm_conv1d_{projection}.weight"),
            &[4, 1, 8192],
            GgmlType::F32,
        );
    }
    add(
        schema,
        format!("blk.{layer}.attn_output.weight"),
        &[8192, 4096],
        GgmlType::Q6_K,
    );
    add(schema, format!("blk.{layer}.ssm_a"), &[64], GgmlType::F32);
    add(
        schema,
        format!("blk.{layer}.ssm_beta.weight"),
        &[4096, 64],
        GgmlType::Q8_0,
    );
    add(
        schema,
        format!("blk.{layer}.ssm_dt.bias"),
        &[8192],
        GgmlType::F32,
    );
    for projection in ["f_a", "g_a"] {
        add(
            schema,
            format!("blk.{layer}.ssm_{projection}.weight"),
            &[4096, 128],
            GgmlType::Q8_0,
        );
    }
    for projection in ["f_b", "g_b"] {
        add(
            schema,
            format!("blk.{layer}.ssm_{projection}.weight"),
            &[128, 8192],
            GgmlType::Q8_0,
        );
    }
    add(
        schema,
        format!("blk.{layer}.ssm_norm.weight"),
        &[128],
        GgmlType::F32,
    );
}

pub(super) fn add_dsa(schema: &mut Schema, layer: usize) {
    add(
        schema,
        format!("blk.{layer}.attn_k_b.weight"),
        &[256, 512, 64],
        GgmlType::Q8_0,
    );
    add(
        schema,
        format!("blk.{layer}.attn_v_b.weight"),
        &[512, 256, 64],
        GgmlType::Q8_0,
    );
    add(
        schema,
        format!("blk.{layer}.attn_kv_a_mqa.weight"),
        &[4096, 512],
        GgmlType::Q8_0,
    );
    add(
        schema,
        format!("blk.{layer}.attn_kv_a_norm.weight"),
        &[512],
        GgmlType::F32,
    );
    let edge = if layer == 11 {
        GgmlType::Q8_0
    } else {
        GgmlType::Q6_K
    };
    add(
        schema,
        format!("blk.{layer}.attn_q_a.weight"),
        &[4096, 1536],
        edge,
    );
    add(
        schema,
        format!("blk.{layer}.attn_q_a_norm.weight"),
        &[1536],
        GgmlType::F32,
    );
    add(
        schema,
        format!("blk.{layer}.attn_q_b.weight"),
        &[1536, 16384],
        GgmlType::Q8_0,
    );
    add(
        schema,
        format!("blk.{layer}.attn_output.weight"),
        &[16384, 4096],
        edge,
    );
    add(
        schema,
        format!("blk.{layer}.indexer.attn_k.weight"),
        &[4096, 128],
        GgmlType::Q8_0,
    );
    add(
        schema,
        format!("blk.{layer}.indexer.attn_q_b.weight"),
        &[1536, 4096],
        GgmlType::Q8_0,
    );
    for suffix in ["weight", "bias"] {
        add(
            schema,
            format!("blk.{layer}.indexer.k_norm.{suffix}"),
            &[128],
            GgmlType::F32,
        );
    }
    add(
        schema,
        format!("blk.{layer}.indexer.proj.weight"),
        &[4096, 32],
        GgmlType::F32,
    );
    add(
        schema,
        format!("blk.{layer}.indexer_compressor_ape.weight"),
        &[128, 4],
        GgmlType::F32,
    );
    add(
        schema,
        format!("blk.{layer}.indexer_compressor_gate.weight"),
        &[4096, 128],
        GgmlType::Q8_0,
    );
}
