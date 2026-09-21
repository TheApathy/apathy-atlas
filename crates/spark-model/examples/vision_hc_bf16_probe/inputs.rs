// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use std::io::Read;
use std::path::Path;

pub const H: usize = 4096;
pub const HC: usize = 4;

pub struct Inputs {
    pub label: String,
    pub rows: usize,
    pub block: Vec<u8>,
    pub residual: Vec<u8>,
    pub post: Vec<u8>,
    pub comb: Vec<u8>,
    pub captured_output: Option<Vec<u8>>,
}

/// Independent integer RNE oracle; used only on finite baseline output.
pub fn rounded_bits(bits: u32) -> u32 {
    assert_ne!(bits & 0x7f80_0000, 0x7f80_0000, "nonfinite baseline");
    bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000
}

const BOUNDARIES: &[u32] = &[
    0, 0x80000000, 0x3f800000, 0x3f807fff, 0x3f808000, 0x3f808001, 0x3f817fff, 0x3f818000,
    0x3f818001, 0x00000001, 0x00007fff, 0x00008000, 0x00008001, 0x00018000, 0x007fffff, 0x00800000,
    0x7f7f0000, 0x7f7f7fff, 0x7f7f8000, 0x7f7f8001, 0xbf808000, 0xbf818000, 0x80008000, 0xff7f8000,
];

fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}

pub fn synthetic(rows: usize, boundary: bool) -> Inputs {
    assert!([1, 12].contains(&rows));
    let block = (0..rows * H)
        .flat_map(|i| {
            half::bf16::from_f32(if boundary {
                0.0
            } else {
                (i % 31) as f32 / 8.0 - 2.0
            })
            .to_bits()
            .to_le_bytes()
        })
        .collect();
    let residual = f32_bytes((0..rows * HC * H).map(|i| {
        if boundary {
            f32::from_bits(BOUNDARIES[i % BOUNDARIES.len()])
        } else {
            (i % 257) as f32 / 71.0 - 1.5
        }
    }));
    let post = f32_bytes((0..rows * HC).map(|i| {
        if boundary {
            0.0
        } else {
            (i % 7) as f32 / 5.0 - 0.4
        }
    }));
    let comb = f32_bytes((0..rows * HC * HC).map(|i| {
        if boundary {
            if (i / HC) % HC == i % HC { 1.0 } else { 0.0 }
        } else {
            (i % 11) as f32 / 13.0 - 0.3
        }
    }));
    Inputs {
        label: format!(
            "synthetic-{}",
            if boundary { "RNE-boundaries" } else { "mixed" }
        ),
        rows,
        block,
        residual,
        post,
        comb,
        captured_output: None,
    }
}

fn read_file(dir: &Path, name: &str, bytes: usize) -> Result<Vec<u8>> {
    let path = dir.join(name);
    let meta = std::fs::symlink_metadata(&path)?;
    ensure!(
        meta.is_file() && meta.len() == bytes as u64,
        "bad capture file {name}"
    );
    let mut data = Vec::with_capacity(bytes);
    std::fs::File::open(path)?
        .take((bytes + 1) as u64)
        .read_to_end(&mut data)?;
    ensure!(data.len() == bytes, "capture file changed: {name}");
    Ok(data)
}

/// Read-only bounded V14 capture control. Does not load checkpoint weights.
pub fn captured(dir: &Path) -> Result<Vec<Inputs>> {
    ensure!(
        dir.is_absolute() && std::fs::symlink_metadata(dir)?.is_dir(),
        "capture must be an absolute real directory"
    );
    let meta = std::fs::symlink_metadata(dir.join("manifest.json"))?;
    ensure!(
        meta.is_file() && meta.len() <= 65536,
        "bad capture manifest file"
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&read_file(dir, "manifest.json", meta.len() as usize)?)?;
    ensure!(
        manifest["schema"] == "atlas-vision-l0-dump-v1"
            && manifest["status"] == "COMPLETE"
            && manifest["byte_order"] == "little"
            && manifest["layer_index"] == 0
            && manifest["token_count"] == 12
            && manifest["hidden_size"] == H
            && manifest["hc_mult"] == HC
            && manifest["vocab_size"] == 129280
            && manifest["payload_bytes"] == 3049392,
        "wrong capture contract"
    );
    let ids = read_file(dir, "token_ids.bin", 48)?;
    let actual: Vec<u32> = ids
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        actual
            == [
                0, 128803, 19905, 418, 9045, 20370, 305, 5760, 3006, 16, 128804, 128822
            ]
            && manifest["token_ids"] == serde_json::json!(actual)
            && manifest["token_ids_file"] == "token_ids.bin"
            && manifest["token_ids_bytes"] == 48,
        "wrong capture token IDs"
    );
    let stages = [
        ("embed", "BF16", vec![12, H]),
        ("hc_expanded", "F32", vec![12, HC, H]),
        ("hc_pre_attn", "BF16", vec![12, H]),
        ("post_attn", "F32", vec![12, HC]),
        ("comb_attn", "F32", vec![12, HC, HC]),
        ("norm_attn", "BF16", vec![12, H]),
        ("attention_out", "BF16", vec![12, H]),
        ("hc_post_attn", "F32", vec![12, HC, H]),
        ("hc_pre_ffn", "BF16", vec![12, H]),
        ("post_ffn", "F32", vec![12, HC]),
        ("comb_ffn", "F32", vec![12, HC, HC]),
        ("norm_ffn", "BF16", vec![12, H]),
        ("moe_out", "BF16", vec![12, H]),
        ("hc_post_ffn", "F32", vec![12, HC, H]),
    ];
    let tensors = manifest["tensors"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("missing tensors"))?;
    ensure!(tensors.len() == stages.len(), "unexpected capture stages");
    let mut payloads = std::collections::BTreeMap::new();
    for (name, dtype, shape) in stages {
        let bytes = shape.iter().product::<usize>() * if dtype == "BF16" { 2 } else { 4 };
        let entry = tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing stage {name}"))?;
        ensure!(
            entry["file"] == format!("{name}.bin")
                && entry["dtype"] == dtype
                && entry["shape"] == serde_json::json!(shape)
                && entry["bytes"] == bytes,
            "bad stage {name}"
        );
        payloads.insert(name, read_file(dir, &format!("{name}.bin"), bytes)?);
    }
    let mut out = Vec::new();
    for rows in [1, 12] {
        for (label, block, residual, post, comb, output) in [
            (
                "capture-attention",
                "attention_out",
                "hc_expanded",
                "post_attn",
                "comb_attn",
                "hc_post_attn",
            ),
            (
                "capture-ffn",
                "moe_out",
                "hc_post_attn",
                "post_ffn",
                "comb_ffn",
                "hc_post_ffn",
            ),
        ] {
            out.push(Inputs {
                label: label.into(),
                rows,
                block: payloads[block][..rows * H * 2].to_vec(),
                residual: payloads[residual][..rows * HC * H * 4].to_vec(),
                post: payloads[post][..rows * HC * 4].to_vec(),
                comb: payloads[comb][..rows * HC * HC * 4].to_vec(),
                captured_output: Some(payloads[output][..rows * HC * H * 4].to_vec()),
            });
        }
    }
    Ok(out)
}
