// SPDX-License-Identifier: AGPL-3.0-only
use crate::{contract::check_bf16, io, pins};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub fn tensor_span(v: &Value, name: &str, header_len: u64, file_len: u64) -> Result<(u64, usize)> {
    ensure!(
        header_len > 0 && header_len <= io::MAX_FILE as u64,
        "header size bound"
    );
    let t = &v[name];
    ensure!(
        t["dtype"].as_str() == Some("BF16"),
        "fc2 weight must be BF16"
    );
    let shape = t["shape"].as_array().context("weight shape")?;
    ensure!(
        shape.len() == 2 && shape[0].as_u64() == Some(1024) && shape[1].as_u64() == Some(2816),
        "weight shape"
    );
    let offsets = t["data_offsets"].as_array().context("weight offsets")?;
    ensure!(offsets.len() == 2, "weight offsets length");
    let lo = offsets[0].as_u64().context("integer offset start")?;
    let hi = offsets[1].as_u64().context("integer offset end")?;
    ensure!(
        hi.checked_sub(lo) == Some(5_767_168),
        "weight span byte count"
    );
    let start = header_len
        .checked_add(8)
        .and_then(|n| n.checked_add(lo))
        .context("weight start overflow")?;
    let end = start
        .checked_add(5_767_168)
        .context("weight end overflow")?;
    ensure!(end <= file_len, "weight span outside shard");
    Ok((start, 5_767_168))
}
pub fn selected(model: &Path, index: &Value, checkpoint: &Value) -> Result<(Vec<u8>, Value)> {
    let name = index["weight_map"][pins::FC2_WEIGHT]
        .as_str()
        .context("fc2 not indexed")?;
    let shards = checkpoint["complete_shards"]
        .as_array()
        .context("retained shard receipts")?;
    let candidates: Vec<_> = shards
        .iter()
        .filter(|s| s["file"].as_str() == Some(name))
        .collect();
    ensure!(
        candidates.len() == 1 && !name.contains('/') && !name.contains('\\'),
        "not a unique pinned shard"
    );
    let old = candidates[0];
    let path = io::absolute(&model.join(name))?;
    ensure!(path.parent() == Some(model), "shard escapes model root");
    let mut f = File::open(&path)?;
    let meta = f.metadata()?;
    ensure!(
        meta.is_file() && Some(meta.len()) == old["bytes"].as_u64(),
        "shard size changed"
    );
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let hlen = u64::from_le_bytes(len);
    ensure!(
        hlen > 0 && hlen <= io::MAX_FILE as u64 && hlen + 8 <= meta.len(),
        "header bound"
    );
    let mut header = vec![0; hlen as usize];
    f.read_exact(&mut header)?;
    let sha = io::hash(&header)?;
    ensure!(
        Some(sha.as_str()) == old["header_sha256"].as_str(),
        "shard header changed"
    );
    let (start, bytes) = tensor_span(&io::json(&header)?, pins::FC2_WEIGHT, hlen, meta.len())?;
    let mut raw = vec![0; bytes];
    f.seek(SeekFrom::Start(start))?;
    f.read_exact(&mut raw)?;
    check_bf16(&raw)?;
    Ok((
        raw.clone(),
        json!({"tensor":pins::FC2_WEIGHT,"dtype":"BF16","shape":[1024,2816],
        "shard":path,"shard_bytes":meta.len(),"header_bytes":hlen,"header_sha256":sha,
        "payload_offset":start,"payload_bytes":bytes,"payload_sha256":io::hash(&raw)?,
        "full_shard_rehashed":false,"download_etag":old["download_etag"]}),
    ))
}
