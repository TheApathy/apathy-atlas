// SPDX-License-Identifier: AGPL-3.0-only
//! Read only one native BF16 tensor, not the 7.6 GB shard or full checkpoint.
use crate::contract::{self, HEADER_CAP, WEIGHT};
use anyhow::{Context, Result, ensure};
use deepseek_vision_p1::{contract::check_bf16, io};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
    path::Path,
};

pub fn selected(model: &Path, index: &Value, checkpoint: &Value) -> Result<(Vec<u8>, Value)> {
    let name = index["weight_map"][WEIGHT]
        .as_str()
        .context("fc1 not indexed")?;
    ensure!(
        name == "model-00010-of-00010.safetensors",
        "unexpected fc1 shard"
    );
    let shards = checkpoint["complete_shards"]
        .as_array()
        .context("shard receipts")?;
    ensure!(shards.len() == 10, "checkpoint shard count");
    let matches: Vec<_> = shards.iter().filter(|s| s["file"] == name).collect();
    ensure!(matches.len() == 1, "shard receipt not unique");
    let old = matches[0];
    let path = model.join(name);
    ensure!(
        !fs::symlink_metadata(&path)?.file_type().is_symlink(),
        "symlink shard rejected"
    );
    ensure!(
        io::absolute(&path)? == path && path.parent() == Some(model),
        "noncanonical shard"
    );
    let mut f = File::open(&path)?;
    let before = f.metadata()?;
    ensure!(
        before.is_file()
            && before.len() == 7_602_973_028
            && old["bytes"].as_u64() == Some(before.len()),
        "shard byte identity"
    );
    let mut prefix = [0; 8];
    f.read_exact(&mut prefix)?;
    let hlen = u64::from_le_bytes(prefix);
    ensure!(
        hlen > 0 && hlen <= HEADER_CAP && hlen + 8 <= before.len(),
        "header size bound"
    );
    let mut header = vec![0; hlen as usize];
    f.read_exact(&mut header)?;
    let header_sha = io::hash(&header)?;
    ensure!(
        header_sha == "10abb63d540ae5529ad51b2d3f87697b8e0977603c270825239542339657e0e2"
            && old["header_sha256"] == header_sha,
        "selected shard header pin"
    );
    let (offset, bytes) = contract::fc1_weight_span(&io::json(&header)?, hlen, before.len())?;
    let mut raw = vec![0; bytes];
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(&mut raw)?;
    check_bf16(&raw)?;
    let after = f.metadata()?;
    let named = fs::symlink_metadata(&path)?;
    let identity = |m: &fs::Metadata| {
        (
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        )
    };
    ensure!(
        identity(&before) == identity(&after)
            && identity(&after) == identity(&named)
            && named.is_file(),
        "selected shard changed during read"
    );
    let receipt = json!({"tensor":WEIGHT,"dtype":"BF16","shape":[5632,1024],"shard":path,
        "shard_bytes":before.len(),"header_bytes":hlen,"header_sha256":header_sha,
        "payload_offset":offset,"payload_bytes":bytes,"payload_sha256":io::hash(&raw)?,
        "full_shard_rehashed":false,"download_etag":old["download_etag"]});
    Ok((raw, receipt))
}
