// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 engram tier: sparse row gather from the two 95 GB n-gram tables.
//!
//! The engram tables are `engram_num_embeddings = [384_006_168, 384_016_682]` rows
//! of 8 heads x 256 fp8 values, living in `model-000{47,48}-of-00048.safetensors`
//! at ~95 GB each. 190 GB against 119.7 GB of unified memory: they can never be
//! resident, so every token's rows come off NVMe.
//!
//! Shape of the problem, and why this is NOT `expert_tier`:
//!   * A record here is 264 B (256 B of fp8 weights + 8 B of ue8m0 scales), not an
//!     expert's 14.45 MB. The per-record residency bookkeeping would outweigh the
//!     payload.
//!   * Rows sit at `r * 256`, so 15 of every 16 are *not* 4 KiB-aligned and
//!     `O_DIRECT` — `UmaArenaTier`'s whole reason to exist — cannot be used.
//!     Aligning by hand just re-implements the page cache without the cache.
//!   * There is no resident tier to demote to, so there is no hot/cold state
//!     machine to run.
//!
//! What DOES carry over is the lesson from the Python reference: batch the reads
//! into few large calls and never loop per row across a lock boundary. Measured on
//! this box (cold, random rows, `/proc/diskstats` cross-checked at 4684 device
//! bytes per requested row, i.e. one 4 KiB page per 256 B row):
//!
//! ```text
//!   prefill, 2048-token chunk (98,304 rows), 256 threads: 104.3 ms -> 19,637 tok/s cap
//!   decode,  1 token (48 rows),               32 threads:   1.21 ms ->    827 tok/s cap
//!   decode,  1 token,                          1 thread:   20.3 ms ->     49 tok/s cap
//! ```
//!
//! So the thread count is the entire difference between "free" and "the
//! bottleneck", and engram I/O costs a ~30 ms decode step about 4%.

use std::collections::HashMap;
use std::fs::File;
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// 256 fp8_e4m3 weight bytes followed by 8 ue8m0 scale bytes, exactly the layout
/// the Python reference hands to the dequantizer.
pub const ENGRAM_ROW_BYTES: usize = 264;
/// Values per row (`engram_head_dim`).
pub const ENGRAM_HEAD_DIM: usize = 256;
/// Values covered by one ue8m0 scale.
pub const ENGRAM_SCALE_GROUP: usize = 32;

/// Default reader threads. This is the single most consequential knob in the
/// module: at the decode point (48 rows) one thread costs 20.3 ms against 1.21 ms
/// at 32, a 17x regression, and it is the only way to make engram a bottleneck.
/// 128 is a good compromise across the decode and prefill batch sizes — the reads
/// block in the kernel, so oversubscribing cores is the point.
pub const DEFAULT_ENGRAM_THREADS: usize = 128;

const FADV_RANDOM: libc::c_int = 1;

/// One engram layer's table: a whole safetensors shard holding exactly the
/// `embed.weight` / `embed.scale` pair for that layer.
pub struct EngramShard {
    path: PathBuf,
    file: File,
    w_off: i64,
    s_off: i64,
    n_rows: u64,
}

impl EngramShard {
    /// Parse the shard header and locate the two tensors for `layer`.
    pub fn open(path: &Path, layer: u32) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open engram shard {}", path.display()))?;
        let (w_off, s_off, n_rows) = read_header(&file, path, layer)?;
        // Random access: kill readahead, which would otherwise pull 128 KiB per
        // 256 B row and turn a 4 KiB-per-row workload into a bandwidth problem.
        let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, FADV_RANDOM) };
        if rc != 0 {
            // Advisory only — a failure costs speed, not correctness.
            tracing::warn!(path = %path.display(), rc, "posix_fadvise(RANDOM) failed on engram shard");
        }
        Ok(Self { path: path.to_path_buf(), file, w_off, s_off, n_rows })
    }

    pub fn n_rows(&self) -> u64 {
        self.n_rows
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

/// Locate `layers.{layer}.engram.embed.{weight,scale}` and return
/// `(weight byte offset, scale byte offset, rows)`.
fn read_header(file: &File, path: &Path, layer: u32) -> Result<(i64, i64, u64)> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = file.try_clone().context("clone engram shard handle")?;
    f.seek(SeekFrom::Start(0))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).context("read safetensors header length")?;
    let n = u64::from_le_bytes(len) as usize;
    if n == 0 || n > 64 << 20 {
        bail!("{}: implausible safetensors header length {n}", path.display());
    }
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).context("read safetensors header")?;
    let hdr: serde_json::Value = serde_json::from_slice(&buf).context("parse safetensors header")?;
    let base = (8 + n) as i64;

    let get = |suffix: &str| -> Result<(i64, Vec<u64>)> {
        let key = format!("layers.{layer}.engram.embed.{suffix}");
        let e = hdr.get(&key).with_context(|| format!("{}: missing tensor {key}", path.display()))?;
        let off = e["data_offsets"][0]
            .as_i64()
            .with_context(|| format!("{key}: no data_offsets"))?;
        let shape: Vec<u64> = e["shape"]
            .as_array()
            .with_context(|| format!("{key}: no shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0))
            .collect();
        Ok((base + off, shape))
    };

    let (w_off, w_shape) = get("weight")?;
    let (s_off, s_shape) = get("scale")?;
    if w_shape.len() != 2 || w_shape[1] as usize != ENGRAM_HEAD_DIM {
        bail!("{}: engram weight shape {w_shape:?}, expected [rows, {ENGRAM_HEAD_DIM}]", path.display());
    }
    let groups = (ENGRAM_HEAD_DIM / ENGRAM_SCALE_GROUP) as u64;
    if s_shape.len() != 2 || s_shape[1] != groups {
        bail!("{}: engram scale shape {s_shape:?}, expected [rows, {groups}]", path.display());
    }
    if w_shape[0] != s_shape[0] {
        bail!("{}: engram weight rows {} != scale rows {}", path.display(), w_shape[0], s_shape[0]);
    }
    Ok((w_off, s_off, w_shape[0]))
}

/// The engram row-gather tier: one shard per engram layer, read with a batched
/// thread fan-out.
pub struct EngramTier {
    shards: Vec<EngramShard>,
    threads: usize,
}

impl EngramTier {
    /// `shards[i]` must correspond to `layer_ids[i]`, in the model's order.
    ///
    /// Prefer [`DEFAULT_ENGRAM_THREADS`] unless you have measured something better
    /// — see the note on that constant before passing a small number.
    pub fn new(shards: Vec<EngramShard>, threads: usize) -> Result<Self> {
        if shards.is_empty() {
            bail!("EngramTier needs at least one shard");
        }
        if threads < 8 {
            // Not an error — a caller may genuinely want a serial read — but this
            // is worth a line in the log, because it is a 17x cliff and it shows
            // up as "the model is slow", never as "the reader is misconfigured".
            tracing::warn!(
                threads,
                "engram tier configured with < 8 reader threads; expect a large \
                 latency penalty (1 thread measures 20.3 ms/token vs 1.21 ms at 32)"
            );
        }
        Ok(Self { shards, threads: threads.clamp(1, 512) })
    }

    /// Open from a model directory and the safetensors weight map.
    pub fn open(
        model_dir: &Path,
        layer_ids: &[u32],
        weight_map: &HashMap<String, String>,
        threads: usize,
    ) -> Result<Self> {
        let mut shards = Vec::with_capacity(layer_ids.len());
        for &layer in layer_ids {
            let key = format!("layers.{layer}.engram.embed.weight");
            let file = weight_map
                .get(&key)
                .with_context(|| format!("weight map has no entry for {key}"))?;
            shards.push(EngramShard::open(&model_dir.join(file), layer)?);
        }
        Self::new(shards, threads)
    }

    pub fn n_layers(&self) -> usize {
        self.shards.len()
    }

    pub fn shard(&self, li: usize) -> Result<&EngramShard> {
        self.shards.get(li).with_context(|| format!("engram layer index {li} out of range"))
    }

    /// Gather `row_ids` for engram layer index `li` into `[row_ids.len() * 264]` bytes.
    ///
    /// One batched call: the fan-out is over dedicated threads, so a caller holding
    /// any lock is never serialised against the per-row loop. Duplicate ids are
    /// read twice — call [`EngramTier::gather_dedup`] when duplicates are likely.
    pub fn gather(&self, li: usize, row_ids: &[u64]) -> Result<Vec<u8>> {
        let shard = self.shard(li)?;
        let mut out = vec![0u8; row_ids.len() * ENGRAM_ROW_BYTES];
        self.gather_into(shard, row_ids, &mut out)?;
        Ok(out)
    }

    /// Gather only the distinct ids, plus the inverse index that rebuilds the
    /// original order (`unique[inverse[i]]` is the row for `row_ids[i]`).
    ///
    /// This mirrors the reference's `np.unique(..., return_inverse=True)`. Within a
    /// prefill chunk exact n-gram repeats are common, and a duplicate costs a full
    /// 4 KiB page fault, so the dedup is worth its hash map.
    pub fn gather_dedup(&self, li: usize, row_ids: &[u64]) -> Result<(Vec<u8>, Vec<u32>)> {
        let mut first: HashMap<u64, u32> = HashMap::with_capacity(row_ids.len());
        let mut unique: Vec<u64> = Vec::with_capacity(row_ids.len());
        let mut inverse: Vec<u32> = Vec::with_capacity(row_ids.len());
        for &r in row_ids {
            let slot = *first.entry(r).or_insert_with(|| {
                unique.push(r);
                (unique.len() - 1) as u32
            });
            inverse.push(slot);
        }
        Ok((self.gather(li, &unique)?, inverse))
    }

    fn gather_into(&self, shard: &EngramShard, row_ids: &[u64], out: &mut [u8]) -> Result<()> {
        let n = row_ids.len();
        if n == 0 {
            return Ok(());
        }
        if out.len() != n * ENGRAM_ROW_BYTES {
            bail!("engram gather: output is {} bytes, need {}", out.len(), n * ENGRAM_ROW_BYTES);
        }
        for (i, &r) in row_ids.iter().enumerate() {
            if r >= shard.n_rows {
                bail!("engram row {r} out of range (index {i}, table has {} rows)", shard.n_rows);
            }
        }

        let nthreads = self.threads.clamp(1, n.max(1));
        let per = n.div_ceil(nthreads);
        let fd = shard.fd();
        let (w_off, s_off) = (shard.w_off, shard.s_off);

        // Hand each thread a disjoint, contiguous slice of the output. No locking,
        // no shared cursor: the whole point is that a thread blocked in `pread`
        // blocks nothing else.
        let mut err: Option<String> = None;
        std::thread::scope(|sc| {
            let mut handles = Vec::with_capacity(nthreads);
            for (t, chunk) in out.chunks_mut(per * ENGRAM_ROW_BYTES).enumerate() {
                let ids = &row_ids[t * per..(t * per + chunk.len() / ENGRAM_ROW_BYTES)];
                handles.push(sc.spawn(move || read_chunk(fd, w_off, s_off, ids, chunk)));
            }
            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        err.get_or_insert(e);
                    }
                    Err(_) => {
                        err.get_or_insert_with(|| "engram reader thread panicked".to_string());
                    }
                }
            }
        });
        match err {
            Some(e) => bail!("{}: {e}", shard.path.display()),
            None => Ok(()),
        }
    }
}

/// One thread's share: `pread` each row's 256 weight bytes and 8 scale bytes.
///
/// The two regions are ~98 GB apart in the file, so they cannot be coalesced into
/// one call. Short reads are an error, never silently zero-filled — a zero row
/// dequantizes to a plausible-looking zero vector and would be invisible
/// downstream.
fn read_chunk(fd: RawFd, w_off: i64, s_off: i64, ids: &[u64], out: &mut [u8]) -> Result<(), String> {
    for (i, &r) in ids.iter().enumerate() {
        let dst = &mut out[i * ENGRAM_ROW_BYTES..(i + 1) * ENGRAM_ROW_BYTES];
        let r = r as i64;
        pread_exact(fd, &mut dst[..ENGRAM_HEAD_DIM], w_off + r * ENGRAM_HEAD_DIM as i64)
            .map_err(|e| format!("weight row {r}: {e}"))?;
        pread_exact(fd, &mut dst[ENGRAM_HEAD_DIM..], s_off + r * 8)
            .map_err(|e| format!("scale row {r}: {e}"))?;
    }
    Ok(())
}

fn pread_exact(fd: RawFd, buf: &mut [u8], off: i64) -> Result<(), String> {
    let mut done = 0usize;
    while done < buf.len() {
        // SAFETY: `fd` is a live read-only handle owned by the shard for the whole
        // scope of the gather; the write stays inside `buf`.
        let n = unsafe {
            libc::pread(
                fd,
                buf[done..].as_mut_ptr() as *mut libc::c_void,
                buf.len() - done,
                off + done as i64,
            )
        };
        match n {
            -1 => {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e.to_string());
            }
            0 => return Err(format!("unexpected EOF at offset {}", off + done as i64)),
            n => done += n as usize,
        }
    }
    Ok(())
}

/// fp8_e4m3fn -> f32, as a 256-entry table built once.
///
/// e4m3**fn**: no infinities, and the single NaN encoding is exponent 15 with
/// mantissa 7. Everything else is an ordinary value, so the max magnitude is 448.
fn e4m3_table() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for (b, slot) in t.iter_mut().enumerate() {
        let b = b as u8;
        let exp = ((b >> 3) & 0x0F) as i32;
        let man = (b & 0x07) as f32;
        let mag = if exp == 0 {
            // subnormal: mantissa * 2^-9
            man * (1.0 / 512.0)
        } else if exp == 0x0F && man == 7.0 {
            f32::NAN
        } else {
            (1.0 + man / 8.0) * exp2i(exp - 7)
        };
        *slot = if b & 0x80 != 0 { -mag } else { mag };
    }
    t
}

/// Dequantize one 264-byte row into 256 f32 values.
///
/// `value[j] = fp8_e4m3(w[j]) * 2^(scale[j / 32] - 127)`. The scale is a power of
/// two and the mantissa is 3 bits, so this is exact in f32.
pub fn dequant_row(row: &[u8], out: &mut [f32]) -> Result<()> {
    if row.len() != ENGRAM_ROW_BYTES || out.len() != ENGRAM_HEAD_DIM {
        bail!("dequant_row: row {} bytes, out {} values", row.len(), out.len());
    }
    let table = e4m3_table();
    for g in 0..ENGRAM_HEAD_DIM / ENGRAM_SCALE_GROUP {
        let scale = exp2i(row[ENGRAM_HEAD_DIM + g] as i32 - 127);
        for j in g * ENGRAM_SCALE_GROUP..(g + 1) * ENGRAM_SCALE_GROUP {
            out[j] = table[row[j] as usize] * scale;
        }
    }
    Ok(())
}

/// `2^e` without touching libm, for the full ue8m0 range (e in -127..=128).
fn exp2i(e: i32) -> f32 {
    if (-126..=127).contains(&e) {
        f32::from_bits(((e + 127) as u32) << 23)
    } else if e < -126 {
        // subnormal tail: step down from the smallest normal
        let mut v = f32::from_bits(1u32 << 23);
        for _ in 0..(-126 - e).min(32) {
            v *= 0.5;
        }
        v
    } else {
        f32::INFINITY
    }
}
