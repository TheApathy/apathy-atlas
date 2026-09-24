// SPDX-License-Identifier: AGPL-3.0-only

//! On-disk checkpoint format.
//!
//! ```text
//! [header, padded to ALIGN]
//!   magic "ATLASCTX" | version u32 | header_len u32 | model_key [32]
//!   n_meta u32     { name_len u16, name, value u64 }*
//!   n_sections u32 { name_len u16, name, offset u64, len u64,
//!                    piece u64, n_pieces u32, crc32 u32 * n_pieces }*
//!   header_crc u32   (crc32 of bytes [0, header_len))
//! [section 0, padded to ALIGN] [section 1, padded] ...
//! ```
//!
//! Token ids are the `tokens` section (u32 LE). Every section carries one
//! CRC32 per [`PIECE`] bytes so a flipped byte anywhere is rejected, and the
//! pieces are verified in parallel on load. The model key binds the file to
//! one (engine, model, recipe) identity; a mismatch is a clean miss.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use super::aligned::{ALIGN, AlignedBuf, round_up};

pub const MAGIC: &[u8; 8] = b"ATLASCTX";
/// Bump on any layout or semantic change; old files become misses.
pub const FORMAT_VERSION: u32 = 1;
/// Checksum granule. Also the unit of parallel verification.
pub const PIECE: usize = 64 << 20;
const TOKENS: &str = "tokens";
const MAX_HEADER: usize = 16 << 20;

/// One named blob of model state.
pub struct CtxSection {
    pub name: String,
    pub data: AlignedBuf,
}

/// Everything needed to resume a sequence at `tokens.len()`.
pub struct CtxSnapshot {
    pub tokens: Vec<u32>,
    pub meta: Vec<(String, u64)>,
    pub sections: Vec<CtxSection>,
}

impl CtxSnapshot {
    pub fn meta(&self, name: &str) -> Option<u64> {
        self.meta.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    /// Required metadata value, as a readable error when absent.
    pub fn meta_req(&self, name: &str) -> Result<u64> {
        self.meta(name)
            .with_context(|| format!("ctx checkpoint is missing meta '{name}'"))
    }

    pub fn section(&self, name: &str) -> Option<&AlignedBuf> {
        self.sections
            .iter()
            .find(|s| s.name == name)
            .map(|s| &s.data)
    }

    /// Section with an exact expected length.
    pub fn section_exact(&self, name: &str, len: usize) -> Result<&AlignedBuf> {
        let s = self
            .section(name)
            .with_context(|| format!("ctx checkpoint is missing section '{name}'"))?;
        ensure!(
            s.len() == len,
            "ctx checkpoint section '{name}' is {} bytes, expected {len}",
            s.len()
        );
        Ok(s)
    }

    /// Bytes of model state (excluding tokens and header).
    pub fn state_bytes(&self) -> usize {
        self.sections.iter().map(|s| s.data.len()).sum()
    }
}

fn crcs(data: &[u8]) -> Vec<u32> {
    let pieces: Vec<&[u8]> = data.chunks(PIECE).collect();
    let mut out = vec![0u32; pieces.len()];
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8);
    let per = pieces.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (outs, ins) in out.chunks_mut(per).zip(pieces.chunks(per)) {
            s.spawn(move || {
                for (o, p) in outs.iter_mut().zip(ins) {
                    *o = crc32fast::hash(p);
                }
            });
        }
    });
    out
}

fn put_name(h: &mut Vec<u8>, name: &str) -> Result<()> {
    let len = u16::try_from(name.len()).context("ctx section name too long")?;
    h.extend_from_slice(&len.to_le_bytes());
    h.extend_from_slice(name.as_bytes());
    Ok(())
}

fn tokens_bytes(tokens: &[u32]) -> Result<AlignedBuf> {
    let mut buf = AlignedBuf::zeroed(tokens.len() * 4)?;
    for (dst, t) in buf.as_mut_slice().chunks_exact_mut(4).zip(tokens) {
        dst.copy_from_slice(&t.to_le_bytes());
    }
    Ok(buf)
}

fn build_header(snap: &CtxSnapshot, model_key: &[u8; 32], tokens: &AlignedBuf) -> Result<Vec<u8>> {
    let all: Vec<(&str, &AlignedBuf)> = std::iter::once((TOKENS, tokens))
        .chain(snap.sections.iter().map(|s| (s.name.as_str(), &s.data)))
        .collect();
    let mut body = Vec::new();
    body.extend_from_slice(&(snap.meta.len() as u32).to_le_bytes());
    for (name, v) in &snap.meta {
        put_name(&mut body, name)?;
        body.extend_from_slice(&v.to_le_bytes());
    }
    body.extend_from_slice(&(all.len() as u32).to_le_bytes());
    // Offsets depend on the header size; entries are fixed-size per name, so
    // compute the header length first, then fill offsets.
    let entry_len: usize = all
        .iter()
        .map(|(n, d)| 2 + n.len() + 8 + 8 + 8 + 4 + 4 * d.len().div_ceil(PIECE))
        .sum();
    let header_len = 8 + 4 + 4 + 32 + body.len() + entry_len;
    let mut offset = round_up(header_len + 4) as u64;
    for (name, data) in &all {
        let c = crcs(data.as_slice());
        put_name(&mut body, name)?;
        body.extend_from_slice(&offset.to_le_bytes());
        body.extend_from_slice(&(data.len() as u64).to_le_bytes());
        body.extend_from_slice(&(PIECE as u64).to_le_bytes());
        body.extend_from_slice(&(c.len() as u32).to_le_bytes());
        for v in c {
            body.extend_from_slice(&v.to_le_bytes());
        }
        offset += data.padded_len() as u64;
    }
    let mut h = Vec::with_capacity(header_len + 4);
    h.extend_from_slice(MAGIC);
    h.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    h.extend_from_slice(&(header_len as u32).to_le_bytes());
    h.extend_from_slice(model_key);
    h.extend_from_slice(&body);
    ensure!(h.len() == header_len, "ctx header length mismatch");
    let crc = crc32fast::hash(&h);
    h.extend_from_slice(&crc.to_le_bytes());
    Ok(h)
}

fn open_direct(path: &Path, write: bool) -> Result<File> {
    let mut o = OpenOptions::new();
    if write {
        o.write(true).create(true).truncate(true);
    } else {
        o.read(true);
    }
    match o.clone().custom_flags(libc::O_DIRECT).open(path) {
        Ok(f) => Ok(f),
        // Filesystems without O_DIRECT (tmpfs) fall back to buffered I/O;
        // the aligned layout is valid either way.
        Err(e) if e.raw_os_error() == Some(libc::EINVAL) => o
            .open(path)
            .with_context(|| format!("open {}", path.display())),
        Err(e) => Err(e).with_context(|| format!("open {}", path.display())),
    }
}

/// Write `snap` to `path` atomically (tmp file, fsync, rename). Returns bytes.
pub fn write_file(path: &Path, snap: &CtxSnapshot, model_key: &[u8; 32]) -> Result<u64> {
    let tokens = tokens_bytes(&snap.tokens)?;
    let header = build_header(snap, model_key, &tokens)?;
    let head = AlignedBuf::from_slice(&header)?;
    let tmp = path.with_extension("tmp");
    let result = (|| -> Result<u64> {
        let mut f = open_direct(&tmp, true)?;
        let mut total = 0u64;
        for buf in std::iter::once(&head)
            .chain(std::iter::once(&tokens))
            .chain(snap.sections.iter().map(|s| &s.data))
        {
            f.write_all(buf.padded_slice())
                .with_context(|| format!("write {}", tmp.display()))?;
            total += buf.padded_len() as u64;
        }
        f.sync_all()?;
        Ok(total)
    })();
    match result {
        Ok(total) => {
            std::fs::rename(&tmp, path)
                .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
            Ok(total)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.at + n <= self.b.len(), "ctx header truncated");
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn name(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        Ok(std::str::from_utf8(self.take(n)?)?.to_string())
    }
}

struct SectionEntry {
    name: String,
    offset: u64,
    len: usize,
    piece: usize,
    crcs: Vec<u32>,
}

fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    f.read_exact_at(buf, offset)
        .with_context(|| format!("read {} bytes at {offset}", buf.len()))
}

/// Read and fully validate a checkpoint. Any mismatch is an error: callers
/// treat an error as a miss and delete the file.
pub fn read_file(path: &Path, model_key: &[u8; 32]) -> Result<CtxSnapshot> {
    let f = open_direct(path, false)?;
    let file_len = f.metadata()?.len();
    let mut first = AlignedBuf::zeroed(ALIGN)?;
    read_exact_at(&f, first.padded_mut_slice(), 0)?;
    let fixed = first.padded_slice();
    ensure!(&fixed[..8] == MAGIC, "not an Atlas ctx checkpoint");
    let version = u32::from_le_bytes(fixed[8..12].try_into()?);
    ensure!(
        version == FORMAT_VERSION,
        "ctx checkpoint format {version} != {FORMAT_VERSION}"
    );
    let header_len = u32::from_le_bytes(fixed[12..16].try_into()?) as usize;
    ensure!(header_len + 4 <= MAX_HEADER, "ctx header too large");
    let header = if header_len + 4 <= ALIGN {
        first
    } else {
        let mut h = AlignedBuf::zeroed(header_len + 4)?;
        read_exact_at(&f, h.padded_mut_slice(), 0)?;
        h
    };
    let hb = &header.padded_slice()[..header_len + 4];
    let stored = u32::from_le_bytes(hb[header_len..].try_into()?);
    ensure!(
        crc32fast::hash(&hb[..header_len]) == stored,
        "ctx header checksum mismatch"
    );
    ensure!(
        &hb[16..48] == model_key,
        "ctx checkpoint belongs to a different engine/model/recipe"
    );
    let mut c = Cursor { b: &hb[..header_len], at: 48 };
    // Counts come from the file: never pre-allocate from them. Every entry
    // is bounds-checked by `Cursor::take`, so growth stops at the header end.
    let n_meta = c.u32()? as usize;
    let mut meta = Vec::new();
    for _ in 0..n_meta {
        let name = c.name()?;
        meta.push((name, c.u64()?));
    }
    let n_sections = c.u32()? as usize;
    let mut entries = Vec::new();
    for _ in 0..n_sections {
        let name = c.name()?;
        let offset = c.u64()?;
        let len = usize::try_from(c.u64()?)?;
        let piece = usize::try_from(c.u64()?)?;
        let n = c.u32()? as usize;
        ensure!(piece > 0 && n == len.div_ceil(piece), "ctx section '{name}' piece table");
        let mut crcs = Vec::new();
        for _ in 0..n {
            crcs.push(c.u32()?);
        }
        ensure!(
            offset % ALIGN as u64 == 0 && offset + round_up(len) as u64 <= file_len,
            "ctx section '{name}' lies outside the file"
        );
        entries.push(SectionEntry { name, offset, len, piece, crcs });
    }
    let mut tokens = None;
    let mut sections = Vec::new();
    for e in entries {
        let mut data = AlignedBuf::zeroed(e.len)?;
        read_exact_at(&f, data.padded_mut_slice(), e.offset)?;
        verify_pieces(&e, data.as_slice())?;
        if e.name == TOKENS {
            ensure!(e.len % 4 == 0, "ctx tokens section is not u32-sized");
            tokens = Some(
                data.as_slice()
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<u32>>(),
            );
        } else {
            sections.push(CtxSection { name: e.name, data });
        }
    }
    let Some(tokens) = tokens else {
        bail!("ctx checkpoint has no tokens section");
    };
    Ok(CtxSnapshot { tokens, meta, sections })
}

fn verify_pieces(e: &SectionEntry, data: &[u8]) -> Result<()> {
    let got = if e.piece == PIECE {
        crcs(data)
    } else {
        data.chunks(e.piece).map(crc32fast::hash).collect()
    };
    if let Some(i) = got.iter().zip(&e.crcs).position(|(a, b)| a != b) {
        bail!(
            "ctx section '{}' checksum mismatch in piece {i} (bytes {}..)",
            e.name,
            i * e.piece
        );
    }
    Ok(())
}

/// Flush helper for callers that want the directory entry durable too.
pub fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> CtxSnapshot {
        let mut big = AlignedBuf::zeroed(3 * ALIGN + 17).unwrap();
        for (i, b) in big.as_mut_slice().iter_mut().enumerate() {
            *b = (i * 31 % 251) as u8;
        }
        CtxSnapshot {
            tokens: (0..1000).collect(),
            meta: vec![("c".into(), 1000), ("block_size".into(), 16)],
            sections: vec![
                CtxSection { name: "kv.L0.k".into(), data: big },
                CtxSection { name: "empty".into(), data: AlignedBuf::zeroed(0).unwrap() },
            ],
        }
    }

    #[test]
    fn round_trip_preserves_everything() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.ckpt");
        let key = [7u8; 32];
        let s = snap();
        let bytes = write_file(&p, &s, &key).unwrap();
        assert_eq!(bytes, std::fs::metadata(&p).unwrap().len());
        assert!(!p.with_extension("tmp").exists());
        let r = read_file(&p, &key).unwrap();
        assert_eq!(r.tokens, s.tokens);
        assert_eq!(r.meta_req("c").unwrap(), 1000);
        assert_eq!(
            r.section("kv.L0.k").unwrap().as_slice(),
            s.section("kv.L0.k").unwrap().as_slice()
        );
        assert_eq!(r.section("empty").unwrap().len(), 0);
    }

    #[test]
    fn a_flipped_byte_in_any_section_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.ckpt");
        let key = [7u8; 32];
        write_file(&p, &snap(), &key).unwrap();
        let clean = std::fs::read(&p).unwrap();
        // Header, tokens section, and state section each get one flip.
        for at in [60usize, ALIGN + 5, 3 * ALIGN + 100] {
            let mut bad = clean.clone();
            bad[at] ^= 0x01;
            std::fs::write(&p, &bad).unwrap();
            let err = read_file(&p, &key).err().expect("corruption must be rejected");
            assert!(format!("{err:#}").contains("mismatch"), "offset {at}: {err:#}");
        }
    }

    #[test]
    fn a_different_model_key_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.ckpt");
        write_file(&p, &snap(), &[7u8; 32]).unwrap();
        let err = read_file(&p, &[8u8; 32]).err().unwrap();
        assert!(format!("{err:#}").contains("different engine/model/recipe"));
    }

    #[test]
    fn absurd_counts_in_a_valid_crc_header_are_rejected_not_allocated() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.ckpt");
        let key = [7u8; 32];
        write_file(&p, &snap(), &key).unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        // n_meta := u32::MAX, then re-seal the header CRC (a forged file).
        bytes[48..52].copy_from_slice(&u32::MAX.to_le_bytes());
        let crc = crc32fast::hash(&bytes[..header_len]);
        bytes[header_len..header_len + 4].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(&p, &bytes).unwrap();
        let err = read_file(&p, &key).err().unwrap();
        assert!(format!("{err:#}").contains("truncated"), "{err:#}");
    }

    #[test]
    fn truncated_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.ckpt");
        write_file(&p, &snap(), &[7u8; 32]).unwrap();
        let clean = std::fs::read(&p).unwrap();
        std::fs::write(&p, &clean[..clean.len() - ALIGN]).unwrap();
        assert!(read_file(&p, &[7u8; 32]).is_err());
    }
}
