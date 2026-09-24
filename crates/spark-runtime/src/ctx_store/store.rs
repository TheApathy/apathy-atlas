// SPDX-License-Identifier: AGPL-3.0-only

//! Directory index, lookup, background writer and disk budget.
//!
//! Layout: `<root>/<model key, 16 hex>/<C, 10 digits>-<prefix hash>.ckpt`.
//! The index is rebuilt from file names at startup, so a restart (or a
//! model swap back) finds every checkpoint without reading any of them.
//! Lookup hashes the prompt only at the distinct checkpoint depths, then
//! the loaded file's stored tokens are compared exactly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use parking_lot::Mutex;

use super::format::{CtxSnapshot, read_file, sync_dir, write_file};
use super::identity::{hex, prefix_hash};

/// One checkpoint on disk.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Tokens covered (the restore position).
    pub c: usize,
    pub hash: u64,
    pub path: PathBuf,
    pub bytes: u64,
    last_used: u64,
}

struct Inner {
    dir: PathBuf,
    key: [u8; 32],
    budget: u64,
    entries: Mutex<Vec<Entry>>,
    busy: AtomicBool,
    clock: AtomicU64,
}

/// Handle to one model key's checkpoint directory. Cheap to clone.
#[derive(Clone)]
pub struct CtxStore {
    inner: Arc<Inner>,
}

fn file_name(c: usize, hash: u64) -> String {
    format!("{c:010}-{hash:016x}.ckpt")
}

fn parse_name(name: &str) -> Option<(usize, u64)> {
    let stem = name.strip_suffix(".ckpt")?;
    let (c, h) = stem.split_once('-')?;
    Some((c.parse().ok()?, u64::from_str_radix(h, 16).ok()?))
}

impl CtxStore {
    /// Open (creating) the directory for `key` under `root` and index it.
    /// Leftover `.tmp` files from an interrupted write are removed.
    pub fn open(root: &Path, key: [u8; 32], budget_bytes: u64) -> Result<Self> {
        let dir = root.join(&hex(&key)[..16]);
        std::fs::create_dir_all(&dir)?;
        let mut entries = Vec::new();
        for e in std::fs::read_dir(&dir)?.filter_map(|e| e.ok()) {
            let path = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if name.ends_with(".tmp") {
                let _ = std::fs::remove_file(&path);
                continue;
            }
            let Some((c, hash)) = parse_name(&name) else {
                continue;
            };
            let meta = e.metadata()?;
            let last_used = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            entries.push(Entry { c, hash, path, bytes: meta.len(), last_used });
        }
        let clock = entries.iter().map(|e| e.last_used).max().unwrap_or(0) + 1;
        Ok(Self {
            inner: Arc::new(Inner {
                dir,
                key,
                budget: budget_bytes,
                entries: Mutex::new(entries),
                busy: AtomicBool::new(false),
                clock: AtomicU64::new(clock),
            }),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.inner.dir
    }

    pub fn key(&self) -> &[u8; 32] {
        &self.inner.key
    }

    pub fn entries(&self) -> Vec<Entry> {
        self.inner.entries.lock().clone()
    }

    pub fn total_bytes(&self) -> u64 {
        self.inner.entries.lock().iter().map(|e| e.bytes).sum()
    }

    /// Deepest checkpoint whose token prefix matches and leaves at least one
    /// suffix token to prefill (logits come from prefilling that suffix).
    pub fn lookup(&self, tokens: &[u32]) -> Option<Entry> {
        let entries = self.inner.entries.lock().clone();
        let mut by_c: HashMap<usize, u64> = HashMap::new();
        let mut best: Option<Entry> = None;
        for e in entries {
            if e.c == 0 || e.c >= tokens.len() {
                continue;
            }
            let h = *by_c.entry(e.c).or_insert_with(|| prefix_hash(&tokens[..e.c]));
            if h == e.hash && best.as_ref().is_none_or(|b| e.c > b.c) {
                best = Some(e);
            }
        }
        best
    }

    /// Read and validate `e` against `tokens`. On any failure the file is
    /// deleted and dropped from the index, so a corrupt checkpoint is
    /// rejected once and never retried.
    pub fn load(&self, e: &Entry, tokens: &[u32]) -> Result<CtxSnapshot> {
        let r = read_file(&e.path, &self.inner.key).and_then(|s| {
            ensure!(
                s.tokens.len() == e.c && tokens.len() >= e.c && s.tokens[..] == tokens[..e.c],
                "ctx checkpoint tokens do not match the prompt prefix (stale or colliding file)"
            );
            Ok(s)
        });
        match r {
            Ok(s) => {
                self.touch(e);
                Ok(s)
            }
            Err(err) => {
                self.forget(e, true);
                Err(err)
            }
        }
    }

    fn touch(&self, e: &Entry) {
        let now = self.inner.clock.fetch_add(1, Ordering::Relaxed);
        if let Some(x) = self.inner.entries.lock().iter_mut().find(|x| x.path == e.path) {
            x.last_used = now;
        }
    }

    fn forget(&self, e: &Entry, delete: bool) {
        self.inner.entries.lock().retain(|x| x.path != e.path);
        if delete {
            let _ = std::fs::remove_file(&e.path);
        }
    }

    /// Whether a background write is in flight.
    pub fn is_busy(&self) -> bool {
        self.inner.busy.load(Ordering::Acquire)
    }

    /// Queue `snap` for a background write. Returns false (dropping the
    /// snapshot) when a write is already in flight: one staging copy at a
    /// time bounds host memory.
    pub fn submit(&self, snap: CtxSnapshot) -> bool {
        if self
            .inner
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        let me = self.clone();
        let spawned = std::thread::Builder::new()
            .name("atlas-ctx-writer".into())
            .spawn(move || {
                if let Err(e) = me.write_now(&snap) {
                    tracing::warn!("ctx-cache: checkpoint write failed: {e:#}");
                }
                me.inner.busy.store(false, Ordering::Release);
            });
        if let Err(e) = spawned {
            tracing::warn!("ctx-cache: could not spawn writer: {e}");
            self.inner.busy.store(false, Ordering::Release);
            return false;
        }
        true
    }

    /// Synchronous write + index update + supersede + budget GC.
    pub fn write_now(&self, snap: &CtxSnapshot) -> Result<Entry> {
        let c = snap.tokens.len();
        let hash = prefix_hash(&snap.tokens);
        let path = self.inner.dir.join(file_name(c, hash));
        let t0 = Instant::now();
        let bytes = write_file(&path, snap, &self.inner.key)?;
        let _ = sync_dir(&self.inner.dir);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let now = self.inner.clock.fetch_add(1, Ordering::Relaxed);
        let entry = Entry { c, hash, path: path.clone(), bytes, last_used: now };
        let removed = {
            let mut entries = self.inner.entries.lock();
            let mut removed = Vec::new();
            // A shallower checkpoint of the same conversation is superseded.
            entries.retain(|x| {
                let superseded = x.path == path
                    || (x.c < c && x.hash == prefix_hash(&snap.tokens[..x.c]));
                if superseded && x.path != path {
                    removed.push(x.path.clone());
                }
                !superseded
            });
            entries.push(entry.clone());
            removed.extend(self.gc_locked(&mut entries, &path));
            removed
        };
        for p in &removed {
            let _ = std::fs::remove_file(p);
        }
        tracing::info!(
            "ctx-cache: checkpoint written: {c} tokens, {:.1} MiB in {ms:.0} ms ({:.2} GB/s), \
             {} evicted/superseded, dir {:.2} GiB",
            bytes as f64 / (1 << 20) as f64,
            bytes as f64 / 1e6 / ms.max(1e-3),
            removed.len(),
            self.total_bytes() as f64 / (1u64 << 30) as f64,
        );
        Ok(entry)
    }

    fn gc_locked(&self, entries: &mut Vec<Entry>, keep: &Path) -> Vec<PathBuf> {
        let mut removed = Vec::new();
        loop {
            let total: u64 = entries.iter().map(|e| e.bytes).sum();
            if total <= self.inner.budget {
                break;
            }
            // Oldest first; the new file goes last and only if it alone
            // exceeds the budget.
            let victim = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.path != keep)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .or_else(|| (!entries.is_empty()).then_some(0));
            let Some(i) = victim else { break };
            removed.push(entries.swap_remove(i).path);
        }
        removed
    }

    /// Block until no write is in flight (gate harness / shutdown).
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let t0 = Instant::now();
        while self.is_busy() {
            if t0.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::aligned::AlignedBuf;
    use super::super::format::CtxSection;
    use super::*;

    fn snap(tokens: Vec<u32>, state: usize) -> CtxSnapshot {
        CtxSnapshot {
            tokens,
            meta: vec![("x".into(), 1)],
            sections: vec![CtxSection {
                name: "s".into(),
                data: AlignedBuf::zeroed(state).unwrap(),
            }],
        }
    }

    #[test]
    fn lookup_picks_the_deepest_strict_prefix() {
        let d = tempfile::tempdir().unwrap();
        let s = CtxStore::open(d.path(), [1; 32], u64::MAX).unwrap();
        let conv: Vec<u32> = (0..500).collect();
        // Two unrelated conversations; the second one's prefix is not ours.
        s.write_now(&snap(conv[..100].to_vec(), 10)).unwrap();
        s.write_now(&snap((1000..1300).collect(), 10)).unwrap();
        let e = s.lookup(&conv).unwrap();
        assert_eq!(e.c, 100);
        // Exact-length prompt: nothing left to prefill -> no hit.
        assert!(s.lookup(&conv[..100]).is_none());
        let loaded = s.load(&e, &conv).unwrap();
        assert_eq!(loaded.tokens, conv[..100].to_vec());
    }

    #[test]
    fn deeper_checkpoint_supersedes_its_prefix() {
        let d = tempfile::tempdir().unwrap();
        let s = CtxStore::open(d.path(), [1; 32], u64::MAX).unwrap();
        let conv: Vec<u32> = (0..500).collect();
        s.write_now(&snap(conv[..100].to_vec(), 10)).unwrap();
        s.write_now(&snap(conv[..300].to_vec(), 10)).unwrap();
        let es = s.entries();
        assert_eq!(es.len(), 1);
        assert_eq!(es[0].c, 300);
        assert_eq!(std::fs::read_dir(s.dir()).unwrap().count(), 1);
    }

    #[test]
    fn index_survives_reopen_and_drops_tmp() {
        let d = tempfile::tempdir().unwrap();
        let conv: Vec<u32> = (0..200).collect();
        {
            let s = CtxStore::open(d.path(), [1; 32], u64::MAX).unwrap();
            s.write_now(&snap(conv[..150].to_vec(), 10)).unwrap();
            std::fs::write(s.dir().join("0000000001-00.tmp"), b"x").unwrap();
        }
        let s = CtxStore::open(d.path(), [1; 32], u64::MAX).unwrap();
        assert_eq!(s.lookup(&conv).unwrap().c, 150);
        assert_eq!(std::fs::read_dir(s.dir()).unwrap().count(), 1);
        // A different model key sees nothing.
        let other = CtxStore::open(d.path(), [2; 32], u64::MAX).unwrap();
        assert!(other.lookup(&conv).is_none());
    }

    #[test]
    fn stale_tokens_under_a_matching_name_are_rejected_and_deleted() {
        let d = tempfile::tempdir().unwrap();
        let s = CtxStore::open(d.path(), [1; 32], u64::MAX).unwrap();
        let real: Vec<u32> = (0..100).collect();
        let fake: Vec<u32> = (5000..5100).collect();
        let e = s.write_now(&snap(fake, 10)).unwrap();
        // Rename the fake conversation's file to claim `real`'s prefix.
        let forged = s.dir().join(file_name(100, prefix_hash(&real)));
        std::fs::rename(&e.path, &forged).unwrap();
        let s = CtxStore::open(d.path(), [1; 32], u64::MAX).unwrap();
        let mut prompt = real.clone();
        prompt.push(7);
        let hit = s.lookup(&prompt).unwrap();
        let err = s.load(&hit, &prompt).err().unwrap();
        assert!(format!("{err:#}").contains("do not match"));
        assert!(!forged.exists());
        assert!(s.lookup(&prompt).is_none());
    }

    #[test]
    fn budget_evicts_least_recently_used() {
        let d = tempfile::tempdir().unwrap();
        let s = CtxStore::open(d.path(), [1; 32], 3 * 20_000).unwrap();
        let a: Vec<u32> = (0..10).collect();
        let b: Vec<u32> = (100..110).collect();
        let c: Vec<u32> = (200..210).collect();
        let ea = s.write_now(&snap(a.clone(), 12_000)).unwrap();
        s.write_now(&snap(b.clone(), 12_000)).unwrap();
        // Use `a` so `b` is the LRU victim.
        let mut pa = a.clone();
        pa.push(0);
        s.load(&ea, &pa).unwrap();
        s.write_now(&snap(c, 12_000)).unwrap();
        let cs: Vec<usize> = s.entries().iter().map(|e| e.hash as usize).collect();
        assert_eq!(s.entries().len(), 2, "{cs:?}");
        let mut pb = b.clone();
        pb.push(0);
        assert!(s.lookup(&pb).is_none());
        assert!(s.lookup(&pa).is_some());
    }

    #[test]
    fn submit_is_single_flight() {
        let d = tempfile::tempdir().unwrap();
        let s = CtxStore::open(d.path(), [1; 32], u64::MAX).unwrap();
        assert!(s.submit(snap((0..10).collect(), 64 << 20)));
        // Either still busy (dropped) or already done (accepted); never both
        // in flight at once.
        let second = s.submit(snap((20..30).collect(), 10));
        assert!(s.wait_idle(Duration::from_secs(30)));
        let n = s.entries().len();
        assert_eq!(n, if second { 2 } else { 1 });
    }
}
