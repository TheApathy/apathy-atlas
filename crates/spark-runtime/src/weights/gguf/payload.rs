// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};

use super::glm53::{Glm53GgufSummary, Glm53QuantProfile};
use super::{GgufDirectory, LocatedTensor};

const MAX_STREAM_BUFFER: usize = 4 * 1024 * 1024;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct FileIdentity {
    len: u64,
    modified: Option<std::time::SystemTime>,
    readonly: bool,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    mode: u32,
    #[cfg(unix)]
    nlink: u64,
    #[cfg(unix)]
    ctime: i64,
    #[cfg(unix)]
    ctime_nsec: i64,
}

impl FileIdentity {
    /// A stable, order-fixed rendering of every captured field, for the shard
    /// identity cache. `ctime` is kernel-managed and cannot be set from
    /// userspace, so any content change — even one that preserves size and
    /// mtime — changes this string.
    pub(super) fn cache_key(&self) -> String {
        #[cfg(unix)]
        {
            format!(
                "v1 len={} mtime={:?} ro={} dev={} ino={} mode={} nlink={} ctime={}.{:09}",
                self.len,
                self.modified
                    .map(|m| m.duration_since(std::time::UNIX_EPOCH).ok()),
                self.readonly,
                self.dev,
                self.ino,
                self.mode,
                self.nlink,
                self.ctime,
                self.ctime_nsec
            )
        }
        #[cfg(not(unix))]
        {
            format!(
                "v1 len={} mtime={:?} ro={}",
                self.len,
                self.modified
                    .map(|m| m.duration_since(std::time::UNIX_EPOCH).ok()),
                self.readonly
            )
        }
    }

    #[cfg(test)]
    pub(super) fn cache_key_for_test(&self) -> String {
        self.cache_key()
    }

    pub(super) fn capture(file: &File) -> Result<Self> {
        let metadata = file.metadata().context("failed to stat pinned GGUF file")?;
        Self::from_metadata(&metadata)
    }

    fn from_metadata(metadata: &Metadata) -> Result<Self> {
        if !metadata.is_file() {
            bail!("pinned GGUF input is not a regular file");
        }
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            readonly: metadata.permissions().readonly(),
            #[cfg(unix)]
            dev: metadata.dev(),
            #[cfg(unix)]
            ino: metadata.ino(),
            #[cfg(unix)]
            mode: metadata.mode(),
            #[cfg(unix)]
            nlink: metadata.nlink(),
            #[cfg(unix)]
            ctime: metadata.ctime(),
            #[cfg(unix)]
            ctime_nsec: metadata.ctime_nsec(),
        })
    }
}

pub(super) struct OpenShard {
    pub(super) split_no: usize,
    pub(super) data_offset: u64,
    pub(super) file: File,
    pub(super) identity: FileIdentity,
}

/// Exact pinned profile files retained open after byte/hash/schema admission.
pub struct Glm53GgufFiles {
    pub(super) shards: Vec<OpenShard>,
    directory: GgufDirectory,
    summary: Glm53GgufSummary,
    profile: Glm53QuantProfile,
}

/// Compatibility alias for the original UD-IQ3_XXS API.
pub type Glm53Iq3Files = Glm53GgufFiles;

impl Glm53GgufFiles {
    pub(super) fn new_with_profile(
        profile: Glm53QuantProfile,
        mut shards: Vec<OpenShard>,
        directory: GgufDirectory,
        summary: Glm53GgufSummary,
    ) -> Result<Self> {
        shards.sort_unstable_by_key(|shard| shard.split_no);
        if shards.len() != directory.split_count
            || shards
                .iter()
                .enumerate()
                .any(|(index, shard)| shard.split_no != index)
        {
            bail!("admitted GGUF file handles do not cover every split");
        }
        Ok(Self {
            shards,
            directory,
            summary,
            profile,
        })
    }

    pub fn summary(&self) -> Glm53GgufSummary {
        self.summary
    }

    pub fn profile(&self) -> Glm53QuantProfile {
        self.profile
    }

    pub fn tensor(&self, name: &str) -> Option<&LocatedTensor> {
        self.directory.tensors.get(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.directory.tensors.keys().map(String::as_str)
    }

    /// Stream one admitted tensor without reopening the shard or allocating its full payload.
    pub fn stream_tensor(
        &mut self,
        name: &str,
        scratch: &mut [u8],
        mut sink: impl FnMut(u64, &[u8]) -> Result<()>,
    ) -> Result<()> {
        if scratch.is_empty() || scratch.len() > MAX_STREAM_BUFFER {
            bail!("GGUF stream buffer must contain 1..={MAX_STREAM_BUFFER} bytes");
        }
        let located = self
            .directory
            .tensors
            .get(name)
            .context("requested tensor is absent from admitted GGUF directory")?
            .clone();
        let shard = self
            .shards
            .get_mut(located.shard_no)
            .context("admitted GGUF tensor references an invalid split")?;
        if FileIdentity::capture(&shard.file)? != shard.identity {
            bail!("pinned GGUF shard changed after admission");
        }
        let start = shard
            .data_offset
            .checked_add(located.info.offset)
            .context("GGUF tensor start overflow")?;
        let end = start
            .checked_add(located.info.byte_len)
            .context("GGUF tensor end overflow")?;
        if end > shard.identity.len {
            bail!("GGUF tensor exceeds its admitted shard");
        }
        let operation = (|| {
            shard.file.seek(SeekFrom::Start(start))?;
            let mut copied = 0u64;
            while copied < located.info.byte_len {
                let chunk =
                    usize::try_from((located.info.byte_len - copied).min(scratch.len() as u64))?;
                shard.file.read_exact(&mut scratch[..chunk])?;
                sink(copied, &scratch[..chunk])?;
                copied += chunk as u64;
            }
            Ok(())
        })();
        crate::weights::evict_page_cache(&shard.file);
        if FileIdentity::capture(&shard.file)? != shard.identity {
            bail!("pinned GGUF shard changed while streaming a tensor");
        }
        operation
    }
}
