// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::path::Path;

use super::decode::read_gguf_header_with_len;
use super::payload::{FileIdentity, Glm53GgufFiles, Glm53Iq3Files, OpenShard};
use super::sha256::HashingReader;
use super::{GgufDirectory, GgufHeader, GgufShardRef, GgufValue, assemble_split_shards};

mod schema;
pub use schema::GLM53_GGUF_VOCAB_SIZE;
pub(super) use schema::expected_schema;

pub const GLM53_GGUF_REVISION: &str = "ac47690c15c8703615ab7d9c1ef2293d45372757";

const IQ3_NAMES: [&str; 4] = [
    "GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf",
    "GLM-5.3-Flash-UD-IQ3_XXS-00002-of-00004.gguf",
    "GLM-5.3-Flash-UD-IQ3_XXS-00003-of-00004.gguf",
    "GLM-5.3-Flash-UD-IQ3_XXS-00004-of-00004.gguf",
];
const IQ3_BYTES: [u64; 4] = [9_429_859, 49_261_511_584, 49_075_832_480, 22_020_797_792];
const IQ3_SHA256: [&str; 4] = [
    "b3282af0711eec07a0462a6126dc75b32bc14b698a39efec1b3b34d3c4217dc5",
    "a4dba331fe92ffbd71317839cc1452c57f0b5b3f971a9e96ee5107f9a615cdac",
    "5b728b5b263c662201ef4a4a2b12fb509fe230d012418d0ed37c220a8a5aa230",
    "4b6d16e0d1381e1b4b7eeb6c9799671bf98153a995be12e75e3da93cd97981d2",
];
const Q2_NAMES: [&str; 4] = [
    "GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf",
    "GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf",
    "GLM-5.3-Flash-UD-Q2_K_XL-00003-of-00004.gguf",
    "GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf",
];
const Q2_BYTES: [u64; 4] = [9_429_859, 49_294_975_936, 49_949_266_048, 9_466_399_584];
const Q2_SHA256: [&str; 4] = [
    "b2aaab111a0f93f04b0627270691fe430b6bb14b17c0d62f9c0f1ac75e7efef4",
    "f4a9e1ab13d5d9620f5590c9a4aba4c169e6707a1554f3be3fe3112f80a66825",
    "330a8ad76c787b3ab6df062dd30abea7aafe6a0e3d5860e8720dfa4abb2434a5",
    "5c294f42edc5d69cf00a79ab444e61f0742e9933105fc5c85f97da54dab8946f",
];

// UD-IQ2_XXS is the smallest published recipe that still uses only quant types
// with a pinned GLM MMQ specialization, and still carries the 29-tensor
// `blk.45` NextN head that speculation depends on. It is the profile that fits
// one Spark with headroom once every tensor is device-resident: Atlas copies
// the whole checkpoint into resident memory rather than mmap-ing it, so unlike
// llama.cpp it cannot rely on the page cache to absorb the difference.
const IQ2_XXS_NAMES: [&str; 4] = [
    "GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf",
    "GLM-5.3-Flash-UD-IQ2_XXS-00002-of-00004.gguf",
    "GLM-5.3-Flash-UD-IQ2_XXS-00003-of-00004.gguf",
    "GLM-5.3-Flash-UD-IQ2_XXS-00004-of-00004.gguf",
];
const IQ2_XXS_BYTES: [u64; 4] = [9_429_888, 49_896_298_080, 49_248_507_840, 2_690_716_000];
const IQ2_XXS_SHA256: [&str; 4] = [
    "d9605eec5aa2c14fc9ccff1ba3341a1e9a1476b08714d41cd43d0e10a4db3f7a",
    "d51a3d9ca05022d53e71753df75ee64ed5fd4e47655650ae023e9717b3ec158d",
    "89180f13784dc1128586b98adf1ca876279881a20bc4f01747703ce352563372",
    "9e0ef47daa1fdffce3cf2ac1594ce4634d176ad018be6e692729757f38d5fbf6",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53QuantProfile {
    UdQ2KXl,
    UdIq3Xxs,
    UdIq2Xxs,
}

impl Glm53QuantProfile {
    pub const PRIMARY: Self = Self::UdQ2KXl;

    pub const fn directory_name(self) -> &'static str {
        match self {
            Self::UdQ2KXl => "UD-Q2_K_XL",
            Self::UdIq3Xxs => "UD-IQ3_XXS",
            Self::UdIq2Xxs => "UD-IQ2_XXS",
        }
    }

    pub const fn canonical_file_names(self) -> &'static [&'static str; 4] {
        match self {
            Self::UdQ2KXl => &Q2_NAMES,
            Self::UdIq3Xxs => &IQ3_NAMES,
            Self::UdIq2Xxs => &IQ2_XXS_NAMES,
        }
    }

    pub const fn shard_bytes(self) -> &'static [u64; 4] {
        match self {
            Self::UdQ2KXl => &Q2_BYTES,
            Self::UdIq3Xxs => &IQ3_BYTES,
            Self::UdIq2Xxs => &IQ2_XXS_BYTES,
        }
    }

    pub const fn shard_sha256(self) -> &'static [&'static str; 4] {
        match self {
            Self::UdQ2KXl => &Q2_SHA256,
            Self::UdIq3Xxs => &IQ3_SHA256,
            Self::UdIq2Xxs => &IQ2_XXS_SHA256,
        }
    }

    pub const fn tensor_bytes(self) -> u64 {
        match self {
            Self::UdQ2KXl => 108_710_550_904,
            Self::UdIq3Xxs => 120_358_051_192,
            Self::UdIq2Xxs => 101_835_431_288,
        }
    }

    pub const fn total_file_bytes(self) -> u64 {
        match self {
            Self::UdQ2KXl => 108_720_071_427,
            Self::UdIq3Xxs => 120_367_571_715,
            Self::UdIq2Xxs => 101_844_951_808,
        }
    }

    const fn file_type(self) -> u64 {
        match self {
            Self::UdQ2KXl => 10,
            Self::UdIq3Xxs => 23,
            Self::UdIq2Xxs => 19,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Iq3Summary {
    pub shards: usize,
    pub tensors: usize,
    pub tensor_bytes: u64,
}

/// Generic name for the retained compatibility summary.
pub type Glm53GgufSummary = Glm53Iq3Summary;

struct OpenVerifiedShard {
    file: File,
    identity: FileIdentity,
    header: GgufHeader,
}

fn split_no(header: &GgufHeader) -> Result<usize> {
    match header.metadata.get("split.no") {
        Some(GgufValue::Unsigned(value)) => Ok(usize::try_from(*value)?),
        _ => bail!("GGUF shard missing unsigned split.no"),
    }
}

fn unsigned(header: &GgufHeader, key: &str) -> Option<u64> {
    match header.metadata.get(key) {
        Some(GgufValue::Unsigned(value)) => Some(*value),
        _ => None,
    }
}

fn array_shape(header: &GgufHeader, key: &str) -> Option<(u32, u64)> {
    match header.metadata.get(key) {
        Some(GgufValue::Array { element_type, len }) => Some((*element_type, *len)),
        _ => None,
    }
}

fn digest_hex(digest: [u8; 32]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(64);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn validate_identity(
    profile: Glm53QuantProfile,
    index: usize,
    file_len: u64,
    digest: &str,
) -> Result<()> {
    if index >= profile.shard_bytes().len()
        || file_len != profile.shard_bytes()[index]
        || digest != profile.shard_sha256()[index]
    {
        bail!("pinned GLM-5.3 GGUF profile shard identity mismatch");
    }
    Ok(())
}

/// Opt-in: trust a previously computed shard SHA-256 when the file's full
/// identity (dev, inode, size, mtime, ctime, ...) is unchanged.
///
/// The hand-rolled SHA-256 in `sha256.rs` runs single-threaded at ~140 MB/s, so
/// the 95 GiB UD-IQ2_XXS checkpoint costs ~7 minutes of hashing on EVERY open.
/// During bring-up that is the dominant cost of each fix-and-retest cycle.
///
/// Safety argument, in order of what matters:
/// 1. The cache only skips *computing* the digest. `validate_identity` still
///    compares the digest against the pinned profile SHA, so a stale or forged
///    cache entry can only produce a false REFUSAL, never a false pass.
/// 2. The key includes `ctime`, which the kernel sets on every content or
///    metadata change and which userspace cannot backdate. Rewriting the file
///    in place — even preserving size and mtime — misses the cache.
/// 3. It is off unless `ATLAS_GLM53_TRUST_SHARD_IDENTITY_CACHE=1`, so serve
///    paths that never set it keep the full hash.
pub const GLM53_SHARD_IDENTITY_CACHE_ENV: &str = "ATLAS_GLM53_TRUST_SHARD_IDENTITY_CACHE";

fn shard_cache_enabled() -> bool {
    std::env::var(GLM53_SHARD_IDENTITY_CACHE_ENV).is_ok_and(|v| v == "1")
}

fn shard_cache_path(path: &Path) -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let name = path.file_name()?.to_str()?;
    Some(
        Path::new(&home)
            .join(".cache")
            .join("atlas")
            .join("glm53-shard-sha256")
            .join(format!("{name}.identity")),
    )
}

/// Returns the cached hex digest if the stored identity matches exactly.
#[cfg(test)]
pub(super) fn shard_cache_lookup_for_test(path: &Path, identity: &FileIdentity) -> Option<String> {
    shard_cache_lookup(path, identity)
}

#[cfg(test)]
pub(super) fn shard_cache_store_for_test(path: &Path, identity: &FileIdentity, hex: &str) {
    shard_cache_store(path, identity, hex)
}

fn shard_cache_lookup(path: &Path, identity: &FileIdentity) -> Option<String> {
    let entry = std::fs::read_to_string(shard_cache_path(path)?).ok()?;
    let mut lines = entry.lines();
    let stored_key = lines.next()?;
    let stored_hex = lines.next()?;
    (stored_key == identity.cache_key() && stored_hex.len() == 64).then(|| stored_hex.to_string())
}

fn shard_cache_store(path: &Path, identity: &FileIdentity, digest_hex: &str) {
    let Some(target) = shard_cache_path(path) else {
        return;
    };
    if let Some(dir) = target.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // Best effort: a failed cache write just means the next open re-hashes.
    let _ = std::fs::write(
        target,
        format!("{}\n{}\n", identity.cache_key(), digest_hex),
    );
}

fn read_verified(path: &Path, profile: Glm53QuantProfile) -> Result<OpenVerifiedShard> {
    let mut file = File::open(path).context("failed to open pinned GLM-5.3 GGUF shard")?;
    let before = FileIdentity::capture(&file)?;
    let file_len = file.metadata()?.len();

    if shard_cache_enabled() {
        if let Some(cached_hex) = shard_cache_lookup(path, &before) {
            // Parse the header without hashing; the pinned-SHA comparison below
            // is what actually admits the shard.
            let header = read_gguf_header_with_len(&mut file, file_len)?;
            let after = FileIdentity::capture(&file)?;
            if before != after {
                bail!("pinned GLM-5.3 GGUF shard changed during validation");
            }
            let index = split_no(&header)?;
            validate_identity(profile, index, file_len, &cached_hex)?;
            tracing::info!(
                "GLM shard {} admitted from identity cache ({GLM53_SHARD_IDENTITY_CACHE_ENV}=1); hash skipped",
                path.display()
            );
            return Ok(OpenVerifiedShard {
                file,
                identity: after,
                header,
            });
        }
    }

    let mut reader = HashingReader::new(&mut file);
    let header = read_gguf_header_with_len(&mut reader, file_len)?;
    let (digest, consumed) = reader
        .finish()
        .context("failed to hash pinned GLM-5.3 GGUF shard")?;
    let after = FileIdentity::capture(&file)?;
    if consumed != file_len || before != after {
        bail!("pinned GLM-5.3 GGUF shard changed during validation");
    }
    let index = split_no(&header)?;
    let hex = digest_hex(digest);
    validate_identity(profile, index, file_len, &hex)?;
    if shard_cache_enabled() {
        shard_cache_store(path, &after, &hex);
    }
    Ok(OpenVerifiedShard {
        file,
        identity: after,
        header,
    })
}

fn validate_headers(
    profile: Glm53QuantProfile,
    headers: &[&GgufHeader],
) -> Result<(Glm53Iq3Summary, GgufDirectory)> {
    if headers.len() != 4 {
        bail!("pinned GLM-5.3 GGUF profile requires four shards");
    }
    let refs: Vec<_> = headers
        .iter()
        .map(|header| GgufShardRef { header })
        .collect();
    let main = headers
        .iter()
        .find(|header| split_no(header).ok() == Some(0))
        .context("pinned GLM-5.3 metadata shard is missing")?;
    if unsigned(main, "general.file_type") != Some(profile.file_type())
        || unsigned(main, "general.quantization_version") != Some(2)
        || unsigned(main, "glm5next.block_count") != Some(46)
        || unsigned(main, "glm5next.nextn_predict_layers") != Some(1)
        || array_shape(main, "glm5next.attention.head_count_kv") != Some((5, 46))
        || array_shape(main, "glm5next.swiglu_clamp_exp") != Some((6, 46))
        || array_shape(main, "glm5next.swiglu_clamp_shexp") != Some((6, 46))
    {
        bail!("pinned GLM-5.3 GGUF profile metadata ABI mismatch");
    }
    let directory = assemble_split_shards(&refs)?;
    if directory.architecture != "glm5next" || directory.tensors.len() != 1412 {
        bail!("pinned GLM-5.3 GGUF profile architecture or tensor count mismatch");
    }
    let expected = expected_schema(profile);
    if expected.len() != directory.tensors.len() {
        bail!("internal GLM-5.3 schema count mismatch");
    }
    let mut tensor_bytes = 0u64;
    for (name, (dimensions, ggml_type)) in expected {
        let tensor = directory
            .tensors
            .get(&name)
            .with_context(|| format!("missing expected GLM-5.3 tensor {name}"))?;
        if tensor.info.dimensions != dimensions || tensor.info.ggml_type != ggml_type {
            bail!("expected GLM-5.3 tensor {name} shape/type mismatch");
        }
        tensor_bytes = tensor_bytes
            .checked_add(tensor.info.byte_len)
            .context("tensor byte total overflow")?;
    }
    if tensor_bytes != profile.tensor_bytes() {
        bail!("GLM-5.3 tensor byte total mismatch: {tensor_bytes}");
    }
    Ok((
        Glm53Iq3Summary {
            shards: 4,
            tensors: 1412,
            tensor_bytes,
        },
        directory,
    ))
}

/// Opens, validates and retains one exact pinned four-shard profile.
pub fn open_glm53_files<P: AsRef<Path>>(
    profile: Glm53QuantProfile,
    paths: &[P],
) -> Result<Glm53GgufFiles> {
    if paths.len() != 4 {
        bail!("pinned GLM-5.3 GGUF profile requires four shard paths");
    }
    let verified = paths
        .iter()
        .map(|path| read_verified(path.as_ref(), profile))
        .collect::<Result<Vec<_>>>()?;
    let headers = verified
        .iter()
        .map(|shard| &shard.header)
        .collect::<Vec<_>>();
    let (summary, directory) = validate_headers(profile, &headers)?;
    let shards = verified
        .into_iter()
        .map(|shard| {
            Ok(OpenShard {
                split_no: split_no(&shard.header)?,
                data_offset: shard.header.data_offset,
                file: shard.file,
                identity: shard.identity,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Glm53GgufFiles::new_with_profile(profile, shards, directory, summary)
}

/// Compatibility wrapper for the original pinned UD-IQ3_XXS callers.
pub fn open_glm53_iq3_files<P: AsRef<Path>>(paths: &[P]) -> Result<Glm53Iq3Files> {
    open_glm53_files(Glm53QuantProfile::UdIq3Xxs, paths)
}

pub fn validate_glm53_files<P: AsRef<Path>>(
    profile: Glm53QuantProfile,
    paths: &[P],
) -> Result<Glm53GgufSummary> {
    Ok(open_glm53_files(profile, paths)?.summary())
}

/// Compatibility wrapper for callers that only need an admission summary.
pub fn validate_glm53_iq3_files<P: AsRef<Path>>(paths: &[P]) -> Result<Glm53Iq3Summary> {
    validate_glm53_files(Glm53QuantProfile::UdIq3Xxs, paths)
}

#[cfg(test)]
pub(super) fn validate_verified_for_test(
    profile: Glm53QuantProfile,
    headers: Vec<GgufHeader>,
) -> Result<Glm53Iq3Summary> {
    let headers = headers.iter().collect::<Vec<_>>();
    validate_headers(profile, &headers).map(|(summary, _)| summary)
}

#[cfg(test)]
pub(super) fn validate_identity_for_test(
    profile: Glm53QuantProfile,
    index: usize,
    file_len: u64,
    digest: &str,
) -> Result<()> {
    validate_identity(profile, index, file_len, digest)
}
