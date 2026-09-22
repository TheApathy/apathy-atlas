// SPDX-License-Identifier: AGPL-3.0-only

//! Metadata-only resolution of an exact four-part GLM-5.3 GGUF profile.

use anyhow::{Context, Result, bail};
use spark_runtime::weights::gguf::Glm53QuantProfile;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

const PARTS: usize = 4;

fn validate_expected_names(expected: &[&str; PARTS]) -> Result<()> {
    let mut common_stem = None;
    for (index, name) in expected.iter().enumerate() {
        let path = Path::new(name);
        if path.components().count() != 1 || path.file_name() != Some(OsStr::new(name)) {
            bail!("GLM-5.3 GGUF shard names must be single path components");
        }

        let suffix = format!("-{:05}-of-{PARTS:05}.gguf", index + 1);
        let Some(stem) = name.strip_suffix(&suffix) else {
            bail!("GLM-5.3 GGUF expected names must be ordered parts 1 through {PARTS}");
        };
        if stem.is_empty() {
            bail!("GLM-5.3 GGUF shard name stem must not be empty");
        }
        match common_stem {
            Some(known) if known != stem => {
                bail!("GLM-5.3 GGUF expected names must share one checkpoint stem");
            }
            None => common_stem = Some(stem),
            _ => {}
        }
    }
    Ok(())
}

fn canonical_regular_directory(directory: &Path) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(directory)
        .with_context(|| format!("inspecting GLM-5.3 GGUF directory {}", directory.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("GLM-5.3 GGUF directory must not be a symlink");
    }
    if !metadata.is_dir() {
        bail!("GLM-5.3 GGUF path must be a directory");
    }

    let canonical = fs::canonicalize(directory).context("canonicalizing GLM-5.3 GGUF directory")?;
    if !fs::symlink_metadata(&canonical)?.is_dir() {
        bail!("canonical GLM-5.3 GGUF path is not a directory");
    }
    Ok(canonical)
}

/// Resolves exact caller-pinned names without reading checkpoint payload bytes.
///
/// Non-GGUF sidecars are ignored. Every `.gguf` entry must be one of the four
/// expected regular files. The returned paths use a canonical, non-symlink
/// directory and remain ordered by shard number regardless of directory order.
fn resolve_exact_named_shards(
    directory: &Path,
    expected: &[&str; PARTS],
) -> Result<[PathBuf; PARTS]> {
    validate_expected_names(expected)?;
    let root = canonical_regular_directory(directory)?;
    let mut found: [Option<PathBuf>; PARTS] = std::array::from_fn(|_| None);

    for entry in fs::read_dir(&root).context("reading GLM-5.3 GGUF directory")? {
        let entry = entry.context("reading GLM-5.3 GGUF directory entry")?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("GLM-5.3 GGUF directory contains a non-UTF-8 name"))?;
        if !name.ends_with(".gguf") {
            continue;
        }

        let Some(index) = expected
            .iter()
            .position(|expected_name| *expected_name == name)
        else {
            bail!("GLM-5.3 GGUF directory contains an unexpected GGUF shard: {name}");
        };
        let path = root.join(&name);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("inspecting GLM-5.3 GGUF shard {name}"))?;
        if metadata.file_type().is_symlink() {
            bail!("GLM-5.3 GGUF shard must not be a symlink: {name}");
        }
        if !metadata.is_file() {
            bail!("GLM-5.3 GGUF shard must be a regular file: {name}");
        }
        if found[index].replace(path).is_some() {
            bail!("GLM-5.3 GGUF shard part {} is duplicated", index + 1);
        }
    }

    let [part_1, part_2, part_3, part_4] = found;
    Ok([
        part_1.context("GLM-5.3 GGUF shard part 1 is missing")?,
        part_2.context("GLM-5.3 GGUF shard part 2 is missing")?,
        part_3.context("GLM-5.3 GGUF shard part 3 is missing")?,
        part_4.context("GLM-5.3 GGUF shard part 4 is missing")?,
    ])
}

/// Resolve the exact filenames sealed by `profile` without reading payloads.
#[allow(dead_code)] // Wired into startup with the typed GGUF target constructor.
pub(crate) fn resolve_exact_glm53_shards(
    directory: &Path,
    profile: Glm53QuantProfile,
) -> Result<[PathBuf; PARTS]> {
    resolve_exact_named_shards(directory, profile.canonical_file_names())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "atlas-glm53-gguf-resolver-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn create_parts(&self, profile: Glm53QuantProfile) {
            for name in profile.canonical_file_names().iter().rev() {
                fs::write(self.0.join(name), []).unwrap();
            }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resolves_both_profiles_in_numeric_order() {
        for profile in [Glm53QuantProfile::UdQ2KXl, Glm53QuantProfile::UdIq3Xxs] {
            let directory = TestDirectory::new();
            directory.create_parts(profile);
            fs::write(directory.path().join("config.json"), b"sidecar").unwrap();

            let paths = resolve_exact_glm53_shards(directory.path(), profile).unwrap();
            let root = fs::canonicalize(directory.path()).unwrap();
            assert_eq!(
                paths,
                profile.canonical_file_names().map(|name| root.join(name))
            );
        }
    }

    #[test]
    fn rejects_missing_and_extra_gguf_parts() {
        let profile = Glm53QuantProfile::PRIMARY;
        let names = profile.canonical_file_names();
        let missing = TestDirectory::new();
        for name in &names[..3] {
            fs::write(missing.path().join(name), []).unwrap();
        }
        assert!(resolve_exact_glm53_shards(missing.path(), profile).is_err());

        let extra = TestDirectory::new();
        extra.create_parts(profile);
        fs::write(extra.path().join("alternate-00002-of-00004.gguf"), []).unwrap();
        assert!(resolve_exact_glm53_shards(extra.path(), profile).is_err());
    }

    #[test]
    fn rejects_non_regular_expected_part() {
        let profile = Glm53QuantProfile::PRIMARY;
        let names = profile.canonical_file_names();
        let directory = TestDirectory::new();
        for name in [names[0], names[2], names[3]] {
            fs::write(directory.path().join(name), []).unwrap();
        }
        fs::create_dir(directory.path().join(names[1])).unwrap();
        assert!(resolve_exact_glm53_shards(directory.path(), profile).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_root_and_expected_part() {
        use std::os::unix::fs::symlink;

        let profile = Glm53QuantProfile::PRIMARY;
        let names = profile.canonical_file_names();
        let directory = TestDirectory::new();
        directory.create_parts(profile);
        fs::remove_file(directory.path().join(names[1])).unwrap();
        symlink(names[0], directory.path().join(names[1])).unwrap();
        assert!(resolve_exact_glm53_shards(directory.path(), profile).is_err());

        let parent = TestDirectory::new();
        let linked = parent.path().join("linked");
        symlink(directory.path(), &linked).unwrap();
        assert!(resolve_exact_glm53_shards(&linked, profile).is_err());
    }

    #[test]
    fn rejects_unsealed_expected_name_sets_and_cross_profile_directories() {
        let profile = Glm53QuantProfile::PRIMARY;
        let names = *profile.canonical_file_names();
        let directory = TestDirectory::new();
        directory.create_parts(profile);

        let mut traversal = names;
        traversal[0] = "../GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf";
        assert!(resolve_exact_named_shards(directory.path(), &traversal).is_err());

        let mut wrong_part = names;
        wrong_part.swap(0, 1);
        assert!(resolve_exact_named_shards(directory.path(), &wrong_part).is_err());

        let mut mixed_stem = names;
        mixed_stem[3] = "other-00004-of-00004.gguf";
        assert!(resolve_exact_named_shards(directory.path(), &mixed_stem).is_err());

        assert!(resolve_exact_glm53_shards(directory.path(), Glm53QuantProfile::UdIq3Xxs).is_err());
    }
}
