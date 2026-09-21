// SPDX-License-Identifier: AGPL-3.0-only

//! Compare resolved checkpoint directories, independent of CLI spelling.

use std::path::Path;

pub(super) fn same_checkpoint(target: &Path, draft: &Path) -> std::io::Result<bool> {
    Ok(target.canonicalize()? == draft.canonicalize()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_target_path_and_symlink_share_the_store() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("target");
        std::fs::create_dir(&target).unwrap();
        assert!(same_checkpoint(&target, &target.join(".")).unwrap());
        #[cfg(unix)]
        {
            let link = root.path().join("draft");
            std::os::unix::fs::symlink(&target, &link).unwrap();
            assert!(same_checkpoint(&target, &link).unwrap());
        }
    }

    #[test]
    fn different_or_unresolved_directories_never_share() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        assert!(!same_checkpoint(a.path(), b.path()).unwrap());
        assert!(same_checkpoint(a.path(), &a.path().join("absent")).is_err());
    }

    #[test]
    fn model_from_path_is_bound_through_resolved_target_not_positional_arg() {
        let source = include_str!("../weights.rs");
        assert!(source.contains("identity::same_checkpoint(target_dir, &drafter_dir)?"));
        assert!(!source.contains("let shares_target_dir = args"));
        let serve = include_str!("../../serve.rs");
        let call = serve
            .split("serve_phases::load_dflash_drafter(")
            .nth(1)
            .unwrap();
        assert!(call.split(")?").next().unwrap().contains("&model_dir"));
    }
}
