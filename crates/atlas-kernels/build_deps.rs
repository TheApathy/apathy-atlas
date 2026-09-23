// SPDX-License-Identifier: AGPL-3.0-only

//! Recursive local quoted-include discovery for kernel build invalidation.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub(crate) fn quoted_local_dependencies(root: &Path) -> Result<Vec<PathBuf>, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot resolve {}: {error}", root.display()))?;
    let mut seen = HashSet::from([root.clone()]);
    let mut stack = vec![root];
    let mut dependencies = Vec::new();
    while let Some(source) = stack.pop() {
        let text = std::fs::read_to_string(&source)
            .map_err(|error| format!("cannot read {}: {error}", source.display()))?;
        for include in text.lines().filter_map(quoted_include) {
            let Some(parent) = source.parent() else {
                continue;
            };
            let candidate = parent.join(include);
            if !candidate.is_file() {
                continue;
            }
            let dependency = candidate
                .canonicalize()
                .map_err(|error| format!("cannot resolve {}: {error}", candidate.display()))?;
            if seen.insert(dependency.clone()) {
                dependencies.push(dependency.clone());
                stack.push(dependency);
            }
        }
    }
    dependencies.sort();
    Ok(dependencies)
}

fn quoted_include(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#include")?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}
