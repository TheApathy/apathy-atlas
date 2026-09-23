// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../build_deps.rs"]
mod build_deps;

#[test]
fn glm_exl3_thin_compunit_tracks_the_shared_iq3_source() {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let exl3 = crate_dir.join("../../kernels/gb10/glm5.3-flash/exl3/glm53_kda.cu");
    let iq3 = crate_dir
        .join("../../kernels/gb10/glm5.3-flash/iq3/glm53_kda.cu")
        .canonicalize()
        .unwrap();
    let dependencies = build_deps::quoted_local_dependencies(&exl3).unwrap();
    assert!(dependencies.contains(&iq3));
}

#[test]
fn recursive_cycle_is_finite_and_each_dependency_is_unique() {
    let unique = format!(
        "atlas-kernel-deps-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir().join(unique);
    std::fs::create_dir(&root).unwrap();
    let a = root.join("a.cu");
    let b = root.join("b.cuh");
    std::fs::write(&a, "#include \"b.cuh\"\n").unwrap();
    std::fs::write(&b, "#include \"a.cu\"\n").unwrap();
    let dependencies = build_deps::quoted_local_dependencies(&a).unwrap();
    assert_eq!(dependencies, vec![b.canonicalize().unwrap()]);
    std::fs::remove_dir_all(root).unwrap();
}
