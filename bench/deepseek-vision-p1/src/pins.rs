// SPDX-License-Identifier: AGPL-3.0-only
pub const OFFICIAL_SHA: &str = "a4f089069310398d42ca17fd4496cec82da64cbbfde9b0230679ce1537cc0bb1";
pub const OFFICIAL_PATH: &str = "/var/tmp/atlas-deepseek-vision-oracle.ONOuQa/vision.py";
pub const OFFICIAL_REV: &str = "6821d6ad3681a4b137b066b76094fa82ebd0a380";
pub const MODEL_REV: &str = "c171bea574201ff25530256fbd63626c7fd20f3c";
pub const REFERENCE_SCRIPTS: [(&str, &str); 2] = [
    (
        "/var/tmp/atlas-deepseek-vision-oracle.ONOuQa/compare_reference.py",
        "208dff152a714bbd8c18a554a1dbb9700d898c7d041d20fa7a86ad2945f482ca",
    ),
    (
        "/var/tmp/atlas-deepseek-vision-oracle.ONOuQa/compare_stages.py",
        "60a660e9174c6e2508c843dd60b8285e74b8041649bbc76c1de024797fd895ef",
    ),
];
pub const PAYLOAD_SHA: &str = "fc9d54d790826fb7d8f60b06c8ae17a45b3cade7bfee44e7ae3f74dc948922b0";
pub const CONFIG_SHA: &str = "28a07138554196d7de70cfb193eb63bf51c39bb42ae4cd4303ba16610b5b1bf5";
pub const INDEX_SHA: &str = "f4df075b9b9d77af5fe1482624a33466a7b5418f96c9d31f53339c34d72338d8";
pub const CORPUS_FILES: [(&str, &str); 4] = [
    (
        "atlas-manifest.json",
        "caff41128f7e6266d3127e51d3a4aca6a0a6c9237559b08851e1a094c214e660",
    ),
    (
        "reference-cuda-math.json",
        "2a32cf81b2e22bea22d6cac362587bbc2a6463904e4310a1e160235f6ddc58ae",
    ),
    (
        "stage-reference-cuda-math.json",
        "ab4f3c0581f90fe5213122a6ce021db6be402374c054bda235c1f735dc66cccb",
    ),
    (
        "stage-reference-cuda-math-bf16-full.json",
        "b0184de244f7402355bd30cfeb21496bda1c92a027f3bea8761ead0e52c08742",
    ),
];
pub const STAGE_MANIFESTS: [&str; 3] = [
    "a0137d3b3f39a944a59ca58d5c1de163483ec426b0a394cf1ac24710ec91b260",
    "53373392894998078138f73e23b37c9eaf0ccb2720b7a4db17fc006ed59d2bfa",
    "971161922e32b57fa87140abaae3d333981166950af95e2eba266d6464dd1f2d",
];
pub const PTX_ROOT: &str =
    "/var/tmp/atlas-deepseek-vision-v6/release/build/atlas-kernels-f0fea15102ef8dc1/out";
pub const PTX_FILES: [(&str, &str); 2] = [
    (
        "t0__deepseek_vision.ptx",
        "4a095166a9c52dce37a80513354d767d9ef0bf7198cf303cf3d28240884a2e25",
    ),
    (
        "t0__deepseek_vision_gemm.ptx",
        "9c06771d350de7ad5a2cf6a01b4ce7244f6bd0a1a00cb327a39024d4d0774669",
    ),
];
pub const PRODUCTION_FILES: [(&str, &str); 5] = [
    (
        "crates/spark-model/src/layers/deepseek_vision/geometry.rs",
        "faa3e794ea41e82ec4a93427390c4dc68ed905f2ec3aa57c24ec157591339a69",
    ),
    (
        "crates/spark-model/src/layers/deepseek_vision/forward.rs",
        "4c8d8b1fb859b28c592407cb257221c5a26bce6476473bba9586f00036f12c55",
    ),
    (
        "kernels/gb10/deepseek-v4-flash/nvfp4/deepseek_vision.cu",
        "7b8a1e678aef500c8d926e1c8f42621d6e618f7d84518fb7e8a7529e3e2c602d",
    ),
    (
        "kernels/gb10/deepseek-v4-flash/nvfp4/deepseek_vision_gemm.cu",
        "f89ff2b9373189e76c722e279d23b140b8a99c51e8c8a1cb42c4b16f5acc7cb4",
    ),
    (
        "crates/spark-runtime/src/cublaslt.rs",
        "e1784cdbc6280c13916b5b24069b4b2714fe7b77701539b78cb0208b475e200e",
    ),
];
pub const FC2_REFERENCE: &str = "e7b96bbc3bdb69bdb3ed3a34a7340cc4fd3675782b77ec944b3f0ea2a9b48d00";
pub const FC2_WEIGHT: &str = "vision.blocks.0.mlp.w2.weight";
pub const SOURCE_FILES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "src/lib.rs",
    "src/main.rs",
    "src/contract.rs",
    "src/host_angles.rs",
    "src/pins.rs",
    "src/io.rs",
    "src/admission.rs",
    "src/weight.rs",
    "src/driver.rs",
    "src/cuda_abi.rs",
    "src/rope.rs",
    "src/fc2.rs",
    "src/lt.rs",
    "src/lt_abi.rs",
    "angles.cu",
    "tests/contract.rs",
    "tests/host_angles.rs",
    "tests/inspection.rs",
    "README.md",
];
