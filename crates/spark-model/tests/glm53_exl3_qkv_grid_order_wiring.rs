// SPDX-License-Identifier: AGPL-3.0-only

const HOST: &str = include_str!("../src/layers/ops/glm53_exl3.rs");
const CUDA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_kda_qkv_row_x.cu"
));
const MANIFEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/gb10/glm5.3-flash/exl3/KERNEL.toml"
));

#[test]
fn qkv_row_x_grid_order_is_strict_default_off_and_swaps_only_slice_indices() {
    assert!(HOST.contains("ATLAS_GLM53_EXL3_KDA_QKV_GRID_ORDER"));
    assert!(HOST.contains("Glm53Exl3StagedQkvGridOrder"));
    assert!(HOST.contains("[row_tiles, 1, GB10_MULTIPROCESSORS]"));
    assert!(CUDA.contains("atlas_original_block_idx"));
    assert!(CUDA.contains("atlas_original_grid_dim"));
    assert!(CUDA.contains("#define blockIdx atlas_swapped_block_idx()"));
    assert!(CUDA.contains("#define gridDim atlas_swapped_grid_dim()"));
    assert!(CUDA.contains("const int row = 16 * blockIdx.x"));
    assert!(CUDA.contains("locks + blockIdx.x * lock_stride"));
    assert!(MANIFEST.contains("glm53_exl3_kda_qkv_row_x = \"glm53_exl3_kda_qkv_row_x\""));
}
