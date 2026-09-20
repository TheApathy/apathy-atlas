// SPDX-License-Identifier: AGPL-3.0-only

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Abi {
    U64,
    U32,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Grid {
    Nt48,
    Fixed4x48,
    Output,
}

#[derive(Clone, Copy, Debug)]
pub struct KernelSpec {
    pub role: &'static str,
    pub cache_dir: &'static str,
    pub function: &'static str,
    pub cubin_sha256: &'static str,
    pub cubin_bytes: u64,
    pub abi: &'static [Abi],
    pub grid: Grid,
    pub block_x: u32,
    pub dynamic_shared: u32,
    pub static_shared: i32,
    pub registers: i32,
    pub local_bytes: i32,
    pub stack_bytes: i32,
}

use Abi::{F32, U32, U64};

pub const CACHE_ROOT: &str = "/home/flocka/.cache/sglang/triton";
pub const NATIVE_ROOT: &str = "/home/flocka/atlas/src/native/gdn-sglang-c143";
pub const MANIFEST_FILE: &str = "frozen_triton_c143_manifest.json";
pub const MANIFEST_BYTES: u64 = 15_331;
pub const MANIFEST_SHA256: &str =
    "ecfa82b3f09651619953b8181f3e60bd613748d5e051cc2ee6c43d5874c61661";
pub const ARTIFACT_ATTESTATION_SHA256: &str =
    "993fdc906f53d8e342b04d8978ca5096fdcbea681d4d38280bcfe8ac215c295c";
pub const GATE_SOURCE_BUNDLE_SHA256: &str =
    "0047632a826e968bc3f17f6e50e6159b69febd914870b7ad5830563160cb4d76";
pub const EXECUTOR_SOURCE_BUNDLE_SHA256: &str =
    "7744aff43aadbaa0312ca9b8f96e36e246c3541bc0689fb971a2c8860b85b333";
pub const SGLANG_COMMIT: &str = "c14312a66420b75ca9a11bf1817c4db1fa26b097";
pub const SGLANG_LICENSE: &str = "Apache-2.0";
pub const TRITON_VERSION: &str = "3.6.0";
pub const TRITON_LICENSE: &str = "MIT";

const ABI_CUMSUM: &[Abi] = &[U64, U64, U64, U64, U32, U64, U64];
const ABI_KKT: &[Abi] = &[U64, U64, U64, U64, U64, U64, U32, U64, U64];
const ABI_WU: &[Abi] = &[U64, U64, U64, U64, U64, U64, U64, U64, U64, U32, U64, U64];
const ABI_STATE: &[Abi] = &[
    U64, U64, U64, U64, U64, U64, U64, U64, U32, U64, U64, U32, U64, U64,
];
const ABI_OUTPUT: &[Abi] = &[U64, U64, U64, U64, U64, U64, U64, U64, F32, U32, U64, U64];

pub const KERNELS: [KernelSpec; 5] = [
    KernelSpec {
        role: "cumsum",
        cache_dir: "N5Z6QMT3PX3KZXYSAYX75DRABTDASWQXND3DY4AYHY5IWVNAUSRQ",
        function: "chunk_local_cumsum_scalar_kernel",
        cubin_sha256: "c8eb8ec44fef0cabe53eb0f0c0a51f642106c9c4a727d200fc5c8fc4b972a309",
        cubin_bytes: 14_568,
        abi: ABI_CUMSUM,
        grid: Grid::Nt48,
        block_x: 256,
        dynamic_shared: 8,
        static_shared: 1_024,
        registers: 18,
        local_bytes: 0,
        stack_bytes: 0,
    },
    KernelSpec {
        role: "kkt_bc16_solve",
        cache_dir: "5PWPRAGLSBHYVO54HQIWYJGDRDVYXWKSFIEJBI5ZAPTL55BVE5NQ",
        function: "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
        cubin_sha256: "ec915e196cce992516def06b3aac5ba39e5d5ef28792a8450a3e6dc599c61df8",
        cubin_bytes: 262_976,
        abi: ABI_KKT,
        grid: Grid::Nt48,
        block_x: 32,
        dynamic_shared: 7_168,
        static_shared: 1_024,
        registers: 246,
        local_bytes: 0,
        stack_bytes: 0,
    },
    KernelSpec {
        role: "recompute_w_u",
        cache_dir: "CWPIHJJSBWG2DJ6LH3ZV3IDYTL2P6G64V6I4Y6NL55TILJM5S3DA",
        function: "recompute_w_u_fwd_kernel",
        cubin_sha256: "50874fa09cd96ce13a0d3075ff96661c685a9823e91943a77be7228f97ae7baa",
        cubin_bytes: 105_608,
        abi: ABI_WU,
        grid: Grid::Nt48,
        block_x: 128,
        dynamic_shared: 28_672,
        static_shared: 1_024,
        registers: 167,
        local_bytes: 0,
        stack_bytes: 0,
    },
    KernelSpec {
        role: "chunk_recurrence_state",
        cache_dir: "Q6BG3XVEHV2GEILZO5FGGEKKN2HU3PX44NLAPUIEBZEUO6RKX5WQ",
        function: "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        cubin_sha256: "ef641ac45c0918e3a84bb2ae1ed35c2126b980e5a786f55db8ca35a9edcaaa4c",
        cubin_bytes: 101_328,
        abi: ABI_STATE,
        grid: Grid::Fixed4x48,
        block_x: 128,
        dynamic_shared: 41_220,
        static_shared: 1_024,
        registers: 168,
        local_bytes: 0,
        stack_bytes: 0,
    },
    KernelSpec {
        role: "output",
        cache_dir: "KTFBEMQNV7CPTR5VWT4OH2W5U4AWJVLJ2QV3CEDL433KBK2JBWNA",
        function: "chunk_fwd_kernel_o",
        cubin_sha256: "ba130d76ac7ae5fc1892cc0a0d85b0ca92bc65d1c348719fad4baa4bc0f8cb9d",
        cubin_bytes: 128_632,
        abi: ABI_OUTPUT,
        grid: Grid::Output,
        block_x: 128,
        dynamic_shared: 18_432,
        static_shared: 1_024,
        registers: 150,
        local_bytes: 0,
        stack_bytes: 0,
    },
];

pub const GATE_SOURCES: &[&str] = &[
    "frozen_triton_c143_artifact.py",
    "frozen_triton_c143_attest.py",
    "frozen_triton_c143_authorization.py",
    "frozen_triton_c143_constants.py",
    "frozen_triton_c143_gate.py",
    "frozen_triton_c143_io.py",
    "frozen_triton_c143_layout.py",
    "frozen_triton_c143_manifest_policy.py",
    "frozen_triton_c143_provenance.py",
    "frozen_triton_c143_receipt.py",
    "frozen_triton_c143_receipt_metrics.py",
];

pub const EXECUTOR_SOURCES: &[&str] = &[
    "run_frozen_triton_c143.py",
    "frozen_triton_executor/__init__.py",
    "frozen_triton_executor/contract.py",
    "frozen_triton_executor/verified_loader.py",
    "frozen_triton_executor/cuda_driver.py",
    "frozen_triton_executor/buffers.py",
    "frozen_triton_executor/launch.py",
    "frozen_triton_executor/references.py",
    "frozen_triton_executor/case.py",
    "frozen_triton_executor/timing.py",
    "frozen_triton_executor/evidence.py",
    "frozen_triton_executor/receipt.py",
    "frozen_triton_executor/publication.py",
    "frozen_triton_executor/main.py",
];

pub const SAME_STREAM_SEQUENCE: &[&str] = &[
    "adapter_qkv_split",
    "adapter_alpha_log_beta_split",
    "adapter_state_hkv_to_hvk",
    "memset_A_zero",
    "memset_output_zero",
    "chunk_local_cumsum_scalar_kernel",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
    "adapter_state_hvk_to_hkv",
    "completion_marker",
];
