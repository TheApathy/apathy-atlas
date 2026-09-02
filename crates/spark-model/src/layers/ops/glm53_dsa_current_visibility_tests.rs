// SPDX-License-Identifier: AGPL-3.0-only

use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};

use super::*;

const HOST: &str = include_str!("glm53_dsa_current_visibility.rs");
const CUDA: &str =
    include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/glm53_dsa_current_visibility.cu");

fn plan(position: u32) -> Glm53DsaCurrentVisibilityPlan {
    Glm53DsaCurrentVisibilityPlan::new(
        1_048_576,
        262_144,
        position,
        position + 1,
        11,
        512,
        128,
        4,
        7,
        9,
    )
    .unwrap()
}

#[test]
fn plan_pins_all_boundaries_and_only_complete_pool_groups() {
    for position in [0, 1, 2, 3, 1_048_575] {
        let plan = plan(position);
        assert_eq!(plan.end_position, position + 1);
        assert_eq!(plan.pool_row, position / 4);
        assert_eq!(plan.writes_pool, position % 4 == 3);
        assert_eq!(plan.latent_overlay_bytes, 11_264);
        assert_eq!(
            plan.pool_overlay_bytes,
            if position % 4 == 3 { 2_816 } else { 0 }
        );
        assert_eq!(
            plan.pool_overlay_validity_bytes,
            if position % 4 == 3 { 11 } else { 0 }
        );
        assert_eq!(plan.persistent_latent_bytes, 11_811_160_064);
        assert_eq!(plan.persistent_pool_bytes, 738_197_504);
        assert_eq!(plan.persistent_pool_validity_bytes, 2_883_584);
        assert_eq!((plan.ends_bytes, plan.generations_bytes), (44, 88));
        assert_eq!((plan.nonces_bytes, plan.statuses_bytes), (88, 44));
    }
    for hostile in [
        Glm53DsaCurrentVisibilityPlan::new(0, 0, 0, 1, 11, 512, 128, 4, 1, 1),
        Glm53DsaCurrentVisibilityPlan::new(
            1_048_577, 262_145, 1_048_575, 1_048_576, 11, 512, 128, 4, 1, 1,
        ),
        Glm53DsaCurrentVisibilityPlan::new(4, 1, 4, 5, 11, 512, 128, 4, 1, 1),
        Glm53DsaCurrentVisibilityPlan::new(4, 1, 3, 3, 11, 512, 128, 4, 1, 1),
        Glm53DsaCurrentVisibilityPlan::new(4, 2, 3, 4, 11, 512, 128, 4, 1, 1),
        Glm53DsaCurrentVisibilityPlan::new(4, 1, 3, 4, 10, 512, 128, 4, 1, 1),
        Glm53DsaCurrentVisibilityPlan::new(4, 1, 3, 4, 11, 511, 128, 4, 1, 1),
        Glm53DsaCurrentVisibilityPlan::new(4, 1, 3, 4, 11, 512, 127, 4, 1, 1),
        Glm53DsaCurrentVisibilityPlan::new(4, 1, 3, 4, 11, 512, 128, 3, 1, 1),
        Glm53DsaCurrentVisibilityPlan::new(4, 1, 3, 4, 11, 512, 128, 4, 0, 1),
        Glm53DsaCurrentVisibilityPlan::new(4, 1, 3, 4, 11, 512, 128, 4, 1, 0),
    ] {
        assert!(hostile.is_err());
    }
    for capacity in [1, 2, 3, 5, 1_048_575] {
        let position = capacity - 1;
        let pool_capacity = capacity / 4;
        let plan = Glm53DsaCurrentVisibilityPlan::new(
            capacity,
            pool_capacity,
            position,
            capacity,
            11,
            512,
            128,
            4,
            1,
            1,
        )
        .unwrap();
        assert_eq!(plan.pool_capacity, pool_capacity);
        assert_eq!(
            plan.persistent_pool_bytes,
            pool_capacity as usize * 11 * 128 * 2
        );
        assert_eq!(
            plan.persistent_pool_validity_bytes,
            pool_capacity as usize * 11
        );
        assert!(!plan.writes_pool);
        let mut overpooled = plan;
        overpooled.pool_capacity += 1;
        assert!(overpooled.validate().is_err());
    }
}

fn materialize_reference(
    position: usize,
    capacity: usize,
    latent_overlay: &[bf16],
    pool_overlay: Option<&[bf16]>,
    persistent_latent: &mut [bf16],
    persistent_pool: &mut [bf16],
    pool_validity: &mut [u8],
) {
    for layer in 0..11 {
        let source = layer * 512;
        let destination = (layer * capacity + position) * 512;
        persistent_latent[destination..destination + 512]
            .copy_from_slice(&latent_overlay[source..source + 512]);
    }
    if position % 4 == 3 {
        let pool_capacity = capacity / 4;
        let pool_row = position / 4;
        let overlay = pool_overlay.unwrap();
        for layer in 0..11 {
            let source = layer * 128;
            let destination = (layer * pool_capacity + pool_row) * 128;
            persistent_pool[destination..destination + 128]
                .copy_from_slice(&overlay[source..source + 128]);
            pool_validity[layer * pool_capacity + pool_row] = 1;
        }
    }
}

#[test]
fn cpu_oracle_writes_only_future_rows_and_rollback_hides_bytes() {
    let capacity = 4;
    let latent_overlay: Vec<bf16> = (0..11 * 512)
        .map(|index| bf16::from_f32(index as f32 + 1.0))
        .collect();
    let pool_overlay: Vec<bf16> = (0..11 * 128)
        .map(|index| bf16::from_f32(index as f32 - 3.0))
        .collect();
    let zero = bf16::from_f32(0.0);
    let mut latent = vec![zero; 11 * capacity * 512];
    let mut pool = vec![zero; 11 * (capacity / 4) * 128];
    let mut validity = vec![0; 11 * (capacity / 4)];
    let logical_lengths = vec![3u32; 11];
    materialize_reference(
        3,
        capacity,
        &latent_overlay,
        Some(&pool_overlay),
        &mut latent,
        &mut pool,
        &mut validity,
    );
    assert_eq!(&latent[3 * 512..4 * 512], &latent_overlay[..512]);
    assert!(latent[..3 * 512].iter().all(|value| *value == zero));
    assert_eq!(&pool[..128], &pool_overlay[..128]);
    assert_eq!(validity, vec![1; 11]);
    assert_eq!(logical_lengths, vec![3; 11]);
    let mut ready_ends = vec![4; 11];
    let mut ready_generations = vec![7; 11];
    let mut ready_statuses = vec![0; 11];
    let mut ready_nonces = vec![9; 11];
    ready_nonces.fill(0);
    ready_statuses.fill(u32::MAX);
    ready_ends.fill(0);
    ready_generations.fill(0);
    assert_eq!(logical_lengths, vec![3; 11]);
    assert!(ready_nonces.iter().all(|nonce| *nonce == 0));
    assert_eq!(&latent[3 * 512..4 * 512], &latent_overlay[..512]);
    let mut short_latent = vec![zero; 11 * 3 * 512];
    materialize_reference(
        2,
        3,
        &latent_overlay,
        None,
        &mut short_latent,
        &mut [],
        &mut [],
    );
    assert_eq!(&short_latent[2 * 512..3 * 512], &latent_overlay[..512]);
}

fn buffer(gpu: &MockGpuBackend, bytes: usize) -> GgmlIqBuffer {
    if bytes == 0 {
        return GgmlIqBuffer {
            ptr: DevicePtr::NULL,
            bytes: 0,
        };
    }
    GgmlIqBuffer {
        ptr: gpu.alloc(bytes).unwrap(),
        bytes,
    }
}

fn buffers(
    gpu: &MockGpuBackend,
    plan: Glm53DsaCurrentVisibilityPlan,
) -> Glm53DsaCurrentVisibilityBuffers {
    Glm53DsaCurrentVisibilityBuffers {
        latent_overlay_bf16: buffer(gpu, plan.latent_overlay_bytes),
        pool_overlay_bf16: buffer(gpu, plan.pool_overlay_bytes),
        pool_overlay_validity_u8: buffer(gpu, plan.pool_overlay_validity_bytes),
        staged_ends_u32: buffer(gpu, plan.ends_bytes),
        staged_generations_u64: buffer(gpu, plan.generations_bytes),
        staged_nonces_u64: buffer(gpu, plan.nonces_bytes),
        staged_statuses_u32: buffer(gpu, plan.statuses_bytes),
        persistent_latent_bf16: buffer(gpu, plan.persistent_latent_bytes),
        persistent_pool_bf16: buffer(gpu, plan.persistent_pool_bytes),
        persistent_pool_validity_u8: buffer(gpu, plan.persistent_pool_validity_bytes),
        ready_ends_u32: buffer(gpu, plan.ends_bytes),
        ready_generations_u64: buffer(gpu, plan.generations_bytes),
        ready_statuses_u32: buffer(gpu, plan.statuses_bytes),
        ready_nonces_u64: buffer(gpu, plan.nonces_bytes),
    }
}

#[test]
fn host_rejects_forgery_extent_alias_alignment_null_and_address_overflow() {
    let gpu = MockGpuBackend::new();
    let kernel = Glm53DsaCurrentVisibilityKernel::load(&gpu).unwrap();
    let plan = Glm53DsaCurrentVisibilityPlan::new(4, 1, 3, 4, 11, 512, 128, 4, 7, 9).unwrap();
    let valid = buffers(&gpu, plan);
    let mut forged = plan;
    forged.pool_row = 2;
    assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
    for hostile in [
        Glm53DsaCurrentVisibilityBuffers {
            latent_overlay_bf16: GgmlIqBuffer {
                bytes: valid.latent_overlay_bf16.bytes - 2,
                ..valid.latent_overlay_bf16
            },
            ..valid
        },
        Glm53DsaCurrentVisibilityBuffers {
            persistent_pool_bf16: GgmlIqBuffer {
                ptr: valid.persistent_latent_bf16.ptr,
                ..valid.persistent_pool_bf16
            },
            ..valid
        },
        Glm53DsaCurrentVisibilityBuffers {
            staged_generations_u64: GgmlIqBuffer {
                ptr: DevicePtr(valid.staged_generations_u64.ptr.0 + 4),
                ..valid.staged_generations_u64
            },
            ..valid
        },
        Glm53DsaCurrentVisibilityBuffers {
            staged_statuses_u32: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                ..valid.staged_statuses_u32
            },
            ..valid
        },
        Glm53DsaCurrentVisibilityBuffers {
            latent_overlay_bf16: GgmlIqBuffer {
                ptr: DevicePtr(u64::MAX - 1),
                ..valid.latent_overlay_bf16
            },
            ..valid
        },
    ] {
        assert!(kernel.launch(&gpu, plan, hostile, 0).is_err());
    }
    assert_eq!(gpu.launch_count(), 0);
    kernel.launch(&gpu, plan, valid, 0).unwrap();
    assert_eq!(gpu.launch_count(), 1);
    kernel.rollback_ready(&gpu, plan, valid, 0).unwrap();
    assert_eq!(gpu.launch_count(), 1);

    let zero_pool_plan =
        Glm53DsaCurrentVisibilityPlan::new(3, 0, 2, 3, 11, 512, 128, 4, 7, 9).unwrap();
    let zero_pool = buffers(&gpu, zero_pool_plan);
    assert_eq!(zero_pool.persistent_pool_bf16.ptr, DevicePtr::NULL);
    assert_eq!(zero_pool.persistent_pool_validity_u8.ptr, DevicePtr::NULL);
    kernel.launch(&gpu, zero_pool_plan, zero_pool, 0).unwrap();
    assert_eq!(gpu.launch_count(), 2);
    let unexpected_persistent_pool = Glm53DsaCurrentVisibilityBuffers {
        persistent_pool_bf16: buffer(&gpu, 2),
        ..zero_pool
    };
    assert!(
        kernel
            .launch(&gpu, zero_pool_plan, unexpected_persistent_pool, 0)
            .is_err()
    );

    let no_pool_plan =
        Glm53DsaCurrentVisibilityPlan::new(4, 1, 2, 3, 11, 512, 128, 4, 7, 9).unwrap();
    let no_pool = buffers(&gpu, no_pool_plan);
    let unexpected_pool = Glm53DsaCurrentVisibilityBuffers {
        pool_overlay_bf16: buffer(&gpu, 2),
        ..no_pool
    };
    assert!(
        kernel
            .launch(&gpu, no_pool_plan, unexpected_pool, 0)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 2);
}

fn ordered(source: &str, terms: &[&str]) -> bool {
    let mut offset = 0;
    terms.iter().all(|term| {
        let Some(found) = source[offset..].find(term) else {
            return false;
        };
        offset += found + term.len();
        true
    })
}

fn swap_unique(source: &str, left: &str, right: &str) -> String {
    assert_eq!(source.matches(left).count(), 1, "ambiguous left ABI token");
    assert_eq!(
        source.matches(right).count(),
        1,
        "ambiguous right ABI token"
    );
    source
        .replace(left, "__GLM53_DSA_VIS_LEFT__")
        .replace(right, left)
        .replace("__GLM53_DSA_VIS_LEFT__", right)
}

fn abi_contract(host: &str, cuda: &str) -> bool {
    const HOST_ABI: &str = ".arg_ptr(buffers.latent_overlay_bf16.ptr)\n            .arg_ptr(buffers.pool_overlay_bf16.ptr)\n            .arg_ptr(buffers.pool_overlay_validity_u8.ptr)\n            .arg_ptr(buffers.staged_ends_u32.ptr)\n            .arg_ptr(buffers.staged_generations_u64.ptr)\n            .arg_ptr(buffers.staged_nonces_u64.ptr)\n            .arg_ptr(buffers.staged_statuses_u32.ptr)\n            .arg_ptr(buffers.persistent_latent_bf16.ptr)\n            .arg_ptr(buffers.persistent_pool_bf16.ptr)\n            .arg_ptr(buffers.persistent_pool_validity_u8.ptr)\n            .arg_ptr(buffers.ready_ends_u32.ptr)\n            .arg_ptr(buffers.ready_generations_u64.ptr)\n            .arg_ptr(buffers.ready_statuses_u32.ptr)\n            .arg_ptr(buffers.ready_nonces_u64.ptr)\n            .arg_u32(plan.capacity)\n            .arg_u32(plan.pool_capacity)\n            .arg_u32(plan.logical_position)\n            .arg_u32(plan.end_position)\n            .arg_u64(plan.generation)\n            .arg_u64(plan.transaction_nonce)";
    const CUDA_ABI: &str = "const __nv_bfloat16 * __restrict__ latent_overlay,\n        const __nv_bfloat16 * __restrict__ pool_overlay,\n        const unsigned char * __restrict__ pool_overlay_validity,\n        const unsigned int * __restrict__ staged_ends,\n        const unsigned long long * __restrict__ staged_generations,\n        const unsigned long long * __restrict__ staged_nonces,\n        const unsigned int * __restrict__ staged_statuses,\n        __nv_bfloat16 * __restrict__ persistent_latent,\n        __nv_bfloat16 * __restrict__ persistent_pool,\n        unsigned char * __restrict__ persistent_pool_validity,\n        unsigned int * __restrict__ ready_ends,\n        unsigned long long * __restrict__ ready_generations,\n        unsigned int * __restrict__ ready_statuses,\n        unsigned long long * __restrict__ ready_nonces,\n        unsigned int capacity, unsigned int pool_capacity,\n        unsigned int logical_position, unsigned int end_position,\n        unsigned long long generation, unsigned long long transaction_nonce";
    host.contains(HOST_ABI) && cuda.contains(CUDA_ABI)
}

#[test]
fn ordered_host_cuda_abi_rejects_pointer_and_scalar_swaps() {
    assert!(abi_contract(HOST, CUDA));
    for (left, right) in [
        (
            ".arg_ptr(buffers.latent_overlay_bf16.ptr)",
            ".arg_ptr(buffers.staged_ends_u32.ptr)",
        ),
        (
            ".arg_ptr(buffers.staged_nonces_u64.ptr)",
            ".arg_ptr(buffers.ready_nonces_u64.ptr)",
        ),
        (
            ".arg_ptr(buffers.persistent_pool_bf16.ptr)",
            ".arg_ptr(buffers.ready_statuses_u32.ptr)",
        ),
        (".arg_u32(plan.capacity)", ".arg_u32(plan.end_position)"),
        (
            ".arg_u32(plan.logical_position)",
            ".arg_u64(plan.generation)",
        ),
    ] {
        assert!(!abi_contract(&swap_unique(HOST, left, right), CUDA));
    }
    for (left, right) in [
        (
            "const __nv_bfloat16 * __restrict__ latent_overlay,",
            "const unsigned int * __restrict__ staged_ends,",
        ),
        (
            "const unsigned long long * __restrict__ staged_nonces,",
            "unsigned long long * __restrict__ ready_nonces,",
        ),
        (
            "__nv_bfloat16 * __restrict__ persistent_pool,",
            "unsigned int * __restrict__ ready_statuses,",
        ),
        ("unsigned int capacity", "unsigned int end_position"),
        (
            "unsigned int logical_position",
            "unsigned long long generation",
        ),
    ] {
        assert!(!abi_contract(HOST, &swap_unique(CUDA, left, right)));
    }
}

#[rustfmt::skip]
#[test]
fn source_contract_rejects_address_status_barrier_and_order_mutations() {
    const REQUIRED: &[&str] = &[
        "staged_ends[layer] != end_position",
        "staged_generations[layer] != generation",
        "staged_nonces[layer] != transaction_nonce",
        "staged_statuses[layer] != 0U",
        "pool_overlay_validity[layer] != 1U",
        "if (invalid_transaction != 0U)",
        "(layer * capacity + logical_position) * GLM53_DSA_VIS_LATENT + feature;\n        persistent_latent[destination] = latent_overlay[item];",
        "(layer * pool_capacity + pool_row) * GLM53_DSA_VIS_INDEX + feature;\n            persistent_pool[destination] = pool_overlay[item];",
        "ready_ends[layer] = end_position",
        "ready_generations[layer] = generation",
        "ready_statuses[layer] = 0U",
        "atomicExch(ready_nonces + layer, transaction_nonce)",
    ];
    const EXACT: &[&str] = &[
        "#define GLM53_DSA_VIS_LAYERS 11U",
        "#define GLM53_DSA_VIS_LATENT 512U", "#define GLM53_DSA_VIS_INDEX 128U", "#define GLM53_DSA_VIS_KPOOL 4U", "#define GLM53_DSA_VIS_MAX_POSITIONS 1048576U", "#define GLM53_DSA_VIS_THREADS 256U",
        "logical_position % GLM53_DSA_VIS_KPOOL == GLM53_DSA_VIS_KPOOL - 1U",
        "const unsigned int computed_pool_capacity =\n        capacity / GLM53_DSA_VIS_KPOOL;",
        "(pool_capacity == 0U &&\n         (persistent_pool != nullptr || persistent_pool_validity != nullptr)) ||\n        (pool_capacity != 0U &&\n         (persistent_pool == nullptr || persistent_pool_validity == nullptr))",
        "(writes_pool && (pool_overlay == nullptr ||\n                         pool_overlay_validity == nullptr))",
        "blockDim.x != GLM53_DSA_VIS_THREADS",
        "blockDim.y != 1U",
        "blockDim.z != 1U",
        "gridDim.x != 1U",
        "gridDim.y != 1U",
        "gridDim.z != 1U",
        "const unsigned int lane = threadIdx.x;",
        "const unsigned long long latent_items =\n        (unsigned long long)GLM53_DSA_VIS_LAYERS * GLM53_DSA_VIS_LATENT;",
        "const unsigned long long layer = item / GLM53_DSA_VIS_LATENT;",
        "const unsigned long long feature = item % GLM53_DSA_VIS_LATENT;",
        "const unsigned long long pool_items =\n            (unsigned long long)GLM53_DSA_VIS_LAYERS * GLM53_DSA_VIS_INDEX;",
        "const unsigned long long layer = item / GLM53_DSA_VIS_INDEX;",
        "const unsigned long long feature = item % GLM53_DSA_VIS_INDEX;",
        "persistent_pool_validity[\n                (unsigned long long)layer * pool_capacity + pool_row] = 1U;\n        }\n    }\n    __syncthreads();\n    __threadfence_system();\n    __syncthreads();\n    for (unsigned int layer = lane;",
        "ready_statuses[layer] = 0U;\n    }\n    __syncthreads();\n    __threadfence_system();\n    __syncthreads();\n    for (unsigned int layer = lane;",
    ];
    fn contract(host: &str, cuda: &str) -> bool {
        let positions: Vec<_> = REQUIRED.iter().map(|term| cuda.find(term)).collect();
        let host_launch = [
            "plan.validate()?;",
            "validate_buffers(plan, buffers)?;",
            "clear_ready(gpu, plan, buffers, stream)?;",
            "KernelLaunch::new(gpu, self.materialize)\n            .grid([1, 1, 1])\n            .block([THREADS, 1, 1])",
        ];
        let clear = &host[host.find("fn clear_ready(").unwrap()..host.find("fn validate_buffers(").unwrap()];
        REQUIRED.iter().all(|term| cuda.matches(term).count() == 1)
            && EXACT.iter().all(|term| cuda.matches(term).count() == 1)
            && positions.iter().all(Option::is_some)
            && positions.windows(2).all(|pair| pair[0] < pair[1])
            && ordered(host, &host_launch)
            && cuda.matches("item += GLM53_DSA_VIS_THREADS").count() == 2
            && cuda.matches("layer += GLM53_DSA_VIS_THREADS").count() == 4
            && cuda.matches("__syncthreads();").count() == 6
            && cuda.matches("__threadfence_system();").count() == 2
            && cuda.matches("ready_statuses[layer] = 0U").count() == 1
            && ordered(clear, &["ready_nonces_u64.ptr", "ready_statuses_u32.ptr", "ready_ends_u32.ptr", "ready_generations_u64.ptr"])
            && host.matches("const THREADS: u32 = 256;").count() == 1
            && !host.contains("logical_lengths_u32")
            && !cuda.contains("logical_lengths")
    }
    assert!(contract(HOST, CUDA));
    for token in REQUIRED {
        assert!(!contract(HOST, &CUDA.replacen(token, "", 1)));
    }
    for token in ["__syncthreads();", "__threadfence_system();"] {
        assert!(!contract(HOST, &CUDA.replacen(token, "", 1)));
    }
    for (from, to) in [
        ("#define GLM53_DSA_VIS_LAYERS 11U", "#define GLM53_DSA_VIS_LAYERS 10U"),
        ("#define GLM53_DSA_VIS_LATENT 512U", "#define GLM53_DSA_VIS_LATENT 128U"), ("#define GLM53_DSA_VIS_INDEX 128U", "#define GLM53_DSA_VIS_INDEX 512U"), ("#define GLM53_DSA_VIS_KPOOL 4U", "#define GLM53_DSA_VIS_KPOOL 3U"), ("#define GLM53_DSA_VIS_MAX_POSITIONS 1048576U", "#define GLM53_DSA_VIS_MAX_POSITIONS 1048575U"), ("#define GLM53_DSA_VIS_THREADS 256U", "#define GLM53_DSA_VIS_THREADS 128U"),
        ("const unsigned int lane = threadIdx.x;", "const unsigned int lane = threadIdx.y;"),
        ("const unsigned int lane = threadIdx.x;", "const unsigned int lane = threadIdx.z;"),
        ("blockDim.x != GLM53_DSA_VIS_THREADS", "false"),
        ("blockDim.y != 1U", "false"),
        ("blockDim.z != 1U", "false"),
        ("gridDim.x != 1U", "false"),
        ("gridDim.y != 1U", "false"),
        ("gridDim.z != 1U", "false"),
        ("item += GLM53_DSA_VIS_THREADS", "item += GLM53_DSA_VIS_LATENT"),
        ("layer += GLM53_DSA_VIS_THREADS", "layer += GLM53_DSA_VIS_LAYERS"),
        ("capacity / GLM53_DSA_VIS_KPOOL", "(capacity + 3U) / GLM53_DSA_VIS_KPOOL"),
        ("layer * capacity + logical_position", "logical_position * capacity + layer"),
        ("layer * pool_capacity + pool_row", "pool_row * pool_capacity + layer"),
        ("staged_statuses[layer] != 0U", "staged_statuses[layer] != 1U"),
        ("ready_statuses[layer] = 0U", "ready_statuses[layer] = 1U"),
        ("logical_position % GLM53_DSA_VIS_KPOOL == GLM53_DSA_VIS_KPOOL - 1U", "logical_position % GLM53_DSA_VIS_KPOOL == GLM53_DSA_VIS_KPOOL - 2U"),
        ("persistent_latent[destination] = latent_overlay[item];", "persistent_latent[destination] = latent_overlay[feature];"),
        ("persistent_pool[destination] = pool_overlay[item];", "persistent_pool[destination] = pool_overlay[feature];"),
        ("(writes_pool && (pool_overlay == nullptr ||\n                         pool_overlay_validity == nullptr))", "false"),
        ("GLM53_DSA_VIS_LAYERS * GLM53_DSA_VIS_LATENT", "GLM53_DSA_VIS_LAYERS * GLM53_DSA_VIS_INDEX"),
        ("item / GLM53_DSA_VIS_LATENT", "item / GLM53_DSA_VIS_INDEX"),
        ("item % GLM53_DSA_VIS_LATENT", "item % GLM53_DSA_VIS_INDEX"),
        ("GLM53_DSA_VIS_LAYERS * GLM53_DSA_VIS_INDEX", "GLM53_DSA_VIS_LAYERS * GLM53_DSA_VIS_LATENT"),
        ("item / GLM53_DSA_VIS_INDEX", "item / GLM53_DSA_VIS_LATENT"),
        ("item % GLM53_DSA_VIS_INDEX", "item % GLM53_DSA_VIS_LATENT"),
    ] {
        let mutant = CUDA.replacen(from, to, 1);
        assert_ne!(mutant, CUDA, "missing mutation source: {from}");
        assert!(!contract(HOST, &mutant));
    }
    assert!(!contract(
        HOST,
        &swap_unique(
            CUDA,
            "ready_statuses[layer] = 0U",
            "atomicExch(ready_nonces + layer, transaction_nonce)",
        ),
    ));
    assert!(!contract(&HOST.replacen("const THREADS: u32 = 256;", "const THREADS: u32 = 128;", 1), CUDA));
    let seam = "__syncthreads();\n    __threadfence_system();\n    __syncthreads();";
    let reordered = "__syncthreads();\n    __syncthreads();\n    __threadfence_system();";
    assert!(!contract(HOST, &CUDA.replace(seam, reordered)));
    assert!(!contract(&HOST.replacen(".block([THREADS, 1, 1])", ".block([128, 1, 1])", 1), CUDA));
    let clear_swapped = HOST.replace("gpu.memset_async(buffers.ready_nonces_u64.ptr, 0, plan.nonces_bytes, stream)?;\n    gpu.memset_async(\n        buffers.ready_statuses_u32.ptr,\n        0xff,\n        plan.statuses_bytes,\n        stream,\n    )?;", "gpu.memset_async(\n        buffers.ready_statuses_u32.ptr,\n        0xff,\n        plan.statuses_bytes,\n        stream,\n    )?;\n    gpu.memset_async(buffers.ready_nonces_u64.ptr, 0, plan.nonces_bytes, stream)?;");
    assert_ne!(clear_swapped, HOST);
    assert!(!contract(&clear_swapped, CUDA));
}
#[path = "glm53_dsa_current_visibility_registration_tests.rs"]
mod registration_tests;
