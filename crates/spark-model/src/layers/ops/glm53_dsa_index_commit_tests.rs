// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

const HOST_SOURCE: &str = include_str!("glm53_dsa_index_commit.rs");
const CUDA_SOURCE: &str =
    include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/glm53_dsa_index_commit.cu");

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReferenceState {
    staged_keys: Vec<u16>,
    staged_gates: Vec<u16>,
    staged_validity: Vec<u8>,
    persistent_keys: Vec<u16>,
    persistent_gates: Vec<u16>,
    persistent_validity: Vec<u8>,
    pool_validity: Vec<u8>,
    lengths: [u32; 11],
    generations: [u64; 11],
    latent_ends: [u32; 11],
    latent_nonces: [u64; 11],
    index_ends: [u32; 11],
    index_nonces: [u64; 11],
    visibility_ends: [u32; 11],
    visibility_nonces: [u64; 11],
    visibility_status: [u32; 11],
}

impl ReferenceState {
    fn new(start: u32, generation: u64, nonce: u64) -> Self {
        let end = start + 1;
        let elements = 11 * 3 * 128;
        let final_tail = (end % 4) as usize;
        let mut pool_validity = vec![0; 11 * (1_048_576 / 4)];
        if start % 4 == 3 {
            for layer in 0..11 {
                pool_validity[layer * (1_048_576 / 4) + start as usize / 4] = 1;
            }
        }
        Self {
            staged_keys: (0..elements).map(|index| index as u16).collect(),
            staged_gates: (0..elements).map(|index| (index as u16) ^ 0x55aa).collect(),
            staged_validity: (0..33)
                .map(|index| u8::from(index % 3 < final_tail))
                .collect(),
            persistent_keys: vec![0x1111; elements],
            persistent_gates: vec![0x2222; elements],
            persistent_validity: vec![9; 33],
            pool_validity,
            lengths: [start; 11],
            generations: [generation; 11],
            latent_ends: [end; 11],
            latent_nonces: [nonce; 11],
            index_ends: [end; 11],
            index_nonces: [nonce; 11],
            visibility_ends: [end; 11],
            visibility_nonces: [nonce; 11],
            visibility_status: [0; 11],
        }
    }

    fn commit(&mut self, accepted: u32, start: u32, generation: u64, nonce: u64) -> bool {
        let end = start + 1;
        for layer in 0..11 {
            let invalid = self.lengths[layer] != start
                || self.generations[layer] != generation
                || self.latent_ends[layer] != end
                || self.latent_nonces[layer] != nonce
                || self.index_ends[layer] != end
                || self.index_nonces[layer] != nonce
                || self.visibility_ends[layer] != end
                || self.visibility_nonces[layer] != nonce
                || self.visibility_status[layer] != 0;
            if invalid {
                return false;
            }
            if accepted == 1 && start % 4 == 3 {
                let pool = layer * (1_048_576 / 4) + start as usize / 4;
                if self.pool_validity[pool] != 1 {
                    return false;
                }
            }
        }
        if accepted == 1 {
            let tail = (end % 4) as usize;
            if self
                .staged_validity
                .iter()
                .enumerate()
                .any(|(item, &valid)| valid != u8::from(item % 3 < tail))
            {
                return false;
            }
            self.persistent_keys.copy_from_slice(&self.staged_keys);
            self.persistent_gates.copy_from_slice(&self.staged_gates);
            self.persistent_validity
                .copy_from_slice(&self.staged_validity);
        }
        self.index_ends.fill(0);
        self.index_nonces.fill(0);
        self.visibility_ends.fill(0);
        self.visibility_nonces.fill(0);
        if accepted == 0 {
            self.latent_ends.fill(0);
            self.latent_nonces.fill(0);
        }
        true
    }
}

#[test]
fn plan_pins_all11_full_1m_and_boundaries() {
    for (start, accepted, tail) in [
        (0, 0, 1),
        (1, 1, 2),
        (2, 1, 3),
        (3, 1, 0),
        (1_048_575, 1, 0),
    ] {
        let plan =
            Glm53DsaIndexCommitPlan::new(accepted, 1_048_576, start, start + 1, 7, 9).unwrap();
        assert_eq!(plan.tail_vector_bytes, 8_448);
        assert_eq!(plan.tail_validity_bytes, 33);
        assert_eq!(plan.pool_validity_bytes, 2_883_584);
        assert_eq!((plan.layer_u32_bytes, plan.layer_u64_bytes), (44, 88));
        assert_eq!(tail, plan.end_position % 4);
    }
    for hostile in [
        Glm53DsaIndexCommitPlan::new(2, 1_048_576, 0, 1, 1, 1),
        Glm53DsaIndexCommitPlan::new(1, 1_048_575, 0, 1, 1, 1),
        Glm53DsaIndexCommitPlan::new(1, 1_048_576, 0, 2, 1, 1),
        Glm53DsaIndexCommitPlan::new(1, 1_048_576, 1_048_576, 1_048_577, 1, 1),
        Glm53DsaIndexCommitPlan::new(1, 1_048_576, 0, 1, 0, 1),
        Glm53DsaIndexCommitPlan::new(1, 1_048_576, 0, 1, 1, 0),
    ] {
        assert!(hostile.is_err());
    }
}

#[test]
fn reference_prevalidates_all_layers_and_separates_accept_from_reject() {
    for start in [0, 1, 2, 3, 1_048_575] {
        let initial = ReferenceState::new(start, 7, 9);
        for fault in 0..7 {
            let mut state = initial.clone();
            match fault {
                0 => state.lengths[10] += 1,
                1 => state.generations[10] += 1,
                2 => state.latent_nonces[10] += 1,
                3 => state.index_ends[10] = 0,
                4 => state.visibility_nonces[10] = 0,
                5 => state.visibility_status[10] = 1,
                _ if start % 4 == 3 => {
                    state.pool_validity[10 * (1_048_576 / 4) + start as usize / 4] = 0
                }
                _ => state.staged_validity[0] ^= 1,
            }
            let before = state.clone();
            assert!(!state.commit(1, start, 7, 9));
            assert_eq!(state, before);
        }
        let mut accepted = initial.clone();
        assert!(accepted.commit(1, start, 7, 9));
        assert_eq!(accepted.persistent_keys, initial.staged_keys);
        assert_eq!(accepted.persistent_gates, initial.staged_gates);
        assert_eq!(accepted.latent_nonces, [9; 11]);
        assert_eq!(accepted.index_nonces, [0; 11]);
        assert_eq!(accepted.visibility_nonces, [0; 11]);

        let mut rejected = initial.clone();
        assert!(rejected.commit(0, start, 7, 9));
        assert_eq!(rejected.persistent_keys, initial.persistent_keys);
        assert_eq!(rejected.persistent_validity, initial.persistent_validity);
        assert_eq!(rejected.latent_nonces, [0; 11]);
    }
}

fn fake_buffers(plan: Glm53DsaIndexCommitPlan) -> Glm53DsaIndexCommitBuffers {
    let mut cursor = 0x10_0000u64;
    let mut next = |bytes: usize| {
        let buffer = GgmlIqBuffer {
            ptr: DevicePtr(cursor),
            bytes,
        };
        cursor = (cursor + u64::try_from(bytes).unwrap() + 0xfff) & !0xfff;
        buffer
    };
    Glm53DsaIndexCommitBuffers {
        staged_tail_keys_bf16: next(plan.tail_vector_bytes),
        staged_tail_gates_bf16: next(plan.tail_vector_bytes),
        staged_tail_validity_u8: next(plan.tail_validity_bytes),
        persistent_tail_keys_bf16: next(plan.tail_vector_bytes),
        persistent_tail_gates_bf16: next(plan.tail_vector_bytes),
        persistent_tail_validity_u8: next(plan.tail_validity_bytes),
        persistent_pool_validity_u8: next(plan.pool_validity_bytes),
        logical_lengths_u32: next(plan.layer_u32_bytes),
        owner_generations_u64: next(plan.layer_u64_bytes),
        latent_ends_u32: next(plan.layer_u32_bytes),
        latent_nonces_u64: next(plan.layer_u64_bytes),
        index_ends_u32: next(plan.layer_u32_bytes),
        index_nonces_u64: next(plan.layer_u64_bytes),
        visibility_ends_u32: next(plan.layer_u32_bytes),
        visibility_nonces_u64: next(plan.layer_u64_bytes),
        visibility_status_u32: next(plan.layer_u32_bytes),
    }
}

#[test]
fn forged_extent_alignment_alias_and_overflow_have_zero_launches() {
    let gpu = MockGpuBackend::new();
    let kernel = Glm53DsaIndexCommitKernel::load(&gpu).unwrap();
    let plan = Glm53DsaIndexCommitPlan::new(1, 1_048_576, 3, 4, 7, 9).unwrap();
    let valid = fake_buffers(plan);
    let mut forged = plan;
    forged.tail_vector_bytes += 2;
    assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
    for hostile in [
        Glm53DsaIndexCommitBuffers {
            staged_tail_keys_bf16: GgmlIqBuffer {
                bytes: plan.tail_vector_bytes - 2,
                ..valid.staged_tail_keys_bf16
            },
            ..valid
        },
        Glm53DsaIndexCommitBuffers {
            owner_generations_u64: GgmlIqBuffer {
                ptr: DevicePtr(valid.owner_generations_u64.ptr.0 + 4),
                ..valid.owner_generations_u64
            },
            ..valid
        },
        Glm53DsaIndexCommitBuffers {
            index_ends_u32: GgmlIqBuffer {
                ptr: valid.latent_ends_u32.ptr,
                ..valid.index_ends_u32
            },
            ..valid
        },
        Glm53DsaIndexCommitBuffers {
            visibility_status_u32: GgmlIqBuffer {
                ptr: DevicePtr::NULL,
                ..valid.visibility_status_u32
            },
            ..valid
        },
        Glm53DsaIndexCommitBuffers {
            staged_tail_gates_bf16: GgmlIqBuffer {
                ptr: DevicePtr(u64::MAX - 1),
                ..valid.staged_tail_gates_bf16
            },
            ..valid
        },
    ] {
        assert!(kernel.launch(&gpu, plan, hostile, 0).is_err());
    }
    assert_eq!(gpu.launch_count(), 0);
    kernel.launch(&gpu, plan, valid, 0).unwrap();
    assert_eq!(gpu.launch_count(), 1);
}

fn swap_unique(source: &str, left: &str, right: &str) -> String {
    assert_eq!(source.matches(left).count(), 1);
    assert_eq!(source.matches(right).count(), 1);
    source
        .replace(left, "__LEFT__")
        .replace(right, left)
        .replace("__LEFT__", right)
}

fn abi_contract(host: &str, cuda: &str) -> bool {
    const HOST: &str = ".arg_ptr(buffers.staged_tail_keys_bf16.ptr)\n            .arg_ptr(buffers.staged_tail_gates_bf16.ptr)\n            .arg_ptr(buffers.staged_tail_validity_u8.ptr)\n            .arg_ptr(buffers.persistent_tail_keys_bf16.ptr)\n            .arg_ptr(buffers.persistent_tail_gates_bf16.ptr)\n            .arg_ptr(buffers.persistent_tail_validity_u8.ptr)\n            .arg_ptr(buffers.persistent_pool_validity_u8.ptr)\n            .arg_ptr(buffers.logical_lengths_u32.ptr)\n            .arg_ptr(buffers.owner_generations_u64.ptr)\n            .arg_ptr(buffers.latent_ends_u32.ptr)\n            .arg_ptr(buffers.latent_nonces_u64.ptr)\n            .arg_ptr(buffers.index_ends_u32.ptr)\n            .arg_ptr(buffers.index_nonces_u64.ptr)\n            .arg_ptr(buffers.visibility_ends_u32.ptr)\n            .arg_ptr(buffers.visibility_nonces_u64.ptr)\n            .arg_ptr(buffers.visibility_status_u32.ptr)\n            .arg_u32(plan.accepted)\n            .arg_u32(plan.capacity)\n            .arg_u32(plan.start_position)\n            .arg_u32(plan.end_position)\n            .arg_u64(plan.owner_generation)\n            .arg_u64(plan.transaction_nonce)";
    const CUDA: &str = "const __nv_bfloat16 * __restrict__ staged_tail_keys,\n        const __nv_bfloat16 * __restrict__ staged_tail_gates,\n        const unsigned char * __restrict__ staged_tail_validity,\n        __nv_bfloat16 * __restrict__ persistent_tail_keys,\n        __nv_bfloat16 * __restrict__ persistent_tail_gates,\n        unsigned char * __restrict__ persistent_tail_validity,\n        const unsigned char * __restrict__ persistent_pool_validity,\n        const unsigned int * __restrict__ logical_lengths,\n        const unsigned long long * __restrict__ owner_generations,\n        unsigned int * __restrict__ latent_ends,\n        unsigned long long * __restrict__ latent_nonces,\n        unsigned int * __restrict__ index_ends,\n        unsigned long long * __restrict__ index_nonces,\n        unsigned int * __restrict__ visibility_ends,\n        unsigned long long * __restrict__ visibility_nonces,\n        const unsigned int * __restrict__ visibility_status,\n        unsigned int accepted, unsigned int capacity,\n        unsigned int start_position, unsigned int end_position,\n        unsigned long long owner_generation,\n        unsigned long long transaction_nonce";
    host.contains(HOST) && cuda.contains(CUDA) && host.matches(".arg_").count() == 22
}

fn ordered(source: &str, tokens: &[&str]) -> bool {
    let mut cursor = 0;
    tokens.iter().all(|token| {
        let Some(offset) = source[cursor..].find(token) else {
            return false;
        };
        cursor += offset + token.len();
        true
    })
}

fn host_contract(host: &str, cuda: &str) -> bool {
    abi_contract(host, cuda)
        && host.contains("const THREADS: u32 = 256;")
        && host.contains(
            "gpu.kernel(\n                \"glm53_dsa_index_commit\",\n                \"atlas_glm53_dsa_index_commit_all11\",",
        )
        && host.contains(".grid([1, 1, 1])\n            .block([THREADS, 1, 1])")
}

fn cuda_contract(source: &str) -> bool {
    const REQUIRED: &[&str] = &[
        "#define GLM53_DSA_COMMIT_LAYERS 11U",
        "#define GLM53_DSA_COMMIT_THREADS 256U",
        "const unsigned int lane = threadIdx.x;",
        "blockDim.x != GLM53_DSA_COMMIT_THREADS",
        "blockDim.y != 1U",
        "blockDim.z != 1U",
        "gridDim.x != 1U",
        "gridDim.y != 1U",
        "gridDim.z != 1U",
        "logical_lengths[layer] != start_position",
        "owner_generations[layer] != owner_generation",
        "latent_ends[layer] != end_position",
        "latent_nonces[layer] != transaction_nonce",
        "index_ends[layer] != end_position",
        "index_nonces[layer] != transaction_nonce",
        "visibility_ends[layer] != end_position",
        "visibility_nonces[layer] != transaction_nonce",
        "visibility_status[layer] != 0U",
        "start_position % GLM53_DSA_COMMIT_KPOOL == 3U",
        "(unsigned long long)layer *\n                    (capacity / GLM53_DSA_COMMIT_KPOOL) +\n                start_position / GLM53_DSA_COMMIT_KPOOL",
        "persistent_pool_validity[pool] != 1U",
        "const unsigned int slot = item % GLM53_DSA_COMMIT_TAIL;",
        "if (invalid_transaction != 0U) {",
        "persistent_tail_keys[element] = staged_tail_keys[element];",
        "persistent_tail_gates[element] = staged_tail_gates[element];",
        "persistent_tail_validity[item] = staged_tail_validity[item];",
        "index_ends[layer] = 0U;",
        "visibility_ends[layer] = 0U;",
        "atomicExch(index_nonces + layer, 0ULL);",
        "atomicExch(visibility_nonces + layer, 0ULL);",
    ];
    const REJECT_ENDS: &str =
        "if (accepted == 0U) {\n            latent_ends[layer] = 0U;\n        }";
    const REJECT_NONCES: &str =
        "if (accepted == 0U) {\n            atomicExch(latent_nonces + layer, 0ULL);\n        }";
    const PAYLOAD_FENCE: &str = "__threadfence_system();\n    __syncthreads();\n\n    for";
    const ENDS_FENCE: &str = "__threadfence_system();\n    __syncthreads();\n    for";
    REQUIRED.iter().all(|token| source.contains(token))
        && source.contains(REJECT_ENDS)
        && source.contains(REJECT_NONCES)
        && source.matches(PAYLOAD_FENCE).count() == 1
        && source.matches(ENDS_FENCE).count() == 1
        && source.matches("__threadfence_system();").count() == 2
        && source.matches("__syncthreads();").count() == 4
        && ordered(
            source,
            &[
                "if (invalid_transaction != 0U)",
                "persistent_tail_keys[element] = staged_tail_keys[element];",
                "persistent_tail_gates[element] = staged_tail_gates[element];",
                "persistent_tail_validity[item] = staged_tail_validity[item];",
                "__threadfence_system();",
                "index_ends[layer] = 0U;",
                "visibility_ends[layer] = 0U;",
                "__threadfence_system();",
                "atomicExch(index_nonces + layer, 0ULL);",
                "atomicExch(visibility_nonces + layer, 0ULL);",
            ],
        )
        && !source.contains("logical_lengths[layer] =")
}

#[test]
fn ordered_host_cuda_abi_and_effect_contract_reject_mutants() {
    assert!(host_contract(HOST_SOURCE, CUDA_SOURCE));
    assert!(cuda_contract(CUDA_SOURCE));
    for (left, right) in [
        (
            ".arg_ptr(buffers.staged_tail_keys_bf16.ptr)",
            ".arg_ptr(buffers.persistent_tail_keys_bf16.ptr)",
        ),
        (
            ".arg_ptr(buffers.latent_ends_u32.ptr)",
            ".arg_ptr(buffers.index_ends_u32.ptr)",
        ),
        (".arg_u32(plan.accepted)", ".arg_u32(plan.start_position)"),
        (
            ".arg_u64(plan.owner_generation)",
            ".arg_u64(plan.transaction_nonce)",
        ),
    ] {
        assert!(!host_contract(
            &swap_unique(HOST_SOURCE, left, right),
            CUDA_SOURCE,
        ));
    }
    for (from, to) in [
        (".grid([1, 1, 1])", ".grid([2, 1, 1])"),
        (".block([THREADS, 1, 1])", ".block([128, 1, 1])"),
    ] {
        assert!(!host_contract(
            &HOST_SOURCE.replacen(from, to, 1),
            CUDA_SOURCE
        ));
    }
    for (from, to) in [
        (
            "#define GLM53_DSA_COMMIT_LAYERS 11U",
            "#define GLM53_DSA_COMMIT_LAYERS 10U",
        ),
        (
            "const unsigned int lane = threadIdx.x;",
            "const unsigned int lane = threadIdx.y;",
        ),
        ("blockDim.x != GLM53_DSA_COMMIT_THREADS || ", ""),
        ("latent_ends[layer] != end_position ||\n            ", ""),
        (
            "visibility_status[layer] != 0U",
            "visibility_status[layer] == 0U",
        ),
        (
            "persistent_tail_gates[element] = staged_tail_gates[element];",
            "",
        ),
        (
            "start_position % GLM53_DSA_COMMIT_KPOOL == 3U",
            "start_position % GLM53_DSA_COMMIT_KPOOL == 2U",
        ),
        (
            "item % GLM53_DSA_COMMIT_TAIL",
            "item / GLM53_DSA_COMMIT_TAIL",
        ),
        (
            "if (accepted == 0U) {\n            latent_ends[layer]",
            "if (accepted == 1U) {\n            latent_ends[layer]",
        ),
        (
            "if (accepted == 0U) {\n            atomicExch(latent_nonces",
            "if (accepted == 1U) {\n            atomicExch(latent_nonces",
        ),
        (
            "__threadfence_system();\n    __syncthreads();",
            "__syncthreads();\n    __threadfence_system();",
        ),
    ] {
        let mutant = CUDA_SOURCE.replacen(from, to, 1);
        assert_ne!(mutant, CUDA_SOURCE, "missing mutation source: {from}");
        assert!(!cuda_contract(&mutant), "false accept: {from}");
    }
    assert!(!cuda_contract(&CUDA_SOURCE.replacen(
        "__threadfence_system();",
        "",
        1
    )));
    assert!(!cuda_contract(&CUDA_SOURCE.replacen(
        "__syncthreads();",
        "",
        1
    )));
}
