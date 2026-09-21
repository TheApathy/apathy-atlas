// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashSet;

use spark_runtime::gpu::DevicePtr;

use super::*;

const EXPERTS: usize = 512;
const INTER: usize = 640;
const HIDDEN: usize = 2_560;

fn exact_slot(
    source: &str,
    shape: &[usize],
    expert: usize,
    projection: PackedMtpProjection,
    offset: usize,
    n: usize,
    k: usize,
) -> Result<String> {
    packed_mtp_nvfp4_cache_slot(
        source,
        shape,
        WeightDtype::BF16,
        expert,
        projection,
        offset,
        n,
        k,
    )
}

#[test]
fn exact_1536_slots_are_stable_distinct_and_provenance_bound() {
    let mut slots = HashSet::new();
    let gate_up_stride = 2 * INTER * HIDDEN * WeightDtype::BF16.byte_size();
    for expert in 0..EXPERTS {
        let slices = [
            (
                PackedMtpProjection::Gate,
                expert * gate_up_stride,
                GATE_UP_NAME,
                [EXPERTS, 2 * INTER, HIDDEN],
                INTER,
                HIDDEN,
            ),
            (
                PackedMtpProjection::Up,
                expert * gate_up_stride + gate_up_stride / 2,
                GATE_UP_NAME,
                [EXPERTS, 2 * INTER, HIDDEN],
                INTER,
                HIDDEN,
            ),
            (
                PackedMtpProjection::Down,
                expert * gate_up_stride / 2,
                DOWN_NAME,
                [EXPERTS, HIDDEN, INTER],
                HIDDEN,
                INTER,
            ),
        ];
        for (projection, offset, source, shape, n, k) in slices {
            let slot = exact_slot(source, &shape, expert, projection, offset, n, k).unwrap();
            assert!(slot.contains(TARGET_QUANT));
            assert!(slot.contains(TARGET_KERNELS));
            assert!(slot.contains("source_dtype=BF16"));
            assert!(slots.insert(slot));
        }
    }
    assert_eq!(slots.len(), 1_536);
}

#[test]
fn hostile_source_slice_and_target_identities_fail_closed() {
    let gate_offset = 7 * 2 * INTER * HIDDEN * WeightDtype::BF16.byte_size();
    let exact = || {
        exact_slot(
            GATE_UP_NAME,
            &[EXPERTS, 2 * INTER, HIDDEN],
            7,
            PackedMtpProjection::Gate,
            gate_offset,
            INTER,
            HIDDEN,
        )
    };
    assert_eq!(exact().unwrap(), exact().unwrap());

    let wrong_name = exact_slot(
        "wrong",
        &[EXPERTS, 2 * INTER, HIDDEN],
        7,
        PackedMtpProjection::Gate,
        gate_offset,
        INTER,
        HIDDEN,
    );
    assert!(wrong_name.is_err());
    let wrong_dtype = packed_mtp_nvfp4_cache_slot(
        GATE_UP_NAME,
        &[EXPERTS, 2 * INTER, HIDDEN],
        WeightDtype::UInt8,
        7,
        PackedMtpProjection::Gate,
        gate_offset,
        INTER,
        HIDDEN,
    );
    assert!(wrong_dtype.is_err());
    let wrong_shape = exact_slot(
        GATE_UP_NAME,
        &[EXPERTS, 2 * INTER - 1, HIDDEN],
        7,
        PackedMtpProjection::Gate,
        gate_offset,
        INTER,
        HIDDEN,
    );
    assert!(wrong_shape.is_err());
    let wrong_expert = exact_slot(
        GATE_UP_NAME,
        &[EXPERTS, 2 * INTER, HIDDEN],
        EXPERTS,
        PackedMtpProjection::Gate,
        0,
        INTER,
        HIDDEN,
    );
    assert!(wrong_expert.is_err());
    let wrong_offset = exact_slot(
        GATE_UP_NAME,
        &[EXPERTS, 2 * INTER, HIDDEN],
        7,
        PackedMtpProjection::Gate,
        0,
        INTER,
        HIDDEN,
    );
    assert!(wrong_offset.is_err());
    let wrong_role = exact_slot(
        DOWN_NAME,
        &[EXPERTS, HIDDEN, INTER],
        7,
        PackedMtpProjection::Up,
        0,
        INTER,
        HIDDEN,
    );
    assert!(wrong_role.is_err());
    let overflow = exact_slot(
        DOWN_NAME,
        &[1, usize::MAX, 32],
        0,
        PackedMtpProjection::Down,
        0,
        usize::MAX,
        32,
    );
    assert!(overflow.is_err());
}

#[test]
fn hostile_null_and_wrapping_device_sources_fail_before_pointer_arithmetic() {
    let null = WeightTensor {
        ptr: DevicePtr::NULL,
        shape: vec![EXPERTS, 2 * INTER, HIDDEN],
        dtype: WeightDtype::BF16,
    };
    let null_result = packed_mtp_cache_slot_for_source(
        GATE_UP_NAME,
        &null,
        0,
        PackedMtpProjection::Gate,
        0,
        INTER,
        HIDDEN,
    );
    assert!(null_result.is_err());

    let wrapping = WeightTensor {
        ptr: DevicePtr(u64::MAX - 1),
        shape: vec![EXPERTS, 2 * INTER, HIDDEN],
        dtype: WeightDtype::BF16,
    };
    let wrapping_result = packed_mtp_cache_slot_for_source(
        GATE_UP_NAME,
        &wrapping,
        0,
        PackedMtpProjection::Gate,
        0,
        INTER,
        HIDDEN,
    );
    assert!(wrapping_result.is_err());
}

#[test]
fn cache_commit_follows_every_native_mtp_transform() {
    let factory = include_str!("../factory/build.rs");
    let loader = include_str!("../weight_loader/qwen35.rs");
    let cache = include_str!("../weight_loader/transform_cache.rs");
    let registry = include_str!("../../../atlas-core/src/registry.rs");
    let cuda = include_str!("../../../spark-runtime/src/cuda_backend/transform_cache_identity.rs");
    const { assert!(crate::weight_loader::transform_cache::CACHE_FORMAT_VERSION >= 4) };
    assert!(factory.contains("configure_construction_mode(use_speculative)?"));
    assert!(cache.contains("hash_construction_mode(&mut fp, construction_mode())"));
    assert!(cache.contains("source_content_digest={source_digest}"));
    assert!(cache.contains("transform_cache_identity()"));
    assert!(registry.contains("loaded_modules_sha256={sha256}"));
    assert!(cuda.contains("identity(registry_identity: &str)"));
    assert!(cuda.contains("{registry_identity};cuda_device={device}"));
    let proposer = factory
        .find("let qwen4_mtp_proposer")
        .expect("native MTP proposer construction must exist");
    let finish = factory[proposer..]
        .find("transform_cache::finish()")
        .map(|offset| proposer + offset)
        .expect("Qwen4 transform cache must be committed");

    assert!(factory[proposer..finish].contains("load_qwen4_mtp_layer"));
    assert!(factory[proposer..finish].contains("load_qwen4_final_mixer"));
    assert!(factory[proposer..finish].contains("Qwen4MtpHead::new"));
    assert!(!loader.contains("transform_cache::finish()"));
}

#[test]
fn packed_cache_call_sites_are_exact_and_numbered_path_stays_distinct() {
    let loader = include_str!("ssm_qwen35.rs");
    assert_eq!(loader.matches("quantize_packed_mtp_slice(").count(), 3);
    assert!(loader.contains("if is_fused_bf16"));
    assert!(loader.contains("experts.push(load_expert(&format!(\"{p}.experts.{e}\"))?);"));
}
