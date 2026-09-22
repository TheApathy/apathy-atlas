// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;

use super::glm53::{expected_schema, validate_identity_for_test, validate_verified_for_test};
use super::{
    GLM53_GGUF_REVISION, GgmlType, GgufHeader, GgufTensorInfo, GgufValue, Glm53Iq3Summary,
    Glm53QuantProfile,
};

fn metadata(profile: Glm53QuantProfile, split_no: usize) -> BTreeMap<String, GgufValue> {
    let mut metadata = BTreeMap::from([
        ("split.count".to_string(), GgufValue::Unsigned(4)),
        ("split.no".to_string(), GgufValue::Unsigned(split_no as u64)),
        // Real Unsloth GLM-5.3 shards store this as INT32, not unsigned —
        // llama.cpp's `gguf_split` writes it signed while `split.count` and
        // `split.no` stay UINT16. Mirroring the on-disk encoding here keeps the
        // whole suite honest about what the loader actually has to parse.
        ("split.tensors.count".to_string(), GgufValue::Signed(1412)),
    ]);
    if split_no == 0 {
        let file_type = match profile {
            Glm53QuantProfile::UdQ2KXl => 10,
            Glm53QuantProfile::UdIq3Xxs => 23,
            Glm53QuantProfile::UdIq2Xxs => 19,
        };
        metadata.extend([
            (
                "general.architecture".to_string(),
                GgufValue::String("glm5next".to_string()),
            ),
            (
                "general.file_type".to_string(),
                GgufValue::Unsigned(file_type),
            ),
            (
                "general.quantization_version".to_string(),
                GgufValue::Unsigned(2),
            ),
            ("glm5next.block_count".to_string(), GgufValue::Unsigned(46)),
            (
                "glm5next.nextn_predict_layers".to_string(),
                GgufValue::Unsigned(1),
            ),
            (
                "glm5next.attention.head_count_kv".to_string(),
                GgufValue::Array {
                    element_type: 5,
                    len: 46,
                },
            ),
            (
                "glm5next.swiglu_clamp_exp".to_string(),
                GgufValue::Array {
                    element_type: 6,
                    len: 46,
                },
            ),
            (
                "glm5next.swiglu_clamp_shexp".to_string(),
                GgufValue::Array {
                    element_type: 6,
                    len: 46,
                },
            ),
        ]);
    }
    metadata
}

fn tensor_counts(profile: Glm53QuantProfile) -> [usize; 4] {
    match profile {
        Glm53QuantProfile::UdQ2KXl => [0, 676, 622, 114],
        Glm53QuantProfile::UdIq3Xxs => [0, 614, 558, 240],
        Glm53QuantProfile::UdIq2Xxs => [0, 717, 676, 19],
    }
}

fn make_headers(profile: Glm53QuantProfile) -> Vec<GgufHeader> {
    let specs: Vec<_> = expected_schema(profile).into_iter().collect();
    let mut cursor = 0;
    tensor_counts(profile)
        .into_iter()
        .enumerate()
        .map(|(split_no, count)| {
            let tensors = specs[cursor..cursor + count]
                .iter()
                .map(|(name, (dimensions, ggml_type))| {
                    let elements = dimensions.iter().product();
                    GgufTensorInfo {
                        name: name.clone(),
                        dimensions: dimensions.clone(),
                        ggml_type: *ggml_type,
                        offset: 0,
                        byte_len: ggml_type.byte_len(elements).unwrap(),
                    }
                })
                .collect();
            cursor += count;
            GgufHeader {
                version: 3,
                alignment: 32,
                data_offset: if split_no == 0 { 9_429_888 } else { 32 },
                file_len: profile.shard_bytes()[split_no],
                metadata: metadata(profile, split_no),
                tensors,
            }
        })
        .collect()
}

fn validate(profile: Glm53QuantProfile, headers: &[GgufHeader]) -> anyhow::Result<Glm53Iq3Summary> {
    validate_verified_for_test(profile, headers.to_vec())
}

fn assert_census(headers: &[GgufHeader], expected: &[(GgmlType, usize, u64)]) {
    let all: Vec<_> = headers.iter().flat_map(|header| &header.tensors).collect();
    for (ggml_type, count, bytes) in expected {
        let selected: Vec<_> = all
            .iter()
            .filter(|tensor| tensor.ggml_type == *ggml_type)
            .collect();
        assert_eq!(selected.len(), *count, "{ggml_type:?}");
        assert_eq!(
            selected.iter().map(|tensor| tensor.byte_len).sum::<u64>(),
            *bytes,
            "{ggml_type:?}"
        );
    }
}

#[test]
fn exact_q2_profile_identity_schema_and_census_are_pinned() {
    let profile = Glm53QuantProfile::UdQ2KXl;
    assert_eq!(Glm53QuantProfile::PRIMARY, profile);
    assert_eq!(profile.directory_name(), "UD-Q2_K_XL");
    assert_eq!(
        GLM53_GGUF_REVISION,
        "ac47690c15c8703615ab7d9c1ef2293d45372757"
    );
    assert_eq!(
        profile.canonical_file_names(),
        &[
            "GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf",
            "GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf",
            "GLM-5.3-Flash-UD-Q2_K_XL-00003-of-00004.gguf",
            "GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf",
        ]
    );
    assert_eq!(
        profile.shard_bytes(),
        &[9_429_859, 49_294_975_936, 49_949_266_048, 9_466_399_584]
    );
    assert_eq!(
        profile.shard_sha256(),
        &[
            "b2aaab111a0f93f04b0627270691fe430b6bb14b17c0d62f9c0f1ac75e7efef4",
            "f4a9e1ab13d5d9620f5590c9a4aba4c169e6707a1554f3be3fe3112f80a66825",
            "330a8ad76c787b3ab6df062dd30abea7aafe6a0e3d5860e8720dfa4abb2434a5",
            "5c294f42edc5d69cf00a79ab444e61f0742e9933105fc5c85f97da54dab8946f",
        ]
    );
    assert_eq!(profile.total_file_bytes(), 108_720_071_427);
    assert_eq!(
        profile.shard_bytes().iter().sum::<u64>(),
        profile.total_file_bytes()
    );
    for index in 0..4 {
        validate_identity_for_test(
            profile,
            index,
            profile.shard_bytes()[index],
            profile.shard_sha256()[index],
        )
        .unwrap();
    }

    let headers = make_headers(profile);
    let summary = validate(profile, &headers).unwrap();
    assert_eq!(summary.shards, 4);
    assert_eq!(summary.tensors, 1412);
    assert_eq!(summary.tensor_bytes, 108_710_550_904);
    assert_census(
        &headers,
        &[
            (GgmlType::F32, 638, 225_467_768),
            (GgmlType::Q8_0, 346, 860_372_992),
            (GgmlType::Q2_K, 2, 1_585_446_912),
            (GgmlType::Q3_K, 1, 1_038_090_240),
            (GgmlType::Q4_K, 1, 356_843_520),
            (GgmlType::Q5_K, 181, 3_251_961_856),
            (GgmlType::Q6_K, 117, 2_358_558_720),
            (GgmlType::IQ2_XS, 82, 57_264_832_512),
            (GgmlType::IQ3_XXS, 41, 37_918_605_312),
            (GgmlType::IQ4_XS, 3, 3_850_371_072),
        ],
    );
}

#[test]
fn exact_iq3_compatibility_schema_is_unchanged() {
    let profile = Glm53QuantProfile::UdIq3Xxs;
    let headers = make_headers(profile);
    let summary = validate(profile, &headers).unwrap();
    assert_eq!(summary.tensor_bytes, 120_358_051_192);
    assert_census(
        &headers,
        &[
            (GgmlType::F32, 638, 225_467_768),
            (GgmlType::Q8_0, 350, 956_186_624),
            (GgmlType::Q2_K, 2, 1_585_446_912),
            (GgmlType::Q3_K, 1, 1_038_090_240),
            (GgmlType::Q6_K, 295, 6_685_163_520),
            (GgmlType::IQ3_S, 41, 42_561_699_840),
            (GgmlType::IQ2_S, 82, 63_455_625_216),
            (GgmlType::IQ4_XS, 3, 3_850_371_072),
        ],
    );
}

#[test]
fn cross_profile_hash_size_and_tensor_type_drift_fail_closed() {
    let q2 = Glm53QuantProfile::UdQ2KXl;
    let iq3 = Glm53QuantProfile::UdIq3Xxs;
    assert!(
        validate_identity_for_test(q2, 0, q2.shard_bytes()[0] - 1, q2.shard_sha256()[0]).is_err()
    );
    assert!(validate_identity_for_test(q2, 0, q2.shard_bytes()[0], iq3.shard_sha256()[0]).is_err());
    assert!(validate_identity_for_test(q2, 4, 0, "").is_err());

    let headers = make_headers(q2);
    assert!(format!("{:#}", validate(iq3, &headers).unwrap_err()).contains("metadata ABI"));

    let mut forged = headers;
    forged[0]
        .metadata
        .insert("general.file_type".to_string(), GgufValue::Unsigned(23));
    assert!(format!("{:#}", validate(iq3, &forged).unwrap_err()).contains("shape/type"));

    let tensor = forged[1]
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "blk.0.attn_q.weight")
        .unwrap();
    tensor.ggml_type = GgmlType::Q8_0;
    assert!(format!("{:#}", validate(q2, &forged).unwrap_err()).contains("metadata ABI"));
    forged[0]
        .metadata
        .insert("general.file_type".to_string(), GgufValue::Unsigned(10));
    assert!(format!("{:#}", validate(q2, &forged).unwrap_err()).contains("shape/type"));
}

#[test]
fn missing_or_duplicate_split_numbers_are_rejected() {
    let profile = Glm53QuantProfile::UdQ2KXl;
    let mut headers = make_headers(profile);
    headers[3]
        .metadata
        .insert("split.no".to_string(), GgufValue::Unsigned(2));
    assert!(format!("{:#}", validate(profile, &headers).unwrap_err()).contains("split.no"));
}

/// UD-IQ2_XXS is the profile that fits one Spark once every tensor is
/// device-resident. This exercises its schema builder so the absolute type
/// assignment and the measured histogram in `schema/ud_iq2_xxs.rs` actually
/// run, and pins the census against the published shard headers.
#[test]
fn exact_iq2_xxs_profile_schema_census_and_residency_are_pinned() {
    let profile = Glm53QuantProfile::UdIq2Xxs;
    assert_eq!(profile.directory_name(), "UD-IQ2_XXS");
    assert_eq!(
        profile.canonical_file_names(),
        &[
            "GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf",
            "GLM-5.3-Flash-UD-IQ2_XXS-00002-of-00004.gguf",
            "GLM-5.3-Flash-UD-IQ2_XXS-00003-of-00004.gguf",
            "GLM-5.3-Flash-UD-IQ2_XXS-00004-of-00004.gguf",
        ]
    );
    assert_eq!(
        profile.shard_bytes(),
        &[9_429_888, 49_896_298_080, 49_248_507_840, 2_690_716_000]
    );
    assert_eq!(
        profile.shard_sha256(),
        &[
            "d9605eec5aa2c14fc9ccff1ba3341a1e9a1476b08714d41cd43d0e10a4db3f7a",
            "d51a3d9ca05022d53e71753df75ee64ed5fd4e47655650ae023e9717b3ec158d",
            "89180f13784dc1128586b98adf1ca876279881a20bc4f01747703ce352563372",
            "9e0ef47daa1fdffce3cf2ac1594ce4634d176ad018be6e692729757f38d5fbf6",
        ]
    );
    for index in 0..4 {
        validate_identity_for_test(
            profile,
            index,
            profile.shard_bytes()[index],
            profile.shard_sha256()[index],
        )
        .unwrap();
    }
    assert_eq!(profile.tensor_bytes(), 101_835_431_288);
    assert_eq!(profile.total_file_bytes(), 101_844_951_808);
    // Shard byte sizes must reconstruct the pinned total exactly.
    assert_eq!(
        profile.shard_bytes().iter().sum::<u64>(),
        profile.total_file_bytes()
    );

    // Builds the schema, which runs the tensor-count and type-histogram
    // assertions inside the retarget.
    let schema = expected_schema(profile);
    assert_eq!(schema.len(), 1412);

    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    for (_, ty) in schema.values() {
        *histogram.entry(format!("{ty:?}")).or_default() += 1;
    }
    // Measured from the four published UD-IQ2_XXS shard headers.
    for (ty, count) in [
        ("F32", 638),
        ("Q8_0", 346),
        ("Q5_K", 248),
        ("IQ2_XXS", 82),
        ("Q6_K", 49),
        ("IQ3_XXS", 39),
        ("IQ4_XS", 3),
        ("Q4_K", 2),
        ("IQ2_S", 2),
        ("Q2_K", 2),
        ("Q3_K", 1),
    ] {
        assert_eq!(histogram.get(ty).copied(), Some(count), "type {ty} drift");
    }

    // The NextN head must survive: strip it and speculation collapses.
    let nextn = schema.keys().filter(|k| k.starts_with("blk.45.")).count();
    assert_eq!(nextn, 29);

    // Every admitted type must have a pinned GLM MMQ specialization or be F32.
    // IQ1_S/IQ1_M/IQ2_XXS-adjacent types without a kernel must never appear.
    for (name, (_, ty)) in &schema {
        assert!(
            !matches!(
                ty,
                GgmlType::IQ1_S | GgmlType::IQ1_M | GgmlType::IQ4_NL | GgmlType::Q4_0
            ),
            "{name} uses a type with no GLM kernel: {ty:?}"
        );
    }

    // The whole point of the profile: it must be materially smaller than the
    // Q2_K_XL recipe it replaces, since Atlas holds weights resident.
    assert!(profile.tensor_bytes() < Glm53QuantProfile::UdQ2KXl.tensor_bytes());
}

/// Split counts must parse from either integer signedness, but never from a
/// negative value or a non-integer. Real shards use INT32 for
/// `split.tensors.count`; rejecting that rejected genuine checkpoints.
#[test]
fn split_counts_accept_either_signedness_but_reject_negative_and_non_integer() {
    let profile = Glm53QuantProfile::UdIq2Xxs;

    // Signed is the real on-disk encoding and is what `metadata()` now emits.
    validate(profile, &make_headers(profile)).unwrap();

    // Unsigned must remain acceptable for writers that emit it that way.
    let mut headers = make_headers(profile);
    for header in &mut headers {
        header
            .metadata
            .insert("split.tensors.count".to_string(), GgufValue::Unsigned(1412));
    }
    validate(profile, &headers).unwrap();

    // A negative count is a corrupt header, not an empty directory.
    let mut headers = make_headers(profile);
    for header in &mut headers {
        header
            .metadata
            .insert("split.tensors.count".to_string(), GgufValue::Signed(-1));
    }
    let error = format!("{:#}", validate(profile, &headers).unwrap_err());
    assert!(error.contains("negative"), "unexpected error: {error}");

    // A non-integer type is still refused outright.
    let mut headers = make_headers(profile);
    for header in &mut headers {
        header.metadata.insert(
            "split.tensors.count".to_string(),
            GgufValue::String("1412".to_string()),
        );
    }
    assert!(validate(profile, &headers).is_err());
}

/// The shard identity cache must round-trip, must miss on any identity change,
/// and must never be the thing that admits a shard — `validate_identity` is.
#[test]
fn shard_identity_cache_round_trips_and_misses_on_identity_change() {
    use super::glm53::{shard_cache_lookup_for_test, shard_cache_store_for_test};
    use std::io::Write;

    let dir = std::env::temp_dir().join(format!("glm53-cache-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Point HOME at the temp dir so the cache lands there, not in the real one.
    let real_home = std::env::var_os("HOME");
    // SAFETY: test-local; restored below. Tests in this module do not run
    // concurrently with anything else that reads HOME.
    unsafe { std::env::set_var("HOME", &dir) };

    let shard = dir.join("GLM-5.3-Flash-UD-IQ2_XXS-00004-of-00004.gguf");
    std::fs::File::create(&shard)
        .unwrap()
        .write_all(b"GGUF-not-really")
        .unwrap();
    let file = std::fs::File::open(&shard).unwrap();
    let identity = super::payload::FileIdentity::capture(&file).unwrap();

    // Miss before store.
    assert!(shard_cache_lookup_for_test(&shard, &identity).is_none());

    // Store, then hit with the identical identity.
    let hex = "d9605eec5aa2c14fc9ccff1ba3341a1e9a1476b08714d41cd43d0e10a4db3f7a";
    shard_cache_store_for_test(&shard, &identity, hex);
    assert_eq!(
        shard_cache_lookup_for_test(&shard, &identity).as_deref(),
        Some(hex)
    );

    // Rewrite the file in place (size preserved). ctime and inode-level state
    // change, so the identity differs and the cache must miss.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::OpenOptions::new()
        .write(true)
        .open(&shard)
        .unwrap()
        .write_all(b"GGUF-not-reallY")
        .unwrap();
    let changed =
        super::payload::FileIdentity::capture(&std::fs::File::open(&shard).unwrap()).unwrap();
    assert_ne!(
        identity.cache_key_for_test(),
        changed.cache_key_for_test(),
        "rewriting the file must change its identity key"
    );
    assert!(shard_cache_lookup_for_test(&shard, &changed).is_none());

    unsafe {
        match real_home {
            Some(h) => std::env::set_var("HOME", h),
            None => std::env::remove_var("HOME"),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
