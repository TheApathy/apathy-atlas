// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::gguf::{GgmlType, GgufDeviceTensor};

use super::{
    Glm53GgufExperts, Glm53GgufF32, Glm53GgufMatrix, Glm53GgufMatrixBank, Glm53LayerNorms,
};

#[path = "glm53_debug_source_sha256.rs"]
mod debug_source_sha256;

const RAW_SOURCE: &str = include_str!("glm53_gguf.rs");
const RAW_SOURCE_SHA256: &str = "486af64bca1c8b5e2e1c1b8aa33bf6bfcedca24be06523255fc21338b8cabfbe";

fn derive_is_not_copy(source: &str, declaration: &str) -> bool {
    let end = source.find(declaration).unwrap();
    let start = source[..end].rfind("\n\n").map_or(0, |offset| offset + 2);
    let header = &source[start..end];
    !header.contains("Clone") && !header.contains("Copy")
}

fn ownership_quarantine(raw: &str, catalogs: &[(&str, &[&str])]) -> bool {
    let raw_types = [
        "pub struct Glm53GgufMatrix",
        "pub struct Glm53GgufExperts",
        "pub struct Glm53GgufF32",
    ];
    raw_types
        .iter()
        .all(|declaration| derive_is_not_copy(raw, declaration))
        && raw.matches("pub(crate) fn new(").count() == 3
        && ["buffer", "ptr", "expert"]
            .iter()
            .all(|method| raw.contains(&format!("pub(crate) fn {method}(&self")))
        && catalogs.iter().all(|(source, declarations)| {
            declarations
                .iter()
                .all(|declaration| derive_is_not_copy(source, declaration))
        })
}

fn producers_are_private(source: &str, methods: &[&str]) -> bool {
    methods
        .iter()
        .all(|method| source.contains(&format!("pub(crate) fn {method}(&self")))
}

#[test]
fn pointer_views_are_noncopy_and_raw_extraction_is_crate_private() {
    let catalogs: &[(&str, &[&str])] = &[
        (
            include_str!("glm53_catalog/attention.rs"),
            &["pub struct Glm53KdaWeights", "pub struct Glm53DsaWeights"],
        ),
        (
            include_str!("glm53_catalog/ffn.rs"),
            &[
                "pub struct Glm53DenseFfnWeights",
                "pub struct Glm53MoeWeights",
                "pub enum Glm53FfnWeights",
            ],
        ),
        (
            include_str!("glm53_catalog/hyper.rs"),
            &[
                "pub struct Glm53LayerNorms",
                "pub struct Glm53HyperBranchWeights",
                "pub struct Glm53HyperWeights",
            ],
        ),
        (
            include_str!("glm53_catalog/layer.rs"),
            &[
                "pub enum Glm53AttentionWeights",
                "pub struct Glm53TargetLayerWeights",
            ],
        ),
        (
            include_str!("glm53_catalog/nextn.rs"),
            &["pub struct Glm53NextnWeights"],
        ),
    ];
    assert!(ownership_quarantine(RAW_SOURCE, catalogs));
    for mutant in [
        RAW_SOURCE.replacen(
            "pub struct Glm53GgufMatrix",
            "#[derive(Clone, Copy)]\npub struct Glm53GgufMatrix",
            1,
        ),
        RAW_SOURCE.replacen("pub(crate) fn buffer(&self", "pub fn buffer(&self", 1),
        RAW_SOURCE.replacen("pub(crate) fn expert(&self", "pub fn expert(&self", 1),
    ] {
        assert!(!ownership_quarantine(&mutant, catalogs));
    }
    let descriptor_source = include_str!("glm53_catalog.rs");
    assert!(descriptor_source.contains("pub struct Glm53LayerDescriptor"));
    assert!(descriptor_source.contains("#[derive(Debug, Clone, Copy, PartialEq, Eq)]"));
    assert!(descriptor_source.contains("pub(crate) fn new(store: &'a GgufDeviceStore)"));
    for (source, methods) in [
        (
            descriptor_source,
            &["token_embedding", "output", "output_norm"][..],
        ),
        (catalogs[0].0, &["kda", "dsa"][..]),
        (catalogs[1].0, &["ffn"][..]),
        (catalogs[2].0, &["norms", "hyper_connections"][..]),
        (catalogs[3].0, &["target_layer"][..]),
        (catalogs[4].0, &["nextn"][..]),
    ] {
        assert!(producers_are_private(source, methods));
    }
    let public_catalog = catalogs[0]
        .0
        .replacen("pub(crate) fn dsa(&self", "pub fn dsa(&self", 1);
    assert!(!producers_are_private(&public_catalog, &["kda", "dsa"]));
}

fn formatter_impl<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("impl fmt::Debug for {name}");
    let start = source.find(&marker)?;
    let rest = &source[start..];
    let end = rest.find("\n}\n")? + 2;
    Some(&rest[..end])
}

fn normalized(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn derives_debug(source: &str) -> bool {
    let compact = compact(source);
    let mut rest = compact.as_str();
    while let Some(start) = rest.find("#[derive(") {
        rest = &rest[start + "#[derive(".len()..];
        let Some(end) = rest.find(")]") else {
            return true;
        };
        if rest[..end]
            .split(',')
            .any(|trait_name| trait_name == "Debug")
        {
            return true;
        }
        rest = &rest[end + 2..];
    }
    false
}

fn debug_formatter_contract(source: &str) -> bool {
    const MATRIX: &str = r#"
impl fmt::Debug for Glm53GgufMatrix {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53GgufMatrix")
            .field("kind", &self.kind)
            .field("inner", &self.inner)
            .field("columns", &self.columns)
            .field("bytes", &self.buffer.bytes)
            .finish()
    }
}"#;
    const EXPERTS: &str = r#"
impl fmt::Debug for Glm53GgufExperts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53GgufExperts")
            .field("kind", &self.kind)
            .field("inner", &self.inner)
            .field("columns", &self.columns)
            .field("experts", &self.experts)
            .field("expert_bytes", &self.expert_bytes)
            .field("total_bytes", &self.total_bytes)
            .finish()
    }
}"#;
    const F32: &str = r#"
impl fmt::Debug for Glm53GgufF32 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53GgufF32")
            .field("elements", &self.elements)
            .field("bytes", &self.bytes)
            .finish()
    }
}"#;
    let compact = compact(source);
    if debug_source_sha256::hex(debug_source_sha256::digest(source.as_bytes())) != RAW_SOURCE_SHA256
        || compact.matches("Debugfor").count() != 3
        || compact.contains("#[cfg")
        || derives_debug(source)
    {
        return false;
    }
    [
        ("Glm53GgufMatrix", MATRIX),
        ("Glm53GgufExperts", EXPERTS),
        ("Glm53GgufF32", F32),
    ]
    .iter()
    .all(|(name, expected)| {
        formatter_impl(source, name)
            .is_some_and(|actual| normalized(actual) == normalized(expected))
    })
}

#[test]
fn debug_views_redact_raw_device_addresses() {
    fn rendered(address: u64) -> [String; 8] {
        let matrix =
            Glm53GgufMatrix::new(&tensor(address, &[128, 1], GgmlType::Q8_0, 136)).unwrap();
        let experts =
            Glm53GgufExperts::new(&tensor(address, &[256, 1, 1], GgmlType::IQ3_S, 110)).unwrap();
        let f32_view = Glm53GgufF32::new(&tensor(address, &[4], GgmlType::F32, 16), &[4]).unwrap();
        let raw = [
            format!("{matrix:?}"),
            format!("{matrix:#?}"),
            format!("{experts:?}"),
            format!("{experts:#?}"),
            format!("{f32_view:?}"),
            format!("{f32_view:#?}"),
        ];
        let nested = Glm53LayerNorms {
            attention: f32_view,
            ffn: Glm53GgufF32::new(&tensor(address, &[4], GgmlType::F32, 16), &[4]).unwrap(),
        };
        [
            raw[0].clone(),
            raw[1].clone(),
            raw[2].clone(),
            raw[3].clone(),
            raw[4].clone(),
            raw[5].clone(),
            format!("{nested:?}"),
            format!("{nested:#?}"),
        ]
    }

    let expected = [
        "Glm53GgufMatrix { kind: Q8_0, inner: 128, columns: 1, bytes: 136 }",
        "Glm53GgufMatrix {\n    kind: Q8_0,\n    inner: 128,\n    columns: 1,\n    bytes: 136,\n}",
        "Glm53GgufExperts { kind: IQ3_S, inner: 256, columns: 1, experts: 1, expert_bytes: 110, total_bytes: 110 }",
        "Glm53GgufExperts {\n    kind: IQ3_S,\n    inner: 256,\n    columns: 1,\n    experts: 1,\n    expert_bytes: 110,\n    total_bytes: 110,\n}",
        "Glm53GgufF32 { elements: 4, bytes: 16 }",
        "Glm53GgufF32 {\n    elements: 4,\n    bytes: 16,\n}",
        "Glm53LayerNorms { attention: Glm53GgufF32 { elements: 4, bytes: 16 }, ffn: Glm53GgufF32 { elements: 4, bytes: 16 } }",
        "Glm53LayerNorms {\n    attention: Glm53GgufF32 {\n        elements: 4,\n        bytes: 16,\n    },\n    ffn: Glm53GgufF32 {\n        elements: 4,\n        bytes: 16,\n    },\n}",
    ];
    const SENTINELS: [u64; 3] = [
        0x8bad_c0de_f00d_be00,
        0x0123_4567_89ab_cd7f,
        0x4000_0000_0000_00aa,
    ];
    for address in SENTINELS {
        let actual = rendered(address);
        assert_eq!(actual.as_slice(), expected);
        for debug in actual {
            assert!(!debug.contains("DevicePtr"));
            assert!(!debug.contains("ptr"));
            assert!(!debug.contains("base"));
            assert!(!debug.contains(&address.to_string()));
            assert!(!debug.to_ascii_lowercase().contains(&format!("{address:x}")));
        }
    }
    assert!(debug_formatter_contract(RAW_SOURCE));

    let matrix = RAW_SOURCE.replacen(
        ".field(\"bytes\", &self.buffer.bytes)",
        ".field(\"bytes\", &(self.buffer().ptr.0 & 0xff))",
        1,
    );
    let experts = RAW_SOURCE.replacen(
        "formatter\n            .debug_struct(\"Glm53GgufExperts\")",
        "let Self { base: address, .. } = self;\n        formatter\n            .debug_struct(\"Glm53GgufExperts\")",
        1,
    );
    let experts = experts.replacen(
        ".field(\"total_bytes\", &self.total_bytes)",
        ".field(\"total_bytes\", &(address.0 & 0xff))",
        1,
    );
    let f32_view = RAW_SOURCE.replacen(
        "formatter\n            .debug_struct(\"Glm53GgufF32\")",
        "let Self { ptr: address, .. } = self;\n        formatter\n            .debug_struct(\"Glm53GgufF32\")",
        1,
    );
    let f32_view = f32_view.replacen(
        ".field(\"bytes\", &self.bytes)",
        ".field(\"bytes\", &(address.0 & 0xff))",
        1,
    );
    let canonical_matrix = formatter_impl(RAW_SOURCE, "Glm53GgufMatrix").unwrap();
    let live_conditional_leak = r#"impl fmt::Debug for Glm53GgufMatrix {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = match self.buffer.ptr.0 {
            0x8bad_c0de_f00d_be00 | 0x0123_4567_89ab_cd7f | 0x4000_0000_0000_00aa => self.buffer.bytes,
            address => (address & 0xff) as usize,
        };
        formatter
            .debug_struct("Glm53GgufMatrix")
            .field("kind", &self.kind)
            .field("inner", &self.inner)
            .field("columns", &self.columns)
            .field("bytes", &bytes)
            .finish()
    }
}"#;
    let disabled_canonical =
        format!("#[cfg(any())]\n{canonical_matrix}\n\n{live_conditional_leak}");
    let duplicate_impl = RAW_SOURCE.replacen(canonical_matrix, &disabled_canonical, 1);
    let commented_and_aliased = format!(
        "/*{canonical_matrix}*/\nuse std::fmt::Debug as D;\n{}",
        live_conditional_leak.replacen("impl fmt::Debug", "impl D", 1)
    );
    let aliased_impl = RAW_SOURCE.replacen(canonical_matrix, &commented_and_aliased, 1);
    for mutant in [matrix, experts, f32_view, duplicate_impl, aliased_impl] {
        assert!(!debug_formatter_contract(&mutant));
    }
}

fn tensor(ptr: u64, dimensions: &[u64], ggml_type: GgmlType, byte_len: usize) -> GgufDeviceTensor {
    GgufDeviceTensor {
        ptr: DevicePtr(ptr),
        dimensions: dimensions.to_vec(),
        ggml_type,
        byte_len,
    }
}

#[test]
fn dense_views_validate_rank_geometry_extent_and_address() {
    let q8 = tensor(0x1000, &[128, 3], GgmlType::Q8_0, 3 * 4 * 34);
    let view = Glm53GgufMatrix::new(&q8).unwrap();
    assert_eq!(view.kind(), GgmlType::Q8_0);
    assert_eq!((view.inner(), view.columns()), (128, 3));
    assert_eq!(view.buffer().ptr, DevicePtr(0x1000));
    let plan = view.plan(2).unwrap();
    // K=128 pads to a full MMQ_ITER_K=256 of activations per row: the MMQ tile
    // loop always consumes 256 K, so the tail block must exist and be zeroed.
    // Pinned against the gate rather than deleted, so the deliberate-regression
    // build asserts the pre-fix extent instead of silently failing.
    if crate::layers::ops::MMQ_ACTIVATION_PADDING {
        assert_eq!(plan.activation_bytes, 2 * 2 * 144);
    } else {
        assert_eq!(plan.activation_bytes, 2 * 1 * 144);
    }
    assert_eq!(plan.output_bytes, 2 * 3 * 2);

    assert!(Glm53GgufMatrix::new(&tensor(1, &[128], GgmlType::Q8_0, 136)).is_err());
    assert!(Glm53GgufMatrix::new(&tensor(1, &[128, 1], GgmlType::Q8_0, 135)).is_err());
    assert!(Glm53GgufMatrix::new(&tensor(1, &[128, 1], GgmlType::F32, 512)).is_err());
    assert!(Glm53GgufMatrix::new(&tensor(0, &[128, 1], GgmlType::Q8_0, 136)).is_err());
    assert!(
        Glm53GgufMatrix::new(&tensor(u64::MAX - 100, &[128, 1], GgmlType::Q8_0, 136,)).is_err()
    );
}

#[test]
fn packed_expert_views_slice_exact_contiguous_matrices() {
    let expert_bytes = 128 * 110;
    let packed = tensor(0x20_000, &[256, 128, 4], GgmlType::IQ3_S, expert_bytes * 4);
    let experts = Glm53GgufExperts::new(&packed).unwrap();
    assert_eq!(experts.len(), 4);
    assert!(!experts.is_empty());
    for index in 0..4 {
        let view = experts.expert(index).unwrap();
        assert_eq!(
            view.buffer().ptr,
            DevicePtr(0x20_000 + index as u64 * expert_bytes as u64)
        );
        assert_eq!(view.buffer().bytes, expert_bytes);
        assert_eq!((view.inner(), view.columns()), (256, 128));
    }
    assert!(experts.expert(4).is_err());
}

#[test]
fn packed_expert_admission_rejects_hostile_metadata() {
    let bytes = 128 * 110;
    assert!(Glm53GgufExperts::new(&tensor(1, &[256, 128], GgmlType::IQ3_S, bytes)).is_err());
    assert!(Glm53GgufExperts::new(&tensor(1, &[256, 128, 0], GgmlType::IQ3_S, 0)).is_err());
    assert!(Glm53GgufExperts::new(&tensor(1, &[128, 1, 2], GgmlType::IQ3_S, 220)).is_err());
    assert!(
        Glm53GgufExperts::new(&tensor(1, &[256, 128, 2], GgmlType::IQ3_S, bytes * 2 - 1)).is_err()
    );
    assert!(
        Glm53GgufExperts::new(&tensor(
            u64::MAX - 100,
            &[256, 128, 2],
            GgmlType::IQ3_S,
            bytes * 2,
        ))
        .is_err()
    );
}

#[test]
fn matrix_banks_cover_dsa_heads_and_f32_views_are_exact() {
    let head_bytes = 512 * (256 / 32) * 34;
    let heads = Glm53GgufMatrixBank::new(&tensor(
        0x40_000,
        &[256, 512, 64],
        GgmlType::Q8_0,
        head_bytes * 64,
    ))
    .unwrap();
    assert_eq!(heads.len(), 64);
    assert_eq!(heads.expert(63).unwrap().buffer().bytes, head_bytes);

    let raw = tensor(0x80_000, &[4, 1, 8192], GgmlType::F32, 4 * 8192 * 4);
    let view = Glm53GgufF32::new(&raw, &[4, 1, 8192]).unwrap();
    assert_eq!(view.ptr(), DevicePtr(0x80_000));
    assert_eq!(view.elements(), 4 * 8192);
    assert_eq!(view.bytes(), 4 * 8192 * 4);
    assert!(Glm53GgufF32::new(&raw, &[8192, 4]).is_err());
    assert!(Glm53GgufF32::new(&tensor(1, &[4], GgmlType::Q8_0, 136), &[4]).is_err());
    assert!(Glm53GgufF32::new(&tensor(1, &[4], GgmlType::F32, 15), &[4]).is_err());
    assert!(Glm53GgufF32::new(&tensor(0, &[4], GgmlType::F32, 16), &[4]).is_err());
    assert!(Glm53GgufF32::new(&tensor(u64::MAX - 8, &[4], GgmlType::F32, 16), &[4]).is_err());
}
