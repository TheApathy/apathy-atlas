// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;

use anyhow::{Context, Result, bail, ensure};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

use super::admission::W3SidecarRequest;
use crate::layers::ops::W3_GROUP_SIZE;

const MAX_HEADER_BYTES: usize = 128 << 20;

#[derive(Clone, Debug)]
pub(super) struct ProjectionPlan {
    pub packed: Range<usize>,
    pub scales: Range<usize>,
    pub scale2_bytes: Range<usize>,
    pub scale2: f32,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Debug)]
pub(super) struct LayerPlan {
    pub gate: ProjectionPlan,
    pub up: ProjectionPlan,
    pub down: ProjectionPlan,
}

pub(super) fn validate_manifest(
    artifact: &[u8],
    request: &W3SidecarRequest,
    layer_prefixes: &[String],
    hidden: usize,
    intermediate: usize,
) -> Result<BTreeMap<usize, LayerPlan>> {
    ensure!(hidden > 0 && hidden.is_multiple_of(W3_GROUP_SIZE));
    ensure!(intermediate > 0 && intermediate.is_multiple_of(W3_GROUP_SIZE));
    let header_len_bytes = artifact
        .get(..8)
        .context("W3 sidecar is shorter than the safetensors length prefix")?;
    let header_len = usize::try_from(u64::from_le_bytes(header_len_bytes.try_into().unwrap()))
        .context("W3 sidecar header length does not fit this host")?;
    ensure!(header_len > 0, "W3 sidecar header is empty");
    ensure!(
        header_len <= MAX_HEADER_BYTES,
        "W3 sidecar header exceeds {MAX_HEADER_BYTES} bytes"
    );
    let data_start = 8_usize
        .checked_add(header_len)
        .context("W3 sidecar header offset overflow")?;
    let header_bytes = artifact
        .get(8..data_start)
        .context("W3 sidecar header extends beyond the artifact")?;
    let header: UniqueJson =
        serde_json::from_slice(header_bytes).context("parse duplicate-free W3 header JSON")?;
    let root = header.object("W3 sidecar header")?;

    let mut layers = BTreeMap::new();
    let mut all_ranges = Vec::new();
    for &layer in request.layers() {
        let layer_prefix = layer_prefixes
            .get(layer)
            .with_context(|| format!("W3 layer {layer} has no model prefix"))?;
        ensure!(!layer_prefix.is_empty(), "W3 layer {layer} prefix is empty");
        let mlp_prefix = format!("{layer_prefix}.mlp.");
        let expected = expected_names(layer_prefix);
        let actual = root
            .keys()
            .filter(|name| name.starts_with(&mlp_prefix))
            .cloned()
            .collect::<BTreeSet<_>>();
        ensure!(
            actual == expected,
            "W3 layer {layer} tensor census differs: expected {expected:?}, got {actual:?}"
        );

        let gate = projection_plan(
            root,
            artifact,
            data_start,
            &format!("{layer_prefix}.mlp.gate_proj"),
            intermediate,
            hidden,
        )?;
        let up = projection_plan(
            root,
            artifact,
            data_start,
            &format!("{layer_prefix}.mlp.up_proj"),
            intermediate,
            hidden,
        )?;
        let down = projection_plan(
            root,
            artifact,
            data_start,
            &format!("{layer_prefix}.mlp.down_proj"),
            hidden,
            intermediate,
        )?;
        for projection in [&gate, &up, &down] {
            all_ranges.push(projection.packed.clone());
            all_ranges.push(projection.scales.clone());
            all_ranges.push(projection.scale2_bytes.clone());
        }
        layers.insert(layer, LayerPlan { gate, up, down });
    }
    ensure!(
        layers.keys().copied().collect::<BTreeSet<_>>() == *request.layers(),
        "W3 validated layer census differs from the request"
    );
    all_ranges.sort_by_key(|range| (range.start, range.end));
    for pair in all_ranges.windows(2) {
        ensure!(
            pair[0].end <= pair[1].start,
            "W3 requested tensor payloads overlap at {:?} and {:?}",
            pair[0],
            pair[1]
        );
    }
    Ok(layers)
}

fn expected_names(layer_prefix: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        for suffix in ["w3_weight", "w3_weight_scale", "w3_weight_scale_2"] {
            names.insert(format!("{layer_prefix}.mlp.{projection}.{suffix}"));
        }
    }
    names
}

fn projection_plan(
    root: &BTreeMap<String, UniqueJson>,
    artifact: &[u8],
    data_start: usize,
    prefix: &str,
    n: usize,
    k: usize,
) -> Result<ProjectionPlan> {
    ensure!(k.is_multiple_of(8), "{prefix}: K={k} is not W3-packable");
    let row_bytes = k
        .checked_div(8)
        .and_then(|octets| octets.checked_mul(3))
        .context("W3 packed row-byte count overflow")?;
    let groups = k / W3_GROUP_SIZE;
    let packed = tensor_range(
        root,
        artifact.len(),
        data_start,
        &format!("{prefix}.w3_weight"),
        "U8",
        &[n, row_bytes],
    )?;
    let scales = tensor_range(
        root,
        artifact.len(),
        data_start,
        &format!("{prefix}.w3_weight_scale"),
        "U8",
        &[n, groups],
    )?;
    let scale2_range = tensor_range(
        root,
        artifact.len(),
        data_start,
        &format!("{prefix}.w3_weight_scale_2"),
        "F32",
        &[1],
    )?;
    let scale2_bytes: [u8; 4] = artifact[scale2_range.clone()]
        .try_into()
        .expect("validated F32 scalar has exactly four bytes");
    let scale2 = f32::from_le_bytes(scale2_bytes);
    ensure!(
        scale2.is_finite() && scale2 > 0.0,
        "{prefix}.w3_weight_scale_2 must be finite and positive"
    );
    Ok(ProjectionPlan {
        packed,
        scales,
        scale2_bytes: scale2_range,
        scale2,
        n,
        k,
    })
}

fn tensor_range(
    root: &BTreeMap<String, UniqueJson>,
    artifact_len: usize,
    data_start: usize,
    name: &str,
    dtype: &str,
    shape: &[usize],
) -> Result<Range<usize>> {
    let metadata = root
        .get(name)
        .with_context(|| format!("W3 sidecar is missing {name}"))?
        .object(name)?;
    ensure!(
        metadata.len() == 3
            && metadata.contains_key("dtype")
            && metadata.contains_key("shape")
            && metadata.contains_key("data_offsets"),
        "{name}: tensor metadata must contain exactly dtype, shape, and data_offsets"
    );
    ensure!(
        metadata["dtype"].string(name)? == dtype,
        "{name}: wrong dtype"
    );
    let actual_shape = metadata["shape"].usize_array(&format!("{name}.shape"))?;
    ensure!(
        actual_shape == shape,
        "{name}: wrong shape {actual_shape:?}"
    );
    let offsets = metadata["data_offsets"].usize_array(&format!("{name}.data_offsets"))?;
    ensure!(
        offsets.len() == 2,
        "{name}: data_offsets must have length two"
    );
    ensure!(offsets[0] <= offsets[1], "{name}: descending data offsets");
    let start = data_start
        .checked_add(offsets[0])
        .context("W3 tensor start offset overflow")?;
    let end = data_start
        .checked_add(offsets[1])
        .context("W3 tensor end offset overflow")?;
    ensure!(
        end <= artifact_len,
        "{name}: tensor payload is out of bounds"
    );
    let elements = shape.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .context("W3 tensor element-count overflow")
    })?;
    let bytes_per_element = match dtype {
        "U8" => 1,
        "F32" => 4,
        _ => unreachable!("caller admits only W3 dtypes"),
    };
    let expected_len = elements
        .checked_mul(bytes_per_element)
        .context("W3 tensor byte-count overflow")?;
    ensure!(
        end - start == expected_len,
        "{name}: tensor span has the wrong length"
    );
    Ok(start..end)
}

#[derive(Debug)]
enum UniqueJson {
    Null,
    Bool,
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(BTreeMap<String, Self>),
}

impl UniqueJson {
    fn object(&self, label: &str) -> Result<&BTreeMap<String, Self>> {
        match self {
            Self::Object(value) => Ok(value),
            _ => bail!("{label} must be a JSON object"),
        }
    }

    fn string(&self, label: &str) -> Result<&str> {
        match self {
            Self::String(value) => Ok(value),
            _ => bail!("{label} must be a JSON string"),
        }
    }

    fn usize_array(&self, label: &str) -> Result<Vec<usize>> {
        let Self::Array(values) = self else {
            bail!("{label} must be a JSON array");
        };
        values
            .iter()
            .map(|value| match value {
                Self::Number(number) => number
                    .as_u64()
                    .context("expected an unsigned integer")
                    .and_then(|raw| usize::try_from(raw).context("integer does not fit this host")),
                _ => bail!("{label} entries must be unsigned integers"),
            })
            .collect()
    }
}

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = UniqueJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("duplicate-free JSON")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        let _ = value;
        Ok(UniqueJson::Bool)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJson::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJson::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(UniqueJson::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueJson::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueJson::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJson::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJson::Null)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element()? {
            values.push(value);
        }
        Ok(UniqueJson::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = BTreeMap::new();
        while let Some((key, value)) = map.next_entry()? {
            if values.insert(key, value).is_some() {
                return Err(serde::de::Error::custom("duplicate JSON object key"));
            }
        }
        Ok(UniqueJson::Object(values))
    }
}
