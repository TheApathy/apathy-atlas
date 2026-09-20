// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use serde_json::{Value, json};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
const PRODUCER_SCHEMA: &str = "qwen38-ssm-qkvz-generated-weight-producer-receipt-v1";
const MANIFEST_SCHEMA: &str = "qwen38-ssm-qkvz-generated-weight-v1";
#[rustfmt::skip]
pub(super) struct OutputPaths { pub packed: PathBuf, pub scales: PathBuf, pub receipt: PathBuf, pub manifest: PathBuf }
#[rustfmt::skip]
fn new_path(name: &str) -> Result<PathBuf> {
    let raw = PathBuf::from(required_env(name)?); ensure!(raw.is_absolute(), "{name} must be absolute");
    let leaf = raw.file_name().context("output path lacks a file name")?;
    let path = raw.parent().context("output path lacks a parent")?.canonicalize()?.join(leaf);
    ensure!(!path.exists(), "{name} must name a new file"); Ok(path)
}
#[rustfmt::skip]
impl OutputPaths {
    pub fn from_env() -> Result<Self> {
        let value = Self { packed: new_path("ATLAS_SSM_QKVZ_WEIGHT_PACKED")?, scales: new_path("ATLAS_SSM_QKVZ_WEIGHT_SCALES")?,
            receipt: new_path("ATLAS_SSM_QKVZ_PRODUCER_RECEIPT")?, manifest: new_path("ATLAS_SSM_QKVZ_WEIGHT_MANIFEST")? };
        ensure!([&value.packed,&value.scales,&value.receipt,&value.manifest].into_iter()
            .collect::<std::collections::BTreeSet<_>>().len() == 4, "producer output paths must be distinct"); Ok(value)
    }
}
#[rustfmt::skip]
pub(super) fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?;
    file.write_all(bytes)?; file.sync_all()?; std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o444))?; Ok(())
}
#[rustfmt::skip]
pub(super) fn write_json_immutable(path: &Path, value: &Value) -> Result<Vec<u8>> {
    let mut bytes=serde_json::to_vec(value)?; bytes.push(b'\n'); write_immutable(path,&bytes)?; Ok(bytes)
}
#[rustfmt::skip]
fn read_bounded_immutable(path: &Path, limit: u64, mode: u32) -> Result<Vec<u8>> {
    ensure!(path.is_absolute(), "evidence path must be absolute"); let path = path.canonicalize()?;
    let mut file = File::open(path)?; let metadata = file.metadata()?;
    ensure!(metadata.is_file() && metadata.len() <= limit, "invalid evidence file"); ensure!(metadata.mode() & 0o777 == mode, "evidence mode mismatch");
    let mut bytes = Vec::with_capacity(metadata.len() as usize); std::io::Read::by_ref(&mut file).take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 == metadata.len(), "evidence changed while reading"); Ok(bytes)
}
#[rustfmt::skip]
struct Tensor { bytes: Vec<u8>, dtype: String, source: Value }
type Expected<'a> = (&'a str, &'a [usize], usize);
#[rustfmt::skip]
fn identity(path: &Path, metadata: &std::fs::Metadata) -> Value {
    json!({"path":path,"device":metadata.dev(),"inode":metadata.ino(),"file_bytes":metadata.len(),"mode":metadata.mode() & 0o777})
}
#[rustfmt::skip]
fn identity_matches(value: &Value, metadata: &std::fs::Metadata) -> bool {
    metadata.is_file() && value.get("device").and_then(Value::as_u64) == Some(metadata.dev())
        && value.get("inode").and_then(Value::as_u64) == Some(metadata.ino())
        && value.get("file_bytes").and_then(Value::as_u64) == Some(metadata.len())
        && value.get("mode").and_then(Value::as_u64) == Some(u64::from(metadata.mode() & 0o777))
}
#[rustfmt::skip]
fn stable_path_hash(path: &Path, expected: &Value) -> Result<String> {
    let before = File::open(path)?.metadata()?; ensure!(identity_matches(expected,&before), "checkpoint source identity drift");
    let hash = sha256_file(path)?; let after = File::open(path)?.metadata()?;
    ensure!(identity_matches(expected,&after), "checkpoint source replaced during hash"); Ok(hash)
}
#[rustfmt::skip]
pub(super) fn stable_checkpoint_index(checkpoint: &Checkpoint) -> Result<(String,Value)> {
    let path=checkpoint.root.join("model.safetensors.index.json").canonicalize()?; let mut file=File::open(&path)?; let metadata=file.metadata()?;
    ensure!(metadata.is_file() && metadata.len() <= 128<<20, "invalid checkpoint index"); let id=identity(&path,&metadata); let mut bytes=Vec::with_capacity(metadata.len() as usize);
    std::io::Read::by_ref(&mut file).take((128<<20)+1).read_to_end(&mut bytes)?; ensure!(bytes.len() as u64 == metadata.len() && identity_matches(&id,&file.metadata()?), "checkpoint index changed");
    let parsed: Value=serde_json::from_slice(&bytes)?; ensure!(parsed.get("weight_map").and_then(Value::as_object) == Some(&checkpoint.index), "checkpoint index snapshot drift");
    Ok((sha256_bytes(&bytes)?,id))
}
#[rustfmt::skip]
fn admit_descriptor(dtype: &str, shape: &[usize], start: u64, end: u64, file_len: u64, expected: &[Expected<'_>]) -> Result<usize> {
    ensure!(end >= start, "reversed offsets"); let len = end-start;
    let exact = expected.iter().find(|(d,s,b)| *d == dtype && *s == shape && *b as u64 == len).context("unexpected tensor dtype/shape/extent")?;
    ensure!(end <= file_len, "tensor out of range"); Ok(exact.2)
}
#[rustfmt::skip]
fn tensor(checkpoint: &Checkpoint, name: &str, expected: &[Expected<'_>]) -> Result<Tensor> {
    let shard = checkpoint.index.get(name).and_then(Value::as_str).with_context(|| format!("index missing {name}"))?;
    let path = checkpoint.root.join(shard).canonicalize()?; ensure!(path.starts_with(&checkpoint.root), "checkpoint shard escapes root");
    let mut file = File::open(&path)?; let metadata = file.metadata()?; ensure!(metadata.is_file(), "checkpoint shard is not regular");
    let source_identity = identity(&path,&metadata); let mut len8 = [0; 8]; file.read_exact(&mut len8)?; let header_len = u64::from_le_bytes(len8);
    ensure!(header_len <= 128 << 20, "safetensors header too large"); let mut header = vec![0; header_len as usize]; file.read_exact(&mut header)?;
    let root: Value = serde_json::from_slice(&header)?; let meta = root.get(name).with_context(|| format!("shard header missing {name}"))?;
    let dtype = meta.get("dtype").and_then(Value::as_str).context("missing dtype")?.to_owned();
    let shape = meta.get("shape").and_then(Value::as_array).context("missing shape")?.iter()
        .map(|v| usize::try_from(v.as_u64().context("bad shape")?).context("shape overflow")).collect::<Result<Vec<_>>>()?;
    let offsets = meta.get("data_offsets").and_then(Value::as_array).context("missing offsets")?; ensure!(offsets.len() == 2, "bad offsets");
    let start = offsets[0].as_u64().context("bad start")?; let end = offsets[1].as_u64().context("bad end")?;
    let data_start = 8u64.checked_add(header_len).context("header overflow")?;
    let relative_file_len = metadata.len().checked_sub(data_start).context("header exceeds shard")?;
    let exact_bytes = admit_descriptor(&dtype,&shape,start,end,relative_file_len,expected)?;
    let absolute = data_start.checked_add(start).context("offset overflow")?;
    let mut bytes = vec![0; exact_bytes]; file.seek(SeekFrom::Start(absolute))?; file.read_exact(&mut bytes)?;
    ensure!(identity_matches(&source_identity,&file.metadata()?), "checkpoint source mutated during read");
    let tensor_sha256 = sha256_bytes(&bytes)?; let shard_sha256 = stable_path_hash(&path,&source_identity)?;
    let source = json!({"identity":source_identity,"shard":shard,"tensor":name,"dtype":dtype,"shape":shape,"start":start,"end":end,
        "tensor_sha256":tensor_sha256,"shard_sha256":shard_sha256}); Ok(Tensor { bytes, dtype, source })
}
#[rustfmt::skip]
fn scalar_f32(checkpoint: &Checkpoint, name: &str) -> Result<(f32,Value)> {
    let value = tensor(checkpoint, name, &[("F32",&[],4),("F32",&[1],4)])?;
    let scalar = f32::from_le_bytes(value.bytes.try_into().unwrap()); ensure!(scalar.is_finite() && scalar > 0.0, "invalid scalar {name}"); Ok((scalar,value.source))
}
#[rustfmt::skip]
fn fp8(bits: u8) -> f32 {
    let sign = bits >> 7; let exp = (bits >> 3) & 15; let mantissa = bits & 7;
    let value = if exp == 0 { mantissa as f32 * (0.015625 / 8.0) } else if exp == 15 && mantissa == 7 { 0.0 }
        else { f32::from_bits(((exp as u32 + 120) << 23) | ((mantissa as u32) << 20)) };
    if sign == 1 { -value } else { value }
}
#[rustfmt::skip]
fn dequant_part(checkpoint: &Checkpoint, prefix: &str, n: usize) -> Result<(Vec<u8>,Vec<Value>)> {
    let compressed = checkpoint.index.contains_key(&format!("{prefix}.weight_packed"));
    let packed_name = format!("{prefix}.{}", if compressed { "weight_packed" } else { "weight" });
    let packed_expect = if compressed { vec![("U8",vec![n,K/2],n*K/2)] } else { vec![("BF16",vec![n,K],n*K*2),("U8",vec![n,K/2],n*K/2)] };
    let packed_specs: Vec<_> = packed_expect.iter().map(|(d,s,b)| (*d,s.as_slice(),*b)).collect();
    let packed = tensor(checkpoint,&packed_name,&packed_specs)?;
    if packed.dtype == "BF16" { return Ok((packed.bytes,vec![packed.source])); }
    let scale_name = format!("{prefix}.weight_scale");
    let scales = tensor(checkpoint,&scale_name,&[("F8_E4M3",&[n,K/16],n*K/16),("F8_E4M3FN",&[n,K/16],n*K/16)])?;
    let (scalar,scalar_source) = scalar_f32(checkpoint, &format!("{prefix}.{}", if compressed { "weight_global_scale" } else { "weight_scale_2" }))?;
    let e2m1: [f32;16] = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0];
    let mut output = Vec::with_capacity(n*K*2);
    for (group, scale) in scales.bytes.iter().copied().enumerate() { let combined = if compressed { fp8(scale)/scalar } else { fp8(scale)*scalar };
        for index in group*16..group*16+16 { let byte = packed.bytes[index/2]; let nibble = if index&1 == 0 { byte&15 } else { byte>>4 };
            output.extend_from_slice(&((e2m1[nibble as usize]*combined).to_bits() >> 16).to_le_bytes()[..2]); } }
    Ok((output,vec![packed.source,scales.source,scalar_source]))
}
#[rustfmt::skip]
pub(super) fn load_qkvz_bf16(checkpoint: &Checkpoint, layer: usize) -> Result<(Vec<u8>,Vec<Value>)> {
    ensure!(layer < 64, "layer must be in 0..64"); let base = format!("model.language_model.layers.{layer}.linear_attn");
    let (mut output,mut sources) = dequant_part(checkpoint, &format!("{base}.in_proj_qkv"), 10_240)?;
    let (z,z_sources) = dequant_part(checkpoint, &format!("{base}.in_proj_z"), 6_144)?; output.extend_from_slice(&z); sources.extend(z_sources);
    ensure!(output.len() == N*K*2, "QKVZ BF16 extent mismatch"); Ok((output,sources))
}
#[rustfmt::skip]
fn expected_for(name: &str, layer: usize) -> Result<Vec<(&'static str,Vec<usize>,usize)>> {
    let base = format!("model.language_model.layers.{layer}.linear_attn."); let rest = name.strip_prefix(&base).context("foreign tensor source")?;
    let (n,suffix) = if let Some(v)=rest.strip_prefix("in_proj_qkv.") {(10_240,v)} else if let Some(v)=rest.strip_prefix("in_proj_z.") {(6_144,v)} else { bail!("foreign tensor source") };
    Ok(match suffix { "weight_packed" => vec![("U8",vec![n,K/2],n*K/2)],
        "weight" => vec![("BF16",vec![n,K],n*K*2),("U8",vec![n,K/2],n*K/2)],
        "weight_scale" => vec![("F8_E4M3",vec![n,K/16],n*K/16),("F8_E4M3FN",vec![n,K/16],n*K/16)],
        "weight_scale_2" | "weight_global_scale" => vec![("F32",vec![],4),("F32",vec![1],4)], _ => bail!("foreign tensor suffix") })
}
#[rustfmt::skip]
fn validate_source_set(layer: usize, sources: &[Value]) -> Result<()> {
    let base=format!("model.language_model.layers.{layer}.linear_attn."); let mut sets: std::collections::BTreeMap<&str,std::collections::BTreeSet<String>>=std::collections::BTreeMap::new(); let mut names=std::collections::BTreeSet::new();
    sets.insert("in_proj_qkv",std::collections::BTreeSet::new()); sets.insert("in_proj_z",std::collections::BTreeSet::new());
    for source in sources { let name=source.get("tensor").and_then(Value::as_str).context("tensor source name")?;
        let rest=name.strip_prefix(&base).context("foreign tensor source")?; let (projection,suffix)=rest.split_once('.').context("bad tensor source")?;
        let set=sets.get_mut(projection).context("foreign projection")?; ensure!(names.insert(name), "duplicate tensor source");
        let dtype=source.get("dtype").and_then(Value::as_str).context("tensor source dtype")?; let dtype=if matches!(dtype,"F8_E4M3"|"F8_E4M3FN") {"E4M3"} else {dtype};
        set.insert(format!("{suffix}:{dtype}")); }
    for set in sets.values() { let valid=set.len()==1 && set.contains("weight:BF16")
        || set.len()==3 && set.contains("weight:U8") && set.contains("weight_scale:E4M3") && set.contains("weight_scale_2:F32")
        || set.len()==3 && set.contains("weight_packed:U8") && set.contains("weight_scale:E4M3") && set.contains("weight_global_scale:F32");
        ensure!(valid,"incomplete or mixed projection source set"); } Ok(())
}
#[rustfmt::skip]
pub(super) fn recheck_tensor_sources(checkpoint: &Checkpoint, layer: usize, sources: &[Value]) -> Result<()> {
    validate_source_set(layer,sources)?;
    for projection in ["in_proj_qkv","in_proj_z"] { let packed=format!("model.language_model.layers.{layer}.linear_attn.{projection}.weight_packed");
        ensure!(checkpoint.index.contains_key(&packed) == sources.iter().any(|v| v.get("tensor").and_then(Value::as_str)==Some(&packed)), "source set disagrees with checkpoint format"); }
    for source in sources { let name=source.get("tensor").and_then(Value::as_str).context("tensor source name")?;
        let owned=expected_for(name,layer)?;
        let expected: Vec<_>=owned.iter().map(|(d,s,b)|(*d,s.as_slice(),*b)).collect();
        ensure!(tensor(checkpoint,name,&expected)?.source == *source, "checkpoint tensor source changed"); }
    Ok(())
}
#[rustfmt::skip]
pub(super) fn bind_tensor_shards(sources: &[Value], shards: &std::collections::BTreeMap<String,String>) -> Result<()> {
    for source in sources { let shard=source.get("shard").and_then(Value::as_str).context("source shard")?;
        ensure!(source.get("shard_sha256").and_then(Value::as_str) == shards.get(shard).map(String::as_str), "tensor/shard hash mismatch"); } Ok(())
}
#[rustfmt::skip]
fn field<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value.get(name).and_then(Value::as_str).with_context(|| format!("missing {name}"))
}
#[rustfmt::skip]
pub(super) fn verify_consumer_inputs(checkpoint: &Checkpoint, layer: usize, executable: &Value,
    sources: &std::collections::BTreeMap<String,String>, bundle: &Value) -> Result<Value> {
    let manifest_path = PathBuf::from(required_env("ATLAS_SSM_QKVZ_WEIGHT_MANIFEST")?);
    let manifest: Value = serde_json::from_slice(&read_bounded_immutable(&manifest_path, 1 << 20, 0o444)?)?;
    ensure!(manifest.get("schema").and_then(Value::as_str) == Some(MANIFEST_SCHEMA), "manifest schema mismatch");
    ensure!(manifest.get("layer").and_then(Value::as_u64) == Some(layer as u64) && manifest.get("n").and_then(Value::as_u64) == Some(N as u64)
        && manifest.get("k").and_then(Value::as_u64) == Some(K as u64), "manifest geometry mismatch");
    let receipt_bytes = read_bounded_immutable(&PathBuf::from(field(&manifest,"producer_receipt_path")?), 1 << 20, 0o444)?;
    ensure!(sha256_bytes(&receipt_bytes)? == field(&manifest,"producer_receipt_sha256")?, "producer receipt hash mismatch");
    let receipt: Value = serde_json::from_slice(&receipt_bytes)?;
    ensure!(receipt.get("schema").and_then(Value::as_str) == Some(PRODUCER_SCHEMA), "producer receipt schema mismatch");
    ensure!(receipt.get("source_sha256") == Some(&serde_json::to_value(sources)?), "producer source identity drift");
    ensure!(receipt.get("embedded_bundle") == Some(bundle), "producer bundle identity drift");
    for name in ["checkpoint_index_sha256","checkpoint_index_identity","checkpoint_source_shards","checkpoint_tensor_sources","layer","n","k","packed_path","packed_sha256",
        "logical_scales_path","logical_scales_sha256","weight_scale2_bits","executable"] { ensure!(manifest.get(name) == receipt.get(name), "producer handoff mismatch: {name}"); }
    ensure!(manifest.get("executable") == Some(executable), "producer and consumer executable identity differ");
    ensure!(field(&manifest,"producer_executable_path")? == field(executable,"path")?
        && field(&manifest,"producer_executable_sha256")? == field(executable,"sha256")?, "producer executable fields differ"); recheck_executable(executable)?;
    let (index_hash,index_identity)=stable_checkpoint_index(checkpoint)?;
    ensure!(field(&manifest,"checkpoint_index_sha256")? == index_hash && manifest.get("checkpoint_index_identity") == Some(&index_identity), "checkpoint index drift");
    ensure!(manifest.get("checkpoint_source_shards") == Some(&serde_json::to_value(checkpoint_shards(checkpoint,layer)?)?), "checkpoint shard drift");
    let tensor_sources=receipt.get("checkpoint_tensor_sources").and_then(Value::as_array).context("missing tensor sources")?;
    recheck_tensor_sources(checkpoint,layer,tensor_sources)?;
    let shard_map: std::collections::BTreeMap<String,String> = serde_json::from_value(receipt["checkpoint_source_shards"].clone())?;
    bind_tensor_shards(tensor_sources,&shard_map)?;
    for (env_name,manifest_name) in [("ATLAS_SSM_QKVZ_WEIGHT_PACKED","packed_path"),("ATLAS_SSM_QKVZ_WEIGHT_SCALES","logical_scales_path")] {
        ensure!(PathBuf::from(required_env(env_name)?).canonicalize()? == PathBuf::from(field(&manifest,manifest_name)?).canonicalize()?, "artifact path mismatch"); }
    Ok(manifest)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] #[rustfmt::skip]
    fn fp8_and_schemas_are_exact() {
        assert_eq!(fp8(0x7e),448.0); assert_eq!(fp8(0xfe),-448.0); assert_eq!(fp8(0x7f),0.0); assert_ne!(PRODUCER_SCHEMA,MANIFEST_SCHEMA);
        let source=include_str!("producer_io.rs"); for marker in ["create_new(true)","mode(0o444)","producer receipt hash mismatch",
            "producer and consumer executable identity differ","recheck_executable(executable)"] { assert!(source.contains(marker)); }
    }
    #[test]
    #[rustfmt::skip]
    fn exact_projection_source_sets_only() {
        let s=|part:&str,suffix:&str,dtype:&str| json!({"tensor":format!("model.language_model.layers.7.linear_attn.{part}.{suffix}"),"dtype":dtype});
        let bf=vec![s("in_proj_qkv","weight","BF16"),s("in_proj_z","weight","BF16")]; assert!(validate_source_set(7,&bf).is_ok());
        let mut mixed=bf.clone(); mixed.push(s("in_proj_qkv","weight_scale","F8_E4M3")); assert!(validate_source_set(7,&mixed).is_err());
        assert!(validate_source_set(7,&bf[..1]).is_err()); let mut duplicate=bf.clone(); duplicate.push(bf[0].clone()); assert!(validate_source_set(7,&duplicate).is_err());
        let mut extra=vec![s("in_proj_qkv","weight","U8"),s("in_proj_qkv","weight_scale","F8_E4M3FN"),s("in_proj_qkv","weight_scale_2","F32"),
            s("in_proj_z","weight_packed","U8"),s("in_proj_z","weight_scale","F8_E4M3"),s("in_proj_z","weight_global_scale","F32")]; assert!(validate_source_set(7,&extra).is_ok());
        extra.push(s("in_proj_z","weight_scale_2","F32")); assert!(validate_source_set(7,&extra).is_err());
    }
    #[test]
    #[rustfmt::skip]
    fn hostile_extent_and_scalar_admission_precede_allocation() {
        let scalar: &[Expected<'_>] = &[("F32",&[],4),("F32",&[1],4)];
        assert_eq!(admit_descriptor("F32",&[],0,4,4,scalar).unwrap(),4);
        assert!(admit_descriptor("F32",&[],0,u64::MAX,u64::MAX,scalar).is_err());
        assert!(admit_descriptor("F32",&[2],0,4,4,scalar).is_err());
        assert!(admit_descriptor("F32",&[],0,5,5,scalar).is_err());
    }
    #[test]
    #[rustfmt::skip]
    fn hostile_mutation_and_replacement_are_detected() {
        let root=std::env::temp_dir().join(format!("atlas-qkvz-source-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap(); let path=root.join("shard"); std::fs::write(&path,b"first").unwrap();
        let id=identity(&path,&File::open(&path).unwrap().metadata().unwrap()); let first=stable_path_hash(&path,&id).unwrap();
        std::fs::write(&path,b"other").unwrap(); assert_ne!(stable_path_hash(&path,&id).unwrap(),first);
        let old=root.join("old"); std::fs::rename(&path,&old).unwrap(); std::fs::write(&path,b"first").unwrap();
        assert!(stable_path_hash(&path,&id).is_err()); std::fs::remove_dir_all(root).unwrap();
    }
}
