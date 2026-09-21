// SPDX-License-Identifier: AGPL-3.0-only
//! Actual WeightStore metadata admission; no model/GPU/weight payload I/O.
#[path = "../src/weight_loader/deepseek_v4/indexer.rs"]
mod indexer;
use atlas_core::config::{ModelConfig, parse_config};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
use std::collections::HashMap;

fn config() -> ModelConfig {
    parse_config(
        r#"{"model_type":"deepseek_v4","hidden_size":4096,
      "num_hidden_layers":4,"num_attention_heads":64,"num_key_value_heads":1,
      "head_dim":512,"vocab_size":129280,"max_position_embeddings":8192,
      "q_lora_rank":1024,"o_lora_rank":1024,"qk_rope_head_dim":64,
      "compress_ratios":[0,0,4,128],"index_n_heads":64,
      "index_head_dim":128,"index_topk":512}"#,
    )
    .unwrap()
}

fn tensors() -> HashMap<String, WeightTensor> {
    use WeightDtype::*;
    [
        ("wq_b.weight", FP8E4M3, vec![8192, 1024]),
        ("wq_b.scale", FP8E8M0, vec![64, 8]),
        ("weights_proj.weight", BF16, vec![64, 4096]),
        ("compressor.wkv.weight", BF16, vec![256, 4096]),
        ("compressor.wgate.weight", BF16, vec![256, 4096]),
        ("compressor.norm.weight", BF16, vec![128]),
        ("compressor.ape", FP32, vec![4, 256]),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (suffix, dtype, shape))| {
        (
            format!("layers.2.attn.indexer.{suffix}"),
            WeightTensor {
                ptr: DevicePtr(0x10000 + i as u64 * 0x1000000),
                shape,
                dtype,
            },
        )
    })
    .collect()
}

#[test]
fn admits_all_seven_native_tensors_without_gpu_or_conversion() {
    let store = WeightStore::from_map(tensors());
    assert_eq!(
        indexer::admit_indexer_weights(&store, &config()).unwrap(),
        1
    );
    assert_eq!(store.len(), 7);
    assert_eq!(
        store
            .get("layers.2.attn.indexer.wq_b.weight")
            .unwrap()
            .dtype,
        WeightDtype::FP8E4M3
    );
    assert_eq!(
        store.get("layers.2.attn.indexer.wq_b.scale").unwrap().dtype,
        WeightDtype::FP8E8M0
    );
}

#[test]
fn every_native_tensor_is_required() {
    for key in tensors().keys() {
        let mut map = tensors();
        map.remove(key);
        assert!(
            indexer::admit_indexer_weights(&WeightStore::from_map(map), &config()).is_err(),
            "{key}"
        );
    }
}

#[test]
fn every_shape_and_dtype_is_exact_not_same_byte_count() {
    for key in tensors().keys() {
        let mut map = tensors();
        map.get_mut(key).unwrap().shape.push(1);
        assert!(
            indexer::admit_indexer_weights(&WeightStore::from_map(map), &config()).is_err(),
            "rank {key}"
        );
        let mut map = tensors();
        let t = map.get_mut(key).unwrap();
        t.dtype = if t.dtype == WeightDtype::BF16 {
            WeightDtype::F16
        } else {
            WeightDtype::UInt8
        };
        assert!(
            indexer::admit_indexer_weights(&WeightStore::from_map(map), &config()).is_err(),
            "dtype {key}"
        );
    }
    for suffix in ["wq_b.weight", "wq_b.scale", "compressor.ape"] {
        let mut map = tensors();
        map.get_mut(&format!("layers.2.attn.indexer.{suffix}"))
            .unwrap()
            .shape
            .reverse();
        assert!(indexer::admit_indexer_weights(&WeightStore::from_map(map), &config()).is_err());
    }
}

#[test]
fn fp8_scale_geometry_is_k128_e8m0_not_attention_k64_or_float_scale() {
    for shape in [
        vec![128, 16],
        vec![64, 16],
        vec![64, 8, 1],
        vec![512],
        vec![usize::MAX, 8],
    ] {
        let mut map = tensors();
        map.get_mut("layers.2.attn.indexer.wq_b.scale")
            .unwrap()
            .shape = shape;
        assert!(indexer::admit_indexer_weights(&WeightStore::from_map(map), &config()).is_err());
    }
    let mut map = tensors();
    map.get_mut("layers.2.attn.indexer.wq_b.scale")
        .unwrap()
        .dtype = WeightDtype::FP32;
    assert!(indexer::admit_indexer_weights(&WeightStore::from_map(map), &config()).is_err());
}

#[test]
fn refuses_null_misaligned_or_overflowing_metadata_addresses() {
    for address in [0, 1, u64::MAX - 1] {
        let mut map = tensors();
        map.get_mut("layers.2.attn.indexer.compressor.ape")
            .unwrap()
            .ptr = DevicePtr(address);
        assert!(indexer::admit_indexer_weights(&WeightStore::from_map(map), &config()).is_err());
    }
}

#[test]
fn extra_indexer_keys_wrong_layers_and_aliases_are_not_silently_ignored() {
    for name in [
        "layers.0.attn.indexer.wq_b.weight",
        "layers.3.attn.indexer.wq_b.weight",
        "mtp.0.attn.indexer.wq_b.weight",
        "layers.2.attn.indexer.wq_b.weight_scale",
        "model.layers.2.attn.indexer.wq_b.weight",
    ] {
        let mut map = tensors();
        map.insert(name.into(), map.values().next().unwrap().clone());
        assert!(
            indexer::admit_indexer_weights(&WeightStore::from_map(map), &config()).is_err(),
            "{name}"
        );
    }
}

#[test]
fn missing_config_is_legacy_only_when_no_indexer_weights_exist() {
    let mut cfg = config();
    cfg.deepseek_v4_indexer = None;
    assert_eq!(
        indexer::admit_indexer_weights(&WeightStore::empty(), &cfg).unwrap(),
        0
    );
    assert!(indexer::admit_indexer_weights(&WeightStore::from_map(tensors()), &cfg).is_err());
}

#[test]
fn all_layers_are_admitted_not_only_the_first_csa_block() {
    let mut cfg = config();
    cfg.num_hidden_layers = 5;
    cfg.compress_ratios.push(4);
    let mut map = tensors();
    assert!(indexer::admit_indexer_weights(&WeightStore::from_map(map.clone()), &cfg).is_err());
    for (key, value) in tensors() {
        map.insert(key.replacen("layers.2", "layers.4", 1), value);
    }
    assert_eq!(
        indexer::admit_indexer_weights(&WeightStore::from_map(map), &cfg).unwrap(),
        2
    );
}
