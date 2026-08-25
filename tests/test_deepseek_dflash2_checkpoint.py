import importlib.util
import json
import pathlib
import tempfile
import unittest

import torch
from safetensors.torch import save_file


SCRIPT = (
    pathlib.Path(__file__).parents[1]
    / "scripts"
    / "validate-deepseek-dflash2-checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("dflash2_checkpoint", SCRIPT)
CHECKPOINT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECKPOINT)


class DeepseekDflash2CheckpointTest(unittest.TestCase):
    @staticmethod
    def valid_config():
        return {
            "hidden_size": 4096,
            "vocab_size": 129280,
            "num_target_layers": 43,
            "num_hidden_layers": 3,
            "block_size": 16,
            "intermediate_size": 11008,
            "head_dim": 128,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "dflash_config": {
                "target_layer_ids": [1, 11, 22, 32, 43],
                "mask_token_id": 129000,
                "causal": False,
            },
        }

    @classmethod
    def valid_shapes(cls):
        config = cls.valid_config()
        hidden = config["hidden_size"]
        intermediate = config["intermediate_size"]
        head_dim = config["head_dim"]
        q_width = config["num_attention_heads"] * head_dim
        kv_width = config["num_key_value_heads"] * head_dim
        shapes = {
            "fc.weight": ([hidden, 5 * hidden], "BF16"),
            "hidden_norm.weight": ([hidden], "BF16"),
            "norm.weight": ([hidden], "BF16"),
        }
        for layer in range(3):
            base = f"layers.{layer}"
            shapes.update(
                {
                    f"{base}.input_layernorm.weight": ([hidden], "BF16"),
                    f"{base}.post_attention_layernorm.weight": ([hidden], "BF16"),
                    f"{base}.self_attn.q_proj.weight": ([q_width, hidden], "BF16"),
                    f"{base}.self_attn.k_proj.weight": ([kv_width, hidden], "BF16"),
                    f"{base}.self_attn.v_proj.weight": ([kv_width, hidden], "BF16"),
                    f"{base}.self_attn.o_proj.weight": ([hidden, q_width], "BF16"),
                    f"{base}.self_attn.q_norm.weight": ([head_dim], "BF16"),
                    f"{base}.self_attn.k_norm.weight": ([head_dim], "BF16"),
                    f"{base}.mlp.gate_proj.weight": ([intermediate, hidden], "BF16"),
                    f"{base}.mlp.up_proj.weight": ([intermediate, hidden], "BF16"),
                    f"{base}.mlp.down_proj.weight": ([hidden, intermediate], "BF16"),
                }
            )
        return shapes

    def test_reads_unindexed_safetensors_shapes_without_allocating_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            save_file(
                {"fc.weight": torch.zeros((2, 3), dtype=torch.bfloat16)},
                root / "model.safetensors",
            )
            shapes = CHECKPOINT.tensor_shapes(root)
        self.assertEqual(shapes, {"fc.weight": ([2, 3], "BF16")})

    def test_reads_indexed_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            save_file({"a": torch.zeros(1)}, root / "one.safetensors")
            save_file({"b": torch.zeros(2)}, root / "two.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}
                )
            )
            shapes = CHECKPOINT.tensor_shapes(root)
        self.assertEqual(set(shapes), {"a", "b"})

    def test_rejects_index_key_or_shard_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            save_file({"a": torch.zeros(1)}, root / "one.safetensors")
            save_file({"b": torch.zeros(2)}, root / "two.safetensors")
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps({"weight_map": {"a": "two.safetensors", "b": "one.safetensors"}})
            )
            with self.assertRaisesRegex(RuntimeError, "wrong shards"):
                CHECKPOINT.tensor_shapes(root)

            index_path.write_text(
                json.dumps(
                    {"weight_map": {"a": "one.safetensors", "b": "one.safetensors"}}
                )
            )
            with self.assertRaisesRegex(RuntimeError, "keys differ"):
                CHECKPOINT.tensor_shapes(root)

    def test_full_metadata_abi_passes_without_allocating_large_tensors(self):
        report = CHECKPOINT.validate_abi(self.valid_config(), self.valid_shapes())
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["capture_width"], 20480)
        self.assertEqual(report["required_tensor_count"], 36)

    def test_rejects_causal_or_mixed_prefix_checkpoint(self):
        config = self.valid_config()
        config["dflash_config"]["causal"] = True
        with self.assertRaisesRegex(RuntimeError, "must be non-causal"):
            CHECKPOINT.validate_abi(config, self.valid_shapes())

        config["dflash_config"]["causal"] = False
        shapes = self.valid_shapes()
        shapes["model.fc.weight"] = shapes["fc.weight"]
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            CHECKPOINT.validate_abi(config, shapes)

    def test_rejects_wrong_tensor_dtype_and_shape(self):
        shapes = self.valid_shapes()
        shapes["fc.weight"] = ([4096, 4096], "BF16")
        with self.assertRaisesRegex(RuntimeError, "expected"):
            CHECKPOINT.validate_abi(self.valid_config(), shapes)

        shapes = self.valid_shapes()
        shapes["norm.weight"] = ([4096], "I64")
        with self.assertRaisesRegex(RuntimeError, "unsupported training dtype"):
            CHECKPOINT.validate_abi(self.valid_config(), shapes)


if __name__ == "__main__":
    unittest.main()
