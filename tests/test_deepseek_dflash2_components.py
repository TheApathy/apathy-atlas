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
    / "validate-deepseek-dflash2-components.py"
)
SPEC = importlib.util.spec_from_file_location("dflash2_components", SCRIPT)
COMPONENTS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COMPONENTS)


class DeepseekDflash2ComponentsTest(unittest.TestCase):
    def test_metadata_requires_preserved_source_and_loader_abi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            components = root / "components"
            source.mkdir()
            components.mkdir()
            for name in COMPONENTS.IDENTICAL_METADATA:
                (source / name).write_text(f"{name}\n")
                (components / name).write_text(f"{name}\n")
            (source / "config.json").write_text('{"model_type":"deepseek_v4"}\n')
            (components / "deepseek_target_config.json").write_text(
                '{"model_type":"deepseek_v4"}\n'
            )
            loader_config = {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 4096,
                "vocab_size": 129280,
                "num_hidden_layers": 3,
                "intermediate_size": 11008,
                "head_dim": 128,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            }
            (components / "config.json").write_text(json.dumps(loader_config))
            hashes = COMPONENTS.validate_metadata(source, components)
            self.assertEqual(set(hashes), set(COMPONENTS.IDENTICAL_METADATA) | {
                "config.json", "deepseek_target_config.json"
            })

            loader_config["vocab_size"] = 1
            (components / "config.json").write_text(json.dumps(loader_config))
            with self.assertRaisesRegex(RuntimeError, "vocab_size"):
                COMPONENTS.validate_metadata(source, components)

    def test_streamed_tensor_digest_detects_value_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            tensor = torch.arange(24, dtype=torch.int32).reshape(6, 4)
            save_file({"weight": tensor}, first)
            save_file({"weight": tensor.clone()}, second)
            self.assertEqual(
                COMPONENTS.tensor_digest(first, "weight", chunk_rows=2),
                COMPONENTS.tensor_digest(second, "weight", chunk_rows=3),
            )
            tensor[5, 3] += 1
            save_file({"weight": tensor}, second)
            self.assertNotEqual(
                COMPONENTS.tensor_digest(first, "weight")[2],
                COMPONENTS.tensor_digest(second, "weight")[2],
            )


if __name__ == "__main__":
    unittest.main()
