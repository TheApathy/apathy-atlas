# SPDX-License-Identifier: AGPL-3.0-only

"""CPU-only contracts for the exact EXL3 max-prefill checkpoint preflight."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check-exl3-prefill-max-model.py"


def load_preflight():
    spec = importlib.util.spec_from_file_location("atlas_prefill_model_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_config() -> dict:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "expert_dtype": "fp4",
        "head_dim": 512,
        "hidden_size": 4096,
        "model_type": "deepseek_v4",
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 43,
        "num_key_value_heads": 1,
        "o_lora_rank": 1024,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "torch_dtype": "bfloat16",
        "vocab_size": 129280,
        "quantization_config": {
            "bits": 2.0,
            "checkpoint_format": "exl3",
            "codebook": "mcg",
            "format": "exl3",
            "group_size": -1,
            "method": "exl3",
            "quant_method": "exl3",
        },
    }


def exact_model_toml() -> str:
    return """\
[model]
name = "deepseek-v4-flash"
layers_total = 43
hidden_dim = 4096
head_dim = 512
q_heads = 64
kv_heads = 1
intermediate_size = 2048
vocab_size = 129280
kv_lora_rank = 512
q_lora_rank = 1024
o_lora_rank = 1024
v_head_dim = 512
qk_rope_head_dim = 64
qk_nope_head_dim = 448
num_experts = 256
num_shared_experts = 1
moe_intermediate_size = 2048
top_k = 6

[[model_types]]
model_type = "deepseek_v4"

[behavior]
default_kv_dtype = "fp8"
fp8_kv_calibration_tokens = 256
"""


def exact_weight_map(module) -> dict[str, str]:
    shards = list(module.EXPECTED_SHARDS)
    weight_map = {}
    position = 0
    for layer in range(43):
        for expert in range(256):
            for projection in ("down_proj", "gate_proj", "up_proj"):
                for component in ("mcg", "suh", "svh", "trellis"):
                    name = (
                        f"model.layers.{layer}.mlp.experts.{expert}."
                        f"{projection}.{component}"
                    )
                    weight_map[name] = shards[position % len(shards)]
                    position += 1
    while len(weight_map) < module.EXPECTED_TENSOR_COUNT:
        name = f"fixture.identity.{len(weight_map):06d}"
        weight_map[name] = shards[position % len(shards)]
        position += 1
    return weight_map


class Exl3PrefillMaxModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preflight = load_preflight()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.model = self.root / "DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1"
        self.repo.mkdir()
        self.model.mkdir()
        (self.repo / "scripts").mkdir()
        target = self.repo / "kernels/gb10/deepseek-v4-flash/nvfp4"
        target.mkdir(parents=True)
        (self.repo / "scripts/exl3-prefill-max.sh").write_text(
            f"export MODEL={self.model}\n", encoding="utf-8"
        )
        (target / "MODEL.toml").write_text(exact_model_toml(), encoding="utf-8")
        (target.parent / "MODEL.toml").write_text(
            exact_model_toml(), encoding="utf-8"
        )
        (self.model / "config.json").write_text(
            json.dumps(exact_config()), encoding="utf-8"
        )
        self.weight_map = exact_weight_map(self.preflight)
        self.write_index(self.weight_map)
        for shard_name in self.preflight.EXPECTED_SHARDS:
            (self.model / shard_name).write_bytes(b"not-hashed-by-preflight")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_index(self, weight_map: dict[str, str]) -> None:
        index = {
            "metadata": {"total_size": self.preflight.EXPECTED_INDEX_TOTAL_SIZE},
            "weight_map": weight_map,
        }
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps(index, separators=(",", ":")), encoding="utf-8"
        )

    def test_exact_profile_emits_bounded_identity_without_hashing_shards(self) -> None:
        hashed = []
        real_sha256_file = self.preflight.sha256_file

        def record_hash(path):
            hashed.append(Path(path))
            return real_sha256_file(path)

        with mock.patch.object(self.preflight, "sha256_file", side_effect=record_hash):
            report = self.preflight.build_report(
                repo_root=self.repo, expected_model=self.model
            )

        self.assertEqual(report["schema"], "atlas-exl3-prefill-max-model-v1")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["profile"]["top_k"], 6)
        self.assertEqual(report["profile"]["num_experts"], 256)
        self.assertEqual(report["profile"]["exl3_bits"], 2.0)
        self.assertEqual(report["kv"]["k_scale_tensors"], 0)
        self.assertEqual(report["kv"]["v_scale_tensors"], 0)
        self.assertEqual(report["kv"]["calibration_tokens"], 256)
        self.assertEqual(report["checkpoint"]["shard_count"], 10)
        self.assertTrue(report["checkpoint"]["shard_stat_manifest_sha256"])
        self.assertLess(len(json.dumps(report, separators=(",", ":"))), 1600)
        self.assertFalse(any(path.suffix == ".safetensors" for path in hashed))

    def test_config_and_behavior_identity_fail_closed(self) -> None:
        config = exact_config()
        config["num_experts_per_tok"] = 5
        with self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
            self.preflight.validate_config(config)

        config = exact_config()
        config["n_routed_experts"] = True
        with self.assertRaisesRegex(ValueError, "n_routed_experts"):
            self.preflight.validate_config(config)

        registry = self.preflight.parse_model_toml(exact_model_toml().encode())
        registry["behavior"]["fp8_kv_calibration_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            self.preflight.validate_model_toml(registry)

    def test_index_rejects_missing_escaping_and_symlink_shards(self) -> None:
        missing = self.model / self.preflight.EXPECTED_SHARDS[-1]
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.preflight.build_report(
                repo_root=self.repo, expected_model=self.model
            )
        missing.write_bytes(b"restored")

        poisoned = dict(self.weight_map)
        poisoned["fixture.identity.132096"] = "../escape.safetensors"
        self.write_index(poisoned)
        with self.assertRaisesRegex(ValueError, "safe shard basename"):
            self.preflight.build_report(
                repo_root=self.repo, expected_model=self.model
            )

        outside = self.root / self.preflight.EXPECTED_SHARDS[0]
        outside.write_bytes(b"outside")
        (self.model / self.preflight.EXPECTED_SHARDS[0]).unlink()
        (self.model / self.preflight.EXPECTED_SHARDS[0]).symlink_to(outside)
        self.write_index(self.weight_map)
        with self.assertRaisesRegex(ValueError, "regular non-symlink"):
            self.preflight.build_report(
                repo_root=self.repo, expected_model=self.model
            )

    def test_canonical_checkpoint_and_zero_scale_calibration_are_mandatory(self) -> None:
        alias = self.root / "model-alias"
        alias.symlink_to(self.model, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "canonical expected path"):
            self.preflight.require_canonical_expected_model(alias)

        scaled = dict(self.weight_map)
        scaled["layers.0.attn.k_scale"] = scaled.pop("fixture.identity.132096")
        self.write_index(scaled)
        with self.assertRaisesRegex(ValueError, "zero K/V scale tensors"):
            self.preflight.build_report(
                repo_root=self.repo, expected_model=self.model
            )

        self.write_index(self.weight_map)
        zero_calibration = exact_model_toml().replace(
            "fp8_kv_calibration_tokens = 256",
            "fp8_kv_calibration_tokens = 0",
        )
        for target in (
            self.repo / "kernels/gb10/deepseek-v4-flash/MODEL.toml",
            self.repo / "kernels/gb10/deepseek-v4-flash/nvfp4/MODEL.toml",
        ):
            target.write_text(zero_calibration, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "positive"):
            self.preflight.build_report(
                repo_root=self.repo, expected_model=self.model
            )

    def test_launcher_must_name_the_same_literal_expected_checkpoint(self) -> None:
        launcher = self.repo / "scripts/exl3-prefill-max.sh"
        launcher.write_text("export MODEL=${HOME}/surprise\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "literal absolute"):
            self.preflight.build_report(
                repo_root=self.repo, expected_model=self.model
            )

        launcher.write_text(
            f"export MODEL={self.root / 'different'}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "does not select"):
            self.preflight.build_report(
                repo_root=self.repo, expected_model=self.model
            )


if __name__ == "__main__":
    unittest.main()
