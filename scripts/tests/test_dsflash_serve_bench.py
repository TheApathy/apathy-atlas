# SPDX-License-Identifier: AGPL-3.0-only

"""CPU-only preflight contracts for the DeepSeek benchmark launcher."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "dsflash-serve-bench.sh"


class DsflashServeBenchTests(unittest.TestCase):
    def test_default_receipt_artifacts_are_gitignored(self):
        for name in (
            "serve-unit.log.planned.receipt.json",
            "serve-unit.log.receipt.json",
        ):
            result = subprocess.run(
                ["git", "check-ignore", "--no-index", "--quiet", name],
                cwd=REPO,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"not gitignored: {name}")

    def test_print_config_builds_bound_receipt_without_starting_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8"
            )
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "model-1.safetensors"}}),
                encoding="utf-8",
            )
            (model / "model-1.safetensors").write_bytes(b"weights")
            (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            executed = root / "executed"
            binary = root / "spark"
            binary.write_text(
                "#!/bin/sh\n"
                "# deepseek_v4 atlas_engine ATLAS_BENCH_ENGINE_USAGE "
                "ATLAS_DSPARK_CAPTURE ATLAS_DFLASH_ADAPTIVE "
                "ATLAS_UNIFIED_MOE_LAYOUT\n"
                'touch "$EXECUTED_MARKER"\n',
                encoding="utf-8",
            )
            binary.chmod(0o755)
            log = root / "serve.log"
            environment = {
                **os.environ,
                "REPO": str(REPO),
                "BIN": str(binary),
                "MODEL": str(model),
                "LOG": str(log),
                "PRINT_CONFIG_ONLY": "1",
                "EXECUTED_MARKER": str(executed),
            }
            result = subprocess.run(
                [str(SCRIPT), "unit", "-"],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            time.sleep(0.05)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("pid=", result.stdout)
            self.assertFalse(
                executed.exists(), "config-only preflight launched the server"
            )
            receipt_path = Path(f"{log}.planned.receipt.json")
            envelope = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(Path(f"{log}.receipt.json").exists())
            manifest = envelope["manifest"]
            self.assertEqual(manifest["receipt_state"], "PLANNED")
            self.assertEqual(manifest["binary"]["path"], str(binary))
            self.assertEqual(manifest["model"]["path"], str(model))
            self.assertEqual(manifest["argv"][:3], [str(binary), "serve", str(model)])
            self.assertNotIn("--dflash", manifest["argv"])
            self.assertEqual(
                manifest["environment"],
                {
                    "_": str(binary),
                    "ATLAS_BENCH_ENGINE_USAGE": "1",
                    "ATLAS_DFLASH_ADAPTIVE": "1",
                    "ATLAS_DSPARK_CAPTURE": "1",
                    "ATLAS_UNIFIED_MOE_LAYOUT": "1",
                },
            )


if __name__ == "__main__":
    unittest.main()
