import importlib.util
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
