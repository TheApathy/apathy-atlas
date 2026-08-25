import importlib.util
import io
import pathlib
import tempfile
import unittest
from unittest import mock

import torch


SCRIPT = (
    pathlib.Path(__file__).parents[1]
    / "scripts"
    / "capture-deepseek-dflash2-offline.py"
)
SPEC = importlib.util.spec_from_file_location("dflash2_capture", SCRIPT)
CAPTURE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CAPTURE)


def bf16_bytes(value: float, rows: int) -> bytes:
    tensor = torch.full((rows, 4096), value, dtype=torch.bfloat16)
    return tensor.view(torch.uint16).numpy().tobytes()


def record(start: int, rows: int, values: list[float], kind: int = 0) -> bytes:
    header = CAPTURE.HEADER.pack(CAPTURE.MAGIC, kind, start, rows, 4096, len(values), 0)
    return header + b"".join(bf16_bytes(value, rows) for value in values)


class DeepseekDflash2CaptureTest(unittest.TestCase):
    def test_reassembles_chunked_layer_major_dump_as_token_major(self):
        stream = io.BytesIO(
            record(0, 2, [1, 2, 3, 4, 5]) + record(2, 1, [6, 7, 8, 9, 10])
        )
        hidden = CAPTURE.read_prefill_row(
            stream, torch, 3, 4, [0, 10, 21, 31, 42], 0.1, 7
        )

        self.assertEqual(tuple(hidden.shape), (4, 5, 4096))
        self.assertTrue(torch.all(hidden[0, 0] == 1))
        self.assertTrue(torch.all(hidden[1, 4] == 5))
        self.assertTrue(torch.all(hidden[2, 0] == 6))
        self.assertTrue(torch.all(hidden[2, 4] == 10))
        self.assertTrue(torch.all(hidden[3] == 0))

    def test_rejects_interleaved_or_out_of_order_record(self):
        stream = io.BytesIO(record(1, 1, [1, 2, 3, 4, 5]))
        with self.assertRaisesRegex(RuntimeError, "capture server was contaminated"):
            CAPTURE.read_prefill_row(stream, torch, 1, 1, [0, 10, 21, 31, 42], 0.1, 0)

    def test_atomic_save_never_publishes_partial_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "row.pt"
            tensor = torch.zeros((2, 3), dtype=torch.bfloat16)
            with mock.patch.object(torch, "save", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    CAPTURE.save_tensor_atomic(torch, tensor, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob("*.tmp")), [])

            CAPTURE.save_tensor_atomic(torch, tensor, output)
            loaded = torch.load(output, weights_only=True)
            self.assertTrue(torch.equal(loaded, tensor))


if __name__ == "__main__":
    unittest.main()
