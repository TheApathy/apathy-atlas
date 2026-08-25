import importlib.util
import json
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("context_sweep.py")
SPEC = importlib.util.spec_from_file_location("context_sweep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())


class ContextSweepTests(unittest.TestCase):
    def test_prompt_is_exact_length_with_one_middle_needle(self):
        tokens = MODULE.build_prompt(FakeTokenizer(), 8192)
        rendered = bytes(tokens).decode()
        self.assertEqual(len(tokens), 8192)
        self.assertEqual(rendered.count(MODULE.SECRET), 1)
        needle_offset = rendered.index(MODULE.SECRET)
        self.assertGreater(needle_offset, 3000)
        self.assertLess(needle_offset, 5000)

    def test_rejects_context_shorter_than_fixed_prompt(self):
        with self.assertRaisesRegex(ValueError, "below fixed prompt size"):
            MODULE.build_prompt(FakeTokenizer(), 1)

    def test_context_contract_reserves_generation_room(self):
        self.assertEqual(
            MODULE.validate_contexts("8192,1000000", 1048576, 128, 1),
            [8192, 1000000],
        )
        with self.assertRaisesRegex(ValueError, "exceeds checkpoint ceiling"):
            MODULE.validate_contexts("1048500", 1048576, 128, 1)

    def test_context_contract_rejects_duplicates_and_zero_reps(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            MODULE.validate_contexts("8192,8192", 1048576, 128, 1)
        with self.assertRaisesRegex(ValueError, "reps must be positive"):
            MODULE.validate_contexts("8192", 1048576, 128, 0)

    def test_atomic_json_output_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "plan.json"
            MODULE.write_json_atomic(output, {"status": "complete"})
            self.assertEqual(json.loads(output.read_text()), {"status": "complete"})
            self.assertEqual(list(output.parent.glob("*.tmp")), [])

    def test_live_validation_rejects_token_drift_and_retrieval_failure(self):
        run = {
            "prompt_tokens": 8191,
            "completion_tokens": 1,
            "decode_seconds": 0.0,
            "retrieval_pass": False,
        }
        failures = MODULE.validate_live_runs(8192, [run])
        self.assertTrue(any("expected 8192" in item for item in failures))
        self.assertTrue(any("no measurable" in item for item in failures))
        self.assertTrue(any("retrieval failed" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
