import importlib.util
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "plan-checkpoint-prune.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_prune", SCRIPT)
PLANNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PLANNER)


class CheckpointPrunePlanTest(unittest.TestCase):
    def test_keeps_latest_step_and_only_reports_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for name, size in (
                ("epoch_0_step_10", 10),
                ("epoch_1_step_20", 20),
                ("not-a-checkpoint", 30),
            ):
                directory = root / name
                directory.mkdir()
                (directory / "model.safetensors").write_bytes(b"x" * size)
            report = PLANNER.plan_root(root, keep=1)

        self.assertEqual([item["step"] for item in report["retained"]], [20])
        self.assertEqual([item["step"] for item in report["candidates"]], [10])
        self.assertEqual(report["candidate_bytes"], 10)
        self.assertEqual(report["status"], "plan-only")


if __name__ == "__main__":
    unittest.main()
