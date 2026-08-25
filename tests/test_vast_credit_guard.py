import importlib.util
import json
import pathlib
import subprocess
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "vast-credit-guard.py"
SPEC = importlib.util.spec_from_file_location("vast_credit_guard", SCRIPT)
GUARD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GUARD)


class VastCreditGuardTest(unittest.TestCase):
    def test_reads_only_numeric_credit(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps({"credit": 12.5}), stderr=""
        )
        with mock.patch.object(GUARD.subprocess, "run", return_value=completed) as run:
            self.assertEqual(GUARD.read_credit(pathlib.Path("vastai")), 12.5)
        self.assertEqual(run.call_args.args[0], ["vastai", "--raw", "show", "user"])

    def test_rejects_missing_credit(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="{}", stderr=""
        )
        with mock.patch.object(GUARD.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "numeric credit"):
                GUARD.read_credit(pathlib.Path("vastai"))

    def test_stop_targets_exact_instance(self):
        with mock.patch.object(GUARD.subprocess, "run") as run:
            GUARD.stop_instance(pathlib.Path("/bin/vastai"), 48572428)
        run.assert_called_once_with(
            ["/bin/vastai", "stop", "instance", "48572428"], check=True
        )


if __name__ == "__main__":
    unittest.main()
