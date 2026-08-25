import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "deepseek-dflash2-bundle.py"
SPEC = importlib.util.spec_from_file_location("dflash2_bundle", SCRIPT)
BUNDLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUNDLE)


class DeepseekDflash2BundleTest(unittest.TestCase):
    def test_build_and_verify_detect_content_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            target = root / "target" / "component"
            source.mkdir()
            target.mkdir(parents=True)
            (source / "config.json").write_text("original")
            (target / "config.json").write_text("original")
            manifest = root / "manifest.json"
            BUNDLE.build([("component", source)], manifest)
            BUNDLE.verify(manifest, root / "target")
            (target / "config.json").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                BUNDLE.verify(manifest, root / "target")

    def test_manifest_has_no_source_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "weights").write_bytes(b"abc")
            manifest = root / "manifest.json"
            BUNDLE.build([("target-components", source)], manifest)
            payload = json.loads(manifest.read_text())
            self.assertEqual(payload["files"][0]["path"], "target-components/weights")
            self.assertNotIn(str(root), manifest.read_text())

    def test_rejects_duplicate_paths_and_inconsistent_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = root / "target"
            target.mkdir()
            (target / "weights").write_bytes(b"abc")
            manifest = root / "manifest.json"
            BUNDLE.build([("weights", target / "weights")], manifest)
            payload = json.loads(manifest.read_text())
            payload["files"].append(dict(payload["files"][0]))
            payload["file_count"] = 2
            payload["total_bytes"] = 6
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "duplicate bundle path"):
                BUNDLE.verify(manifest, target)

            payload["files"].pop()
            payload["file_count"] = 1
            payload["total_bytes"] = 99
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "total_bytes"):
                BUNDLE.verify(manifest, target)

    def test_rejects_symlink_even_when_target_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            (source / "weights").write_bytes(b"abc")
            manifest = root / "manifest.json"
            BUNDLE.build([("component", source)], manifest)
            (target / "component").mkdir()
            (target / "component" / "weights").symlink_to(source / "weights")
            with self.assertRaisesRegex(RuntimeError, "may not contain a symlink"):
                BUNDLE.verify(manifest, target)

    def test_rejects_symlinked_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            target = root / "target"
            outside = root / "outside"
            source.mkdir()
            target.mkdir()
            outside.mkdir()
            (source / "weights").write_bytes(b"abc")
            (outside / "weights").write_bytes(b"abc")
            manifest = root / "manifest.json"
            BUNDLE.build([("component", source)], manifest)
            (target / "component").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "may not contain a symlink"):
                BUNDLE.verify(manifest, target)

    def test_failed_atomic_replace_preserves_previous_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "weights").write_bytes(b"abc")
            manifest = root / "manifest.json"
            manifest.write_text("previous\n")
            with mock.patch.object(BUNDLE.os, "replace", side_effect=OSError("full")):
                with self.assertRaisesRegex(OSError, "full"):
                    BUNDLE.build([("component", source)], manifest)
            self.assertEqual(manifest.read_text(), "previous\n")
            self.assertEqual(list(root.glob(".manifest.json.*.tmp")), [])

    def test_rejects_malformed_digest_before_reading_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "atlas-deepseek-dflash2-bundle-v1",
                        "files": [{"path": "x", "bytes": 0, "sha256": "nope"}],
                        "file_count": 1,
                        "total_bytes": 0,
                    }
                )
            )
            with mock.patch.object(BUNDLE, "sha256") as digest:
                with self.assertRaisesRegex(RuntimeError, "invalid sha256"):
                    BUNDLE.verify(manifest, root)
                digest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
