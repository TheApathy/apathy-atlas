import pathlib
import unittest


ROOT = pathlib.Path(__file__).parents[1]


class DeepseekDflash2TrainingContractTest(unittest.TestCase):
    def test_capture_validation_and_training_share_post_filter_limit(self):
        capture = (ROOT / "scripts/capture-deepseek-dflash2-offline.py").read_text()
        validator = (ROOT / "scripts/validate-deepseek-dflash2-offline.py").read_text()
        trainer_patch = (
            ROOT
            / "3rdparty_patches/specforge/offline_train_row_limit.patch"
        ).read_text()
        cache_patch = (
            ROOT
            / "3rdparty_patches/specforge/content_addressed_preprocess_cache.patch"
        ).read_text()
        launcher = (ROOT / "scripts/train-deepseek-dflash2-vast.sh").read_text()

        self.assertLess(capture.index("dataset = dataset.filter"), capture.index("if args.limit:"))
        self.assertLess(
            validator.index("dataset = dataset.filter"), validator.index("if args.limit:")
        )
        self.assertIn("--max-train-rows", trainer_patch)
        self.assertIn("if args.max_train_rows:", trainer_patch)
        for source in (capture, validator, cache_patch):
            self.assertIn("deepseek-dflash2-preprocess-v2", source)
            self.assertIn("corpus_sha256", source)
            self.assertIn("tokenizer_sha256", source)
            self.assertIn("preprocessing_sha256", source)
        self.assertIn("default=128", capture)
        self.assertIn("default=128", validator)
        self.assertIn('--limit "$TRAIN_ROWS"', launcher)
        self.assertIn('--max-train-rows "$TRAIN_ROWS"', launcher)
        self.assertIn("2824835f81288541eaa6a97362cd7e308", launcher)
        self.assertIn("corpus SHA-256 mismatch", launcher)
        self.assertLess(
            launcher.index('if [[ "$PREFLIGHT_ONLY" == 1 ]]'),
            launcher.index("torchrun --standalone"),
        )
        self.assertIn("refusing to start torchrun", launcher)


if __name__ == "__main__":
    unittest.main()
