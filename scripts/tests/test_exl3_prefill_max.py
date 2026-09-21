# SPDX-License-Identifier: AGPL-3.0-only

import os
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts" / "exl3-prefill-max.sh"


class Exl3PrefillMaxTests(unittest.TestCase):
    def test_printed_profile_is_plain_single_pass_and_complete(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "PRINT_CONFIG_ONLY": "1",
                "GAMMA": "17",
                "DSPARK_TOKENS": "16",
                "ATLAS_PROFILE": "1",
                "ATLAS_DIAG_V4_ALL_LAYERS": "1",
                "ATLAS_OP_DUMP": "/tmp/must-not-survive",
                "DFLASH_TRAIN_DUMP": "/tmp/must-not-survive-either",
                "ATLAS_DUMP_EXPERT_IDS": "1",
                "ATLAS_EXL3_SHARED_PREFILL_FP8": "1",
                "ATLAS_V4_PROJ_FP8MMA": "1",
                "ATLAS_V4_PREFILL_HC_RMS_FUSED": "0",
                "ATLAS_V4_PREFILL_QB_ROPE_CACHE_FUSED": "1",
                "ATLAS_V4_PREFILL_QB_ROPE_FUSED": "1",
                "ATLAS_V4_PREFILL_TC2_WARP0": "1",
                "ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN_N256": "1",
                "ATLAS_EXL3_PREFILL_FUSED_BLEND": "1",
                "ATLAS_FP8_KV_EMA_RECAL": "1",
                "ATLAS_FP8_KV_HEADROOM": "99",
                "ATLAS_DEBUG_SYNC_KERNELS": "1",
                "ATLAS_PREFILL_HOST_TIMING": "1",
                "ATLAS_V4_STAGE_SYNCS": "1",
                "ATLAS_MOE_PREFILL_ZERO": "1",
                "ATLAS_DUMP_EMBED": "1",
                "CUDA_LAUNCH_BLOCKING": "1",
                "ATLAS_EXL3_FIXED_K2": "0",
                "ATLAS_EXL3_SPLIT": "1",
                "ATLAS_EXL3_VERIFY_WORKLIST": "1",
                "ATLAS_EXL3_FUSED": "0",
                "ATLAS_PREFILL_MAX_REQUIRE_ARMS": "0",
                "ATLAS_MAX_BATCH_TOKENS": "999999",
                "ATLAS_KV_EXTERNAL_RESERVE_GB": "99",
                "ATLAS_PEAK_MEM_MULT": "99",
                "FP8_KV_CALIBRATION_TOKENS": "256",
                "REPO": "/tmp/not-the-launcher-checkout",
                "MODEL": "/tmp/not-the-k2-checkpoint",
            }
        )
        result = subprocess.run(
            [str(WRAPPER)],
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        output = result.stdout
        self.assertIn("spec : <none>", output)
        self.assertIn("max_seq=4096", output)
        self.assertIn("max_prefill=4096", output)
        self.assertIn(f"serve: {ROOT / 'target/release/spark'}", output)
        self.assertIn(
            "model: /home/flocka/models/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1",
            output,
        )
        self.assertIn("fp8_kv_calibration=model-default", output)
        for setting in [
            "ATLAS_V4_PREFILL_CUBLASLT=1",
            "ATLAS_V4_ATTN_RELEASE_BF16=0",
            "ATLAS_V4_PREFILL_HC_RMS_FUSED=1",
            "ATLAS_V4_PREFILL_TC2_WARP0=0",
            "ATLAS_V4_PREFILL_QB_ROPE_FUSED=0",
            "ATLAS_V4_PREFILL_KV_ALIAS=1",
            "ATLAS_V4_PREFILL_INVERSE_ROPE_FUSED=1",
            "ATLAS_PREFILL_MAX_REQUIRE_ARMS=1",
            "ATLAS_EXL3_PREFILL_W2A8=1",
            "ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN=1",
            "ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN_N256=0",
            "ATLAS_EXL3_PREFILL_W2A8_N256_DOWN=1",
            "ATLAS_EXL3_PREFILL_FUSED_UNPERMUTE=1",
            "ATLAS_EXL3_PREFILL_FUSED_BLEND=0",
            "ATLAS_KV_OVERCOMMIT=0",
            "ATLAS_MOE_SHARED_K64=4",
            "ATLAS_EXL3_PREFILL_M128=0",
            "ATLAS_EXL3_PREFILL_K64=0",
            "ATLAS_EXL3_PREFILL_N128=0",
            "ATLAS_EXL3_PREFILL_N256=0",
        ]:
            self.assertIn(setting, output)
        scrub_line = next(
            line for line in output.splitlines() if line.startswith("scrub:")
        )
        for unsafe_setting in [
            "ATLAS_V4_PREFILL_QB_ROPE_CACHE_FUSED=1",
            "ATLAS_DUMP_EXPERT_IDS=1",
            "ATLAS_EXL3_SHARED_PREFILL_FP8=1",
            "ATLAS_V4_PROJ_FP8MMA=1",
            "ATLAS_FP8_KV_EMA_RECAL=1",
            "ATLAS_FP8_KV_HEADROOM=99",
            "ATLAS_DSPARK_DUMP=",
        ]:
            self.assertNotIn(unsafe_setting, output)
        for scrubbed_name in [
            "ATLAS_V4_PREFILL_QB_ROPE_CACHE_FUSED",
            "ATLAS_V4_PREFILL_QB_ROPE_FUSED",
            "ATLAS_V4_PREFILL_TC2_WARP0",
            "ATLAS_FP8_KV_EMA_RECAL",
            "ATLAS_FP8_KV_HEADROOM",
            "ATLAS_DEBUG_SYNC_KERNELS",
            "ATLAS_PREFILL_HOST_TIMING",
            "ATLAS_V4_STAGE_SYNCS",
            "ATLAS_MOE_PREFILL_ZERO",
            "ATLAS_DUMP_EMBED",
            "CUDA_LAUNCH_BLOCKING",
            "ATLAS_EXL3_FIXED_K2",
            "ATLAS_EXL3_SPLIT",
            "ATLAS_EXL3_VERIFY_WORKLIST",
            "ATLAS_EXL3_FUSED",
            "ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN_N256",
            "ATLAS_EXL3_PREFILL_FUSED_BLEND",
            "ATLAS_MAX_BATCH_TOKENS",
            "ATLAS_KV_EXTERNAL_RESERVE_GB",
            "ATLAS_PEAK_MEM_MULT",
        ]:
            self.assertIn(scrubbed_name, scrub_line)

    def test_explicit_trailing_assignment_remains_an_ablation_override(self) -> None:
        env = os.environ.copy()
        env["PRINT_CONFIG_ONLY"] = "1"
        result = subprocess.run(
            [str(WRAPPER), "ATLAS_V4_PREFILL_KV_ALIAS=0"],
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        env_line = next(line for line in result.stdout.splitlines() if line.startswith("env  :"))
        self.assertLess(
            env_line.index("ATLAS_V4_PREFILL_KV_ALIAS=1"),
            env_line.rindex("ATLAS_V4_PREFILL_KV_ALIAS=0"),
        )

    def test_real_launch_preflights_model_before_build_receipt(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        model_preflight = (
            '"$repo_root/scripts/check-exl3-prefill-max-model.py"'
        )
        build_verification = (
            '"$repo_root/scripts/build-exl3-prefill-max.sh" --verify-only'
        )
        self.assertIn(model_preflight, source)
        self.assertLess(source.index(model_preflight), source.index(build_verification))


if __name__ == "__main__":
    unittest.main()
