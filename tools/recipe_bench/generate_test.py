# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for generate.py. No GPU, no server, no network."""
import textwrap

import generate


def parse(yaml_text: str) -> dict:
    import yaml

    return yaml.safe_load(textwrap.dedent(yaml_text))


def test_env_reads_the_atlas_env_block_verbatim():
    recipe = parse(
        """
        defaults: {}
        atlas:
          env:
            ATLAS_FOO: "1"
            ATLAS_BAR: "0"
        """
    )
    assert generate.build_env(recipe) == ["ATLAS_FOO=1", "ATLAS_BAR=0"]


def test_env_handles_null_value_as_empty_string():
    recipe = parse(
        """
        defaults: {}
        atlas:
          env:
            ATLAS_EMPTY:
        """
    )
    assert generate.build_env(recipe) == ["ATLAS_EMPTY="]


def test_argv_underscore_keys_are_hyphenated():
    recipe = parse(
        """
        defaults:
          max_batch_size: 1
          kv_cache_dtype: nvfp4
        """
    )
    argv = generate.build_argv(recipe, "/models/m", "m", [])
    assert "--max-batch-size" in argv
    assert "--kv-cache-dtype" in argv
    assert "max_batch_size" not in " ".join(argv)


def test_argv_applies_renames():
    recipe = parse(
        """
        defaults:
          max_model_len: 4096
          host: 0.0.0.0
        """
    )
    argv = generate.build_argv(recipe, "/models/m", "m", [])
    assert "--max-seq-len" in argv
    assert "--bind" in argv
    assert "--max-model-len" not in argv
    assert "--host" not in argv


def test_argv_presence_only_flags_carry_no_value():
    """The bug this test exists to catch: a presence-only flag's true/false
    value must never be emitted as a second token. An earlier draft of this
    generator listed 'qwen4_qsa' (underscored) in PRESENCE, but PRESENCE is
    checked against the flag name AFTER hyphenation ('qwen4-qsa'), so the
    membership test silently failed and `--qwen4-qsa true` came out as two
    argv tokens instead of the bare flag. Caught here before the tool ever
    touched a real server — the mismatch was found by inspecting the argv
    output, not by a GPU run."""
    recipe = parse(
        """
        defaults:
          qwen4_qsa: true
          dflash: true
          no_tui: true
          max_batch_size: 1
        """
    )
    argv = generate.build_argv(recipe, "/models/m", "m", [])
    assert "--qwen4-qsa" in argv
    assert "--dflash" in argv
    # presence flags contribute exactly one token each: the flag itself,
    # never followed by "true"/"false" as a second token.
    for flag in ("--qwen4-qsa", "--dflash"):
        i = argv.index(flag)
        assert i + 1 == len(argv) or not argv[i + 1] in ("true", "false"), (
            f"{flag} was followed by a value token; presence flags take none"
        )


def test_argv_presence_only_flags_omitted_when_false():
    recipe = parse(
        """
        defaults:
          dflash: false
        """
    )
    argv = generate.build_argv(recipe, "/models/m", "m", [])
    assert "--dflash" not in argv


def test_argv_override_replaces_a_default():
    recipe = parse(
        """
        defaults:
          port: 8888
        """
    )
    argv = generate.build_argv(recipe, "/models/m", "m", ["port=8897"])
    i = argv.index("--port")
    assert argv[i + 1] == "8897"


def test_argv_override_del_removes_a_default():
    recipe = parse(
        """
        defaults:
          port: 8888
          max_batch_size: 1
        """
    )
    argv = generate.build_argv(recipe, "/models/m", "m", ["port=DEL"])
    assert "--port" not in argv
    assert "--max-batch-size" in argv


def test_argv_always_carries_model_from_path_and_no_tui():
    recipe = parse("defaults: {}")
    argv = generate.build_argv(recipe, "/models/qwen", "m", [])
    assert argv[:5] == ["serve", "--model-from-path", "/models/qwen", "--model-name", "m"]
    assert "--no-tui" in argv


def test_env_and_argv_do_not_cross_contaminate():
    """defaults: overrides must never leak into the env: output — env mode
    doesn't take overrides as meaningful input at all."""
    recipe = parse(
        """
        defaults:
          port: 8888
        atlas:
          env:
            ATLAS_FOO: "1"
        """
    )
    assert generate.build_env(recipe) == ["ATLAS_FOO=1"]
