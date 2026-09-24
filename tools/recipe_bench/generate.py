#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Emit env (KEY=VAL lines) or argv (one token per line) for a built-in recipe
YAML, mirroring crates/spark-server/src/recipe/schema.rs: `defaults:` maps to
`--flag value`, with renames and presence-only booleans; `atlas.env` maps
straight to environment variables.

History: this replaces two independent hand-transcriptions of recipe env
blocks (one for Qwen3.8-27B via bench/qwen38-gb10/serve.sh, one for GLM/FN)
that each silently dropped real vars — 4 missing for Q3.8, ~all of them for
GLM and FN on the first pass. Every prior "the recipe is slow" finding this
session turned out to be that, not a real regression. This generator exists
so nobody hand-transcribes a recipe again.

usage:
  generate.py <recipe.yaml> <model_dir> <model_name> env
  generate.py <recipe.yaml> <model_dir> <model_name> argv [key=value overrides...]
    (override value 'DEL' removes that default)
"""
import sys
import yaml


# Recipe keys whose CLI flag name differs from `key.replace('_', '-')`.
RENAMES = {"max_model_len": "max-seq-len", "tensor_parallel": "tp-size", "host": "bind"}

# Flags that take no value — presence alone means true. Checked AFTER
# hyphenation/renaming, so members here must already be hyphenated
# ('qwen4-qsa', not 'qwen4_qsa'). A version of this generator that checked
# PRESENCE against the underscored key emitted `--qwen4-qsa true` as two
# tokens instead of the bare flag — caught by generate_test.py before this
# tool was ever run against a real server.
PRESENCE = {"dflash", "no-tui", "qwen4-qsa"}


def build_env(recipe: dict) -> list[str]:
    env = recipe.get("atlas", {}).get("env", {})
    return [f"{k}={'' if v is None else v}" for k, v in env.items()]


def build_argv(recipe: dict, model_dir: str, model_name: str, overrides: list[str]) -> list[str]:
    defaults = {
        k: ("true" if v is True else "false" if v is False else str(v))
        for k, v in recipe["defaults"].items()
    }
    for o in overrides:
        k, v = o.split("=", 1)
        if v == "DEL":
            defaults.pop(k, None)
        else:
            defaults[k] = v

    out = ["serve", "--model-from-path", model_dir, "--model-name", model_name, "--no-tui"]
    for k, v in defaults.items():
        flag = RENAMES.get(k, k.replace("_", "-"))
        if flag in PRESENCE:
            if v == "true":
                out.append("--" + flag)
            continue
        out += ["--" + flag, v]
    return out


def main() -> None:
    recipe_path, model_dir, model_name, mode, *overrides = sys.argv[1:]
    recipe = yaml.safe_load(open(recipe_path))
    if mode == "env":
        print("\n".join(build_env(recipe)))
    elif mode == "argv":
        print("\n".join(build_argv(recipe, model_dir, model_name, overrides)))
    else:
        raise SystemExit(f"mode must be 'env' or 'argv', got {mode!r}")


if __name__ == "__main__":
    main()
