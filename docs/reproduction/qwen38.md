# Qwen3.8 coding reproduction

This package is the public, secret-free reproduction interface for Apathy
Atlas, TheApathy's performance-focused Atlas fork. It records benchmark
evidence without publishing model weights, credentials, host paths, process
state, full generated responses or machine-specific logs.

## Current status

The harness and sanitized reference evidence are public. The exact Qwen3.8
source stack and the native-v2 drafter do not yet have immutable public
revisions, so `manifest.json` deliberately marks the release as incomplete.
The command currently verifies an already-running compatible server. It will
gain build, model-download and serve phases only after those revisions exist.

## Reproduce against a server

```bash
git clone --depth 1 --branch feat/qwen38-repro-public \
  https://github.com/TheApathy/apathy-atlas.git
cd apathy-atlas
./scripts/reproduce qwen38-coding
```

Use a different endpoint when necessary:

```bash
./scripts/reproduce qwen38-coding \
  --endpoint http://127.0.0.1:8896/v1/chat/completions
```

The gate requires three sequential requests, temperature zero,
`reasoning_effort: none`, 1,500 completion tokens, one stable output hash and a
40 tok/s median floor. The checked-in reference is 40.9466 tok/s median across
three deterministic runs (40.9466, 40.9857 and 40.5193 tok/s), each producing
1,500 completion tokens and the same content fingerprint. Results are written
under ignored `runs/` and contain hashes and metrics, not response text.

## Evidence policy

Tracked evidence may contain:

- canonical request hashes;
- token counts and finish reason;
- content and stable-output hashes;
- per-run server decode TPS and client wall time;
- deterministic pass/fail state.

The checked-in gamma sweep retains every per-run server TPS value, the common
output fingerprint and the SHA-256 of each complete private-side source
artifact. This corroborates the published width comparison without exposing
the raw logs or local filesystem identities embedded in those source files.

Tracked evidence must not contain:

- API tokens or credential files;
- absolute user or model-cache paths;
- model weights or quantized sidecars;
- full prompts from private workloads;
- full generated responses;
- hostnames, process IDs, environment dumps or raw serve logs.

Run the public-safety gate before committing:

```bash
./scripts/reproduce verify-public-safety
```

## Release completion requirements

Before creating an immutable `qwen38-vX.Y.Z` tag, populate every null field in
`manifest.json` with:

1. a reviewed source commit;
2. immutable target and drafter revisions;
3. their manifest hashes;
4. the exact build environment;
5. the frozen binary hash;
6. a clean three-run result produced from the tag.

Once complete, the user-facing command will target the immutable tag rather
than a development branch.
