# F10 packaging qualification, 2026-09-05

CPU-only; no HTTP, server or GPU run with this new binary.

The existing V4 13-test suite was copied first and failed on missing functions.
After the implementation transfer, all13 tests passed with `-Dwarnings`.
The release client builds offline. Scoped formatting passes. A root comparison
checked nine sections against the frozen V4 source: helpers, run/main, options,
prompts, workload, evidence, provenance, transport and tests. All matched after
removing only the module-boundary `pub(super)` visibility annotations. The last
four files are byte-identical. All source files are at most200 lines.

Reproduction from the Atlas `src` checkout:

```sh
CARGO_TARGET_DIR=/var/tmp/atlas-prefill-census-f10-cpu RUSTFLAGS=-Dwarnings \
  cargo test --locked --offline --manifest-path bench/prefill-census/Cargo.toml
CARGO_TARGET_DIR=/var/tmp/atlas-prefill-census-f10-cpu RUSTFLAGS=-Dwarnings \
  cargo build --locked --offline --release --manifest-path bench/prefill-census/Cargo.toml
cargo fmt --manifest-path bench/prefill-census/Cargo.toml -- --check
```

The new client is
`/var/tmp/atlas-prefill-census-f10-cpu/release/atlas-prefill-census`, SHA256
`e0532b9911767f44b85e4d9541e9371166429b13fa738777eee5bb6be9620e38`.
Cargo.toml SHA256
`d9c1c979ee7398471baeb741b20be6e5d76617a402b3759955f3162a322af236`;
Cargo.lock SHA256
`3fa37ee5c7bdda2c056ae4bd098598eeb8f38013342ecfd1f8f6d6307b324889`.

This does not upgrade historical measurements or qualify additional model
templates. Inherited semantic-review and producer-identity limits are explicitly
listed in README.md. No parent Cargo.toml, production branch recipe, runtime
source, kernel, checkpoint or historical evidence was changed.
