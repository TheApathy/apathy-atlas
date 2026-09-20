# Provenance and license boundary

This directory is an isolated, default-unrouted numerical shadow. The CUDA
implementation is derived from Atlas's AGPL-3.0-only
`gated_delta_rule_fla.cu` at commit
`6d79e14cc49bbc0c21a9b68901f5d78a522a2caa`, source SHA-256
`454590abfb614ba9a87601626a8ab6085041db8673882d8de2fa8e5d2508aa6e`.
That implementation remains AGPL-3.0-only.

The equations and API were checked against SGLang commit
`c14312a66420b75ca9a11bf1817c4db1fa26b097`, distributed under Apache-2.0
(LICENSE SHA-256
`1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747`).
No SGLang source text is copied into the CUDA implementation. Audited donor
files and SHA-256 identities:

- `chunk.py`: `8edab1f6fc35b86300a91dc6afd61c2456bd7a4ed3986564456977fdb098f2b2`
- `chunk_fwd.py`: `e6ee7b4601ca12ccda6fd93050acedae25d2b6e6a27a27ebf194a58533a4140c`
- `chunk_delta_h.py`: `580a24d2e91c885ef180f5135978c3cc35f01e96a17776baa4b13fe06533bb60`
- `chunk_o.py`: `c5e5b0f7ccdaa744c5e0eede8ec73a5767b322132a72ce46a56f04bfe4c07564`

The port accepts FP32 log-decay directly, matching SGLang. It is approximate
FLA chunk arithmetic and must never be represented as bit-exact WY32 parity.

The versioned v2 host ABI is an Atlas-side layout adapter, not additional
SGLang-derived code. It accepts Atlas's retained linear FP32 alpha and beta
rows, converts alpha to `log(max(alpha, 1e-30))` on the caller's CUDA stream,
and feeds that compact result to the same port core. It also exposes explicit
Q/K/V base offsets and row strides so the exact Qwen3.8 C1 `[M,10240]` conv
layout is not reinterpreted as the v1 compact tensors. The original v1 ABI is
retained unchanged as the compact provenance oracle; the runtime harness must
prove v1/v2 output and state bytes identical when v1 consumes v2's converted
workspace tail.

The versioned v3 path is an Atlas AGPL CUDA redesign informed by the pinned
SGLang program geometry and equations; no SGLang source text is copied. It
retains the v2 ABI and conversion, preserves v1/v2 as oracles, and mirrors the
audited donor's `(NT,H)` KKT/WU grid, `BV=32` state tiling, `BV=64` output
tiling, BF16 tensor-core contractions, FP32 master state, and explicit solved-A
dataflow. The isolated four-way harness is the sole admission gate and must not
attribute the donor's measured timing to v3 before v3 itself is run.
