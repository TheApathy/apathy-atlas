// SPDX-License-Identifier: AGPL-3.0-only
//
// CB3 -> packed FP4 (e2m1) register decode for DeepSeek-V4.1 experts.
//
// THE PRIMITIVE THE V4.1 LOADER NAMES AS MISSING. `deepseek_v41.rs` stops with
// "CB3 3-bit decode kernels -- none exist in kernels/"; this is that decoder,
// transcribed from the PTX the Python prefill path already runs
// (`tools/cb3_moe.py::_cb3_asm`), which is itself inline PTX rather than
// Triton IR — so this is a transcription of the same instructions, not a
// reimplementation of an algorithm.
//
// FORMAT. A weight is a 3-bit codebook index: two low bits packed 4-per-byte in
// the `lo` plane, one high bit packed 8-per-byte in the `hi` plane. The index
// selects one of 8 codebook entries, held as two uint32 byte-vectors A and B
// (8 bytes = 8 e2m1 nibble pairs), and `prmt` does the selection as a byte
// permute. One call produces EIGHT weights as four packed-FP4 bytes.
//
// WHY IT IS THIS SHAPE. Two lop3 fusions carry it: `a | (b & c)` (immLut 0xF8)
// folds the high-bit mask and the or into one instruction, and `(a | b) & c`
// (immLut 0xA8) does the same for the first step of byte-lane -> nibble
// compaction. The high bit is pre-positioned by shifting H so it lands at bit 2
// of its byte lane, which is where the 3-bit index wants it. Nine instructions
// per half, 20 per invocation for eight weights — about 0.08 warp-instructions
// per weight, which is why decoding in registers beats a scratch buffer: the
// Python path measured 2.3 GB written plus 2.3 GB read per layer for the
// scratch it replaces.
#pragma once
#include <cstdint>

// One half: four weights from `shift`/`bit` positions, selected out of {A,B}.
#define ATLAS_CB3_HALF(OUT, L, H, A, B, SHIFT, BIT, MOVE)                       \
    asm volatile(                                                               \
        "{\n\t"                                                                 \
        ".reg .b32 a, b, ie, t, r;\n\t"                                         \
        "shr.b32 a, %1, " #SHIFT ";\n\t"                                        \
        "and.b32 a, a, 0x03030303;\n\t"                                         \
        MOVE                                                                    \
        "lop3.b32 ie, a, b, 0x04040404, 0xF8;\n\t"                              \
        "shr.b32 t, ie, 4;\n\t"                                                 \
        "lop3.b32 r, ie, t, 0x00FF00FF, 0xA8;\n\t"                              \
        "shr.b32 t, r, 8;\n\t"                                                  \
        "or.b32  r, r, t;\n\t"                                                  \
        "prmt.b32 %0, %2, %3, r;\n\t"                                           \
        "}"                                                                     \
        : "=r"(OUT)                                                             \
        : "r"(L), "r"(A), "r"(B), "r"(H))

// `bit >= 2` shifts right to put the wanted high bit at bit 2; below that it
// must shift LEFT, and getting this branch wrong silently selects a neighbouring
// codebook entry rather than failing.
#define ATLAS_CB3_MOVE_SHR(BIT) "shr.b32 b, %4, " #BIT ";\n\t"
#define ATLAS_CB3_MOVE_SHL(BIT) "shl.b32 b, %4, " #BIT ";\n\t"

/// Eight weights -> four packed-FP4 bytes. `SH` is the lo-plane bit offset
/// (0 or 4); `HB` the hi-plane bit index (0, 2, 4 or 6).
template <int SH, int HB>
__device__ __forceinline__ uint32_t atlas_cb3_decode8(uint32_t L, uint32_t H,
                                                      uint32_t A, uint32_t B);

#define ATLAS_CB3_SPECIALISE(SH, HB, M0, M1)                                    \
    template <>                                                                 \
    __device__ __forceinline__ uint32_t atlas_cb3_decode8<SH, HB>(              \
        uint32_t L, uint32_t H, uint32_t A, uint32_t B) {                       \
        uint32_t ne, no;                                                        \
        ATLAS_CB3_HALF(ne, L, H, A, B, SH, HB, M0);                             \
        ATLAS_CB3_HALF(no, L, H, A, B, SH + 2, HB + 1, M1);                     \
        return ne | (no << 4);                                                  \
    }

// The four call sites `_cb3_tile128` uses, with their MOVE direction resolved
// at compile time from `hb` exactly as the generator does at trace time.
ATLAS_CB3_SPECIALISE(0, 0, ATLAS_CB3_MOVE_SHL(2), ATLAS_CB3_MOVE_SHL(1))
ATLAS_CB3_SPECIALISE(4, 2, ATLAS_CB3_MOVE_SHR(0), ATLAS_CB3_MOVE_SHR(1))
ATLAS_CB3_SPECIALISE(0, 4, ATLAS_CB3_MOVE_SHR(2), ATLAS_CB3_MOVE_SHR(3))
ATLAS_CB3_SPECIALISE(4, 6, ATLAS_CB3_MOVE_SHR(4), ATLAS_CB3_MOVE_SHR(5))
