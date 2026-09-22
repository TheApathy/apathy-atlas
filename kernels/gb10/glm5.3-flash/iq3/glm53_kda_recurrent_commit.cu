// SPDX-License-Identifier: AGPL-3.0-only
// One-thread completion marker for the GLM-5.3 KDA all-34 recurrent commit.
//
// The recurrent payload itself is moved by device-to-device copies enqueued on
// the same stream ahead of this launch (ordinals 0..33, 4 MiB each). This
// kernel publishes nothing but the single completion slot that proves those
// copies were ordered behind the caller's exclusive stream, so the host
// readback can distinguish "copies enqueued and retired" from "never issued".
//
// It writes exactly one 48-byte slot and never touches recurrent state, so a
// forged or reordered launch cannot corrupt persistent memory — the worst it
// can do is publish a slot the host expectation then rejects.

#include <stdint.h>

// Mirrors device_completion.rs: ABI version, success sentinel, slot stride and
// the fixed slot the KdaRecurrentCommit phase reads back (after the 34 conv
// slots 0..33).
#define GLM53_COMPLETION_ABI_VERSION 1U
#define GLM53_COMPLETION_SUCCESS 0x354d4c47U
#define GLM53_COMPLETION_SLOT_U32 12U
#define GLM53_COMPLETION_SLOTS 45U
#define GLM53_KDA_RECURRENT_COMMIT_SLOT 34U
#define GLM53_KDA_RECURRENT_COMMIT_TAG 6U
#define GLM53_KDA_RECURRENT_ORDINALS 34U
#define GLM53_KDA_RECURRENT_ORDINAL_BYTES 4194304ULL
#define GLM53_COMPLETION_MAX_POSITIONS 1048576U

// `slab` is the completion slab base (45 slots x 48 bytes). Every argument is
// re-checked here even though the host prevalidates: this is the last barrier
// before a slot claims success, and a slot that claims success without the
// copies having been enqueued is exactly the corruption this phase exists to
// prevent.
extern "C" __global__ void __launch_bounds__(1, 1)
atlas_glm53_kda_recurrent_commit_marker(
        unsigned int * __restrict__ slab,
        unsigned int slot,
        unsigned int tag,
        unsigned int layer_id,
        unsigned int accepted,
        unsigned int exclusive_end,
        unsigned int capacity,
        unsigned int phase_incarnation,
        unsigned long long owner_generation,
        unsigned long long publish_nonce,
        unsigned long long copied_bytes) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    // Fail closed: publish nothing rather than a slot the host would have to
    // interpret. A missing slot poisons the phase; a wrong slot could retire a
    // commit that never moved its payload.
    if (slab == nullptr ||
        slot != GLM53_KDA_RECURRENT_COMMIT_SLOT ||
        tag != GLM53_KDA_RECURRENT_COMMIT_TAG ||
        accepted != 1U ||
        capacity != GLM53_COMPLETION_MAX_POSITIONS ||
        exclusive_end == 0U || exclusive_end > capacity ||
        phase_incarnation == 0U ||
        owner_generation == 0ULL ||
        publish_nonce == 0ULL ||
        copied_bytes !=
            (unsigned long long)GLM53_KDA_RECURRENT_ORDINALS *
                GLM53_KDA_RECURRENT_ORDINAL_BYTES) {
        return;
    }

    unsigned int * __restrict__ out = slab + (size_t)slot * GLM53_COMPLETION_SLOT_U32;

    // Field order mirrors RawCompletionSlot::decode exactly. `status` is written
    // last so a torn write can never present success over stale identity.
    out[2] = tag;
    out[3] = layer_id;
    out[4] = accepted;
    out[5] = exclusive_end;
    out[6] = capacity;
    out[7] = phase_incarnation;
    out[8] = (unsigned int)(owner_generation & 0xffffffffULL);
    out[9] = (unsigned int)(owner_generation >> 32);
    out[10] = (unsigned int)(publish_nonce & 0xffffffffULL);
    out[11] = (unsigned int)(publish_nonce >> 32);
    out[0] = GLM53_COMPLETION_ABI_VERSION;
    __threadfence_system();
    out[1] = GLM53_COMPLETION_SUCCESS;
}
