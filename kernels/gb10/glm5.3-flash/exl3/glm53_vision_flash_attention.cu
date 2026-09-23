// SPDX-License-Identifier: AGPL-3.0-only

// Compile the established tensor-core online-softmax kernel for GLM vision's
// exact 64-wide heads. A separate module avoids changing language attention.
#define HDIM 64
// Upstream signature (see the vendored header), not this tree's common copy.
#include "../iq3/glm53_inferspark_prefill_upstream.cuh"
