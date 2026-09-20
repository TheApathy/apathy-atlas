// SPDX-License-Identifier: AGPL-3.0-only
extern "C" __global__ __launch_bounds__(160, 1)
void moe_w4a16_routed_k16_v2_plan(
    const unsigned int* routes,
    const v2::Contract* contract,
    v2::Plan* plan,
    float* staged_activation,
    const int* parent_status,
    int* status
) {
    __shared__ unsigned short ids[v2::ROUTES];
    __shared__ unsigned char leaders[v2::ROUTES], tails[v2::ROUTES];
    __shared__ unsigned char leader_slots[v2::DESCRIPTORS], tail_slots[v2::ROUTES];
    __shared__ unsigned int homogeneous_count, tail_count;
    __shared__ int admission, invalid;
    const unsigned int slot = threadIdx.x;
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != v2::ROUTES || blockDim.y != 1 || blockDim.z != 1 ||
        status == nullptr) return;

    if (slot == 0) {
        admission = 0;
        invalid = 0;
        const v2::Range sr{status, sizeof(int)}, psr{parent_status, sizeof(int)};
        const v2::Range rr{routes, v2::ROUTES * sizeof(unsigned int)};
        const v2::Range cr{contract, sizeof(v2::Contract)};
        const v2::Range pr{plan, sizeof(v2::Plan)};
        const v2::Range ar{staged_activation, v2::ACTIVATION_BYTES};
        const bool status_safe = v2::aligned_to(status, 4) && v2::end_ok(status, sizeof(int));
        const bool status_isolated = !v2::overlap_by_distance(sr, psr) &&
            !v2::overlap_by_distance(sr, rr) && !v2::overlap_by_distance(sr, cr) &&
            !v2::overlap_by_distance(sr, pr) && !v2::overlap_by_distance(sr, ar);
        if (!status_safe || !status_isolated) {
            // Preserve an unsafe control word and every protected byte.
        } else if (!v2::aligned_to(parent_status, 4) || !v2::aligned_to(routes, 4) ||
            !v2::aligned_to(contract, 16) || !v2::aligned_to(plan, 16) ||
            !v2::aligned_to(staged_activation, 16)) {
            status[0] = v2::ERR_ALIGNMENT;
        } else if (!v2::end_ok(psr.ptr, psr.bytes) || !v2::end_ok(rr.ptr, rr.bytes) ||
            !v2::end_ok(cr.ptr, cr.bytes) || !v2::end_ok(pr.ptr, pr.bytes) ||
            !v2::end_ok(ar.ptr, ar.bytes)) {
            status[0] = v2::ERR_OVERFLOW;
        } else if (v2::overlap(psr, rr) || v2::overlap(psr, cr) ||
            v2::overlap(psr, pr) || v2::overlap(psr, ar) || v2::overlap(pr, rr) ||
            v2::overlap(pr, cr) || v2::overlap(pr, ar) || v2::overlap(ar, rr) ||
            v2::overlap(ar, cr)) {
            status[0] = v2::ERR_ALIAS;
        } else if (parent_status[0] != READY) {
            status[0] = v2::ERR_PARENT;
        } else if (status[0] != PENDING || !v2::contract_ok(contract)) {
            status[0] = v2::ERR_ABI;
        } else {
            status[0] = v2::BUILDING;
            plan->ready_magic = 0;
            admission = 1;
        }
    }
    __syncthreads();
    if (!admission) return;

    const unsigned int expert = routes[slot];
    if (expert >= EXPERTS) atomicExch(&invalid, 1);
    ids[slot] = static_cast<unsigned short>(expert);
    leaders[slot] = 0;
    tails[slot] = 0;
    plan->route_snapshot[slot] = expert;
    __syncthreads();
    if (invalid) {
        if (slot == 0) status[0] = v2::ERR_ROUTE;
        return;
    }

    unsigned int rank = 0, total = 0;
    for (unsigned int i = 0; i < v2::ROUTES; ++i) {
        total += ids[i] == expert;
        rank += i < slot && ids[i] == expert;
    }
    leaders[slot] = rank % v2::DESCRIPTOR_WIDTH == 0 &&
        rank + v2::DESCRIPTOR_WIDTH <= total;
    tails[slot] = rank >= total - total % v2::DESCRIPTOR_WIDTH;
    __syncthreads();

    unsigned int leader_index = 0, tail_index = 0;
    for (unsigned int i = 0; i < slot; ++i) {
        leader_index += leaders[i];
        tail_index += tails[i];
    }
    if (leaders[slot]) leader_slots[leader_index] = static_cast<unsigned char>(slot);
    if (tails[slot]) tail_slots[tail_index] = static_cast<unsigned char>(slot);
    __syncthreads();

    if (slot == 0) {
        homogeneous_count = 0;
        tail_count = 0;
        for (unsigned int i = 0; i < v2::ROUTES; ++i) {
            homogeneous_count += leaders[i];
            tail_count += tails[i];
        }
        if (homogeneous_count > v2::DESCRIPTORS ||
            tail_count % v2::DESCRIPTOR_WIDTH != 0 ||
            homogeneous_count + tail_count / v2::DESCRIPTOR_WIDTH != v2::DESCRIPTORS) {
            invalid = 1;
        }
        plan->version = v2::VERSION;
        plan->descriptor_count = v2::DESCRIPTORS;
        plan->homogeneous_count = homogeneous_count;
        plan->heterogeneous_count = tail_count / v2::DESCRIPTOR_WIDTH;
        plan->route_count = v2::ROUTES;
        plan->reserved0 = plan->reserved1 = plan->reserved2 = 0;
    }
    __syncthreads();

    if (slot < v2::DESCRIPTORS && !invalid) {
        v2::Descriptor descriptor{};
        if (slot < homogeneous_count) {
            const unsigned int leader = leader_slots[slot];
            descriptor.shared_expert = ids[leader];
            descriptor.mode = v2::HOMOGENEOUS;
            unsigned int found = 0;
            for (unsigned int i = leader; i < v2::ROUTES && found < v2::DESCRIPTOR_WIDTH; ++i)
                if (ids[i] == ids[leader]) descriptor.slots[found++] = static_cast<unsigned char>(i);
            if (found != v2::DESCRIPTOR_WIDTH) atomicExch(&invalid, 1);
        } else {
            descriptor.shared_expert = v2::HETEROGENEOUS_EXPERT;
            descriptor.mode = v2::HETEROGENEOUS;
            const unsigned int base = (slot - homogeneous_count) * v2::DESCRIPTOR_WIDTH;
            for (unsigned int i = 0; i < v2::DESCRIPTOR_WIDTH; ++i)
                descriptor.slots[i] = tail_slots[base + i];
        }
        plan->descriptors[slot] = descriptor;
    }
    __syncthreads();

    if (slot < v2::DESCRIPTORS && !v2::descriptor_ok(plan, plan->descriptors[slot]))
        atomicExch(&invalid, 1);
    unsigned int coverage = 0;
    for (unsigned int d = 0; d < v2::DESCRIPTORS; ++d)
        for (unsigned int i = 0; i < v2::DESCRIPTOR_WIDTH; ++i)
            coverage += plan->descriptors[d].slots[i] == slot;
    if (coverage != 1) atomicExch(&invalid, 1);
    __syncthreads();

    if (slot == 0) {
        if (invalid) {
            status[0] = v2::ERR_PLAN;
            return;
        }
        __threadfence();
        plan->ready_magic = v2::MAGIC;
        __threadfence();
        status[0] = v2::PLANNED;
    }
}
