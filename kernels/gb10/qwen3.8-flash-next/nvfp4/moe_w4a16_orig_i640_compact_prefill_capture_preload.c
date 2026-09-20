// SPDX-License-Identifier: AGPL-3.0-only
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define OI640_EXPERTS 512U
#define OI640_BYTES ((OI640_EXPERTS + 1U) * sizeof(uint32_t))
#define OI640_MAGIC UINT64_C(0x4f49363430455631)

typedef int (*copy_fn)(void*, uint64_t, size_t);
static copy_fn real_copy;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static _Atomic int captured_2013;
static _Atomic int captured_8192;

struct capture_event {
    uint64_t magic;
    uint32_t version;
    uint32_t pid;
    uint32_t endpoint;
    uint32_t bytes;
    uint64_t source_device_ptr;
    int64_t realtime_seconds;
    int64_t realtime_nanoseconds;
    char nonce[64];
};

static void resolve_copy(void) {
    real_copy = (copy_fn)dlsym(RTLD_NEXT, "cuMemcpyDtoH_v2");
}

static int exact_mode(void) {
    const char* grid = getenv("ATLAS_MOE_EXACT_PREFILL_GRID");
    const char* mode = getenv("ATLAS_OI640_CAPTURE_MODE");
    const char* nonce = getenv("ATLAS_OI640_CAPTURE_NONCE");
    if (!grid || strcmp(grid, "1") || !mode || strcmp(mode, "real-target-route-v1") ||
        !nonce || strlen(nonce) != 64) return 0;
    for (size_t i = 0; i < 64; ++i)
        if (!((nonce[i] >= '0' && nonce[i] <= '9') || (nonce[i] >= 'a' && nonce[i] <= 'f')))
            return 0;
    return 1;
}

static int valid_offsets(const uint32_t* values, uint32_t endpoint) {
    if (values[0] != 0 || values[OI640_EXPERTS] != endpoint) return 0;
    for (size_t i = 1; i <= OI640_EXPERTS; ++i)
        if (values[i] < values[i - 1]) return 0;
    return 1;
}

static int write_all(int fd, const void* source, size_t bytes) {
    const unsigned char* p = (const unsigned char*)source;
    while (bytes) {
        ssize_t n = write(fd, p, bytes);
        if (n <= 0) return 0;
        p += (size_t)n;
        bytes -= (size_t)n;
    }
    return 1;
}

static int seal_file(const char* path, const void* data, size_t bytes) {
    if (!path || path[0] != '/') return 0;
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, bytes) && fsync(fd) == 0 && fchmod(fd, 0444) == 0;
    if (close(fd) != 0) ok = 0;
    return ok;
}

static void capture(const uint32_t* values, uint64_t source, uint32_t endpoint) {
    _Atomic int* state = endpoint == 20130 ? &captured_2013 : &captured_8192;
    int expected = 0;
    if (!atomic_compare_exchange_strong(state, &expected, 1)) return;
    const char* offsets_path = getenv(endpoint == 20130
        ? "ATLAS_OI640_CAPTURE_M2013" : "ATLAS_OI640_CAPTURE_M8192");
    const char* event_path = getenv(endpoint == 20130
        ? "ATLAS_OI640_CAPTURE_M2013_EVENT" : "ATLAS_OI640_CAPTURE_M8192_EVENT");
    const char* nonce = getenv("ATLAS_OI640_CAPTURE_NONCE");
    struct timespec now = {0, 0};
    (void)clock_gettime(CLOCK_REALTIME, &now);
    struct capture_event event = {OI640_MAGIC, 1, (uint32_t)getpid(), endpoint,
        OI640_BYTES, source, now.tv_sec, now.tv_nsec, {0}};
    memcpy(event.nonce, nonce, 64);
    if (!seal_file(offsets_path, values, OI640_BYTES) ||
        !seal_file(event_path, &event, sizeof(event))) atomic_store(state, -1);
}

int cuMemcpyDtoH_v2(void* destination, uint64_t source, size_t bytes) {
    pthread_once(&resolve_once, resolve_copy);
    if (!real_copy) return 999;
    int result = real_copy(destination, source, bytes);
    if (result == 0 && bytes == OI640_BYTES && exact_mode()) {
        const uint32_t* values = (const uint32_t*)destination;
        uint32_t endpoint = values[OI640_EXPERTS];
        if ((endpoint == 20130 || endpoint == 81920) && valid_offsets(values, endpoint))
            capture(values, source, endpoint);
    }
    return result;
}
