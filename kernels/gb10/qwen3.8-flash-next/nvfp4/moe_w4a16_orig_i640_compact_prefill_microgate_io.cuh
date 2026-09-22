// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

namespace oi640_gate {

bool hex_digest(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::string hash_fd(int fd) {
    char command[128]{};
    const int n = std::snprintf(command, sizeof(command),
        "/usr/bin/sha256sum /proc/%ld/fd/%d", static_cast<long>(getpid()), fd);
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(command)) return {};
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) return {};
    char digest[65]{};
    const int scanned = std::fscanf(pipe, "%64[0-9a-f]", digest);
    const int closed = pclose(pipe);
    return scanned == 1 && closed == 0 && hex_digest(digest) ? digest : "";
}

struct FrozenFile {
    int fd = -1;
    struct stat identity{};
    std::string digest;
    FrozenFile() = default;
    FrozenFile(const FrozenFile&) = delete;
    ~FrozenFile() { if (fd >= 0) close(fd); }

    bool open_exact(const char* path, const std::string& expected) {
        fd = open(path, O_RDONLY | O_NOFOLLOW);
        if (fd < 0 || fstat(fd, &identity) != 0 || !S_ISREG(identity.st_mode)) return false;
        digest = hash_fd(fd);
        struct stat after{};
        return hex_digest(expected) && digest == expected && fstat(fd, &after) == 0 &&
            identity.st_dev == after.st_dev && identity.st_ino == after.st_ino &&
            identity.st_size == after.st_size && identity.st_mtim.tv_sec == after.st_mtim.tv_sec &&
            identity.st_mtim.tv_nsec == after.st_mtim.tv_nsec;
    }

    bool read_exact(void* output, size_t bytes) {
        if (identity.st_size != static_cast<off_t>(bytes) || lseek(fd, 0, SEEK_SET) != 0) return false;
        auto* out = static_cast<unsigned char*>(output);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t got = read(fd, out + done, bytes - done);
            if (got <= 0) return false;
            done += static_cast<size_t>(got);
        }
        unsigned char extra = 0;
        return read(fd, &extra, 1) == 0;
    }

    bool read_bounded(std::string& output, size_t limit) {
        if (identity.st_size <= 0 || static_cast<size_t>(identity.st_size) > limit ||
            lseek(fd, 0, SEEK_SET) != 0) return false;
        output.resize(static_cast<size_t>(identity.st_size));
        size_t done = 0;
        while (done < output.size()) {
            const ssize_t got = read(fd, output.data() + done, output.size() - done);
            if (got <= 0) return false;
            done += static_cast<size_t>(got);
        }
        unsigned char extra = 0;
        return read(fd, &extra, 1) == 0;
    }
};

struct FileIdentity {
    std::string path, digest, build_id;
    uint64_t device = 0, inode = 0, size = 0;
    uint32_t mode = 0;
};

std::string build_id_fd(int fd) {
    char command[160]{};
    const int n = std::snprintf(command, sizeof(command),
        "/usr/bin/readelf --notes --wide /proc/%ld/fd/%d", static_cast<long>(getpid()), fd);
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(command)) return {};
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) return {};
    char line[512]{}, value[129]{};
    int found = 0;
    while (std::fgets(line, sizeof(line), pipe) != nullptr) {
        const char* marker = std::strstr(line, "Build ID: ");
        if (marker != nullptr && std::sscanf(marker + 10, "%128[0-9a-f]", value) == 1) ++found;
    }
    const int closed = pclose(pipe);
    const size_t length = std::strlen(value);
    return closed == 0 && found == 1 && length >= 16 && length <= 128 ? value : "";
}

bool file_identity(const char* path, FileIdentity& output, bool executable) {
    if (path == nullptr || path[0] != '/') return false;
    char resolved[PATH_MAX]{};
    if (realpath(path, resolved) == nullptr) return false;
    const int fd = open(resolved, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat stat_value{};
    const bool ok = fstat(fd, &stat_value) == 0 && S_ISREG(stat_value.st_mode);
    if (!ok) { close(fd); return false; }
    output.path = resolved;
    output.digest = hash_fd(fd);
    output.build_id = executable ? build_id_fd(fd) : "";
    output.device = static_cast<uint64_t>(stat_value.st_dev);
    output.inode = static_cast<uint64_t>(stat_value.st_ino);
    output.size = static_cast<uint64_t>(stat_value.st_size);
    output.mode = static_cast<uint32_t>(stat_value.st_mode & 0777);
    close(fd);
    return hex_digest(output.digest) && (!executable || !output.build_id.empty());
}

bool same_identity(const FileIdentity& a, const FileIdentity& b) {
    return a.path == b.path && a.digest == b.digest && a.build_id == b.build_id &&
        a.device == b.device && a.inode == b.inode && a.size == b.size && a.mode == b.mode;
}

const char* required_env(const char* key) {
    const char* value = std::getenv(key);
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

bool frozen_artifact(const char* path_key, const char* hash_key, const char* build_hash) {
    const char* path = required_env(path_key);
    const char* expected = required_env(hash_key);
    if (path == nullptr || expected == nullptr || std::string(expected) != build_hash) return false;
    FrozenFile file;
    return file.open_exact(path, expected);
}

bool runtime_artifact(const char* path_key, const char* hash_key) {
    const char* path = required_env(path_key);
    const char* expected = required_env(hash_key);
    if (path == nullptr || expected == nullptr) return false;
    FrozenFile file;
    return file.open_exact(path, expected) && (file.identity.st_mode & 0222) == 0;
}

bool valid_nonce(const char* nonce) {
    if (nonce == nullptr || !hex_digest(nonce)) return false;
    std::array<bool, 16> seen{};
    for (unsigned char c : std::string(nonce))
        seen[c <= '9' ? c - '0' : c - 'a' + 10] = true;
    return std::count(seen.begin(), seen.end(), true) >= 8;
}

bool read_offsets(const char* path_key, const char* hash_key, uint32_t rows,
                  std::vector<int>& offsets, std::string& digest) {
    const char* path = required_env(path_key);
    const char* expected = required_env(hash_key);
    if (path == nullptr || expected == nullptr || !hex_digest(expected)) return false;
    FrozenFile file;
    offsets.resize(oi640::EXPERTS + 1);
    if (!file.open_exact(path, expected) || (file.identity.st_mode & 0222) != 0 ||
        !file.read_exact(offsets.data(), offsets.size() * sizeof(int))) return false;
    digest = file.digest;
    if (offsets.front() != 0 || offsets.back() != static_cast<int>(rows * oi640::TOP_K))
        return false;
    for (size_t i = 1; i < offsets.size(); ++i)
        if (offsets[i] < offsets[i - 1]) return false;
    return true;
}

std::string executable_hash() {
    const int fd = open(("/proc/" + std::to_string(getpid()) + "/exe").c_str(), O_RDONLY);
    if (fd < 0) return {};
    const std::string result = hash_fd(fd);
    close(fd);
    return result;
}

bool executable_identity(FileIdentity& output) {
    const std::string path = "/proc/" + std::to_string(getpid()) + "/exe";
    return file_identity(path.c_str(), output, true) && output.mode == 0555;
}

bool write_receipt(const std::string& body) {
    const char* path = required_env("ATLAS_OI640_RECEIPT_OUT");
    if (path == nullptr) return false;
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) return false;
    size_t done = 0;
    while (done < body.size()) {
        const ssize_t wrote = write(fd, body.data() + done, body.size() - done);
        if (wrote <= 0) { close(fd); return false; }
        done += static_cast<size_t>(wrote);
    }
    const bool ok = fsync(fd) == 0 && fchmod(fd, 0444) == 0 && close(fd) == 0;
    return ok;
}

} // namespace oi640_gate
