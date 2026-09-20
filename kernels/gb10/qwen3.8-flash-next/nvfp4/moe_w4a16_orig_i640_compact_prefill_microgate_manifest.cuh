// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
namespace oi640_gate {
struct Manifest {
    std::map<std::string, std::string> fields;
    std::string digest;
    const std::string* find(const std::string& key) const {
        const auto it = fields.find(key);
        return it == fields.end() ? nullptr : &it->second;
    }
};
bool manifest_key(const std::string& key) {
    return !key.empty() && key.size() <= 96 && std::all_of(key.begin(), key.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
    });
}
bool parse_manifest(const std::string& body, const std::string& digest, Manifest& output) {
    if (body.empty() || body.back() != '\n' || body.find('\0') != std::string::npos) return false;
    std::string previous;
    size_t begin = 0;
    while (begin < body.size()) {
        const size_t end = body.find('\n', begin);
        if (end == std::string::npos || end == begin) return false;
        const std::string line = body.substr(begin, end - begin);
        const size_t equal = line.find('=');
        if (equal == std::string::npos || equal == 0 || equal + 1 == line.size()) return false;
        const std::string key = line.substr(0, equal), value = line.substr(equal + 1);
        if (!manifest_key(key) || key <= previous || value.size() > 4096 ||
            !output.fields.emplace(key, value).second) return false;
        previous = key;
        begin = end + 1;
    }
    output.digest = digest;
    return output.fields.size() <= 2048;
}
bool read_manifest(const char* path_key, const char* hash_key, Manifest& output, size_t limit = 1 << 20) {
    const char* path = required_env(path_key);
    const char* expected = required_env(hash_key);
    if (path == nullptr || expected == nullptr || !hex_digest(expected)) return false;
    FrozenFile file;
    std::string body;
    return file.open_exact(path, expected) && (file.identity.st_mode & 0222) == 0 &&
        file.read_bounded(body, limit) && parse_manifest(body, file.digest, output);
}
bool field(const Manifest& manifest, const std::string& key, const std::string& value) {
    const auto* found = manifest.find(key);
    return found != nullptr && *found == value;
}
bool digest_field(const Manifest& manifest, const std::string& key) {
    const auto* value = manifest.find(key); return value != nullptr && hex_digest(*value);
}
bool decimal(const Manifest& manifest, const std::string& key, uint64_t& output) {
    const auto* value = manifest.find(key);
    if (value == nullptr || value->empty() || value->size() > 20 ||
        !std::all_of(value->begin(), value->end(), [](unsigned char c) { return c >= '0' && c <= '9'; }))
        return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value->c_str(), &end, 10);
    if (errno != 0 || end != value->c_str() + value->size()) return false;
    output = parsed;
    return true;
}
bool identity_fields(const Manifest& manifest, const std::string& prefix,
                     const FileIdentity& identity) {
    uint64_t device = 0, inode = 0, size = 0;
    return field(manifest, prefix + ".path", identity.path) &&
        field(manifest, prefix + ".sha256", identity.digest) &&
        field(manifest, prefix + ".build_id", identity.build_id) &&
        field(manifest, prefix + ".mode", "0555") &&
        decimal(manifest, prefix + ".dev", device) && device == identity.device &&
        decimal(manifest, prefix + ".ino", inode) && inode == identity.inode &&
        decimal(manifest, prefix + ".size", size) && size == identity.size && identity.mode == 0555;
}
bool manifest_source(const Manifest& manifest, const std::string& path,
                     const std::string& digest) {
    for (const auto& [key, value] : manifest.fields) {
        if (key.size() > 5 && key.compare(key.size() - 5, 5, ".path") == 0 && value == path) {
            const auto* found = manifest.find(key.substr(0, key.size() - 5) + ".sha256");
            return found != nullptr && *found == digest;
        }
    }
    return false;
}
bool verify_model_manifest(const Manifest& model, const Manifest& build,
                           const Manifest& capture) {
    const char* root = required_env("ATLAS_OI640_MODEL_ROOT");
    uint64_t count = 0;
    if (root == nullptr || root[0] != '/' || model.digest != OI640_MODEL_MANIFEST_SHA256 ||
        !field(model, "schema", "oi640-model-v1") ||
        !decimal(model, "shard.count", count) || count == 0 || count > 256 ||
        !field(build, "model.manifest.sha256", model.digest) ||
        !field(capture, "model.manifest.sha256", model.digest) ||
        !field(build, "model.config.sha256", OI640_CONFIG_SHA256) ||
        !field(build, "model.index.sha256", OI640_INDEX_SHA256) ||
        !field(capture, "model.config.sha256", OI640_CONFIG_SHA256) ||
        !field(capture, "model.index.sha256", OI640_INDEX_SHA256)) return false;
    auto exact_file = [&](const std::string& name, const std::string& digest, uint64_t size) {
        if (name.empty() || name.find('/') != std::string::npos || name.find("..") != std::string::npos)
            return false;
        const std::string path = std::string(root) + "/" + name;
        FrozenFile file;
        return file.open_exact(path.c_str(), digest) && (size == UINT64_MAX ||
            static_cast<uint64_t>(file.identity.st_size) == size);
    };
    if (!field(model, "config.sha256", OI640_CONFIG_SHA256) ||
        !field(model, "index.sha256", OI640_INDEX_SHA256) ||
        !exact_file("config.json", OI640_CONFIG_SHA256, UINT64_MAX) ||
        !exact_file("model.safetensors.index.json", OI640_INDEX_SHA256, UINT64_MAX))
        return false;
    for (uint64_t i = 0; i < count; ++i) {
        char prefix[32]{};
        std::snprintf(prefix, sizeof(prefix), "shard.%03llu", static_cast<unsigned long long>(i));
        const auto* name = model.find(std::string(prefix) + ".path");
        const auto* digest = model.find(std::string(prefix) + ".sha256");
        uint64_t size = 0;
        if (name == nullptr || digest == nullptr || !hex_digest(*digest) ||
            !decimal(model, std::string(prefix) + ".size", size) || !exact_file(*name, *digest, size))
            return false;
    }
    return field(build, "model.shard.count", std::to_string(count)) &&
        field(capture, "model.shard.count", std::to_string(count));
}
struct CaptureEvent { uint64_t magic; uint32_t version, pid, endpoint, bytes; uint64_t source;
    int64_t seconds, nanoseconds; char nonce[64]; }; static_assert(sizeof(CaptureEvent) == 112);
bool artifact(const Manifest& manifest, const std::string& prefix, std::string* body = nullptr,
              size_t limit = 1 << 20) {
    const auto* path = manifest.find(prefix + ".path");
    const auto* digest = manifest.find(prefix + ".sha256");
    if (path == nullptr || digest == nullptr || !hex_digest(*digest)) return false;
    FrozenFile file;
    if (!file.open_exact(path->c_str(), *digest) || (file.identity.st_mode & 0222) != 0) return false;
    return body == nullptr || file.read_bounded(*body, limit);
}
size_t occurrences(const std::string& body, const std::string& needle) {
    size_t count = 0, position = 0;
    while ((position = body.find(needle, position)) != std::string::npos) {
        ++count; position += needle.size();
    }
    return count;
}
} // namespace oi640_gate
