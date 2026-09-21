// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
namespace oi640_gate {
constexpr const char* SOURCE_PREFIX = "kernels/gb10/qwen3.8-flash-next/nvfp4/";
constexpr uint64_t SOURCE_COUNT = 22;
struct Provenance {
    Manifest build, capture, model, producer, hook, server;
    FileIdentity executable, capture_binary, server_binary;
};
bool manifest_artifact(const Manifest& parent, const std::string& prefix, Manifest& output) {
    const auto* path = parent.find(prefix + ".path");
    const auto* digest = parent.find(prefix + ".sha256");
    FrozenFile file;
    std::string body;
    return path != nullptr && digest != nullptr && hex_digest(*digest) &&
        file.open_exact(path->c_str(), *digest) && (file.identity.st_mode & 0222) == 0 &&
        file.read_bounded(body, 1 << 20) && parse_manifest(body, file.digest, output);
}
bool identity_any(const Manifest& manifest, const std::string& prefix,
                  const FileIdentity& identity) {
    uint64_t device = 0, inode = 0, size = 0, mode = 0;
    const auto* mode_text = manifest.find(prefix + ".mode");
    if (mode_text == nullptr || mode_text->size() != 4 || (*mode_text)[0] != '0') return false;
    for (char c : mode_text->substr(1)) {
        if (c < '0' || c > '7') return false;
        mode = mode * 8 + static_cast<uint64_t>(c - '0');
    }
    return field(manifest, prefix + ".path", identity.path) &&
        field(manifest, prefix + ".sha256", identity.digest) &&
        field(manifest, prefix + ".build_id", identity.build_id) &&
        decimal(manifest, prefix + ".dev", device) && device == identity.device &&
        decimal(manifest, prefix + ".ino", inode) && inode == identity.inode &&
        decimal(manifest, prefix + ".size", size) && size == identity.size &&
        mode == identity.mode;
}
bool sha_manifest_artifact(const Manifest& receipt, const std::string& prefix,
                           size_t expected, bool ptx) {
    std::string body;
    if (!artifact(receipt, prefix + ".manifest", &body)) return false;
    std::set<std::string> names;
    size_t begin = 0, count = 0;
    while (begin < body.size()) {
        const size_t end = body.find('\n', begin);
        if (end == std::string::npos || end <= begin + 66 ||
            body.compare(begin + 64, 2, "  ") != 0) return false;
        const std::string digest = body.substr(begin, 64);
        const std::string name = body.substr(begin + 66, end - begin - 66);
        if (!hex_digest(digest) || name.empty() || name.find_first_of(" \t\r\0") != std::string::npos ||
            !names.insert(name).second || (ptx && (name.rfind("./t0__", 0) != 0 ||
            name.size() < 5 || name.compare(name.size() - 4, 4, ".ptx") != 0))) return false;
        ++count; begin = end + 1;
    }
    return count == expected;
}
bool verify_producer(const Manifest& producer, const Manifest& build,
                     const Manifest& capture) {
    if (producer.fields.size() != 58 || !field(producer, "schema", "oi640-producer-sources-v1") ||
        !field(producer, "source.count", "22") ||
        !field(producer, "source.bundle_sha256", OI640_SOURCE_BUNDLE_SHA256) ||
        !field(capture, "source.bundle_sha256", OI640_SOURCE_BUNDLE_SHA256) ||
        !field(producer, "producer.count", "4") ||
        !digest_field(producer, "producer.bundle_sha256") ||
        !field(capture, "producer.bundle_sha256", *producer.find("producer.bundle_sha256"))) return false;
    std::set<std::string> source_paths;
    for (uint64_t i = 0; i < SOURCE_COUNT; ++i) {
        char prefix[32]{}; std::snprintf(prefix, sizeof(prefix), "source.%02llu",
            static_cast<unsigned long long>(i));
        const auto* path = producer.find(std::string(prefix) + ".path");
        const auto* digest = producer.find(std::string(prefix) + ".sha256");
        if (path == nullptr || digest == nullptr || !hex_digest(*digest) || !source_paths.insert(*path).second ||
            !manifest_source(build, *path, *digest)) return false;
    }
    const std::string entry = std::string(SOURCE_PREFIX) +
        "moe_w4a16_orig_i640_compact_prefill_capture.py";
    const auto* entry_digest = producer.find("producer.entry.sha256");
    const auto* python_path = producer.find("python.path");
    FileIdentity python;
    return field(producer, "producer.entry.path", entry) && entry_digest != nullptr &&
        manifest_source(build, entry, *entry_digest) && python_path != nullptr &&
        file_identity(python_path->c_str(), python, true) && identity_any(producer, "python", python);
}
bool verify_hook(const Manifest& hook, const Manifest& build, const Manifest& capture,
                 const FileIdentity& binary) {
    const std::string source = std::string(SOURCE_PREFIX) +
        "moe_w4a16_orig_i640_compact_prefill_capture_preload.c";
    const auto* digest = hook.find("source.sha256");
    const auto* compiler_path = hook.find("compiler.path");
    const auto* source_absolute = hook.find("compile.argv.07");
    const auto* nonce = hook.find("build.nonce");
    FileIdentity compiler;
    FrozenFile source_file;
    if (hook.fields.size() != 36 || digest == nullptr || compiler_path == nullptr || nonce == nullptr ||
        source_absolute == nullptr || !field(hook, "schema", "oi640-hook-compile-v1") ||
        !field(hook, "profile", "release") || !field(hook, "target", "aarch64-linux-gnu") ||
        !field(hook, "source.bundle_sha256", OI640_SOURCE_BUNDLE_SHA256) ||
        !field(hook, "source.path", source) || !manifest_source(build, source, *digest) ||
        !field(capture, "hook.source.sha256", *digest) || !identity_fields(hook, "binary", binary) ||
        !file_identity(compiler_path->c_str(), compiler, true) || !identity_any(hook, "compiler", compiler) ||
        !source_file.open_exact(source_absolute->c_str(), *digest) ||
        source_absolute->size() <= source.size() ||
        source_absolute->compare(source_absolute->size() - source.size(), source.size(), source) != 0 ||
        !field(hook, "compile.argc", "12") || !valid_nonce(nonce->c_str()) ||
        !digest_field(hook, "compiler.version_sha256") || !digest_field(hook, "compile.env_sha256")) return false;
    const std::array<std::string, 12> argv = {compiler.path, "-std=c11", "-O2", "-DNDEBUG",
        "-fPIC", "-shared", "-Wl,-z,relro,-z,now", *source_absolute, "-o", binary.path,
        "-Wl,--build-id=sha1", "-ldl"};
    for (size_t i = 0; i < argv.size(); ++i) {
        char key[32]{}; std::snprintf(key, sizeof(key), "compile.argv.%02zu", i);
        if (!field(hook, key, argv[i])) return false;
    }
    return true;
}
bool verify_server(const Manifest& server, const FileIdentity& binary) {
    return server.fields.size() == 24 && field(server, "schema", "oi640-server-build-v1") &&
        field(server, "target.signature", "gb10|qwen3.8-flash-next|nvfp4|sm_121f") &&
        field(server, "kernel.ptx_count", "154") && field(server, "kernel.override_count", "9") &&
        field(server, "kernel.target_ptx_set_count", "1") && field(server, "kernel.v2_count", "0") &&
        field(server, "source.count", "1839") && field(server, "selected.count", "25") &&
        field(server, "ptx.count", "154") &&
        field(server, "binary.path", "/var/tmp/atlas-flash-b62-build-D6fncRTX/release/spark") &&
        field(server, "binary.sha256", "77fe8bc1ad3ed90c3b3fb4db5323e55d36c5fedc5b76852f36a445d25193d5bd") &&
        field(server, "binary.build_id", "5d9d0cc7cf59c20cd2f12af01e6221a332698645") &&
        field(server, "binary.dev", "66306") && field(server, "binary.ino", "27698744") &&
        field(server, "binary.size", "38569952") &&
        field(server, "source.manifest.path", "/var/tmp/atlas-flash-b62-build-D6fncRTX/source-manifest.sha256") &&
        field(server, "source.manifest.sha256", "703d2d4c651c1ae5c3a868f172f530000fd4c14050b569c74572b893c600b504") &&
        field(server, "selected.manifest.path", "/var/tmp/atlas-flash-b62-build-D6fncRTX/selected-source-manifest.sha256") &&
        field(server, "selected.manifest.sha256", "997218b8bce625a57bc2bfd52d384d54abd070d2ace0e8d85d079fe3b1e54ef7") &&
        field(server, "ptx.manifest.path", "/var/tmp/atlas-flash-b62-build-D6fncRTX/ptx-manifest.sha256") &&
        field(server, "ptx.manifest.sha256", "cbe5fd1ad856ae41a41a46f03870aa1e66563f38d9cf377152b5e40957281253") &&
        field(server, "upstream.receipt.path", "/var/tmp/atlas-flash-b62-build-D6fncRTX/BUILD_RECEIPT.md") &&
        field(server, "upstream.receipt.sha256", "c40ad52e50c6af734e6ff0c573d2ca34db80da9e8525a8dedf17b41bc793547a") &&
        identity_fields(server, "binary", binary) &&
        sha_manifest_artifact(server, "source", 1839, false) &&
        sha_manifest_artifact(server, "selected", 25, false) &&
        sha_manifest_artifact(server, "ptx", 154, true) && artifact(server, "upstream.receipt");
}
bool shape_evidence(const Manifest& capture, const std::string& label, uint32_t rows,
                    const char* offset_path, const char* offset_digest) {
    uint64_t pid = 0, source = 0, listener = 0;
    const auto* receipt_path = capture.find(label + ".offset.path");
    if (offset_path == nullptr || offset_digest == nullptr || receipt_path == nullptr ||
        *receipt_path != offset_path || !field(capture, label + ".offset.sha256", offset_digest) ||
        !artifact(capture, label + ".offset") || !artifact(capture, label + ".request") ||
        !artifact(capture, label + ".response") || !decimal(capture, label + ".process.pid", pid) ||
        !decimal(capture, label + ".process.listener_inode", listener) || listener == 0 ||
        !decimal(capture, label + ".event.source", source) || pid == 0 || source == 0 ||
        !(field(capture, label + ".process.exit", "0") || field(capture, label + ".process.exit", "-15"))) return false;
    for (const char* suffix : {"process.cmdline_sha256", "process.environ_sha256"}) {
        const auto* digest = capture.find(label + "." + suffix);
        if (digest == nullptr || !hex_digest(*digest)) return false;
    }
    const auto* event_path = capture.find(label + ".event.path");
    const auto* event_digest = capture.find(label + ".event.sha256");
    const auto* nonce = capture.find("capture.nonce");
    FrozenFile event_file; CaptureEvent event{};
    if (event_path == nullptr || event_digest == nullptr || nonce == nullptr || !valid_nonce(nonce->c_str()) ||
        !event_file.open_exact(event_path->c_str(), *event_digest) || (event_file.identity.st_mode & 0222) != 0 ||
        !event_file.read_exact(&event, sizeof(event)) || event.magic != 0x4f49363430455631ULL ||
        event.version != 1 || event.pid != pid || event.endpoint != rows * 10 || event.bytes != 2052 ||
        event.source != source || event.seconds <= 0 || event.nanoseconds < 0 ||
        event.nanoseconds >= 1000000000 || std::memcmp(event.nonce, nonce->data(), 64) != 0) return false;
    std::string log;
    return artifact(capture, label + ".log", &log, 64 << 20) &&
        occurrences(log, "QWEN4_PREFILL_SELECTOR_RECEIPT") == 1 &&
        log.find("family=attention") != std::string::npos &&
        log.find("serialized_fallback=false") != std::string::npos &&
        log.find("M=" + std::to_string(rows)) != std::string::npos &&
        log.find("H=2560 L=48 E=512 TOPK=10 I=640 SI=640") != std::string::npos &&
        occurrences(log, "ATLAS_EXPERT_LOAD: n_tokens=" + std::to_string(rows)) == 1;
}
bool load_provenance(Provenance& proof) {
    if (!read_manifest("ATLAS_OI640_BUILD_MANIFEST", "ATLAS_OI640_BUILD_MANIFEST_SHA256", proof.build) ||
        !read_manifest("ATLAS_OI640_CAPTURE_RECEIPT", "ATLAS_OI640_CAPTURE_RECEIPT_SHA256", proof.capture) ||
        !read_manifest("ATLAS_OI640_MODEL_MANIFEST", "ATLAS_OI640_MODEL_MANIFEST_SHA256", proof.model) ||
        !manifest_artifact(proof.capture, "producer.receipt", proof.producer) ||
        !manifest_artifact(proof.capture, "hook.compile_receipt", proof.hook) ||
        !manifest_artifact(proof.capture, "server.build_receipt", proof.server) ||
        !executable_identity(proof.executable)) return false;
    const auto* hook_path = proof.capture.find("capture.binary.path");
    const auto* server_path = proof.capture.find("server.binary.path");
    if (hook_path == nullptr || server_path == nullptr ||
        !file_identity(hook_path->c_str(), proof.capture_binary, true) ||
        !file_identity(server_path->c_str(), proof.server_binary, true)) return false;
    const bool geometry = field(proof.capture, "schema", "oi640-capture-v1") &&
        field(proof.capture, "hidden", "2560") && field(proof.capture, "intermediate", "640") &&
        field(proof.capture, "experts", "512") && field(proof.capture, "top_k", "10") &&
        field(proof.capture, "capture.profile", "release") &&
        field(proof.capture, "m2013.prompt_tokens", "2013") && field(proof.capture, "m8192.prompt_tokens", "8192");
    const bool build = field(proof.build, "schema", "oi640-build-v1") &&
        field(proof.build, "profile", "release") && field(proof.build, "target", "sm_121a") &&
        field(proof.build, "source.count", "22") && field(proof.build, "source.bundle_sha256", OI640_SOURCE_BUNDLE_SHA256) &&
        field(proof.build, "compile_receipt.schema", "oi640-compile-v1") &&
        field(proof.build, "compile_receipt.profile", "release") && field(proof.build, "compile_receipt.target", "sm_121a") &&
        field(proof.build, "compile_receipt.source.count", "22") &&
        field(proof.build, "compile_receipt.source.bundle_sha256", OI640_SOURCE_BUNDLE_SHA256) &&
        field(proof.build, "compile_receipt.model.manifest.sha256", OI640_MODEL_MANIFEST_SHA256) &&
        identity_fields(proof.build, "compile_receipt.binary", proof.executable) &&
        identity_fields(proof.build, "binary", proof.executable) &&
        digest_field(proof.build, "compile_receipt.compile.argv_sha256") &&
        digest_field(proof.build, "compile_receipt.compile.env_sha256") &&
        digest_field(proof.build, "compile_receipt.toolchain.sha256") &&
        proof.build.find("compile_receipt.build.nonce") != nullptr &&
        valid_nonce(proof.build.find("compile_receipt.build.nonce")->c_str());
    const char* m2013 = required_env("ATLAS_OI640_M2013_OFFSETS_SHA256");
    const char* m8192 = required_env("ATLAS_OI640_M8192_OFFSETS_SHA256");
    return geometry && build && artifact(proof.capture, "capture.plan") &&
        identity_fields(proof.capture, "capture.binary", proof.capture_binary) &&
        identity_fields(proof.capture, "server.binary", proof.server_binary) &&
        verify_producer(proof.producer, proof.build, proof.capture) &&
        verify_hook(proof.hook, proof.build, proof.capture, proof.capture_binary) &&
        verify_server(proof.server, proof.server_binary) &&
        shape_evidence(proof.capture, "m2013", 2013, required_env("ATLAS_OI640_M2013_OFFSETS"), m2013) &&
        shape_evidence(proof.capture, "m8192", 8192, required_env("ATLAS_OI640_M8192_OFFSETS"), m8192) &&
        verify_model_manifest(proof.model, proof.build, proof.capture);
}
bool provenance_stable(const Provenance& proof) {
    Provenance current;
    return load_provenance(current) && current.build.digest == proof.build.digest &&
        current.capture.digest == proof.capture.digest && current.model.digest == proof.model.digest &&
        current.producer.digest == proof.producer.digest && current.hook.digest == proof.hook.digest &&
        current.server.digest == proof.server.digest && same_identity(current.executable, proof.executable) &&
        same_identity(current.capture_binary, proof.capture_binary) &&
        same_identity(current.server_binary, proof.server_binary);
}
} // namespace oi640_gate
