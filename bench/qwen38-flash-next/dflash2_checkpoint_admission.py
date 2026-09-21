#!/usr/bin/env python3
"""CPU-only fail-closed admission for native Qwen3.8-Flash-Next DFlash2."""
import argparse
import hashlib
import json
import math
import os
import pathlib
import stat
import struct
FILES = ("config.json", "dflash.py", "dflash2.py", "dflash2_export_receipt.json", "model.safetensors")
COUNT, PARAMS, RAW, BF16_NONFINITE = 96, 695_497_216, 1_390_994_432, __import__("re").compile(rb"\A(?:[\x00-\xff]{2})*?[\x80-\xff][\x7f\xff]", __import__("re").DOTALL)
SOURCE_SHA = "1ad27069394d507f98c5d1c1486c372676adb62ef8836fae2e3ab4bf1e4b58da"
UPSTREAM = {"repository": "NVIDIA-NeMo/Automodel", "commit": "7a36b6f2e1abd63d2027e5b3c32c4102e6685079", "license": "Apache-2.0", "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", "source_sha256": {"dflash2_core.py": "8a95bfc3eb9e0ade2f873aca4d4dd41a1513d4c2c2018fba169ff63b3b7a99fc", "draft_qwen3_dflash2.py": "594e62242c07c40c3d19a8444de384d44240cc6c48db432a7d044d4d595109a4", "train_dflash2.py": "f9e0e9eb5cc6deee9233c514615e8e48806d587ca84451225b1800affae330f4"}}
AdmissionError = RuntimeError
def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
def valid_digest(value):
    return type(value) is str and len(value) == 64 and all(x in "0123456789abcdef" for x in value)
def expected_specs():
    h, i, q, kv, hd = 2560, 8704, 20, 4, 128
    specs = {"fc.weight": (h, h * 8), "hidden_norm.weight": (h,), "norm.weight": (h,)}
    for layer in range(6):
        p = f"layers.{layer}"
        specs.update({f"{p}.input_layernorm.weight": (h,), f"{p}.post_attention_layernorm.weight": (h,), f"{p}.self_attn.q_proj.weight": (q * hd, h), f"{p}.self_attn.k_proj.weight": (kv * hd, h), f"{p}.self_attn.v_proj.weight": (kv * hd, h), f"{p}.self_attn.o_proj.weight": (h, q * hd), f"{p}.self_attn.q_norm.weight": (hd,), f"{p}.self_attn.k_norm.weight": (hd,), f"{p}.mlp.gate_proj.weight": (i, h), f"{p}.mlp.up_proj.weight": (i, h), f"{p}.mlp.down_proj.weight": (h, i)})
        for sub in ("attention_conv", "mlp_conv"):
            specs[f"{p}.{sub}.base_kernel"] = (2, 2, h)
            specs[f"{p}.{sub}.kernel_projection.weight"] = (640, h)
    specs.update({"candidate_selector.hidden_projection.weight": (256, h), "candidate_selector.predecessor_codebook": (248320, 256), "candidate_selector.successor_codebook": (248320, 256)})
    return specs
def expected_config():
    return {"architectures": ["DFlash2DraftModel"], "attention_bias": False, "attention_dropout": 0.0, "auto_map": {"AutoModel": "dflash2.DFlash2DraftModel"}, "block_size": 16, "bos_token_id": 248044, "dflash_config": {"block_size": 16, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248077, "selector_rank": 256, "selector_top_k": 16, "selector_vocab_size": 248077, "target_layer_ids": [1, 7, 13, 20, 26, 33, 39, 46]}, "dtype": "bfloat16", "eos_token_id": 248044, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2560, "initializer_range": 0.02, "intermediate_size": 8704, "is_causal": False, "layer_types": ["sliding_attention"] * 5 + ["full_attention"], "max_position_embeddings": 262144, "max_window_layers": 6, "model_type": "qwen3", "num_attention_heads": 20, "num_hidden_layers": 6, "num_key_value_heads": 4, "num_target_layers": 48, "pad_token_id": None, "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10_000_000, "rope_type": "default"}, "sliding_window": 4096, "tie_word_embeddings": False, "use_cache": True, "use_sliding_window": True, "vocab_size": 248320}
def inventory_sha():
    value = [{"name": name, "shape": list(shape), "dtype": "BF16", "finite": True} for name, shape in sorted(expected_specs().items())]
    return hashlib.sha256(canonical(value)).hexdigest()
def expected_export(model_sha, config_sha):
    return {"schema": "atlas_flash_next_dflash2_export_v1", "architecture": "DFlash2DraftModel", "upstream": UPSTREAM, "tensor_count": COUNT, "parameter_count": PARAMS, "raw_bf16_bytes": RAW, "physical_vocab_size": 248320, "selector_vocab_size": 248077, "model_sha256": model_sha, "config_sha256": config_sha, "inventory_sha256": inventory_sha(), "all_tensors_bfloat16": True, "all_tensors_finite": True}
def secure_stat(path):
    path = pathlib.Path(path)
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise AdmissionError(f"{path} is not a regular file")
    if value.st_nlink != 1:
        raise AdmissionError(f"{path} must have exactly one link")
    if value.st_mode & 0o222:
        raise AdmissionError(f"{path} must be immutable (no write bits)")
    return {"dev": value.st_dev, "inode": value.st_ino, "size": value.st_size, "mtime_ns": value.st_mtime_ns, "mode": stat.S_IMODE(value.st_mode), "nlink": value.st_nlink}
def _open(path, identity):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    value = os.fstat(fd)
    actual = {"dev": value.st_dev, "inode": value.st_ino, "size": value.st_size, "mtime_ns": value.st_mtime_ns, "mode": stat.S_IMODE(value.st_mode), "nlink": value.st_nlink}
    if actual != identity:
        os.close(fd)
        raise AdmissionError(f"{path} identity changed before open")
    return fd
def sha256_file(path, identity=None):
    identity = identity or secure_stat(path)
    digest = hashlib.sha256()
    with os.fdopen(_open(path, identity), "rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
def _pairs(items):
    if len(result := dict(items)) != len(items):
        raise AdmissionError("duplicate JSON key")
    return result
def read_canonical_json(path, limit=2 * 1024 * 1024):
    identity = secure_stat(path)
    if identity["size"] > limit:
        raise AdmissionError(f"{path} exceeds JSON size limit")
    fd = _open(path, identity)
    try:
        raw = b""
        while len(raw) < identity["size"]:
            chunk = os.read(fd, identity["size"] - len(raw))
            if not chunk:
                raise AdmissionError(f"truncated JSON file {path}")
            raw += chunk
    finally:
        os.close(fd)
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdmissionError(f"invalid JSON in {path}: {error}") from error
    if raw != canonical(value):
        raise AdmissionError(f"{path} is not canonical JSON")
    return value
def validate_config(value):
    if canonical(value) != canonical(expected_config()):
        raise AdmissionError("DFlash2 config differs from the exact native contract")
def validate_export(value, model_sha, config_sha):
    if canonical(value) != canonical(expected_export(model_sha, config_sha)):
        raise AdmissionError("DFlash2 export receipt does not close exact hashes/schema/sources")
def _header(fd, size, specs, expected_params):
    prefix = os.pread(fd, 8, 0)
    if len(prefix) != 8:
        raise AdmissionError("truncated safetensors length")
    length = struct.unpack("<Q", prefix)[0]
    if length < 2 or length % 8 or length > 16 * 1024 * 1024 or 8 + length > size:
        raise AdmissionError("invalid or oversized safetensors header")
    raw = os.pread(fd, length, 8)
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdmissionError(f"invalid safetensors header: {error}") from error
    if type(value) is not dict:
        raise AdmissionError("safetensors header must be a JSON object")
    metadata = value.pop("__metadata__", {})
    if metadata not in ({}, {"format": "pt"}) or set(value) != set(specs):
        raise AdmissionError("noncanonical metadata or tensor names")
    spans = []
    for name, shape in specs.items():
        item = value[name]
        if type(item) is not dict or set(item) != {"dtype", "shape", "data_offsets"} or item["dtype"] != "BF16":
            raise AdmissionError(f"invalid tensor descriptor: {name}")
        observed, offsets = item["shape"], item["data_offsets"]
        if type(observed) is not list or any(type(x) is not int or x <= 0 for x in observed) or tuple(observed) != shape:
            raise AdmissionError(f"invalid tensor shape: {name}")
        if type(offsets) is not list or len(offsets) != 2 or any(type(x) is not int or x < 0 for x in offsets):
            raise AdmissionError(f"invalid tensor offsets: {name}")
        if offsets[1] - offsets[0] != math.prod(shape) * 2:
            raise AdmissionError(f"invalid tensor byte length: {name}")
        spans.append((offsets[0], offsets[1], name))
    if sum(math.prod(x) for x in specs.values()) != expected_params:
        raise AdmissionError("parameter-count contract drift")
    cursor = 0
    for start, end, _ in sorted(spans):
        if start != cursor or end < start:
            raise AdmissionError("safetensors offsets overlap or contain a gap")
        cursor = end
    if cursor != expected_params * 2 or size != 8 + length + cursor:
        raise AdmissionError("safetensors payload has a gap or trailing bytes")
    return 8 + length, {name: (start, end) for start, end, name in spans}
def inspect_model(path, specs, expected_params, full_payload):
    identity, digest = secure_stat(path), hashlib.sha256()
    fd = _open(path, identity)
    try:
        data_start, _ = _header(fd, identity["size"], specs, expected_params)
        os.lseek(fd, 0, os.SEEK_SET)
        position = 0
        while chunk := os.read(fd, 4 * 1024 * 1024):
            digest.update(chunk)
            if full_payload and position + len(chunk) > data_start:
                payload = chunk[max(0, data_start - position):]
                if BF16_NONFINITE.search(payload):
                    raise AdmissionError("model contains a nonfinite BF16 encoding")
            position += len(chunk)
    finally:
        os.close(fd)
    return {"sha256": digest.hexdigest(), "qualification": "FULL_PAYLOAD_BF16_FINITE" if full_payload else "HEADER_ONLY_NO_PAYLOAD_FINITE_SCAN", "tensor_count": len(specs), "parameter_count": expected_params, "raw_bf16_bytes": expected_params * 2}
def shared_digest(path, specs):
    identity, digest = secure_stat(path), hashlib.sha256()
    fd = _open(path, identity)
    try:
        data_start, spans = _header(fd, identity["size"], specs, PARAMS)
        names = sorted(name for name in specs if "_conv." not in name and not name.startswith("candidate_selector."))
        for name in names:
            start, end = spans[name]
            digest.update(name.encode())
            digest.update(json.dumps(list(specs[name]), separators=(",", ":")).encode())
            digest.update(b"torch.bfloat16")
            offset = start
            while offset < end:
                chunk = os.pread(fd, min(4 * 1024 * 1024, end - offset), data_start + offset)
                if not chunk:
                    raise AdmissionError("truncated shared tensor payload")
                digest.update(chunk)
                offset += len(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()
def write_receipt(path, value):
    value = dict(value)
    value["receipt_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    raw = canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o444)
    with os.fdopen(fd, "wb") as handle:
        if handle.write(raw) != len(raw):
            raise AdmissionError("short write while publishing admission receipt")
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), 0o444)
    return value
def admit(args):
    root = pathlib.Path(args.checkpoint)
    if pathlib.Path(args.output).resolve(strict=False).is_relative_to(root.resolve(strict=False)):
        raise AdmissionError("receipt output must resolve outside checkpoint")
    directory = root.lstat()
    if not stat.S_ISDIR(directory.st_mode) or directory.st_mode & 0o222:
        raise AdmissionError("checkpoint must be a real immutable directory")
    if {path.name for path in root.iterdir()} != set(FILES):
        raise AdmissionError("checkpoint must contain exactly the five canonical files")
    directory_before = {"dev": directory.st_dev, "inode": directory.st_ino, "mode": stat.S_IMODE(directory.st_mode), "mtime_ns": directory.st_mtime_ns}
    paths = {name: root / name for name in FILES}
    before = {name: secure_stat(path) for name, path in paths.items()}
    config = read_canonical_json(paths["config.json"])
    validate_config(config)
    config_sha = sha256_file(paths["config.json"], before["config.json"])
    model = inspect_model(paths["model.safetensors"], expected_specs(), PARAMS, args.full_payload)
    for name in ("dflash.py", "dflash2.py"):
        if sha256_file(paths[name], before[name]) != SOURCE_SHA:
            raise AdmissionError(f"{name} source identity drift")
    export = read_canonical_json(paths["dflash2_export_receipt.json"])
    validate_export(export, model["sha256"], config_sha)
    hashes = {"config.json": config_sha, "model.safetensors": model["sha256"]}
    hashes.update({name: sha256_file(paths[name], before[name]) for name in FILES if name not in hashes})
    lineage = None
    if bool(args.training_lineage) != bool(args.expected_training_lineage_sha256):
        raise AdmissionError("training lineage path and expected SHA-256 must be supplied together")
    if args.training_lineage:
        lineage_path = pathlib.Path(args.training_lineage)
        lineage_before = secure_stat(lineage_path)
        lineage_sha = sha256_file(lineage_path, lineage_before)
        lineage_value = read_canonical_json(lineage_path)
        if not valid_digest(args.expected_training_lineage_sha256) or lineage_sha != args.expected_training_lineage_sha256 or type(lineage_value) is not dict or lineage_value.get("schema") != "atlas_dflash_training_lineage_v1":
            raise AdmissionError("training-lineage identity/schema mismatch")
        lineage_after = secure_stat(lineage_path)
        if lineage_before != lineage_after:
            raise AdmissionError("training-lineage identity changed during admission")
        lineage = {"path": str(lineage_path), "sha256": lineage_sha, "identity_pre": lineage_before, "identity_post": lineage_after}
    if args.expected_v3_shared_state_digest and not valid_digest(args.expected_v3_shared_state_digest):
        raise AdmissionError("expected V3 shared-state digest is malformed")
    shared = shared_digest(paths["model.safetensors"], expected_specs()) if args.expected_v3_shared_state_digest else None
    if shared != args.expected_v3_shared_state_digest:
        raise AdmissionError("V3 shared-state digest mismatch")
    after = {name: secure_stat(path) for name, path in paths.items()}
    if before != after:
        raise AdmissionError("checkpoint identity changed during admission")
    directory = root.lstat()
    directory_after = {"dev": directory.st_dev, "inode": directory.st_ino, "mode": stat.S_IMODE(directory.st_mode), "mtime_ns": directory.st_mtime_ns}
    if directory_before != directory_after:
        raise AdmissionError("checkpoint directory identity changed during admission")
    tool_sha = hashlib.sha256(pathlib.Path(__file__).resolve().read_bytes()).hexdigest()
    receipt = {"schema": "atlas_flash_next_dflash2_checkpoint_admission_v1", "qualification": model.pop("qualification"), "checkpoint": str(root.resolve()), "directory_identity_pre": directory_before, "directory_identity_post": directory_after, "files": {name: {"identity_pre": before[name], "identity_post": after[name], "sha256": hashes[name]} for name in FILES}, "model": model, "export_receipt_sha256": hashes["dflash2_export_receipt.json"], "source_alias_sha256": SOURCE_SHA, "v3_shared_state_digest": shared, "training_lineage": lineage, "admission_tool_sha256": tool_sha}
    return write_receipt(args.output, receipt)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-payload", action="store_true")
    for option in ("expected-v3-shared-state-digest", "training-lineage", "expected-training-lineage-sha256"):
        parser.add_argument(f"--{option}")
    args = parser.parse_args()
    try:
        receipt = admit(args)
    except (AdmissionError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"output": args.output, "qualification": receipt["qualification"], "receipt_payload_sha256": receipt["receipt_payload_sha256"]}, sort_keys=True))
if __name__ == "__main__":
    main()
