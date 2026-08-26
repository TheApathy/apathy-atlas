#!/usr/bin/env python3
"""Stream Qwen3.8-Flash-Next's 51B-parameter PLE table to NVFP4.

The official FP8 checkpoint splits the PLE embedding into 128 tensors of
shape [2_500_012, 160].  This tool rewrites only files containing those
tensors and hard-links every other safetensors shard.  Peak host memory is
bounded by one input file plus one output file; quantization itself is
chunked by rows.

Each output shard uses Atlas/ModelOpt's standard group-16 representation:

  shard_N.weight          uint8 [rows, 80]   (two E2M1 values per byte)
  shard_N.weight_scale    FP8   [rows, 10]   (one scale per 16 values)
  shard_N.weight_scale_2  FP32  [1]          (per-shard global scale)

Dequantization is: E2M1[nibble] * weight_scale * weight_scale_2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PLE_MARKER = ".ple.ple_embedding.ngram_embedding.shard_"
GLOBAL_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
FORMAT_TAG = "qwen38-flash-next-ple-nvfp4-v1"
OFFLOAD_FORMAT_TAG = "qwen38-flash-next-ple-nvfp4-direct-v1"
OFFLOAD_PAGE_BYTES = 8192
OFFLOAD_RECORD_BYTES = 90  # 80 packed E2M1 bytes + 10 FP8 scale bytes
OFFLOAD_RECORDS_PER_PAGE = OFFLOAD_PAGE_BYTES // OFFLOAD_RECORD_BYTES  # 91
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def is_ple_weight(name: str) -> bool:
    return PLE_MARKER in name and name.endswith(".weight")


def quantize_chunk(x: torch.Tensor, scale2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact CPU equivalent of ModelOpt NVFP4QTensor.quantize(block_size=16)."""
    if x.ndim != 2 or x.shape[1] % 16:
        raise ValueError(f"expected [rows, K] with K divisible by 16, got {tuple(x.shape)}")
    rows, width = x.shape
    grouped = x.float().reshape(rows, width // 16, 16)
    block_amax = grouped.abs().amax(dim=-1)
    block_scale = block_amax / (6.0 * scale2)
    block_scale[block_scale == 0] = 1.0
    block_scale = block_scale.to(torch.float8_e4m3fn)
    scaled = grouped / (block_scale.float() * scale2).unsqueeze(-1)
    flat = scaled.reshape(rows, width)

    magnitude = flat.abs()
    ordinal = torch.searchsorted(E2M1_BOUNDS, magnitude, out_int32=True).to(torch.uint8)
    # ModelOpt uses round-to-even at the three boundaries whose upper code is odd.
    odd_tie = (magnitude.unsqueeze(-1) == E2M1_BOUNDS[[1, 3, 5]]).any(dim=-1)
    code = ((flat < 0).to(torch.uint8) << 3) + ordinal + odd_tie.to(torch.uint8)
    packed = (code[:, 1::2] << 4) | code[:, 0::2]
    return packed.contiguous(), block_scale.contiguous()


def dequantize(packed: torch.Tensor, scales: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
    code = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8)
    code[:, 0::2] = packed & 0x0F
    code[:, 1::2] = packed >> 4
    values = E2M1_VALUES[code.long()].reshape(packed.shape[0], scales.shape[1], 16)
    return (values * scales.float().unsqueeze(-1) * scale2).flatten(1)


def tensor_absmax(tensor: torch.Tensor, source_scale: float, chunk_rows: int) -> float:
    maximum = 0.0
    for start in range(0, tensor.shape[0], chunk_rows):
        chunk = tensor[start : start + chunk_rows].float()
        maximum = max(maximum, float(chunk.abs().max()) * source_scale)
    return maximum


def quantize_tensor(
    source: torch.Tensor, source_scale: float, chunk_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, width = source.shape
    if width % 16:
        raise ValueError(f"PLE width {width} is not divisible by group size 16")
    maximum = tensor_absmax(source, source_scale, chunk_rows)
    scale2 = torch.tensor([maximum / (6.0 * 448.0) if maximum else 1.0], dtype=torch.float32)
    packed = torch.empty((rows, width // 2), dtype=torch.uint8)
    scales = torch.empty((rows, width // 16), dtype=torch.float8_e4m3fn)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        x = source[start:end].float().mul_(source_scale)
        packed[start:end], scales[start:end] = quantize_chunk(x, scale2)
    return packed, scales, scale2


def find_source_scale(files: list[Path]) -> float:
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.endswith(GLOBAL_SCALE_SUFFIX):
                    return float(handle.get_tensor(name).float().item())
    raise RuntimeError(f"could not find *{GLOBAL_SCALE_SUFFIX} in source checkpoint")


def completed_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.metadata().get("atlas_repack") == FORMAT_TAG
    except Exception:
        return False


def rewrite_file(source: Path, destination: Path, source_scale: float, chunk_rows: int) -> int:
    tensors: dict[str, torch.Tensor] = {}
    converted = 0
    with safe_open(source, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if not is_ple_weight(name):
                tensors[name] = tensor
                continue
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name}: expected FP8 E4M3, got {tensor.dtype}")
            packed, scales, scale2 = quantize_tensor(tensor, source_scale, chunk_rows)
            tensors[name] = packed
            prefix = name.removesuffix(".weight")
            tensors[f"{prefix}.weight_scale"] = scales
            tensors[f"{prefix}.weight_scale_2"] = scale2
            converted += 1
            print(
                f"  {name}: {tuple(tensor.shape)} -> packed={tuple(packed.shape)} "
                f"scale2={float(scale2):.9g}",
                flush=True,
            )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    save_file(tensors, temporary, metadata={"atlas_repack": FORMAT_TAG})
    os.replace(temporary, destination)
    return converted


def write_offload_tensor(
    output: Path, packed: torch.Tensor, scales: torch.Tensor
) -> tuple[int, str]:
    """Write page-indexable records; every row is served by one aligned read.

    91 records fit in an 8 KiB direct-I/O page with only two padding bytes.
    This avoids the 6-byte-per-row waste of a 4 KiB/96-byte layout while
    retaining exactly one O_DIRECT request for any row.
    """
    rows = packed.shape[0]
    temporary = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    zero_page = bytearray(OFFLOAD_PAGE_BYTES)
    with temporary.open("wb", buffering=0) as handle:
        for start in range(0, rows, OFFLOAD_RECORDS_PER_PAGE):
            end = min(start + OFFLOAD_RECORDS_PER_PAGE, rows)
            count = end - start
            records = torch.cat(
                (packed[start:end], scales[start:end].view(torch.uint8)), dim=1
            ).contiguous()
            raw = records.numpy().tobytes()
            page = zero_page.copy()
            page[: count * OFFLOAD_RECORD_BYTES] = raw
            handle.write(page)
            digest.update(page)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output.stat().st_size, digest.hexdigest()


def rewrite_file_offload(
    source: Path,
    destination: Path,
    offload_dir: Path,
    source_scale: float,
    chunk_rows: int,
) -> list[dict[str, object]]:
    """Remove PLE weights from the GPU checkpoint and emit NVMe sidecars."""
    tensors: dict[str, torch.Tensor] = {}
    entries: list[dict[str, object]] = []
    with safe_open(source, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if not is_ple_weight(name):
                tensors[name] = tensor
                continue
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name}: expected FP8 E4M3, got {tensor.dtype}")
            packed, scales, scale2 = quantize_tensor(tensor, source_scale, chunk_rows)
            shard = int(name.rsplit(".shard_", 1)[1].removesuffix(".weight"))
            filename = f"ple-ngram-nvfp4-{shard:03d}.bin"
            size, sha256 = write_offload_tensor(offload_dir / filename, packed, scales)
            entries.append(
                {
                    "tensor": name,
                    "shard": shard,
                    "file": filename,
                    "rows": tensor.shape[0],
                    "width": tensor.shape[1],
                    "scale2": float(scale2),
                    "bytes": size,
                    "sha256": sha256,
                }
            )
            print(
                f"  {name}: offload={filename} bytes={size} scale2={float(scale2):.9g}",
                flush=True,
            )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    save_file(tensors, temporary, metadata={"atlas_repack": OFFLOAD_FORMAT_TAG})
    os.replace(temporary, destination)
    return entries


def copy_metadata(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != "model.safetensors.index.json":
            shutil.copy2(path, destination / path.name)


def rebuild_index(destination: Path) -> None:
    weight_map: dict[str, str] = {}
    total_size = 0
    dtype_bytes = {
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
        "I32": 4, "U32": 4, "F32": 4,
        "I64": 8, "U64": 8, "F64": 8,
    }
    for path in sorted(destination.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in weight_map:
                    raise RuntimeError(f"duplicate output tensor {name}")
                weight_map[name] = path.name
                view = handle.get_slice(name)
                elements = 1
                for dim in view.get_shape():
                    elements *= dim
                total_size += elements * dtype_bytes[view.get_dtype()]
    payload = {"metadata": {"total_size": total_size, "atlas_repack": FORMAT_TAG}, "weight_map": weight_map}
    index = destination / "model.safetensors.index.json"
    temporary = index.with_suffix(index.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, index)


def self_test() -> None:
    torch.manual_seed(38)
    x = torch.randn(37, 160, dtype=torch.float32) * 0.07
    scale2 = torch.tensor([float(x.abs().max()) / (6.0 * 448.0)])
    packed, scales = quantize_chunk(x, scale2)
    restored = dequantize(packed, scales, scale2)
    if not torch.isfinite(restored).all():
        raise AssertionError("non-finite NVFP4 reconstruction")
    relative_l2 = float(torch.linalg.vector_norm(restored - x) / torch.linalg.vector_norm(x))
    if relative_l2 > 0.16:
        raise AssertionError(f"unexpected NVFP4 error: relative_l2={relative_l2}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.bin"
        size, _ = write_offload_tensor(path, packed, scales)
        expected_pages = (x.shape[0] + OFFLOAD_RECORDS_PER_PAGE - 1) // OFFLOAD_RECORDS_PER_PAGE
        if size != expected_pages * OFFLOAD_PAGE_BYTES:
            raise AssertionError(f"offload size {size} != {expected_pages} pages")
        raw = path.read_bytes()
        for row in (0, x.shape[0] - 1):
            page = row // OFFLOAD_RECORDS_PER_PAGE
            slot = row % OFFLOAD_RECORDS_PER_PAGE
            offset = page * OFFLOAD_PAGE_BYTES + slot * OFFLOAD_RECORD_BYTES
            record = raw[offset : offset + OFFLOAD_RECORD_BYTES]
            if record[:80] != packed[row].numpy().tobytes():
                raise AssertionError(f"packed offload row {row} mismatch")
            if record[80:] != scales[row].view(torch.uint8).numpy().tobytes():
                raise AssertionError(f"scale offload row {row} mismatch")
    try:
        from modelopt.torch.quantization.qtensor import NVFP4QTensor

        reference, reference_scales, reference_scale2 = NVFP4QTensor.quantize(
            x.clone(), 16, weights_scaling_factor_2=scale2
        )
        if not torch.equal(packed, reference._quantized_data):
            raise AssertionError("packed E2M1 data differs from ModelOpt")
        if not torch.equal(scales.view(torch.uint8), reference_scales.view(torch.uint8)):
            raise AssertionError("FP8 group scales differ from ModelOpt")
        if not torch.equal(scale2, reference_scale2):
            raise AssertionError("global scale differs from ModelOpt")
        modelopt = "; exact ModelOpt parity"
    except ImportError:
        modelopt = "; ModelOpt not installed (parity check skipped)"
    print(f"self-test passed: relative_l2={relative_l2:.6f}{modelopt}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("destination", nargs="?", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=32768)
    parser.add_argument(
        "--mode", choices=("resident", "offload"), default="resident",
        help="resident writes standard NVFP4 tensors; offload writes 8-KiB-page NVMe sidecars",
    )
    parser.add_argument(
        "--offload-dir", type=Path,
        help="sidecar directory for --mode offload (default: DESTINATION/ple-offload)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.source is None or args.destination is None:
        parser.error("source and destination are required unless --self-test is used")
    source = args.source.resolve()
    destination = args.destination.resolve()
    if source == destination:
        parser.error("destination must differ from source")
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    files = sorted(source.glob("*.safetensors"))
    if not files:
        parser.error(f"no safetensors files found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    offload_dir = (args.offload_dir or destination / "ple-offload").resolve()
    if args.mode == "offload":
        offload_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata(source, destination)
    source_scale = find_source_scale(files)
    print(f"official PLE FP8 dequant scale: {source_scale:.9g}", flush=True)

    total = 0
    offload_entries: list[dict[str, object]] = []
    for number, path in enumerate(files, 1):
        output = destination / path.name
        with safe_open(path, framework="pt", device="cpu") as handle:
            contains_ple = any(is_ple_weight(name) for name in handle.keys())
        if not contains_ple:
            if not output.exists():
                os.link(path, output)
            continue
        if args.mode == "resident" and completed_output(output):
            print(f"[{number}/{len(files)}] resume {path.name}", flush=True)
            continue
        print(f"[{number}/{len(files)}] rewrite {path.name}", flush=True)
        if args.mode == "resident":
            total += rewrite_file(path, output, source_scale, args.chunk_rows)
        else:
            entries = rewrite_file_offload(
                path, output, offload_dir, source_scale, args.chunk_rows
            )
            offload_entries.extend(entries)
            total += len(entries)

    rebuild_index(destination)
    if args.mode == "offload":
        manifest = {
            "format": OFFLOAD_FORMAT_TAG,
            "page_bytes": OFFLOAD_PAGE_BYTES,
            "record_bytes": OFFLOAD_RECORD_BYTES,
            "records_per_page": OFFLOAD_RECORDS_PER_PAGE,
            "packed_bytes": 80,
            "scale_bytes": 10,
            "group_size": 16,
            "entries": sorted(offload_entries, key=lambda item: int(item["shard"])),
        }
        manifest_path = offload_dir / "manifest.json"
        temporary = manifest_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, manifest_path)
    print(f"complete: converted {total} PLE shards into {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
