"""Read oracle capture taps (DSV41_PORT/oracle/ref/<run>) as numpy arrays.

bf16 taps are stored as their raw uint16 bits; `load(..., raw=True)` keeps the bits (for
bit-identity checks), otherwise they are widened to float32 exactly.
"""
from __future__ import annotations

import json
import os

import numpy as np

REF = os.environ.get("DSV41_ORACLE_REF", "/home/flocka/atlas/DSV41_PORT/oracle/ref")

_NP = {"bfloat16": np.uint16, "float32": np.float32, "int64": np.int64, "bool": np.bool_,
       "float64": np.float64, "int32": np.int32, "uint16": np.uint16, "uint8": np.uint8}


class Run:
    def __init__(self, name: str):
        self.dir = os.path.join(REF, name)
        m = json.load(open(os.path.join(self.dir, "manifest.json")))  # manifest = DONE line
        self.manifest = m
        self.entries = {e["file"]: e for e in m["tensors"]}

    def meta(self, layer: int, tap: str, occ: int = 0) -> dict:
        return self.entries[f"L{layer:02d}.{tap}.{occ:03d}.bin"]

    def has(self, layer: int, tap: str, occ: int = 0) -> bool:
        return f"L{layer:02d}.{tap}.{occ:03d}.bin" in self.entries

    def load(self, layer: int, tap: str, occ: int = 0, raw: bool = False) -> np.ndarray:
        e = self.meta(layer, tap, occ)
        a = np.fromfile(os.path.join(self.dir, e["file"]), dtype=_NP[e["stored_dtype"]])
        a = a.reshape(e["shape"])
        if e["dtype"] == "bfloat16" and not raw:
            a = bf16_to_f32(a)
        return a


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even, as torch's .to(bfloat16) does (finite inputs)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return r.astype(np.uint16)


def round_bf16(x: np.ndarray) -> np.ndarray:
    return bf16_to_f32(f32_to_bf16_bits(x))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
