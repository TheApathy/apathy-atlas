# SPDX-License-Identifier: AGPL-3.0-only
"""Minimal CUDA-driver loader for the five externally sealed cubins."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from frozen_triton_c143_io import GateError

from .verified_loader import load_verified_file

CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
TRITON_RESERVED_SHARED_BYTES = 1024
U64 = ctypes.c_uint64
U32 = ctypes.c_uint32
F32 = ctypes.c_float


def device_pointer(value: Any) -> tuple[type[ctypes._SimpleCData], int]:
    return U64, value.data_ptr()


class CudaDriver:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_LOCAL)
        self._prototypes()
        self.check(self.lib.cuInit(0), "cuInit")
        device = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        self.device = device
        context = ctypes.c_void_p()
        self.check(
            self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),
            "cuDevicePrimaryCtxRetain",
        )
        self.check(self.lib.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        self.context = context
        self.modules: list[ctypes.c_void_p] = []

    def _prototypes(self) -> None:
        u32 = ctypes.c_uint
        ptr = ctypes.c_void_p
        self.lib.cuInit.argtypes = [u32]
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGetName.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ptr), ctypes.c_int]
        self.lib.cuCtxSetCurrent.argtypes = [ptr]
        self.lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ptr), ptr]
        self.lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ptr),
            ptr,
            ctypes.c_char_p,
        ]
        self.lib.cuFuncGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ptr,
        ]
        self.lib.cuFuncSetAttribute.argtypes = [ptr, ctypes.c_int, ctypes.c_int]
        self.lib.cuLaunchKernel.argtypes = [
            ptr,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            ptr,
            ctypes.POINTER(ptr),
            ptr,
        ]
        self.lib.cuModuleUnload.argtypes = [ptr]
        self.lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]

    def check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        name = ctypes.c_char_p()
        message = ctypes.c_char_p()
        self.lib.cuGetErrorName(status, ctypes.byref(name))
        self.lib.cuGetErrorString(status, ctypes.byref(message))
        detail = (name.value or b"CUDA_ERROR").decode("ascii", "replace")
        text = (message.value or b"unknown").decode("utf-8", "replace")
        raise GateError(f"{operation}: {detail}: {text}")

    def device_identity(self) -> dict[str, Any]:
        def attribute(number: int) -> int:
            value = ctypes.c_int()
            self.check(
                self.lib.cuDeviceGetAttribute(ctypes.byref(value), number, self.device),
                "cuDeviceGetAttribute",
            )
            return value.value

        name = ctypes.create_string_buffer(256)
        self.check(
            self.lib.cuDeviceGetName(name, len(name), self.device), "cuDeviceGetName"
        )
        result = {
            "ordinal": 0,
            "name": name.value.decode("utf-8", "strict"),
            "major": attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            "minor": attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
            "multiprocessor_count": attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        }
        if "GB10" not in result["name"] or (
            result["major"],
            result["minor"],
            result["multiprocessor_count"],
        ) != (12, 1, 48):
            raise GateError(f"executor requires exact GB10 SM12.1/48SM, got {result}")
        return result

    def load_function(
        self, path: Path, name: str, contract: dict[str, Any]
    ) -> ctypes.c_void_p:
        expected_sha, expected_size = contract["files"]["cubin"]

        def load_bytes(raw: bytes) -> ctypes.c_void_p:
            storage = ctypes.create_string_buffer(raw, len(raw))
            if __import__("hashlib").sha256(storage.raw).hexdigest() != expected_sha:
                raise GateError(f"{name}: cubin memory image drift")
            module = ctypes.c_void_p()
            self.check(
                self.lib.cuModuleLoadData(
                    ctypes.byref(module), ctypes.cast(storage, ctypes.c_void_p)
                ),
                "cuModuleLoadData",
            )
            if __import__("hashlib").sha256(storage.raw).hexdigest() != expected_sha:
                raise GateError(f"{name}: cubin memory image changed during load")
            return module

        module = load_verified_file(
            path, f"executor cubin:{name}", expected_sha, expected_size, load_bytes
        )
        function = ctypes.c_void_p()
        self.check(
            self.lib.cuModuleGetFunction(ctypes.byref(function), module, name.encode()),
            "cuModuleGetFunction",
        )
        expected = {
            CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: contract["static_shared_bytes"]
            - TRITON_RESERVED_SHARED_BYTES,
            CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: contract["local_bytes"],
            CU_FUNC_ATTRIBUTE_NUM_REGS: contract["registers_per_thread"],
        }
        for attribute, wanted in expected.items():
            actual = ctypes.c_int()
            self.check(
                self.lib.cuFuncGetAttribute(ctypes.byref(actual), attribute, function),
                "cuFuncGetAttribute",
            )
            if actual.value != wanted:
                raise GateError(
                    f"{name}: CUDA function resource drift: "
                    f"attribute={attribute} wanted={wanted} actual={actual.value}"
                )
        maximum = ctypes.c_int()
        self.check(
            self.lib.cuFuncGetAttribute(
                ctypes.byref(maximum), CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function
            ),
            "cuFuncGetAttribute",
        )
        if maximum.value < contract["block"][0]:
            raise GateError(f"{name}: insufficient maximum threads")
        self.check(
            self.lib.cuFuncSetAttribute(
                function,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                contract["dynamic_shared_bytes"],
            ),
            "cuFuncSetAttribute",
        )
        self.modules.append(module)
        return function

    def launch(
        self,
        function: ctypes.c_void_p,
        grid: list[int],
        block: list[int],
        shared: int,
        stream: int,
        arguments: list[tuple[type[ctypes._SimpleCData], int | float]],
    ) -> None:
        holders = [kind(value) for kind, value in arguments]
        pointers = (ctypes.c_void_p * len(holders))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in holders)
        )
        self.check(
            self.lib.cuLaunchKernel(
                function, *grid, *block, shared, ctypes.c_void_p(stream), pointers, None
            ),
            "cuLaunchKernel",
        )

    def launch_typed(
        self,
        function: ctypes.c_void_p,
        plan: dict[str, Any],
        stream: int,
        arguments: list[tuple[type[ctypes._SimpleCData], int | float]],
    ) -> None:
        types = [entry[1] for entry in plan["driver_params"]]
        actual = [{U64: "u64", U32: "u32", F32: "f32"}[kind] for kind, _ in arguments]
        if actual != types or arguments[-2:] != [(U64, 0), (U64, 0)]:
            raise GateError(f"{plan['role']}: executor driver ABI drift")
        self.launch(
            function,
            plan["grid"],
            plan["block"],
            plan["dynamic_shared_bytes"],
            stream,
            arguments,
        )

    def close_modules(self) -> None:
        for module in reversed(self.modules):
            self.check(self.lib.cuModuleUnload(module), "cuModuleUnload")
        self.modules.clear()
