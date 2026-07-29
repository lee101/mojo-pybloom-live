"""ctypes bridge to the Mojo Bloom-filter kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYBLOOM_LIVE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pybloom-live.so"
)

I = ctypes.c_int64
U64 = ctypes.c_uint64
P = ctypes.c_void_p
_I64_MAX = (1 << 63) - 1

_SIGNATURES = {
    "mpbl_hash64": ([P, I, U64], U64),
    "mpbl_contains": ([P, P, I, I, I], I),
    "mpbl_add": ([P, P, I, I, I, I], I),
    "mpbl_contains_many": ([P, P, P, I, I, I, P], None),
    "mpbl_add_many": ([P, P, P, I, I, I, P], None),
    "mpbl_scalable_add": ([P, P, P, I, P, I, I], I),
    "mpbl_or": ([P, P, P, I, I], None),
    "mpbl_and": ([P, P, P, I, I], None),
}

_library: ctypes.CDLL | None = None
_max_runtime: ctypes.CDLL | None = None
_bytes_address = ctypes.pythonapi.PyBytes_AsString
_bytes_address.argtypes = [ctypes.py_object]
_bytes_address.restype = ctypes.c_void_p


def build() -> str:
    subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        check=True,
    )
    return LIB


def lib() -> ctypes.CDLL:
    global _library, _max_runtime
    if _library is None:
        if not os.path.exists(LIB):
            build()
        max_runtime = os.path.join(sys.prefix, "lib", "libmax.so")
        if os.path.exists(max_runtime):
            _max_runtime = ctypes.CDLL(
                max_runtime, mode=ctypes.RTLD_GLOBAL
            )
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def bytes_buffer(value: bytes) -> tuple[bytes, int]:
    address = int(_bytes_address(value))
    if not address:
        raise RuntimeError("CPython returned a null bytes buffer")
    return value, address


def encode_keys(keys) -> tuple[list, bytes, np.ndarray, int]:
    values = list(keys)
    encoded = [key_bytes(key) for key in values]
    offsets = np.empty(len(encoded) + 1, dtype=np.int64)
    offsets[0] = 0
    total = 0
    for index, item in enumerate(encoded, 1):
        total += len(item)
        if total > _I64_MAX:
            raise OverflowError("encoded key data exceeds the Mojo ABI limit")
        offsets[index] = total
    storage, address = bytes_buffer(b"".join(encoded))
    return values, storage, offsets, address


def key_bytes(key) -> bytes:
    if isinstance(key, str):
        return key.encode("utf-8")
    return str(key).encode("utf-8")


def bitarray_address(bits) -> int:
    address = bits.buffer_info().address
    if not address:
        raise RuntimeError("bitarray returned a null buffer")
    return address


def array_address(array: np.ndarray) -> int:
    if not array.flags.c_contiguous:
        raise ValueError("FFI arrays must be C-contiguous")
    return int(array.ctypes.data)


def hash64(value: bytes, seed: int = 0) -> int:
    storage, address = bytes_buffer(value)
    return int(lib().mpbl_hash64(address, len(value), seed))
