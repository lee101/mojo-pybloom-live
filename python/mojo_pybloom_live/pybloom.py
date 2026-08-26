"""BloomFilter and ScalableBloomFilter with Mojo hashing and bit kernels."""

from __future__ import annotations

import copy
import math
import sys
from struct import calcsize, error as StructError, pack, unpack

import numpy as np
import xxhash
from bitarray import bitarray

from ._lib import (
    array_address,
    bitarray_address,
    bytes_address,
    encode_keys,
    key_bytes,
    lib,
)

_SECOND_SEED = 0x9E3779B185EBCA87
_SET_PARALLEL_THRESHOLD = 1 << 27


def make_hashfuncs(num_slices, num_bits):
    """Return this port's XXH64 double-hash position generator."""

    def _hash_maker(key):
        value = key_bytes(key)
        first = xxhash.xxh64_intdigest(value)
        step = xxhash.xxh64_intdigest(value, seed=_SECOND_SEED) | 1
        for index in range(num_slices):
            yield ((first + index * step) & 0xFFFFFFFFFFFFFFFF) % num_bits

    return _hash_maker, xxhash.xxh64


class BloomFilter:
    FILE_FMT = b"<dQQQQ"

    def __init__(self, capacity, error_rate=0.001):
        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")
        if not capacity > 0:
            raise ValueError("Capacity must be > 0")
        num_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))
        bits_per_slice = int(
            math.ceil(
                (capacity * abs(math.log(error_rate)))
                / (num_slices * (math.log(2) ** 2))
            )
        )
        self._setup(error_rate, num_slices, bits_per_slice, capacity, 0)
        self._set_bitarray(bitarray(self.num_bits, endian="little"))
        self.bitarray.setall(False)

    def _setup(
        self, error_rate, num_slices, bits_per_slice, capacity, count
    ) -> None:
        if not math.isfinite(error_rate) or not 0 < error_rate < 1:
            raise ValueError("invalid BloomFilter error rate")
        if (
            not isinstance(num_slices, int)
            or not isinstance(bits_per_slice, int)
            or not isinstance(capacity, int)
            or not isinstance(count, int)
            or num_slices <= 0
            or bits_per_slice <= 0
            or capacity <= 0
            or count < 0
        ):
            raise ValueError("invalid BloomFilter metadata")
        if num_slices > sys.maxsize // bits_per_slice:
            raise ValueError("BloomFilter size exceeds platform limits")
        self.error_rate = error_rate
        self.num_slices = num_slices
        self.bits_per_slice = bits_per_slice
        self.capacity = capacity
        self.num_bits = num_slices * bits_per_slice
        self.count = count
        self.make_hashes, self.hashfn = make_hashfuncs(
            self.num_slices, self.bits_per_slice
        )

    def _set_bitarray(self, bits) -> None:
        self.bitarray = bits
        self._bits_address = bitarray_address(bits)

    def __contains__(self, key) -> bool:
        value = key_bytes(key)
        found = bool(
            lib().mpbl_contains(
                self._bits_address,
                bytes_address(value),
                len(value),
                self.num_slices,
                self.bits_per_slice,
            )
        )
        return found

    def __len__(self) -> int:
        return self.count

    def add(self, key, skip_check=False) -> bool:
        if self.count > self.capacity:
            raise IndexError("BloomFilter is at capacity")
        value = key_bytes(key)
        found = bool(
            lib().mpbl_add(
                self._bits_address,
                bytes_address(value),
                len(value),
                self.num_slices,
                self.bits_per_slice,
                skip_check,
            )
        )
        if skip_check or not found:
            self.count += 1
            return False
        return True

    def contains_many(self, keys) -> np.ndarray:
        """Return a boolean array of membership results in one FFI call."""
        values, storage, offsets, keys_address = encode_keys(keys)
        result = np.empty(len(values), dtype=np.uint8)
        if values:
            lib().mpbl_contains_many(
                self._bits_address,
                keys_address,
                int(offsets.ctypes.data),
                len(values),
                self.num_slices,
                self.bits_per_slice,
                int(result.ctypes.data),
            )
        del storage
        return result.astype(bool)

    def update(self, keys) -> int:
        """Add an iterable in one FFI call and return the number newly added."""
        values, storage, offsets, keys_address = encode_keys(keys)
        if len(values) > self.capacity + 1 - self.count:
            added = sum(not self.add(key) for key in values)
            return added
        result = np.empty(len(values), dtype=np.uint8)
        if values:
            lib().mpbl_add_many(
                self._bits_address,
                keys_address,
                int(offsets.ctypes.data),
                len(values),
                self.num_slices,
                self.bits_per_slice,
                int(result.ctypes.data),
            )
        del storage
        added = int(np.count_nonzero(result == 0))
        self.count += added
        return added

    def copy(self):
        new_filter = self.__class__.__new__(self.__class__)
        new_filter._setup(
            self.error_rate,
            self.num_slices,
            self.bits_per_slice,
            self.capacity,
            0,
        )
        new_filter._set_bitarray(self.bitarray.copy())
        return new_filter

    def _empty_like(self):
        new_filter = self.__class__.__new__(self.__class__)
        new_filter._setup(
            self.error_rate,
            self.num_slices,
            self.bits_per_slice,
            self.capacity,
            0,
        )
        new_filter._set_bitarray(bitarray(self.num_bits, endian="little"))
        return new_filter

    def _compatible(self, other, operation: str) -> None:
        if (
            not isinstance(other, BloomFilter)
            or self.capacity != other.capacity
            or self.error_rate != other.error_rate
        ):
            raise ValueError(
                f"{operation} filters requires both filters to have both "
                "the same capacity and error rate"
            )

    def union(self, other):
        self._compatible(other, "Unioning")
        new_bloom = self._empty_like()
        lib().mpbl_or(
            new_bloom._bits_address,
            self._bits_address,
            other._bits_address,
            self.bitarray.nbytes,
            _SET_PARALLEL_THRESHOLD,
        )
        return new_bloom

    def __or__(self, other):
        return self.union(other)

    def intersection(self, other):
        self._compatible(other, "Intersecting")
        new_bloom = self._empty_like()
        lib().mpbl_and(
            new_bloom._bits_address,
            self._bits_address,
            other._bits_address,
            self.bitarray.nbytes,
            _SET_PARALLEL_THRESHOLD,
        )
        return new_bloom

    def __and__(self, other):
        return self.intersection(other)

    def tofile(self, f) -> None:
        f.write(
            pack(
                self.FILE_FMT,
                self.error_rate,
                self.num_slices,
                self.bits_per_slice,
                self.capacity,
                self.count,
            )
        )
        f.write(self.bitarray.tobytes())

    @classmethod
    def fromfile(cls, f, n=-1):
        headerlen = calcsize(cls.FILE_FMT)
        if 0 < n < headerlen:
            raise ValueError("n too small!")
        header = f.read(headerlen)
        if len(header) != headerlen:
            raise ValueError("Bit length mismatch!")
        new_filter = cls(1)
        try:
            new_filter._setup(*unpack(cls.FILE_FMT, header))
        except (OverflowError, StructError, TypeError, ValueError) as error:
            raise ValueError("Invalid BloomFilter header!") from error
        expected_bytes = (new_filter.num_bits + 7) // 8
        if n > 0 and n - headerlen != expected_bytes:
            raise ValueError("Bit length mismatch!")
        payload = f.read(expected_bytes) if n > 0 else f.read()
        if len(payload) != expected_bytes:
            raise ValueError("Bit length mismatch!")
        loaded = bitarray(endian="little")
        loaded.frombytes(payload)
        new_filter._set_bitarray(loaded)
        return new_filter

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("make_hashes", None)
        state.pop("_bits_address", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.make_hashes, self.hashfn = make_hashfuncs(
            self.num_slices, self.bits_per_slice
        )
        self._bits_address = bitarray_address(self.bitarray)


class ScalableBloomFilter:
    SMALL_SET_GROWTH = 2
    LARGE_SET_GROWTH = 4
    FILE_FMT = "<idQd"

    def __init__(
        self, initial_capacity=100, error_rate=0.001, mode=LARGE_SET_GROWTH
    ):
        if not error_rate or error_rate < 0:
            raise ValueError("Error_Rate must be a decimal less than 0.")
        self._setup(mode, 0.9, initial_capacity, error_rate)
        self.filters = []
        self._kernel_metadata = None

    def _setup(self, mode, ratio, initial_capacity, error_rate) -> None:
        self.scale = mode
        self.ratio = ratio
        self.initial_capacity = initial_capacity
        self.error_rate = error_rate

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_kernel_metadata"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._refresh_kernel_metadata()

    def __contains__(self, key) -> bool:
        return any(key in bloom for bloom in reversed(self.filters))

    def _refresh_kernel_metadata(self) -> None:
        if np.dtype(np.uintp).itemsize != 8:
            raise RuntimeError("mojo-pybloom-live requires a 64-bit platform")
        bits = np.asarray(
            [bloom._bits_address for bloom in self.filters], dtype=np.uintp
        )
        slices = np.asarray(
            [bloom.num_slices for bloom in self.filters], dtype=np.int64
        )
        sizes = np.asarray(
            [bloom.bits_per_slice for bloom in self.filters], dtype=np.int64
        )
        self._kernel_metadata = (
            bits,
            slices,
            sizes,
            array_address(bits),
            array_address(slices),
            array_address(sizes),
        )

    def add(self, key) -> bool:
        value = key_bytes(key)
        address = bytes_address(value)
        if not self.filters:
            bloom = BloomFilter(
                capacity=self.initial_capacity,
                error_rate=self.error_rate * self.ratio,
            )
            self.filters.append(bloom)
            self._refresh_kernel_metadata()
        else:
            bloom = self.filters[-1]
            if bloom.count >= bloom.capacity:
                if self._kernel_metadata is None:
                    self._refresh_kernel_metadata()
                _, _, _, bits_address, slices_address, sizes_address = (
                    self._kernel_metadata
                )
                found = lib().mpbl_scalable_add(
                    bits_address,
                    slices_address,
                    sizes_address,
                    len(self.filters),
                    address,
                    len(value),
                    0,
                )
                if found == 1:
                    return True
                bloom = BloomFilter(
                    capacity=bloom.capacity * self.scale,
                    error_rate=bloom.error_rate * self.ratio,
                )
                self.filters.append(bloom)
                self._refresh_kernel_metadata()
        if self._kernel_metadata is None:
            self._refresh_kernel_metadata()
        _, _, _, bits_address, slices_address, sizes_address = (
            self._kernel_metadata
        )
        found = bool(
            lib().mpbl_scalable_add(
                bits_address,
                slices_address,
                sizes_address,
                len(self.filters),
                address,
                len(value),
                1,
            )
        )
        if not found:
            bloom.count += 1
        return found

    def update(self, keys) -> int:
        return sum(not self.add(key) for key in keys)

    def contains_many(self, keys) -> np.ndarray:
        values = list(keys)
        result = np.zeros(len(values), dtype=bool)
        for bloom in reversed(self.filters):
            missing = np.flatnonzero(~result)
            if not missing.size:
                break
            result[missing] = bloom.contains_many(values[index] for index in missing)
        return result

    def union(self, other):
        if (
            not isinstance(other, ScalableBloomFilter)
            or
            self.scale != other.scale
            or self.initial_capacity != other.initial_capacity
            or self.error_rate != other.error_rate
        ):
            raise ValueError(
                "Unioning two scalable bloom filters requires both filters "
                "to have both the same mode, initial capacity and error rate"
            )
        if len(self.filters) > len(other.filters):
            larger = copy.deepcopy(self)
            smaller = other
        else:
            larger = copy.deepcopy(other)
            smaller = self
        merged = [
            larger.filters[index] | smaller.filters[index]
            for index in range(len(smaller.filters))
        ]
        merged.extend(larger.filters[len(smaller.filters) :])
        larger.filters = merged
        larger._refresh_kernel_metadata()
        return larger

    def __or__(self, other):
        return self.union(other)

    @property
    def capacity(self) -> int:
        return sum(bloom.capacity for bloom in self.filters)

    @property
    def count(self) -> int:
        return len(self)

    def tofile(self, f) -> None:
        f.write(
            pack(
                self.FILE_FMT,
                self.scale,
                self.ratio,
                self.initial_capacity,
                self.error_rate,
            )
        )
        f.write(pack(b"<l", len(self.filters)))
        if self.filters:
            payloads = []
            for bloom in self.filters:
                from io import BytesIO

                stream = BytesIO()
                bloom.tofile(stream)
                payloads.append(stream.getvalue())
            f.write(pack(b"<" + b"Q" * len(payloads), *(map(len, payloads))))
            for payload in payloads:
                f.write(payload)

    @classmethod
    def fromfile(cls, f):
        new_filter = cls()
        headerlen = calcsize(cls.FILE_FMT)
        header = f.read(headerlen)
        if len(header) != headerlen:
            raise ValueError("Invalid ScalableBloomFilter header!")
        try:
            mode, ratio, initial_capacity, error_rate = unpack(
                cls.FILE_FMT, header
            )
        except StructError as error:
            raise ValueError("Invalid ScalableBloomFilter header!") from error
        if (
            mode not in (cls.SMALL_SET_GROWTH, cls.LARGE_SET_GROWTH)
            or not math.isfinite(ratio)
            or not 0 < ratio < 1
            or initial_capacity <= 0
            or not math.isfinite(error_rate)
            or not 0 < error_rate < 1
        ):
            raise ValueError("Invalid ScalableBloomFilter header!")
        new_filter._setup(mode, ratio, initial_capacity, error_rate)
        count_bytes = f.read(calcsize(b"<l"))
        if len(count_bytes) != calcsize(b"<l"):
            raise ValueError("Invalid ScalableBloomFilter header!")
        (count,) = unpack(b"<l", count_bytes)
        if count < 0:
            raise ValueError("Invalid ScalableBloomFilter filter count!")
        new_filter.filters = []
        if count:
            lengths_fmt = b"<" + b"Q" * count
            lengths_bytes = f.read(calcsize(lengths_fmt))
            if len(lengths_bytes) != calcsize(lengths_fmt):
                raise ValueError("Invalid ScalableBloomFilter lengths!")
            lengths = unpack(lengths_fmt, lengths_bytes)
            for length in lengths:
                new_filter.filters.append(BloomFilter.fromfile(f, length))
        new_filter._refresh_kernel_metadata()
        return new_filter

    def __len__(self) -> int:
        return sum(bloom.count for bloom in self.filters)
