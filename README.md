# mojo-pybloom-live

Bloom filters and scalable Bloom filters with their hashing and bit operations
implemented in [Mojo](https://www.modular.com/mojo).

The Python API follows `pybloom-live` 4.0.0: `BloomFilter` and
`ScalableBloomFilter` keep the same constructor and method signatures for the
covered surface. The port adds `update()` and `contains_many()` so batches cross
the FFI once; those are the paths where compiled hashing makes the largest
difference.

## Install

The repository pins the Mojo nightly used to build and test it.

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-pybloom-live.so`. All tests and benchmarks run
inside the same environment:

```bash
pixi run test
pixi run bench
```

## Usage

```python
from mojo_pybloom_live import BloomFilter, ScalableBloomFilter

bloom = BloomFilter(capacity=100_000, error_rate=0.001)
bloom.update(f"user:{number}" for number in range(50_000))

assert "user:1234" in bloom
assert bloom.contains_many(["user:1", "missing"]).tolist() == [True, False]

growing = ScalableBloomFilter(initial_capacity=1_000, error_rate=0.001)
growing.add("event-1")
assert "event-1" in growing
```

This example is exercised by the same built library used by the test suite.

## Coverage

`BloomFilter` covers construction, `add(key, skip_check=False)`, membership,
`len()`, `copy()`, union (`union` and `|`), intersection (`intersection` and
`&`), file serialization, and pickle state.

`ScalableBloomFilter` covers both growth modes, automatic filter growth,
`add`, membership, `len`, `count`, `capacity`, union, and file serialization.
Both classes additionally provide bulk `update` and `contains_many` methods.

The constructor sizing formulas, capacity boundary, duplicate return values,
scalable growth parameters, serialized header shape, and probabilistic error
bound are parity-tested against the real `pybloom-live==4.0.0` package.

This port does not reproduce upstream's internal selection among XXH3-128 and
SHA-family hashes. It uses two seeded XXH64 hashes and standard double hashing,
all inside Mojo. Consequently, a file written here round-trips here and has the
same compact header and size as upstream, but its membership bit pattern is not
interchangeable with a `pybloom-live` file. There is also no accelerated
cross-filter bulk operation for `ScalableBloomFilter`; its scalar growth logic
remains in Python.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Python
3.13.14. Times are the best of three runs and include Python key conversion,
allocation, and FFI. A ratio above 1 means this port is faster.

| case | Mojo port | pybloom-live | upstream / Mojo |
| --- | ---: | ---: | ---: |
| bulk insert 250k integers | 185.5 ms | 1895.0 ms | 10.21x |
| bulk lookup 250k present | 181.4 ms | 1218.4 ms | 6.72x |
| bulk lookup 250k absent | 166.8 ms | 603.4 ms | 3.62x |
| scalar add 25k integers | 87.6 ms | 104.5 ms | 1.19x |
| union, capacity 5M | 3.2 ms | 5.5 ms | 1.74x |
| parallel union, capacity 75M | 133.3 ms | 504.4 ms | 3.78x |
| scalable insert 25k | 279.0 ms | 529.6 ms | 1.90x |

Bulk calls win by amortizing ctypes overhead and avoiding Python's per-slice
hash-position loop. Scalar calls pass their encoded Python bytes to Mojo without
an intermediate ctypes copy. Union and intersection use SIMD with a scalar tail;
large buffers are split into independent tasks, while smaller buffers avoid
parallel launch overhead. Scalable insertion hashes each key once in one FFI
call and reuses those hashes across every filter level.

There is no GPU path. Set operations perform one bitwise operation for every
three bytes moved, and hashing is followed by sparse, irregular bit probes.
Neither workload has enough sustained arithmetic intensity to offset device
transfer and launch costs, so CPU remains the only execution device.

## How it works

Python converts keys with the same rule as upstream: strings become UTF-8 and
other objects use `str(key).encode("utf-8")`. A scalar call passes the existing
Python bytes address through `ctypes` without copying; a bulk call packs key
bytes contiguously beside an `int64` offset array. C ABI buffers cross as integer
addresses, and the Mojo exports rebuild
`UnsafePointer[..., AnyOrigin[mut=True]]` values internally.

Mojo computes two XXH64 values and derives one position per slice using double
hashing. Storage is a little-endian `bitarray`, one bit per position, divided
into the same independent slices and with the same size formula as upstream.
Mojo writes directly into the `bitarray` buffer, so no FFI bitset copy is
needed. Union and intersection stream across the underlying bytes with
unaligned-safe SIMD loads and stores.

All kernels and C exports live in one compilation unit,
`src/bloom.mojo`, to pay the fixed Mojo build cost once.

MIT.
