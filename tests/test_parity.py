from __future__ import annotations

import copy
import io
import pickle
from struct import pack

import numpy as np
import pytest
import xxhash
from pybloom_live import BloomFilter as PythonBloomFilter
from pybloom_live import ScalableBloomFilter as PythonScalableBloomFilter

from mojo_pybloom_live import BloomFilter, ScalableBloomFilter
from mojo_pybloom_live._lib import hash64, lib
from mojo_pybloom_live.pybloom import make_hashfuncs


@pytest.mark.parametrize(
    "capacity,error_rate",
    [(1, 0.1), (100, 0.01), (10_000, 0.001), (1_000_000, 1e-6)],
)
def test_constructor_dimensions_match_upstream(capacity, error_rate):
    ours = BloomFilter(capacity, error_rate)
    upstream = PythonBloomFilter(capacity, error_rate)
    assert ours.num_slices == upstream.num_slices
    assert ours.bits_per_slice == upstream.bits_per_slice
    assert ours.num_bits == upstream.num_bits
    assert len(ours.bitarray) == len(upstream.bitarray)


@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity_matches_upstream(capacity):
    with pytest.raises(ValueError, match="Capacity"):
        BloomFilter(capacity)
    with pytest.raises(ValueError, match="Capacity"):
        PythonBloomFilter(capacity)


@pytest.mark.parametrize("error_rate", [0, 1, -0.1, 1.1])
def test_invalid_error_rate_matches_upstream(error_rate):
    with pytest.raises(ValueError, match="Error_Rate"):
        BloomFilter(10, error_rate)
    with pytest.raises(ValueError, match="Error_Rate"):
        PythonBloomFilter(10, error_rate)


def test_add_duplicate_membership_and_length_match_upstream():
    ours = BloomFilter(10_000)
    upstream = PythonBloomFilter(10_000)
    values = [f"key-{index}" for index in range(500)]
    assert [ours.add(value) for value in values] == [
        upstream.add(value) for value in values
    ]
    assert [ours.add(value) for value in values] == [
        upstream.add(value) for value in values
    ]
    assert len(ours) == len(upstream) == len(values)
    assert all(value in ours and value in upstream for value in values)
    assert all(
        (f"absent-{index}" in ours) == (f"absent-{index}" in upstream) == False
        for index in range(100)
    )


def test_key_conversion_matches_upstream_contract():
    values = ["hello", b"hello", 42, 3.25, ("a", 1), None]
    ours = BloomFilter(100)
    upstream = PythonBloomFilter(100)
    for value in values:
        assert ours.add(value) == upstream.add(value)
    for value in values:
        assert value in ours
        assert value in upstream


def test_skip_check_and_capacity_boundary_match_upstream():
    ours = BloomFilter(2)
    upstream = PythonBloomFilter(2)
    for value in range(3):
        assert ours.add(value, skip_check=True) is False
        assert upstream.add(value, skip_check=True) is False
    assert len(ours) == len(upstream) == 3
    with pytest.raises(IndexError, match="capacity"):
        ours.add("overflow")
    with pytest.raises(IndexError, match="capacity"):
        upstream.add("overflow")


def test_copy_preserves_upstream_membership_and_count_semantics():
    ours = BloomFilter(100)
    upstream = PythonBloomFilter(100)
    for value in range(25):
        ours.add(value)
        upstream.add(value)
    ours_copy = ours.copy()
    upstream_copy = upstream.copy()
    assert len(ours_copy) == len(upstream_copy) == 0
    assert all(value in ours_copy and value in upstream_copy for value in range(25))


def test_union_and_intersection_match_upstream_membership():
    ours_a, ours_b = BloomFilter(1000), BloomFilter(1000)
    py_a, py_b = PythonBloomFilter(1000), PythonBloomFilter(1000)
    for value in range(0, 150):
        ours_a.add(value)
        py_a.add(value)
    for value in range(75, 225):
        ours_b.add(value)
        py_b.add(value)
    ours_union, py_union = ours_a | ours_b, py_a | py_b
    ours_intersection, py_intersection = ours_a & ours_b, py_a & py_b
    assert all(value in ours_union and value in py_union for value in range(225))
    assert all(
        value in ours_intersection and value in py_intersection
        for value in range(75, 150)
    )
    assert all(
        value not in ours_intersection and value not in py_intersection
        for value in list(range(0, 75)) + list(range(150, 225))
    )


@pytest.mark.parametrize("capacity", [101, 600_000])
@pytest.mark.parametrize(
    "operation,numpy_operation",
    [("union", np.bitwise_or), ("intersection", np.bitwise_and)],
)
def test_set_kernels_handle_simd_tails(capacity, operation, numpy_operation):
    left, right = BloomFilter(capacity), BloomFilter(capacity)
    left_bytes = np.frombuffer(left.bitarray, dtype=np.uint8)
    right_bytes = np.frombuffer(right.bitarray, dtype=np.uint8)
    pattern = np.arange(left_bytes.size, dtype=np.uint64)
    left_bytes[:] = ((pattern * 17 + 3) & 0xFF).astype(np.uint8)
    right_bytes[:] = ((pattern * 29 + 11) & 0xFF).astype(np.uint8)
    expected = numpy_operation(left_bytes, right_bytes)
    assert left.bitarray.nbytes % 32
    combined = getattr(left, operation)(right)
    assert np.array_equal(
        np.frombuffer(combined.bitarray, dtype=np.uint8), expected
    )


@pytest.mark.parametrize(
    "kernel,numpy_operation",
    [("mpbl_or", np.bitwise_or), ("mpbl_and", np.bitwise_and)],
)
def test_set_kernels_parallelize_only_at_threshold(kernel, numpy_operation):
    threshold = 1 << 22
    for nbytes in (threshold - 3, threshold + 17):
        indices = np.arange(nbytes, dtype=np.uint64)
        left = ((indices * 17 + 3) & 0xFF).astype(np.uint8)
        right = ((indices * 29 + 11) & 0xFF).astype(np.uint8)
        result = np.empty(nbytes, dtype=np.uint8)
        getattr(lib(), kernel)(
            int(result.ctypes.data),
            int(left.ctypes.data),
            int(right.ctypes.data),
            nbytes,
            threshold,
        )
        assert np.array_equal(result, numpy_operation(left, right))


@pytest.mark.parametrize("operation", ["union", "intersection"])
def test_incompatible_set_operation_matches_upstream(operation):
    ours_a, ours_b = BloomFilter(100), BloomFilter(101)
    py_a, py_b = PythonBloomFilter(100), PythonBloomFilter(101)
    with pytest.raises(ValueError):
        getattr(ours_a, operation)(ours_b)
    with pytest.raises(ValueError):
        getattr(py_a, operation)(py_b)


def test_bloom_file_roundtrip_and_size_match_upstream():
    ours = BloomFilter(1234, 0.01)
    upstream = PythonBloomFilter(1234, 0.01)
    ours.update(range(400))
    for value in range(400):
        upstream.add(value)
    ours_stream, py_stream = io.BytesIO(), io.BytesIO()
    ours.tofile(ours_stream)
    upstream.tofile(py_stream)
    assert len(ours_stream.getvalue()) == len(py_stream.getvalue())
    restored = BloomFilter.fromfile(io.BytesIO(ours_stream.getvalue()))
    assert restored.count == ours.count
    assert restored.capacity == ours.capacity
    assert restored.error_rate == ours.error_rate
    assert restored.bitarray.tobytes() == ours.bitarray.tobytes()
    assert restored.contains_many(range(400)).all()


def test_bloom_fromfile_n_and_bad_lengths_match_upstream():
    with pytest.raises(ValueError, match="n too small"):
        BloomFilter.fromfile(io.BytesIO(b"\0" * 10), n=10)
    with pytest.raises(ValueError, match="n too small"):
        PythonBloomFilter.fromfile(io.BytesIO(b"\0" * 10), n=10)
    bloom = BloomFilter(100)
    stream = io.BytesIO()
    bloom.tofile(stream)
    with pytest.raises(ValueError, match="Bit length mismatch"):
        BloomFilter.fromfile(io.BytesIO(stream.getvalue()[:-2]))


def test_pickle_roundtrip():
    bloom = BloomFilter(1000)
    bloom.update(range(100))
    restored = pickle.loads(pickle.dumps(bloom))
    assert restored.count == bloom.count
    assert restored.bitarray == bloom.bitarray
    assert restored.contains_many(range(100)).all()


def test_bulk_update_matches_scalar_exactly():
    values = [f"value-{index % 800}" for index in range(2000)]
    bulk, scalar = BloomFilter(5000), BloomFilter(5000)
    added = bulk.update(values)
    scalar_results = [scalar.add(value) for value in values]
    assert added == sum(not found for found in scalar_results)
    assert bulk.count == scalar.count
    assert bulk.bitarray == scalar.bitarray
    queries = values[::11] + [f"missing-{index}" for index in range(100)]
    assert np.array_equal(
        bulk.contains_many(queries), np.array([key in scalar for key in queries])
    )


def test_empty_keys_and_empty_batches_are_safe():
    bloom = BloomFilter(100)
    assert bloom.update([]) == 0
    assert bloom.contains_many([]).dtype == np.bool_
    assert bloom.contains_many([]).size == 0
    assert bloom.add("") is False
    assert "" in bloom
    assert bloom.contains_many(["", b"", "missing"]).tolist() == [
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    "header",
    [
        pack(BloomFilter.FILE_FMT, 0.01, 0, 10, 100, 0),
        pack(BloomFilter.FILE_FMT, 0.01, 10, 0, 100, 0),
        pack(BloomFilter.FILE_FMT, float("nan"), 10, 10, 100, 0),
        pack(BloomFilter.FILE_FMT, 0.01, 1 << 63, 3, 100, 0),
    ],
)
def test_bloom_fromfile_rejects_unsafe_metadata(header):
    with pytest.raises(ValueError):
        BloomFilter.fromfile(io.BytesIO(header))


def test_scalable_fromfile_rejects_truncated_and_invalid_metadata():
    with pytest.raises(ValueError):
        ScalableBloomFilter.fromfile(io.BytesIO(b""))
    header = pack(
        ScalableBloomFilter.FILE_FMT,
        ScalableBloomFilter.LARGE_SET_GROWTH,
        0.9,
        100,
        0.001,
    )
    with pytest.raises(ValueError):
        ScalableBloomFilter.fromfile(io.BytesIO(header))
    with pytest.raises(ValueError):
        ScalableBloomFilter.fromfile(io.BytesIO(header + pack(b"<l", -1)))


def test_xxh64_kernel_matches_reference_vectors():
    for value in [b"", b"a", b"hello", bytes(range(256)), b"x" * 10_000]:
        for seed in [0, 1, 0x9E3779B185EBCA87]:
            assert hash64(value, seed) == xxhash.xxh64_intdigest(value, seed=seed)


def test_hash_positions_correspond_to_bits_set_by_mojo():
    bloom = BloomFilter(1000)
    value = "position-check"
    bloom.add(value)
    make_hashes, hashfn = make_hashfuncs(
        bloom.num_slices, bloom.bits_per_slice
    )
    assert hashfn is xxhash.xxh64
    for slice_index, position in enumerate(make_hashes(value)):
        assert bloom.bitarray[
            slice_index * bloom.bits_per_slice + position
        ]


def test_false_positive_rate_is_within_requested_bound():
    bloom = BloomFilter(20_000, error_rate=0.01)
    bloom.update(range(20_000))
    false_positives = bloom.contains_many(range(100_000, 120_000)).sum()
    assert false_positives < 350


def test_scalable_growth_matches_upstream():
    ours = ScalableBloomFilter(
        initial_capacity=10,
        error_rate=0.01,
        mode=ScalableBloomFilter.SMALL_SET_GROWTH,
    )
    upstream = PythonScalableBloomFilter(
        initial_capacity=10,
        error_rate=0.01,
        mode=PythonScalableBloomFilter.SMALL_SET_GROWTH,
    )
    inserted = []
    candidate = 0
    while len(inserted) < 75:
        if candidate not in ours and candidate not in upstream:
            assert ours.add(candidate) is False
            assert upstream.add(candidate) is False
            inserted.append(candidate)
        candidate += 1
    assert len(ours) == len(upstream) == 75
    assert ours.capacity == upstream.capacity
    assert [item.capacity for item in ours.filters] == [
        item.capacity for item in upstream.filters
    ]
    assert [item.error_rate for item in ours.filters] == pytest.approx(
        [item.error_rate for item in upstream.filters]
    )
    assert ours.contains_many(inserted).all()


@pytest.mark.parametrize(
    "mode",
    [ScalableBloomFilter.SMALL_SET_GROWTH, ScalableBloomFilter.LARGE_SET_GROWTH],
)
def test_both_scalable_growth_modes(mode):
    bloom = ScalableBloomFilter(initial_capacity=2, mode=mode)
    bloom.update(range(20))
    assert len(bloom.filters) >= 2
    assert all(value in bloom for value in range(20))
    assert all(
        current.capacity == previous.capacity * mode
        for previous, current in zip(bloom.filters, bloom.filters[1:])
    )


def test_scalable_duplicates_and_properties_match_upstream():
    ours, upstream = ScalableBloomFilter(), PythonScalableBloomFilter()
    for value in range(50):
        ours.add(value)
        upstream.add(value)
    assert [ours.add(value) for value in range(50)] == [
        upstream.add(value) for value in range(50)
    ]
    assert ours.count == upstream.count
    assert ours.capacity == upstream.capacity


def test_scalable_duplicate_in_older_filter_does_not_trigger_growth():
    bloom = ScalableBloomFilter(initial_capacity=2)
    assert bloom.add("first") is False
    assert bloom.add("second") is False
    assert bloom.add("third") is False
    assert len(bloom.filters) == 2
    counts = [item.count for item in bloom.filters]
    assert bloom.add("first") is True
    assert len(bloom.filters) == 2
    assert [item.count for item in bloom.filters] == counts


def test_scalable_deepcopy_rebuilds_kernel_addresses():
    original = ScalableBloomFilter(initial_capacity=2)
    original.update(range(10))
    original_bits = [item.bitarray.copy() for item in original.filters]
    cloned = copy.deepcopy(original)
    assert cloned.add("clone-only") is False
    assert [item.bitarray for item in original.filters] == original_bits
    assert "clone-only" in cloned


def test_scalable_union_matches_upstream_membership():
    ours_a, ours_b = ScalableBloomFilter(10), ScalableBloomFilter(10)
    py_a = PythonScalableBloomFilter(10)
    py_b = PythonScalableBloomFilter(10)
    for value in range(100):
        (ours_a if value % 2 else ours_b).add(value)
        (py_a if value % 2 else py_b).add(value)
    ours_union, py_union = ours_a | ours_b, py_a | py_b
    assert all(value in ours_union and value in py_union for value in range(100))


def test_scalable_pickle_roundtrip():
    bloom = ScalableBloomFilter(initial_capacity=2)
    bloom.update(range(20))
    restored = pickle.loads(pickle.dumps(bloom))
    assert restored.count == bloom.count
    assert restored.capacity == bloom.capacity
    assert restored.contains_many(range(20)).all()
    assert restored.add("after-pickle") is False


def test_scalable_file_roundtrip_and_size_match_upstream():
    ours, upstream = ScalableBloomFilter(10), PythonScalableBloomFilter(10)
    for value in range(100):
        ours.add(value)
        upstream.add(value)
    ours_stream, py_stream = io.BytesIO(), io.BytesIO()
    ours.tofile(ours_stream)
    upstream.tofile(py_stream)
    assert len(ours_stream.getvalue()) == len(py_stream.getvalue())
    restored = ScalableBloomFilter.fromfile(
        io.BytesIO(ours_stream.getvalue())
    )
    assert restored.count == ours.count
    assert restored.capacity == ours.capacity
    assert restored.contains_many(range(100)).all()
