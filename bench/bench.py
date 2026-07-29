"""Benchmarks against pybloom-live 4.0.0 on identical keys."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

from mojo_pybloom_live import BloomFilter, ScalableBloomFilter  # noqa: E402
from pybloom_live import BloomFilter as PythonBloomFilter  # noqa: E402
from pybloom_live import ScalableBloomFilter as PythonScalableBloomFilter  # noqa: E402


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def add_loop(filter_type, capacity, keys):
    bloom = filter_type(capacity)
    for key in keys:
        bloom.add(key)
    return bloom


def contains_loop(bloom, keys):
    return [key in bloom for key in keys]


def scalable_add_loop(filter_type, keys):
    bloom = filter_type()
    for key in keys:
        bloom.add(key)
    return bloom


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    BloomFilter(10).add("warm-up")
    cases = []

    insert_keys = list(range(250_000))
    cases.append(
        (
            "bulk insert 250k integers",
            lambda: BloomFilter(300_000).update(insert_keys),
            lambda: add_loop(PythonBloomFilter, 300_000, insert_keys),
        )
    )

    ours = BloomFilter(300_000)
    ours.update(insert_keys)
    upstream = add_loop(PythonBloomFilter, 300_000, insert_keys)
    cases.append(
        (
            "bulk lookup 250k present",
            lambda: ours.contains_many(insert_keys),
            lambda: contains_loop(upstream, insert_keys),
        )
    )

    absent_keys = list(range(1_000_000, 1_250_000))
    cases.append(
        (
            "bulk lookup 250k absent",
            lambda: ours.contains_many(absent_keys),
            lambda: contains_loop(upstream, absent_keys),
        )
    )

    scalar_keys = list(range(25_000))
    cases.append(
        (
            "scalar add 25k integers",
            lambda: add_loop(BloomFilter, 30_000, scalar_keys),
            lambda: add_loop(PythonBloomFilter, 30_000, scalar_keys),
        )
    )

    left = BloomFilter(5_000_000)
    right = BloomFilter(5_000_000)
    py_left = PythonBloomFilter(5_000_000)
    py_right = PythonBloomFilter(5_000_000)
    left.update(range(0, 250_000))
    right.update(range(250_000, 500_000))
    for key in range(0, 250_000):
        py_left.add(key)
    for key in range(250_000, 500_000):
        py_right.add(key)
    cases.append(
        (
            "union, capacity 5M",
            lambda: left | right,
            lambda: py_left | py_right,
        )
    )

    large_left = BloomFilter(75_000_000)
    large_right = BloomFilter(75_000_000)
    py_large_left = PythonBloomFilter(75_000_000)
    py_large_right = PythonBloomFilter(75_000_000)
    large_left.update(range(0, 250_000))
    large_right.update(range(250_000, 500_000))
    for key in range(0, 250_000):
        py_large_left.add(key)
    for key in range(250_000, 500_000):
        py_large_right.add(key)
    cases.append(
        (
            "parallel union, capacity 75M",
            lambda: large_left | large_right,
            lambda: py_large_left | py_large_right,
        )
    )

    scalable_keys = list(range(25_000))
    cases.append(
        (
            "scalable insert 25k",
            lambda: scalable_add_loop(ScalableBloomFilter, scalable_keys),
            lambda: scalable_add_loop(
                PythonScalableBloomFilter, scalable_keys
            ),
        )
    )

    print(f"Machine: {cpu_name()}; Python {platform.python_version()}")
    print()
    print("| case | Mojo port | pybloom-live | upstream / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, python_function in cases:
        mojo_function()
        python_function()
        mojo_time = best_time(mojo_function)
        python_time = best_time(python_function)
        ratio = python_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1e3:.1f} ms | "
            f"{python_time * 1e3:.1f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
