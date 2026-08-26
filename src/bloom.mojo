"""XXH64 hashing and Bloom-filter bit operations exposed through a C ABI."""

from max.algorithm import parallelize
from std.sys.info import simd_width_of

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]

comptime P64_1: UInt64 = 0x9E3779B185EBCA87
comptime P64_2: UInt64 = 0xC2B2AE3D27D4EB4F
comptime P64_3: UInt64 = 0x165667B19E3779F9
comptime P64_4: UInt64 = 0x85EBCA77C2B2AE63
comptime P64_5: UInt64 = 0x27D4EB2F165667C5
comptime SECOND_SEED: UInt64 = 0x9E3779B185EBCA87


def read32(p: BPtr, offset: Int) -> UInt32:
    return (
        p.unsafe_offset(offset)
        .unsafe_bitcast[UInt32]()
        .unsafe_load[alignment=1]()
    )


def read64(p: BPtr, offset: Int) -> UInt64:
    return (
        p.unsafe_offset(offset)
        .unsafe_bitcast[UInt64]()
        .unsafe_load[alignment=1]()
    )


def rotl64(value: UInt64, amount: UInt64) -> UInt64:
    return (value << amount) | (value >> (UInt64(64) - amount))


def xxh64_round(acc: UInt64, lane: UInt64) -> UInt64:
    return rotl64(acc + lane * P64_2, 31) * P64_1


def xxh64_merge(acc: UInt64, lane: UInt64) -> UInt64:
    return (acc ^ xxh64_round(0, lane)) * P64_1 + P64_4


def xxh64(p: BPtr, n: Int, seed: UInt64) -> UInt64:
    var i = 0
    var h: UInt64
    if n >= 32:
        var v1 = seed + P64_1 + P64_2
        var v2 = seed + P64_2
        var v3 = seed
        var v4 = seed - P64_1
        while i <= n - 32:
            v1 = xxh64_round(v1, read64(p, i))
            v2 = xxh64_round(v2, read64(p, i + 8))
            v3 = xxh64_round(v3, read64(p, i + 16))
            v4 = xxh64_round(v4, read64(p, i + 24))
            i += 32
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18)
        h = xxh64_merge(h, v1)
        h = xxh64_merge(h, v2)
        h = xxh64_merge(h, v3)
        h = xxh64_merge(h, v4)
    else:
        h = seed + P64_5
    h += UInt64(n)
    while i <= n - 8:
        h ^= xxh64_round(0, read64(p, i))
        h = rotl64(h, 27) * P64_1 + P64_4
        i += 8
    if i <= n - 4:
        h ^= UInt64(read32(p, i)) * P64_1
        h = rotl64(h, 23) * P64_2 + P64_3
        i += 4
    while i < n:
        h ^= UInt64(p[unsafe_offset=i]) * P64_5
        h = rotl64(h, 11) * P64_1
        i += 1
    h ^= h >> 33
    h *= P64_2
    h ^= h >> 29
    h *= P64_3
    h ^= h >> 32
    return h


def has_key(
    bits: BPtr, key: BPtr, key_len: Int, num_slices: Int, bits_per_slice: Int
) -> Bool:
    var h1 = xxh64(key, key_len, 0)
    var h2 = xxh64(key, key_len, SECOND_SEED) | UInt64(1)
    return has_hashes(bits, h1, h2, num_slices, bits_per_slice)


def has_hashes(
    bits: BPtr,
    h1: UInt64,
    h2: UInt64,
    num_slices: Int,
    bits_per_slice: Int,
) -> Bool:
    for i in range(num_slices):
        var within = Int((h1 + UInt64(i) * h2) % UInt64(bits_per_slice))
        var bit_index = i * bits_per_slice + within
        var mask = UInt8(1 << (bit_index & 7))
        if (bits[unsafe_offset=bit_index >> 3] & mask) == 0:
            return False
    return True


def add_key(
    bits: BPtr,
    key: BPtr,
    key_len: Int,
    num_slices: Int,
    bits_per_slice: Int,
    skip_check: Bool,
) -> Bool:
    var h1 = xxh64(key, key_len, 0)
    var h2 = xxh64(key, key_len, SECOND_SEED) | UInt64(1)
    return add_hashes(bits, h1, h2, num_slices, bits_per_slice, skip_check)


def add_hashes(
    bits: BPtr,
    h1: UInt64,
    h2: UInt64,
    num_slices: Int,
    bits_per_slice: Int,
    skip_check: Bool,
) -> Bool:
    var found_all = True
    for i in range(num_slices):
        var within = Int((h1 + UInt64(i) * h2) % UInt64(bits_per_slice))
        var bit_index = i * bits_per_slice + within
        var byte_index = bit_index >> 3
        var mask = UInt8(1 << (bit_index & 7))
        if not skip_check and (bits[unsafe_offset=byte_index] & mask) == 0:
            found_all = False
        bits[unsafe_offset=byte_index] |= mask
    return found_all


def combine_range[
    is_union: Bool
](dst: BPtr, a: BPtr, b: BPtr, start: Int, end: Int):
    comptime W = simd_width_of[DType.float64]()
    comptime BYTES_PER_VECTOR = W * 8
    var i = start
    var vector_end = end - ((end - start) % BYTES_PER_VECTOR)
    var dst64 = dst.unsafe_offset(start).unsafe_bitcast[UInt64]()
    var a64 = a.unsafe_offset(start).unsafe_bitcast[UInt64]()
    var b64 = b.unsafe_offset(start).unsafe_bitcast[UInt64]()
    var word = 0
    while i < vector_end:
        var av = a64.unsafe_load[width=W, alignment=1](word)
        var bv = b64.unsafe_load[width=W, alignment=1](word)
        comptime if is_union:
            dst64.unsafe_store[alignment=1](word, av | bv)
        else:
            dst64.unsafe_store[alignment=1](word, av & bv)
        word += W
        i += BYTES_PER_VECTOR
    while i < end:
        comptime if is_union:
            dst[unsafe_offset=i] = a[unsafe_offset=i] | b[unsafe_offset=i]
        else:
            dst[unsafe_offset=i] = a[unsafe_offset=i] & b[unsafe_offset=i]
        i += 1


def combine[
    is_union: Bool
](
    dst_addr: Int,
    a_addr: Int,
    b_addr: Int,
    nbytes: Int,
    parallel_threshold: Int,
):
    comptime CHUNK_BYTES = 1 << 22
    if nbytes < parallel_threshold:
        combine_range[is_union](
            BPtr(unsafe_from_address=dst_addr),
            BPtr(unsafe_from_address=a_addr),
            BPtr(unsafe_from_address=b_addr),
            0,
            nbytes,
        )
        return
    var chunks = (nbytes + CHUNK_BYTES - 1) // CHUNK_BYTES

    @__copy_capture(dst_addr, a_addr, b_addr, nbytes)
    @__parameter
    def work(chunk: Int):
        var start = chunk * CHUNK_BYTES
        var end = min(start + CHUNK_BYTES, nbytes)
        combine_range[is_union](
            BPtr(unsafe_from_address=dst_addr),
            BPtr(unsafe_from_address=a_addr),
            BPtr(unsafe_from_address=b_addr),
            start,
            end,
        )

    parallelize[work](chunks, min(chunks, 8))


@export("mpbl_hash64")
def mpbl_hash64(key_addr: Int, key_len: Int, seed: UInt64) abi("C") -> UInt64:
    return xxh64(BPtr(unsafe_from_address=key_addr), key_len, seed)


@export("mpbl_contains")
def mpbl_contains(
    bits_addr: Int,
    key_addr: Int,
    key_len: Int,
    num_slices: Int,
    bits_per_slice: Int,
) abi("C") -> Int:
    return 1 if has_key(
        BPtr(unsafe_from_address=bits_addr),
        BPtr(unsafe_from_address=key_addr),
        key_len,
        num_slices,
        bits_per_slice,
    ) else 0


@export("mpbl_add")
def mpbl_add(
    bits_addr: Int,
    key_addr: Int,
    key_len: Int,
    num_slices: Int,
    bits_per_slice: Int,
    skip_check: Int,
) abi("C") -> Int:
    return 1 if add_key(
        BPtr(unsafe_from_address=bits_addr),
        BPtr(unsafe_from_address=key_addr),
        key_len,
        num_slices,
        bits_per_slice,
        skip_check != 0,
    ) else 0


@export("mpbl_contains_many")
def mpbl_contains_many(
    bits_addr: Int,
    keys_addr: Int,
    offsets_addr: Int,
    count: Int,
    num_slices: Int,
    bits_per_slice: Int,
    result_addr: Int,
) abi("C"):
    var bits = BPtr(unsafe_from_address=bits_addr)
    var keys = BPtr(unsafe_from_address=keys_addr)
    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    var result = BPtr(unsafe_from_address=result_addr)
    for i in range(count):
        var start = Int(offsets[unsafe_offset=i])
        var key_len = Int(offsets[unsafe_offset=i + 1]) - start
        result[unsafe_offset=i] = UInt8(
            1 if has_key(
                bits,
                keys.unsafe_offset(start),
                key_len,
                num_slices,
                bits_per_slice,
            ) else 0
        )


@export("mpbl_add_many")
def mpbl_add_many(
    bits_addr: Int,
    keys_addr: Int,
    offsets_addr: Int,
    count: Int,
    num_slices: Int,
    bits_per_slice: Int,
    result_addr: Int,
) abi("C"):
    var bits = BPtr(unsafe_from_address=bits_addr)
    var keys = BPtr(unsafe_from_address=keys_addr)
    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    var result = BPtr(unsafe_from_address=result_addr)
    for i in range(count):
        var start = Int(offsets[unsafe_offset=i])
        var key_len = Int(offsets[unsafe_offset=i + 1]) - start
        result[unsafe_offset=i] = UInt8(
            1 if add_key(
                bits,
                keys.unsafe_offset(start),
                key_len,
                num_slices,
                bits_per_slice,
                False,
            ) else 0
        )


@export("mpbl_scalable_add")
def mpbl_scalable_add(
    bits_addrs_addr: Int,
    num_slices_addr: Int,
    bits_per_slice_addr: Int,
    filter_count: Int,
    key_addr: Int,
    key_len: Int,
    add_to_last: Int,
) abi("C") -> Int:
    var bits_addrs = U64Ptr(unsafe_from_address=bits_addrs_addr)
    var num_slices = I64Ptr(unsafe_from_address=num_slices_addr)
    var bits_per_slice = I64Ptr(unsafe_from_address=bits_per_slice_addr)
    var key = BPtr(unsafe_from_address=key_addr)
    var h1 = xxh64(key, key_len, 0)
    var h2 = xxh64(key, key_len, SECOND_SEED) | UInt64(1)
    var last = filter_count - 1
    var first_to_check = last - 1 if add_to_last != 0 else last
    for offset in range(first_to_check + 1):
        var index = first_to_check - offset
        var bits = BPtr(
            unsafe_from_address=Int(bits_addrs[unsafe_offset=index])
        )
        if has_hashes(
            bits,
            h1,
            h2,
            Int(num_slices[unsafe_offset=index]),
            Int(bits_per_slice[unsafe_offset=index]),
        ):
            return 1
    if add_to_last == 0:
        return -1
    var last_bits = BPtr(
        unsafe_from_address=Int(bits_addrs[unsafe_offset=last])
    )
    return 1 if add_hashes(
        last_bits,
        h1,
        h2,
        Int(num_slices[unsafe_offset=last]),
        Int(bits_per_slice[unsafe_offset=last]),
        False,
    ) else 0


@export("mpbl_or")
def mpbl_or(
    dst_addr: Int,
    a_addr: Int,
    b_addr: Int,
    nbytes: Int,
    parallel_threshold: Int,
) abi("C"):
    combine[True](dst_addr, a_addr, b_addr, nbytes, parallel_threshold)


@export("mpbl_and")
def mpbl_and(
    dst_addr: Int,
    a_addr: Int,
    b_addr: Int,
    nbytes: Int,
    parallel_threshold: Int,
) abi("C"):
    combine[False](dst_addr, a_addr, b_addr, nbytes, parallel_threshold)
