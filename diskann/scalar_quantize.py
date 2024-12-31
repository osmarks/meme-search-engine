import numpy as np
import msgpack
import math

n_dims = 1152
n_buckets = n_dims
#n_buckets = n_dims // 2 # we now have one quant scale per pair of components
#pair_separation = 16 # for efficient dot product computation, we need to have the second element of a pair exactly chunk_size after the first
n_dims_per_bucket = n_dims // n_buckets
data = np.fromfile("embeddings.bin", dtype=np.float16).reshape(-1, n_dims).astype(np.float32) # sorry

CUTOFF = 1e-3 / 2

print("computing quantiles")
smin = np.quantile(data, CUTOFF, axis=0)
smax = np.quantile(data, 1 - CUTOFF, axis=0)

# naive O(n²) greedy algorithm
# probably overbuilt for the 2-components-per-bucket case but I'm not getting rid of it
def assign_buckets():
    import random
    intervals = list(enumerate(zip(smin, smax)))
    random.shuffle(intervals)
    buckets = [ [ intervals.pop() ] for _ in range(n_buckets) ]
    def bucket_cost(bucket):
        bmin = min(cmin for id, (cmin, cmax) in bucket)
        bmax = max(cmax for id, (cmin, cmax) in bucket)
        #print("MIN", bmin, "MAX", bmax)
        return sum(abs(cmin - bmin) + abs(cmax - bmax) for id, (cmin, cmax) in bucket)
    while len(intervals):
        for bucket in buckets:
            def new_interval_cost(interval):
                return bucket_cost(bucket + [interval[1]])
            i, interval = min(enumerate(intervals), key=new_interval_cost)
            bucket.append(intervals.pop(i))
    return buckets

ranges = smax - smin
# TODO: it is possible to do better assignment to buckets
#order = np.argsort(ranges)
print("bucket assignment")
order = np.arange(n_dims) # np.concatenate(np.stack([ [ id for id, (cmin, cmax) in bucket ] for bucket in assign_buckets() ]))

bucket_ranges = []
bucket_centres = []
bucket_absmax = []
bucket_gmins = []

for bucket_min in range(0, n_dims, n_dims_per_bucket):
    bucket_max = bucket_min + n_dims_per_bucket
    indices = order[bucket_min:bucket_max]
    gmin = float(np.min(smin[indices]))
    gmax = float(np.max(smax[indices]))
    bucket_range = gmax - gmin
    bucket_centre = (gmax + gmin) / 2
    bucket_gmins.append(gmin)
    bucket_ranges.append(bucket_range)
    bucket_centres.append(bucket_centre)
    bucket_absmax.append(max(abs(gmin), abs(gmax)))

print("determining scales")
scales = [] # multiply by float and convert to quantize
offsets = []
q_offsets = [] # int16 value to add at dot time
q_scales = [] # rescales channel up at dot time; must be proportional(ish) to square of scale factor but NOT cause overflow in accumulation or PLMULLW
scale_factor_bound = float("inf")
for bucket in range(n_buckets):
    step_size = bucket_ranges[bucket] / 255
    scales.append(1 / step_size)
    q_offset = int(bucket_gmins[bucket] / step_size)
    q_offsets.append(q_offset)
    nsfb = (2**31 - 1) / (n_dims_per_bucket * abs((255**2) + 2 * q_offset * 255 + q_offset ** 2)) / 2
    # we are bounded both by overflow in accumulation and PLMULLW (u8 plus offset times scale factor)
    scale_factor_bound = min(scale_factor_bound, nsfb, (2**15 - 1) // (q_offset + 255))
    offsets.append(bucket_gmins[bucket])

for bucket in range(n_buckets):
    sfb = scale_factor_bound / max(map(lambda x: x ** 2, bucket_ranges))
    sf = (bucket_ranges[bucket]) ** 2 * sfb
    q_scales.append(int(sf))

print(bucket_ranges, bucket_centres, bucket_absmax)
print(scales, offsets, q_offsets, q_scales)

"""
interleave = np.concatenate([
    np.arange(0, n_dims, n_dims_per_bucket) + a
    for a in range(n_dims_per_bucket)
])
"""

"""
interleave = np.arange(0, n_dims)
for base in range(0, n_dims, 2 * pair_separation):
    interleave[base:base + pair_separation] = np.arange(base, base + 2 * pair_separation, 2)
    interleave[base + pair_separation:base + 2 * pair_separation] = np.arange(base + 1, base + 2 * pair_separation + 1, 2)
"""

#print(bucket_ranges, bucket_centres, order[interleave])
#print(ranges[order][interleave].tolist())
#print(ranges.tolist())

with open("quantizer.msgpack", "wb") as f:
    msgpack.pack({
        "permutation": order.tolist(),
        "offsets": offsets,
        "scales": scales,
        "q_offsets": q_offsets,
        "q_scales": q_scales
    }, f)

def rquantize(vec):
    out = np.zeros(len(vec), dtype=np.uint8)
    for i, p in enumerate(order[interleave]):
        bucket = p % n_buckets
        raw = vec[i]
        raw = (raw - offsets[bucket]) * scales[bucket]
        raw = min(max(raw, 0.0), 255.0)
        out[p] = round(raw)
    return out

def rdquantize(bytes):
    vec = np.zeros(n_dims, dtype=np.float32)
    for i, p in enumerate(order[interleave]):
        bucket = p % n_buckets
        raw = float(bytes[p])
        vec[i] = raw / scales[bucket] + offsets[bucket]
    return vec

def rdot(x, y):
    xq_offsets = np.array(q_offsets, dtype=np.int16)
    xq_scales = np.array(q_scales, dtype=np.int16)
    assert x.shape == y.shape
    assert x.dtype == np.uint8 == y.dtype
    acc = 0
    for i in range(0, len(x), n_buckets):
        x1 = x[i:i+n_buckets].astype(np.int16) + xq_offsets
        y1 = y[i:i+n_buckets].astype(np.int16) + xq_offsets
        x1 *= xq_scales
        acc += np.dot(x1.astype(np.int32), y1.astype(np.int32))
    return acc

def cmp(i, j):
    return np.dot(data[i], data[j]) / rdot(rquantize(data[i]), rquantize(data[j]))

def rdot_cmp(a, b):
    x = rquantize(a)
    y = rquantize(b)
    a = a[order[interleave]]
    b = b[order[interleave]]
    xq_offsets = np.array(q_offsets, dtype=np.int16)
    xq_scales = np.array(q_scales, dtype=np.int16)
    assert x.shape == y.shape
    assert x.dtype == np.uint8 == y.dtype
    acc = 0
    for i in range(0, len(x), n_buckets):
        x1 = x[i:i+n_buckets].astype(np.int16) + xq_offsets
        y1 = y[i:i+n_buckets].astype(np.int16) + xq_offsets
        x1 *= xq_scales
        component = np.dot(x1.astype(np.int32), y1.astype(np.int32))
        a1 = a[i:i+n_buckets]
        b1 = b[i:i+n_buckets]
        component_exact = np.dot(a1, b1)
        print(x1, a1, sep="\n")
        print(component, component_exact, component / component_exact)
        acc += component
    return acc
