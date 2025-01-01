use core::f32;

use half::f16;
use simsimd::SpatialSimilarity;
use fastrand::Rng;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct Vector(Vec<f16>);
#[derive(Debug, Clone)]
pub struct SVector(Vec<f32>);

pub type VectorRef<'a> = &'a [f16];
pub type QVectorRef<'a> = &'a [u8];
pub type SVectorRef<'a> = &'a [f32];

impl SVector {
    pub fn zero(d: usize) -> Self {
        SVector(vec![0.0; d])
    }
    pub fn half(&self) -> Vector {
        Vector(self.0.iter().map(|a| f16::from_f32(*a)).collect())
    }
}

fn box_muller(rng: &mut Rng) -> f32 {
    loop {
        let u = rng.f32();
        let v = rng.f32();
        let x = (v * std::f32::consts::TAU).cos() * (-2.0 * u.ln()).sqrt();
        if x.is_finite() {
            return x;
        }
    }
}

impl Vector {
    pub fn zero(d: usize) -> Self {
        Vector(vec![f16::from_f32(0.0); d])
    }

    pub fn randn(rng: &mut Rng, d: usize) -> Self {
        Vector(Vec::from_iter((0..d).map(|_| f16::from_f32(box_muller(rng)))))
    }
}

// Floats are vaguely annoying and not sortable (trivially), so we mostly represent dot products as integers
pub const SCALE: f32 = 1099511627776.0;
pub const SCALE_F64: f64 = SCALE as f64;

pub fn dot<'a>(x: VectorRef<'a>, y: VectorRef<'a>) -> i64 {
    // safety is not real
    scale_dot_result_f64(simsimd::f16::dot(unsafe { std::mem::transmute(x) }, unsafe { std::mem::transmute(y) }).unwrap())
}

pub fn to_svector(vec: VectorRef) -> SVector {
    SVector(vec.iter().map(|a| a.to_f32()).collect())
}

impl<'a> std::ops::AddAssign<VectorRef<'a>> for SVector {
    fn add_assign(&mut self, other: VectorRef<'a>) {
        self.0.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b.to_f32());
    }
}

impl std::ops::Div<f32> for SVector {
    type Output = Self;

    fn div(self, b: f32) -> Self::Output {
        SVector(self.0.iter().map(|a| a / b).collect())
    }
}

impl std::ops::Deref for Vector {
    type Target = [f16];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::Deref for SVector {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::Add<&SVector> for SVector {
    type Output = Self;

    fn add(self, other: &Self) -> Self::Output {
        SVector(self.0.iter().zip(other.0.iter()).map(|(a, b)| a + b).collect())
    }
}

impl std::ops::Sub<&SVector> for SVector {
    type Output = Self;

    fn sub(self, other: &Self) -> Self::Output {
        SVector(self.0.iter().zip(other.0.iter()).map(|(a, b)| a - b).collect())
    }
}

impl std::ops::AddAssign for SVector {
    fn add_assign(&mut self, other: Self) {
        self.0.iter_mut().zip(other.0.iter()).for_each(|(a, b)| *a += b);
    }
}

impl std::ops::Mul<f32> for SVector {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        SVector(self.0.iter().map(|a| *a * other).collect())
    }
}

#[derive(Debug, Clone)]
pub struct VectorList {
    pub d_emb: usize,
    pub length: usize,
    pub data: Vec<f16>
}

impl std::ops::Index<usize> for VectorList {
    type Output = [f16];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.d_emb..(index + 1) * self.d_emb]
    }
}

pub struct VectorListIterator<'a> {
    list: &'a VectorList,
    index: usize
}

impl<'a> Iterator for VectorListIterator<'a> {
    type Item = VectorRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.list.len() {
            let ret = &self.list[self.index];
            self.index += 1;
            Some(ret)
        } else {
            None
        }
    }
}


impl VectorList {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn iter(&self) -> VectorListIterator {
        VectorListIterator {
            list: self,
            index: 0
        }
    }

    pub fn empty(d: usize) -> Self {
        VectorList {
            d_emb: d,
            length: 0,
            data: Vec::new()
        }
    }

    pub fn from_f16s(f16s: Vec<f16>, d: usize) -> Self {
        assert!(f16s.len() % d == 0);
        VectorList {
            d_emb: d,
            length: f16s.len() / d,
            data: f16s
        }
    }

    pub fn push(&mut self, vec: VectorRef) {
        self.length += 1;
        self.data.extend_from_slice(vec);
    }
}

// SimSIMD has its own, but ours prefetches concurrently, is unrolled more, ignores inconveniently-sized vectors and does a cheaper reduction
// Also, we return an int because floats are annoying (not Ord)
// On Tiger Lake (i5-1135G7) we have about a 3x performance advantage ignoring the prefetching
// (it would be better to use AVX512 for said CPU but this also has to run on Zen 3)
pub fn fast_dot(x: VectorRef, y: VectorRef, prefetch: VectorRef) -> i64 {
    use std::arch::x86_64::*;

    debug_assert!(x.len() == y.len());
    debug_assert!(prefetch.len() == x.len());
    debug_assert!(x.len() % 64 == 0);

    // safety is not real
    // it's probably fine I guess
    unsafe {
        let mut x_ptr = x.as_ptr();
        let mut y_ptr = y.as_ptr();
        let end = x_ptr.add(x.len());
        let mut prefetch_ptr = prefetch.as_ptr();

        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut acc4 = _mm256_setzero_ps();

        while x_ptr < end {
            // fetch chunks and prefetch next vector
            let x1 = _mm256_loadu_si256(x_ptr as *const __m256i);
            let y1 = _mm256_loadu_si256(y_ptr as *const __m256i);
            let x2 = _mm256_loadu_si256(x_ptr.add(16) as *const __m256i);
            let y2 = _mm256_loadu_si256(y_ptr.add(16) as *const __m256i);
            // technically, we only have to do this once per cache line but I don't care enough to test every way to optimize this
            _mm_prefetch(prefetch_ptr as *const i8, _MM_HINT_T0);
            x_ptr = x_ptr.add(32); // move 16 f16s at a time
            y_ptr = y_ptr.add(32);
            prefetch_ptr = prefetch_ptr.add(32);

            // unpack f32 to f16
            let x1lo = _mm256_cvtph_ps(_mm256_extractf128_si256(x1, 0));
            let x1hi = _mm256_cvtph_ps(_mm256_extractf128_si256(x1, 1));
            let y1lo = _mm256_cvtph_ps(_mm256_extractf128_si256(y1, 0));
            let y1hi = _mm256_cvtph_ps(_mm256_extractf128_si256(y1, 1));
            let x2lo = _mm256_cvtph_ps(_mm256_extractf128_si256(x2, 0));
            let x2hi = _mm256_cvtph_ps(_mm256_extractf128_si256(x2, 1));
            let y2lo = _mm256_cvtph_ps(_mm256_extractf128_si256(y2, 0));
            let y2hi = _mm256_cvtph_ps(_mm256_extractf128_si256(y2, 1));

            acc1 = _mm256_fmadd_ps(x1lo, y1lo, acc1);
            acc2 = _mm256_fmadd_ps(x1hi, y1hi, acc2);
            acc3 = _mm256_fmadd_ps(x2lo, y2lo, acc3);
            acc4 = _mm256_fmadd_ps(x2hi, y2hi, acc4);
        }

        // reduce
        let acc1 = _mm256_add_ps(acc1, acc2);
        let acc2 = _mm256_add_ps(acc3, acc4);

        let hsum = _mm256_hadd_ps(acc1, acc2);
        let hsum_lo = _mm256_extractf128_ps(hsum, 0);
        let hsum_hi = _mm256_extractf128_ps(hsum, 1);
        let hsum = _mm_add_ps(hsum_lo, hsum_hi);

        let floatval = f32::from_bits(_mm_extract_ps::<0>(hsum) as u32) + f32::from_bits(_mm_extract_ps::<1>(hsum) as u32) + f32::from_bits(_mm_extract_ps::<2>(hsum) as u32) + f32::from_bits(_mm_extract_ps::<3>(hsum) as u32);
        (floatval * SCALE) as i64
    }
}

// same as above, without prefetch pointer
pub fn fast_dot_noprefetch(x: VectorRef, y: VectorRef) -> i64 {
    use std::arch::x86_64::*;

    debug_assert!(x.len() == y.len());
    debug_assert!(x.len() % 64 == 0);

    unsafe {
        let mut x_ptr = x.as_ptr();
        let mut y_ptr = y.as_ptr();
        let end = x_ptr.add(x.len());

        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut acc4 = _mm256_setzero_ps();

        while x_ptr < end {
            let x1 = _mm256_loadu_si256(x_ptr as *const __m256i);
            let y1 = _mm256_loadu_si256(y_ptr as *const __m256i);
            let x2 = _mm256_loadu_si256(x_ptr.add(16) as *const __m256i);
            let y2 = _mm256_loadu_si256(y_ptr.add(16) as *const __m256i);
            x_ptr = x_ptr.add(32);
            y_ptr = y_ptr.add(32);

            let x1lo = _mm256_cvtph_ps(_mm256_extractf128_si256(x1, 0));
            let x1hi = _mm256_cvtph_ps(_mm256_extractf128_si256(x1, 1));
            let y1lo = _mm256_cvtph_ps(_mm256_extractf128_si256(y1, 0));
            let y1hi = _mm256_cvtph_ps(_mm256_extractf128_si256(y1, 1));
            let x2lo = _mm256_cvtph_ps(_mm256_extractf128_si256(x2, 0));
            let x2hi = _mm256_cvtph_ps(_mm256_extractf128_si256(x2, 1));
            let y2lo = _mm256_cvtph_ps(_mm256_extractf128_si256(y2, 0));
            let y2hi = _mm256_cvtph_ps(_mm256_extractf128_si256(y2, 1));

            acc1 = _mm256_fmadd_ps(x1lo, y1lo, acc1);
            acc2 = _mm256_fmadd_ps(x1hi, y1hi, acc2);
            acc3 = _mm256_fmadd_ps(x2lo, y2lo, acc3);
            acc4 = _mm256_fmadd_ps(x2hi, y2hi, acc4);
        }

        // reduce
        let acc1 = _mm256_add_ps(acc1, acc2);
        let acc2 = _mm256_add_ps(acc3, acc4);

        let hsum = _mm256_hadd_ps(acc1, acc2);
        let hsum_lo = _mm256_extractf128_ps(hsum, 0);
        let hsum_hi = _mm256_extractf128_ps(hsum, 1);
        let hsum = _mm_add_ps(hsum_lo, hsum_hi);

        let floatval = f32::from_bits(_mm_extract_ps::<0>(hsum) as u32) + f32::from_bits(_mm_extract_ps::<1>(hsum) as u32) + f32::from_bits(_mm_extract_ps::<2>(hsum) as u32) + f32::from_bits(_mm_extract_ps::<3>(hsum) as u32);
        (floatval * SCALE) as i64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    centroids: Vec<f32>,
    transform: Vec<f32>, // D*D orthonormal matrix
    pub n_dims_per_code: usize,
    pub n_dims: usize
}

// chunk * centroid_index
pub struct DistanceLUT(Vec<f32>);

impl ProductQuantizer {
    pub fn apply_transform(&self, x: &[f32]) -> Vec<f32> {
        let dim = self.n_dims;
        let n_vectors = x.len() / dim;
        let mut transformed = vec![0.0; n_vectors * dim];
        // transform_matrix (D * D) @ batch.T (D * B)
        unsafe {
            matrixmultiply::sgemm(dim, dim, n_vectors, 1.0, self.transform.as_ptr(), dim as isize, 1, x.as_ptr(), 1, dim as isize, 0.0, transformed.as_mut_ptr(), 1, dim as isize);
        }
        transformed
    }

    pub fn quantize_batch(&self, x: &[f32]) -> Vec<u8> {
        // x is B * D
        let dim = self.n_dims;
        assert_eq!(dim * dim, self.transform.len());
        let n_vectors = x.len() / dim;
        let n_centroids = self.centroids.len() / dim;
        assert!(n_centroids <= 256);
        let transformed = self.apply_transform(&x); // B * D, as we write sgemm result in a weird order
        let mut codes = vec![0; n_vectors * dim / self.n_dims_per_code];
        let vec_len_codes = dim / self.n_dims_per_code;

        // B * C buffer of similarity of each vector to each centroid, within subspace
        let mut scratch = vec![0.0; n_vectors * n_centroids];

        for i in 0..(dim / self.n_dims_per_code) {
            let offset = i * self.n_dims_per_code;
            // transformed_batch[:, range] (B * D_r) @ centroids[:, range].T (D_r * C)
            unsafe {
                matrixmultiply::sgemm(n_vectors, self.n_dims_per_code, n_centroids, 1.0, transformed.as_ptr().add(offset), dim as isize, 1, self.centroids.as_ptr().add(offset), 1, dim as isize, 0.0, scratch.as_mut_ptr(), n_centroids as isize, 1);
            }
            // assign this component to best centroid
            for i_vec in 0..n_vectors {
                let mut best = f32::NEG_INFINITY;
                for i_centroid in 0..n_centroids {
                    let score = scratch[i_vec * n_centroids + i_centroid];
                    if score > best {
                        best = score;
                        codes[i_vec * vec_len_codes + i] = i_centroid as u8;
                    }
                }
            }
        }
        codes
    }

    // not particularly performance-sensitive right now; do unbatched
    pub fn preprocess_query(&self, query: &[f32]) -> DistanceLUT {
        let transformed = self.apply_transform(query);
        let n_chunks = self.n_dims / self.n_dims_per_code;
        let n_centroids = self.centroids.len() / self.n_dims;
        let mut lut = Vec::with_capacity(n_chunks * n_centroids);

        for i in 0..n_chunks {
            let vec_component = &transformed[i * self.n_dims_per_code..(i + 1) * self.n_dims_per_code];
            for j in 0..n_centroids {
                let centroid = &self.centroids[j * self.n_dims..(j + 1) * self.n_dims];
                let centroid_component = &centroid[i * self.n_dims_per_code..(i + 1) * self.n_dims_per_code];
                let score = SpatialSimilarity::dot(vec_component, centroid_component).unwrap();
                lut.push(score as f32);
            }
        }

        DistanceLUT(lut)
    }

    // compute dot products of query against product-quantized vectors
    pub fn asymmetric_dot_product(&self, query: &DistanceLUT, pq_vectors: &[u8]) -> Vec<i64> {
        let n_chunks = self.n_dims / self.n_dims_per_code;
        let n_vectors = pq_vectors.len() / n_chunks;
        let mut scores = vec![0.0; n_vectors];
        let n_centroids = self.centroids.len() / self.n_dims;

        for i in 0..n_chunks {
            for j in 0..n_vectors {
                let code = pq_vectors[j * n_chunks + i];
                let chunk_score = query.0[i * n_centroids + code as usize];
                scores[j] += chunk_score;
            }
        }

        // I have no idea why but we somehow have significant degradation in search quality
        // if this accumulates in integers. As such, do floats and convert at the end.
        // I'm sure there are fascinating reasons for this, but God is dead, God remains dead, etc.
        scores.into_iter().map(scale_dot_result).collect()
    }
}

#[inline]
pub fn scale_dot_result(x: f32) -> i64 {
    (x * SCALE) as i64
}

#[inline]
pub fn scale_dot_result_f64(x: f64) -> i64 {
    (x * SCALE_F64) as i64
}

#[cfg(test)]
mod bench {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_dot(be: &mut Bencher) {
        let mut rng = fastrand::Rng::with_seed(1);
        let a = Vector::randn(&mut rng, 1024);
        let b = Vector::randn(&mut rng, 1024);
        be.iter(|| {
            dot(&a, &b)
        });
    }

    #[bench]
    fn bench_fastdot(be: &mut Bencher) {
        let mut rng = fastrand::Rng::with_seed(1);
        let a = Vector::randn(&mut rng, 1024);
        let b = Vector::randn(&mut rng, 1024);
        be.iter(|| {
            fast_dot(&a, &b, &a)
        });
    }

    #[bench]
    fn bench_fastdot_noprefetch(be: &mut Bencher) {
        let mut rng = fastrand::Rng::with_seed(1);
        let a = Vector::randn(&mut rng, 1024);
        let b = Vector::randn(&mut rng, 1024);
        be.iter(|| {
            fast_dot_noprefetch(&a, &b)
        });
    }
}
