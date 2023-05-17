#include "dot_product.h"
#include "rt.h"
#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>

// AVX2 / AVX512 optimized dot_product functions:

float  dot_product_f32(const float* p, const float* q, uint64_t n);
double dot_product_f64(const double* p, const double* q, uint64_t n);

typedef struct avx2_if {
    void   (*init)(void);
    double (*dot_product_f64)(const double* p, const double* q, uint64_t n);
    float  (*dot_product_f32)(const float* p, const float* q, uint64_t n);
} avx2_if;

typedef struct avx512_if {
    void   (*init)(void);
    double (*dot_product_f64)(const double* p, const double* q, uint64_t n);
    float  (*dot_product_f32)(const float* p, const float* q, uint64_t n);
} avx512_if;

// _MM_HINT_T0 (temporal data) — prefetch data into all levels of the caches.
// _MM_HINT_T1 (temporal data with respect to first level cache misses) —
//              into L2 and higher.
// _MM_HINT_T2 (temporal data with respect to second level cache misses) —
//              into L3 and higher, or an implementation-specific choice.
// _MM_HINT_NTA (non-temporal data with respect to all cache levels) -
//              into non-temporal cache structure and into a location close
//              to the processor, minimizing cache pollution.

// Performance measuremnts confirm that prefetching into _MM_HINT_T0 is fastest

#define prefetch2_L1L2L3(p, q) do {              \
    _mm_prefetch((const char*)(p), _MM_HINT_T0); \
    _mm_prefetch((const char*)(q), _MM_HINT_T0); \
} while (0)

static void avx2_init(void);
static void avx512_init(void);

static avx2_if avx2   = { .init = avx2_init };
static avx2_if avx512 = { .init = avx512_init };

static inline double c_dot_product_f64(const double* p, const double* q, uint64_t n) {
    double sum = 0;
    const double* e = p + n;
    while (p < e) { sum += *p++ * *q++; }
    return sum;
}

static inline float c_dot_product_f32(const float* p, const float* q, uint64_t n) {
    float sum = 0;
    const float* e = p + n;
    while (p < e) { sum += *p++ * *q++; }
    return sum;
}

double dot_product_f64(const double* p, const double* q, uint64_t n) {
    prefetch2_L1L2L3(p, q);
    static bool init;
    if (!init) { avx2.init(); avx512.init(); init = true;}
    if (n >= 8 && avx512.dot_product_f64 != null) {
        return avx512.dot_product_f64(p, q, n);
    } else if (n >= 4 && avx2.dot_product_f64 != null) {
        return avx2.dot_product_f64(p, q, n);
    } else {
        return c_dot_product_f64(p, q, n);
    }
}

float dot_product_f32(const float* p, const float* q, uint64_t n) {
    prefetch2_L1L2L3(p, q);
    static bool init;
    if (!init) { avx2.init(); avx512.init(); init = true;}
    if (n >= 16 && avx512.dot_product_f64 != null) {
        return avx512.dot_product_f32(p, q, n);
    } else if (n >= 8 && avx2.dot_product_f64 != null) {
        return avx2.dot_product_f32(p, q, n);
    } else {
        return c_dot_product_f32(p, q, n);
    }
}

// f64_t double
#define f64x2_t __m128d
#define f64x4_t __m256d
#define f64x8_t __m512d

// f32_t float
#define f32x4_t __m128
#define f32x8_t __m256
#define f32x16_t __m512

static double avx2_dot_product_f64(const double* p, const double* q, uint64_t n) {
    assertion(n >= 4);
    prefetch2_L1L2L3(p, q);
    f64x4_t mul_add_f64x4 = _mm256_setzero_pd();
    while (n >= 4) {
        f64x4_t a = _mm256_loadu_pd(p);
        f64x4_t b = _mm256_loadu_pd(q);
        n -= 4; p += 4; q += 4;
        if (n > 0) { prefetch2_L1L2L3(p, q); }
        mul_add_f64x4 = _mm256_fmadd_pd(a, b, mul_add_f64x4);
    }
    double result[2];
    _mm_store_pd(result, _mm_add_pd(
        _mm256_castpd256_pd128(mul_add_f64x4),      // 0, 1
        _mm256_extractf64x2_pd(mul_add_f64x4, 1))); // 2, 3
    double sum = result[0] + result[1];
    while (n > 0) { sum += *p++ * *q++; n--; }
    return sum;
}

static float avx2_dot_product_f32(const float* p, const float* q, uint64_t n) {
    assertion(n >= 8);
    prefetch2_L1L2L3(p, q);
    f32x8_t mul_add_f32x8 = _mm256_setzero_ps();
    while (n >= 8) {
        f32x8_t a = _mm256_loadu_ps(p);
        f32x8_t b = _mm256_loadu_ps(q);
        n -= 8; p += 8; q += 8;
        if (n > 0) { prefetch2_L1L2L3(p, q); }
        mul_add_f32x8 = _mm256_fmadd_ps(a, b, mul_add_f32x8);
    }
    float result[4];
    _mm_store_ps(result, _mm_add_ps(
        _mm256_extractf32x4_ps(mul_add_f32x8, 0),   // 0,1,2,3
        _mm256_extractf32x4_ps(mul_add_f32x8, 1))); // 4,5,6,7
    float sum = result[0] + result[1] + result[2] + result[3];
    while (n > 0) { sum += *p++ * *q++; n--; }
    return sum;
}

static double avx512_dot_product_f64(const double* p, const double* q, uint64_t n) {
    assertion(n >= 8);
//  prefetch2_L1L2L3(p, q); (already done by caller)
    f64x8_t mul_add_f64x8 = _mm512_setzero_pd(); // multiply and add
    while (n >= 8) {
        f64x8_t a = _mm512_loadu_pd(p); // f64x8
        f64x8_t b = _mm512_loadu_pd(q); // f64x8
        n -= 8; p += 8; q += 8;
        if (n > 0) { prefetch2_L1L2L3(p, q); }
        mul_add_f64x8 = _mm512_fmadd_pd(a, b, mul_add_f64x8);
    }
    // Reduce the 512-bit sum to a single 128-bit sum using AVX
    f64x4_t f64x4 = _mm256_add_pd(
            _mm512_castpd512_pd256(mul_add_f64x8),     // 0,1,2,3
            _mm512_extractf64x4_pd(mul_add_f64x8, 1)); // 4,5,6,7
    f64x2_t f64x2 = _mm_add_pd(
        _mm256_castpd256_pd128(f64x4),     // 0,1
        _mm256_extractf64x2_pd(f64x4, 1)); // 2,3
    double f64[2];
    _mm_store_pd(f64, f64x2);
    double sum = f64[0] + f64[1];
    while (n > 0) { sum += *p++ * *q++; n--; }
    return sum;
}

static float avx512_dot_product_f32(const float* p, const float* q, uint64_t n) {
    assertion(n >= 16);
//  prefetch2_L1L2L3(p, q); (already done by caller)
    f32x16_t mul_add_f32x16 = _mm512_setzero_ps(); // multiply and add
    while (n >= 16) {
        f32x16_t a = _mm512_loadu_ps(p); // f32x8
        f32x16_t b = _mm512_loadu_ps(q); // f32x8
        n -= 16; p += 16; q += 16;
        if (n > 0) { prefetch2_L1L2L3(p, q); }
        mul_add_f32x16 = _mm512_fmadd_ps(a, b, mul_add_f32x16);
    }
    // Reduce the 512-bit sum to a single 128-bit sum using AVX
    f32x8_t f32x8 = _mm256_add_ps(
            _mm512_castps512_ps256(mul_add_f32x16),     // 0,1,2,3,4,5,6,7
            _mm512_extractf32x8_ps(mul_add_f32x16, 1)); // 8,9,10,11,12,13,14,15
    f32x4_t f32x4 = _mm_add_ps(
        _mm256_castps256_ps128(f32x8),     // 0,1,2,3
        _mm256_extractf32x4_ps(f32x8, 1)); // 4,5,6,7
    float f32[4];
    _mm_store_ps(f32, f32x4);
    float sum = f32[0] + f32[1] + f32[2] + f32[3];
    while (n > 0) { sum += *p++ * *q++; n--; }
    return sum;
}

static void test_dot_product_f32() {
    float a[21] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    float b[21] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    for (int i = 1; i < countof(a); i++) {
        float sum0 = c_dot_product_f32(a, b, i);
        static const float avx_flt_epsilon = FLT_EPSILON; // 1.192092896E-07f
        if (i >= 16) {
            float sum1 = avx2_dot_product_f32(a, b, i);
            assertion(fabs(sum1 - sum0) < avx_flt_epsilon,
                "cpu: %.16f avx: %.16f delta: %.16e FLT_EPSILON: %.16e",
                sum0, sum1, sum0 - sum1, FLT_EPSILON);
            float sum2 = avx512_dot_product_f32(a, b, i);
            assertion(fabs(sum2 - sum0) < avx_flt_epsilon,
                "cpu: %.16f avx: %.16f delta: %.16e FLT_EPSILON: %.16e",
                sum0, sum2, sum0 - sum2, FLT_EPSILON);
             (void)sum1; (void)sum2; // unused in release
        } else if (i >= 8) {
            float sum1 = avx2_dot_product_f32(a, b, i);
            assertion(fabs(sum1 - sum0) < avx_flt_epsilon,
                "cpu: %.16f avx: %.16f delta: %.16e FLT_EPSILON: %.16e",
                sum0, sum1, sum0 - sum1, FLT_EPSILON);
            (void)sum1; // unused in release
        }
        (void)sum0; // unused in release
    }
}

static void test_dot_product_f64() {
    double a[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    double b[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    for (int i = 1; i < countof(a); i++) {
        double sum0 = c_dot_product_f64(a, b, i);
        static const double avx_dbl_epsilon = 2.0e-13;
        if (i >= 8) {
            double sum1 = avx2_dot_product_f64(a, b, i);
            assertion(fabs(sum1 - sum0) < avx_dbl_epsilon,
                "cpu: %.16f avx: %.16f delta: %.16e DBL_EPSILON: %.16e",
                sum0, sum1, sum0 - sum1, DBL_EPSILON);
            double sum2 = avx512_dot_product_f64(a, b, i);
            assertion(fabs(sum2 - sum0) < avx_dbl_epsilon,
                "cpu: %.16f avx: %.16f delta: %.16e DBL_EPSILON: %.16e",
                sum0, sum2, sum0 - sum2, DBL_EPSILON);
             (void)sum1; (void)sum2; // unused in release
       } else if (i >= 4) {
            double sum1 = avx2_dot_product_f64(a, b, i);
            assertion(fabs(sum1 - sum0) < avx_dbl_epsilon,
                "cpu: %.16f avx: %.16f delta: %.16e DBL_EPSILON: %.16e",
                sum0, sum1, sum0 - sum1, DBL_EPSILON);
            (void)sum1; // unused in release
        }
        (void)sum0; // unused in release
    }
}

#undef TEST_DOT_PRODUCT_PERFORMANCE

#ifdef TEST_DOT_PRODUCT_PERFORMANCE

static void dot_product_performance() {
    threads.realtime();
    static float a[1024][16 * 1024];
    static float b[1024][16 * 1024];
    float total = 0;
    uint32_t seed = 0;
    for (int i = 0; i < countof(a); i++) {
        for (int j = 0; j < countof(a[i]); j++) {
            a[i][j] = random32(&seed) / (float)UINT32_MAX - 0.5f;
            b[i][j] = random32(&seed) / (float)UINT32_MAX - 0.5f;
        }
    }
    static byte flush_L1L2L3[128 * 1024 * 1024];
    memset(flush_L1L2L3, 0xFF, sizeof(flush_L1L2L3));
    // AVX-512
    double ms_avx512 = seconds() * 1000;
    for (int k = 0; k < 1000; k++) {
        for (int i = 0; i < countof(a); i++) {
            total += avx512_dot_product_f32(a[i], b[i], countof(a[i]));
        }
    }
    ms_avx512 = seconds() * 1000 - ms_avx512;
    // AVX-2
    double ms_avx2 = seconds() * 1000;
    for (int k = 0; k < 1000; k++) {
        for (int i = 0; i < countof(a); i++) {
            total += avx2_dot_product_f32(a[i], b[i], countof(a[i]));
        }
    }
    ms_avx2 = seconds() * 1000 - ms_avx2;
    // C
    double ms_c = seconds() * 1000;
    for (int k = 0; k < 1000; k++) {
        for (int i = 0; i < countof(a); i++) {
            total += c_dot_product_f32(a[i], b[i], countof(a[i]));
        }
    }
    ms_c = seconds() * 1000 - ms_c;
    double ns_c      = (1000 * (ms_c / countof(a))) / countof(a[0]);
    double ns_avx2   = (1000 * (ms_avx2 / countof(a))) / countof(a[0]);
    double ns_avx512 = (1000 * (ms_avx512 / countof(a))) / countof(a[0]);
    traceln("C     : %7.3f nanoseconds\n", ns_c);
    traceln("avx2  : %7.3f nanoseconds\n", ns_avx2);
    traceln("avx512: %7.3f nanoseconds\n", ns_avx512);
    // GFlops (2 flops per element)
    double gfps_c      = 2.0 / ns_c;
    double gfps_avx2   = 2.0 / ns_avx2;
    double gfps_avx512 = 2.0 / ns_avx512;
    traceln("C     : %7.3f Gflops\n", gfps_c);
    traceln("avx2  : %7.3f Gflops\n", gfps_avx2);
    traceln("avx512: %7.3f Gflops\n", gfps_avx512);
    // total referenced to prevent compiler from optimizing out
    if (total == 0) { exit(1); } // what are the odds of that?!
    exit(0);
}

// 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz / 4.00 GHz
// f32_t [16 * 1024] x [16 * 1024] dot product
// per add/mult flop:
// C     :   0.906 nanoseconds
// avx2  :   0.283 nanoseconds (faster then avx512 - probably due to memory access)
// avx512:   0.315 nanoseconds
// C     :   2.207 Gflops
// avx2  :   7.057 Gflops
// avx512:   6.343 Gflops

#endif // TEST_DOT_PRODUCT_PERFORMANCE

// 1. AXV512 on Gen-11 Intel CPU's measures slower then AVX2
// 2. AVX512-FP16
// https://cdrdv2-public.intel.com/678970/intel-avx512-fp16.pdf
// https://web.archive.org/web/20230426181123/https://cdrdv2-public.intel.com/678970/intel-avx512-fp16.pdf
//    is cool, but MAY NOT be supported on Gen-12 Intel CPU's:
// https://gist.github.com/FCLC/56e4b3f4a4d98cfd274d1430fabb9458
// 3. GPUs support FP16 for a long time and Microsoft has weak provision for that:
// https://learn.microsoft.com/en-us/windows/win32/dxmath/half-data-type

static void avx512_init(void) {
    double d0[16] = { 0 };
    double d1[16] = { 0 };
    __try {
        avx512_dot_product_f64(d0, d1, countof(d0));
        avx512.dot_product_f64 = avx512_dot_product_f64;
        avx512.dot_product_f32 = avx512_dot_product_f32;
    } __except(1) {
    }
}

static void avx2_init(void) {
    double d0[16] = { 0 };
    double d1[16] = { 0 };
    __try {
        avx2_dot_product_f64(d0, d1, countof(d0));
        avx2.dot_product_f64 = avx2_dot_product_f64;
        avx2.dot_product_f32 = avx2_dot_product_f32;
    } __except(1) {
    }
}


