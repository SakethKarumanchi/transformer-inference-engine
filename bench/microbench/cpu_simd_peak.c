/* Microbenchmark 4 -- CPU SIMD peak throughput.
 *
 * An FMA loop at the widest vector width this CPU supports, with enough
 * independent accumulator chains that the measurement is issue-rate limited
 * rather than latency limited. Single threaded and stated as such: this is the
 * per-core ceiling, not an aggregate over the package.
 *
 * A scalar version of the identical loop runs alongside. If the compiler had
 * silently dropped the vectorisation, the two figures would match; the unit
 * test asserts the vector run is the faster of the two, which is the only
 * evidence that vectorisation actually happened.
 *
 * STAGE 0b CHANGE -- thread placement, and a checksum carried out of each path.
 *
 * The vectorised configuration exceeded the 5% standard-deviation limit in both
 * Stage 0 suite runs (5.86%, 5.60%) while the scalar reference of the same loop
 * did not (0.21%, 2.51%). Whatever explains that has to explain a difference
 * between two paths through one benchmark, not a property of the machine in
 * general. The retained per-sample timings do: the vector path's dispersion is
 * carried by DOWNWARD excursions -- contiguous blocks of samples that run
 * FASTER than the median (run 1 median 5.977 ms, min 5.008 ms; run 2 median
 * 5.911 ms, min 4.922 ms), recovering smoothly over several samples. External
 * interference can only ever make a sample slower, so interference cannot be
 * the cause here. Removing the k largest samples barely moves the figure
 * (5.862% -> 5.729% at k = 3), confirming it is not a tail effect either. The
 * shape is that of the core changing frequency during the run, which a
 * benchmark cannot control; the scalar path, which draws a fraction of the
 * power, never leaves its settled frequency.
 *
 * What IS controllable, and was not controlled, is where the thread ran and at
 * what priority. Both are fixed below, applied once and outside every timed
 * bracket. This addresses migration and preemption; it does NOT address
 * frequency variation, and this stage does not claim it will. The trip count,
 * the arithmetic and the flop derivation are deliberately unchanged: lengthening
 * a sample so it averages over frequency excursions would lower the reported
 * standard deviation by measuring something other than what Stage 0 measured,
 * which is engineering the validity test rather than the measurement.
 *
 * The checksums exist so tests/test_cpu_simd_peak.cu can prove the two paths
 * compute the same arithmetic rather than assume it. Every vector lane starts
 * from the same value as the matching scalar chain, so the vector checksum is
 * MB_SIMD_LANES times the scalar one in exact arithmetic.
 */
#include "microbench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <intrin.h>
#endif
#include <immintrin.h>

#define MB_SIMD_ACC   8
#define MB_SIMD_ITERS 2000000

/* ---- widest ISA the CPU reports --------------------------------------- */
const char *mb_cpu_widest_isa(void)
{
#if defined(_WIN32) || defined(__x86_64__) || defined(_M_X64)
    int r[4];
    __cpuid(r, 0);
    int maxleaf = r[0];
    __cpuid(r, 1);
    int ecx1 = r[2];
    int have_osxsave = (ecx1 >> 27) & 1;
    int have_avx     = (ecx1 >> 28) & 1;
    unsigned long long xcr0 = 0;
    if (have_osxsave && have_avx) xcr0 = _xgetbv(0);
    int os_ymm = (xcr0 & 0x6)  == 0x6;
    int os_zmm = (xcr0 & 0xe6) == 0xe6;

    if (maxleaf >= 7) {
        __cpuidex(r, 7, 0);
        int ebx7 = r[1];
        if (((ebx7 >> 16) & 1) && os_zmm) return "AVX512F";
        if (((ebx7 >>  5) & 1) && os_ymm) return "AVX2";
    }
    if (have_avx && os_ymm) return "AVX";
    return "SSE2";
#else
    return "unknown";
#endif
}

/* ---- what this translation unit was actually compiled for ------------- */
#if defined(__AVX512F__)
#  define MB_SIMD_ISA_COMPILED "AVX512F"
#  define MB_SIMD_WIDTH_BITS   512
#elif defined(__AVX2__)
#  define MB_SIMD_ISA_COMPILED "AVX2"
#  define MB_SIMD_WIDTH_BITS   256
#elif defined(__AVX__)
#  define MB_SIMD_ISA_COMPILED "AVX"
#  define MB_SIMD_WIDTH_BITS   256
#else
#  define MB_SIMD_ISA_COMPILED "SSE2"
#  define MB_SIMD_WIDTH_BITS   128
#endif

#if MB_SIMD_WIDTH_BITS >= 256
#  define MB_SIMD_LANES 8
typedef __m256 mb_vec;
#  define MB_VSET(x)        _mm256_set1_ps(x)
#  define MB_VFMA(a, b, c)  _mm256_fmadd_ps((a), (b), (c))
#  define MB_VADD(a, b)     _mm256_add_ps((a), (b))
#else
#  define MB_SIMD_LANES 4
typedef __m128 mb_vec;
#  define MB_VSET(x)        _mm_set1_ps(x)
#  define MB_VFMA(a, b, c)  _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#  define MB_VADD(a, b)     _mm_add_ps((a), (b))
#endif

typedef struct {
    int   iters;
    volatile float sink;
    /* Per-iteration total of each path, carried out for the unit test. Every
     * iteration starts from the same accumulators and runs the same trip count,
     * so the value is identical across iterations and deterministic. */
    double vector_total;
    double scalar_total;
} simd_ctx;

static double simd_vector_body(void *vctx, int iteration)
{
    (void)iteration;
    simd_ctx *c = (simd_ctx *)vctx;
    mb_vec acc[MB_SIMD_ACC];
    mb_vec a = MB_VSET(1.0000001f), b = MB_VSET(0.9999999f);
    for (int i = 0; i < MB_SIMD_ACC; ++i) acc[i] = MB_VSET((float)(i + 1));

    double t0 = bench_cpu_time_seconds();
    for (int it = 0; it < c->iters; ++it) {
        for (int i = 0; i < MB_SIMD_ACC; ++i) acc[i] = MB_VFMA(acc[i], b, a);
    }
    double t1 = bench_cpu_time_seconds();

    mb_vec s = acc[0];
    for (int i = 1; i < MB_SIMD_ACC; ++i) s = MB_VADD(s, acc[i]);
    float tmp[MB_SIMD_LANES];
#if MB_SIMD_WIDTH_BITS >= 256
    _mm256_storeu_ps(tmp, s);
#else
    _mm_storeu_ps(tmp, s);
#endif
    float total = 0.0f;
    for (int i = 0; i < MB_SIMD_LANES; ++i) total += tmp[i];
    c->vector_total = (double)total;
    c->sink += total;
    return (t1 - t0) * 1.0e3;
}

static double simd_scalar_body(void *vctx, int iteration)
{
    (void)iteration;
    simd_ctx *c = (simd_ctx *)vctx;
    float acc[MB_SIMD_ACC];
    const float a = 1.0000001f, b = 0.9999999f;
    for (int i = 0; i < MB_SIMD_ACC; ++i) acc[i] = (float)(i + 1);

    double t0 = bench_cpu_time_seconds();
#if defined(_MSC_VER)
#  pragma loop(no_vector)   /* the scalar reference must stay scalar */
#endif
    for (int it = 0; it < c->iters; ++it) {
        for (int i = 0; i < MB_SIMD_ACC; ++i) acc[i] = acc[i] * b + a;
    }
    double t1 = bench_cpu_time_seconds();

    float total = 0.0f;
    for (int i = 0; i < MB_SIMD_ACC; ++i) total += acc[i];
    c->scalar_total = (double)total;
    c->sink += total;
    return (t1 - t0) * 1.0e3;
}

int mb_cpu_simd_peak_run(int warmup, int samples, mb_simd_peak_result *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);

    snprintf(out->isa_compiled, sizeof out->isa_compiled, "%s", MB_SIMD_ISA_COMPILED);
    snprintf(out->isa_widest_supported, sizeof out->isa_widest_supported, "%s",
             mb_cpu_widest_isa());
    out->vector_width_bits = MB_SIMD_WIDTH_BITS;

    int n_alloc = samples > BENCH_MIN_SAMPLES ? samples : BENCH_MIN_SAMPLES;
    double *raw_v = (double *)malloc((size_t)n_alloc * sizeof(double));
    double *raw_s = (double *)malloc((size_t)n_alloc * sizeof(double));
    if (!raw_v || !raw_s) { free(raw_v); free(raw_s); return 1; }

    simd_ctx c;
    c.iters = MB_SIMD_ITERS;
    c.sink  = 0.0f;
    c.vector_total = 0.0;
    c.scalar_total = 0.0;

    /* OUTSIDE every timed bracket: applied once here, restored once at the end,
     * and recorded whether or not it took. */
    out->placement = bench_pin_current_thread(-1);

    int eff_w = 0, eff_s = 0;
    out->vector_stats = bench_run(simd_vector_body, &c, warmup, samples, raw_v, &eff_w, &eff_s);
    out->scalar_stats = bench_run(simd_scalar_body, &c, warmup, samples, raw_s, &eff_w, &eff_s);
    out->warmup  = eff_w;
    out->samples = eff_s;
    out->lanes           = MB_SIMD_LANES;
    out->vector_checksum = c.vector_total;
    out->scalar_checksum = c.scalar_total;

    /* FLOPs derived from the trip count: iterations * chains * lanes * 2. */
    double vec_flops = (double)c.iters * MB_SIMD_ACC * MB_SIMD_LANES * 2.0;
    double sca_flops = (double)c.iters * MB_SIMD_ACC * 2.0;
    out->flops_per_iteration = vec_flops;
    out->vector_gflops = (out->vector_stats.median > 0.0)
        ? vec_flops / (out->vector_stats.median * 1.0e-3) / 1.0e9 : 0.0;
    out->scalar_gflops = (out->scalar_stats.median > 0.0)
        ? sca_flops / (out->scalar_stats.median * 1.0e-3) / 1.0e9 : 0.0;

    {
        bench_kv_num nums[] = {
            { "vector_width_bits",  (double)MB_SIMD_WIDTH_BITS },
            { "lanes",              (double)MB_SIMD_LANES },
            { "chains",             (double)MB_SIMD_ACC },
            { "inner_iterations",   (double)c.iters },
            { "scalar_gflops",      out->scalar_gflops },
            { "vector_over_scalar", out->scalar_gflops > 0.0
                                    ? out->vector_gflops / out->scalar_gflops : 0.0 },
            { "vector_checksum",    out->vector_checksum },
            { "scalar_checksum",    out->scalar_checksum },
            { "thread_pinned",      (double)out->placement.pinned },
            { "thread_logical_cpu", (double)out->placement.logical_cpu },
        };
        bench_kv_str strs[] = {
            { "isa_compiled",           out->isa_compiled },
            { "isa_widest_supported",   out->isa_widest_supported },
            { "threading",              "single threaded, one core" },
            { "flop_count_derivation",  "iterations * chains * lanes * 2 flops per FMA" },
            { "tag",                    "measured" },
            { "thread_placement",       out->placement.detail },
            { "timed_bracket_contains", "the FMA loop only; accumulator setup, the "
                                        "reduction, thread placement and the "
                                        "checksum are all outside it" },
        };
        bench_record recs[2];
        memset(recs, 0, sizeof recs);
        recs[0].benchmark = "cpu_simd_peak";
        recs[0].configuration = "vectorised FMA loop, " MB_SIMD_ISA_COMPILED;
        recs[0].units = "GFLOP/s";
        recs[0].value = out->vector_gflops;
        recs[0].warmup = eff_w; recs[0].samples_requested = eff_s;
        recs[0].samples_ms = raw_v; recs[0].n_samples = eff_s;
        recs[0].stats = out->vector_stats;
        recs[0].meta_num = nums; recs[0].n_meta_num = (int)(sizeof nums / sizeof nums[0]);
        recs[0].meta_str = strs; recs[0].n_meta_str = (int)(sizeof strs / sizeof strs[0]);

        recs[1] = recs[0];
        recs[1].configuration = "scalar reference of the same loop";
        recs[1].value = out->scalar_gflops;
        recs[1].samples_ms = raw_s;
        recs[1].stats = out->scalar_stats;

        bench_write_results(NULL, "cpu_simd_peak", recs, 2);
    }

    free(raw_v);
    free(raw_s);
    bench_restore_current_thread();
    return 0;
}

#ifndef BENCH_NO_MAIN
int main(int argc, char **argv)
{
    int warmup  = (argc > 1) ? atoi(argv[1]) : BENCH_MIN_WARMUP;
    int samples = (argc > 2) ? atoi(argv[2]) : BENCH_MIN_SAMPLES;

    mb_simd_peak_result r;
    if (mb_cpu_simd_peak_run(warmup, samples, &r) != 0) {
        fprintf(stderr, "cpu_simd_peak: FAILED\n");
        return 1;
    }
    printf("cpu_simd_peak  %.3f GFLOP/s (%s, %d-bit, single thread)  stddev %.3f%%  %s\n",
           r.vector_gflops, r.isa_compiled, r.vector_width_bits,
           r.vector_stats.stddev_pct_of_median, r.vector_stats.valid ? "VALID" : "INVALID");
    printf("  scalar reference %.3f GFLOP/s, speedup %.2fx\n",
           r.scalar_gflops, r.scalar_gflops > 0 ? r.vector_gflops / r.scalar_gflops : 0.0);
    printf("  widest ISA CPUID reports: %s\n", r.isa_widest_supported);
    printf("  thread placement: %s\n", r.placement.detail);
    if (!r.vector_stats.valid) printf("  INVALID: %s\n", r.vector_stats.invalid_reason);
    return (r.vector_stats.valid && r.scalar_stats.valid) ? 0 : 2;
}
#endif
