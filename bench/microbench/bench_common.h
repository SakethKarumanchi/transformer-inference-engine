/* bench_common.h -- shared measurement harness for every Stage 0 microbenchmark.
 *
 * The run-structure rules from BENCHMARK_PROTOCOL.md live here ONCE and nowhere
 * else: minimum warmup, minimum timed samples, and the 5%-of-median standard
 * deviation validity rule. No microbenchmark may restate or relax them.
 *
 * Linkage is extern "C" so the two CPU benchmarks (.c) and the seven CUDA
 * benchmarks (.cu) share one harness object.
 */
#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- protocol constants (single source of truth) ---------------------- */
#define BENCH_MIN_WARMUP        5
#define BENCH_MIN_SAMPLES      20
#define BENCH_MAX_STDDEV_PCT  5.0   /* stddev > 5% of median => run is INVALID */

#define BENCH_STAGE_ID        "stage-0"
#define BENCH_REASON_LEN       256
#define BENCH_LABEL_LEN        128

/* ---- statistics ------------------------------------------------------- */
/* Standard deviation is the SAMPLE standard deviation (Bessel corrected,
 * divisor n-1). Median of an even-length set is the mean of the two middle
 * elements. Both are asserted against hand-computed values in
 * tests/test_bench_common.cu. */
typedef struct {
    int    n;
    double mean;
    double median;
    double min;
    double max;
    double stddev;
    double stddev_pct_of_median;
    int    valid;                        /* 1 = valid, 0 = INVALID */
    char   invalid_reason[BENCH_REASON_LEN];
} bench_stats;

/* Computes every statistic unconditionally, then applies the validity rules.
 * An INVALID verdict never suppresses the numbers -- an invalid run is
 * reported as invalid, never averaged away and never silently retried. */
bench_stats bench_compute_stats(const double *samples, int n);

/* Marks a stats block invalid for a reason the numbers cannot show by
 * themselves (background load, throttling, battery power). Appends to any
 * reason already present. */
void bench_mark_invalid(bench_stats *s, const char *reason);

/* ---- timing ----------------------------------------------------------- */
/* Highest-resolution monotonic counter available on the platform.
 * Windows: QueryPerformanceCounter (clock_gettime(CLOCK_MONOTONIC) is not
 * available under MSVC). Never wall clock. Returns seconds. */
double bench_cpu_time_seconds(void);
const char *bench_cpu_timer_name(void);

/* ---- warmup-and-sample driver ----------------------------------------- */
/* body(ctx, iteration) performs one iteration and returns its elapsed time in
 * milliseconds. The driver calls it (warmup + samples) times and writes only
 * the post-warmup timings into out_samples. Warmup iterations are discarded
 * and never appear in the reported statistics.
 *
 * warmup is raised to BENCH_MIN_WARMUP and samples to BENCH_MIN_SAMPLES if the
 * caller asks for fewer; the effective values are written back through
 * eff_warmup / eff_samples so the results file records what actually ran.
 * out_samples must have room for max(samples, BENCH_MIN_SAMPLES) doubles. */
typedef double (*bench_body_fn)(void *ctx, int iteration);

bench_stats bench_run(bench_body_fn body, void *ctx,
                      int warmup, int samples,
                      double *out_samples,
                      int *eff_warmup, int *eff_samples);

/* ---- CUDA event timing ------------------------------------------------ */
/* Available only in translation units compiled by nvcc. Kernel launches are
 * asynchronous, so every one of these synchronizes the device before reading
 * the timer. Timing brackets computation only. */
#ifdef __CUDACC__
typedef struct bench_cuda_timer bench_cuda_timer;
bench_cuda_timer *bench_cuda_timer_create(void);
void   bench_cuda_timer_destroy(bench_cuda_timer *t);
void   bench_cuda_timer_start(bench_cuda_timer *t);
double bench_cuda_timer_stop_ms(bench_cuda_timer *t);   /* synchronizes */
#endif

/* ---- structured results file ------------------------------------------ */
/* One JSON object per run, written to bench/results/. The format is a
 * contract: the Stage 10 performance model reads these files directly. Raw
 * per-sample timings are retained, not just summary statistics. */
typedef struct {
    const char *key;
    const char *value;      /* written as a JSON string */
} bench_kv_str;

typedef struct {
    const char *key;
    double      value;      /* written as a JSON number */
} bench_kv_num;

typedef struct {
    const char   *benchmark;        /* e.g. "gpu_bandwidth" */
    const char   *configuration;    /* e.g. "N=67108864 stride=1" */
    const char   *units;            /* units of the derived figure, e.g. "GB/s" */
    double        value;            /* the derived figure itself */
    int           warmup;
    int           samples_requested;
    const double *samples_ms;       /* raw per-sample timings, milliseconds */
    int           n_samples;
    bench_stats   stats;
    const bench_kv_str *meta_str;   int n_meta_str;
    const bench_kv_num *meta_num;   int n_meta_num;
} bench_record;

/* Writes one JSON file to <results_dir>/<benchmark>.json containing the stage
 * identifier, git commit hash, timestamp, device, compiler and flags, the raw
 * sample array, the computed statistics and any supplied counters.
 * Returns 0 on success. Multiple records may be written in one file. */
int bench_write_results(const char *results_dir,
                        const char *file_stem,
                        const bench_record *records,
                        int n_records);

/* Same payload, emitted to an already-open stream. Exposed for the unit test,
 * which parses the emitted text to prove it is valid JSON. */
int bench_write_results_fp(void *fp,
                           const bench_record *records,
                           int n_records);

/* Build/environment provenance, filled in by the build system. */
const char *bench_git_commit(void);
const char *bench_build_timestamp(void);
const char *bench_cxx_compiler(void);
const char *bench_cxx_flags(void);
const char *bench_cuda_compiler(void);
const char *bench_cuda_flags(void);

/* Device identity string ("NVIDIA GeForce ... sm_75"); "cpu-only-build" when
 * the harness was linked without a usable CUDA device. */
const char *bench_device_name(void);

/* Resolves the repository's bench/results directory from the executable's
 * location. Returns buf. */
char *bench_default_results_dir(char *buf, size_t buflen);

#ifdef __cplusplus
}   /* extern "C" */
#endif

#endif /* BENCH_COMMON_H */
