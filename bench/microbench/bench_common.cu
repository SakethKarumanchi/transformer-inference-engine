/* bench_common.cu -- implementation of the shared Stage 0 measurement harness.
 * See bench_common.h for the contract. Compiled by nvcc, exported with C
 * linkage so the .c microbenchmarks link against the same object. */

#include "bench_common.h"
#include "build_info.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

/* ===================== statistics ===================================== */

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

extern "C" bench_stats bench_compute_stats(const double *samples, int n)
{
    bench_stats s;
    memset(&s, 0, sizeof s);
    s.n = n;

    if (n <= 0 || samples == NULL) {
        s.valid = 0;
        snprintf(s.invalid_reason, BENCH_REASON_LEN, "no samples supplied");
        return s;
    }

    double *sorted = (double *)malloc((size_t)n * sizeof(double));
    if (!sorted) {
        s.valid = 0;
        snprintf(s.invalid_reason, BENCH_REASON_LEN, "allocation failed in bench_compute_stats");
        return s;
    }
    memcpy(sorted, samples, (size_t)n * sizeof(double));
    qsort(sorted, (size_t)n, sizeof(double), cmp_double);

    s.min = sorted[0];
    s.max = sorted[n - 1];
    s.median = (n % 2) ? sorted[n / 2]
                       : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);

    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += samples[i];
    s.mean = sum / (double)n;

    /* sample standard deviation, divisor n-1 */
    if (n > 1) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = samples[i] - s.mean;
            acc += d * d;
        }
        s.stddev = sqrt(acc / (double)(n - 1));
    } else {
        s.stddev = 0.0;
    }

    s.stddev_pct_of_median = (s.median != 0.0)
                           ? 100.0 * s.stddev / fabs(s.median)
                           : 0.0;
    free(sorted);

    /* --- validity rules. Both are checked; both reasons are reported. --- */
    s.valid = 1;
    s.invalid_reason[0] = '\0';

    if (n < BENCH_MIN_SAMPLES) {
        char r[BENCH_REASON_LEN];
        snprintf(r, sizeof r,
                 "sample count %d is below the protocol minimum of %d",
                 n, BENCH_MIN_SAMPLES);
        bench_mark_invalid(&s, r);
    }
    if (s.stddev_pct_of_median > BENCH_MAX_STDDEV_PCT) {
        char r[BENCH_REASON_LEN];
        snprintf(r, sizeof r,
                 "stddev is %.3f%% of median, above the %.1f%% protocol limit",
                 s.stddev_pct_of_median, (double)BENCH_MAX_STDDEV_PCT);
        bench_mark_invalid(&s, r);
    }
    return s;
}

extern "C" void bench_mark_invalid(bench_stats *s, const char *reason)
{
    if (!s || !reason) return;
    s->valid = 0;
    size_t used = strlen(s->invalid_reason);
    if (used == 0) {
        snprintf(s->invalid_reason, BENCH_REASON_LEN, "%s", reason);
    } else if (used + 3 < BENCH_REASON_LEN) {
        snprintf(s->invalid_reason + used, BENCH_REASON_LEN - used, "; %s", reason);
    }
}

/* ===================== monotonic CPU timing =========================== */

extern "C" const char *bench_cpu_timer_name(void)
{
#if defined(_WIN32)
    return "QueryPerformanceCounter";      /* monotonic, highest resolution on Windows */
#else
    return "clock_gettime(CLOCK_MONOTONIC)";
#endif
}

extern "C" double bench_cpu_time_seconds(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int have_freq = 0;
    if (!have_freq) { QueryPerformanceFrequency(&freq); have_freq = 1; }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

/* ===================== warmup-and-sample driver ======================= */

extern "C" bench_stats bench_run(bench_body_fn body, void *ctx,
                                 int warmup, int samples,
                                 double *out_samples,
                                 int *eff_warmup, int *eff_samples)
{
    if (warmup  < BENCH_MIN_WARMUP)  warmup  = BENCH_MIN_WARMUP;
    if (samples < BENCH_MIN_SAMPLES) samples = BENCH_MIN_SAMPLES;
    if (eff_warmup)  *eff_warmup  = warmup;
    if (eff_samples) *eff_samples = samples;

    /* Warmup iterations run exactly like timed ones and are then discarded;
     * their timings never reach out_samples. */
    for (int i = 0; i < warmup; ++i) (void)body(ctx, i);

    for (int i = 0; i < samples; ++i) out_samples[i] = body(ctx, warmup + i);

    return bench_compute_stats(out_samples, samples);
}

/* ===================== CUDA event timing ============================== */

struct bench_cuda_timer { cudaEvent_t start, stop; };

extern "C" bench_cuda_timer *bench_cuda_timer_create(void)
{
    bench_cuda_timer *t = (bench_cuda_timer *)malloc(sizeof(bench_cuda_timer));
    if (!t) return NULL;
    if (cudaEventCreate(&t->start) != cudaSuccess ||
        cudaEventCreate(&t->stop)  != cudaSuccess) { free(t); return NULL; }
    return t;
}

extern "C" void bench_cuda_timer_destroy(bench_cuda_timer *t)
{
    if (!t) return;
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
    free(t);
}

extern "C" void bench_cuda_timer_start(bench_cuda_timer *t)
{
    cudaEventRecord(t->start, 0);
}

extern "C" double bench_cuda_timer_stop_ms(bench_cuda_timer *t)
{
    cudaEventRecord(t->stop, 0);
    /* Launches are asynchronous. Synchronize before reading the timer, or the
     * measurement is launch overhead rather than execution. */
    cudaEventSynchronize(t->stop);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return (double)ms;
}

/* ===================== provenance ===================================== */

extern "C" const char *bench_git_commit(void)      { return BENCH_GIT_COMMIT; }
extern "C" const char *bench_build_timestamp(void) { return BENCH_BUILD_TIMESTAMP; }
extern "C" const char *bench_cxx_compiler(void)    { return BENCH_CXX_COMPILER; }
extern "C" const char *bench_cxx_flags(void)       { return BENCH_CXX_FLAGS; }
extern "C" const char *bench_cuda_compiler(void)   { return BENCH_CUDA_COMPILER; }
extern "C" const char *bench_cuda_flags(void)      { return BENCH_CUDA_FLAGS; }

extern "C" const char *bench_device_name(void)
{
    static char name[256];
    static int resolved = 0;
    if (!resolved) {
        resolved = 1;
        int n = 0;
        cudaDeviceProp p;
        if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0 &&
            cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
            snprintf(name, sizeof name, "%s sm_%d%d", p.name, p.major, p.minor);
        } else {
            snprintf(name, sizeof name, "cpu-only-build");
        }
    }
    return name;
}

extern "C" char *bench_default_results_dir(char *buf, size_t buflen)
{
    const char *env = getenv("BENCH_RESULTS_DIR");
    snprintf(buf, buflen, "%s", (env && *env) ? env : BENCH_RESULTS_DIR);
    return buf;
}

/* ===================== JSON emission ================================== */

static void json_str(FILE *f, const char *s)
{
    fputc('"', f);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
            switch (*p) {
                case '"':  fputs("\\\"", f); break;
                case '\\': fputs("\\\\", f); break;
                case '\n': fputs("\\n",  f); break;
                case '\r': fputs("\\r",  f); break;
                case '\t': fputs("\\t",  f); break;
                default:
                    if (*p < 0x20) fprintf(f, "\\u%04x", *p);
                    else           fputc(*p, f);
            }
        }
    }
    fputc('"', f);
}

/* Emits a finite double, or JSON null for NaN/inf. A non-finite figure must
 * never be written as a number that downstream code would treat as real. */
static void json_num(FILE *f, double v)
{
    if (isfinite(v)) fprintf(f, "%.17g", v);
    else             fputs("null", f);
}

static void json_stats(FILE *f, const bench_stats *s)
{
    fputs("{", f);
    fprintf(f, "\"n\":%d,", s->n);
    fputs("\"mean_ms\":", f);   json_num(f, s->mean);   fputc(',', f);
    fputs("\"median_ms\":", f); json_num(f, s->median); fputc(',', f);
    fputs("\"min_ms\":", f);    json_num(f, s->min);    fputc(',', f);
    fputs("\"max_ms\":", f);    json_num(f, s->max);    fputc(',', f);
    fputs("\"stddev_ms\":", f); json_num(f, s->stddev); fputc(',', f);
    fputs("\"stddev_pct_of_median\":", f); json_num(f, s->stddev_pct_of_median);
    fprintf(f, ",\"valid\":%s,", s->valid ? "true" : "false");
    fputs("\"invalid_reason\":", f);
    if (s->invalid_reason[0]) json_str(f, s->invalid_reason); else fputs("null", f);
    fputs("}", f);
}

static void json_record(FILE *f, const bench_record *r)
{
    fputs("{", f);
    fputs("\"benchmark\":", f);     json_str(f, r->benchmark);      fputc(',', f);
    fputs("\"configuration\":", f); json_str(f, r->configuration);  fputc(',', f);
    fputs("\"units\":", f);         json_str(f, r->units);          fputc(',', f);
    fputs("\"value\":", f);         json_num(f, r->value);          fputc(',', f);
    fprintf(f, "\"warmup_iterations\":%d,", r->warmup);
    fprintf(f, "\"samples_requested\":%d,", r->samples_requested);
    fputs("\"raw_samples_ms\":[", f);
    for (int i = 0; i < r->n_samples; ++i) {
        if (i) fputc(',', f);
        json_num(f, r->samples_ms[i]);
    }
    fputs("],", f);
    fputs("\"statistics\":", f); json_stats(f, &r->stats); fputc(',', f);

    fputs("\"counters\":{", f);
    for (int i = 0; i < r->n_meta_num; ++i) {
        if (i) fputc(',', f);
        json_str(f, r->meta_num[i].key); fputc(':', f);
        json_num(f, r->meta_num[i].value);
    }
    fputs("},", f);

    fputs("\"annotations\":{", f);
    for (int i = 0; i < r->n_meta_str; ++i) {
        if (i) fputc(',', f);
        json_str(f, r->meta_str[i].key); fputc(':', f);
        json_str(f, r->meta_str[i].value);
    }
    fputs("}", f);
    fputs("}", f);
}

extern "C" int bench_write_results_fp(void *fpv,
                                      const bench_record *records,
                                      int n_records)
{
    FILE *f = (FILE *)fpv;
    if (!f || !records || n_records <= 0) return -1;

    char ts[64];
    time_t now = time(NULL);
    struct tm gmt;
#if defined(_WIN32)
    gmtime_s(&gmt, &now);
#else
    gmtime_r(&now, &gmt);
#endif
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", &gmt);

    fputs("{", f);
    fputs("\"stage\":", f);            json_str(f, BENCH_STAGE_ID);           fputc(',', f);
    fputs("\"git_commit\":", f);       json_str(f, bench_git_commit());       fputc(',', f);
    fputs("\"run_timestamp_utc\":", f);json_str(f, ts);                       fputc(',', f);
    fputs("\"build_timestamp\":", f);  json_str(f, bench_build_timestamp());  fputc(',', f);
    fputs("\"device\":", f);           json_str(f, bench_device_name());      fputc(',', f);
    fputs("\"cpu_timer\":", f);        json_str(f, bench_cpu_timer_name());   fputc(',', f);
    fputs("\"gpu_timer\":", f);        json_str(f, "cudaEvent + cudaDeviceSynchronize"); fputc(',', f);
    fputs("\"cxx_compiler\":", f);     json_str(f, bench_cxx_compiler());     fputc(',', f);
    fputs("\"cxx_flags\":", f);        json_str(f, bench_cxx_flags());        fputc(',', f);
    fputs("\"cuda_compiler\":", f);    json_str(f, bench_cuda_compiler());    fputc(',', f);
    fputs("\"cuda_flags\":", f);       json_str(f, bench_cuda_flags());       fputc(',', f);
    fprintf(f, "\"protocol\":{\"min_warmup\":%d,\"min_samples\":%d,\"max_stddev_pct_of_median\":%.1f},",
            BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, (double)BENCH_MAX_STDDEV_PCT);

    fputs("\"records\":[", f);
    for (int i = 0; i < n_records; ++i) {
        if (i) fputc(',', f);
        json_record(f, &records[i]);
    }
    fputs("]}", f);
    fputc('\n', f);
    return 0;
}

extern "C" int bench_write_results(const char *results_dir,
                                   const char *file_stem,
                                   const bench_record *records,
                                   int n_records)
{
    char dir[1024];
    if (results_dir && *results_dir) snprintf(dir, sizeof dir, "%s", results_dir);
    else                             bench_default_results_dir(dir, sizeof dir);

    char path[1200];
    snprintf(path, sizeof path, "%s/%s.json", dir, file_stem);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "bench_write_results: cannot open %s\n", path);
        return -1;
    }
    int rc = bench_write_results_fp(f, records, n_records);
    fclose(f);
    if (rc == 0) printf("results written: %s\n", path);
    return rc;
}
