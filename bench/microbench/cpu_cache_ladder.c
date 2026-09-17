/* Microbenchmark 3 -- CPU cache ladder.
 *
 * A streaming read swept across working-set sizes from well below L1 to well
 * beyond L3. The plateau edges in the resulting bandwidth curve reveal the
 * effective cache sizes, which often differ from what the OS reports. Both the
 * measured edges and the OS-reported sizes are emitted; the measured ones are
 * what later stages use, but they never silently replace the reported ones.
 *
 * Timed with the platform's highest-resolution monotonic counter via
 * bench_cpu_time_seconds(); never wall clock.
 */
#include "microbench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#if defined(_WIN32)
#  include <intrin.h>
#  include <malloc.h>
#  define MB_ALIGNED_ALLOC(sz, al) _aligned_malloc((sz), (al))
#  define MB_ALIGNED_FREE(p)       _aligned_free(p)
#else
#  include <stdlib.h>
#  define MB_ALIGNED_ALLOC(sz, al) aligned_alloc((al), (sz))
#  define MB_ALIGNED_FREE(p)       free(p)
#endif

#define MB_LADDER_MIN_BYTES   (4u * 1024u)
#define MB_LADDER_MAX_BYTES   (128u * 1024u * 1024u)
/* Bytes read per timed sample, the same at every working-set size. Sized so a
 * sample lasts long enough that scheduler interference averages out instead of
 * landing on one short sample and skewing it. */
#define MB_LADDER_TARGET_WORK (256u * 1024u * 1024u)
#define MB_EDGE_DROP_PCT      12.0                   /* relative drop marking an edge */

/* ---- OS / CPUID reported cache sizes ---------------------------------- */
int mb_cpu_os_cache_sizes(size_t *l1d, size_t *l2, size_t *l3)
{
    if (l1d) *l1d = 0;
    if (l2)  *l2  = 0;
    if (l3)  *l3  = 0;
#if defined(_WIN32) || defined(__x86_64__) || defined(_M_X64)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 4) return 1;
    for (int i = 0; i < 16; ++i) {
        __cpuidex(r, 4, i);
        int type = r[0] & 0x1f;
        if (type == 0) break;                    /* no more cache levels */
        int level = (r[0] >> 5) & 0x7;
        long long ways  = ((r[1] >> 22) & 0x3ff) + 1;
        long long parts = ((r[1] >> 12) & 0x3ff) + 1;
        long long line  = (r[1] & 0xfff) + 1;
        long long sets  = (long long)(unsigned)r[2] + 1;
        size_t sz = (size_t)(ways * parts * line * sets);
        if (level == 1 && type == 1 && l1d) *l1d = sz;   /* data cache only */
        else if (level == 2 && l2) *l2 = sz;
        else if (level == 3 && l3) *l3 = sz;
    }
    return 0;
#else
    return 1;
#endif
}

/* ---- the streaming read ------------------------------------------------ */
typedef struct {
    unsigned *buf;
    size_t    n_words;
    int       passes;
    unsigned  sink;
} ladder_ctx;

/* Two properties this loop has to have, both learned the hard way:
 *
 *  - The read must be wide enough that the loop is limited by the memory
 *    hierarchy rather than by scalar issue rate. A scalar version tops out
 *    around 15 GB/s on this class of CPU even when the working set fits in L1,
 *    which flattens the very plateaus the ladder exists to find.
 *
 *  - The loop shape must not change with the working-set size. An outer
 *    "repeat this buffer N times" loop charges the small sizes a per-pass
 *    setup cost that the large sizes do not pay, and the curve then rises from
 *    4 KiB to 16 KiB -- an artefact of the harness, read as if it were cache
 *    behaviour. Instead a single flat loop performs a fixed number of vector
 *    loads for every size, walking a plain forward pointer that is reset when
 *    it reaches the end of the working set.
 */
static double ladder_body(void *vctx, int iteration)
{
    (void)iteration;
    ladder_ctx *c = (ladder_ctx *)vctx;
    unsigned tail = 0;
    const size_t total_vecs = (size_t)MB_LADDER_TARGET_WORK / 32;   /* 32 B per vector */
    double t0 = bench_cpu_time_seconds();
#if defined(__AVX2__)
    {
        const __m256i *b   = (const __m256i *)c->buf;
        const size_t   nv  = c->n_words / 8;     /* 8 unsigned per 256-bit vector */
        const __m256i *end = b + nv;
        const __m256i *p   = b;
        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
        __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
        /* A plainly increasing pointer, reset at the end of the working set.
         * The hardware prefetcher sees a clean forward stream; an AND-masked
         * index instead of the reset costs measurable bandwidth at 4 KiB,
         * where the wrap aliases. */
        for (size_t done = 0; done < total_vecs; done += 4) {
            a0 = _mm256_add_epi32(a0, _mm256_load_si256(p + 0));
            a1 = _mm256_add_epi32(a1, _mm256_load_si256(p + 1));
            a2 = _mm256_add_epi32(a2, _mm256_load_si256(p + 2));
            a3 = _mm256_add_epi32(a3, _mm256_load_si256(p + 3));
            p += 4;
            if (p == end) p = b;
        }
        double t1 = bench_cpu_time_seconds();
        __m256i s = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
        unsigned lanes[8];
        _mm256_storeu_si256((__m256i *)lanes, s);
        for (int k = 0; k < 8; ++k) tail += lanes[k];
        c->sink += tail;
        return (t1 - t0) * 1.0e3;
    }
#else
    {
        const unsigned *b = c->buf, *end = b + c->n_words, *p = b;
        unsigned s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (size_t done = 0; done < total_vecs * 8; done += 4) {
            s0 += p[0]; s1 += p[1]; s2 += p[2]; s3 += p[3];
            p += 4;
            if (p == end) p = b;
        }
        double t1 = bench_cpu_time_seconds();
        tail += s0 + s1 + s2 + s3;
        c->sink += tail;
        return (t1 - t0) * 1.0e3;
    }
#endif
}

int mb_cpu_cache_ladder_run(int warmup, int samples, mb_cache_ladder_result *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);
    mb_cpu_os_cache_sizes(&out->os_l1d_bytes, &out->os_l2_bytes, &out->os_l3_bytes);
    out->bytes_per_sample = (size_t)MB_LADDER_TARGET_WORK;

    unsigned *buf = (unsigned *)MB_ALIGNED_ALLOC(MB_LADDER_MAX_BYTES, 64);
    if (!buf) { fprintf(stderr, "cpu_cache_ladder: allocation failed\n"); return 1; }
    for (size_t i = 0; i < MB_LADDER_MAX_BYTES / sizeof(unsigned); ++i)
        buf[i] = (unsigned)i;

    int n_alloc = samples > BENCH_MIN_SAMPLES ? samples : BENCH_MIN_SAMPLES;
    double *raw = (double *)malloc((size_t)n_alloc * sizeof(double));
    if (!raw) { MB_ALIGNED_FREE(buf); return 1; }

    bench_record  recs[MB_LADDER_MAX_POINTS];
    double       *raws[MB_LADDER_MAX_POINTS];
    char          cfgs[MB_LADDER_MAX_POINTS][BENCH_LABEL_LEN];
    memset(recs, 0, sizeof recs);

    int eff_w = 0, eff_s = 0, np = 0;
    for (size_t bytes = MB_LADDER_MIN_BYTES;
         bytes <= MB_LADDER_MAX_BYTES && np < MB_LADDER_MAX_POINTS;
         bytes *= 2) {

        ladder_ctx c;
        c.buf     = buf;
        c.n_words = bytes / sizeof(unsigned);
        c.passes  = (int)(MB_LADDER_TARGET_WORK / bytes);   /* recorded, not used to time */
        c.sink    = 0;

        bench_stats st = bench_run(ladder_body, &c, warmup, samples, raw, &eff_w, &eff_s);

        /* Every point reads the same number of bytes, so the loop shape is
         * identical across the ladder and only the working-set size varies. */
        double bytes_touched = (double)MB_LADDER_TARGET_WORK;
        double gbs = (st.median > 0.0) ? bytes_touched / (st.median * 1.0e-3) / 1.0e9 : 0.0;

        out->sizes[np]    = bytes;
        out->gb_per_s[np] = gbs;
        out->stats[np]    = st;

        raws[np] = (double *)malloc((size_t)eff_s * sizeof(double));
        if (!raws[np]) { free(raw); MB_ALIGNED_FREE(buf); return 1; }
        memcpy(raws[np], raw, (size_t)eff_s * sizeof(double));

        snprintf(cfgs[np], BENCH_LABEL_LEN,
                 "working_set=%zu B, %u B read per sample", bytes,
                 (unsigned)MB_LADDER_TARGET_WORK);
        recs[np].benchmark         = "cpu_cache_ladder";
        recs[np].configuration     = cfgs[np];
        recs[np].units             = "GB/s";
        recs[np].value             = gbs;
        recs[np].warmup            = eff_w;
        recs[np].samples_requested = eff_s;
        recs[np].samples_ms        = raws[np];
        recs[np].n_samples         = eff_s;
        recs[np].stats             = st;
        ++np;
    }
    out->n_points = np;
    out->warmup   = eff_w;
    out->samples  = eff_s;

    /* Plateau edges: the last size before a relative bandwidth drop larger
     * than MB_EDGE_DROP_PCT. Reported, not used to overwrite the OS figures. */
    for (int i = 0; i + 1 < np && out->n_edges < MB_LADDER_MAX_EDGES; ++i) {
        if (out->gb_per_s[i] <= 0.0) continue;
        double drop = 100.0 * (out->gb_per_s[i] - out->gb_per_s[i + 1]) / out->gb_per_s[i];
        if (drop > MB_EDGE_DROP_PCT) {
            out->edge_below[out->n_edges]    = out->sizes[i];
            out->edge_drop_pct[out->n_edges] = drop;
            ++out->n_edges;
        }
    }

    {
        bench_kv_num nums[3 + MB_LADDER_MAX_EDGES];
        int nn = 0;
        nums[nn].key = "os_l1d_bytes"; nums[nn++].value = (double)out->os_l1d_bytes;
        nums[nn].key = "os_l2_bytes";  nums[nn++].value = (double)out->os_l2_bytes;
        nums[nn].key = "os_l3_bytes";  nums[nn++].value = (double)out->os_l3_bytes;
        static const char *edge_keys[MB_LADDER_MAX_EDGES] = {
            "measured_edge_below_bytes_0", "measured_edge_below_bytes_1",
            "measured_edge_below_bytes_2", "measured_edge_below_bytes_3",
            "measured_edge_below_bytes_4", "measured_edge_below_bytes_5",
            "measured_edge_below_bytes_6", "measured_edge_below_bytes_7" };
        for (int e = 0; e < out->n_edges; ++e) {
            nums[nn].key = edge_keys[e];
            nums[nn++].value = (double)out->edge_below[e];
        }
        for (int i = 0; i < np; ++i) { recs[i].meta_num = nums; recs[i].n_meta_num = nn; }
        bench_write_results(NULL, "cpu_cache_ladder", recs, np);
    }

    for (int i = 0; i < np; ++i) free(raws[i]);
    free(raw);
    MB_ALIGNED_FREE(buf);
    return 0;
}

#ifndef BENCH_NO_MAIN
int main(int argc, char **argv)
{
    int warmup  = (argc > 1) ? atoi(argv[1]) : BENCH_MIN_WARMUP;
    int samples = (argc > 2) ? atoi(argv[2]) : BENCH_MIN_SAMPLES;

    mb_cache_ladder_result r;
    if (mb_cpu_cache_ladder_run(warmup, samples, &r) != 0) {
        fprintf(stderr, "cpu_cache_ladder: FAILED\n");
        return 1;
    }
    printf("cpu_cache_ladder  timer=%s\n", bench_cpu_timer_name());
    printf("  OS-reported: L1d %zu B, L2 %zu B, L3 %zu B\n",
           r.os_l1d_bytes, r.os_l2_bytes, r.os_l3_bytes);
    int all_valid = 1;
    for (int i = 0; i < r.n_points; ++i) {
        printf("  %10zu B  %8.2f GB/s  stddev %6.3f%%  %s\n",
               r.sizes[i], r.gb_per_s[i], r.stats[i].stddev_pct_of_median,
               r.stats[i].valid ? "VALID" : "INVALID");
        if (!r.stats[i].valid) { all_valid = 0; printf("      INVALID: %s\n", r.stats[i].invalid_reason); }
    }
    printf("  measured plateau edges (%d):\n", r.n_edges);
    for (int e = 0; e < r.n_edges; ++e)
        printf("    below %zu B, drop %.1f%%\n", r.edge_below[e], r.edge_drop_pct[e]);
    return all_valid ? 0 : 2;
}
#endif
