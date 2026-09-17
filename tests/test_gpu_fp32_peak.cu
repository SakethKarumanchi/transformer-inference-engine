/* Unit test for microbenchmark 2.
 *
 * The "no memory traffic in the inner loop" property is asserted by reading
 * the PTX the build emitted, not by trusting the source comment. */
#include "test_util.h"
#include "microbench.h"

#include <cuda_runtime.h>

#ifndef TIE_PTX_DIR
#  define TIE_PTX_DIR "."
#endif

/* Returns the slice of the PTX belonging to the named kernel. The caller owns
 * nothing; the pointers index into text. */
static int ptx_kernel_span(const char *text, const char *kernel,
                           const char **begin, const char **end)
{
    const char *k = strstr(text, kernel);
    if (!k) return 0;
    const char *open = strchr(k, '{');
    if (!open) return 0;
    int depth = 0;
    const char *p = open;
    for (; *p; ++p) {
        if (*p == '{') ++depth;
        else if (*p == '}') { if (--depth == 0) { ++p; break; } }
    }
    *begin = open;
    *end   = p;
    return 1;
}

static int count_in_span(const char *begin, const char *end, const char *needle)
{
    int c = 0;
    size_t n = strlen(needle);
    for (const char *p = begin; p + n <= end; ++p)
        if (strncmp(p, needle, n) == 0) ++c;
    return c;
}

int main(void)
{
    printf("test_gpu_fp32_peak\n");
    tie_redirect_results();

    /* --- structural: the inner loop performs no global memory traffic --- */
    {
        char path[1024];
        snprintf(path, sizeof path, "%s/gpu_fp32_peak.ptx", TIE_PTX_DIR);
        char *ptx = tie_read_file(path);
        CHECK(ptx != NULL, "the build emitted PTX for the FP32 peak kernel (%s)", path);
        if (ptx) {
            const char *b = NULL, *e = NULL;
            CHECK(ptx_kernel_span(ptx, "mb_fp32_peak_kernel", &b, &e),
                  "the FP32 peak kernel is present in the PTX");
            if (b && e) {
                int lds = count_in_span(b, e, "ld.global");
                int sts = count_in_span(b, e, "st.global");
                CHECK(lds == 0,
                      "the kernel performs no global loads at all (found %d)", lds);
                CHECK(sts <= 1,
                      "the kernel performs at most the one dead-code-defeating "
                      "global store (found %d)", sts);

                /* The store must sit after the loop-closing backward branch,
                 * so it cannot be inside the timed inner loop. */
                const char *last_bra = NULL;
                for (const char *p = b; p + 4 <= e; ++p)
                    if (strncmp(p, "bra", 3) == 0) last_bra = p;
                const char *st = NULL;
                for (const char *p = b; p + 9 <= e; ++p)
                    if (strncmp(p, "st.global", 9) == 0) { st = p; break; }
                if (sts == 1) {
                    CHECK(last_bra != NULL && st > last_bra,
                          "the single global store follows the last branch, so it is "
                          "outside the loop body");
                } else {
                    CHECK(1, "no global store to place relative to the loop");
                }
            }
            free(ptx);
        }
    }

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) {
        printf("  FAIL no CUDA device\n");
        return 1;
    }

    mb_fp32_peak_result r;
    int rc = mb_gpu_fp32_peak_run(BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, &r);
    CHECK(rc == 0, "the benchmark runs to completion");
    if (rc != 0) TIE_SUMMARY("test_gpu_fp32_peak");

    CHECK(isfinite(r.gflops), "reported throughput is finite");
    CHECK(r.gflops > 0.0, "reported throughput is strictly positive");

    /* Theoretical peak recomputed here from the queried SM count and clock and
     * the [spec] cores-per-SM figure. */
    double theo = (double)dev.sm_count * (double)dev.cores_per_sm_spec * 2.0 *
                  ((double)(dev.max_sm_clock_khz ? dev.max_sm_clock_khz : dev.clock_khz) * 1.0e3)
                  / 1.0e9;
    CHECK_NEAR(r.theoretical_gflops, theo, 1e-6,
               "theoretical peak matches the independent recomputation");
    CHECK(r.gflops <= theo,
          "measured %.2f GFLOP/s does not exceed the theoretical peak %.2f GFLOP/s",
          r.gflops, theo);

    /* The FLOP count must come from the loop trip count, not a constant. */
    double expected_flops = (double)r.threads * (double)r.inner_iterations *
                            (double)r.accumulators * 2.0;
    CHECK_NEAR(r.flops_per_iteration, expected_flops, 1.0,
               "the FLOP count equals threads * iterations * accumulators * 2, "
               "derived from the loop trip count rather than hardcoded");
    CHECK(r.threads == dev.sm_count * (dev.max_threads_per_sm / 256) * 256,
          "the launch geometry is derived from the queried SM count and "
          "maximum resident threads per SM (%d threads)", r.threads);
    CHECK(r.inner_iterations > 0 && r.accumulators > 0,
          "the trip count and chain count are both recorded");

    CHECK(r.warmup >= BENCH_MIN_WARMUP && r.samples >= BENCH_MIN_SAMPLES,
          "warmup and sample counts meet the protocol minimums");

    remove("gpu_fp32_peak.json");
    TIE_SUMMARY("test_gpu_fp32_peak");
}
