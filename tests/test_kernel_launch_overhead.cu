/* Unit test for microbenchmark 7. */
#include "test_util.h"
#include "microbench.h"

#include <cuda_runtime.h>

#ifndef TIE_PTX_DIR
#  define TIE_PTX_DIR "."
#endif

/* Extracts the body of the named PTX kernel. */
static int ptx_kernel_body(const char *text, const char *kernel,
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
        else if (*p == '}') { if (--depth == 0) break; }
    }
    *begin = open + 1;
    *end   = p;
    return 1;
}

int main(void)
{
    printf("test_kernel_launch_overhead\n");
    tie_redirect_results();

    /* --- structural: the kernel body is genuinely empty ---------------- */
    {
        char path[1024];
        snprintf(path, sizeof path, "%s/kernel_launch_overhead.ptx", TIE_PTX_DIR);
        char *ptx = tie_read_file(path);
        CHECK(ptx != NULL, "the build emitted PTX for the empty kernel (%s)", path);
        if (ptx) {
            const char *b = NULL, *e = NULL;
            CHECK(ptx_kernel_body(ptx, "mb_empty_kernel", &b, &e),
                  "the empty kernel is present in the PTX");
            if (b && e) {
                /* Walk the body counting instructions. Directives (lines
                 * starting '.'), labels and blank lines are not instructions. */
                int instructions = 0, rets = 0;
                const char *line = b;
                char buf[512];
                while (line < e) {
                    const char *nl = (const char *)memchr(line, '\n', (size_t)(e - line));
                    const char *stop = nl ? nl : e;
                    size_t len = (size_t)(stop - line);
                    if (len >= sizeof buf) len = sizeof buf - 1;
                    memcpy(buf, line, len);
                    buf[len] = '\0';
                    char *s = buf;
                    while (*s == ' ' || *s == '\t' || *s == '\r') ++s;
                    size_t sl = strlen(s);
                    while (sl && (s[sl - 1] == ' ' || s[sl - 1] == '\r')) s[--sl] = '\0';
                    if (sl && s[0] != '.' && s[0] != '/' && s[sl - 1] != ':') {
                        ++instructions;
                        if (strncmp(s, "ret", 3) == 0) ++rets;
                        else printf("       unexpected instruction in body: %s\n", s);
                    }
                    line = stop + 1;
                }
                CHECK(rets == 1, "the body contains exactly one ret");
                CHECK(instructions == rets,
                      "the body contains nothing but that ret (%d instructions)",
                      instructions);
            }
            free(ptx);
        }
    }

    mb_launch_result r;
    int rc = mb_kernel_launch_overhead_run(BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, &r);
    CHECK(rc == 0, "the benchmark runs to completion");
    if (rc != 0) TIE_SUMMARY("test_kernel_launch_overhead");

    CHECK(isfinite(r.sync_per_launch_us), "the sync-per-launch figure is finite");
    CHECK(r.sync_per_launch_us > 0.0,
          "the sync-per-launch figure is strictly positive (%.4f us)",
          r.sync_per_launch_us);
    CHECK(isfinite(r.back_to_back_us), "the back-to-back figure is finite");
    CHECK(r.back_to_back_us > 0.0,
          "the back-to-back figure is strictly positive (%.4f us)", r.back_to_back_us);

    /* --- the two cases are reported separately, as the model needs ---- */
    CHECK(r.sync_per_launch_us != r.back_to_back_us,
          "the synchronized-per-launch and back-to-back cases are distinct values");
    CHECK(r.sync_stats.median != r.back_to_back_stats.median,
          "each case carries its own statistics");

    /* --- the synchronize is really there ------------------------------
     * Launch-to-completion must cost more than a pipelined launch. If the
     * measurement had captured enqueue time only, the two would match. */
    CHECK(r.sync_per_launch_us > r.back_to_back_us,
          "launch-to-completion (%.4f us) costs more than an amortised pipelined "
          "launch (%.4f us), which is the evidence the device synchronize is in "
          "the timed region", r.sync_per_launch_us, r.back_to_back_us);

    CHECK(r.batch_size >= 100,
          "each sample batches %d launches, so per-launch resolution is not "
          "limited by the timer", r.batch_size);
    CHECK(r.warmup >= BENCH_MIN_WARMUP && r.samples >= BENCH_MIN_SAMPLES,
          "warmup and sample counts meet the protocol minimums");

    remove("kernel_launch_overhead.json");
    TIE_SUMMARY("test_kernel_launch_overhead");
}
