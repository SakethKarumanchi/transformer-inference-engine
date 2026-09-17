/* Unit test for microbenchmark 1. Structural: it checks the benchmark is
 * correctly constructed, not that this machine is fast. */
#include "test_util.h"
#include "microbench.h"

#include <cuda_runtime.h>

int main(void)
{
    printf("test_gpu_bandwidth\n");
    tie_redirect_results();

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) {
        printf("  FAIL no CUDA device; the benchmark cannot be validated\n");
        return 1;
    }

    mb_bandwidth_result r;
    int rc = mb_gpu_bandwidth_run(BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, 0, &r);
    CHECK(rc == 0, "the benchmark runs to completion");
    if (rc != 0) TIE_SUMMARY("test_gpu_bandwidth");

    /* --- the reported figure is a real number ------------------------- */
    CHECK(isfinite(r.gb_per_s), "reported bandwidth is finite");
    CHECK(r.gb_per_s > 0.0, "reported bandwidth is strictly positive");

    /* --- and does not exceed what the hardware can do. A result above the
     *     theoretical peak indicates a timing or byte-counting error, not a
     *     fast machine. ---------------------------------------------------- */
    double theo = 2.0 * ((double)dev.mem_clock_khz * 1.0e3) *
                  ((double)dev.bus_width_bits / 8.0) / 1.0e9;
    CHECK_NEAR(r.theoretical_peak_gb_per_s, theo, 1e-6,
               "theoretical peak is recomputed here from the queried bus width "
               "and memory clock and matches what the benchmark reported");
    double ceiling = (dev.theoretical_dram_gb_s_at_max_clock > theo)
                   ? dev.theoretical_dram_gb_s_at_max_clock : theo;
    CHECK(r.gb_per_s <= ceiling,
          "measured %.3f GB/s does not exceed the theoretical ceiling %.3f GB/s",
          r.gb_per_s, ceiling);

    /* --- the working set really is beyond L2 -------------------------- */
    CHECK(r.l2_bytes == dev.l2_bytes,
          "the benchmark used the queried L2 size (%zu B)", dev.l2_bytes);
    CHECK(r.working_set_bytes > dev.l2_bytes,
          "the working set (%zu B) exceeds the queried L2 size (%zu B)",
          r.working_set_bytes, dev.l2_bytes);
    CHECK(r.working_set_bytes > dev.l2_bytes * 16,
          "the working set exceeds L2 by more than a factor of 16, so the copy "
          "is not partly cache resident");
    CHECK(r.bytes_moved_per_iter == r.working_set_bytes,
          "bytes counted per iteration equal one read plus one write of each buffer");

    /* --- protocol minimums ------------------------------------------- */
    CHECK(r.warmup >= BENCH_MIN_WARMUP,
          "warmup count %d meets the protocol minimum of %d", r.warmup, BENCH_MIN_WARMUP);
    CHECK(r.samples >= BENCH_MIN_SAMPLES,
          "sample count %d meets the protocol minimum of %d", r.samples, BENCH_MIN_SAMPLES);
    CHECK(r.stats.n == r.samples, "the statistics cover exactly the timed samples");

    remove("gpu_bandwidth.json");
    TIE_SUMMARY("test_gpu_bandwidth");
}
