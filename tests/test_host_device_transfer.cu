/* Unit test for microbenchmark 5. */
#include "test_util.h"
#include "microbench.h"

#include <cuda_runtime.h>

/* A local empty kernel so the test can measure launch overhead for itself,
 * without depending on microbenchmark 7 having run. */
__global__ void tst_empty_kernel(void) { }

static double measure_launch_overhead_us(void)
{
    const int n = 2000;
    for (int i = 0; i < 200; ++i) { tst_empty_kernel<<<1, 1>>>(); }
    cudaDeviceSynchronize();
    double t0 = bench_cpu_time_seconds();
    for (int i = 0; i < n; ++i) { tst_empty_kernel<<<1, 1>>>(); cudaDeviceSynchronize(); }
    double t1 = bench_cpu_time_seconds();
    return (t1 - t0) * 1.0e6 / (double)n;
}

int main(void)
{
    printf("test_host_device_transfer\n");
    tie_redirect_results();

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) { printf("  FAIL no CUDA device\n"); return 1; }

    double launch_us = measure_launch_overhead_us();
    printf("       locally measured launch overhead: %.3f us\n", launch_us);

    mb_transfer_result r;
    int rc = mb_host_device_transfer_run(BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, 0, &r);
    CHECK(rc == 0, "the benchmark runs to completion");
    if (rc != 0) TIE_SUMMARY("test_host_device_transfer");

    /* --- all four figures are real numbers ---------------------------- */
    const double v[4] = { r.pinned_h2d_gb_s, r.pinned_d2h_gb_s,
                          r.pageable_h2d_gb_s, r.pageable_d2h_gb_s };
    const char *n[4]  = { "pinned H2D", "pinned D2H", "pageable H2D", "pageable D2H" };
    for (int i = 0; i < 4; ++i) {
        CHECK(isfinite(v[i]), "%s is finite", n[i]);
        CHECK(v[i] > 0.0, "%s is strictly positive (%.3f GB/s)", n[i], v[i]);
    }

    /* --- pinned and pageable are distinct values, never merged -------- */
    CHECK(r.pinned_h2d_gb_s != r.pageable_h2d_gb_s,
          "pinned and pageable host-to-device are reported as distinct values");
    CHECK(r.pinned_d2h_gb_s != r.pageable_d2h_gb_s,
          "pinned and pageable device-to-host are reported as distinct values");
    CHECK(r.pinned_h2d_gb_s != r.pinned_d2h_gb_s,
          "the two directions are reported as distinct values");
    {
        /* Four separate statistics blocks, not one shared one. */
        int distinct = (r.pinned_h2d.median   != r.pageable_h2d.median) &&
                       (r.pinned_d2h.median   != r.pageable_d2h.median);
        CHECK(distinct, "each of the four configurations carries its own statistics");
    }

    /* --- the transfer is long enough that launch overhead is irrelevant --- */
    {
        double shortest_us = r.measured_median_ms_min * 1.0e3;
        CHECK(shortest_us > launch_us * 100.0,
              "the shortest measured transfer (%.1f us) exceeds kernel launch "
              "overhead (%.3f us) by more than 100x", shortest_us, launch_us);
        CHECK(r.transfer_bytes >= 64u * 1024u * 1024u,
              "the transfer size is %zu B, large enough for the figure to be "
              "bandwidth rather than fixed overhead", r.transfer_bytes);
    }

    CHECK(r.warmup >= BENCH_MIN_WARMUP && r.samples >= BENCH_MIN_SAMPLES,
          "warmup and sample counts meet the protocol minimums");

    remove("host_device_transfer.json");
    TIE_SUMMARY("test_host_device_transfer");
}
