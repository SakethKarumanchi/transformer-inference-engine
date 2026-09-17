/* Unit test for microbenchmark 8.
 *
 * Deliberately does NOT compare shared bandwidth against measured global
 * bandwidth: unit tests run in the offline gate before any timed measurement
 * exists, so that figure is unavailable here. bench/microbench/run_all.py makes
 * that comparison after both benchmarks have run and reports it as a finding.
 */
#include "test_util.h"
#include "microbench.h"

#include <cuda_runtime.h>

int main(void)
{
    printf("test_shared_mem_bandwidth\n");
    tie_redirect_results();

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) { printf("  FAIL no CUDA device\n"); return 1; }

    mb_shmem_result r;
    int rc = mb_shared_mem_bandwidth_run(BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, &r);
    CHECK(rc == 0, "the benchmark runs to completion");
    if (rc != 0) TIE_SUMMARY("test_shared_mem_bandwidth");

    /* --- the working set fits the queried shared memory per block ----- */
    CHECK(r.queried_shared_per_block == dev.shared_per_block,
          "the benchmark used the queried shared memory per block (%zu B)",
          dev.shared_per_block);
    CHECK(r.shared_bytes_per_block > 0,
          "the working set is recorded (%zu B)", r.shared_bytes_per_block);
    CHECK(r.shared_bytes_per_block <= dev.shared_per_block,
          "the working set (%zu B) fits within the queried limit (%zu B)",
          r.shared_bytes_per_block, dev.shared_per_block);

    /* --- the access pattern is stated --------------------------------- */
    CHECK(strlen(r.access_pattern) > 0, "the access pattern is stated in the result");
    CHECK(strstr(r.access_pattern, "bank") != NULL,
          "the stated pattern describes the bank mapping: %s", r.access_pattern);
    CHECK(r.conflict_stride == 32,
          "the conflicting variant strides by %d words, one full bank cycle",
          r.conflict_stride);

    /* --- both variants measured, both real numbers -------------------- */
    CHECK(isfinite(r.conflict_free_gb_s), "the conflict-free figure is finite");
    CHECK(r.conflict_free_gb_s > 0.0,
          "the conflict-free figure is strictly positive (%.2f GB/s)",
          r.conflict_free_gb_s);
    CHECK(isfinite(r.conflicting_gb_s), "the conflicting figure is finite");
    CHECK(r.conflicting_gb_s > 0.0,
          "the conflicting figure is strictly positive (%.2f GB/s)", r.conflicting_gb_s);

    /* --- the conflict-free pattern really is the conflict-free one ----
     * If the two patterns compiled to the same access behaviour, they would
     * run at the same rate. They do not. */
    CHECK(r.conflicting_gb_s < r.conflict_free_gb_s,
          "the %d-way conflicting variant (%.2f GB/s) is slower than the "
          "conflict-free one (%.2f GB/s)",
          r.conflict_stride, r.conflicting_gb_s, r.conflict_free_gb_s);
    CHECK(r.conflict_free_stats.median != r.conflicting_stats.median,
          "the two variants carry separate statistics");

    CHECK(r.warmup >= BENCH_MIN_WARMUP && r.samples >= BENCH_MIN_SAMPLES,
          "warmup and sample counts meet the protocol minimums");

    remove("shared_mem_bandwidth.json");
    TIE_SUMMARY("test_shared_mem_bandwidth");
}
