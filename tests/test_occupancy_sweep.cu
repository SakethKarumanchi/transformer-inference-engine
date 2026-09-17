/* Unit test for microbenchmark 9. */
#include "test_util.h"
#include "microbench.h"

#include <cuda_runtime.h>

int main(void)
{
    printf("test_occupancy_sweep\n");
    tie_redirect_results();

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) { printf("  FAIL no CUDA device\n"); return 1; }

    mb_occ_result r;
    int rc = mb_occupancy_sweep_run(BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, &r);
    CHECK(rc == 0, "the benchmark runs to completion");
    if (rc != 0) TIE_SUMMARY("test_occupancy_sweep");

    CHECK(r.n_points > 0, "the sweep produced %d configurations", r.n_points);
    CHECK(r.warp_size == dev.warp_size &&
          r.max_threads_per_block == dev.max_threads_per_block &&
          r.max_threads_per_sm == dev.max_threads_per_sm &&
          r.regs_per_sm == dev.regs_per_sm,
          "the sweep recorded the queried device properties it swept against");

    /* --- block sizes are legal ---------------------------------------- */
    {
        int bad_multiple = 0, too_large = 0;
        for (int i = 0; i < r.n_points; ++i) {
            if (r.points[i].block_size % dev.warp_size != 0) ++bad_multiple;
            if (r.points[i].block_size > dev.max_threads_per_block) ++too_large;
        }
        CHECK(bad_multiple == 0,
              "every swept block size is a multiple of the queried warp size (%d)",
              dev.warp_size);
        CHECK(too_large == 0,
              "no swept block size exceeds the queried maximum threads per block (%d)",
              dev.max_threads_per_block);
    }

    /* --- theoretical occupancy is computed, not hardcoded -------------
     * Recompute it here from the queried properties and the occupancy API's
     * block count, and require an exact match. */
    {
        int mismatches = 0;
        for (int i = 0; i < r.n_points; ++i) {
            const mb_occ_point *p = &r.points[i];
            double expect = 100.0 * (double)(p->blocks_per_sm_theoretical * p->block_size)
                          / (double)dev.max_threads_per_sm;
            if (fabs(p->theoretical_occupancy_pct - expect) > 1e-9) ++mismatches;
            if (p->theoretical_occupancy_pct > 100.0 + 1e-9) ++mismatches;
        }
        CHECK(mismatches == 0,
              "every theoretical occupancy equals blocksPerSM * blockSize / "
              "maxThreadsPerSM * 100, computed from queried properties");
    }

    /* --- achieved occupancy is not assumed equal to theoretical ------- */
    {
        int assumed = 0, pending = 0;
        for (int i = 0; i < r.n_points; ++i) {
            if (r.points[i].achieved_occupancy_pct < 0.0) ++pending;
            else if (fabs(r.points[i].achieved_occupancy_pct -
                          r.points[i].theoretical_occupancy_pct) < 1e-12) ++assumed;
        }
        CHECK(pending == r.n_points,
              "achieved occupancy is left unset (%d of %d points) for the profiler "
              "to fill in; it is never copied from theoretical", pending, r.n_points);
        CHECK(assumed == 0, "no point has achieved occupancy silently set to theoretical");
    }

    /* --- register pressure actually varies, and the sweep says where -- */
    {
        int min_regs = 1 << 30, max_regs = 0;
        for (int i = 0; i < r.n_points; ++i) {
            if (r.points[i].regs_per_thread < min_regs) min_regs = r.points[i].regs_per_thread;
            if (r.points[i].regs_per_thread > max_regs) max_regs = r.points[i].regs_per_thread;
        }
        CHECK(max_regs > min_regs,
              "the sweep spans a real range of register pressure (%d to %d regs/thread)",
              min_regs, max_regs);
        CHECK(r.n_register_limited >= 1,
              "at least one configuration is register-limited (%d found)",
              r.n_register_limited);

        int named = 0;
        for (int i = 0; i < r.n_points; ++i)
            if (r.points[i].limiter == MB_OCC_LIMIT_REGISTERS) {
                ++named;
                printf("       register-limited: acc=%d block=%d regs=%d blocks/SM=%d "
                       "theoretical %.2f%%\n",
                       r.points[i].accumulators, r.points[i].block_size,
                       r.points[i].regs_per_thread,
                       r.points[i].blocks_per_sm_theoretical,
                       r.points[i].theoretical_occupancy_pct);
            }
        CHECK(named == r.n_register_limited,
              "the sweep names which configurations are register-limited");
    }

    /* --- non-launchable configurations are reported as such ----------- */
    {
        int placeholders = 0;
        for (int i = 0; i < r.n_points; ++i)
            if (r.points[i].median_ms < 0.0 && r.points[i].stats.invalid_reason[0] == '\0')
                ++placeholders;
        CHECK(placeholders == 0,
              "a configuration that could not run carries a stated reason rather "
              "than a placeholder timing");
    }

    CHECK(r.warmup >= BENCH_MIN_WARMUP && r.samples >= BENCH_MIN_SAMPLES,
          "warmup and sample counts meet the protocol minimums");

    remove("occupancy_sweep.json");
    TIE_SUMMARY("test_occupancy_sweep");
}
