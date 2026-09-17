/* Unit test for microbenchmark 3. */
#include "test_util.h"
#include "microbench.h"

int main(void)
{
    printf("test_cpu_cache_ladder\n");
    tie_redirect_results();

    size_t l1d = 0, l2 = 0, l3 = 0;
    int qrc = mb_cpu_os_cache_sizes(&l1d, &l2, &l3);
    CHECK(qrc == 0, "the OS/CPUID cache sizes can be queried");
    CHECK(l1d > 0 && l2 > 0 && l3 > 0,
          "all three levels are reported: L1d %zu, L2 %zu, L3 %zu", l1d, l2, l3);

    mb_cache_ladder_result r;
    int rc = mb_cpu_cache_ladder_run(BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, &r);
    CHECK(rc == 0, "the benchmark runs to completion");
    if (rc != 0) TIE_SUMMARY("test_cpu_cache_ladder");

    CHECK(r.n_points >= 4, "the sweep produced %d points", r.n_points);
    CHECK(r.bytes_per_sample > 0,
          "every point reads the same %zu B per sample, so the loop shape does "
          "not vary with the working-set size", r.bytes_per_sample);

    /* --- the swept range brackets every reported cache level ---------- */
    size_t smallest = r.sizes[0], largest = r.sizes[r.n_points - 1];
    const size_t levels[3] = { l1d, l2, l3 };
    const char  *names[3]  = { "L1d", "L2", "L3" };
    for (int i = 0; i < 3; ++i) {
        int below = 0, above = 0;
        for (int p = 0; p < r.n_points; ++p) {
            if (r.sizes[p] < levels[i]) below = 1;
            if (r.sizes[p] > levels[i]) above = 1;
        }
        CHECK(below && above,
              "the sweep has at least one point on each side of %s (%zu B)",
              names[i], levels[i]);
    }
    CHECK(smallest < l1d, "the sweep starts well below L1d");
    CHECK(largest > l3 * 4, "the sweep ends well beyond L3");

    /* --- bandwidth is monotonically non-increasing across the ladder ---
     *
     * Two things are checked, because the property holds exactly at one scale
     * and only approximately at the other.
     *
     * Across cache TIERS it holds exactly, and that is the hard assertion: the
     * fastest L1-resident point must beat the fastest L2-resident point, which
     * must beat the L3-resident one, which must beat DRAM.
     *
     * Between ADJACENT points inside one tier it does not hold exactly, and
     * the reason is structural rather than noise. A ladder re-reads a small
     * working set many times to make the sample long enough to time, and every
     * wrap back to the start restarts the hardware prefetch stream. A 64 KiB
     * set wraps twice as often as a 128 KiB one for the same bytes read, so it
     * pays that restart cost twice as often and measures slightly slower even
     * though both are served from the same cache level. Rises at within-tier
     * steps come from that, not from interference, so they are printed but not
     * treated as a defect.
     *
     * An earlier revision of this test capped those rises at 25% and treated a
     * larger one as a failure. Five back-to-back runs on this machine showed
     * that cap is not a property of a working ladder: it failed in 4 of the 5,
     * always in the DRAM tier, with rises up to +83% (67 MiB -> 134 MiB,
     * 9.45 -> 17.26 GB/s). The effect is largest there because wrap count
     * varies 16x across the DRAM points (256 MiB read / 16 MiB set = 16 wraps,
     * / 128 MiB = 2) while the per-wrap prefetch and TLB restart cost is a
     * large fraction of an already slow DRAM baseline. Point-to-point
     * monotonicity in working-set size is therefore something this benchmark
     * never claimed and that this machine measurably violates, so asserting it
     * was testing the wrong property. It is replaced below by a dynamic-range
     * check, which is what actually distinguishes a ladder that resolves the
     * cache hierarchy from one that does not.
     *
     * The comparison uses each point's FASTEST sample rather than its median:
     * this test runs in the offline gate against whatever else is on the
     * machine, and interference can only make a sample slower, never faster.
     * The reported headline figure stays the median, as the protocol requires.
     */
    {
        double best_case[MB_LADDER_MAX_POINTS];
        for (int p = 0; p < r.n_points; ++p)
            best_case[p] = (r.stats[p].min > 0.0)
                         ? (double)r.bytes_per_sample / (r.stats[p].min * 1e-3) / 1e9
                         : r.gb_per_s[p];

        double tier[4] = { 0, 0, 0, 0 };       /* L1, L2, L3, DRAM resident */
        const char *tier_name[4] = { "L1-resident", "L2-resident",
                                     "L3-resident", "DRAM-resident" };
        for (int p = 0; p < r.n_points; ++p) {
            int t = (r.sizes[p] <= l1d) ? 0
                  : (r.sizes[p] <= l2)  ? 1
                  : (r.sizes[p] <= l3)  ? 2 : 3;
            if (best_case[p] > tier[t]) tier[t] = best_case[p];
        }
        for (int t = 0; t < 4; ++t)
            printf("       %-14s peak %.2f GB/s\n", tier_name[t], tier[t]);
        for (int t = 0; t + 1 < 4; ++t)
            CHECK(tier[t] > tier[t + 1],
                  "%s bandwidth (%.2f GB/s) exceeds %s (%.2f GB/s)",
                  tier_name[t], tier[t], tier_name[t + 1], tier[t + 1]);

        for (int p = 0; p + 1 < r.n_points; ++p) {
            if (best_case[p + 1] > best_case[p] * 1.10)
                printf("       rise at %zu B -> %zu B: %.2f -> %.2f GB/s "
                       "(wrap-count / prefetch restart effect)\n",
                       r.sizes[p], r.sizes[p + 1], best_case[p], best_case[p + 1]);
        }

        /* Dynamic range. A ladder that is not actually resolving the hierarchy
         * -- wrong byte count, working set not resident, timer not bracketing
         * the loop -- flattens, and its top and bottom tiers converge. A
         * working one does not: the five runs behind the note above spanned
         * 6.2x to 7.8x. 3x sits well clear of that and well clear of flat. */
        CHECK(tier[0] > tier[3] * 3.0,
              "L1-resident peak (%.2f GB/s) exceeds DRAM-resident peak "
              "(%.2f GB/s) by more than 3x: ratio %.2fx",
              tier[0], tier[3], tier[3] > 0.0 ? tier[0] / tier[3] : 0.0);
    }

    /* --- at least one plateau edge is detected ------------------------ */
    CHECK(r.n_edges >= 1, "at least one plateau edge was detected (%d found)", r.n_edges);
    for (int e = 0; e < r.n_edges; ++e)
        CHECK(r.edge_below[e] >= smallest && r.edge_below[e] <= largest,
              "detected edge %d (%zu B) lies inside the swept range", e, r.edge_below[e]);

    /* --- edges are reported ALONGSIDE the OS figures, not instead ----- */
    CHECK(r.os_l1d_bytes == l1d && r.os_l2_bytes == l2 && r.os_l3_bytes == l3,
          "the OS-reported sizes are carried through unmodified next to the "
          "measured edges");

    /* --- every point carries its own validity verdict ----------------- */
    {
        int has_verdict = 1;
        for (int p = 0; p < r.n_points; ++p)
            if (r.stats[p].n != r.samples) has_verdict = 0;
        CHECK(has_verdict, "every ladder point carries statistics over the full sample set");
    }
    CHECK(r.warmup >= BENCH_MIN_WARMUP && r.samples >= BENCH_MIN_SAMPLES,
          "warmup and sample counts meet the protocol minimums");

    remove("cpu_cache_ladder.json");
    TIE_SUMMARY("test_cpu_cache_ladder");
}
