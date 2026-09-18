/* Unit test for microbenchmark 4. */
#include "test_util.h"
#include "microbench.h"

int main(void)
{
    printf("test_cpu_simd_peak\n");
    tie_redirect_results();

    mb_simd_peak_result r;
    int rc = mb_cpu_simd_peak_run(BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, &r);
    CHECK(rc == 0, "the benchmark runs to completion");
    if (rc != 0) TIE_SUMMARY("test_cpu_simd_peak");

    /* --- the ISA compiled for is the widest the CPU reports ----------- */
    const char *widest = mb_cpu_widest_isa();
    printf("       CPUID widest: %s, compiled for: %s\n", widest, r.isa_compiled);
    CHECK(strcmp(r.isa_compiled, r.isa_widest_supported) == 0,
          "the object was compiled for the widest ISA this CPU reports "
          "(compiled %s, CPU reports %s)", r.isa_compiled, r.isa_widest_supported);
    CHECK(strcmp(r.isa_widest_supported, widest) == 0,
          "the recorded widest ISA matches a fresh CPUID query");
    CHECK(r.vector_width_bits >= 128,
          "the vector width is recorded (%d bits)", r.vector_width_bits);

    /* --- the figures are real numbers --------------------------------- */
    CHECK(isfinite(r.vector_gflops), "vector throughput is finite");
    CHECK(r.vector_gflops > 0.0, "vector throughput is strictly positive");
    CHECK(isfinite(r.scalar_gflops), "scalar throughput is finite");
    CHECK(r.scalar_gflops > 0.0, "scalar throughput is strictly positive");

    /* --- vectorisation actually happened ------------------------------
     * If the compiler had silently dropped the vector form, the two loops
     * would run at the same rate. They do not. */
    CHECK(r.scalar_gflops < r.vector_gflops,
          "the scalar reference (%.2f GFLOP/s) is slower than the vectorised run "
          "(%.2f GFLOP/s), which is the evidence that vectorisation occurred",
          r.scalar_gflops, r.vector_gflops);
    CHECK(r.vector_gflops / r.scalar_gflops > 1.5,
          "the speedup is %.2fx, well beyond measurement noise",
          r.vector_gflops / r.scalar_gflops);

    /* --- Stage 0b: the two paths compute the same arithmetic -----------
     *
     * The speedup above shows the vector form was not silently dropped. It does
     * not show the two loops are doing the same work, and a "faster" loop doing
     * different arithmetic would be a measurement of nothing. Each path now
     * carries its result out. Every vector lane starts from the same value as
     * the matching scalar chain and runs the same trip count, so in exact
     * arithmetic the vector checksum is MB_SIMD_LANES times the scalar one; in
     * float they differ only by the order of the final reduction. */
    CHECK(r.lanes >= 4, "the lane count is recorded (%d)", r.lanes);
    CHECK(isfinite(r.vector_checksum) && isfinite(r.scalar_checksum),
          "both checksums are finite (vector %.6f, scalar %.6f)",
          r.vector_checksum, r.scalar_checksum);
    CHECK(r.scalar_checksum != 0.0, "the scalar path produced a non-zero result");
    {
        double per_lane = r.vector_checksum / (double)r.lanes;
        double rel = fabs(per_lane - r.scalar_checksum) /
                     (fabs(r.scalar_checksum) > 0.0 ? fabs(r.scalar_checksum) : 1.0);
        printf("       vector checksum %.6f over %d lanes = %.6f per lane; "
               "scalar %.6f\n", r.vector_checksum, r.lanes, per_lane,
               r.scalar_checksum);
        CHECK(rel < 1e-4,
              "the vectorised and scalar paths agree on the same input to within "
              "float tolerance: relative difference %.3g", rel);
    }

    /* --- the vectorised path was not optimised away --------------------
     * A loop the compiler deleted produces its starting accumulators unchanged.
     * The starting chains are 1..MB_SIMD_ACC, summing to 36 per lane for 8
     * chains; two million multiply-adds move that a long way from 36. */
    CHECK(fabs(r.vector_checksum) > 0.0,
          "the vectorised loop produced a result rather than being deleted");
    {
        double untouched = 0.0;
        for (int i = 1; i <= 8; ++i) untouched += (double)i;
        untouched *= (double)r.lanes;          /* what a deleted loop would leave */
        CHECK(fabs(r.vector_checksum - untouched) > 1.0,
              "the vector checksum (%.6f) is not the untouched starting value "
              "(%.6f), so the FMA loop actually executed",
              r.vector_checksum, untouched);
    }

    /* --- the flop count is derived from the trip count, not asserted --- */
    CHECK(r.flops_per_iteration > 0.0 &&
          fabs(r.flops_per_iteration -
               2.0e6 * 8.0 * (double)r.lanes * 2.0) < 1.0,
          "flops per iteration (%.0f) is iterations * chains * lanes * 2",
          r.flops_per_iteration);

    /* --- Stage 0b: thread placement is recorded, never assumed --------- */
    printf("       thread placement: %s\n", r.placement.detail);
    CHECK(r.placement.detail[0] != '\0',
          "the run records where the measured thread ran and at what priority");
    CHECK(r.placement.pinned == 0 || r.placement.pinned == 1,
          "the pinned flag is a verdict, not a guess (%d)", r.placement.pinned);

    CHECK(r.warmup >= BENCH_MIN_WARMUP && r.samples >= BENCH_MIN_SAMPLES,
          "warmup and sample counts meet the protocol minimums");

    remove("cpu_simd_peak.json");
    TIE_SUMMARY("test_cpu_simd_peak");
}
