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

    CHECK(r.warmup >= BENCH_MIN_WARMUP && r.samples >= BENCH_MIN_SAMPLES,
          "warmup and sample counts meet the protocol minimums");

    remove("cpu_simd_peak.json");
    TIE_SUMMARY("test_cpu_simd_peak");
}
