/* microbench.h -- API surface of the nine Stage 0 microbenchmarks.
 *
 * Every microbenchmark is a library function plus a thin main(). The unit
 * tests link the library function directly (the sources are compiled a second
 * time with BENCH_NO_MAIN defined), which is why these are reusable tested
 * code rather than throwaway scripts: later stages re-run them to confirm the
 * machine has not drifted, and the Stage 10 model consumes their output.
 *
 * Every run_* function returns 0 on success and non-zero on failure. A failure
 * never leaves a plausible placeholder in *out.
 */
#ifndef MICROBENCH_H
#define MICROBENCH_H

#include <stddef.h>
#include "bench_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- queried device facts shared by the CUDA benchmarks --------------- */
typedef struct {
    char   name[256];
    int    major, minor;
    int    sm_count;
    int    warp_size;
    int    max_threads_per_block;
    int    max_threads_per_sm;
    int    max_blocks_per_sm;
    int    regs_per_sm;
    int    regs_per_block;
    size_t shared_per_block;
    size_t shared_per_sm;
    size_t l2_bytes;
    size_t total_global_bytes;
    int    bus_width_bits;
    int    clock_khz;            /* cudaDevAttrClockRate */
    int    mem_clock_khz;        /* cudaDevAttrMemoryClockRate */
    int    max_sm_clock_khz;     /* NVML max SM clock; 0 if NVML unavailable */
    int    max_mem_clock_khz;    /* NVML max memory clock; 0 if unavailable */
    int    nvml_available;
    int    cores_per_sm_spec;    /* [spec] CC -> cores/SM table, not queried */
    double theoretical_dram_gb_s;    /* [derived] 2 * memclk * bus/8 */
    double theoretical_fp32_gflops;  /* [derived] SMs * cores/SM * 2 * clock */
    /* Ceiling at the highest clock the driver reports this part can reach.
     * A measurement may legitimately exceed the rated-boost figure above when
     * the card opportunistically boosts, so bounds checks use this one. */
    double theoretical_fp32_gflops_at_max_clock;
    double theoretical_dram_gb_s_at_max_clock;
} mb_device_info;

int mb_query_device(mb_device_info *out);

/* ---- 1. GPU bandwidth ------------------------------------------------- */
typedef struct {
    double      gb_per_s;
    double      theoretical_peak_gb_per_s;
    size_t      working_set_bytes;      /* src + dst, must exceed L2 */
    size_t      l2_bytes;
    size_t      bytes_moved_per_iter;   /* read + write */
    int         stride_elements;
    int         warmup, samples;
    bench_stats stats;
} mb_bandwidth_result;
int mb_gpu_bandwidth_run(int warmup, int samples, size_t bytes_per_buffer,
                         mb_bandwidth_result *out);

/* ---- 2. GPU FP32 peak ------------------------------------------------- */
typedef struct {
    double      gflops;
    double      theoretical_gflops;
    double      flops_per_iteration;    /* derived from the loop trip count */
    int         inner_iterations;
    int         accumulators;
    int         threads;
    int         warmup, samples;
    bench_stats stats;
} mb_fp32_peak_result;
int mb_gpu_fp32_peak_run(int warmup, int samples, mb_fp32_peak_result *out);

/* ---- 3. CPU cache ladder ---------------------------------------------- */
#define MB_LADDER_MAX_POINTS 32
#define MB_LADDER_MAX_EDGES   8
typedef struct {
    int         n_points;
    size_t      sizes[MB_LADDER_MAX_POINTS];
    double      gb_per_s[MB_LADDER_MAX_POINTS];
    bench_stats stats[MB_LADDER_MAX_POINTS];
    int         n_edges;
    size_t      edge_below[MB_LADDER_MAX_EDGES];  /* last size before the drop */
    double      edge_drop_pct[MB_LADDER_MAX_EDGES];
    /* reported alongside, never replacing, the measured edges */
    size_t      os_l1d_bytes, os_l2_bytes, os_l3_bytes;
    /* Bytes read per timed sample. The same at every point, so the loop shape
     * does not vary with the working-set size. */
    size_t      bytes_per_sample;
    int         warmup, samples;
} mb_cache_ladder_result;
int mb_cpu_cache_ladder_run(int warmup, int samples, mb_cache_ladder_result *out);
/* Cache sizes as CPUID leaf 4 reports them. Returns 0 on success. */
int mb_cpu_os_cache_sizes(size_t *l1d, size_t *l2, size_t *l3);

/* ---- 4. CPU SIMD peak ------------------------------------------------- */
typedef struct {
    double      vector_gflops;
    double      scalar_gflops;
    char        isa_compiled[32];           /* what this object was built for */
    char        isa_widest_supported[32];   /* what CPUID reports */
    int         vector_width_bits;
    double      flops_per_iteration;
    int         warmup, samples;
    bench_stats vector_stats;
    bench_stats scalar_stats;
} mb_simd_peak_result;
int mb_cpu_simd_peak_run(int warmup, int samples, mb_simd_peak_result *out);
/* Widest SIMD ISA the CPU reports, as a short token ("AVX512F","AVX2","AVX","SSE2"). */
const char *mb_cpu_widest_isa(void);

/* ---- 5. Host/device transfer ------------------------------------------ */
typedef struct {
    double      pinned_h2d_gb_s,  pinned_d2h_gb_s;
    double      pageable_h2d_gb_s, pageable_d2h_gb_s;
    bench_stats pinned_h2d,  pinned_d2h;
    bench_stats pageable_h2d, pageable_d2h;
    size_t      transfer_bytes;
    double      measured_median_ms_min;   /* smallest of the four medians */
    int         warmup, samples;
} mb_transfer_result;
int mb_host_device_transfer_run(int warmup, int samples, size_t transfer_bytes,
                                mb_transfer_result *out);

/* ---- 6. cuBLAS SGEMM reference ---------------------------------------- */
#define MB_GEMM_MAX_POINTS 64
#define MB_GEMM_N_SHAPES    5
typedef struct {
    char        shape_name[40];
    int         M, N, K;
    int         is_decode;          /* 1 when M == 1 */
    double      gflops;
    bench_stats stats;
} mb_gemm_point;
typedef struct {
    int           n_points;
    mb_gemm_point points[MB_GEMM_MAX_POINTS];
    int           n_m_values;
    int           m_values[16];
    char          provenance[256];
    int           warmup, samples;
} mb_gemm_result;
int mb_cublas_sgemm_ref_run(int warmup, int samples, mb_gemm_result *out);
/* The five architecture-derived shapes, N and K only (M is the token count). */
typedef struct { const char *name; int N, K; } mb_gemm_shape;
const mb_gemm_shape *mb_gpt2_small_shapes(int *count);
const char          *mb_gpt2_small_provenance(void);
const int           *mb_gemm_m_sweep(int *count);

/* ---- 7. Kernel launch overhead ---------------------------------------- */
typedef struct {
    double      sync_per_launch_us;   /* launch-to-completion, one sync each */
    double      back_to_back_us;      /* amortised over a batch, one sync */
    int         batch_size;
    bench_stats sync_stats;
    bench_stats back_to_back_stats;
    int         warmup, samples;
} mb_launch_result;
int mb_kernel_launch_overhead_run(int warmup, int samples, mb_launch_result *out);

/* ---- 8. Shared memory bandwidth --------------------------------------- */
typedef struct {
    double      conflict_free_gb_s;
    double      conflicting_gb_s;
    bench_stats conflict_free_stats;
    bench_stats conflicting_stats;
    size_t      shared_bytes_per_block;      /* working set, must fit the queried limit */
    size_t      queried_shared_per_block;
    int         conflict_stride;             /* stride of the conflicting variant */
    char        access_pattern[128];
    int         warmup, samples;
} mb_shmem_result;
int mb_shared_mem_bandwidth_run(int warmup, int samples, mb_shmem_result *out);

/* ---- 9. Occupancy sweep ----------------------------------------------- */
#define MB_OCC_MAX_POINTS 64
enum { MB_OCC_LIMIT_BLOCKS = 0, MB_OCC_LIMIT_WARPS, MB_OCC_LIMIT_REGISTERS,
       MB_OCC_LIMIT_SHARED };
typedef struct {
    int    block_size;
    int    accumulators;              /* register-pressure knob */
    int    regs_per_thread;           /* cudaFuncGetAttributes().numRegs */
    int    blocks_per_sm_theoretical;
    double theoretical_occupancy_pct;
    double achieved_occupancy_pct;    /* -1 until bench/profile.py fills it in */
    int    limiter;
    double median_ms;
    bench_stats stats;
} mb_occ_point;
typedef struct {
    int          n_points;
    mb_occ_point points[MB_OCC_MAX_POINTS];
    int          warp_size, max_threads_per_block, max_threads_per_sm;
    int          max_blocks_per_sm, regs_per_sm;
    int          n_register_limited;
    int          warmup, samples;
} mb_occ_result;
int mb_occupancy_sweep_run(int warmup, int samples, mb_occ_result *out);

#ifdef __cplusplus
}
#endif
#endif /* MICROBENCH_H */
