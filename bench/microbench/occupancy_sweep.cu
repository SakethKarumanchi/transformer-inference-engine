/* Microbenchmark 9 -- occupancy sweep.
 *
 * One kernel swept across block sizes and register pressures. Theoretical
 * occupancy is computed from QUERIED device properties with the arithmetic
 * recorded in the results file; it is never hardcoded. Achieved occupancy is
 * left at -1 here and filled in by bench/profile.py from the profiler's
 * sm__warps_active counter -- it is never assumed equal to theoretical.
 *
 * The register-pressure knob is the accumulator count, a template parameter,
 * so each variant genuinely compiles to a different register footprint rather
 * than being annotated as if it did.
 */
#include "microbench.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

template <int ACC>
__global__ void mb_occ_kernel(float *sink, int iters, float a, float b)
{
    float acc[ACC];
#pragma unroll
    for (int i = 0; i < ACC; ++i) acc[i] = a + (float)i;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < ACC; ++i) acc[i] = fmaf(acc[i], b, a);
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < ACC; ++i) s += acc[i];
    if (s == -1.0e30f) sink[threadIdx.x] = s;
}

static const int k_acc_variants[]   = { 4, 16, 40, 72 };
static const int k_block_sizes[]    = { 32, 64, 128, 256, 512, 1024 };
#define MB_OCC_N_ACC   ((int)(sizeof k_acc_variants / sizeof k_acc_variants[0]))
#define MB_OCC_N_BLOCK ((int)(sizeof k_block_sizes  / sizeof k_block_sizes[0]))
#define MB_OCC_ITERS   4096

static const void *occ_func_ptr(int acc)
{
    switch (acc) {
        case  4: return (const void *)mb_occ_kernel<4>;
        case 16: return (const void *)mb_occ_kernel<16>;
        case 40: return (const void *)mb_occ_kernel<40>;
        case 72: return (const void *)mb_occ_kernel<72>;
        default: return NULL;
    }
}

static void occ_launch(int acc, int grid, int block, float *sink, int iters)
{
    switch (acc) {
        case  4: mb_occ_kernel<4> <<<grid, block>>>(sink, iters, 1.0f, 1.0000001f); break;
        case 16: mb_occ_kernel<16><<<grid, block>>>(sink, iters, 1.0f, 1.0000001f); break;
        case 40: mb_occ_kernel<40><<<grid, block>>>(sink, iters, 1.0f, 1.0000001f); break;
        case 72: mb_occ_kernel<72><<<grid, block>>>(sink, iters, 1.0f, 1.0000001f); break;
        default: break;
    }
}

typedef struct {
    int acc, grid, block, iters;
    float *sink;
    bench_cuda_timer *timer;
} occ_ctx;

static double occ_body(void *vctx, int iteration)
{
    (void)iteration;
    occ_ctx *c = (occ_ctx *)vctx;
    bench_cuda_timer_start(c->timer);
    occ_launch(c->acc, c->grid, c->block, c->sink, c->iters);
    return bench_cuda_timer_stop_ms(c->timer);
}

extern "C" int mb_occupancy_sweep_run(int warmup, int samples, mb_occ_result *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);

    mb_device_info dev;
    if (mb_query_device(&dev) != 0) return 1;

    out->warp_size            = dev.warp_size;
    out->max_threads_per_block= dev.max_threads_per_block;
    out->max_threads_per_sm   = dev.max_threads_per_sm;
    out->max_blocks_per_sm    = dev.max_blocks_per_sm;
    out->regs_per_sm          = dev.regs_per_sm;

    float *sink = NULL;
    if (cudaMalloc(&sink, 1024 * sizeof(float)) != cudaSuccess) return 1;
    cudaMemset(sink, 0, 1024 * sizeof(float));

    bench_cuda_timer *timer = bench_cuda_timer_create();
    if (!timer) { cudaFree(sink); return 1; }

    int n_alloc = samples > BENCH_MIN_SAMPLES ? samples : BENCH_MIN_SAMPLES;
    double *raw = (double *)malloc((size_t)n_alloc * sizeof(double));
    bench_record *recs = (bench_record *)calloc(MB_OCC_MAX_POINTS, sizeof(bench_record));
    double **raws = (double **)calloc(MB_OCC_MAX_POINTS, sizeof(double *));
    char (*cfgs)[BENCH_LABEL_LEN] =
        (char (*)[BENCH_LABEL_LEN])calloc(MB_OCC_MAX_POINTS, BENCH_LABEL_LEN);
    static bench_kv_num per[MB_OCC_MAX_POINTS][8];
    if (!raw || !recs || !raws || !cfgs) {
        free(raw); free(recs); free(raws); free(cfgs);
        bench_cuda_timer_destroy(timer); cudaFree(sink); return 1;
    }

    int np = 0, eff_w = 0, eff_s = 0;
    for (int ai = 0; ai < MB_OCC_N_ACC; ++ai) {
        int acc = k_acc_variants[ai];
        cudaFuncAttributes fa;
        memset(&fa, 0, sizeof fa);
        cudaFuncGetAttributes(&fa, occ_func_ptr(acc));

        for (int bi = 0; bi < MB_OCC_N_BLOCK && np < MB_OCC_MAX_POINTS; ++bi) {
            int block = k_block_sizes[bi];
            if (block % dev.warp_size != 0)        continue;
            if (block > dev.max_threads_per_block)  continue;

            int blocks_per_sm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
                                                          occ_func_ptr(acc), block, 0);

            /* Arithmetic recorded rather than trusted: which resource binds. */
            int lim_blocks = dev.max_blocks_per_sm;
            int lim_warps  = dev.max_threads_per_sm / block;
            int lim_regs   = (fa.numRegs > 0)
                           ? dev.regs_per_sm / (fa.numRegs * block)
                           : lim_blocks;
            int lim_shared = lim_blocks;    /* this kernel uses no shared memory */

            int limiter = MB_OCC_LIMIT_BLOCKS, best = lim_blocks;
            if (lim_warps  < best) { best = lim_warps;  limiter = MB_OCC_LIMIT_WARPS; }
            if (lim_regs   < best) { best = lim_regs;   limiter = MB_OCC_LIMIT_REGISTERS; }
            if (lim_shared < best) { best = lim_shared; limiter = MB_OCC_LIMIT_SHARED; }

            mb_occ_point *pt = &out->points[np];
            pt->block_size               = block;
            pt->accumulators             = acc;
            pt->regs_per_thread          = fa.numRegs;
            pt->blocks_per_sm_theoretical= blocks_per_sm;
            pt->theoretical_occupancy_pct=
                100.0 * (double)(blocks_per_sm * block) / (double)dev.max_threads_per_sm;
            pt->achieved_occupancy_pct   = -1.0;   /* profile.py fills this in */
            pt->limiter                  = limiter;
            if (limiter == MB_OCC_LIMIT_REGISTERS) ++out->n_register_limited;

            if (blocks_per_sm <= 0) {
                /* Not launchable at this register pressure and block size. Recorded
                 * as such; no placeholder timing is invented for it. */
                pt->median_ms = -1.0;
                memset(&pt->stats, 0, sizeof pt->stats);
                pt->stats.valid = 0;
                snprintf(pt->stats.invalid_reason, BENCH_REASON_LEN,
                         "not launchable: %d regs/thread * %d threads exceeds the "
                         "per-SM register file of %d", fa.numRegs, block, dev.regs_per_sm);
                ++np;
                continue;
            }

            occ_ctx c;
            c.acc = acc; c.block = block; c.iters = MB_OCC_ITERS;
            c.grid = dev.sm_count * blocks_per_sm;
            c.sink = sink; c.timer = timer;

            bench_stats st = bench_run(occ_body, &c, warmup, samples, raw, &eff_w, &eff_s);
            cudaError_t last = cudaGetLastError();
            if (last != cudaSuccess) {
                fprintf(stderr, "occupancy_sweep: acc=%d block=%d: %s\n",
                        acc, block, cudaGetErrorString(last));
                pt->median_ms = -1.0;
                pt->stats.valid = 0;
                snprintf(pt->stats.invalid_reason, BENCH_REASON_LEN,
                         "launch failed: %s", cudaGetErrorString(last));
                ++np;
                continue;
            }

            pt->median_ms = st.median;
            pt->stats     = st;

            raws[np] = (double *)malloc((size_t)eff_s * sizeof(double));
            if (!raws[np]) goto done;
            memcpy(raws[np], raw, (size_t)eff_s * sizeof(double));

            static const char *lim_names[] = { "blocks", "warps", "registers", "shared" };
            snprintf(cfgs[np], BENCH_LABEL_LEN,
                     "acc=%d block=%d regs/thread=%d blocks/SM=%d limiter=%s",
                     acc, block, fa.numRegs, blocks_per_sm, lim_names[limiter]);
            per[np][0].key = "block_size";                per[np][0].value = block;
            per[np][1].key = "accumulators";              per[np][1].value = acc;
            per[np][2].key = "regs_per_thread";           per[np][2].value = fa.numRegs;
            per[np][3].key = "blocks_per_sm_theoretical"; per[np][3].value = blocks_per_sm;
            per[np][4].key = "theoretical_occupancy_pct"; per[np][4].value = pt->theoretical_occupancy_pct;
            per[np][5].key = "achieved_occupancy_pct";    per[np][5].value = -1.0;
            per[np][6].key = "limiter_code";              per[np][6].value = limiter;
            per[np][7].key = "max_threads_per_sm_queried";per[np][7].value = dev.max_threads_per_sm;

            recs[np].benchmark         = "occupancy_sweep";
            recs[np].configuration     = cfgs[np];
            recs[np].units             = "ms";
            recs[np].value             = st.median;
            recs[np].warmup            = eff_w;
            recs[np].samples_requested = eff_s;
            recs[np].samples_ms        = raws[np];
            recs[np].n_samples         = eff_s;
            recs[np].stats             = st;
            recs[np].meta_num          = per[np];
            recs[np].n_meta_num        = 8;
            ++np;
        }
    }

    out->n_points = np;
    out->warmup   = eff_w;
    out->samples  = eff_s;

    {
        static bench_kv_str strs[] = {
            { "theoretical_occupancy_derivation",
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor * blockSize / "
              "maxThreadsPerMultiProcessor[queried] * 100" },
            { "achieved_occupancy_source",
              "bench/profile.py, sm__warps_active.avg.pct_of_peak_sustained_active; "
              "-1 means not yet collected" },
            { "tag", "measured" },
        };
        int written = 0;
        for (int i = 0; i < np; ++i) {
            if (!recs[i].benchmark) continue;    /* non-launchable point, no record */
            recs[written] = recs[i];
            recs[written].meta_str = strs;
            recs[written].n_meta_str = 3;
            ++written;
        }
        if (written > 0) bench_write_results(NULL, "occupancy_sweep", recs, written);
    }

done:
    for (int i = 0; i < MB_OCC_MAX_POINTS; ++i) free(raws[i]);
    free(raws); free(recs); free(cfgs); free(raw);
    bench_cuda_timer_destroy(timer);
    cudaFree(sink);
    return 0;
}

#ifndef BENCH_NO_MAIN
int main(int argc, char **argv)
{
    int warmup  = (argc > 1) ? atoi(argv[1]) : BENCH_MIN_WARMUP;
    int samples = (argc > 2) ? atoi(argv[2]) : BENCH_MIN_SAMPLES;

    mb_occ_result r;
    if (mb_occupancy_sweep_run(warmup, samples, &r) != 0) {
        fprintf(stderr, "occupancy_sweep: FAILED\n");
        return 1;
    }
    static const char *lim_names[] = { "blocks", "warps", "registers", "shared" };
    printf("occupancy_sweep  warp %d, maxThreads/block %d, maxThreads/SM %d, "
           "maxBlocks/SM %d, regs/SM %d\n",
           r.warp_size, r.max_threads_per_block, r.max_threads_per_sm,
           r.max_blocks_per_sm, r.regs_per_sm);
    int all_valid = 1;
    for (int i = 0; i < r.n_points; ++i) {
        const mb_occ_point *p = &r.points[i];
        printf("  acc=%-3d block=%-5d regs=%-4d blocks/SM=%-3d theo=%6.2f%% "
               "achieved=%s limiter=%-9s median=%s\n",
               p->accumulators, p->block_size, p->regs_per_thread,
               p->blocks_per_sm_theoretical, p->theoretical_occupancy_pct,
               p->achieved_occupancy_pct < 0 ? "pending" : "collected",
               lim_names[p->limiter],
               p->median_ms < 0 ? "not launchable" : "see results file");
        if (p->median_ms >= 0 && !p->stats.valid) all_valid = 0;
    }
    printf("  register-limited configurations: %d\n", r.n_register_limited);
    return all_valid ? 0 : 2;
}
#endif
