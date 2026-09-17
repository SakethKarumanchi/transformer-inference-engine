/* device_info.cu -- one queried snapshot of the primary CUDA device, shared by
 * the seven GPU microbenchmarks and by their unit tests.
 *
 * Everything here is [queried] except cores_per_sm_spec, which is the vendor
 * compute-capability -> FP32-cores-per-SM table. CUDA exposes no such property,
 * so it is tagged [spec] and is only ever used to compute a theoretical ceiling
 * that a measurement is checked against -- never as a measurement itself.
 */
#include "microbench.h"

#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <string.h>

static int cores_per_sm_for_cc(int major, int minor)
{
    /* [spec] vendor table. Turing (7.5) is 64 FP32 lanes per SM. */
    switch (major * 10 + minor) {
        case 30: case 32: case 35: case 37: return 192;   /* Kepler  */
        case 50: case 52: case 53: return 128;            /* Maxwell */
        case 60: return 64;                               /* Pascal GP100 */
        case 61: case 62: return 128;                     /* Pascal  */
        case 70: case 72: return 64;                      /* Volta   */
        case 75: return 64;                               /* Turing  */
        case 80: return 64;                               /* Ampere A100 */
        case 86: case 87: case 89: return 128;            /* Ampere/Ada */
        case 90: return 128;                              /* Hopper  */
        default: return 0;                                /* unknown: refuse to guess */
    }
}

extern "C" int mb_query_device(mb_device_info *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        fprintf(stderr, "mb_query_device: no CUDA device\n");
        return 1;
    }
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        fprintf(stderr, "mb_query_device: cudaGetDeviceProperties failed\n");
        return 1;
    }

    snprintf(out->name, sizeof out->name, "%s", p.name);
    out->major                 = p.major;
    out->minor                 = p.minor;
    out->sm_count              = p.multiProcessorCount;
    out->warp_size             = p.warpSize;
    out->max_threads_per_block = p.maxThreadsPerBlock;
    out->max_threads_per_sm    = p.maxThreadsPerMultiProcessor;
    out->max_blocks_per_sm     = p.maxBlocksPerMultiProcessor;
    out->regs_per_sm           = p.regsPerMultiprocessor;
    out->regs_per_block        = p.regsPerBlock;
    out->shared_per_block      = p.sharedMemPerBlock;
    out->shared_per_sm         = p.sharedMemPerMultiprocessor;
    out->l2_bytes              = (size_t)p.l2CacheSize;
    out->total_global_bytes    = p.totalGlobalMem;
    out->bus_width_bits        = p.memoryBusWidth;

    /* CUDA 13 removed cudaDeviceProp::clockRate and ::memoryClockRate; the
     * values now come from the attribute API. Recorded here because a later
     * stage reading this code needs to know why the props struct is not used. */
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, 0);       out->clock_khz     = v;
    cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, 0); out->mem_clock_khz = v;

    out->cores_per_sm_spec = cores_per_sm_for_cc(p.major, p.minor);

    /* The CUDA runtime exposes no "maximum reachable clock". NVML does, and it
     * is the only honest ceiling to bound a measurement against: the rated
     * boost clock above can be exceeded by opportunistic boost. NVML failing
     * is recorded, never papered over with a substituted value. */
    if (nvmlInit_v2() == NVML_SUCCESS) {
        nvmlDevice_t nd;
        unsigned int mhz = 0;
        if (nvmlDeviceGetHandleByIndex_v2(0, &nd) == NVML_SUCCESS) {
            if (nvmlDeviceGetMaxClockInfo(nd, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS)
                out->max_sm_clock_khz = (int)mhz * 1000;
            if (nvmlDeviceGetMaxClockInfo(nd, NVML_CLOCK_MEM, &mhz) == NVML_SUCCESS)
                out->max_mem_clock_khz = (int)mhz * 1000;
            out->nvml_available = 1;
        }
        nvmlShutdown();
    }

    /* [derived] DRAM ceiling = 2 transfers/clock * clock * bus bytes */
    out->theoretical_dram_gb_s =
        2.0 * ((double)out->mem_clock_khz * 1.0e3) * ((double)out->bus_width_bits / 8.0) / 1.0e9;

    /* [derived] FP32 ceiling = SMs * lanes/SM * 2 flops/FMA * clock */
    out->theoretical_fp32_gflops =
        (double)out->sm_count * (double)out->cores_per_sm_spec * 2.0 *
        ((double)out->clock_khz * 1.0e3) / 1.0e9;

    int max_sm  = out->max_sm_clock_khz  ? out->max_sm_clock_khz  : out->clock_khz;
    int max_mem = out->max_mem_clock_khz ? out->max_mem_clock_khz : out->mem_clock_khz;
    out->theoretical_fp32_gflops_at_max_clock =
        (double)out->sm_count * (double)out->cores_per_sm_spec * 2.0 *
        ((double)max_sm * 1.0e3) / 1.0e9;
    out->theoretical_dram_gb_s_at_max_clock =
        2.0 * ((double)max_mem * 1.0e3) * ((double)out->bus_width_bits / 8.0) / 1.0e9;

    return 0;
}
