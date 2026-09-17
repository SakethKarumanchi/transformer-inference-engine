/* Microbenchmark 6 -- cuBLAS SGEMM reference at GPT-2 small's real shapes.
 *
 * This is the prefill denominator for the entire project, so it runs at the
 * shapes the model actually performs. None of them is square. The M dimension
 * is the token count and is swept from the decode case (M = 1) up to the
 * maximum context the architecture supports; prefill and decode are reported
 * as separate records, never as one combined figure.
 *
 * The architecture parameters are DETERMINED here, not assumed, and are
 * UNCONFIRMED until Stage 1 verifies them against the config file shipped with
 * the weights. See mb_gpt2_small_provenance().
 */
#include "microbench.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GPT-2 small: n_embd 768, n_head 12, n_layer 12, n_ctx 1024, vocab 50257. */
#define GPT2_N_EMBD 768
#define GPT2_N_CTX  1024
#define GPT2_VOCAB  50257

static const mb_gemm_shape k_shapes[MB_GEMM_N_SHAPES] = {
    /* name,                    N (output features),      K (input features) */
    { "qkv_projection",         3 * GPT2_N_EMBD,          GPT2_N_EMBD     },
    { "attn_output_projection", GPT2_N_EMBD,              GPT2_N_EMBD     },
    { "ffn_up",                 4 * GPT2_N_EMBD,          GPT2_N_EMBD     },
    { "ffn_down",               GPT2_N_EMBD,              4 * GPT2_N_EMBD },
    { "lm_head",                GPT2_VOCAB,               GPT2_N_EMBD     },
};

/* M = token count. 1 is the decode case; the rest span short prefill to the
 * architecture's maximum context (n_ctx = 1024). The benchmark prompt set and
 * its token counts are not yet decided -- Stage 3 fixes those. */
static const int k_m_values[] = { 1, 8, 16, 32, 64, 128, 256, 512, 1024 };

extern "C" const mb_gemm_shape *mb_gpt2_small_shapes(int *count)
{
    if (count) *count = MB_GEMM_N_SHAPES;
    return k_shapes;
}

extern "C" const int *mb_gemm_m_sweep(int *count)
{
    if (count) *count = (int)(sizeof k_m_values / sizeof k_m_values[0]);
    return k_m_values;
}

extern "C" const char *mb_gpt2_small_provenance(void)
{
    return "GPT-2 small architecture parameters (n_embd=768, n_head=12, n_layer=12, "
           "n_ctx=1024, vocab_size=50257) read from the published config.json shipped "
           "with the openai-community/gpt2 weights, fetched 2026-09-16. UNCONFIRMED "
           "until Stage 1 verifies them against the config file it loads with the weights.";
}

typedef struct {
    cublasHandle_t h;
    int M, N, K;
    const float *A, *B;
    float *C;
    bench_cuda_timer *timer;
    cublasStatus_t last_status;
} gemm_ctx;

static double gemm_body(void *vctx, int iteration)
{
    (void)iteration;
    gemm_ctx *c = (gemm_ctx *)vctx;
    const float alpha = 1.0f, beta = 0.0f;
    bench_cuda_timer_start(c->timer);
    /* Row-major C[M,N] = A[M,K] * B[K,N] expressed for a column-major library. */
    c->last_status = cublasSgemm(c->h, CUBLAS_OP_N, CUBLAS_OP_N,
                                 c->N, c->M, c->K,
                                 &alpha, c->B, c->N, c->A, c->K,
                                 &beta,  c->C, c->N);
    return bench_cuda_timer_stop_ms(c->timer);
}

extern "C" int mb_cublas_sgemm_ref_run(int warmup, int samples, mb_gemm_result *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);
    snprintf(out->provenance, sizeof out->provenance, "%s", mb_gpt2_small_provenance());

    int n_m = 0;
    const int *ms = mb_gemm_m_sweep(&n_m);
    out->n_m_values = n_m;
    for (int i = 0; i < n_m && i < 16; ++i) out->m_values[i] = ms[i];

    cublasHandle_t h = NULL;
    cublasStatus_t cs = cublasCreate(&h);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas_sgemm_ref: cublasCreate failed, status %d\n", (int)cs);
        return 1;
    }

    int n_alloc = samples > BENCH_MIN_SAMPLES ? samples : BENCH_MIN_SAMPLES;
    double *raw = (double *)malloc((size_t)n_alloc * sizeof(double));
    bench_record *recs = (bench_record *)calloc(MB_GEMM_MAX_POINTS, sizeof(bench_record));
    double **raws = (double **)calloc(MB_GEMM_MAX_POINTS, sizeof(double *));
    char (*cfgs)[BENCH_LABEL_LEN] =
        (char (*)[BENCH_LABEL_LEN])calloc(MB_GEMM_MAX_POINTS, BENCH_LABEL_LEN);
    if (!raw || !recs || !raws || !cfgs) {
        free(raw); free(recs); free(raws); free(cfgs); cublasDestroy(h); return 1;
    }

    bench_cuda_timer *timer = bench_cuda_timer_create();
    int rc = 1, np = 0, eff_w = 0, eff_s = 0;
    if (!timer) goto done;

    for (int si = 0; si < MB_GEMM_N_SHAPES; ++si) {
        for (int mi = 0; mi < n_m && np < MB_GEMM_MAX_POINTS; ++mi) {
            int M = ms[mi], N = k_shapes[si].N, K = k_shapes[si].K;

            float *A = NULL, *B = NULL, *C = NULL;
            if (cudaMalloc(&A, (size_t)M * K * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&B, (size_t)K * N * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&C, (size_t)M * N * sizeof(float)) != cudaSuccess) {
                fprintf(stderr, "cublas_sgemm_ref: allocation failed for %s M=%d\n",
                        k_shapes[si].name, M);
                cudaFree(A); cudaFree(B); cudaFree(C);
                goto done;
            }
            cudaMemset(A, 0x3c, (size_t)M * K * sizeof(float));
            cudaMemset(B, 0x3c, (size_t)K * N * sizeof(float));

            gemm_ctx c;
            c.h = h; c.M = M; c.N = N; c.K = K; c.A = A; c.B = B; c.C = C;
            c.timer = timer; c.last_status = CUBLAS_STATUS_SUCCESS;

            bench_stats st = bench_run(gemm_body, &c, warmup, samples, raw, &eff_w, &eff_s);

            if (c.last_status != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "cublas_sgemm_ref: cublasSgemm status %d for %s M=%d\n",
                        (int)c.last_status, k_shapes[si].name, M);
                cudaFree(A); cudaFree(B); cudaFree(C);
                goto done;
            }

            double flops  = 2.0 * (double)M * (double)N * (double)K;
            double gflops = (st.median > 0.0) ? flops / (st.median * 1.0e-3) / 1.0e9 : 0.0;

            mb_gemm_point *pt = &out->points[np];
            snprintf(pt->shape_name, sizeof pt->shape_name, "%s", k_shapes[si].name);
            pt->M = M; pt->N = N; pt->K = K;
            pt->is_decode = (M == 1);
            pt->gflops = gflops;
            pt->stats  = st;

            raws[np] = (double *)malloc((size_t)eff_s * sizeof(double));
            if (!raws[np]) { cudaFree(A); cudaFree(B); cudaFree(C); goto done; }
            memcpy(raws[np], raw, (size_t)eff_s * sizeof(double));

            snprintf(cfgs[np], BENCH_LABEL_LEN, "%s %s M=%d N=%d K=%d",
                     pt->is_decode ? "decode" : "prefill", k_shapes[si].name, M, N, K);
            recs[np].benchmark         = "cublas_sgemm_ref";
            recs[np].configuration     = cfgs[np];
            recs[np].units             = "GFLOP/s";
            recs[np].value             = gflops;
            recs[np].warmup            = eff_w;
            recs[np].samples_requested = eff_s;
            recs[np].samples_ms        = raws[np];
            recs[np].n_samples         = eff_s;
            recs[np].stats             = st;
            ++np;

            cudaFree(A); cudaFree(B); cudaFree(C);
        }
    }

    out->n_points = np;
    out->warmup   = eff_w;
    out->samples  = eff_s;

    {
        /* Per-record phase and shape metadata, so the Stage 10 model can split
         * prefill from decode without re-deriving anything. */
        for (int i = 0; i < np; ++i) {
            static bench_kv_str shared_strs[2];
            shared_strs[0].key = "architecture_provenance";
            shared_strs[0].value = out->provenance;
            shared_strs[1].key = "tag";
            shared_strs[1].value = "measured";
            recs[i].meta_str = shared_strs; recs[i].n_meta_str = 2;
        }
        static bench_kv_num per[MB_GEMM_MAX_POINTS][4];
        for (int i = 0; i < np; ++i) {
            per[i][0].key = "M"; per[i][0].value = out->points[i].M;
            per[i][1].key = "N"; per[i][1].value = out->points[i].N;
            per[i][2].key = "K"; per[i][2].value = out->points[i].K;
            per[i][3].key = "is_decode"; per[i][3].value = out->points[i].is_decode;
            recs[i].meta_num = per[i]; recs[i].n_meta_num = 4;
        }
        bench_write_results(NULL, "cublas_sgemm_ref", recs, np);
    }
    rc = 0;

done:
    if (timer) bench_cuda_timer_destroy(timer);
    for (int i = 0; i < MB_GEMM_MAX_POINTS; ++i) free(raws[i]);
    free(raws); free(recs); free(cfgs); free(raw);
    cublasDestroy(h);       /* handle destroyed cleanly on every path */
    return rc;
}

#ifndef BENCH_NO_MAIN
int main(int argc, char **argv)
{
    int warmup  = (argc > 1) ? atoi(argv[1]) : BENCH_MIN_WARMUP;
    int samples = (argc > 2) ? atoi(argv[2]) : BENCH_MIN_SAMPLES;

    mb_gemm_result r;
    if (mb_cublas_sgemm_ref_run(warmup, samples, &r) != 0) {
        fprintf(stderr, "cublas_sgemm_ref: FAILED\n");
        return 1;
    }
    printf("cublas_sgemm_ref  %d points\n%s\n", r.n_points, r.provenance);
    int all_valid = 1;
    printf("-- decode (M=1) --\n");
    for (int i = 0; i < r.n_points; ++i) if (r.points[i].is_decode) {
        printf("  %-24s M=%-5d N=%-6d K=%-5d %9.2f GFLOP/s  stddev %6.3f%%  %s\n",
               r.points[i].shape_name, r.points[i].M, r.points[i].N, r.points[i].K,
               r.points[i].gflops, r.points[i].stats.stddev_pct_of_median,
               r.points[i].stats.valid ? "VALID" : "INVALID");
        if (!r.points[i].stats.valid) all_valid = 0;
    }
    printf("-- prefill (M>1) --\n");
    for (int i = 0; i < r.n_points; ++i) if (!r.points[i].is_decode) {
        printf("  %-24s M=%-5d N=%-6d K=%-5d %9.2f GFLOP/s  stddev %6.3f%%  %s\n",
               r.points[i].shape_name, r.points[i].M, r.points[i].N, r.points[i].K,
               r.points[i].gflops, r.points[i].stats.stddev_pct_of_median,
               r.points[i].stats.valid ? "VALID" : "INVALID");
        if (!r.points[i].stats.valid) all_valid = 0;
    }
    return all_valid ? 0 : 2;
}
#endif
