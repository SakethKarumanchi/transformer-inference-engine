/* Unit test for microbenchmark 6 -- the prefill denominator for the project. */
#include "test_util.h"
#include "microbench.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

/* The architecture-derived shapes this test independently expects. If Stage 1
 * finds different values in the config shipped with the weights, this list and
 * the benchmark's must both change together. */
#define T_N_EMBD 768
#define T_VOCAB  50257
typedef struct { const char *name; int N, K; } expected_shape;
static const expected_shape k_expected[] = {
    { "qkv_projection",         3 * T_N_EMBD, T_N_EMBD     },
    { "attn_output_projection", T_N_EMBD,     T_N_EMBD     },
    { "ffn_up",                 4 * T_N_EMBD, T_N_EMBD     },
    { "ffn_down",               T_N_EMBD,     4 * T_N_EMBD },
    { "lm_head",                T_VOCAB,      T_N_EMBD     },
};
#define T_N_EXPECTED ((int)(sizeof k_expected / sizeof k_expected[0]))

int main(void)
{
    printf("test_cublas_sgemm_ref\n");
    tie_redirect_results();

    /* --- the enumerated shapes ---------------------------------------- */
    int n_shapes = 0;
    const mb_gemm_shape *shapes = mb_gpt2_small_shapes(&n_shapes);
    CHECK(n_shapes == T_N_EXPECTED,
          "five GEMM shapes are enumerated (QKV, attention output, FFN up, "
          "FFN down, LM head)");
    for (int i = 0; i < n_shapes && i < T_N_EXPECTED; ++i) {
        CHECK(strcmp(shapes[i].name, k_expected[i].name) == 0,
              "shape %d is %s", i, k_expected[i].name);
        CHECK(shapes[i].N == k_expected[i].N && shapes[i].K == k_expected[i].K,
              "%s has N=%d K=%d, matching the architecture-derived values",
              k_expected[i].name, k_expected[i].N, k_expected[i].K);
    }

    /* --- the M sweep -------------------------------------------------- */
    int n_m = 0;
    const int *ms = mb_gemm_m_sweep(&n_m);
    int has_one = 0, has_ctx = 0;
    for (int i = 0; i < n_m; ++i) {
        if (ms[i] == 1)    has_one = 1;
        if (ms[i] == 1024) has_ctx = 1;
    }
    CHECK(has_one, "the M sweep includes M = 1, the decode case");
    CHECK(has_ctx, "the M sweep reaches n_ctx = 1024, the architecture's maximum context");
    CHECK(n_m >= 5, "the M sweep spans short prefill to maximum context (%d values)", n_m);

    /* --- no benchmarked shape is square ------------------------------- */
    for (int s = 0; s < n_shapes; ++s)
        for (int m = 0; m < n_m; ++m) {
            int M = ms[m], N = shapes[s].N, K = shapes[s].K;
            CHECK(!(M == N && N == K),
                  "%s at M=%d is not square (M=%d N=%d K=%d)",
                  shapes[s].name, M, M, N, K);
        }

    /* --- cuBLAS handle lifecycle and status checking ------------------ */
    {
        cublasHandle_t h = NULL;
        cublasStatus_t cs = cublasCreate(&h);
        CHECK(cs == CUBLAS_STATUS_SUCCESS,
              "cuBLAS is available and a handle can be created (status %d)", (int)cs);
        if (cs == CUBLAS_STATUS_SUCCESS) {
            CHECK(cublasDestroy(h) == CUBLAS_STATUS_SUCCESS,
                  "the handle is destroyed cleanly");
        }
    }

    /* --- run it and check the emitted record -------------------------- */
    mb_gemm_result r;
    int rc = mb_cublas_sgemm_ref_run(BENCH_MIN_WARMUP, BENCH_MIN_SAMPLES, &r);
    CHECK(rc == 0, "the benchmark runs to completion with every cuBLAS status checked");
    if (rc != 0) TIE_SUMMARY("test_cublas_sgemm_ref");

    CHECK(r.n_points == n_shapes * n_m,
          "every shape was run at every M value (%d points)", r.n_points);

    int decode_points = 0, prefill_points = 0, square = 0, mismatched = 0;
    for (int i = 0; i < r.n_points; ++i) {
        const mb_gemm_point *p = &r.points[i];
        if (p->is_decode) ++decode_points; else ++prefill_points;
        if (p->M == p->N && p->N == p->K) ++square;
        int ok = 0;
        for (int e = 0; e < T_N_EXPECTED; ++e)
            if (strcmp(p->shape_name, k_expected[e].name) == 0 &&
                p->N == k_expected[e].N && p->K == k_expected[e].K) ok = 1;
        if (!ok) ++mismatched;
        if (p->is_decode) CHECK(p->M == 1, "a decode point has M = 1");
    }
    CHECK(square == 0, "no benchmarked shape is square");
    CHECK(mismatched == 0, "every benchmarked N and K matches the enumerated values");
    CHECK(decode_points == n_shapes, "the decode case was run for every shape");
    CHECK(prefill_points > 0, "prefill points were run and are counted separately");

    CHECK(strlen(r.provenance) > 0 &&
          strstr(r.provenance, "config.json") != NULL &&
          strstr(r.provenance, "UNCONFIRMED") != NULL,
          "the provenance string records where the architecture parameters came "
          "from and that they are unconfirmed until Stage 1");

    {
        char *json = tie_read_file("cublas_sgemm_ref.json");
        CHECK(json != NULL, "the results file was written");
        if (json) {
            CHECK(strstr(json, "architecture_provenance") != NULL,
                  "the results file records the provenance string");
            CHECK(strstr(json, "\"is_decode\"") != NULL,
                  "the results file marks decode points separately from prefill");
            CHECK(tie_count(json, "\"benchmark\":\"cublas_sgemm_ref\"") == r.n_points,
                  "the results file holds one record per measured point");
            free(json);
        }
        remove("cublas_sgemm_ref.json");
    }

    TIE_SUMMARY("test_cublas_sgemm_ref");
}
