/* stage2_forward_bench.c -- the Stage 2 timing driver.
 *
 * THIS MECHANISM IS TEMPORARY. Stage 3 builds the benchmark harness and
 * replaces it. What is NOT temporary is the results FORMAT, which
 * BENCHMARK_PROTOCOL.md section 9 makes a contract the Stage 10 model reads,
 * and which bench_common already implements. Nothing here reimplements warmup,
 * sampling, statistics, the 5%-of-median validity rule or the results writer;
 * it calls bench_run and bench_write_results. There is no harness here, no
 * registration machinery, no regression detection and no second results path.
 *
 * THE TIMED BRACKET CONTAINS COMPUTATION ONLY. Weight loading, tokenizer
 * loading, tokenization, scratch allocation and the results write are all
 * outside it, per BENCHMARK_PROTOCOL.md section 2. model_reserve is called
 * before the first warmup iteration so no iteration allocates.
 *
 * THREAD PLACEMENT IS APPLIED AND RECORDED. An unpinned single-threaded CPU
 * measurement samples two cores and reports the mixture as variance; Stage 0b
 * established that on this machine and pinned both CPU microbenchmarks. This
 * driver pins the same way, outside every timed bracket, and records where it
 * actually ran -- never assuming the affinity call took.
 *
 * PREFILL AND DECODE ARE SEPARATE RECORDS, ALWAYS. So is the isolated GEMM
 * configuration, which is the only counter-free evidence this stage has for
 * separating a scalar-issue-rate limit from a memory-hierarchy limit.
 *
 * INPUTS ARE PLACEHOLDERS (D3 is open and owned by Stage 3). The flat
 * prompt_set_* annotations below carry that status into the results file,
 * because a prose-only label is invisible to the Stage 10 consumer. They are
 * flat rather than a nested object only because bench_common writes strings as
 * JSON strings and this driver does not fork the writer; the nested
 * "prompt_set" object is written by the correctness script alongside the
 * divergence statistics.
 */
#include "bench_common.h"
#include "model.h"
#include "tokenizer.h"
#include "gemm/gemm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TIE_MODEL_DIR
#define TIE_MODEL_DIR "models/gpt2"
#endif
#ifndef TIE_REPO_DIR
#define TIE_REPO_DIR "."
#endif

#define MAX_SAMPLES 64
#define MAX_RECORDS 16

/* ------------------------------------------------------------ contexts ---- */

typedef struct {
    model         *m;
    const int32_t *ids;
    size_t         n;
    float         *logits;
    size_t         cap;
} fwd_ctx;

static double prefill_body(void *vc, int iteration)
{
    fwd_ctx *c = (fwd_ctx *)vc;
    (void)iteration;
    double t0 = bench_cpu_time_seconds();
    model_prefill(c->m, c->ids, c->n, c->logits, c->cap);
    return (bench_cpu_time_seconds() - t0) * 1000.0;
}

static double decode_body(void *vc, int iteration)
{
    fwd_ctx *c = (fwd_ctx *)vc;
    (void)iteration;
    double t0 = bench_cpu_time_seconds();
    model_decode_step(c->m, c->ids, c->n, c->logits, c->cap);
    return (bench_cpu_time_seconds() - t0) * 1000.0;
}

typedef struct {
    int M, N, K;
    const float *A, *B;
    float *C;
} gemm_ctx;

static double gemm_body(void *vc, int iteration)
{
    gemm_ctx *g = (gemm_ctx *)vc;
    (void)iteration;
    double t0 = bench_cpu_time_seconds();
    gemm_naive(g->M, g->N, g->K, g->A, g->K, g->B, g->N, g->C, g->N);
    return (bench_cpu_time_seconds() - t0) * 1000.0;
}

/* ---------------------------------------------------------- record slot ---- */

typedef struct {
    double       samples[MAX_SAMPLES];
    bench_stats  stats;
    char         configuration[160];
    bench_kv_num num[16];
    int          n_num;
    bench_kv_str str[16];
    int          n_str;
    bench_record rec;
} slot;

static void add_num(slot *s, const char *k, double v)
{
    if (s->n_num < 16) { s->num[s->n_num].key = k; s->num[s->n_num].value = v; ++s->n_num; }
}
static void add_str(slot *s, const char *k, const char *v)
{
    if (s->n_str < 16) { s->str[s->n_str].key = k; s->str[s->n_str].value = v; ++s->n_str; }
}

/* WITHIN-RUN DRIFT, DIAGNOSTIC ONLY.
 *
 * The 5%-of-median rule tests dispersion. It does not test DIRECTION: a run
 * whose second half is systematically slower than its first can pass the
 * std-dev test while drifting, and on this machine that is the signature a
 * sustained single-core load would leave -- the CPU has no clock lock and this
 * machine has no live package-temperature source, so the timings themselves are
 * the only evidence available. The figures below therefore characterise the
 * distribution: they never replace the reported median, and they never convert
 * an INVALID run into a valid one.
 */
static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_of(const double *v, int n)
{
    if (n <= 0) return 0.0;
    double tmp[MAX_SAMPLES];
    if (n > MAX_SAMPLES) n = MAX_SAMPLES;
    memcpy(tmp, v, (size_t)n * sizeof(double));
    qsort(tmp, (size_t)n, sizeof(double), cmp_double);
    return (n % 2) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
}

static void add_drift(slot *s, int n_samples)
{
    int h = n_samples / 2;
    double m1 = median_of(s->samples, h);
    double m2 = median_of(s->samples + h, n_samples - h);
    add_num(s, "diagnostic_first_half_median_ms", m1);
    add_num(s, "diagnostic_second_half_median_ms", m2);
    add_num(s, "diagnostic_drift_pct_second_vs_first",
            m1 > 0.0 ? (m2 - m1) / m1 * 100.0 : 0.0);
    add_num(s, "diagnostic_drift_n_first_half", h);
    add_num(s, "diagnostic_drift_n_second_half", n_samples - h);
    add_str(s, "diagnostic_drift_note",
            "DIAGNOSTIC: median of the second half of the samples against the first, "
            "signed. Characterises the distribution; never replaces the reported median "
            "and never converts an INVALID run into a valid one");
}

/* Loads the placeholder prompt text: the first data row of the fixture, third
 * tab-separated column. Read once, outside every timed bracket. */
static int load_placeholder_prompt(const char *path, char *out, size_t cap)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char *t1 = strchr(line, '\t');
        if (!t1) continue;
        char *t2 = strchr(t1 + 1, '\t');
        if (!t2) continue;
        char *text = t2 + 1;
        size_t n = strlen(text);
        while (n > 0 && (text[n - 1] == '\n' || text[n - 1] == '\r')) text[--n] = '\0';
        if (n == 0 || n + 1 > cap) { fclose(f); return 0; }
        memcpy(out, text, n + 1);
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    int warmup = 25;            /* BENCHMARK_PROTOCOL.md section 4.2, adjusted value */
    int samples = 30;           /* section 3: 30 where the budget allows, never below 20 */
    int probe_only = 0;
    const char *fixture = TIE_REPO_DIR "/tests/fixtures/stage2_placeholder_prompts.tsv";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc)  { warmup  = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--samples") && i + 1 < argc) { samples = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--probe")) { probe_only = 1; continue; }
        if (!strcmp(argv[i], "--fixture") && i + 1 < argc) { fixture = argv[++i]; continue; }
        fprintf(stderr, "usage: %s [--warmup N] [--samples N] [--probe] [--fixture PATH]\n",
                argv[0]);
        return 2;
    }
    if (samples > MAX_SAMPLES) samples = MAX_SAMPLES;

    /* ---- placement, applied and RECORDED, outside every timed bracket ---- */
    bench_thread_placement place = bench_pin_current_thread(-1);
    printf("thread placement: %s (pinned=%d, logical cpu %d, priority raised=%d)\n",
           place.detail, place.pinned, place.logical_cpu, place.priority_raised);
    printf("cpu timer: %s\n", bench_cpu_timer_name());

    /* ---- load everything OUTSIDE the timed brackets ---- */
    char prompt[8192];
    if (!load_placeholder_prompt(fixture, prompt, sizeof prompt)) {
        fprintf(stderr, "cannot read the placeholder prompt fixture: %s\n", fixture);
        return 1;
    }

    tokenizer *tok = NULL;
    if (tok_load(TIE_MODEL_DIR "/tokenizer.json", &tok) != TOK_OK) {
        fprintf(stderr, "tokenizer load failed\n");
        return 1;
    }
    model *m = NULL;
    model_status rc = model_load(TIE_MODEL_DIR "/model.safetensors",
                                 TIE_REPO_DIR "/src/gpt2_tensor_inventory.json",
                                 TIE_MODEL_DIR "/config.json",
                                 MODEL_CPROJ_AS_STORED, &m);
    if (rc != MODEL_OK) { fprintf(stderr, "model load: %s\n", model_strerror(rc)); return 1; }
    const model_config *cfg = model_config_of(m);

    int32_t *ids = (int32_t *)malloc((strlen(prompt) + 1) * sizeof(int32_t));
    size_t n_ids = 0;
    if (tok_encode(tok, (const unsigned char *)prompt, strlen(prompt),
                   ids, strlen(prompt) + 1, &n_ids) != TOK_OK) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }
    printf("placeholder prompt encodes to %zu tokens (D3 open, owned by Stage 3; W11)\n", n_ids);

    const int lengths[] = { 32, 64 };
    const int contexts[] = { 32, 64, 128 };
    const size_t V = (size_t)cfg->vocab_size;
    size_t max_needed = 128;
    if (n_ids < max_needed) {
        fprintf(stderr, "the placeholder prompt is %zu tokens; the longest configuration "
                        "needs %zu. Not padding and not shortening: fix the fixture.\n",
                n_ids, max_needed);
        return 1;
    }

    float *logits = (float *)malloc(64 * V * sizeof(float));
    if (!logits) { fprintf(stderr, "out of memory for logits\n"); return 1; }
    model_reserve(m, max_needed);

    /* ---- the time-budget probe: ONE untimed iteration at the shortest
     *      configuration, so the sample count is chosen from a measured
     *      iteration cost rather than from a guess ---- */
    {
        fwd_ctx c = { m, ids, 32, logits, 32 * V };
        double t0 = bench_cpu_time_seconds();
        model_prefill(m, c.ids, c.n, c.logits, c.cap);
        double t = bench_cpu_time_seconds() - t0;
        printf("probe: one untimed prefill at L=32 took %.6f s\n", t);
        printf("probe: (warmup %d + samples %d) x %.6f s = %.1f s for that configuration\n",
               warmup, samples, t, (warmup + samples) * t);
    }
    if (probe_only) {
        model_free(m); tok_free(tok); free(ids); free(logits);
        return 0;
    }

    static slot slots[MAX_RECORDS];
    static bench_record recs[MAX_RECORDS];
    int n = 0;

    char pinbuf[128];
    snprintf(pinbuf, sizeof pinbuf, "%s", place.detail);

    /* ---------------------------------------------------------- PREFILL ---- */
    for (size_t li = 0; li < sizeof lengths / sizeof *lengths; ++li) {
        int L = lengths[li];
        slot *s = &slots[n];
        fwd_ctx c = { m, ids, (size_t)L, logits, (size_t)L * V };
        int ew = 0, es = 0;
        s->stats = bench_run(prefill_body, &c, warmup, samples, s->samples, &ew, &es);
        snprintf(s->configuration, sizeof s->configuration,
                 "prefill, L=%d tokens, head at every position, no KV cache", L);
        add_num(s, "tokens", L);
        add_num(s, "context_tokens", L);
        add_num(s, "vocab_size", (double)V);
        add_num(s, "n_layer", cfg->n_layer);
        add_num(s, "prompt_set_owning_stage", 3);
        add_num(s, "thread_pinned", place.pinned);
        add_num(s, "thread_logical_cpu", place.logical_cpu);
        add_str(s, "workload", "prefill");
        add_str(s, "matmul", model_gemm_of(m)->name);
        add_str(s, "kv_cache", "none -- Stage 4 adds it");
        add_str(s, "causal_mask", "regenerated as a loop bound; the stored mask buffers are not read");
        add_str(s, "attn_c_proj_reading", "as-stored [input, output]");
        add_str(s, "prompt_set_status", "PLACEHOLDER");
        add_str(s, "prompt_set_open_decision", "D3");
        add_str(s, "prompt_set_work_item", "W11");
        add_str(s, "prompt_set_source", "tests/fixtures/stage2_placeholder_prompts.tsv");
        add_str(s, "thread_placement", pinbuf);
        add_str(s, "timed_bracket_contains",
                "model_prefill only; weight load, tokenizer load, tokenization and scratch "
                "allocation are all outside it");
        add_str(s, "tag", "measured");
        add_drift(s, es);
        s->rec.benchmark = "stage2_forward";
        s->rec.configuration = s->configuration;
        s->rec.units = "ms";
        s->rec.value = s->stats.median;
        s->rec.warmup = ew;
        s->rec.samples_requested = samples;
        s->rec.samples_ms = s->samples;
        s->rec.n_samples = es;
        s->rec.stats = s->stats;
        s->rec.meta_str = s->str; s->rec.n_meta_str = s->n_str;
        s->rec.meta_num = s->num; s->rec.n_meta_num = s->n_num;
        recs[n] = s->rec;
        printf("prefill L=%3d : median %10.3f ms  min %10.3f  max %10.3f  stddev %6.3f%%  %s\n",
               L, s->stats.median, s->stats.min, s->stats.max,
               s->stats.stddev_pct_of_median, s->stats.valid ? "VALID" : "INVALID");
        printf("               drift (DIAGNOSTIC): first half %10.3f ms, second half %10.3f ms, "
               "%+.3f%%\n", s->num[s->n_num - 5].value, s->num[s->n_num - 4].value,
               s->num[s->n_num - 3].value);
        ++n;
    }

    /* ----------------------------------------------------------- DECODE ---- */
    for (size_t ci = 0; ci < sizeof contexts / sizeof *contexts; ++ci) {
        int C = contexts[ci];
        slot *s = &slots[n];
        fwd_ctx c = { m, ids, (size_t)C, logits, V };
        int ew = 0, es = 0;
        s->stats = bench_run(decode_body, &c, warmup, samples, s->samples, &ew, &es);
        snprintf(s->configuration, sizeof s->configuration,
                 "decode, one step at context c=%d tokens, head at the last position only, "
                 "no KV cache", C);
        add_num(s, "context_tokens", C);
        add_num(s, "tokens_generated", 1);
        add_num(s, "vocab_size", (double)V);
        add_num(s, "n_layer", cfg->n_layer);
        add_num(s, "prompt_set_owning_stage", 3);
        add_num(s, "thread_pinned", place.pinned);
        add_num(s, "thread_logical_cpu", place.logical_cpu);
        add_str(s, "workload", "decode");
        add_str(s, "matmul", model_gemm_of(m)->name);
        add_str(s, "kv_cache", "none -- a decode step re-runs the whole forward pass");
        add_str(s, "causal_mask", "regenerated as a loop bound; the stored mask buffers are not read");
        add_str(s, "attn_c_proj_reading", "as-stored [input, output]");
        add_str(s, "prompt_set_status", "PLACEHOLDER");
        add_str(s, "prompt_set_open_decision", "D3");
        add_str(s, "prompt_set_work_item", "W11");
        add_str(s, "prompt_set_source", "tests/fixtures/stage2_placeholder_prompts.tsv");
        add_str(s, "thread_placement", pinbuf);
        add_str(s, "timed_bracket_contains", "model_decode_step only");
        add_str(s, "tag", "measured");
        add_drift(s, es);
        s->rec.benchmark = "stage2_forward";
        s->rec.configuration = s->configuration;
        s->rec.units = "ms";
        s->rec.value = s->stats.median;
        s->rec.warmup = ew;
        s->rec.samples_requested = samples;
        s->rec.samples_ms = s->samples;
        s->rec.n_samples = es;
        s->rec.stats = s->stats;
        s->rec.meta_str = s->str; s->rec.n_meta_str = s->n_str;
        s->rec.meta_num = s->num; s->rec.n_meta_num = s->n_num;
        recs[n] = s->rec;
        printf("decode  c=%3d : median %10.3f ms  min %10.3f  max %10.3f  stddev %6.3f%%  %s\n",
               C, s->stats.median, s->stats.min, s->stats.max,
               s->stats.stddev_pct_of_median, s->stats.valid ? "VALID" : "INVALID");
        printf("               drift (DIAGNOSTIC): first half %10.3f ms, second half %10.3f ms, "
               "%+.3f%%\n", s->num[s->n_num - 5].value, s->num[s->n_num - 4].value,
               s->num[s->n_num - 3].value);
        ++n;
    }

    /* ------------------------------------------------- ISOLATED GEMM ------ */
    /* One shape family at two working-set sizes that sit in different cache
     * tiers, so the gap explanation has evidence for separating a scalar
     * issue-rate limit from a memory-hierarchy limit. M and K are held fixed
     * and only N moves, so the two records differ in the size of the operand
     * being streamed and in nothing else.
     *   N =  768 : B is 768 x  768 x 4 B = 2.25 MiB -- inside the 8 MiB L3
     *   N = 3072 : B is 768 x 3072 x 4 B = 9.00 MiB -- outside it
     * Both are real model shapes: the attention output projection and the
     * feed-forward expansion, at a prefill of 32 tokens. */
    {
        const int Ns[] = { 768, 3072 };
        const char *shape_names[] = {
            "attn.c_proj shape, B = 2.25 MiB, L3-resident",
            "mlp.c_fc shape, B = 9.00 MiB, exceeds the 8 MiB L3"
        };
        const int M = 32, K = 768;
        for (size_t gi = 0; gi < sizeof Ns / sizeof *Ns; ++gi) {
            int N = Ns[gi];
            float *A = (float *)malloc((size_t)M * K * sizeof(float));
            float *B = (float *)malloc((size_t)K * N * sizeof(float));
            float *Cm = (float *)malloc((size_t)M * N * sizeof(float));
            if (!A || !B || !Cm) { fprintf(stderr, "out of memory for the GEMM case\n"); return 1; }
            unsigned int st = 2026u;
            for (size_t i = 0; i < (size_t)M * K; ++i) {
                st = st * 1664525u + 1013904223u;
                A[i] = (float)((double)(st >> 8) / 8388608.0 - 1.0);
            }
            for (size_t i = 0; i < (size_t)K * N; ++i) {
                st = st * 1664525u + 1013904223u;
                B[i] = (float)((double)(st >> 8) / 8388608.0 - 1.0);
            }

            slot *s = &slots[n];
            gemm_ctx g = { M, N, K, A, B, Cm };
            int ew = 0, es = 0;
            s->stats = bench_run(gemm_body, &g, warmup, samples, s->samples, &ew, &es);
            double flops = 2.0 * M * N * K;
            double gflops = s->stats.median > 0 ? flops / (s->stats.median * 1e-3) / 1e9 : 0.0;
            snprintf(s->configuration, sizeof s->configuration,
                     "isolated gemm_naive, M=%d N=%d K=%d, %s", M, N, K, shape_names[gi]);
            add_num(s, "M", M); add_num(s, "N", N); add_num(s, "K", K);
            add_num(s, "flops", flops);
            add_num(s, "b_operand_bytes", (double)K * N * 4.0);
            add_num(s, "a_operand_bytes", (double)M * K * 4.0);
            add_num(s, "c_operand_bytes", (double)M * N * 4.0);
            add_num(s, "gflops", gflops);
            add_num(s, "thread_pinned", place.pinned);
            add_num(s, "thread_logical_cpu", place.logical_cpu);
            add_str(s, "workload", "isolated_gemm");
            add_str(s, "matmul", "gemm_naive");
            add_str(s, "loop_order", "ijk, dot-product inner loop, one accumulator, scalar");
            add_str(s, "thread_placement", pinbuf);
            add_str(s, "timed_bracket_contains", "the gemm_naive call only; operand fill is outside it");
            add_str(s, "flop_count_derivation", "2 * M * N * K");
            add_str(s, "tag", "measured");
            add_drift(s, es);
            s->rec.benchmark = "stage2_forward";
            s->rec.configuration = s->configuration;
            s->rec.units = "ms";
            s->rec.value = s->stats.median;
            s->rec.warmup = ew;
            s->rec.samples_requested = samples;
            s->rec.samples_ms = s->samples;
            s->rec.n_samples = es;
            s->rec.stats = s->stats;
            s->rec.meta_str = s->str; s->rec.n_meta_str = s->n_str;
            s->rec.meta_num = s->num; s->rec.n_meta_num = s->n_num;
            recs[n] = s->rec;
            printf("gemm  N=%4d : median %10.3f ms  %7.3f GFLOP/s  stddev %6.3f%%  %s\n",
                   N, s->stats.median, gflops, s->stats.stddev_pct_of_median,
                   s->stats.valid ? "VALID" : "INVALID");
            printf("               drift (DIAGNOSTIC): first half %10.3f ms, second half %10.3f ms, "
                   "%+.3f%%\n", s->num[s->n_num - 5].value, s->num[s->n_num - 4].value,
                   s->num[s->n_num - 3].value);
            ++n;
            free(A); free(B); free(Cm);
        }
    }

    if (bench_write_results(NULL, "stage2_forward", recs, n) != 0) {
        fprintf(stderr, "writing the results file FAILED\n");
        return 1;
    }
    char dir[512];
    printf("results written: %s/stage2_forward.json (stage id \"%s\", %d records)\n",
           bench_default_results_dir(dir, sizeof dir), bench_stage_id(), n);

    bench_restore_current_thread();
    free(logits); free(ids);
    model_free(m);
    tok_free(tok);

    int invalid = 0;
    for (int i = 0; i < n; ++i) if (!recs[i].stats.valid) ++invalid;
    if (invalid) printf("%d of %d configurations are INVALID and are reported as invalid\n",
                        invalid, n);
    return 0;
}
