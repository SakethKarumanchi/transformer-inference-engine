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
 *
 * STAGE 3 AMENDMENT -- PARAMETERISATION ONLY, 2026-10-04. Stage 3 resolved D3
 * and needed this driver pointed at the fixed prompt set and at the four D3
 * prefill lengths. Per that stage's DETERMINE #10 the harness WRAPS this driver
 * as a subprocess rather than replacing it: the timing code stays here, in
 * bench_common, where it is already unit-tested, and "replaces" means the
 * harness becomes the top-level ORCHESTRATOR. What changed, and nothing else:
 *
 *   - the fixture loader reads EVERY row rather than only the first, and takes
 *     the prompt text from the LAST tab-separated column, so it reads both the
 *     Stage 2 placeholder layout (id, intended_use, text) and the D3 layout
 *     (id, target_tokens, verified_tokens, text) without a second parser;
 *   - --lengths and --contexts accept the configuration list, defaulting to the
 *     Stage 2 values so a bare invocation reproduces Stage 2 exactly;
 *   - --out sets the results file stem, default unchanged;
 *   - --prompt-set-status / --prompt-set-decision / --prompt-set-work-item
 *     label the set, defaulting to the Stage 2 PLACEHOLDER values, and
 *     prompt_set_source now reports the fixture actually read rather than a
 *     hardcoded path;
 *   - --probe reports one untimed iteration of EVERY requested configuration,
 *     in a parseable form, because the harness selects its W3 construction from
 *     a measured probe per configuration;
 *   - a row whose target count is declared is CHECKED against what the
 *     tokenizer produces and the run hard-fails on a mismatch, rather than
 *     padding or truncating an inexact row into place.
 *
 * NOT CHANGED: bench_run, the timed bodies, the bracket contents, the warmup
 * and sample handling, the statistics, the 5%-of-median validity rule, the
 * drift diagnostic, and every output field that already existed.
 *
 * STAGE 4 AMENDMENT -- THE TWO ENGINE PATHS AND THE W3 REFUSAL, 2026-10-05.
 * Extended in place again, for the same reason: the timing code is here and in
 * bench_common, where it is unit-tested, and a second driver would be a second
 * place for the bracket to be wrong. What changed, and nothing else:
 *
 *   - --prefill-path and --decode-path select the engine path, each DEFAULTING
 *     to nocache so a bare invocation reproduces Stage 3 exactly;
 *   - --configs takes an explicit ORDERED configuration list, which is how the
 *     paired interleaved order (cache, then its no-cache control, at each
 *     length) is expressed in ONE invocation. One invocation runs every
 *     configuration, so the order the driver is given is the order the machine
 *     sees, and session drift falls on both members of a pair;
 *   - --repeat N per W3: N = 1 is accepted and RECORDED as applied; N > 1 is
 *     REFUSED with a non-zero exit, because the batched C bracket is not
 *     implemented and dividing an unconfirmed bracket by R would fabricate a
 *     per-iteration figure. Every record carries repeat_requested and
 *     repeat_applied as integers so the harness can check rather than trust;
 *   - for cached decode the cache is pre-filled with the first c-1 tokens
 *     OUTSIDE every bracket, and each iteration restores the length to c-1
 *     OUTSIDE the bracket before timing the single step;
 *   - before a cached decode configuration's warmup begins, its step's logits
 *     are asserted BIT-IDENTICAL to the no-cache step's at the same context,
 *     and a mismatch ABORTS that configuration with a non-zero exit rather
 *     than timing a wrong computation.
 *
 * STILL NOT CHANGED: bench_run, the warmup and sample handling, the
 * statistics, the validity rule, the drift diagnostic, and the contents of
 * every timed bracket apart from WHICH model call is made inside it.
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
/* Raised from 16 by Stage 4: the configuration set is now explicit and carries
 * a cache and a no-cache member per length. This is a slot count, not a
 * protocol value -- the warmup, sample count and validity rule are untouched. */
#define MAX_RECORDS 32

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

/* THE CACHED DECODE STEP. The cache length is restored to c-1 here, BEFORE t0,
 * so the restoration is outside the timed bracket and every timed iteration
 * runs at the same context against the same cached positions. The bracket
 * contains one model call and nothing else, exactly as the two bodies above. */
static double decode_cached_body(void *vc, int iteration)
{
    fwd_ctx *c = (fwd_ctx *)vc;
    (void)iteration;
    model_kv_set_length(c->m, (int)c->n - 1);
    double t0 = bench_cpu_time_seconds();
    model_decode_step_cached(c->m, c->ids, c->n, c->logits, c->cap);
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

/* ---------------------------------------------------------- the fixture ---- */
/* Reads EVERY data row. The prompt text is the LAST tab-separated column, which
 * is what makes one parser serve both fixture layouts: the Stage 2 placeholder
 * file is (id, intended_use, text) and the D3 fixed set is (id, target_tokens,
 * verified_tokens, text). The second column is taken as a declared target token
 * count when it parses as a positive integer and as "no declared count" when it
 * does not, which is how a placeholder row keeps its truncate-to-fit behaviour
 * while a D3 row is held to an exact count. Read once, outside every timed
 * bracket. */
#define MAX_ROWS   16
#define MAX_PROMPT 8192

typedef struct {
    char     id[64];
    int      target_tokens;     /* 0 = none declared; truncate to fit */
    char     text[MAX_PROMPT];
    int32_t *ids;
    size_t   n_ids;
} fixture_row;

static int parse_positive_int(const char *s, int *out)
{
    if (!s || !*s) return 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || v <= 0) return 0;
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0') return 0;
    *out = (int)v;
    return 1;
}

static int load_fixture_rows(const char *path, fixture_row *rows, int cap, int *n_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char line[MAX_PROMPT + 512];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;

        /* Split on tabs in place. The text is the last field. */
        char *fields[8];
        int nf = 0;
        fields[nf++] = line;
        for (char *p = line; *p && nf < 8; ++p) {
            if (*p == '\t') { *p = '\0'; fields[nf++] = p + 1; }
        }
        if (nf < 3) continue;                   /* not a data row */
        if (n >= cap) { fclose(f); return 0; }  /* more rows than slots: a real failure */

        const char *text = fields[nf - 1];
        if (*text == '\0' || strlen(text) + 1 > MAX_PROMPT) { fclose(f); return 0; }
        snprintf(rows[n].id, sizeof rows[n].id, "%s", fields[0]);
        rows[n].target_tokens = 0;
        parse_positive_int(fields[1], &rows[n].target_tokens);
        memcpy(rows[n].text, text, strlen(text) + 1);
        rows[n].ids = NULL;
        rows[n].n_ids = 0;
        ++n;
    }
    fclose(f);
    *n_out = n;
    return n > 0;
}

/* Selects the row that serves a configuration of `tokens` tokens: the row whose
 * declared target count equals it, else the first row with no declared count,
 * whose id sequence is then truncated to fit. Returns NULL when neither exists. */
static fixture_row *row_for(fixture_row *rows, int n, int tokens)
{
    for (int i = 0; i < n; ++i)
        if (rows[i].target_tokens == tokens && rows[i].n_ids >= (size_t)tokens) return &rows[i];
    for (int i = 0; i < n; ++i)
        if (rows[i].target_tokens == 0 && rows[i].n_ids >= (size_t)tokens) return &rows[i];
    return NULL;
}

/* Parses "16,32,64,128" into out[]. Returns the count, or -1 on a bad list. */
static int parse_int_list(const char *s, int *out, int cap)
{
    int n = 0;
    while (*s && n < cap) {
        char *end = NULL;
        long v = strtol(s, &end, 10);
        if (end == s || v <= 0) return -1;
        out[n++] = (int)v;
        s = end;
        while (*s == ',' || *s == ' ') ++s;
    }
    return (*s == '\0') ? n : -1;
}

/* ------------------------------------------------ the configuration list ----
 * ONE ORDERED LIST, so the order the machine sees is the order the caller asked
 * for. Stage 4 needs a cache configuration and its no-cache control adjacent at
 * each length, which neither --lengths nor --contexts can express on its own.
 * A bare invocation builds this list from --lengths, --contexts and the two
 * path flags, all of which default to their pre-Stage-4 values, so the default
 * list is the Stage 3 list in the Stage 3 order. */

typedef enum { CFG_PREFILL = 0, CFG_DECODE = 1, CFG_GEMM = 2 } cfg_kind;

typedef struct {
    cfg_kind kind;
    int      tokens;   /* prefill L, decode c, or the GEMM's N */
    int      cache;    /* 1 = the cached engine path; ignored for CFG_GEMM */
} cfg_item;

static const char *path_name(int cache) { return cache ? "cache" : "nocache"; }

/* Parses "prefill:16:cache,prefill:16:nocache,decode:32:cache,gemm:768".
 * Returns the count, or -1 naming nothing -- the caller prints the usage. */
static int parse_cfg_list(const char *s, cfg_item *out, int cap)
{
    int n = 0;
    char buf[1024];
    if (strlen(s) + 1 > sizeof buf) return -1;
    memcpy(buf, s, strlen(s) + 1);

    char *save = buf;
    while (*save) {
        char *item = save;
        char *comma = strchr(save, ',');
        if (comma) { *comma = '\0'; save = comma + 1; } else { save = item + strlen(item); }
        if (*item == '\0') continue;
        if (n >= cap) return -1;

        char *c1 = strchr(item, ':');
        if (!c1) return -1;
        *c1 = '\0';
        char *rest = c1 + 1;
        char *c2 = strchr(rest, ':');
        if (c2) *c2 = '\0';

        int tokens = 0;
        if (!parse_positive_int(rest, &tokens)) return -1;

        if (!strcmp(item, "gemm")) {
            if (c2) return -1;                     /* a GEMM has no engine path */
            out[n].kind = CFG_GEMM; out[n].cache = 0;
        } else if (!strcmp(item, "prefill") || !strcmp(item, "decode")) {
            out[n].kind = !strcmp(item, "prefill") ? CFG_PREFILL : CFG_DECODE;
            if (!c2) return -1;                    /* the path is required here */
            if (!strcmp(c2 + 1, "cache")) out[n].cache = 1;
            else if (!strcmp(c2 + 1, "nocache")) out[n].cache = 0;
            else return -1;
        } else {
            return -1;
        }
        out[n].tokens = tokens;
        ++n;
    }
    return n > 0 ? n : -1;
}

/* "cache"/"nocache" -> 1/0, or -1 for anything else. */
static int parse_path(const char *s)
{
    if (!strcmp(s, "cache")) return 1;
    if (!strcmp(s, "nocache")) return 0;
    return -1;
}

int main(int argc, char **argv)
{
    int warmup = 25;            /* BENCHMARK_PROTOCOL.md section 4.2, adjusted value */
    int samples = 30;           /* section 3: 30 where the budget allows, never below 20 */
    int probe_only = 0;
    const char *fixture = TIE_REPO_DIR "/tests/fixtures/stage2_placeholder_prompts.tsv";
    const char *out_stem = "stage2_forward";
    const char *ps_status = "PLACEHOLDER";
    const char *ps_decision = "D3";
    const char *ps_work_item = "W11";

    /* Stage 2's own configuration set, which stays the default so that a bare
     * invocation reproduces Stage 2 exactly. */
    int lengths[MAX_RECORDS]  = { 32, 64 };          int n_lengths  = 2;
    int contexts[MAX_RECORDS] = { 32, 64, 128 };     int n_contexts = 3;

    /* Stage 4. Both default to the pre-Stage-4 path, so a bare invocation runs
     * the Stage 2 / Stage 3 engine and nothing about it moves. */
    int prefill_cache = 0, decode_cache = 0;
    int repeat_requested = 1;
    cfg_item cfgs[MAX_RECORDS];
    int n_cfgs = 0;                                  /* 0 = build the default list */

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc)  { warmup  = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--samples") && i + 1 < argc) { samples = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--probe")) { probe_only = 1; continue; }
        if (!strcmp(argv[i], "--fixture") && i + 1 < argc) { fixture = argv[++i]; continue; }
        if (!strcmp(argv[i], "--out") && i + 1 < argc) { out_stem = argv[++i]; continue; }
        if (!strcmp(argv[i], "--prompt-set-status") && i + 1 < argc)
            { ps_status = argv[++i]; continue; }
        if (!strcmp(argv[i], "--prompt-set-decision") && i + 1 < argc)
            { ps_decision = argv[++i]; continue; }
        if (!strcmp(argv[i], "--prompt-set-work-item") && i + 1 < argc)
            { ps_work_item = argv[++i]; continue; }
        if (!strcmp(argv[i], "--lengths") && i + 1 < argc) {
            n_lengths = parse_int_list(argv[++i], lengths, MAX_RECORDS);
            if (n_lengths < 0) { fprintf(stderr, "bad --lengths list\n"); return 2; }
            continue;
        }
        if (!strcmp(argv[i], "--contexts") && i + 1 < argc) {
            n_contexts = parse_int_list(argv[++i], contexts, MAX_RECORDS);
            if (n_contexts < 0) { fprintf(stderr, "bad --contexts list\n"); return 2; }
            continue;
        }
        if (!strcmp(argv[i], "--prefill-path") && i + 1 < argc) {
            prefill_cache = parse_path(argv[++i]);
            if (prefill_cache < 0) { fprintf(stderr, "bad --prefill-path\n"); return 2; }
            continue;
        }
        if (!strcmp(argv[i], "--decode-path") && i + 1 < argc) {
            decode_cache = parse_path(argv[++i]);
            if (decode_cache < 0) { fprintf(stderr, "bad --decode-path\n"); return 2; }
            continue;
        }
        if (!strcmp(argv[i], "--configs") && i + 1 < argc) {
            n_cfgs = parse_cfg_list(argv[++i], cfgs, MAX_RECORDS);
            if (n_cfgs < 0) { fprintf(stderr, "bad --configs list\n"); return 2; }
            continue;
        }
        /* W3. R = 1 is the only construction this driver implements. A request
         * for R > 1 is REFUSED here rather than ignored: a driver that silently
         * ran one iteration while the harness divided the bracket by R would
         * report a per-iteration figure that no iteration produced. The C-side
         * batching loop is owned by the first stage that selects R > 1. */
        if (!strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat_requested = atoi(argv[++i]);
            if (repeat_requested != 1) {
                fprintf(stderr,
                        "--repeat %d REFUSED. PERSISTENT.md section 8, W3: the batched C "
                        "bracket is NOT IMPLEMENTED in this driver. Only --repeat 1 is "
                        "accepted, and it is recorded as applied. No figure is produced "
                        "from an unconfirmed repeat.\n", repeat_requested);
                return 2;
            }
            continue;
        }
        fprintf(stderr,
                "usage: %s [--warmup N] [--samples N] [--probe] [--fixture PATH]\n"
                "          [--lengths L,L,...] [--contexts C,C,...] [--out STEM]\n"
                "          [--prompt-set-status S] [--prompt-set-decision D]\n"
                "          [--prompt-set-work-item W]\n"
                "          [--prefill-path cache|nocache] [--decode-path cache|nocache]\n"
                "          [--configs prefill:16:cache,decode:32:nocache,gemm:768,...]\n"
                "          [--repeat 1]\n"
                "an unrecognised flag is REFUSED with a non-zero exit, never ignored.\n",
                argv[0]);
        return 2;
    }
    if (samples > MAX_SAMPLES) samples = MAX_SAMPLES;

    /* The default configuration list: the Stage 3 set in the Stage 3 order, at
     * whatever paths the two path flags name (both nocache unless asked). */
    if (n_cfgs == 0) {
        for (int i = 0; i < n_lengths && n_cfgs < MAX_RECORDS; ++i) {
            cfgs[n_cfgs].kind = CFG_PREFILL;
            cfgs[n_cfgs].tokens = lengths[i];
            cfgs[n_cfgs].cache = prefill_cache;
            ++n_cfgs;
        }
        for (int i = 0; i < n_contexts && n_cfgs < MAX_RECORDS; ++i) {
            cfgs[n_cfgs].kind = CFG_DECODE;
            cfgs[n_cfgs].tokens = contexts[i];
            cfgs[n_cfgs].cache = decode_cache;
            ++n_cfgs;
        }
        cfgs[n_cfgs].kind = CFG_GEMM; cfgs[n_cfgs].tokens = 768;  cfgs[n_cfgs].cache = 0; ++n_cfgs;
        cfgs[n_cfgs].kind = CFG_GEMM; cfgs[n_cfgs].tokens = 3072; cfgs[n_cfgs].cache = 0; ++n_cfgs;
    }

    /* Whatever the list came from, the probe pass and the timed pass walk the
     * SAME list in the SAME order, so --lengths and --contexts are rebuilt from
     * it for the row-availability check below. */
    n_lengths = 0; n_contexts = 0;
    for (int i = 0; i < n_cfgs; ++i) {
        if (cfgs[i].kind == CFG_PREFILL) {
            int seen = 0;
            for (int j = 0; j < n_lengths; ++j) if (lengths[j] == cfgs[i].tokens) seen = 1;
            if (!seen) lengths[n_lengths++] = cfgs[i].tokens;
        } else if (cfgs[i].kind == CFG_DECODE) {
            int seen = 0;
            for (int j = 0; j < n_contexts; ++j) if (contexts[j] == cfgs[i].tokens) seen = 1;
            if (!seen) contexts[n_contexts++] = cfgs[i].tokens;
        }
    }

    /* ---- placement, applied and RECORDED, outside every timed bracket ---- */
    bench_thread_placement place = bench_pin_current_thread(-1);
    printf("thread placement: %s (pinned=%d, logical cpu %d, priority raised=%d)\n",
           place.detail, place.pinned, place.logical_cpu, place.priority_raised);
    printf("cpu timer: %s\n", bench_cpu_timer_name());

    /* ---- load everything OUTSIDE the timed brackets ---- */
    static fixture_row rows[MAX_ROWS];
    int n_rows = 0;
    if (!load_fixture_rows(fixture, rows, MAX_ROWS, &n_rows)) {
        fprintf(stderr, "cannot read the prompt fixture: %s\n", fixture);
        return 1;
    }
    printf("fixture: %s (%d rows, status %s)\n", fixture, n_rows, ps_status);

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

    /* Every row is encoded once, here, outside every timed bracket. A row that
     * declares a target count is CHECKED against it: an inexact row is an error
     * in the fixture, not something to pad or shorten into place. */
    for (int i = 0; i < n_rows; ++i) {
        size_t len = strlen(rows[i].text);
        rows[i].ids = (int32_t *)malloc((len + 1) * sizeof(int32_t));
        if (!rows[i].ids) { fprintf(stderr, "out of memory for the id buffer\n"); return 1; }
        if (tok_encode(tok, (const unsigned char *)rows[i].text, len,
                       rows[i].ids, len + 1, &rows[i].n_ids) != TOK_OK) {
            fprintf(stderr, "encode failed for fixture row %s\n", rows[i].id);
            return 1;
        }
        printf("  row %-8s declares %-4d tokens, the committed C tokenizer encodes %zu%s\n",
               rows[i].id, rows[i].target_tokens, rows[i].n_ids,
               rows[i].target_tokens == 0 ? "  (no declared count; truncated to fit)" : "");
        if (rows[i].target_tokens != 0 && rows[i].n_ids != (size_t)rows[i].target_tokens) {
            fprintf(stderr, "fixture row %s declares %d tokens and encodes to %zu. Not padding "
                            "and not shortening: fix the fixture.\n",
                    rows[i].id, rows[i].target_tokens, rows[i].n_ids);
            return 1;
        }
    }

    const size_t V = (size_t)cfg->vocab_size;

    /* Every requested configuration must have a row that serves it before any
     * iteration runs, so a missing row fails before the machine is loaded
     * rather than halfway through a timed set. */
    size_t max_needed = 0;
    for (int i = 0; i < n_lengths; ++i) {
        if (!row_for(rows, n_rows, lengths[i])) {
            fprintf(stderr, "no fixture row serves a prefill of %d tokens\n", lengths[i]);
            return 1;
        }
        if ((size_t)lengths[i] > max_needed) max_needed = (size_t)lengths[i];
    }
    for (int i = 0; i < n_contexts; ++i) {
        if (!row_for(rows, n_rows, contexts[i])) {
            fprintf(stderr, "no fixture row serves a decode context of %d tokens\n", contexts[i]);
            return 1;
        }
        if ((size_t)contexts[i] > max_needed) max_needed = (size_t)contexts[i];
    }

    float *logits = (float *)malloc(max_needed * V * sizeof(float));
    if (!logits) { fprintf(stderr, "out of memory for logits\n"); return 1; }
    model_reserve(m, max_needed);

    /* ---- the cache, allocated and first-touched ONCE, here, outside every
     *      bracket, and only if some configuration actually asks for it. A run
     *      of no-cache configurations allocates nothing and holds zero cache
     *      bytes. ---- */
    int any_cache = 0;
    for (int i = 0; i < n_cfgs; ++i)
        if (cfgs[i].kind != CFG_GEMM && cfgs[i].cache) any_cache = 1;
    if (any_cache) {
        model_status krc = model_kv_reserve(m, max_needed);
        if (krc != MODEL_OK) {
            fprintf(stderr, "kv cache reserve for %zu positions: %s\n",
                    max_needed, model_strerror(krc));
            return 1;
        }
        printf("kv cache: capacity %d positions, %zu bytes allocated and touched "
               "(= n_layer %d x 2 x capacity x n_embd %d x 4 B), outside every bracket\n",
               model_kv_capacity(m), model_kv_footprint_bytes(m),
               cfg->n_layer, cfg->n_embd);
    } else {
        printf("kv cache: NOT allocated -- every configuration in this run is on the "
               "no-cache path, so the resident cache footprint is 0 bytes\n");
    }

    /* A second logits buffer, for the pre-warmup bit-identity assertion on each
     * cached decode configuration. Allocated here, outside every bracket. */
    float *logits_ref = (float *)malloc(V * sizeof(float));
    if (!logits_ref) { fprintf(stderr, "out of memory for the comparison buffer\n"); return 1; }

    /* ---- the untimed probe. ONE iteration of EVERY requested configuration,
     *      because the harness selects its W3 timing construction per
     *      configuration from a measured probe rather than from an expectation.
     *      A probe is a scheduling input, NOT a measurement, and is labelled so
     *      on every line it appears. ---- */
    /*      The probe walks the configuration list in the SAME ORDER the timed
     *      pass will, and each probe runs the path its configuration names, so
     *      the construction is selected from a probe of the code that will
     *      actually be timed. */
    for (int i = 0; i < n_cfgs; ++i) {
        if (cfgs[i].kind == CFG_GEMM) continue;      /* the GEMM cases are fixed shapes */
        fixture_row *r = row_for(rows, n_rows, cfgs[i].tokens);
        const int tok = cfgs[i].tokens;
        model_set_decode_path(m, cfgs[i].cache ? MODEL_PATH_CACHE : MODEL_PATH_NOCACHE);
        double t;
        if (cfgs[i].kind == CFG_PREFILL) {
            double t0 = bench_cpu_time_seconds();
            model_prefill(m, r->ids, (size_t)tok, logits, (size_t)tok * V);
            t = bench_cpu_time_seconds() - t0;
            printf("probe prefill L=%d row=%s seconds=%.6f path=%s  (NOT A MEASUREMENT) "
                   "budget=(%d+%d)x=%.1f s\n",
                   tok, r->id, t, path_name(cfgs[i].cache),
                   warmup, samples, (warmup + samples) * t);
        } else if (cfgs[i].cache) {
            /* The cache is filled with the first c-1 tokens outside the probe's
             * own bracket, exactly as the timed pass will fill it. */
            model_prefill(m, r->ids, (size_t)tok - 1, logits, ((size_t)tok - 1) * V);
            model_kv_set_length(m, tok - 1);
            double t0 = bench_cpu_time_seconds();
            model_decode_step_cached(m, r->ids, (size_t)tok, logits, V);
            t = bench_cpu_time_seconds() - t0;
            printf("probe decode c=%d row=%s seconds=%.6f path=cache  (NOT A MEASUREMENT) "
                   "budget=(%d+%d)x=%.1f s\n",
                   tok, r->id, t, warmup, samples, (warmup + samples) * t);
        } else {
            double t0 = bench_cpu_time_seconds();
            model_decode_step(m, r->ids, (size_t)tok, logits, V);
            t = bench_cpu_time_seconds() - t0;
            printf("probe decode c=%d row=%s seconds=%.6f path=nocache  (NOT A MEASUREMENT) "
                   "budget=(%d+%d)x=%.1f s\n",
                   tok, r->id, t, warmup, samples, (warmup + samples) * t);
        }
    }
    model_set_decode_path(m, MODEL_PATH_NOCACHE);
    fflush(stdout);
    if (probe_only) {
        model_free(m); tok_free(tok); free(logits); free(logits_ref);
        for (int i = 0; i < n_rows; ++i) free(rows[i].ids);
        return 0;
    }

    static slot slots[MAX_RECORDS];
    static bench_record recs[MAX_RECORDS];
    int n = 0;

    char pinbuf[128];
    snprintf(pinbuf, sizeof pinbuf, "%s", place.detail);

    /* ---- ONE LOOP, IN THE ORDER THE CONFIGURATION LIST GIVES. Prefill,
     *      decode and the isolated GEMM are still three separate record
     *      shapes and are still never combined into one figure; what changed is
     *      that their ORDER is the caller's, so a cache configuration and its
     *      no-cache control are adjacent in time. ---- */
    for (int ci = 0; ci < n_cfgs; ++ci) {
    if (cfgs[ci].kind == CFG_PREFILL) {
        int L = cfgs[ci].tokens;
        const int use_cache = cfgs[ci].cache;
        fixture_row *row = row_for(rows, n_rows, L);
        slot *s = &slots[n];
        fwd_ctx c = { m, row->ids, (size_t)L, logits, (size_t)L * V };
        int ew = 0, es = 0;
        /* The path is selected OUTSIDE the bracket, once per configuration. */
        model_set_decode_path(m, use_cache ? MODEL_PATH_CACHE : MODEL_PATH_NOCACHE);
        s->stats = bench_run(prefill_body, &c, warmup, samples, s->samples, &ew, &es);
        snprintf(s->configuration, sizeof s->configuration,
                 "prefill, L=%d tokens, head at every position, KV cache %s", L,
                 use_cache ? "WRITTEN (cache path)" : "none (no-cache path)");
        add_num(s, "tokens", L);
        add_num(s, "repeat_requested", repeat_requested);
        add_num(s, "repeat_applied", 1);
        add_num(s, "cache_capacity", use_cache ? model_kv_capacity(m) : 0);
        add_num(s, "cache_bytes", use_cache ? (double)model_kv_footprint_bytes(m) : 0.0);
        add_str(s, "prefill_path", path_name(use_cache));
        add_num(s, "context_tokens", L);
        add_num(s, "vocab_size", (double)V);
        add_num(s, "n_layer", cfg->n_layer);
        add_num(s, "prompt_set_owning_stage", 3);
        add_num(s, "thread_pinned", place.pinned);
        add_num(s, "thread_logical_cpu", place.logical_cpu);
        add_str(s, "workload", "prefill");
        add_str(s, "matmul", model_gemm_of(m)->name);
        add_str(s, "kv_cache", use_cache
                ? "WRITTEN: every position's keys and values, per layer, stores only"
                : "none -- the no-cache path, kept selectable by Stage 4");
        add_str(s, "cache_state_construction", use_cache
                ? "allocated and first-touched once, outside every bracket; prefill writes "
                  "positions 0..L-1 inside the timed call as part of the pass"
                : "no cache allocated for this configuration");
        add_str(s, "causal_mask", "regenerated as a loop bound; the stored mask buffers are not read");
        add_str(s, "attn_c_proj_reading", "as-stored [input, output]");
        add_str(s, "prompt_set_status", ps_status);
        add_str(s, "prompt_set_open_decision", ps_decision);
        add_str(s, "prompt_set_work_item", ps_work_item);
        add_str(s, "prompt_set_source", fixture);
        add_str(s, "prompt_row_id", row->id);
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
        printf("prefill L=%3d %-7s : median %10.3f ms  min %10.3f  max %10.3f  "
               "stddev %6.3f%%  %s\n",
               L, path_name(use_cache), s->stats.median, s->stats.min, s->stats.max,
               s->stats.stddev_pct_of_median, s->stats.valid ? "VALID" : "INVALID");
        printf("               drift (DIAGNOSTIC): first half %10.3f ms, second half %10.3f ms, "
               "%+.3f%%\n", s->num[s->n_num - 5].value, s->num[s->n_num - 4].value,
               s->num[s->n_num - 3].value);
        ++n;
    }

    /* ----------------------------------------------------------- DECODE ---- */
    else if (cfgs[ci].kind == CFG_DECODE) {
        int C = cfgs[ci].tokens;
        const int use_cache = cfgs[ci].cache;
        fixture_row *row = row_for(rows, n_rows, C);
        slot *s = &slots[n];
        fwd_ctx c = { m, row->ids, (size_t)C, logits, V };
        int ew = 0, es = 0;

        model_set_decode_path(m, use_cache ? MODEL_PATH_CACHE : MODEL_PATH_NOCACHE);

        if (use_cache) {
            /* ---- EVERYTHING HERE IS OUTSIDE EVERY TIMED BRACKET ----
             * The cache is filled with the first c-1 tokens, and then the
             * cached step's logits are asserted BIT-IDENTICAL to the no-cache
             * step's at the same context. A mismatch aborts THIS configuration
             * with a non-zero exit: timing a computation that gives a different
             * answer would produce a number that means nothing. */
            model_set_decode_path(m, MODEL_PATH_NOCACHE);
            model_status drc = model_decode_step(m, row->ids, (size_t)C, logits_ref, V);
            if (drc != MODEL_OK) {
                fprintf(stderr, "the no-cache reference step at c=%d failed: %s\n",
                        C, model_strerror(drc));
                return 1;
            }
            model_set_decode_path(m, MODEL_PATH_CACHE);
            model_status prc = model_prefill(m, row->ids, (size_t)C - 1, logits,
                                             ((size_t)C - 1) * V);
            if (prc != MODEL_OK) {
                fprintf(stderr, "filling the cache to %d positions failed: %s\n",
                        C - 1, model_strerror(prc));
                return 1;
            }
            model_kv_set_length(m, C - 1);
            drc = model_decode_step_cached(m, row->ids, (size_t)C, logits, V);
            if (drc != MODEL_OK) {
                fprintf(stderr, "the cached step at c=%d failed: %s\n", C, model_strerror(drc));
                return 1;
            }
            size_t differing = 0;
            double maxdiff = 0.0;
            for (size_t v = 0; v < V; ++v)
                if (logits[v] != logits_ref[v]) {
                    ++differing;
                    double d = (double)logits[v] - (double)logits_ref[v];
                    if (d < 0) d = -d;
                    if (d > maxdiff) maxdiff = d;
                }
            if (differing != 0) {
                fprintf(stderr,
                        "ABORTED decode c=%d (cache): the cached step and the no-cache step "
                        "disagree at the same context -- %zu of %zu logits differ, max "
                        "absolute difference %g. This configuration is NOT timed. A faster "
                        "wrong answer is a regression.\n", C, differing, V, maxdiff);
                return 1;
            }
            printf("decode  c=%3d cache   : pre-warmup check PASSED -- the cached step and "
                   "the no-cache step agree BIT FOR BIT over %zu logits\n", C, V);
            fflush(stdout);
        }

        s->stats = bench_run(use_cache ? decode_cached_body : decode_body,
                             &c, warmup, samples, s->samples, &ew, &es);
        snprintf(s->configuration, sizeof s->configuration,
                 "decode, one step at context c=%d tokens, head at the last position only, "
                 "KV cache %s", C, use_cache ? "USED (cache path)" : "none (no-cache path)");
        add_num(s, "context_tokens", C);
        add_num(s, "tokens_generated", 1);
        add_num(s, "repeat_requested", repeat_requested);
        add_num(s, "repeat_applied", 1);
        add_num(s, "cache_capacity", use_cache ? model_kv_capacity(m) : 0);
        add_num(s, "cache_bytes", use_cache ? (double)model_kv_footprint_bytes(m) : 0.0);
        add_num(s, "cached_positions", use_cache ? C - 1 : 0);
        add_str(s, "decode_path", path_name(use_cache));
        add_str(s, "cache_state_construction", use_cache
                ? "length restored outside bracket"
                : "no cache allocated for this configuration");
        add_str(s, "pre_warmup_bit_identity_check", use_cache
                ? "PASSED: the cached step's logits are bit-identical to the no-cache step's "
                  "at the same context; a mismatch aborts the configuration untimed"
                : "not applicable: this configuration IS the no-cache path");
        add_num(s, "vocab_size", (double)V);
        add_num(s, "n_layer", cfg->n_layer);
        add_num(s, "prompt_set_owning_stage", 3);
        add_num(s, "thread_pinned", place.pinned);
        add_num(s, "thread_logical_cpu", place.logical_cpu);
        add_str(s, "workload", "decode");
        add_str(s, "matmul", model_gemm_of(m)->name);
        add_str(s, "kv_cache", use_cache
                ? "USED: one token through the weights, attention over the cached positions"
                : "none -- a decode step re-runs the whole forward pass");
        add_str(s, "causal_mask", "regenerated as a loop bound; the stored mask buffers are not read");
        add_str(s, "attn_c_proj_reading", "as-stored [input, output]");
        add_str(s, "prompt_set_status", ps_status);
        add_str(s, "prompt_set_open_decision", ps_decision);
        add_str(s, "prompt_set_work_item", ps_work_item);
        add_str(s, "prompt_set_source", fixture);
        add_str(s, "prompt_row_id", row->id);
        add_str(s, "thread_placement", pinbuf);
        add_str(s, "timed_bracket_contains", use_cache
                ? "model_decode_step_cached only; the cache allocation, the pre-fill of the "
                  "first c-1 positions and the per-iteration length restoration are all "
                  "outside it"
                : "model_decode_step only");
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
        printf("decode  c=%3d %-7s : median %10.3f ms  min %10.3f  max %10.3f  "
               "stddev %6.3f%%  %s\n",
               C, path_name(use_cache), s->stats.median, s->stats.min, s->stats.max,
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
    else {
        const int M = 32, K = 768;
        {
            int N = cfgs[ci].tokens;
            const char *shape_name =
                (N == 768)  ? "attn.c_proj shape, B = 2.25 MiB, L3-resident" :
                (N == 3072) ? "mlp.c_fc shape, B = 9.00 MiB, exceeds the 8 MiB L3" :
                              "a GEMM shape named on the command line";
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
                     "isolated gemm_naive, M=%d N=%d K=%d, %s", M, N, K, shape_name);
            add_num(s, "M", M); add_num(s, "N", N); add_num(s, "K", K);
            add_num(s, "repeat_requested", repeat_requested);
            add_num(s, "repeat_applied", 1);
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
    }   /* end of the one ordered configuration loop */

    /* The path is left at the default once the timed set is finished, so nothing
     * after this point depends on which configuration ran last. */
    model_set_decode_path(m, MODEL_PATH_NOCACHE);

    if (bench_write_results(NULL, out_stem, recs, n) != 0) {
        fprintf(stderr, "writing the results file FAILED\n");
        return 1;
    }
    char dir[512];
    printf("results written: %s/%s.json (stage id \"%s\", %d records)\n",
           bench_default_results_dir(dir, sizeof dir), out_stem, bench_stage_id(), n);

    bench_restore_current_thread();
    free(logits); free(logits_ref);
    for (int i = 0; i < n_rows; ++i) free(rows[i].ids);
    model_free(m);
    tok_free(tok);

    int invalid = 0;
    for (int i = 0; i < n; ++i) if (!recs[i].stats.valid) ++invalid;
    if (invalid) printf("%d of %d configurations are INVALID and are reported as invalid\n",
                        invalid, n);
    return 0;
}
