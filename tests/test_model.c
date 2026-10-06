/* Unit test for src/model.c -- the Stage 2 forward pass.
 *
 * REDUCED TOKEN COUNTS, STATED. This implementation is slow by design: one
 * token through the twelve blocks and the tied head is roughly a quarter of a
 * GFLOP of scalar work. Every check below therefore runs at 4 to 7 tokens
 * rather than at the 32 and 64 the timed configurations use. The reduction is
 * permitted for correctness tests specifically and applies to NOTHING in the
 * benchmark: the timed configurations are not shortened.
 *
 * None of these checks is a tolerance. BENCHMARK_PROTOCOL.md section 5 leaves
 * the numerical tolerance to Stage 3 (PERSISTENT.md section 1, D2) and this
 * stage does not invent one. What is asserted here are properties that hold
 * exactly or to machine epsilon regardless of any tolerance: a probability
 * distribution sums to one, a normalized row has zero mean and unit variance, a
 * causal model cannot see forward, two routes through the same arithmetic agree
 * bit for bit, and the head is the embedding rather than a copy of it.
 *
 * STAGE 4 ADDITIONS, 2026-10-05, all at the SAME reduced counts (4 to 7 tokens,
 * stated): the cached path against prefill at every position (check 3), the
 * cached step against the no-cache step (check 4), causality through the cached
 * path including the non-vacuous half (check 5), restoration repeatability with
 * the cache bytes compared before and after (check 7), attention rows summing
 * to 1 in the cached path (check 10), the capacity refusals (check 8), and the
 * no-cache path's logits against the PRE-CHANGE engine's dump. Every existing
 * assertion above is unchanged. The check numbers are the Stage 4 prompt's
 * DECISION E numbering, so the entry and the output can be read against each
 * other.
 */
#include "test_util.h"
#include "model.h"
#include "tokenizer.h"

#include <stdint.h>

#ifndef TIE_MODEL_DIR
#define TIE_MODEL_DIR "models/gpt2"
#endif
#ifndef TIE_REPO_DIR
#define TIE_REPO_DIR "."
#endif

static const char *WEIGHTS   = TIE_MODEL_DIR "/model.safetensors";
static const char *CONFIG    = TIE_MODEL_DIR "/config.json";
static const char *TOKJSON   = TIE_MODEL_DIR "/tokenizer.json";
static const char *INVENTORY = TIE_REPO_DIR "/src/gpt2_tensor_inventory.json";

int main(void)
{
    printf("test_model\n");
    printf("  reduced token counts: 4 to 7 tokens per check, stated deliberately; the "
           "timed configurations are NOT reduced\n");

    /* ---- the tokenizer's vocabulary against the config's ---- */
    tokenizer *tok = NULL;
    tok_status trc = tok_load(TOKJSON, &tok);
    CHECK(trc == TOK_OK, "the tokenizer artifact loads: %s", tok_strerror(trc));
    if (trc != TOK_OK) TIE_SUMMARY("test_model");

    model *m = NULL;
    model_status rc = model_load(WEIGHTS, INVENTORY, CONFIG, MODEL_CPROJ_AS_STORED, &m);
    CHECK(rc == MODEL_OK, "the model loads from the weight file, the inventory and the "
          "config: %s", model_strerror(rc));
    if (rc != MODEL_OK) {
        printf("       the weight file is gitignored; a fresh clone must place the "
               "artifacts under models/gpt2/ before the suite can run.\n");
        TIE_SUMMARY("test_model");
    }
    const model_config *c = model_config_of(m);
    CHECK(tok_vocab_size(tok) == (size_t)c->vocab_size,
          "tok_vocab_size %zu equals the config's vocab_size %d",
          tok_vocab_size(tok), c->vocab_size);
    CHECK(c->n_embd == c->n_head * c->head_dim,
          "n_embd %d = n_head %d x head_dim %d", c->n_embd, c->n_head, c->head_dim);

    /* ---- the head is TIED: the same storage, not a copy ---- */
    CHECK(model_head_weight(m) == model_token_embedding(m),
          "the head weights and the token embedding are the SAME STORAGE (tied, "
          "not copied)");

    /* ---- layernorm: zero mean and unit variance BEFORE the affine transform ---- */
    {
        enum { N = 768 };
        float *x = (float *)malloc(N * sizeof(float));
        float *y = (float *)malloc(N * sizeof(float));
        uint32_t s = 11u;
        for (int i = 0; i < N; ++i) {
            s = s * 1664525u + 1013904223u;
            x[i] = (float)((double)(s >> 8) / 8388608.0 - 1.0) * 7.0f + 3.0f;
        }
        model_layernorm_normalize(x, N, c->layer_norm_epsilon, y);
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < N; ++i) mean += y[i];
        mean /= N;
        for (int i = 0; i < N; ++i) var += (y[i] - mean) * (y[i] - mean);
        var /= N;
        CHECK_NEAR(mean, 0.0, 1e-5, "layernorm output has zero mean per row");
        CHECK_NEAR(var, 1.0, 1e-4, "layernorm output has unit variance per row");
        free(x); free(y);
    }

    /* ---- encode a short prompt; every check below uses it ---- */
    const char *text = "The capital city of France is called";
    int32_t ids[64];
    size_t n_ids = 0;
    trc = tok_encode(tok, (const unsigned char *)text, strlen(text), ids, 64, &n_ids);
    CHECK(trc == TOK_OK && n_ids >= 5, "the probe prompt encodes to %zu tokens", n_ids);
    if (trc != TOK_OK || n_ids < 5) TIE_SUMMARY("test_model");
    size_t T = n_ids > 7 ? 7 : n_ids;

    const size_t V = (size_t)c->vocab_size;
    float *logits  = (float *)malloc(T * V * sizeof(float));
    float *logits2 = (float *)malloc(T * V * sizeof(float));
    float *one     = (float *)malloc(V * sizeof(float));
    if (!logits || !logits2 || !one) { printf("  FAIL out of memory\n"); return 1; }

    /* ---- attention weights sum to 1 along the attended axis, for every head,
     *      every position and every layer of one full forward pass ---- */
    model_collect_attn_stats(m, 1);
    rc = model_prefill(m, ids, T, logits, T * V);
    CHECK(rc == MODEL_OK, "prefill over %zu tokens: %s", T, model_strerror(rc));
    {
        double lo = 0, hi = 0; size_t rows = 0;
        model_attn_rowsum_range(m, &lo, &hi, &rows);
        CHECK(rows == (size_t)c->n_layer * (size_t)c->n_head * T,
              "every attention row was checked: %zu rows = %d layers x %d heads x %zu positions",
              rows, c->n_layer, c->n_head, T);
        CHECK(lo > 1.0 - 1e-5 && hi < 1.0 + 1e-5,
              "every attention row sums to 1 along the attended axis "
              "(min %.9f, max %.9f over %zu rows)", lo, hi, rows);
    }
    model_collect_attn_stats(m, 0);

    /* ---- causality: changing the token at position j must not move any logit
     *      at any position i < j. Bit-exact, not approximate: with a causal
     *      mask the earlier rows are computed from identical inputs. ---- */
    {
        int32_t alt[64];
        memcpy(alt, ids, T * sizeof(int32_t));
        size_t j = T - 1;
        alt[j] = (ids[j] + 1234) % c->vocab_size;
        rc = model_prefill(m, alt, T, logits2, T * V);
        CHECK(rc == MODEL_OK, "prefill with the token at position %zu changed: %s",
              j, model_strerror(rc));
        int moved = 0, changed_at_j = 0;
        for (size_t t = 0; t < j; ++t)
            for (size_t v = 0; v < V; ++v)
                if (logits[t * V + v] != logits2[t * V + v]) { moved = 1; break; }
        for (size_t v = 0; v < V; ++v)
            if (logits[j * V + v] != logits2[j * V + v]) { changed_at_j = 1; break; }
        CHECK(!moved, "changing the token at position %zu moved NO logit at any earlier "
              "position (bit-exact over %zu positions x %zu vocabulary)", j, j, V);
        CHECK(changed_at_j, "the logits AT position %zu did change, so the test is not "
              "passing vacuously", j);
    }

    /* ---- prefill/decode self-consistency: with no KV cache, a decode step at
     *      context t+1 must reproduce the prefill logits at position t ---- */
    {
        int exact = 1;
        size_t checked = 0;
        for (size_t t = 3; t < T; ++t) {
            rc = model_decode_step(m, ids, t + 1, one, V);
            if (rc != MODEL_OK) { exact = 0; break; }
            for (size_t v = 0; v < V; ++v)
                if (one[v] != logits[t * V + v]) { exact = 0; break; }
            ++checked;
        }
        CHECK(exact && checked > 0,
              "a decode step at context t+1 reproduces the prefill logits at position t, "
              "bit for bit, at %zu context lengths", checked);
    }

    /* ================================================================
     * STAGE 4 -- THE KV CACHE. Reduced token counts, as above: every check
     * below runs at the same 4 to 7 tokens. `logits` currently holds the
     * NO-CACHE prefill over T tokens and is the reference the cached path is
     * compared against.
     * ================================================================ */

    /* ---- the no-cache path is BYTE FOR BYTE the pre-change engine ----
     * The guard against a cache that quietly changed the path it was supposed
     * to leave alone. The constant below is FNV-1a 64 over the float payload of
     * `gpt2_tool --dump-logits` produced by the MAIN-BRANCH build, before the
     * first edit to src/ in this stage, on this prompt at this token count; the
     * full dump is kept outside the repository and its SHA-256 is recorded in
     * the Stage 4 MEASUREMENTS.md entry. A hash rather than a committed 1.4 MB
     * array: the property being asserted is identity, and identity is what a
     * hash decides. */
    {
        CHECK(model_decode_path_of(m) == MODEL_PATH_NOCACHE,
              "the decode-path switch DEFAULTS to no-cache, so every pre-Stage-4 call site "
              "keeps its behaviour without being changed");

        const int32_t expect_ids[7] = { 464, 3139, 1748, 286, 4881, 318, 1444 };
        int ids_match = (T == 7);
        for (size_t i = 0; i < T && ids_match; ++i)
            if (ids[i] != expect_ids[i]) ids_match = 0;
        CHECK(ids_match, "the probe prompt still encodes to the 7 ids the pre-change dump was "
                         "taken over, so the comparison below is against the same computation");
        if (ids_match) {
            uint64_t h = 0xcbf29ce484222325ull;
            const unsigned char *p = (const unsigned char *)logits;
            for (size_t i = 0; i < T * V * sizeof(float); ++i) {
                h ^= (uint64_t)p[i];
                h *= 0x100000001b3ull;
            }
            CHECK(h == 0x344664b8843fccf8ull,
                  "the no-cache prefill logits are BIT-IDENTICAL to the pre-change engine's "
                  "dump (FNV-1a 64 = 0x%016llx)", (unsigned long long)h);
        }
    }

    /* ---- the cache itself: lazy, so no-cache holds zero bytes ---- */
    {
        CHECK(model_kv_footprint_bytes(m) == 0u && model_kv_cache_of(m) == NULL,
              "with the switch left at no-cache and no reserve call, the model holds NO cache "
              "and its resident cache footprint is exactly 0 bytes");

        rc = model_kv_reserve(m, T + 2);
        CHECK(rc == MODEL_OK, "model_kv_reserve(%zu) allocates the cache: %s",
              T + 2, model_strerror(rc));
        size_t want = (size_t)c->n_layer * 2u * (T + 2) * (size_t)c->n_embd * sizeof(float);
        CHECK(model_kv_footprint_bytes(m) == want,
              "the allocated footprint is n_layer x 2 x capacity x n_embd x 4 = %zu bytes",
              want);
        CHECK(model_kv_capacity(m) == (int)(T + 2), "capacity is recorded as %zu", T + 2);
    }

    float *cached = (float *)malloc(T * V * sizeof(float));
    float *nocache_one = (float *)malloc(V * sizeof(float));
    if (!cached || !nocache_one) { printf("  FAIL out of memory\n"); return 1; }

    /* ---- CHECK 3: the cached path at EVERY position, against prefill ----
     * The matrix is assembled the way gpt2_tool --via-decode assembles it: a
     * one-token prefill, then one cached decode step per remaining position. */
    {
        model_set_decode_path(m, MODEL_PATH_CACHE);
        model_collect_attn_stats(m, 1);
        rc = model_prefill(m, ids, 1, cached, V);
        CHECK(rc == MODEL_OK, "cache-mode prefill over the first token: %s", model_strerror(rc));
        CHECK(model_kv_length(m) == 1,
              "the cache holds 1 position after a one-token prefill");
        for (size_t t = 1; t < T && rc == MODEL_OK; ++t)
            rc = model_decode_step_cached(m, ids, t + 1, cached + t * V, V);
        CHECK(rc == MODEL_OK, "one cached decode step per remaining position: %s",
              model_strerror(rc));
        CHECK(model_kv_length(m) == (int)T,
              "the cache holds all %zu positions after the last step", T);

        size_t differing = 0;
        double maxdiff = 0.0;
        for (size_t t = 0; t < T; ++t)
            for (size_t v = 0; v < V; ++v) {
                float a = logits[t * V + v], b = cached[t * V + v];
                if (a != b) {
                    ++differing;
                    double d = (double)b - (double)a;
                    if (d < 0) d = -d;
                    if (d > maxdiff) maxdiff = d;
                }
            }
        CHECK(differing == 0,
              "CHECK 3: the cached path reproduces the prefill logits at every one of %zu "
              "positions BIT FOR BIT (%zu differing elements of %zu, max abs difference %g)",
              T, differing, T * V, maxdiff);

        /* ---- CHECK 10: attention rows still sum to 1 in the cached path ---- */
        {
            double lo = 0, hi = 0; size_t rows = 0;
            model_attn_rowsum_range(m, &lo, &hi, &rows);
            /* One row per head per layer for the 1-token prefill, plus one per
             * head per layer for each of the T-1 cached steps. */
            size_t expect_rows = (size_t)c->n_layer * (size_t)c->n_head * T;
            CHECK(rows == expect_rows,
                  "CHECK 10: every attention row of the cached assembly was checked: %zu rows "
                  "= %d layers x %d heads x %zu steps", rows, c->n_layer, c->n_head, T);
            CHECK(lo > 1.0 - 1e-5 && hi < 1.0 + 1e-5,
                  "CHECK 10: every attention row in the CACHED path sums to 1 within 1e-5 "
                  "(min %.9f, max %.9f)", lo, hi);
        }
        model_collect_attn_stats(m, 0);
    }

    /* ---- CHECK 4: the cached step against the NO-CACHE step, same context ---- */
    {
        int exact = 1, checked = 0;
        size_t first_bad = 0, differing = 0;
        double maxdiff = 0.0;
        for (size_t ctx = 3; ctx <= T; ++ctx) {
            model_set_decode_path(m, MODEL_PATH_NOCACHE);
            rc = model_decode_step(m, ids, ctx, nocache_one, V);
            if (rc != MODEL_OK) { exact = 0; break; }

            model_set_decode_path(m, MODEL_PATH_CACHE);
            /* The cache is restored to the context the step expects, OUTSIDE
             * anything being compared -- the same restoration the timing driver
             * performs outside its bracket. */
            if (model_kv_set_length(m, (int)ctx - 1) != MODEL_OK) { exact = 0; break; }
            rc = model_decode_step_cached(m, ids, ctx, one, V);
            if (rc != MODEL_OK) { exact = 0; break; }

            for (size_t v = 0; v < V; ++v)
                if (one[v] != nocache_one[v]) {
                    if (!differing) first_bad = ctx;
                    ++differing;
                    double d = (double)one[v] - (double)nocache_one[v];
                    if (d < 0) d = -d;
                    if (d > maxdiff) maxdiff = d;
                }
            ++checked;
        }
        CHECK(exact && checked > 0 && differing == 0,
              "CHECK 4: at %d contexts the cached step and the no-cache step agree BIT FOR BIT "
              "(%zu differing elements, max abs difference %g, first disagreement at context "
              "%zu)", checked, differing, maxdiff, first_bad);
    }

    /* ---- CHECK 7: restoration repeatability, which the driver relies on ----
     * Fill to c-1, run a step, restore the length, run it again: the two logit
     * vectors must be bit-identical AND the cache contents at positions 0..c-2
     * must be byte-identical before and after. The second is the one that
     * matters: a step that corrupted an earlier cached position would still
     * produce the same logits once and only diverge on the iteration after. */
    {
        const size_t ctx = T;
        model_set_decode_path(m, MODEL_PATH_CACHE);
        rc = model_prefill(m, ids, ctx - 1, cached, (ctx - 1) * V);
        CHECK(rc == MODEL_OK, "cache-mode prefill to length %zu: %s", ctx - 1,
              model_strerror(rc));

        const kv_cache *kv = model_kv_cache_of(m);
        const size_t head_bytes = (ctx - 1) * (size_t)c->n_embd * sizeof(float);
        float *snap_k = (float *)malloc(head_bytes * (size_t)c->n_layer);
        float *snap_v = (float *)malloc(head_bytes * (size_t)c->n_layer);
        if (!snap_k || !snap_v) { printf("  FAIL out of memory\n"); return 1; }
        for (int l = 0; l < c->n_layer; ++l) {
            memcpy((char *)snap_k + (size_t)l * head_bytes, kv_cache_keys(kv, l), head_bytes);
            memcpy((char *)snap_v + (size_t)l * head_bytes, kv_cache_values(kv, l), head_bytes);
        }

        rc = model_decode_step_cached(m, ids, ctx, one, V);
        CHECK(rc == MODEL_OK, "the first cached step at context %zu: %s", ctx,
              model_strerror(rc));
        rc = model_kv_set_length(m, (int)ctx - 1);
        CHECK(rc == MODEL_OK, "the length is restored to %zu between the two steps", ctx - 1);
        rc = model_decode_step_cached(m, ids, ctx, nocache_one, V);
        CHECK(rc == MODEL_OK, "the repeated cached step at the same context: %s",
              model_strerror(rc));

        size_t differing = 0;
        for (size_t v = 0; v < V; ++v) if (one[v] != nocache_one[v]) ++differing;
        CHECK(differing == 0,
              "CHECK 7: a step, a length restoration and the same step again produce "
              "BIT-IDENTICAL logits (%zu differing elements)", differing);

        size_t moved = 0;
        for (int l = 0; l < c->n_layer; ++l) {
            if (memcmp((char *)snap_k + (size_t)l * head_bytes,
                       kv_cache_keys(kv, l), head_bytes) != 0) ++moved;
            if (memcmp((char *)snap_v + (size_t)l * head_bytes,
                       kv_cache_values(kv, l), head_bytes) != 0) ++moved;
        }
        CHECK(moved == 0,
              "CHECK 7: the cache contents at positions 0..%zu are BYTE-IDENTICAL before and "
              "after the two steps (%zu of %d buffers moved)", ctx - 2, moved,
              2 * c->n_layer);
        free(snap_k); free(snap_v);
    }

    /* ---- CHECK 5: causality THROUGH THE CACHED PATH, non-vacuously ---- */
    {
        const size_t ctx = T;
        int32_t alt[64];
        memcpy(alt, ids, ctx * sizeof(int32_t));
        const size_t j = ctx - 1;
        alt[j] = (ids[j] + 1234) % c->vocab_size;

        model_set_decode_path(m, MODEL_PATH_CACHE);
        /* The unchanged sequence through the cached path, assembled position by
         * position, then the same with the token at j replaced. */
        rc = model_prefill(m, ids, 1, cached, V);
        for (size_t t = 1; t < ctx && rc == MODEL_OK; ++t)
            rc = model_decode_step_cached(m, ids, t + 1, cached + t * V, V);
        CHECK(rc == MODEL_OK, "the cached assembly of the unchanged sequence: %s",
              model_strerror(rc));

        float *alt_mat = (float *)malloc(ctx * V * sizeof(float));
        if (!alt_mat) { printf("  FAIL out of memory\n"); return 1; }
        rc = model_prefill(m, alt, 1, alt_mat, V);
        for (size_t t = 1; t < ctx && rc == MODEL_OK; ++t)
            rc = model_decode_step_cached(m, alt, t + 1, alt_mat + t * V, V);
        CHECK(rc == MODEL_OK, "the cached assembly with position %zu changed: %s", j,
              model_strerror(rc));

        size_t moved = 0, changed_at_j = 0;
        for (size_t t = 0; t < j; ++t)
            for (size_t v = 0; v < V; ++v)
                if (cached[t * V + v] != alt_mat[t * V + v]) ++moved;
        for (size_t v = 0; v < V; ++v)
            if (cached[j * V + v] != alt_mat[j * V + v]) ++changed_at_j;
        CHECK(moved == 0,
              "CHECK 5: through the CACHED path, changing the token at position %zu moved NO "
              "logit at any earlier position (bit-exact, %zu differing elements over %zu "
              "positions x %zu vocabulary)", j, moved, j, V);
        CHECK(changed_at_j > 0,
              "CHECK 5: the logits AT position %zu did move (%zu elements), so the causality "
              "check is not passing vacuously", j, changed_at_j);
        free(alt_mat);
    }

    /* ---- CHECK 8: a context beyond capacity is refused before arithmetic ----
     * ONE extra model load serves this whole block, with a deliberately small
     * cache. A second full load costs about half a gigabyte of resident weights
     * and several seconds, so the bounds are checked on one model with capacity
     * 3 rather than on a fresh model per case. */
    {
        model *small = NULL;
        rc = model_load(WEIGHTS, INVENTORY, CONFIG, MODEL_CPROJ_AS_STORED, &small);
        CHECK(rc == MODEL_OK, "a model with a deliberately small cache for the bounds check: "
              "%s", model_strerror(rc));
        if (rc == MODEL_OK) {
            model_set_decode_path(small, MODEL_PATH_CACHE);
            model_reserve(small, 8);

            /* Cache mode with NO cache allocated at all: refused, not faulted. */
            CHECK(model_decode_step_cached(small, ids, 2, one, V) == MODEL_ERR_ARG,
                  "CHECK 8: cache mode with no cache allocated is refused, not faulted");
            CHECK(model_prefill(small, ids, 2, cached, 2 * V) == MODEL_ERR_ARG,
                  "CHECK 8: a cache-mode prefill with no cache allocated is refused");

            rc = model_kv_reserve(small, 3);           /* capacity THREE positions */
            CHECK(rc == MODEL_OK && model_kv_capacity(small) == 3,
                  "a cache of capacity 3: %s", model_strerror(rc));
            CHECK(model_decode_step_cached(small, ids, 5, one, V) == MODEL_ERR_CAPACITY,
                  "CHECK 8: a cached decode step at a context beyond capacity returns "
                  "MODEL_ERR_CAPACITY");
            CHECK(model_prefill(small, ids, 5, cached, 5 * V) == MODEL_ERR_CAPACITY,
                  "CHECK 8: a cache-mode prefill longer than the cache's capacity is refused");
            /* The length is untouched by a refusal, so the next legitimate call
             * is not poisoned by the rejected one. */
            CHECK(model_kv_length(small) == 0,
                  "CHECK 8: the refusals left the cache length at 0");
            CHECK(model_kv_set_length(small, 9) == MODEL_ERR_CAPACITY,
                  "CHECK 8: a length restoration beyond capacity is refused");
            /* A cached step whose cache does not hold exactly c-1 positions is
             * refused rather than answered from the wrong context. */
            model_kv_set_length(small, 1);
            CHECK(model_decode_step_cached(small, ids, 3, one, V) == MODEL_ERR_ARG,
                  "CHECK 8: a cached step whose cache holds the wrong number of positions is "
                  "refused rather than run against it");
            model_free(small);
        }
    }

    /* Back to the default for everything that follows. */
    model_set_decode_path(m, MODEL_PATH_NOCACHE);
    free(cached); free(nocache_one);

    /* ---- both readings of the twelve square attn.c_proj tensors are
     *      exercised, so the selected one is selected by observation ---- */
    float *last_as_stored = (float *)malloc(V * sizeof(float));
    memcpy(last_as_stored, logits + (T - 1) * V, V * sizeof(float));
    {
        const model_tensor_record *rec = NULL;
        for (size_t i = 0; i < model_tensor_record_count(m); ++i) {
            const model_tensor_record *r = model_tensor_record_at(m, i);
            if (strstr(r->name, "attn.c_proj")) rec = r;
        }
        CHECK(rec != NULL && rec->transposed_at_load == 0,
              "the default reading records no transpose at load, and the inventory calls it "
              "\"%s\"", rec ? rec->inventory_orientation : "(missing)");
    }
    model_free(m);
    m = NULL;

    rc = model_load(WEIGHTS, INVENTORY, CONFIG, MODEL_CPROJ_TRANSPOSED, &m);
    CHECK(rc == MODEL_OK, "the model also loads under the TRANSPOSED reading of "
          "h.*.attn.c_proj.weight: %s", model_strerror(rc));
    if (rc == MODEL_OK) {
        const model_tensor_record *rec = NULL;
        for (size_t i = 0; i < model_tensor_record_count(m); ++i) {
            const model_tensor_record *r = model_tensor_record_at(m, i);
            if (strstr(r->name, "attn.c_proj")) rec = r;
        }
        CHECK(rec && rec->transposed_at_load == 1,
              "the transposed reading is recorded as a transpose applied AT LOAD, not as a "
              "transpose inside the arithmetic");
        rc = model_prefill(m, ids, T, logits2, T * V);
        CHECK(rc == MODEL_OK, "prefill under the transposed reading: %s", model_strerror(rc));
        int differs = 0;
        double maxdiff = 0.0;
        for (size_t v = 0; v < V; ++v) {
            double d = (double)logits2[(T - 1) * V + v] - (double)last_as_stored[v];
            if (d < 0) d = -d;
            if (d > maxdiff) maxdiff = d;
            if (logits2[(T - 1) * V + v] != last_as_stored[v]) differs = 1;
        }
        CHECK(differs, "the two readings produce DIFFERENT logits (max absolute difference "
              "%.6f), so the choice between them is decidable by observation and was not "
              "assumed", maxdiff);
        model_free(m);
    }

    free(last_as_stored); free(one); free(logits2); free(logits);
    tok_free(tok);
    TIE_SUMMARY("test_model");
}
