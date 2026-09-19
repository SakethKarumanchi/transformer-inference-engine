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
