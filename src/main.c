/* main.c -- the Stage 2 driver: load, encode, prefill, generate, decode, print.
 *
 * Greedy selection only. No pseudo-random sampling, no seed, no PRNG anywhere
 * in this program. Greedy is what the correctness gate requires -- a token
 * sequence that must match the reference exactly -- and a seed introduced here
 * would become a fixed condition every later stage has to hold constant for no
 * gain at this one.
 *
 * This program times nothing. Timing lives in bench/stage2_forward_bench.c,
 * which brackets computation only. The wall-clock figures printed below are
 * progress output for a human watching a deliberately slow baseline run, are
 * labelled as such, and are never recorded as measurements.
 *
 *   --dump-logits FILE   writes the raw prefill logits for the correctness
 *                        comparison, as a small binary the Python oracle reads:
 *                        8-byte magic "TIE2LOGI", int32 positions, int32
 *                        vocab_size, then positions*vocab_size float32 in row
 *                        order. Text would round the very quantity being
 *                        compared.
 *
 * STAGE 4 AMENDMENT, 2026-10-05. Three flags added; every existing flag keeps
 * its behaviour and a BARE INVOCATION IS UNCHANGED -- it runs the no-cache
 * path, so a Stage 2 or Stage 3 command line reproduces exactly what it did.
 *
 *   --kv-cache on|off    selects the engine path. DEFAULT off, which is what a
 *                        bare invocation did before this stage existed.
 *   --via-decode         with --dump-logits, assembles the positions x vocab
 *                        matrix from the CACHED DECODE PATH ALONE: prefill over
 *                        the first token, then one cached decode step per
 *                        remaining token. Same TIE2LOGI format, so the same
 *                        comparator reads it. Implies --kv-cache on, because
 *                        there is no such thing as a cached step without the
 *                        cache, and says so rather than silently disagreeing
 *                        with an explicit --kv-cache off.
 *   --footprint          prints the cache's allocated bytes and the shapes they
 *                        were computed from, then continues.
 */
#include "model.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef TIE_MODEL_DIR
#define TIE_MODEL_DIR "models/gpt2"
#endif

/* The main() lives behind GPT2_MAIN, following the pattern src/safetensors.c
 * established: one source compiles twice -- into the runnable tool with its
 * main enabled by a define, and into an object library without it. */
#ifdef GPT2_MAIN

static void usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s [options]\n"
        "  --prompt TEXT          prompt text (default: a short English sentence)\n"
        "  --prompt-file FILE     read the prompt from a file instead\n"
        "  --truncate N           keep only the first N prompt tokens\n"
        "  --generate N           greedily generate N tokens (default 0)\n"
        "  --greedy N             synonym for --generate\n"
        "  --dump-logits FILE     write raw prefill logits for every position\n"
        "  --kv-cache on|off      engine path (default off: the no-cache path)\n"
        "  --via-decode           with --dump-logits, assemble the matrix from the\n"
        "                         cached decode path alone; implies --kv-cache on\n"
        "  --footprint            print the KV cache's allocated bytes\n"
        "  --cproj as-stored|transposed   reading of h.*.attn.c_proj.weight\n"
        "  --weights/--config/--inventory/--tokenizer PATH\n"
        "  --records              print the per-tensor orientation reconciliation\n",
        argv0);
}

static char *read_all(const char *path, size_t *n_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    if (n < 0) { fclose(f); return NULL; }
    char *b = (char *)malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    *n_out = fread(b, 1, (size_t)n, f);
    b[*n_out] = '\0';
    fclose(f);
    return b;
}

int main(int argc, char **argv)
{
    const char *weights   = TIE_MODEL_DIR "/model.safetensors";
    const char *config    = TIE_MODEL_DIR "/config.json";
    const char *tokjson   = TIE_MODEL_DIR "/tokenizer.json";
    const char *inventory = "src/gpt2_tensor_inventory.json";
    const char *prompt    = "The capital of France is";
    const char *prompt_file = NULL;
    const char *dump_path = NULL;
    int generate = 0, truncate_to = 0, show_records = 0;
    model_cproj_reading reading = MODEL_CPROJ_AS_STORED;
    /* Stage 4: default OFF, so a bare invocation is the pre-Stage-4 invocation. */
    model_decode_path path = MODEL_PATH_NOCACHE;
    int via_decode = 0, show_footprint = 0, kv_flag_seen = 0;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        int has_next = (i + 1 < argc);
#define ARG(name, var) if (!strcmp(a, name) && has_next) { var = argv[++i]; continue; }
        ARG("--weights",   weights)
        ARG("--config",    config)
        ARG("--inventory", inventory)
        ARG("--tokenizer", tokjson)
        ARG("--prompt",    prompt)
        ARG("--prompt-file", prompt_file)
        ARG("--dump-logits", dump_path)
#undef ARG
        if (!strcmp(a, "--generate") && has_next) { generate = atoi(argv[++i]); continue; }
        if (!strcmp(a, "--greedy") && has_next)   { generate = atoi(argv[++i]); continue; }
        if (!strcmp(a, "--truncate") && has_next) { truncate_to = atoi(argv[++i]); continue; }
        if (!strcmp(a, "--records")) { show_records = 1; continue; }
        if (!strcmp(a, "--via-decode")) { via_decode = 1; continue; }
        if (!strcmp(a, "--footprint")) { show_footprint = 1; continue; }
        if (!strcmp(a, "--kv-cache") && has_next) {
            const char *v = argv[++i];
            kv_flag_seen = 1;
            if (!strcmp(v, "on")) path = MODEL_PATH_CACHE;
            else if (!strcmp(v, "off")) path = MODEL_PATH_NOCACHE;
            else { usage(argv[0]); return 2; }
            continue;
        }
        if (!strcmp(a, "--cproj") && has_next) {
            const char *v = argv[++i];
            if (!strcmp(v, "transposed")) reading = MODEL_CPROJ_TRANSPOSED;
            else if (!strcmp(v, "as-stored")) reading = MODEL_CPROJ_AS_STORED;
            else { usage(argv[0]); return 2; }
            continue;
        }
        usage(argv[0]);
        return 2;
    }

    /* --via-decode is a request for the cached path. Refusing the contradiction
     * rather than resolving it quietly: a run that asked for both and got one
     * would be a measurement whose path is not what its command line says. */
    if (via_decode) {
        if (kv_flag_seen && path == MODEL_PATH_NOCACHE) {
            fprintf(stderr, "--via-decode assembles the matrix from the CACHED decode path "
                            "and contradicts --kv-cache off. Pass one or the other.\n");
            return 2;
        }
        path = MODEL_PATH_CACHE;
    }

    char *prompt_buf = NULL;
    size_t prompt_len;
    if (prompt_file) {
        prompt_buf = read_all(prompt_file, &prompt_len);
        if (!prompt_buf) { fprintf(stderr, "cannot read %s\n", prompt_file); return 1; }
        while (prompt_len > 0 && (prompt_buf[prompt_len - 1] == '\n' ||
                                  prompt_buf[prompt_len - 1] == '\r')) --prompt_len;
        prompt = prompt_buf;
    } else {
        prompt_len = strlen(prompt);
    }

    /* ---- load. Outside every timed bracket, by protocol section 2. ---- */
    tokenizer *tok = NULL;
    tok_status trc = tok_load(tokjson, &tok);
    if (trc != TOK_OK) { fprintf(stderr, "tokenizer: %s\n", tok_strerror(trc)); return 1; }

    model *m = NULL;
    model_status rc = model_load(weights, inventory, config, reading, &m);
    if (rc != MODEL_OK) { fprintf(stderr, "model: %s\n", model_strerror(rc)); return 1; }
    const model_config *c = model_config_of(m);

    if (tok_vocab_size(tok) != (size_t)c->vocab_size) {
        fprintf(stderr, "vocab_size disagreement: tokenizer %zu, config %d\n",
                tok_vocab_size(tok), c->vocab_size);
        return 1;
    }

    printf("model: %d layers, %d heads, n_embd %d, head_dim %d, n_ctx %d, vocab %d, "
           "eps %g, activation %s\n",
           c->n_layer, c->n_head, c->n_embd, c->head_dim, c->n_ctx, c->vocab_size,
           (double)c->layer_norm_epsilon, c->activation_function);
    printf("matmul: %s   attn.c_proj reading: %s   head: TIED to wte.weight (%s storage)\n",
           model_gemm_of(m)->name,
           model_cproj_reading_of(m) == MODEL_CPROJ_TRANSPOSED ? "transposed" : "as-stored",
           model_head_weight(m) == model_token_embedding(m) ? "same" : "DIFFERENT -- NOT TIED");
    printf("kv cache: %s%s\n",
           path == MODEL_PATH_CACHE ? "ON" : "off (the no-cache path)",
           via_decode ? "   logit matrix assembled from the cached decode path alone" : "");

    if (show_records) {
        for (size_t i = 0; i < model_tensor_record_count(m); ++i) {
            const model_tensor_record *r = model_tensor_record_at(m, i);
            printf("  %-28s %-10s inventory=%-26s transposed=%d  %s\n",
                   r->name, r->stored_shape, r->inventory_orientation,
                   r->transposed_at_load, r->reconciliation);
        }
    }

    /* ---- encode ---- */
    size_t cap = prompt_len + 1;
    int32_t *ids = (int32_t *)malloc(cap * sizeof(int32_t) + (size_t)generate * sizeof(int32_t));
    size_t n_ids = 0;
    trc = tok_encode(tok, (const unsigned char *)prompt, prompt_len, ids, cap, &n_ids);
    if (trc != TOK_OK) { fprintf(stderr, "encode: %s\n", tok_strerror(trc)); return 1; }
    if (truncate_to > 0 && (size_t)truncate_to < n_ids) n_ids = (size_t)truncate_to;
    printf("prompt: %zu tokens\n", n_ids);

    float *logits = (float *)malloc((size_t)c->vocab_size * n_ids * sizeof(float));
    if (!logits) { fprintf(stderr, "out of memory for logits\n"); return 1; }

    /* ---- the path, and the cache when it is selected. Both outside every
     *      bracket; this program times nothing. ---- */
    model_set_decode_path(m, path);
    const size_t want_ctx = n_ids + (size_t)generate;
    model_reserve(m, want_ctx);
    if (path == MODEL_PATH_CACHE) {
        rc = model_kv_reserve(m, want_ctx);
        if (rc != MODEL_OK) { fprintf(stderr, "kv cache: %s\n", model_strerror(rc)); return 1; }
    }
    if (show_footprint) {
        printf("kv cache footprint: %zu bytes allocated "
               "(= n_layer %d x 2 x capacity %d x n_embd %d x 4 B)\n",
               model_kv_footprint_bytes(m), c->n_layer, model_kv_capacity(m), c->n_embd);
        if (path != MODEL_PATH_CACHE)
            printf("                    the no-cache path is selected, so NO cache is "
                   "allocated and the resident footprint is exactly 0\n");
    }

    /* ---- prefill ---- */
    clock_t t0 = clock();
    if (via_decode) {
        /* The whole matrix from the CACHED PATH ALONE: prefill over the first
         * token, then one cached decode step per remaining token. Each step
         * writes the row for its own position, so the assembled matrix is
         * positions x vocab in the same layout prefill would have written. */
        rc = model_prefill(m, ids, 1, logits, (size_t)c->vocab_size);
        if (rc != MODEL_OK) { fprintf(stderr, "prefill(1): %s\n", model_strerror(rc)); return 1; }
        for (size_t t = 1; t < n_ids && rc == MODEL_OK; ++t) {
            rc = model_decode_step_cached(m, ids, t + 1,
                                          logits + t * (size_t)c->vocab_size,
                                          (size_t)c->vocab_size);
            if (rc != MODEL_OK)
                fprintf(stderr, "cached decode step at context %zu: %s\n",
                        t + 1, model_strerror(rc));
        }
        if (rc != MODEL_OK) return 1;
        printf("logit matrix assembled from %zu cached decode steps after a 1-token prefill "
               "(progress only, not a measurement: %.1f s wall)\n",
               n_ids - 1, (double)(clock() - t0) / CLOCKS_PER_SEC);
    } else {
        rc = model_prefill(m, ids, n_ids, logits, (size_t)c->vocab_size * n_ids);
        if (rc != MODEL_OK) { fprintf(stderr, "prefill: %s\n", model_strerror(rc)); return 1; }
        printf("prefill done (progress only, not a measurement: %.1f s wall)\n",
               (double)(clock() - t0) / CLOCKS_PER_SEC);
    }

    if (dump_path) {
        FILE *d = fopen(dump_path, "wb");
        if (!d) { fprintf(stderr, "cannot write %s\n", dump_path); return 1; }
        int32_t np = (int32_t)n_ids, nv = c->vocab_size;
        fwrite("TIE2LOGI", 1, 8, d);
        fwrite(&np, sizeof np, 1, d);
        fwrite(&nv, sizeof nv, 1, d);
        fwrite(logits, sizeof(float), (size_t)np * (size_t)nv, d);
        fclose(d);
        printf("logits written: %s (%d positions x %d)\n", dump_path, np, nv);
    }

    /* ---- greedy generation, one full forward pass per token ---- */
    int32_t next = model_argmax(logits + (n_ids - 1) * (size_t)c->vocab_size,
                                (size_t)c->vocab_size);
    for (int g = 0; g < generate; ++g) {
        ids[n_ids++] = next;
        if (g + 1 == generate) break;
        t0 = clock();
        rc = model_decode_step(m, ids, n_ids, logits, (size_t)c->vocab_size);
        if (rc != MODEL_OK) { fprintf(stderr, "decode: %s\n", model_strerror(rc)); return 1; }
        next = model_argmax(logits, (size_t)c->vocab_size);
        printf("  token %d/%d at context %zu (progress only: %.1f s wall)\n",
               g + 2, generate, n_ids, (double)(clock() - t0) / CLOCKS_PER_SEC);
        fflush(stdout);
    }

    /* ---- decode to text ---- */
    size_t out_cap = 16 * n_ids + 64, n_out = 0;
    unsigned char *text = (unsigned char *)malloc(out_cap);
    trc = tok_decode(tok, ids, n_ids, text, out_cap, &n_out);
    if (trc != TOK_OK) { fprintf(stderr, "decode text: %s\n", tok_strerror(trc)); return 1; }
    printf("ids:");
    for (size_t i = 0; i < n_ids; ++i) printf(" %d", ids[i]);
    printf("\ntext: ");
    fwrite(text, 1, n_out, stdout);
    printf("\n");

    free(text); free(logits); free(ids); free(prompt_buf);
    model_free(m);
    tok_free(tok);
    return 0;
}

#endif /* GPT2_MAIN */
