/* model.c -- GPT-2 forward pass, Stage 2 baseline. No KV cache, no blocking,
 * no SIMD, no threading, no CUDA. Every one of those belongs to a later stage.
 *
 * THREE DETERMINATIONS THIS FILE MAKES, EACH WITH ITS REASON.
 *
 * 1. THE CAUSAL MASK IS REGENERATED, NOT LOADED. The twelve h.*.attn.bias
 *    tensors are registered mask buffers -- Stage 1 verified every element is
 *    0 or 1 and the [1024, 1024] plane is exactly lower-triangular -- and they
 *    occupy 50,331,648 bytes, 9.2% of the data segment. This file does not read
 *    them and does not materialize a mask at all: causality is the loop bound
 *    j <= t in the score loop plus an explicit zero above the diagonal, which
 *    is two integer comparisons in place of 50 MB of resident memory and 50 MB
 *    of file reading. The check that this is equivalent is a test, not a
 *    comment: tests/test_model.c changes a later token and asserts no earlier
 *    logit moves.
 *
 * 2. ORIENTATION IS RECONCILED HERE, AT LOAD TIME, AND RECORDED PER TENSOR.
 *    The file stores every rectangular 2-D weight as [input, output] (Stage 1
 *    pinned each one by its own bias length). That is exactly the B layout
 *    gemm_f32 consumes, so the reconciliation for those tensors is "no
 *    transpose, and here is why" -- recorded in model_tensor_record, not left
 *    implicit. wte.weight is [vocab, n_embd] = [output, input] and is consumed
 *    through the transposed-B form rather than being copied into a transposed
 *    buffer, because tying the head means USING that storage. The twelve square
 *    h.*.attn.c_proj.weight tensors are unresolved by shape; the reading is the
 *    cproj_reading parameter, and when the transposed reading is selected the
 *    transpose happens once, here, at load.
 *
 * 3. THE HEAD IS TIED. No tensor is named lm_head and exactly one matrix
 *    carries the vocabulary size against the embedding width, so the head is
 *    wte.weight itself -- the same allocation, not a copy of it.
 *
 * STAGE 4 AMENDMENT, 2026-10-05 -- THE KV CACHE, BEHIND A RUNTIME SWITCH.
 *
 * Two paths now exist and both are kept. MODEL_PATH_NOCACHE is the default and
 * is the Stage 2 code above, unchanged: no cache is allocated, nothing is
 * stored, no branch runs inside an inner loop. MODEL_PATH_CACHE makes prefill
 * store every position's keys and values per layer, and gives a decode step a
 * path that processes ONE token and attends over the stored positions.
 *
 * WHY THE TWO PATHS AGREE BIT FOR BIT, structurally rather than by tolerance.
 * gemm_naive computes each output row from one single-float accumulator walking
 * k upward, and no row of any GEMM here depends on another row, so the rows a
 * cached step computes are the rows prefill computes for the same position.
 * Attention is the only place where that could break, and it does not:
 *   - the cache's layout IS the activation layout, so key row j of a layer
 *     holds the same n_embd floats, in the same order, that m->qkv row j holds
 *     at offset n_embd, and a head is the same 64-float slice at stride
 *     n_embd rather than at stride 3*n_embd;
 *   - the score row for the new position reduces over j = 0..c-1 in increasing
 *     j, which is what gemm_naive_bt does for row c-1 of the full square;
 *   - the softmax runs over exactly the c entries the no-cache path's row c-1
 *     runs over, with the same maximum subtracted in the same order;
 *   - the weighted sum reduces over the same j in the same increasing order.
 * The no-cache path's LAST row never has a masked tail -- at t = T-1 every
 * j <= t -- so for the position a decode step produces there is not even a
 * zeroed upper triangle to differ about.
 *
 * WHAT THE CACHED PATH DOES NOT DO. It introduces no arithmetic routine of its
 * own: every matmul still goes through m->gemm, the same two functions, with
 * M = 1. The attention it performs is one query row over c keys, so there is no
 * discarded upper triangle and its performed-FLOP count is 36,864*c per step
 * against the no-cache path's full-square 36,864*c^2 (PERSISTENT.md section 8,
 * W13). No loop is reordered, blocked or vectorised here; those are Stages 5
 * and 6 and taking any of them now would consume their measured gain.
 *
 * Every architecture value is read from models/gpt2/config.json. Every tensor's
 * shape, dtype, absolute file offset and byte length is read from
 * src/gpt2_tensor_inventory.json and checked against the weight file's own
 * header before any byte is read. Nothing here is compiled in except the tensor
 * NAMES, because a loader has to ask the file for something.
 */
#include "model.h"
#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ JSON ---
 * The smallest scanner that reads these two artifacts and refuses anything it
 * does not understand. Both files are machine-generated, flat, and free of
 * escapes; a general JSON parser would be more code with more to go wrong. Both
 * are validated by what is read out of them, not by a grammar. */

static const char *json_after_key(const char *s, const char *key)
{
    char pat[96];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(s, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    if (*p != ':') return NULL;
    ++p;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    return p;
}

static int json_number(const char *s, const char *key, double *out)
{
    const char *p = json_after_key(s, key);
    if (!p) return 0;
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p) return 0;
    *out = v;
    return 1;
}

static int json_int(const char *s, const char *key, int *out)
{
    double v;
    if (!json_number(s, key, &v)) return 0;
    *out = (int)v;
    return 1;
}

static int json_string(const char *s, const char *key, char *out, size_t cap)
{
    const char *p = json_after_key(s, key);
    if (!p || *p != '"') return 0;
    ++p;
    size_t i = 0;
    while (*p && *p != '"') {
        if (i + 1 >= cap) return 0;
        out[i++] = *p++;
    }
    if (*p != '"') return 0;
    out[i] = '\0';
    return 1;
}

/* Reads a whole text file into a NUL-terminated buffer; caller frees. */
static char *read_text_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);
    return buf;
}

/* ------------------------------------------------------------- inventory ---
 * One tensor per line in the committed inventory, each an object carrying
 * name, dtype, shape, begin, end, file_offset, nbytes and the orientation the
 * Stage 1 loader determined. Looked up by name; the line is the record. */

typedef struct {
    char     dtype[16];
    int      n_dims;
    long long dims[8];
    unsigned long long file_offset;
    unsigned long long nbytes;
    char     orientation[48];
} inv_entry;

static int inventory_find(const char *inv, const char *name, inv_entry *out)
{
    char pat[160];
    snprintf(pat, sizeof pat, "{\"name\": \"%s\",", name);
    const char *p = strstr(inv, pat);
    if (!p) return 0;
    const char *end = strchr(p, '\n');
    size_t len = end ? (size_t)(end - p) : strlen(p);
    char line[1024];
    if (len >= sizeof line) return 0;
    memcpy(line, p, len);
    line[len] = '\0';

    memset(out, 0, sizeof *out);
    if (!json_string(line, "dtype", out->dtype, sizeof out->dtype)) return 0;
    if (!json_string(line, "orientation", out->orientation, sizeof out->orientation)) return 0;

    double v;
    if (!json_number(line, "file_offset", &v)) return 0;
    out->file_offset = (unsigned long long)v;
    if (!json_number(line, "nbytes", &v)) return 0;
    out->nbytes = (unsigned long long)v;

    const char *sh = json_after_key(line, "shape");
    if (!sh || *sh != '[') return 0;
    ++sh;
    while (*sh && *sh != ']') {
        while (*sh == ' ' || *sh == ',') ++sh;
        if (*sh == ']') break;
        char *e = NULL;
        long long d = strtoll(sh, &e, 10);
        if (e == sh) return 0;
        if (out->n_dims >= 8) return 0;
        out->dims[out->n_dims++] = d;
        sh = e;
    }
    return 1;
}

/* ------------------------------------------------------------ the model ---- */

typedef struct {
    float *ln_1_w, *ln_1_b;
    float *c_attn_w, *c_attn_b;       /* [n_embd, 3*n_embd], [3*n_embd] */
    float *attn_proj_w, *attn_proj_b; /* [n_embd, n_embd],   [n_embd]   */
    float *ln_2_w, *ln_2_b;
    float *c_fc_w, *c_fc_b;           /* [n_embd, 4*n_embd], [4*n_embd] */
    float *mlp_proj_w, *mlp_proj_b;   /* [4*n_embd, n_embd], [n_embd]   */
} model_layer;

struct model {
    model_config        cfg;
    model_cproj_reading cproj_reading;
    model_decode_path   decode_path;   /* Stage 4; MODEL_PATH_NOCACHE by default */
    kv_cache           *kv;            /* allocated LAZILY by model_kv_reserve   */
    const gemm_impl    *gemm;

    float *wte;      /* [vocab_size, n_embd] -- token embedding AND the tied head */
    float *wpe;      /* [n_ctx, n_embd] -- learned absolute positions             */
    float *ln_f_w;
    float *ln_f_b;
    model_layer *layers;

    /* activation scratch, grown by model_reserve, never inside a timed bracket */
    size_t scratch_tokens;
    float *x;        /* [T, n_embd]      residual stream                */
    float *xb;       /* [T, n_embd]      normalized copy                */
    float *qkv;      /* [T, 3*n_embd]                                   */
    float *attn;     /* [T, n_embd]      concatenated head outputs      */
    float *scores;   /* [T, T]           one head's scores/probabilities */
    float *ff;       /* [T, 4*n_embd]                                   */

    model_tensor_record *records;
    size_t               n_records;

    int    collect_attn_stats;
    double attn_rowsum_min;
    double attn_rowsum_max;
    size_t attn_rows;
};

const char *model_strerror(model_status s)
{
    switch (s) {
        case MODEL_OK:            return "ok";
        case MODEL_ERR_OPEN:      return "an artifact could not be opened";
        case MODEL_ERR_CONFIG:    return "config.json is missing a required field";
        case MODEL_ERR_INVENTORY: return "the inventory disagrees with the weight file";
        case MODEL_ERR_WEIGHTS:   return "the weight file failed to open, parse or read";
        case MODEL_ERR_SHAPE:     return "a tensor's shape contradicts the config";
        case MODEL_ERR_DTYPE:     return "a tensor is not F32";
        case MODEL_ERR_RANGE:     return "a token id or position is out of range";
        case MODEL_ERR_CAPACITY:  return "a caller-owned buffer is too small";
        case MODEL_ERR_NOMEM:     return "allocation failed";
        case MODEL_ERR_ARG:       return "a null or unusable argument";
    }
    return "unknown";
}

/* ------------------------------------------------------------ loading ------ */

static void record_tensor(model *m, const char *name, const inv_entry *e,
                          int transposed, const char *reconciliation)
{
    if (m->n_records >= (size_t)(12 * 4 + 8)) return;
    model_tensor_record *r = &m->records[m->n_records++];
    snprintf(r->name, sizeof r->name, "%s", name);
    if (e->n_dims == 2)
        snprintf(r->stored_shape, sizeof r->stored_shape, "%lldx%lld",
                 (long long)e->dims[0], (long long)e->dims[1]);
    else
        snprintf(r->stored_shape, sizeof r->stored_shape, "%lld",
                 (long long)e->dims[0]);
    snprintf(r->inventory_orientation, sizeof r->inventory_orientation, "%s", e->orientation);
    r->transposed_at_load = transposed;
    snprintf(r->reconciliation, sizeof r->reconciliation, "%s", reconciliation);
}

/* Loads one tensor. The inventory entry and the weight file's own descriptor
 * must agree on dtype, shape and byte length before anything is read, and the
 * expected extents from the config must match both. */
static model_status load_tensor(model *m, const st_file *f, const char *inv,
                                const char *name, int nd, long long d0, long long d1,
                                float **out, inv_entry *entry_out)
{
    inv_entry e;
    if (!inventory_find(inv, name, &e)) return MODEL_ERR_INVENTORY;
    if (strcmp(e.dtype, "F32") != 0) return MODEL_ERR_DTYPE;
    if (e.n_dims != nd) return MODEL_ERR_SHAPE;
    if (e.dims[0] != d0) return MODEL_ERR_SHAPE;
    if (nd == 2 && e.dims[1] != d1) return MODEL_ERR_SHAPE;

    const st_tensor *t = NULL;
    if (st_find(f, name, &t) != ST_OK) return MODEL_ERR_INVENTORY;
    if (t->dtype != ST_DTYPE_F32) return MODEL_ERR_DTYPE;
    if (t->n_dims != nd) return MODEL_ERR_INVENTORY;
    for (int i = 0; i < nd; ++i)
        if (t->dims[i] != e.dims[i]) return MODEL_ERR_INVENTORY;
    if (t->nbytes != e.nbytes) return MODEL_ERR_INVENTORY;
    if (t->file_offset != e.file_offset) return MODEL_ERR_INVENTORY;

    size_t n_elem = (size_t)d0 * (size_t)(nd == 2 ? d1 : 1);
    if (t->nbytes != n_elem * sizeof(float)) return MODEL_ERR_SHAPE;

    float *buf = (float *)malloc(t->nbytes);
    if (!buf) return MODEL_ERR_NOMEM;
    if (st_tensor_read(f, t, buf, t->nbytes) != ST_OK) { free(buf); return MODEL_ERR_WEIGHTS; }

    *out = buf;
    if (entry_out) *entry_out = e;
    return MODEL_OK;
}

/* Transposes an n x n matrix in place. Used for exactly one case: the square
 * attention output projection under MODEL_CPROJ_TRANSPOSED. */
static void transpose_square(float *a, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            float t = a[(size_t)i * n + j];
            a[(size_t)i * n + j] = a[(size_t)j * n + i];
            a[(size_t)j * n + i] = t;
        }
}

model_status model_load(const char *weights_path,
                        const char *inventory_path,
                        const char *config_path,
                        model_cproj_reading cproj_reading,
                        model **out)
{
    if (!weights_path || !inventory_path || !config_path || !out) return MODEL_ERR_ARG;
    *out = NULL;

    char *cfg_text = read_text_file(config_path);
    if (!cfg_text) return MODEL_ERR_OPEN;
    char *inv_text = read_text_file(inventory_path);
    if (!inv_text) { free(cfg_text); return MODEL_ERR_OPEN; }

    model *m = (model *)calloc(1, sizeof *m);
    if (!m) { free(cfg_text); free(inv_text); return MODEL_ERR_NOMEM; }
    m->gemm = &gemm_impl_naive;
    m->cproj_reading = cproj_reading;

    model_status rc = MODEL_OK;
    st_file *f = NULL;

    /* --- config: every value from the artifact, none defaulted --- */
    double eps;
    if (!json_int(cfg_text, "n_layer",    &m->cfg.n_layer)    ||
        !json_int(cfg_text, "n_head",     &m->cfg.n_head)     ||
        !json_int(cfg_text, "n_embd",     &m->cfg.n_embd)     ||
        !json_int(cfg_text, "n_ctx",      &m->cfg.n_ctx)      ||
        !json_int(cfg_text, "vocab_size", &m->cfg.vocab_size) ||
        !json_number(cfg_text, "layer_norm_epsilon", &eps)    ||
        !json_string(cfg_text, "activation_function",
                     m->cfg.activation_function, sizeof m->cfg.activation_function)) {
        rc = MODEL_ERR_CONFIG;
        goto fail;
    }
    m->cfg.layer_norm_epsilon = (float)eps;
    if (m->cfg.n_head <= 0 || m->cfg.n_embd % m->cfg.n_head != 0) { rc = MODEL_ERR_CONFIG; goto fail; }
    m->cfg.head_dim = m->cfg.n_embd / m->cfg.n_head;
    /* The activation is taken from the config and is not substituted. A
     * checkpoint naming something this file does not implement fails the load
     * rather than silently running a different nonlinearity. */
    if (strcmp(m->cfg.activation_function, "gelu_new") != 0) { rc = MODEL_ERR_CONFIG; goto fail; }

    /* The inventory records the config values it was produced against. If they
     * disagree with the config being read now, the two artifacts describe
     * different models and nothing below is trustworthy. */
    {
        int inv_embd = 0, inv_layer = 0, inv_head = 0, inv_vocab = 0;
        const char *cv = strstr(inv_text, "\"config_values_used\"");
        if (!cv) { rc = MODEL_ERR_INVENTORY; goto fail; }
        if (!json_int(cv, "n_embd", &inv_embd) || !json_int(cv, "n_layer", &inv_layer) ||
            !json_int(cv, "n_head", &inv_head) || !json_int(cv, "vocab_size", &inv_vocab)) {
            rc = MODEL_ERR_INVENTORY; goto fail;
        }
        if (inv_embd != m->cfg.n_embd || inv_layer != m->cfg.n_layer ||
            inv_head != m->cfg.n_head || inv_vocab != m->cfg.vocab_size) {
            rc = MODEL_ERR_INVENTORY; goto fail;
        }
    }

    const int E  = m->cfg.n_embd;
    const int E3 = 3 * E;
    const int E4 = 4 * E;

    if (st_open(weights_path, &f) != ST_OK) { rc = MODEL_ERR_WEIGHTS; goto fail; }

    m->records = (model_tensor_record *)calloc(12 * 4 + 8, sizeof *m->records);
    m->layers  = (model_layer *)calloc((size_t)m->cfg.n_layer, sizeof *m->layers);
    if (!m->records || !m->layers) { rc = MODEL_ERR_NOMEM; goto fail; }

    inv_entry e;
    /* wte: [vocab_size, n_embd]. Both the token embedding and, tied, the head. */
    rc = load_tensor(m, f, inv_text, "wte.weight", 2, m->cfg.vocab_size, E, &m->wte, &e);
    if (rc != MODEL_OK) goto fail;
    record_tensor(m, "wte.weight", &e, 0,
                  "rows are tokens; consumed as-is for lookup and as B^T for the tied head");

    rc = load_tensor(m, f, inv_text, "wpe.weight", 2, m->cfg.n_ctx, E, &m->wpe, &e);
    if (rc != MODEL_OK) goto fail;
    record_tensor(m, "wpe.weight", &e, 0, "rows are positions; learned absolute, consumed as-is");

    rc = load_tensor(m, f, inv_text, "ln_f.weight", 1, E, 0, &m->ln_f_w, &e);
    if (rc != MODEL_OK) goto fail;
    rc = load_tensor(m, f, inv_text, "ln_f.bias",   1, E, 0, &m->ln_f_b, &e);
    if (rc != MODEL_OK) goto fail;

    for (int l = 0; l < m->cfg.n_layer; ++l) {
        model_layer *ly = &m->layers[l];
        char name[64];

#define LOAD(field, fmt, nd, d0, d1)                                            \
        do {                                                                     \
            snprintf(name, sizeof name, fmt, l);                                 \
            rc = load_tensor(m, f, inv_text, name, nd, d0, d1, &ly->field, &e);  \
            if (rc != MODEL_OK) goto fail;                                       \
        } while (0)

        LOAD(ln_1_w,      "h.%d.ln_1.weight",        1, E,  0);
        LOAD(ln_1_b,      "h.%d.ln_1.bias",          1, E,  0);
        LOAD(c_attn_w,    "h.%d.attn.c_attn.weight", 2, E,  E3);
        if (l == 0) record_tensor(m, "h.*.attn.c_attn.weight", &e, 0,
                                  "[input, output] as stored; gemm_f32 B layout, no transpose");
        LOAD(c_attn_b,    "h.%d.attn.c_attn.bias",   1, E3, 0);
        LOAD(attn_proj_w, "h.%d.attn.c_proj.weight", 2, E,  E);
        if (cproj_reading == MODEL_CPROJ_TRANSPOSED) transpose_square(ly->attn_proj_w, E);
        if (l == 0) record_tensor(m, "h.*.attn.c_proj.weight", &e,
                                  cproj_reading == MODEL_CPROJ_TRANSPOSED,
                                  cproj_reading == MODEL_CPROJ_TRANSPOSED
                                      ? "unresolved by shape; TRANSPOSED at load to [input, output]"
                                      : "unresolved by shape; read AS STORED, i.e. [input, output]");
        LOAD(attn_proj_b, "h.%d.attn.c_proj.bias",   1, E,  0);
        LOAD(ln_2_w,      "h.%d.ln_2.weight",        1, E,  0);
        LOAD(ln_2_b,      "h.%d.ln_2.bias",          1, E,  0);
        LOAD(c_fc_w,      "h.%d.mlp.c_fc.weight",    2, E,  E4);
        if (l == 0) record_tensor(m, "h.*.mlp.c_fc.weight", &e, 0,
                                  "[input, output] as stored; gemm_f32 B layout, no transpose");
        LOAD(c_fc_b,      "h.%d.mlp.c_fc.bias",      1, E4, 0);
        LOAD(mlp_proj_w,  "h.%d.mlp.c_proj.weight",  2, E4, E);
        if (l == 0) record_tensor(m, "h.*.mlp.c_proj.weight", &e, 0,
                                  "[input, output] as stored; gemm_f32 B layout, no transpose");
        LOAD(mlp_proj_b,  "h.%d.mlp.c_proj.bias",    1, E,  0);
#undef LOAD
    }

    st_close(f);
    free(cfg_text);
    free(inv_text);
    *out = m;
    return MODEL_OK;

fail:
    if (f) st_close(f);
    free(cfg_text);
    free(inv_text);
    model_free(m);
    return rc;
}

void model_free(model *m)
{
    if (!m) return;
    free(m->wte); free(m->wpe); free(m->ln_f_w); free(m->ln_f_b);
    if (m->layers) {
        for (int l = 0; l < m->cfg.n_layer; ++l) {
            model_layer *ly = &m->layers[l];
            free(ly->ln_1_w); free(ly->ln_1_b);
            free(ly->c_attn_w); free(ly->c_attn_b);
            free(ly->attn_proj_w); free(ly->attn_proj_b);
            free(ly->ln_2_w); free(ly->ln_2_b);
            free(ly->c_fc_w); free(ly->c_fc_b);
            free(ly->mlp_proj_w); free(ly->mlp_proj_b);
        }
        free(m->layers);
    }
    free(m->records);
    free(m->x); free(m->xb); free(m->qkv); free(m->attn); free(m->scores); free(m->ff);
    kv_cache_destroy(m->kv);
    free(m);
}

const model_config *model_config_of(const model *m) { return m ? &m->cfg : NULL; }
model_cproj_reading model_cproj_reading_of(const model *m) { return m->cproj_reading; }

/* ---- Stage 4: the decode-path switch and the cache's state -------------- */

void model_set_decode_path(model *m, model_decode_path p)
{
    if (m && (p == MODEL_PATH_NOCACHE || p == MODEL_PATH_CACHE)) m->decode_path = p;
}

model_decode_path model_decode_path_of(const model *m)
{
    return m ? m->decode_path : MODEL_PATH_NOCACHE;
}

/* LAZY, and outside every timed bracket. A model left on the no-cache path that
 * never calls this holds no cache at all, which is why the no-cache path's
 * resident cache footprint is exactly zero rather than merely unused. */
model_status model_kv_reserve(model *m, size_t capacity)
{
    if (!m) return MODEL_ERR_ARG;
    if (capacity == 0) return MODEL_ERR_ARG;
    if ((int)capacity > m->cfg.n_ctx) return MODEL_ERR_RANGE;
    if (m->kv && kv_cache_capacity(m->kv) >= (int)capacity) return MODEL_OK;

    kv_cache *fresh = NULL;
    kv_status ks = kv_cache_create(m->cfg.n_layer, m->cfg.n_embd, (int)capacity, &fresh);
    if (ks == KV_ERR_NOMEM) return MODEL_ERR_NOMEM;
    if (ks != KV_OK) return MODEL_ERR_ARG;
    kv_cache_destroy(m->kv);
    m->kv = fresh;
    return MODEL_OK;
}

size_t model_kv_footprint_bytes(const model *m) { return m ? kv_cache_footprint_bytes(m->kv) : 0u; }
int    model_kv_capacity(const model *m)        { return m ? kv_cache_capacity(m->kv) : 0; }
int    model_kv_length(const model *m)          { return m ? kv_cache_length(m->kv) : 0; }
const kv_cache *model_kv_cache_of(const model *m){ return m ? m->kv : NULL; }

model_status model_kv_set_length(model *m, int length)
{
    if (!m || !m->kv) return MODEL_ERR_ARG;
    kv_status ks = kv_cache_set_length(m->kv, length);
    if (ks == KV_ERR_CAPACITY) return MODEL_ERR_CAPACITY;
    return ks == KV_OK ? MODEL_OK : MODEL_ERR_ARG;
}
void model_set_gemm(model *m, const gemm_impl *impl) { if (m && impl) m->gemm = impl; }
const gemm_impl *model_gemm_of(const model *m) { return m ? m->gemm : NULL; }
size_t model_tensor_record_count(const model *m) { return m ? m->n_records : 0; }
const model_tensor_record *model_tensor_record_at(const model *m, size_t i)
{
    return (m && i < m->n_records) ? &m->records[i] : NULL;
}
const float *model_token_embedding(const model *m) { return m ? m->wte : NULL; }
/* The tie, as storage rather than as a claim: the head IS the token embedding. */
const float *model_head_weight(const model *m) { return m ? m->wte : NULL; }

void model_collect_attn_stats(model *m, int enable)
{
    if (!m) return;
    m->collect_attn_stats = enable;
    m->attn_rowsum_min = 0.0;
    m->attn_rowsum_max = 0.0;
    m->attn_rows = 0;
}

void model_attn_rowsum_range(const model *m, double *lo, double *hi, size_t *rows)
{
    if (lo)   *lo   = m->attn_rowsum_min;
    if (hi)   *hi   = m->attn_rowsum_max;
    if (rows) *rows = m->attn_rows;
}

/* ------------------------------------------------------------- scratch ----- */

model_status model_reserve(model *m, size_t T)
{
    if (!m) return MODEL_ERR_ARG;
    if (T == 0) T = 1;                       /* realloc(p, 0) is not a growth request */
    if (T <= m->scratch_tokens) return MODEL_OK;
    const size_t E = (size_t)m->cfg.n_embd;

    float *x      = (float *)realloc(m->x,      T * E * sizeof(float));
    float *xb     = (float *)realloc(m->xb,     T * E * sizeof(float));
    float *qkv    = (float *)realloc(m->qkv,    T * 3 * E * sizeof(float));
    float *attn   = (float *)realloc(m->attn,   T * E * sizeof(float));
    float *scores = (float *)realloc(m->scores, T * T * sizeof(float));
    float *ff     = (float *)realloc(m->ff,     T * 4 * E * sizeof(float));
    if (x)      m->x = x;
    if (xb)     m->xb = xb;
    if (qkv)    m->qkv = qkv;
    if (attn)   m->attn = attn;
    if (scores) m->scores = scores;
    if (ff)     m->ff = ff;
    if (!x || !xb || !qkv || !attn || !scores || !ff) return MODEL_ERR_NOMEM;

    m->scratch_tokens = T;
    return MODEL_OK;
}

/* ------------------------------------------------------- elementwise ------- */

void model_layernorm_normalize(const float *x, int n, float eps, float *out)
{
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= (double)n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) { double d = x[i] - mean; var += d * d; }
    var /= (double)n;
    float inv = (float)(1.0 / sqrt(var + (double)eps));
    for (int i = 0; i < n; ++i) out[i] = (float)((x[i] - mean) * inv);
}

static void layernorm(const float *x, const float *g, const float *b,
                      int n, float eps, float *out)
{
    model_layernorm_normalize(x, n, eps, out);
    for (int i = 0; i < n; ++i) out[i] = out[i] * g[i] + b[i];
}

/* gelu_new: the tanh approximation the config names. Not the erf form; the two
 * differ by roughly 1e-3 in the tails, which is larger than the divergence this
 * stage measures against the reference. */
static void gelu_new_inplace(float *v, size_t n)
{
    const float k = 0.7978845608028654f;   /* sqrt(2/pi) */
    for (size_t i = 0; i < n; ++i) {
        float x = v[i];
        float inner = k * (x + 0.044715f * x * x * x);
        v[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/* Numerically stable softmax over v[0..n), maximum subtracted. Returns the sum
 * of the unnormalized exponentials' normalized result, i.e. 1 up to rounding --
 * returned so the attention row-sum check measures the engine's own arithmetic
 * rather than recomputing it. */
static double softmax_inplace(float *v, int n)
{
    float mx = v[0];
    for (int i = 1; i < n; ++i) if (v[i] > mx) mx = v[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { float e = expf(v[i] - mx); v[i] = e; sum += e; }
    float inv = (float)(1.0 / sum);
    double check = 0.0;
    for (int i = 0; i < n; ++i) { v[i] *= inv; check += v[i]; }
    return check;
}

/* ------------------------------------------------------------- forward ----- */

/* One pass over T tokens through the twelve blocks, leaving the final
 * normalized activations in m->xb. The head is NOT applied here: prefill
 * applies it at every position and a decode step applies it once, and that is
 * the only difference between them. */
static model_status forward_blocks(model *m, const int32_t *ids, size_t T)
{
    const model_config *c = &m->cfg;
    const int E = c->n_embd, H = c->n_head, D = c->head_dim;
    const int E3 = 3 * E, E4 = 4 * E;
    const float scale = (float)(1.0 / sqrt((double)D));

    /* The switch, read ONCE per call into a local, outside every loop. In
     * no-cache mode `use_cache` is 0 and the single test below is the whole
     * cost of the switch in this function: twelve tests per call, none of them
     * inside an inner loop and none of them per token or per head. */
    const int use_cache = (m->decode_path == MODEL_PATH_CACHE);

    if (T == 0) return MODEL_ERR_ARG;
    if ((int)T > c->n_ctx) return MODEL_ERR_RANGE;
    for (size_t t = 0; t < T; ++t)
        if (ids[t] < 0 || ids[t] >= c->vocab_size) return MODEL_ERR_RANGE;
    /* Refused before any arithmetic, not halfway through the layers. */
    if (use_cache) {
        if (!m->kv) return MODEL_ERR_ARG;
        if ((int)T > kv_cache_capacity(m->kv)) return MODEL_ERR_CAPACITY;
    }

    /* embeddings: token + learned absolute position */
    for (size_t t = 0; t < T; ++t) {
        const float *we = m->wte + (size_t)ids[t] * E;
        const float *pe = m->wpe + t * E;
        float *dst = m->x + t * E;
        for (int i = 0; i < E; ++i) dst[i] = we[i] + pe[i];
    }

    for (int l = 0; l < c->n_layer; ++l) {
        model_layer *ly = &m->layers[l];

        /* pre-layernorm, then fused QKV projection */
        for (size_t t = 0; t < T; ++t)
            layernorm(m->x + t * E, ly->ln_1_w, ly->ln_1_b, E, c->layer_norm_epsilon,
                      m->xb + t * E);

        m->gemm->mul((int)T, E3, E, m->xb, E, ly->c_attn_w, E3, m->qkv, E3);
        for (size_t t = 0; t < T; ++t)
            for (int j = 0; j < E3; ++j) m->qkv[t * E3 + j] += ly->c_attn_b[j];

        /* STAGE 4, CACHE MODE ONLY. This layer's keys and values for every
         * position, stored AFTER the bias is added -- the cache holds what
         * attention reads, not an intermediate. Stores only: no arithmetic is
         * performed here and the logits are unaffected. The source rows are the
         * K and V thirds of the fused QKV activation, and the destination row is
         * the same n_embd floats in the same order, which is what makes a
         * cached read and a recomputed read the same element. */
        if (use_cache) {
            for (size_t t = 0; t < T; ++t)
                if (kv_cache_write(m->kv, l, (int)t,
                                   m->qkv + t * E3 + (size_t)E,
                                   m->qkv + t * E3 + 2 * (size_t)E) != KV_OK)
                    return MODEL_ERR_CAPACITY;
        }

        for (int h = 0; h < H; ++h) {
            const float *q = m->qkv + (size_t)h * D;
            const float *k = m->qkv + (size_t)E + (size_t)h * D;
            const float *v = m->qkv + (size_t)2 * E + (size_t)h * D;

            /* scores = Q K^T, scaled, causally masked, softmaxed per row. The
             * mask is the loop bound plus an explicit zero above the diagonal:
             * no mask tensor is read and none is materialized. */
            m->gemm->mul_bt((int)T, (int)T, D, q, E3, k, E3, m->scores, (int)T);
            for (size_t t = 0; t < T; ++t) {
                float *row = m->scores + t * T;
                for (size_t j = 0; j <= t; ++j) row[j] *= scale;
                double s = softmax_inplace(row, (int)t + 1);
                for (size_t j = t + 1; j < T; ++j) row[j] = 0.0f;
                if (m->collect_attn_stats) {
                    if (m->attn_rows == 0) { m->attn_rowsum_min = m->attn_rowsum_max = s; }
                    else {
                        if (s < m->attn_rowsum_min) m->attn_rowsum_min = s;
                        if (s > m->attn_rowsum_max) m->attn_rowsum_max = s;
                    }
                    ++m->attn_rows;
                }
            }

            /* context = P V, written straight into this head's slice */
            m->gemm->mul((int)T, D, (int)T, m->scores, (int)T, v, E3,
                         m->attn + (size_t)h * D, E);
        }

        m->gemm->mul((int)T, E, E, m->attn, E, ly->attn_proj_w, E, m->xb, E);
        for (size_t t = 0; t < T; ++t)
            for (int i = 0; i < E; ++i)
                m->x[t * E + i] += m->xb[t * E + i] + ly->attn_proj_b[i];

        /* feed-forward */
        for (size_t t = 0; t < T; ++t)
            layernorm(m->x + t * E, ly->ln_2_w, ly->ln_2_b, E, c->layer_norm_epsilon,
                      m->xb + t * E);

        m->gemm->mul((int)T, E4, E, m->xb, E, ly->c_fc_w, E4, m->ff, E4);
        for (size_t t = 0; t < T; ++t)
            for (int j = 0; j < E4; ++j) m->ff[t * E4 + j] += ly->c_fc_b[j];
        gelu_new_inplace(m->ff, T * (size_t)E4);

        m->gemm->mul((int)T, E, E4, m->ff, E4, ly->mlp_proj_w, E, m->xb, E);
        for (size_t t = 0; t < T; ++t)
            for (int i = 0; i < E; ++i)
                m->x[t * E + i] += m->xb[t * E + i] + ly->mlp_proj_b[i];
    }

    for (size_t t = 0; t < T; ++t)
        layernorm(m->x + t * E, m->ln_f_w, m->ln_f_b, E, c->layer_norm_epsilon,
                  m->xb + t * E);

    /* The cache now holds positions 0..T-1 for every layer, so the length is T
     * and a cached decode step can continue from it. */
    if (use_cache) kv_cache_set_length(m->kv, (int)T);
    return MODEL_OK;
}

model_status model_prefill(model *m, const int32_t *ids, size_t T,
                           float *logits, size_t cap)
{
    if (!m || !ids || !logits) return MODEL_ERR_ARG;
    if (cap < T * (size_t)m->cfg.vocab_size) return MODEL_ERR_CAPACITY;
    model_status rc = model_reserve(m, T);
    if (rc != MODEL_OK) return rc;
    rc = forward_blocks(m, ids, T);
    if (rc != MODEL_OK) return rc;

    /* The tied head, at EVERY position. wte is [vocab, n_embd] -- output by
     * input -- so it enters as the transposed operand rather than as a
     * transposed copy. */
    m->gemm->mul_bt((int)T, m->cfg.vocab_size, m->cfg.n_embd,
                    m->xb, m->cfg.n_embd, m->wte, m->cfg.n_embd,
                    logits, m->cfg.vocab_size);
    return MODEL_OK;
}

/* ONE CACHED DECODE STEP. Positions 0..T-2 are already in the cache; this runs
 * token T-1 alone through the twelve blocks, appends its keys and values, and
 * attends over all T positions.
 *
 * SCRATCH: row 0 of the activation buffers model_reserve already grew, so the
 * step allocates nothing. The score row needs T floats and m->scores holds
 * scratch_tokens^2, which covers it for every T the caller has reserved.
 *
 * THE REDUCTION ORDER IS PREFILL'S, ELEMENT FOR ELEMENT. See the file header
 * for why that makes the agreement structural. */
static model_status forward_one_cached(model *m, int32_t id, size_t T)
{
    const model_config *c = &m->cfg;
    const int E = c->n_embd, H = c->n_head, D = c->head_dim;
    const int E3 = 3 * E, E4 = 4 * E;
    const float scale = (float)(1.0 / sqrt((double)D));
    const size_t pos = T - 1;

    if (id < 0 || id >= c->vocab_size) return MODEL_ERR_RANGE;

    /* embedding of the one new token, at its own position */
    {
        const float *we = m->wte + (size_t)id * E;
        const float *pe = m->wpe + pos * E;
        for (int i = 0; i < E; ++i) m->x[i] = we[i] + pe[i];
    }

    for (int l = 0; l < c->n_layer; ++l) {
        model_layer *ly = &m->layers[l];

        layernorm(m->x, ly->ln_1_w, ly->ln_1_b, E, c->layer_norm_epsilon, m->xb);

        m->gemm->mul(1, E3, E, m->xb, E, ly->c_attn_w, E3, m->qkv, E3);
        for (int j = 0; j < E3; ++j) m->qkv[j] += ly->c_attn_b[j];

        /* Appended BEFORE attention, so the new position is among the keys this
         * step attends over -- and appended at row T-1, which is where the score
         * row's last reduction step reads it. */
        if (kv_cache_write(m->kv, l, (int)pos,
                           m->qkv + (size_t)E, m->qkv + 2 * (size_t)E) != KV_OK)
            return MODEL_ERR_CAPACITY;

        for (int h = 0; h < H; ++h) {
            const float *q  = m->qkv + (size_t)h * D;
            const float *kc = kv_cache_keys(m->kv, l)   + (size_t)h * D;
            const float *vc = kv_cache_values(m->kv, l) + (size_t)h * D;

            /* One query row against T cached key rows. The cache's row stride is
             * n_embd where the activation buffer's was 3*n_embd; the 64-float
             * head slice and the reduction over D are otherwise identical. No
             * mask and no upper triangle: every one of the T positions is at or
             * before this one, which is what causality means for the last row. */
            m->gemm->mul_bt(1, (int)T, D, q, E3, kc, E, m->scores, (int)T);
            for (size_t j = 0; j < T; ++j) m->scores[j] *= scale;
            double s = softmax_inplace(m->scores, (int)T);
            if (m->collect_attn_stats) {
                if (m->attn_rows == 0) { m->attn_rowsum_min = m->attn_rowsum_max = s; }
                else {
                    if (s < m->attn_rowsum_min) m->attn_rowsum_min = s;
                    if (s > m->attn_rowsum_max) m->attn_rowsum_max = s;
                }
                ++m->attn_rows;
            }

            m->gemm->mul(1, D, (int)T, m->scores, (int)T, vc, E,
                         m->attn + (size_t)h * D, E);
        }

        m->gemm->mul(1, E, E, m->attn, E, ly->attn_proj_w, E, m->xb, E);
        for (int i = 0; i < E; ++i) m->x[i] += m->xb[i] + ly->attn_proj_b[i];

        layernorm(m->x, ly->ln_2_w, ly->ln_2_b, E, c->layer_norm_epsilon, m->xb);

        m->gemm->mul(1, E4, E, m->xb, E, ly->c_fc_w, E4, m->ff, E4);
        for (int j = 0; j < E4; ++j) m->ff[j] += ly->c_fc_b[j];
        gelu_new_inplace(m->ff, (size_t)E4);

        m->gemm->mul(1, E, E4, m->ff, E4, ly->mlp_proj_w, E, m->xb, E);
        for (int i = 0; i < E; ++i) m->x[i] += m->xb[i] + ly->mlp_proj_b[i];
    }

    layernorm(m->x, m->ln_f_w, m->ln_f_b, E, c->layer_norm_epsilon, m->xb);
    kv_cache_set_length(m->kv, (int)T);
    return MODEL_OK;
}

model_status model_decode_step_cached(model *m, const int32_t *ids, size_t T,
                                      float *logits, size_t cap)
{
    if (!m || !ids || !logits) return MODEL_ERR_ARG;
    if (cap < (size_t)m->cfg.vocab_size) return MODEL_ERR_CAPACITY;
    if (T == 0) return MODEL_ERR_ARG;
    if ((int)T > m->cfg.n_ctx) return MODEL_ERR_RANGE;

    /* Every refusal below happens BEFORE any arithmetic. A step run against a
     * cache holding the wrong number of positions would return a plausible
     * wrong answer, which is worse than an error code. */
    if (!m->kv) return MODEL_ERR_ARG;
    if ((int)T > kv_cache_capacity(m->kv)) return MODEL_ERR_CAPACITY;
    if (kv_cache_length(m->kv) != (int)(T - 1)) return MODEL_ERR_ARG;

    model_status rc = model_reserve(m, T);
    if (rc != MODEL_OK) return rc;
    rc = forward_one_cached(m, ids[T - 1], T);
    if (rc != MODEL_OK) return rc;

    /* The head once, at this one position -- the same call prefill makes for its
     * last row and the same call the no-cache decode step makes. */
    m->gemm->mul_bt(1, m->cfg.vocab_size, m->cfg.n_embd,
                    m->xb, m->cfg.n_embd, m->wte, m->cfg.n_embd,
                    logits, m->cfg.vocab_size);
    return MODEL_OK;
}

model_status model_decode_step(model *m, const int32_t *ids, size_t T,
                               float *logits, size_t cap)
{
    if (!m || !ids || !logits) return MODEL_ERR_ARG;
    /* The switch, read once per call. Under MODEL_PATH_NOCACHE everything below
     * is the Stage 2 path unchanged. */
    if (m->decode_path == MODEL_PATH_CACHE)
        return model_decode_step_cached(m, ids, T, logits, cap);
    if (cap < (size_t)m->cfg.vocab_size) return MODEL_ERR_CAPACITY;
    model_status rc = model_reserve(m, T);
    if (rc != MODEL_OK) return rc;
    rc = forward_blocks(m, ids, T);
    if (rc != MODEL_OK) return rc;

    /* The head once, at the last position only. */
    m->gemm->mul_bt(1, m->cfg.vocab_size, m->cfg.n_embd,
                    m->xb + (T - 1) * (size_t)m->cfg.n_embd, m->cfg.n_embd,
                    m->wte, m->cfg.n_embd,
                    logits, m->cfg.vocab_size);
    return MODEL_OK;
}

int32_t model_argmax(const float *v, size_t n)
{
    int32_t best = 0;
    for (size_t i = 1; i < n; ++i) if (v[i] > v[best]) best = (int32_t)i;
    return best;
}
