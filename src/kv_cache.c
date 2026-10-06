/* kv_cache.c -- the Stage 4 key/value cache. Storage only; see kv_cache.h for
 * the layout and for why it is the activation layout rather than a new one.
 *
 * TWO ALLOCATIONS PER LAYER, one for keys and one for values, each
 * [capacity, n_embd] float32. Not one fused buffer: attention reads the key
 * rows and the value rows through two separate matmul calls with their own
 * leading dimensions, and interleaving them would put a stride between
 * consecutive key rows that the existing attention loop does not use.
 *
 * EVERY BYTE IS WRITTEN AT CREATION. malloc plus memset rather than calloc: a
 * calloc of this size can be served by lazily-zeroed pages, and then the first
 * store to each page takes a fault -- inside a timed bracket, if the first
 * decode step after allocation is the one being timed. memset touches every
 * page here, outside every bracket, which is the behaviour this file promises.
 */
#include "kv_cache.h"

#include <stdlib.h>
#include <string.h>

struct kv_cache {
    int     n_layer;
    int     n_embd;
    int     capacity;
    int     length;
    float **keys;      /* n_layer buffers, each [capacity, n_embd] */
    float **values;
    size_t  bytes;     /* the total footprint, computed once at creation */
};

const char *kv_strerror(kv_status s)
{
    switch (s) {
        case KV_OK:           return "ok";
        case KV_ERR_ARG:      return "a null pointer or a non-positive shape";
        case KV_ERR_CAPACITY: return "the position or length is at or beyond capacity";
        case KV_ERR_NOMEM:    return "allocation failed";
    }
    return "unknown";
}

size_t kv_cache_footprint_for(int n_layer, int n_embd, int capacity)
{
    if (n_layer <= 0 || n_embd <= 0 || capacity <= 0) return 0;
    return (size_t)n_layer * 2u * (size_t)capacity * (size_t)n_embd * sizeof(float);
}

kv_status kv_cache_create(int n_layer, int n_embd, int capacity, kv_cache **out)
{
    if (!out) return KV_ERR_ARG;
    *out = NULL;
    if (n_layer <= 0 || n_embd <= 0 || capacity <= 0) return KV_ERR_ARG;

    kv_cache *c = (kv_cache *)calloc(1, sizeof *c);
    if (!c) return KV_ERR_NOMEM;
    c->n_layer  = n_layer;
    c->n_embd   = n_embd;
    c->capacity = capacity;
    c->length   = 0;
    c->bytes    = kv_cache_footprint_for(n_layer, n_embd, capacity);

    c->keys   = (float **)calloc((size_t)n_layer, sizeof *c->keys);
    c->values = (float **)calloc((size_t)n_layer, sizeof *c->values);
    if (!c->keys || !c->values) { kv_cache_destroy(c); return KV_ERR_NOMEM; }

    const size_t per_layer = (size_t)capacity * (size_t)n_embd * sizeof(float);
    for (int l = 0; l < n_layer; ++l) {
        c->keys[l]   = (float *)malloc(per_layer);
        c->values[l] = (float *)malloc(per_layer);
        if (!c->keys[l] || !c->values[l]) { kv_cache_destroy(c); return KV_ERR_NOMEM; }
        /* Written here, outside every timed bracket, so no first-touch page
         * fault can land inside one. */
        memset(c->keys[l],   0, per_layer);
        memset(c->values[l], 0, per_layer);
    }

    *out = c;
    return KV_OK;
}

void kv_cache_destroy(kv_cache *c)
{
    if (!c) return;
    if (c->keys) {
        for (int l = 0; l < c->n_layer; ++l) free(c->keys[l]);
        free(c->keys);
    }
    if (c->values) {
        for (int l = 0; l < c->n_layer; ++l) free(c->values[l]);
        free(c->values);
    }
    free(c);
}

int    kv_cache_n_layer(const kv_cache *c)        { return c ? c->n_layer : 0; }
int    kv_cache_n_embd(const kv_cache *c)         { return c ? c->n_embd : 0; }
int    kv_cache_capacity(const kv_cache *c)       { return c ? c->capacity : 0; }
size_t kv_cache_footprint_bytes(const kv_cache *c){ return c ? c->bytes : 0u; }
int    kv_cache_length(const kv_cache *c)         { return c ? c->length : 0; }

kv_status kv_cache_set_length(kv_cache *c, int length)
{
    if (!c || length < 0) return KV_ERR_ARG;
    if (length > c->capacity) return KV_ERR_CAPACITY;
    c->length = length;
    return KV_OK;
}

kv_status kv_cache_write(kv_cache *c, int layer, int pos,
                         const float *k, const float *v)
{
    if (!c || !k || !v) return KV_ERR_ARG;
    if (layer < 0 || layer >= c->n_layer) return KV_ERR_ARG;
    if (pos < 0) return KV_ERR_ARG;
    /* Checked BEFORE either store, so a refusal leaves every byte unchanged. */
    if (pos >= c->capacity) return KV_ERR_CAPACITY;

    const size_t off = (size_t)pos * (size_t)c->n_embd;
    const size_t n   = (size_t)c->n_embd * sizeof(float);
    memcpy(c->keys[layer]   + off, k, n);
    memcpy(c->values[layer] + off, v, n);
    return KV_OK;
}

kv_status kv_cache_append(kv_cache *c, int layer, const float *k, const float *v)
{
    if (!c) return KV_ERR_ARG;
    return kv_cache_write(c, layer, c->length, k, v);
}

kv_status kv_cache_advance(kv_cache *c)
{
    if (!c) return KV_ERR_ARG;
    if (c->length >= c->capacity) return KV_ERR_CAPACITY;
    ++c->length;
    return KV_OK;
}

const float *kv_cache_keys(const kv_cache *c, int layer)
{
    if (!c || layer < 0 || layer >= c->n_layer) return NULL;
    return c->keys[layer];
}

const float *kv_cache_values(const kv_cache *c, int layer)
{
    if (!c || layer < 0 || layer >= c->n_layer) return NULL;
    return c->values[layer];
}

const float *kv_cache_key_row(const kv_cache *c, int layer, int pos)
{
    const float *base = kv_cache_keys(c, layer);
    if (!base || pos < 0 || pos >= c->capacity) return NULL;
    return base + (size_t)pos * (size_t)c->n_embd;
}

const float *kv_cache_value_row(const kv_cache *c, int layer, int pos)
{
    const float *base = kv_cache_values(c, layer);
    if (!base || pos < 0 || pos >= c->capacity) return NULL;
    return base + (size_t)pos * (size_t)c->n_embd;
}
