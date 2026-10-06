/* kv_cache.h -- the Stage 4 key/value cache.
 *
 * WHAT THIS IS FOR. Without a cache a decode step re-runs the whole forward
 * pass over the whole context, so its cost grows with context length; that is
 * the Stage 2 baseline and it is what this file exists to flatten. With the
 * cache, every position's keys and values are kept per layer, and a decode step
 * does one token's work through the weights and attends over what is already
 * stored.
 *
 * THE LAYOUT IS NOT A NEW ONE. Each layer's key buffer and value buffer are
 * [capacity, n_embd] float32, row-major, with POSITION as the row and the heads
 * laid out contiguously along the row -- so head h of position p is the 64-float
 * slice at offset h * head_dim within row p, addressed at stride n_embd. That is
 * exactly how src/model.c already lays out the fused QKV activation buffer and
 * exactly how its attention loop already addresses a head. The cached and the
 * recomputed paths therefore read the same element in the same order, which is
 * what makes bit-for-bit agreement between them a structural property rather
 * than a hope.
 *
 * FOOTPRINT = n_layer * 2 * capacity * n_embd * 4 bytes. At this model's shapes
 * that is 73,728 bytes per position, and 75,497,472 bytes at capacity n_ctx =
 * 1024. It is computed from the arguments, never from a compiled-in constant.
 *
 * NO ARITHMETIC LIVES HERE. This file allocates, stores, hands back row
 * pointers and counts bytes. It does not include the GEMM or the model, it
 * performs no floating-point operation on the values it carries, and it has no
 * opinion about attention. Every matmul in the engine stays behind gemm_impl.
 *
 * ALLOCATION IS OUTSIDE EVERY TIMED BRACKET, and the buffers are written once at
 * creation so that no first-touch page fault can land inside one.
 */
#ifndef TIE_KV_CACHE_H
#define TIE_KV_CACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KV_OK = 0,
    KV_ERR_ARG,        /* a null pointer, or a non-positive shape         */
    KV_ERR_CAPACITY,   /* the position or length is at or beyond capacity */
    KV_ERR_NOMEM
} kv_status;

typedef struct kv_cache kv_cache;

/* Byte footprint for a set of shapes, WITHOUT allocating. Returns 0 when any
 * argument is non-positive, so a caller cannot derive a size from nonsense. */
size_t kv_cache_footprint_for(int n_layer, int n_embd, int capacity);

/* Creates the cache: one allocation per buffer, zero-filled and touched here so
 * that a later store inside a timed bracket cannot take a first-touch fault. */
kv_status kv_cache_create(int n_layer, int n_embd, int capacity, kv_cache **out);

/* Safe on NULL, following the convention model_free already sets. */
void kv_cache_destroy(kv_cache *c);

int    kv_cache_n_layer(const kv_cache *c);
int    kv_cache_n_embd(const kv_cache *c);
int    kv_cache_capacity(const kv_cache *c);
size_t kv_cache_footprint_bytes(const kv_cache *c);

/* ---- the length counter, one per cache ---------------------------------
 * One counter for the whole cache rather than one per layer: a position is
 * either present in every layer or in none, and a per-layer counter would make
 * a half-written position representable. The counter is what the timing driver
 * restores between iterations, so that every timed decode step runs at the same
 * context. */
int       kv_cache_length(const kv_cache *c);
kv_status kv_cache_set_length(kv_cache *c, int length);

/* ---- writing ----------------------------------------------------------
 * kv_cache_write stores one position's keys and values for ONE layer at an
 * explicit row. Prefill needs it: it fills every position of a layer before
 * moving to the next layer, so the length counter cannot advance per store.
 *
 * kv_cache_append is that same store at the CURRENT length, which is what a
 * decode step does; kv_cache_advance then moves the counter on by one position
 * once every layer has stored.
 *
 * Both refuse at capacity with KV_ERR_CAPACITY and write NOTHING when they
 * refuse -- a partially appended position would be worse than a rejected one.
 * k and v are each n_embd contiguous floats, the layer's whole row. */
kv_status kv_cache_write(kv_cache *c, int layer, int pos,
                         const float *k, const float *v);
kv_status kv_cache_append(kv_cache *c, int layer, const float *k, const float *v);
kv_status kv_cache_advance(kv_cache *c);

/* ---- reading ----------------------------------------------------------
 * The base pointer of a layer's [capacity, n_embd] buffer, and the row for one
 * position. Row p of a layer begins at offset p * n_embd, which is the stride
 * attention passes to the matmul as its leading dimension. NULL on a bad
 * argument rather than an out-of-range pointer. */
const float *kv_cache_keys(const kv_cache *c, int layer);
const float *kv_cache_values(const kv_cache *c, int layer);
const float *kv_cache_key_row(const kv_cache *c, int layer, int pos);
const float *kv_cache_value_row(const kv_cache *c, int layer, int pos);

const char *kv_strerror(kv_status s);

#ifdef __cplusplus
}
#endif

#endif /* TIE_KV_CACHE_H */
