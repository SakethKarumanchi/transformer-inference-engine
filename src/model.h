/* model.h -- GPT-2 forward pass in C, Stage 2 baseline.
 *
 * WHAT THIS INTERFACE PROMISES, AND WHAT IT DELIBERATELY DOES NOT.
 *
 * It promises a forward pass built entirely from artifacts: every architecture
 * value comes from models/gpt2/config.json, every tensor's shape and byte range
 * from src/gpt2_tensor_inventory.json cross-checked against the weight file's
 * own header. No architecture constant is compiled in. The one thing this file
 * names is the set of tensor NAMES it looks for, because a loader must ask the
 * file for something.
 *
 * It does NOT promise a KV cache. There is none, by design: a decode step
 * re-runs the whole forward pass over the whole context. Stage 4 adds the
 * cache, and the point of this stage is to be the thing Stage 4 is measured
 * against.
 *
 * BUFFERS ARE CALLER-OWNED, matching the Stage 1 convention: the caller
 * allocates the logits buffer and passes its capacity, and the model never
 * hands out a pointer to storage it may later reallocate. Scratch space for the
 * activations is internal and is grown by model_reserve.
 *
 * ORIENTATION IS RECONCILED AT LOAD TIME, IN model.c, EXPLICITLY AND PER
 * TENSOR -- never as transposes scattered through the arithmetic. The weight
 * file stores every rectangular 2-D weight as [input, output]; the forward pass
 * consumes that layout directly through gemm_f32. The twelve square
 * h.*.attn.c_proj.weight tensors are UNRESOLVED BY SHAPE (Stage 1), so the
 * reading is a load-time parameter here and was selected by measurement, not by
 * assumption -- see model_cproj_reading.
 */
#ifndef TIE_MODEL_H
#define TIE_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "gemm/gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MODEL_OK = 0,
    MODEL_ERR_OPEN,           /* a required artifact could not be opened        */
    MODEL_ERR_CONFIG,         /* config.json is missing a field this needs      */
    MODEL_ERR_INVENTORY,      /* the inventory is missing a tensor, or disagrees
                                 with the weight file's own header              */
    MODEL_ERR_WEIGHTS,        /* the weight file failed to open, parse or read  */
    MODEL_ERR_SHAPE,          /* a tensor's shape contradicts the config        */
    MODEL_ERR_DTYPE,          /* a tensor is not F32                            */
    MODEL_ERR_RANGE,          /* token id or position outside the model's range */
    MODEL_ERR_CAPACITY,       /* a caller-owned buffer is too small             */
    MODEL_ERR_NOMEM,
    MODEL_ERR_ARG
} model_status;

/* The reading of the twelve square h.*.attn.c_proj.weight [768, 768] tensors.
 * Shape cannot settle it: both extents are n_embd and the sibling bias [768]
 * matches either axis, so it discriminates nothing. Both readings are therefore
 * runnable, and Stage 2 selected one by comparing each against the reference
 * oracle and against text coherence. The evidence is recorded in the
 * MEASUREMENTS.md entry; this enum is what made the comparison possible. */
typedef enum {
    MODEL_CPROJ_AS_STORED  = 0,  /* [input, output], like every rectangular weight */
    MODEL_CPROJ_TRANSPOSED = 1   /* [output, input], the transposed reading        */
} model_cproj_reading;

/* Every value here is read from config.json at load time. Nothing is defaulted
 * and nothing is compiled in; a missing field fails the load. */
typedef struct {
    int    n_layer;
    int    n_head;
    int    n_embd;
    int    n_ctx;
    int    vocab_size;
    int    head_dim;                    /* n_embd / n_head, derived and checked */
    float  layer_norm_epsilon;
    char   activation_function[32];     /* "gelu_new" in this checkpoint        */
} model_config;

typedef struct model model;

/* Loads config, inventory and weights. The inventory is CONSUMED, not
 * re-derived: every tensor this model uses is looked up there and its name,
 * dtype, shape, absolute file offset and byte length are checked against what
 * the weight file's own header declares before a single byte is read.
 *
 * The twelve causal-mask buffers h.*.attn.bias are deliberately NOT loaded --
 * see model.c for the determination and its reason. */
model_status model_load(const char *weights_path,
                        const char *inventory_path,
                        const char *config_path,
                        model_cproj_reading cproj_reading,
                        model **out);
void model_free(model *m);

const model_config *model_config_of(const model *m);
model_cproj_reading model_cproj_reading_of(const model *m);

/* ---- the orientation reconciliation, recorded per tensor ---------------
 * One record per 2-D weight the loader read, filled in at load time: the shape
 * as the file stores it, the orientation the committed inventory claims, and
 * what the loader did about it. This is the record the MEASUREMENTS.md entry is
 * written from, and it exists so the reconciliation is inspectable rather than
 * asserted. */
typedef struct {
    char name[64];
    char stored_shape[32];            /* "768x2304", as the file declares it   */
    char inventory_orientation[48];   /* the inventory's own orientation field */
    int  transposed_at_load;          /* 1 when the loader transposed the bytes */
    char reconciliation[96];          /* what the forward pass therefore sees   */
} model_tensor_record;

size_t                     model_tensor_record_count(const model *m);
const model_tensor_record *model_tensor_record_at(const model *m, size_t i);

/* The matmul implementation in use. Stages 5 and 6 substitute here; the forward
 * pass itself does not change. Defaults to gemm_impl_naive. */
void              model_set_gemm(model *m, const gemm_impl *impl);
const gemm_impl  *model_gemm_of(const model *m);

/* Grows the internal activation scratch so a later prefill of up to max_tokens
 * allocates nothing. Callers that time a forward pass call this outside the
 * timed bracket. */
model_status model_reserve(model *m, size_t max_tokens);

/* PREFILL: one forward pass over n_ids tokens, writing the logits for EVERY
 * position. logits_out must hold n_ids * vocab_size floats; logits_cap is that
 * count, in floats. Row t holds the logits at position t.
 *
 * The head is computed at every position deliberately: the elementwise logit
 * comparison against the reference oracle compares all of them, and the oracle
 * is built to match this choice. */
model_status model_prefill(model *m, const int32_t *ids, size_t n_ids,
                           float *logits_out, size_t logits_cap);

/* DECODE: one decode step at a context of n_ids tokens, writing the logits for
 * the LAST position only. logits_cap is in floats and must be at least
 * vocab_size.
 *
 * With no KV cache this re-runs the twelve layers over the whole context and
 * then computes the head once. That cost growing with context length is the
 * baseline Stage 4 exists to flatten; it is not an oversight. */
model_status model_decode_step(model *m, const int32_t *ids, size_t n_ids,
                               float *logits_out, size_t logits_cap);

/* Greedy selection: the index of the maximum, ties broken by lowest index. */
int32_t model_argmax(const float *logits, size_t n);

/* ---- tie of the language-model head ------------------------------------
 * No tensor in the weight file is named lm_head and exactly one matrix carries
 * the vocabulary size against the embedding width, so the head is TIED to
 * wte.weight. These two accessors return the SAME POINTER when the head is
 * tied, which is how tests/test_model.c verifies the tie: not by comparing
 * values, which a transposed copy would also pass, but by identity of storage. */
const float *model_token_embedding(const model *m);
const float *model_head_weight(const model *m);

/* ---- pieces exposed for the unit test ----------------------------------
 * The normalization step of layernorm WITHOUT the affine transform, so a test
 * can assert zero mean and unit variance per row -- which the affine transform
 * would otherwise hide. The forward pass calls this and then applies gain and
 * bias, so the test exercises the same code the engine runs. */
void model_layernorm_normalize(const float *x, int n, float eps, float *out);

/* Attention row-sum statistics, collected only when explicitly enabled, so a
 * timed run never pays for them. When on, every softmax row of every head, every
 * position and every layer of the most recent forward pass is accumulated into
 * a min/max pair, which must bracket 1. Off by default; enabling resets it. */
void model_collect_attn_stats(model *m, int enable);
void model_attn_rowsum_range(const model *m, double *out_min, double *out_max,
                             size_t *out_rows);

const char *model_strerror(model_status s);

#ifdef __cplusplus
}
#endif

#endif /* TIE_MODEL_H */
