/* tokenizer.h -- byte-level BPE tokenizer, Stage 1.
 *
 * The interface takes and returns BYTES WITH AN EXPLICIT LENGTH, never a
 * NUL-terminated string. The round-trip corpus this is tested against contains
 * embedded NUL bytes and byte sequences that are not valid UTF-8 at all, and a
 * char* interface would silently truncate the first and mangle the second.
 *
 * What it loads, and why that artifact: models/gpt2/tokenizer.json. See the
 * determination recorded at the top of tokenizer.c.
 */
#ifndef TIE_TOKENIZER_H
#define TIE_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TOK_OK = 0,
    TOK_ERR_OPEN,            /* the artifact could not be opened            */
    TOK_ERR_READ,            /* a read failed or returned short             */
    TOK_ERR_BAD_JSON,        /* the artifact is not the accepted JSON subset */
    TOK_ERR_NO_VOCAB,        /* no model.vocab member                       */
    TOK_ERR_NO_MERGES,       /* no model.merges member                      */
    TOK_ERR_BAD_MERGE,       /* a merge entry is not "<left> <right>"       */
    TOK_ERR_UNKNOWN_SYMBOL,  /* a symbol produced by the merge loop is not in the vocabulary */
    TOK_ERR_BAD_ID,          /* decode was handed an id outside the vocabulary */
    TOK_ERR_TRUNCATED,       /* the caller's output buffer is too small     */
    TOK_ERR_NOMEM,
    TOK_ERR_ARG
} tok_status;

typedef struct tokenizer tokenizer;

/* Loads vocabulary, merge ranks and added tokens from the artifact. */
tok_status tok_load(const char *tokenizer_json_path, tokenizer **out);
void       tok_free(tokenizer *t);

/* The vocabulary size the artifact declares: the number of entries in its
 * vocabulary map. Read from the artifact, never assumed. */
size_t tok_vocab_size(const tokenizer *t);

/* Encodes n_bytes bytes into ids.
 * An id sequence can never be longer than the input in bytes -- every byte
 * starts as exactly one symbol and merges only ever shorten the sequence -- so
 * a caller that passes cap >= n_bytes can never see TOK_ERR_TRUNCATED. On
 * truncation *n_out holds the number of ids written before the buffer ran out. */
tok_status tok_encode(const tokenizer *t, const unsigned char *bytes, size_t n_bytes,
                      int32_t *ids, size_t cap, size_t *n_out);

/* Decodes ids back to bytes. Special tokens decode to their literal text, which
 * is what makes the round-trip byte-exact. */
tok_status tok_decode(const tokenizer *t, const int32_t *ids, size_t n_ids,
                      unsigned char *out, size_t cap, size_t *n_out);

/* The vocabulary spelling of an id (in the byte-level alphabet, NOT the bytes
 * it decodes to), or NULL if the id is outside the vocabulary. */
const char *tok_token_string(const tokenizer *t, int32_t id);

/* The merge rank the artifact gives this pair, or -1 if the pair is not a
 * merge. Rank 0 is the first merge in the artifact and therefore the strongest. */
int tok_merge_rank(const tokenizer *t, const char *left, const char *right);

size_t      tok_merge_count(const tokenizer *t);
size_t      tok_added_token_count(const tokenizer *t);
const char *tok_added_token_at(const tokenizer *t, size_t i, int32_t *id);

const char *tok_strerror(tok_status s);

#ifdef __cplusplus
}
#endif

#endif /* TIE_TOKENIZER_H */
