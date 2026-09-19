/* tokenizer.c -- byte-level BPE tokenizer, Stage 1.
 *
 * ---------------------------------------------------------------------------
 * WHICH ARTIFACT THIS CONSUMES, AND WHY
 * ---------------------------------------------------------------------------
 * Three tokenizer artifacts shipped with the weights:
 *     vocab.json        50257 entries, a flat string -> id object
 *     merges.txt        a "#version: 0.2" header line then 50000 merge lines
 *     tokenizer.json    the combined file: the same vocabulary, the same
 *                       merges, plus the added-token table and the
 *                       pre-tokenizer / decoder configuration
 *
 * They were compared before anything was written, and they AGREE EXACTLY: the
 * vocabulary object in tokenizer.json is equal to vocab.json entry for entry,
 * and its merge list is equal to merges.txt line for line after the version
 * header. No disagreement to report.
 *
 * This implementation reads tokenizer.json, for three reasons:
 *   1. It is the only artifact that DECLARES THE ADDED TOKENS. The reference
 *      splits "<|endoftext|>" out of ordinary text as id 50256 before the
 *      regex ever sees it; built from vocab.json and merges.txt alone, a
 *      tokenizer would have to be TOLD that such a token exists, which is
 *      exactly the kind of assumption this stage exists to avoid.
 *   2. It is the only artifact that declares the pre-tokenizer configuration
 *      (ByteLevel, add_prefix_space false), which changes the ids of every
 *      string that starts with a letter.
 *   3. It is the easier parse, not the harder one: tokenizer.json uses only
 *      the \" and \\ escapes (311 and 121 occurrences), while vocab.json uses
 *      \uXXXX 35908 times and would drag UTF-16 surrogate reassembly into the
 *      inference path for no gain.
 *
 * ---------------------------------------------------------------------------
 * NO LIBRARY DOES THE TOKENIZING, AND NONE DOES THE PARSING
 * ---------------------------------------------------------------------------
 * The JSON scan below is purpose-built for this artifact, for the same reason
 * safetensors.c has its own: a dependency here is a dependency in the inference
 * path forever. The two scanners are deliberately NOT shared -- the subsets
 * differ (this one handles a 1.3 MB nested document with escapes and a 50257
 * entry string table; that one handles a 14 KB flat document with none), and a
 * single general parser covering both would be the very dependency-shaped thing
 * being avoided.
 *
 * ---------------------------------------------------------------------------
 * THE PRE-TOKENIZER, AND WHERE ITS FACTS CAME FROM
 * ---------------------------------------------------------------------------
 * Byte-level BPE does not run the merge loop over the whole input: it first
 * splits the text into pieces and runs the merges inside each piece. The split
 * is the GPT-2 pattern
 *
 *     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 *
 * applied leftmost-first, alternative by alternative. It is implemented here
 * directly rather than with a regex engine. Every behaviour below was CHECKED
 * against the reference pre-tokenizer before it was written, not deduced:
 *
 *     "a b  c   d"  ->  a | Gb | G | Gc | GG | Gd      (G = the space symbol)
 *     "   "         ->  GGG            (a run at end of input stays whole)
 *     "x  \n  y"    ->  x | GGCG | Gy  (a run before a non-space gives up its
 *                                       last character to the next piece)
 *     "x\r\ny"      ->  x | \r | \n    (same rule, twice)
 *     "IT'S"        ->  IT | ' | S     (the contraction forms are lower case)
 *     "a1b"         ->  a | 1 | b      (letters and numbers never share a piece)
 *
 * The \s+(?!\S) / \s+ pair is what produces the third and fourth lines: the
 * lookahead makes the longest whitespace run that is NOT followed by a
 * non-space character win, which for a run followed by text is the run minus
 * its last character, and for a run at end of input is all of it.
 *
 * CODEPOINT CLASSES. \p{L}, \p{N} and \s are Unicode properties, and which
 * codepoints they cover depends on the Unicode version of whatever engine
 * evaluates them. Rather than assume a version, the table at the bottom of this
 * file was DERIVED FROM THE REFERENCE ITSELF: every codepoint from U+0000 to
 * U+10FFFF was put through the reference pre-tokenizer and classified by which
 * piece it landed in. That mattered -- the result disagrees with Python 3.12's
 * own unicodedata tables at 5008 codepoints, so a table built from the more
 * convenient source would have been wrong in exactly the places nobody tests.
 *
 * BYTES THAT ARE NOT VALID UTF-8. The reference cannot be consulted: its API
 * takes a string, and both bytes and a surrogate-escaped string are rejected
 * with TypeError "TextInputSequence must be str". There is therefore no
 * reference id sequence for that class of input and none is invented. The rule
 * here, chosen and documented rather than discovered: a byte that does not
 * begin a well-formed UTF-8 sequence (including an overlong encoding, a
 * surrogate, or a value above U+10FFFF) is consumed as ONE byte of class
 * "other". Encoding stays total, decoding stays byte-exact, and the id sequence
 * for such input is this implementation's own, not a match to anything.
 *
 * THE BYTE-LEVEL ALPHABET. Bytes are mapped to printable codepoints before the
 * merges run, by the byte-level construction (the printable ASCII and Latin-1
 * ranges map to themselves; the remaining 68 bytes map to U+0100 upward in
 * increasing byte order). Checked against the reference rather than trusted:
 * 243 of the 256 byte values were observed in the reference's own mapped output
 * and every one agreed, and the 256-character alphabet set matches the
 * reference's alphabet exactly. The 13 that could not be checked -- 0xC0, 0xC1
 * and 0xF5..0xFF -- are precisely the bytes that cannot appear in valid UTF-8,
 * so the reference can never be made to emit them; they are exercised instead
 * by the invalid-UTF-8 records of the round-trip corpus.
 */

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- tables --- */

/* Codepoint classes, as the reference's own regex engine draws them. */
#define TC_OTHER  0
#define TC_LETTER 1
#define TC_NUMBER 2
#define TC_SPACE  3

typedef struct { uint32_t lo, hi; unsigned char cls; } tok_range;

/* The table itself is at the bottom of this file, where its size does not push
 * the code out of view. */
static int cp_class(uint32_t cp);

/* --------------------------------------------------------------- structs --- */

typedef struct {
    char    *left;
    char    *right;
    int      rank;
} tok_merge;

typedef struct {
    char    *content;
    int32_t  id;
} tok_added;

struct tokenizer {
    /* vocabulary */
    char   **id_to_str;      /* id -> spelling, NUL terminated               */
    size_t   n_vocab;
    int32_t *vhash;          /* open addressing; -1 empty                    */
    size_t   vhash_mask;

    /* merges */
    tok_merge *merges;
    size_t     n_merges;
    int32_t   *mhash;        /* index into merges; -1 empty                  */
    size_t     mhash_mask;

    /* added tokens */
    tok_added *added;
    size_t     n_added;

    /* the byte-level alphabet */
    uint32_t byte_to_cp[256];
    char     byte_to_sym[256][4];   /* the UTF-8 spelling of byte_to_cp[b]   */
    int      byte_to_sym_len[256];
    int      cp_to_byte[512];       /* -1 where the codepoint is not one     */
};

const char *tok_strerror(tok_status s)
{
    switch (s) {
    case TOK_OK:                  return "ok";
    case TOK_ERR_OPEN:            return "the artifact could not be opened";
    case TOK_ERR_READ:            return "a read failed or returned short";
    case TOK_ERR_BAD_JSON:        return "the artifact is not the accepted JSON subset";
    case TOK_ERR_NO_VOCAB:        return "the artifact declares no vocabulary";
    case TOK_ERR_NO_MERGES:       return "the artifact declares no merges";
    case TOK_ERR_BAD_MERGE:       return "a merge entry is not a left/right pair";
    case TOK_ERR_UNKNOWN_SYMBOL:  return "a merged symbol is not in the vocabulary";
    case TOK_ERR_BAD_ID:          return "the id is outside the vocabulary";
    case TOK_ERR_TRUNCATED:       return "the output buffer is too small";
    case TOK_ERR_NOMEM:           return "allocation failed";
    case TOK_ERR_ARG:             return "invalid argument";
    }
    return "unknown status";
}

/* ----------------------------------------------------------------- hash ---- */

static uint64_t fnv1a(const char *p, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= (unsigned char)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static size_t pow2_at_least(size_t n)
{
    size_t c = 16;
    while (c < n * 2) c *= 2;
    return c;
}

static void vhash_put(tokenizer *t, int32_t id)
{
    const char *s = t->id_to_str[id];
    size_t i = (size_t)fnv1a(s, strlen(s)) & t->vhash_mask;
    while (t->vhash[i] != -1) i = (i + 1) & t->vhash_mask;
    t->vhash[i] = id;
}

static int32_t vhash_get(const tokenizer *t, const char *s, size_t n)
{
    size_t i = (size_t)fnv1a(s, n) & t->vhash_mask;
    while (t->vhash[i] != -1) {
        const char *c = t->id_to_str[t->vhash[i]];
        if (strlen(c) == n && memcmp(c, s, n) == 0) return t->vhash[i];
        i = (i + 1) & t->vhash_mask;
    }
    return -1;
}

static uint64_t pair_hash(const char *l, size_t ln, const char *r, size_t rn)
{
    uint64_t h = fnv1a(l, ln);
    h ^= 0x9e3779b97f4a7c15ull;
    h *= 1099511628211ull;
    h ^= fnv1a(r, rn);
    return h;
}

static void mhash_put(tokenizer *t, int32_t idx)
{
    const tok_merge *m = &t->merges[idx];
    size_t i = (size_t)pair_hash(m->left, strlen(m->left),
                                 m->right, strlen(m->right)) & t->mhash_mask;
    while (t->mhash[i] != -1) i = (i + 1) & t->mhash_mask;
    t->mhash[i] = idx;
}

static int mhash_get(const tokenizer *t, const char *l, size_t ln,
                     const char *r, size_t rn)
{
    size_t i = (size_t)pair_hash(l, ln, r, rn) & t->mhash_mask;
    while (t->mhash[i] != -1) {
        const tok_merge *m = &t->merges[t->mhash[i]];
        if (strlen(m->left) == ln && memcmp(m->left, l, ln) == 0 &&
            strlen(m->right) == rn && memcmp(m->right, r, rn) == 0)
            return m->rank;
        i = (i + 1) & t->mhash_mask;
    }
    return -1;
}

/* ---------------------------------------------------- the JSON scan ------- */
/*
 * Accepted: objects, arrays, strings with the escapes \" \\ \/ \b \f \n \r \t,
 * unsigned integers, true, false and null (skipped -- the artifact carries them
 * in members this reader does not need), and whitespace between tokens.
 * Rejected: \uXXXX (the artifact contains none), signed or fractional numbers,
 * and anything unterminated or unbalanced.
 */

typedef struct { const char *p, *end; } tj_scan;

static void tj_ws(tj_scan *s)
{
    while (s->p < s->end && (*s->p == ' ' || *s->p == '\t' ||
                             *s->p == '\r' || *s->p == '\n')) ++s->p;
}
static int tj_eat(tj_scan *s, char c)
{
    tj_ws(s);
    if (s->p < s->end && *s->p == c) { ++s->p; return 1; }
    return 0;
}
static int tj_peek(tj_scan *s, char c) { tj_ws(s); return s->p < s->end && *s->p == c; }

/* Reads a string into a freshly allocated buffer. Returns NULL on rejection. */
static char *tj_string(tj_scan *s, size_t *out_len)
{
    tj_ws(s);
    if (s->p >= s->end || *s->p != '"') return NULL;
    const char *q = s->p + 1;
    /* measure first, so the copy is exact */
    size_t n = 0;
    while (q < s->end && *q != '"') {
        if (*q == '\\') {
            if (q + 1 >= s->end) return NULL;
            if (q[1] == 'u') return NULL;
            q += 2;
        } else if ((unsigned char)*q < 0x20) {
            return NULL;
        } else {
            ++q;
        }
        ++n;
    }
    if (q >= s->end) return NULL;

    char *buf = (char *)malloc(n + 1);
    if (!buf) return NULL;
    size_t k = 0;
    const char *r = s->p + 1;
    while (r < q) {
        if (*r == '\\') {
            switch (r[1]) {
            case '"':  buf[k++] = '"';  break;
            case '\\': buf[k++] = '\\'; break;
            case '/':  buf[k++] = '/';  break;
            case 'b':  buf[k++] = '\b'; break;
            case 'f':  buf[k++] = '\f'; break;
            case 'n':  buf[k++] = '\n'; break;
            case 'r':  buf[k++] = '\r'; break;
            case 't':  buf[k++] = '\t'; break;
            default:   free(buf); return NULL;
            }
            r += 2;
        } else {
            buf[k++] = *r++;
        }
    }
    buf[k] = '\0';
    s->p = q + 1;
    if (out_len) *out_len = k;
    return buf;
}

static int tj_uint(tj_scan *s, uint64_t *out)
{
    tj_ws(s);
    if (s->p >= s->end || *s->p < '0' || *s->p > '9') return -1;
    uint64_t v = 0;
    while (s->p < s->end && *s->p >= '0' && *s->p <= '9') {
        v = v * 10u + (uint64_t)(*s->p - '0');
        ++s->p;
    }
    if (s->p < s->end && (*s->p == '.' || *s->p == 'e' || *s->p == 'E')) return -1;
    *out = v;
    return 0;
}

static int tj_skip(tj_scan *s, int depth)
{
    if (depth > 64) return -1;
    tj_ws(s);
    if (s->p >= s->end) return -1;
    if (*s->p == '"') {
        char *tmp = tj_string(s, NULL);
        if (!tmp) return -1;
        free(tmp);
        return 0;
    }
    if (*s->p == '{' || *s->p == '[') {
        char open = *s->p, close = (open == '{') ? '}' : ']';
        ++s->p;
        if (tj_eat(s, close)) return 0;
        for (;;) {
            if (open == '{') {
                char *k = tj_string(s, NULL);
                if (!k) return -1;
                free(k);
                if (!tj_eat(s, ':')) return -1;
            }
            if (tj_skip(s, depth + 1) != 0) return -1;
            if (tj_eat(s, ',')) continue;
            if (tj_eat(s, close)) return 0;
            return -1;
        }
    }
    if (s->end - s->p >= 4 && memcmp(s->p, "true", 4) == 0) { s->p += 4; return 0; }
    if (s->end - s->p >= 5 && memcmp(s->p, "false", 5) == 0) { s->p += 5; return 0; }
    if (s->end - s->p >= 4 && memcmp(s->p, "null", 4) == 0) { s->p += 4; return 0; }
    {
        uint64_t dummy;
        return tj_uint(s, &dummy);
    }
}

/* ------------------------------------------------------------ byte level --- */

static int utf8_put(uint32_t cp, char *out)
{
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Decodes one well-formed UTF-8 sequence at p. Returns its length in bytes and
 * writes the codepoint, or returns 0 for anything malformed -- an unexpected
 * continuation byte, a truncated sequence, an overlong encoding, a surrogate,
 * or a value above U+10FFFF. */
static int utf8_next(const unsigned char *p, size_t avail, uint32_t *cp)
{
    if (avail == 0) return 0;
    unsigned char c = p[0];
    if (c < 0x80) { *cp = c; return 1; }
    if (c < 0xC2) return 0;                       /* continuation, or overlong lead */
    if (c < 0xE0) {
        if (avail < 2 || (p[1] & 0xC0) != 0x80) return 0;
        *cp = ((uint32_t)(c & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
        return 2;
    }
    if (c < 0xF0) {
        if (avail < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return 0;
        uint32_t v = ((uint32_t)(c & 0x0F) << 12) |
                     ((uint32_t)(p[1] & 0x3F) << 6) | (uint32_t)(p[2] & 0x3F);
        if (v < 0x800) return 0;                  /* overlong */
        if (v >= 0xD800 && v <= 0xDFFF) return 0; /* surrogate */
        *cp = v;
        return 3;
    }
    if (c < 0xF5) {
        if (avail < 4 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 ||
            (p[3] & 0xC0) != 0x80) return 0;
        uint32_t v = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
                     ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
        if (v < 0x10000 || v > 0x10FFFF) return 0;
        *cp = v;
        return 4;
    }
    return 0;
}

static void build_byte_alphabet(tokenizer *t)
{
    /* The byte-level construction: the printable ASCII range and the two
     * printable Latin-1 ranges map to themselves; every other byte takes the
     * next free codepoint from U+0100 upward, in increasing byte order. */
    int taken[256];
    memset(taken, 0, sizeof taken);
    for (int b = '!'; b <= '~'; ++b)     taken[b] = 1;
    for (int b = 0xA1; b <= 0xAC; ++b)   taken[b] = 1;
    for (int b = 0xAE; b <= 0xFF; ++b)   taken[b] = 1;

    uint32_t next = 256;
    for (int b = 0; b < 256; ++b)
        t->byte_to_cp[b] = taken[b] ? (uint32_t)b : next++;

    for (int i = 0; i < 512; ++i) t->cp_to_byte[i] = -1;
    for (int b = 0; b < 256; ++b) {
        t->byte_to_sym_len[b] = utf8_put(t->byte_to_cp[b], t->byte_to_sym[b]);
        t->byte_to_sym[b][t->byte_to_sym_len[b]] = '\0';
        t->cp_to_byte[t->byte_to_cp[b]] = b;
    }
}

/* ----------------------------------------------------------------- load ---- */

static char *read_whole_file(const char *path, size_t *len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long n = ftell(fp);
    if (n < 0 || fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, fp);
    fclose(fp);
    buf[got] = '\0';
    if (len) *len = got;
    return buf;
}

static tok_status parse_vocab(tj_scan *s, tokenizer *t)
{
    if (!tj_eat(s, '{')) return TOK_ERR_BAD_JSON;
    size_t cap = 0;
    if (!tj_peek(s, '}')) {
        for (;;) {
            size_t klen = 0;
            char *key = tj_string(s, &klen);
            if (!key) return TOK_ERR_BAD_JSON;
            if (!tj_eat(s, ':')) { free(key); return TOK_ERR_BAD_JSON; }
            uint64_t id;
            if (tj_uint(s, &id) != 0) { free(key); return TOK_ERR_BAD_JSON; }

            if ((size_t)id + 1 > cap) {
                size_t ncap = cap ? cap : 1024;
                while (ncap < (size_t)id + 1) ncap *= 2;
                char **n = (char **)realloc(t->id_to_str, ncap * sizeof *n);
                if (!n) { free(key); return TOK_ERR_NOMEM; }
                memset(n + cap, 0, (ncap - cap) * sizeof *n);
                t->id_to_str = n;
                cap = ncap;
            }
            if ((size_t)id + 1 > t->n_vocab) t->n_vocab = (size_t)id + 1;
            free(t->id_to_str[id]);
            t->id_to_str[id] = key;

            if (tj_eat(s, ',')) continue;
            break;
        }
    }
    if (!tj_eat(s, '}')) return TOK_ERR_BAD_JSON;
    return TOK_OK;
}

static tok_status parse_merges(tj_scan *s, tokenizer *t)
{
    if (!tj_eat(s, '[')) return TOK_ERR_BAD_JSON;
    size_t cap = 0;
    if (!tj_peek(s, ']')) {
        for (;;) {
            size_t len = 0;
            char *line = tj_string(s, &len);
            if (!line) return TOK_ERR_BAD_JSON;
            char *sp = strchr(line, ' ');
            if (!sp || sp == line || sp[1] == '\0') { free(line); return TOK_ERR_BAD_MERGE; }
            *sp = '\0';
            char *left = line;
            char *right = sp + 1;

            if (t->n_merges + 1 > cap) {
                size_t ncap = cap ? cap * 2 : 1024;
                tok_merge *n = (tok_merge *)realloc(t->merges, ncap * sizeof *n);
                if (!n) { free(line); return TOK_ERR_NOMEM; }
                t->merges = n;
                cap = ncap;
            }
            char *lc = (char *)malloc(strlen(left) + 1);
            char *rc = (char *)malloc(strlen(right) + 1);
            if (!lc || !rc) { free(lc); free(rc); free(line); return TOK_ERR_NOMEM; }
            memcpy(lc, left, strlen(left) + 1);
            memcpy(rc, right, strlen(right) + 1);
            t->merges[t->n_merges].left = lc;
            t->merges[t->n_merges].right = rc;
            t->merges[t->n_merges].rank = (int)t->n_merges;
            ++t->n_merges;
            free(line);

            if (tj_eat(s, ',')) continue;
            break;
        }
    }
    if (!tj_eat(s, ']')) return TOK_ERR_BAD_JSON;
    return TOK_OK;
}

static tok_status parse_model(tj_scan *s, tokenizer *t)
{
    if (!tj_eat(s, '{')) return TOK_ERR_BAD_JSON;
    if (tj_eat(s, '}')) return TOK_ERR_NO_VOCAB;
    for (;;) {
        char *key = tj_string(s, NULL);
        if (!key) return TOK_ERR_BAD_JSON;
        if (!tj_eat(s, ':')) { free(key); return TOK_ERR_BAD_JSON; }
        tok_status rc = TOK_OK;
        if (strcmp(key, "vocab") == 0)       rc = parse_vocab(s, t);
        else if (strcmp(key, "merges") == 0) rc = parse_merges(s, t);
        else if (tj_skip(s, 0) != 0)         rc = TOK_ERR_BAD_JSON;
        free(key);
        if (rc != TOK_OK) return rc;
        if (tj_eat(s, ',')) continue;
        break;
    }
    if (!tj_eat(s, '}')) return TOK_ERR_BAD_JSON;
    return TOK_OK;
}

static tok_status parse_added(tj_scan *s, tokenizer *t)
{
    if (!tj_eat(s, '[')) return TOK_ERR_BAD_JSON;
    if (tj_eat(s, ']')) return TOK_OK;
    size_t cap = 0;
    for (;;) {
        if (!tj_eat(s, '{')) return TOK_ERR_BAD_JSON;
        char *content = NULL;
        uint64_t id = 0;
        int have_id = 0;
        if (!tj_peek(s, '}')) {
            for (;;) {
                char *key = tj_string(s, NULL);
                if (!key) { free(content); return TOK_ERR_BAD_JSON; }
                if (!tj_eat(s, ':')) { free(key); free(content); return TOK_ERR_BAD_JSON; }
                if (strcmp(key, "content") == 0) {
                    free(content);
                    content = tj_string(s, NULL);
                    if (!content) { free(key); return TOK_ERR_BAD_JSON; }
                } else if (strcmp(key, "id") == 0) {
                    if (tj_uint(s, &id) != 0) { free(key); free(content); return TOK_ERR_BAD_JSON; }
                    have_id = 1;
                } else if (tj_skip(s, 0) != 0) {
                    free(key); free(content); return TOK_ERR_BAD_JSON;
                }
                free(key);
                if (tj_eat(s, ',')) continue;
                break;
            }
        }
        if (!tj_eat(s, '}')) { free(content); return TOK_ERR_BAD_JSON; }
        if (!content || !have_id) { free(content); return TOK_ERR_BAD_JSON; }

        if (t->n_added + 1 > cap) {
            size_t ncap = cap ? cap * 2 : 8;
            tok_added *n = (tok_added *)realloc(t->added, ncap * sizeof *n);
            if (!n) { free(content); return TOK_ERR_NOMEM; }
            t->added = n;
            cap = ncap;
        }
        t->added[t->n_added].content = content;
        t->added[t->n_added].id = (int32_t)id;
        ++t->n_added;

        if (tj_eat(s, ',')) continue;
        break;
    }
    if (!tj_eat(s, ']')) return TOK_ERR_BAD_JSON;
    return TOK_OK;
}

tok_status tok_load(const char *path, tokenizer **out)
{
    if (!path || !out) return TOK_ERR_ARG;
    *out = NULL;

    size_t len = 0;
    char *text = read_whole_file(path, &len);
    if (!text) return TOK_ERR_OPEN;

    tokenizer *t = (tokenizer *)calloc(1, sizeof *t);
    if (!t) { free(text); return TOK_ERR_NOMEM; }
    build_byte_alphabet(t);

    tj_scan s = { text, text + len };
    tok_status rc = TOK_OK;
    if (!tj_eat(&s, '{')) rc = TOK_ERR_BAD_JSON;
    while (rc == TOK_OK) {
        char *key = tj_string(&s, NULL);
        if (!key) { rc = TOK_ERR_BAD_JSON; break; }
        if (!tj_eat(&s, ':')) { free(key); rc = TOK_ERR_BAD_JSON; break; }
        if (strcmp(key, "model") == 0)             rc = parse_model(&s, t);
        else if (strcmp(key, "added_tokens") == 0) rc = parse_added(&s, t);
        else if (tj_skip(&s, 0) != 0)              rc = TOK_ERR_BAD_JSON;
        free(key);
        if (rc != TOK_OK) break;
        if (tj_eat(&s, ',')) continue;
        if (!tj_eat(&s, '}')) rc = TOK_ERR_BAD_JSON;
        break;
    }
    free(text);

    if (rc == TOK_OK && t->n_vocab == 0)  rc = TOK_ERR_NO_VOCAB;
    if (rc == TOK_OK && t->n_merges == 0) rc = TOK_ERR_NO_MERGES;

    /* Every id below the declared size must have a spelling; a hole would make
     * decode ambiguous rather than merely wrong. */
    for (size_t i = 0; rc == TOK_OK && i < t->n_vocab; ++i)
        if (!t->id_to_str[i]) rc = TOK_ERR_NO_VOCAB;

    if (rc == TOK_OK) {
        size_t vs = pow2_at_least(t->n_vocab);
        t->vhash = (int32_t *)malloc(vs * sizeof *t->vhash);
        if (!t->vhash) rc = TOK_ERR_NOMEM;
        else {
            memset(t->vhash, 0xFF, vs * sizeof *t->vhash);
            t->vhash_mask = vs - 1;
            for (size_t i = 0; i < t->n_vocab; ++i) vhash_put(t, (int32_t)i);
        }
    }
    if (rc == TOK_OK) {
        size_t ms = pow2_at_least(t->n_merges);
        t->mhash = (int32_t *)malloc(ms * sizeof *t->mhash);
        if (!t->mhash) rc = TOK_ERR_NOMEM;
        else {
            memset(t->mhash, 0xFF, ms * sizeof *t->mhash);
            t->mhash_mask = ms - 1;
            for (size_t i = 0; i < t->n_merges; ++i) mhash_put(t, (int32_t)i);
        }
    }

    if (rc != TOK_OK) { tok_free(t); return rc; }
    *out = t;
    return TOK_OK;
}

void tok_free(tokenizer *t)
{
    if (!t) return;
    for (size_t i = 0; i < t->n_vocab; ++i) free(t->id_to_str[i]);
    free(t->id_to_str);
    free(t->vhash);
    for (size_t i = 0; i < t->n_merges; ++i) {
        free(t->merges[i].left);
        free(t->merges[i].right);
    }
    free(t->merges);
    free(t->mhash);
    for (size_t i = 0; i < t->n_added; ++i) free(t->added[i].content);
    free(t->added);
    free(t);
}

size_t tok_vocab_size(const tokenizer *t)      { return t ? t->n_vocab : 0; }
size_t tok_merge_count(const tokenizer *t)     { return t ? t->n_merges : 0; }
size_t tok_added_token_count(const tokenizer *t) { return t ? t->n_added : 0; }

const char *tok_added_token_at(const tokenizer *t, size_t i, int32_t *id)
{
    if (!t || i >= t->n_added) return NULL;
    if (id) *id = t->added[i].id;
    return t->added[i].content;
}

const char *tok_token_string(const tokenizer *t, int32_t id)
{
    if (!t || id < 0 || (size_t)id >= t->n_vocab) return NULL;
    return t->id_to_str[id];
}

int tok_merge_rank(const tokenizer *t, const char *left, const char *right)
{
    if (!t || !left || !right) return -1;
    return mhash_get(t, left, strlen(left), right, strlen(right));
}

/* --------------------------------------------------------- pre-tokenizer --- */

/* Class of the sequence starting at p, and its length in bytes. A byte that
 * does not begin a well-formed UTF-8 sequence is one byte of class "other". */
static int next_char(const unsigned char *p, size_t avail, int *cls)
{
    uint32_t cp;
    int n = utf8_next(p, avail, &cp);
    if (n == 0) { *cls = TC_OTHER; return 1; }
    *cls = cp_class(cp);
    return n;
}

static const char *const CONTRACTIONS[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
static const size_t N_CONTRACTIONS = sizeof CONTRACTIONS / sizeof CONTRACTIONS[0];

/* The end of the piece that starts at i, following the pattern's alternatives
 * in order. Always advances by at least one character. */
static size_t piece_end(const unsigned char *b, size_t n, size_t i)
{
    /* 1. the contraction forms */
    if (b[i] == '\'') {
        for (size_t k = 0; k < N_CONTRACTIONS; ++k) {
            size_t len = strlen(CONTRACTIONS[k]);
            if (i + len <= n && memcmp(b + i, CONTRACTIONS[k], len) == 0)
                return i + len;
        }
    }

    /* 2-4. an optional single space, then a run of one class */
    for (int target = 0; target < 3; ++target) {
        int want = (target == 0) ? TC_LETTER : (target == 1) ? TC_NUMBER : TC_OTHER;
        size_t j = i;
        if (b[j] == ' ' && j + 1 < n) ++j;
        if (j >= n) continue;
        int cls;
        int adv = next_char(b + j, n - j, &cls);
        if (cls != want) continue;
        size_t e = j + (size_t)adv;
        while (e < n) {
            int c2;
            int a2 = next_char(b + e, n - e, &c2);
            if (c2 != want) break;
            e += (size_t)a2;
        }
        return e;
    }

    /* 5-6. whitespace. The lookahead in \s+(?!\S) makes a run that is followed
     * by a non-space character give up its last character to the next piece;
     * a run that reaches the end of the input stays whole. */
    {
        int cls;
        int adv = next_char(b + i, n - i, &cls);
        if (cls == TC_SPACE) {
            size_t e = i + (size_t)adv, last = i;
            while (e < n) {
                int c2;
                int a2 = next_char(b + e, n - e, &c2);
                if (c2 != TC_SPACE) break;
                last = e;
                e += (size_t)a2;
            }
            if (e >= n) return e;          /* run reaches end of input       */
            if (last > i) return last;     /* run minus its last character   */
            return e;                      /* a single whitespace character  */
        }
        /* Not whitespace and not matched above: one character of "other". */
        return i + (size_t)adv;
    }
}

/* ---------------------------------------------------------------- encode --- */

typedef struct {
    size_t start, len;   /* into the mapped symbol buffer */
} sym;

static tok_status encode_piece(const tokenizer *t, const unsigned char *b,
                               size_t n, int32_t *ids, size_t cap, size_t *n_out)
{
    if (n == 0) return TOK_OK;

    /* map every byte to its symbol; the mapped form is at most 2 bytes per byte */
    char *buf = (char *)malloc(n * 2 + 1);
    sym  *syms = (sym *)malloc(n * sizeof *syms);
    if (!buf || !syms) { free(buf); free(syms); return TOK_ERR_NOMEM; }

    size_t bn = 0, sn = 0;
    for (size_t i = 0; i < n; ++i) {
        int L = t->byte_to_sym_len[b[i]];
        memcpy(buf + bn, t->byte_to_sym[b[i]], (size_t)L);
        syms[sn].start = bn;
        syms[sn].len = (size_t)L;
        ++sn;
        bn += (size_t)L;
    }
    buf[bn] = '\0';

    /* the merge loop: repeatedly take the lowest-ranked adjacent pair the
     * artifact knows, and merge every non-overlapping occurrence of it */
    for (;;) {
        int best = -1;
        size_t best_at = 0;
        for (size_t i = 0; i + 1 < sn; ++i) {
            int r = mhash_get(t, buf + syms[i].start, syms[i].len,
                              buf + syms[i + 1].start, syms[i + 1].len);
            if (r >= 0 && (best < 0 || r < best)) { best = r; best_at = i; }
        }
        if (best < 0) break;

        const char *bl = buf + syms[best_at].start;
        size_t blen = syms[best_at].len;
        const char *br = buf + syms[best_at + 1].start;
        size_t brlen = syms[best_at + 1].len;

        size_t w = 0;
        for (size_t i = 0; i < sn; ) {
            if (i + 1 < sn &&
                syms[i].len == blen && memcmp(buf + syms[i].start, bl, blen) == 0 &&
                syms[i + 1].len == brlen && memcmp(buf + syms[i + 1].start, br, brlen) == 0) {
                /* adjacent symbols are contiguous in buf, so a merge is a join */
                syms[w].start = syms[i].start;
                syms[w].len = syms[i].len + syms[i + 1].len;
                ++w;
                i += 2;
            } else {
                syms[w++] = syms[i++];
            }
        }
        sn = w;
    }

    tok_status rc = TOK_OK;
    for (size_t i = 0; i < sn; ++i) {
        int32_t id = vhash_get(t, buf + syms[i].start, syms[i].len);
        if (id < 0) { rc = TOK_ERR_UNKNOWN_SYMBOL; break; }
        if (*n_out >= cap) { rc = TOK_ERR_TRUNCATED; break; }
        ids[(*n_out)++] = id;
    }
    free(buf);
    free(syms);
    return rc;
}

/* Length of the added token matching at b+i, longest first, or 0 for none. */
static size_t added_match(const tokenizer *t, const unsigned char *b, size_t n,
                          size_t i, int32_t *id)
{
    size_t best = 0;
    for (size_t k = 0; k < t->n_added; ++k) {
        size_t len = strlen(t->added[k].content);
        if (len > best && i + len <= n &&
            memcmp(b + i, t->added[k].content, len) == 0) {
            best = len;
            *id = t->added[k].id;
        }
    }
    return best;
}

/* Pre-tokenizes [from, to) and runs the merges inside each piece. */
static tok_status encode_segment(const tokenizer *t, const unsigned char *bytes,
                                 size_t from, size_t to,
                                 int32_t *ids, size_t cap, size_t *n_out)
{
    for (size_t p = from; p < to; ) {
        size_t e = piece_end(bytes, to, p);
        tok_status rc = encode_piece(t, bytes + p, e - p, ids, cap, n_out);
        if (rc != TOK_OK) return rc;
        p = e;
    }
    return TOK_OK;
}

tok_status tok_encode(const tokenizer *t, const unsigned char *bytes, size_t n_bytes,
                      int32_t *ids, size_t cap, size_t *n_out)
{
    if (!t || (!bytes && n_bytes) || (!ids && cap) || !n_out) return TOK_ERR_ARG;
    *n_out = 0;

    /* Added tokens are matched against the raw bytes first and split the input,
     * exactly as the reference does: it turns a literal "<|endoftext|>" in
     * ordinary text into its own id rather than letting the merges see it. */
    size_t seg = 0, i = 0;
    while (i < n_bytes) {
        int32_t sid = -1;
        size_t mlen = added_match(t, bytes, n_bytes, i, &sid);
        if (mlen == 0) { ++i; continue; }

        tok_status rc = encode_segment(t, bytes, seg, i, ids, cap, n_out);
        if (rc != TOK_OK) return rc;
        if (*n_out >= cap) return TOK_ERR_TRUNCATED;
        ids[(*n_out)++] = sid;
        i += mlen;
        seg = i;
    }
    return encode_segment(t, bytes, seg, n_bytes, ids, cap, n_out);
}

/* ---------------------------------------------------------------- decode --- */

tok_status tok_decode(const tokenizer *t, const int32_t *ids, size_t n_ids,
                      unsigned char *out, size_t cap, size_t *n_out)
{
    if (!t || (!ids && n_ids) || (!out && cap) || !n_out) return TOK_ERR_ARG;
    *n_out = 0;
    for (size_t i = 0; i < n_ids; ++i) {
        if (ids[i] < 0 || (size_t)ids[i] >= t->n_vocab) return TOK_ERR_BAD_ID;
        const char *s = t->id_to_str[ids[i]];
        size_t len = strlen(s);
        for (size_t p = 0; p < len; ) {
            uint32_t cp;
            int adv = utf8_next((const unsigned char *)s + p, len - p, &cp);
            if (adv == 0 || cp >= 512 || t->cp_to_byte[cp] < 0)
                return TOK_ERR_UNKNOWN_SYMBOL;
            if (*n_out >= cap) return TOK_ERR_TRUNCATED;
            out[(*n_out)++] = (unsigned char)t->cp_to_byte[cp];
            p += (size_t)adv;
        }
    }
    return TOK_OK;
}

/* ===========================================================================
 * THE CODEPOINT CLASS TABLE
 *
 * GENERATED, NOT HAND-WRITTEN, AND NOT COPIED FROM A UNICODE DATA FILE. Every
 * codepoint U+0000..U+10FFFF was put through the reference pre-tokenizer during
 * Stage 1 and classified by the piece it landed in: a codepoint that joins a
 * preceding letter is a letter, one that joins a preceding digit is a number,
 * one that joins a preceding punctuation character is "other", and one that
 * joins none of them is whitespace. Codepoints absent from the table are class
 * "other". Surrogates never appear in well-formed UTF-8 and are recorded as
 * "other" too.
 *
 * The table therefore describes the classification the reference actually
 * performs, which is NOT the same as Python 3.12's unicodedata: the two
 * disagree at 5008 codepoints, among them U+001C..U+001F (whitespace to Python,
 * "other" to the reference) and a block of codepoints Python's tables still
 * call unassigned.
 * ===========================================================================*/

static const tok_range TOK_CLASS_RANGES[] = {
    {0x00009,0x0000D,3}, {0x00020,0x00020,3}, {0x00030,0x00039,2}, {0x00041,0x0005A,1},
    {0x00061,0x0007A,1}, {0x00085,0x00085,3}, {0x000A0,0x000A0,3}, {0x000AA,0x000AA,1},
    {0x000B2,0x000B3,2}, {0x000B5,0x000B5,1}, {0x000B9,0x000B9,2}, {0x000BA,0x000BA,1},
    {0x000BC,0x000BE,2}, {0x000C0,0x000D6,1}, {0x000D8,0x000F6,1}, {0x000F8,0x002C1,1},
    {0x002C6,0x002D1,1}, {0x002E0,0x002E4,1}, {0x002EC,0x002EC,1}, {0x002EE,0x002EE,1},
    {0x00370,0x00374,1}, {0x00376,0x00377,1}, {0x0037A,0x0037D,1}, {0x0037F,0x0037F,1},
    {0x00386,0x00386,1}, {0x00388,0x0038A,1}, {0x0038C,0x0038C,1}, {0x0038E,0x003A1,1},
    {0x003A3,0x003F5,1}, {0x003F7,0x00481,1}, {0x0048A,0x0052F,1}, {0x00531,0x00556,1},
    {0x00559,0x00559,1}, {0x00560,0x00588,1}, {0x005D0,0x005EA,1}, {0x005EF,0x005F2,1},
    {0x00620,0x0064A,1}, {0x00660,0x00669,2}, {0x0066E,0x0066F,1}, {0x00671,0x006D3,1},
    {0x006D5,0x006D5,1}, {0x006E5,0x006E6,1}, {0x006EE,0x006EF,1}, {0x006F0,0x006F9,2},
    {0x006FA,0x006FC,1}, {0x006FF,0x006FF,1}, {0x00710,0x00710,1}, {0x00712,0x0072F,1},
    {0x0074D,0x007A5,1}, {0x007B1,0x007B1,1}, {0x007C0,0x007C9,2}, {0x007CA,0x007EA,1},
    {0x007F4,0x007F5,1}, {0x007FA,0x007FA,1}, {0x00800,0x00815,1}, {0x0081A,0x0081A,1},
    {0x00824,0x00824,1}, {0x00828,0x00828,1}, {0x00840,0x00858,1}, {0x00860,0x0086A,1},
    {0x00870,0x00887,1}, {0x00889,0x0088E,1}, {0x008A0,0x008C9,1}, {0x00904,0x00939,1},
    {0x0093D,0x0093D,1}, {0x00950,0x00950,1}, {0x00958,0x00961,1}, {0x00966,0x0096F,2},
    {0x00971,0x00980,1}, {0x00985,0x0098C,1}, {0x0098F,0x00990,1}, {0x00993,0x009A8,1},
    {0x009AA,0x009B0,1}, {0x009B2,0x009B2,1}, {0x009B6,0x009B9,1}, {0x009BD,0x009BD,1},
    {0x009CE,0x009CE,1}, {0x009DC,0x009DD,1}, {0x009DF,0x009E1,1}, {0x009E6,0x009EF,2},
    {0x009F0,0x009F1,1}, {0x009F4,0x009F9,2}, {0x009FC,0x009FC,1}, {0x00A05,0x00A0A,1},
    {0x00A0F,0x00A10,1}, {0x00A13,0x00A28,1}, {0x00A2A,0x00A30,1}, {0x00A32,0x00A33,1},
    {0x00A35,0x00A36,1}, {0x00A38,0x00A39,1}, {0x00A59,0x00A5C,1}, {0x00A5E,0x00A5E,1},
    {0x00A66,0x00A6F,2}, {0x00A72,0x00A74,1}, {0x00A85,0x00A8D,1}, {0x00A8F,0x00A91,1},
    {0x00A93,0x00AA8,1}, {0x00AAA,0x00AB0,1}, {0x00AB2,0x00AB3,1}, {0x00AB5,0x00AB9,1},
    {0x00ABD,0x00ABD,1}, {0x00AD0,0x00AD0,1}, {0x00AE0,0x00AE1,1}, {0x00AE6,0x00AEF,2},
    {0x00AF9,0x00AF9,1}, {0x00B05,0x00B0C,1}, {0x00B0F,0x00B10,1}, {0x00B13,0x00B28,1},
    {0x00B2A,0x00B30,1}, {0x00B32,0x00B33,1}, {0x00B35,0x00B39,1}, {0x00B3D,0x00B3D,1},
    {0x00B5C,0x00B5D,1}, {0x00B5F,0x00B61,1}, {0x00B66,0x00B6F,2}, {0x00B71,0x00B71,1},
    {0x00B72,0x00B77,2}, {0x00B83,0x00B83,1}, {0x00B85,0x00B8A,1}, {0x00B8E,0x00B90,1},
    {0x00B92,0x00B95,1}, {0x00B99,0x00B9A,1}, {0x00B9C,0x00B9C,1}, {0x00B9E,0x00B9F,1},
    {0x00BA3,0x00BA4,1}, {0x00BA8,0x00BAA,1}, {0x00BAE,0x00BB9,1}, {0x00BD0,0x00BD0,1},
    {0x00BE6,0x00BF2,2}, {0x00C05,0x00C0C,1}, {0x00C0E,0x00C10,1}, {0x00C12,0x00C28,1},
    {0x00C2A,0x00C39,1}, {0x00C3D,0x00C3D,1}, {0x00C58,0x00C5A,1}, {0x00C5D,0x00C5D,1},
    {0x00C60,0x00C61,1}, {0x00C66,0x00C6F,2}, {0x00C78,0x00C7E,2}, {0x00C80,0x00C80,1},
    {0x00C85,0x00C8C,1}, {0x00C8E,0x00C90,1}, {0x00C92,0x00CA8,1}, {0x00CAA,0x00CB3,1},
    {0x00CB5,0x00CB9,1}, {0x00CBD,0x00CBD,1}, {0x00CDD,0x00CDE,1}, {0x00CE0,0x00CE1,1},
    {0x00CE6,0x00CEF,2}, {0x00CF1,0x00CF2,1}, {0x00D04,0x00D0C,1}, {0x00D0E,0x00D10,1},
    {0x00D12,0x00D3A,1}, {0x00D3D,0x00D3D,1}, {0x00D4E,0x00D4E,1}, {0x00D54,0x00D56,1},
    {0x00D58,0x00D5E,2}, {0x00D5F,0x00D61,1}, {0x00D66,0x00D78,2}, {0x00D7A,0x00D7F,1},
    {0x00D85,0x00D96,1}, {0x00D9A,0x00DB1,1}, {0x00DB3,0x00DBB,1}, {0x00DBD,0x00DBD,1},
    {0x00DC0,0x00DC6,1}, {0x00DE6,0x00DEF,2}, {0x00E01,0x00E30,1}, {0x00E32,0x00E33,1},
    {0x00E40,0x00E46,1}, {0x00E50,0x00E59,2}, {0x00E81,0x00E82,1}, {0x00E84,0x00E84,1},
    {0x00E86,0x00E8A,1}, {0x00E8C,0x00EA3,1}, {0x00EA5,0x00EA5,1}, {0x00EA7,0x00EB0,1},
    {0x00EB2,0x00EB3,1}, {0x00EBD,0x00EBD,1}, {0x00EC0,0x00EC4,1}, {0x00EC6,0x00EC6,1},
    {0x00ED0,0x00ED9,2}, {0x00EDC,0x00EDF,1}, {0x00F00,0x00F00,1}, {0x00F20,0x00F33,2},
    {0x00F40,0x00F47,1}, {0x00F49,0x00F6C,1}, {0x00F88,0x00F8C,1}, {0x01000,0x0102A,1},
    {0x0103F,0x0103F,1}, {0x01040,0x01049,2}, {0x01050,0x01055,1}, {0x0105A,0x0105D,1},
    {0x01061,0x01061,1}, {0x01065,0x01066,1}, {0x0106E,0x01070,1}, {0x01075,0x01081,1},
    {0x0108E,0x0108E,1}, {0x01090,0x01099,2}, {0x010A0,0x010C5,1}, {0x010C7,0x010C7,1},
    {0x010CD,0x010CD,1}, {0x010D0,0x010FA,1}, {0x010FC,0x01248,1}, {0x0124A,0x0124D,1},
    {0x01250,0x01256,1}, {0x01258,0x01258,1}, {0x0125A,0x0125D,1}, {0x01260,0x01288,1},
    {0x0128A,0x0128D,1}, {0x01290,0x012B0,1}, {0x012B2,0x012B5,1}, {0x012B8,0x012BE,1},
    {0x012C0,0x012C0,1}, {0x012C2,0x012C5,1}, {0x012C8,0x012D6,1}, {0x012D8,0x01310,1},
    {0x01312,0x01315,1}, {0x01318,0x0135A,1}, {0x01369,0x0137C,2}, {0x01380,0x0138F,1},
    {0x013A0,0x013F5,1}, {0x013F8,0x013FD,1}, {0x01401,0x0166C,1}, {0x0166F,0x0167F,1},
    {0x01680,0x01680,3}, {0x01681,0x0169A,1}, {0x016A0,0x016EA,1}, {0x016EE,0x016F0,2},
    {0x016F1,0x016F8,1}, {0x01700,0x01711,1}, {0x0171F,0x01731,1}, {0x01740,0x01751,1},
    {0x01760,0x0176C,1}, {0x0176E,0x01770,1}, {0x01780,0x017B3,1}, {0x017D7,0x017D7,1},
    {0x017DC,0x017DC,1}, {0x017E0,0x017E9,2}, {0x017F0,0x017F9,2}, {0x01810,0x01819,2},
    {0x01820,0x01878,1}, {0x01880,0x01884,1}, {0x01887,0x018A8,1}, {0x018AA,0x018AA,1},
    {0x018B0,0x018F5,1}, {0x01900,0x0191E,1}, {0x01946,0x0194F,2}, {0x01950,0x0196D,1},
    {0x01970,0x01974,1}, {0x01980,0x019AB,1}, {0x019B0,0x019C9,1}, {0x019D0,0x019DA,2},
    {0x01A00,0x01A16,1}, {0x01A20,0x01A54,1}, {0x01A80,0x01A89,2}, {0x01A90,0x01A99,2},
    {0x01AA7,0x01AA7,1}, {0x01B05,0x01B33,1}, {0x01B45,0x01B4C,1}, {0x01B50,0x01B59,2},
    {0x01B83,0x01BA0,1}, {0x01BAE,0x01BAF,1}, {0x01BB0,0x01BB9,2}, {0x01BBA,0x01BE5,1},
    {0x01C00,0x01C23,1}, {0x01C40,0x01C49,2}, {0x01C4D,0x01C4F,1}, {0x01C50,0x01C59,2},
    {0x01C5A,0x01C7D,1}, {0x01C80,0x01C8A,1}, {0x01C90,0x01CBA,1}, {0x01CBD,0x01CBF,1},
    {0x01CE9,0x01CEC,1}, {0x01CEE,0x01CF3,1}, {0x01CF5,0x01CF6,1}, {0x01CFA,0x01CFA,1},
    {0x01D00,0x01DBF,1}, {0x01E00,0x01F15,1}, {0x01F18,0x01F1D,1}, {0x01F20,0x01F45,1},
    {0x01F48,0x01F4D,1}, {0x01F50,0x01F57,1}, {0x01F59,0x01F59,1}, {0x01F5B,0x01F5B,1},
    {0x01F5D,0x01F5D,1}, {0x01F5F,0x01F7D,1}, {0x01F80,0x01FB4,1}, {0x01FB6,0x01FBC,1},
    {0x01FBE,0x01FBE,1}, {0x01FC2,0x01FC4,1}, {0x01FC6,0x01FCC,1}, {0x01FD0,0x01FD3,1},
    {0x01FD6,0x01FDB,1}, {0x01FE0,0x01FEC,1}, {0x01FF2,0x01FF4,1}, {0x01FF6,0x01FFC,1},
    {0x02000,0x0200A,3}, {0x02028,0x02029,3}, {0x0202F,0x0202F,3}, {0x0205F,0x0205F,3},
    {0x02070,0x02070,2}, {0x02071,0x02071,1}, {0x02074,0x02079,2}, {0x0207F,0x0207F,1},
    {0x02080,0x02089,2}, {0x02090,0x0209C,1}, {0x02102,0x02102,1}, {0x02107,0x02107,1},
    {0x0210A,0x02113,1}, {0x02115,0x02115,1}, {0x02119,0x0211D,1}, {0x02124,0x02124,1},
    {0x02126,0x02126,1}, {0x02128,0x02128,1}, {0x0212A,0x0212D,1}, {0x0212F,0x02139,1},
    {0x0213C,0x0213F,1}, {0x02145,0x02149,1}, {0x0214E,0x0214E,1}, {0x02150,0x02182,2},
    {0x02183,0x02184,1}, {0x02185,0x02189,2}, {0x02460,0x0249B,2}, {0x024EA,0x024FF,2},
    {0x02776,0x02793,2}, {0x02C00,0x02CE4,1}, {0x02CEB,0x02CEE,1}, {0x02CF2,0x02CF3,1},
    {0x02CFD,0x02CFD,2}, {0x02D00,0x02D25,1}, {0x02D27,0x02D27,1}, {0x02D2D,0x02D2D,1},
    {0x02D30,0x02D67,1}, {0x02D6F,0x02D6F,1}, {0x02D80,0x02D96,1}, {0x02DA0,0x02DA6,1},
    {0x02DA8,0x02DAE,1}, {0x02DB0,0x02DB6,1}, {0x02DB8,0x02DBE,1}, {0x02DC0,0x02DC6,1},
    {0x02DC8,0x02DCE,1}, {0x02DD0,0x02DD6,1}, {0x02DD8,0x02DDE,1}, {0x02E2F,0x02E2F,1},
    {0x03000,0x03000,3}, {0x03005,0x03006,1}, {0x03007,0x03007,2}, {0x03021,0x03029,2},
    {0x03031,0x03035,1}, {0x03038,0x0303A,2}, {0x0303B,0x0303C,1}, {0x03041,0x03096,1},
    {0x0309D,0x0309F,1}, {0x030A1,0x030FA,1}, {0x030FC,0x030FF,1}, {0x03105,0x0312F,1},
    {0x03131,0x0318E,1}, {0x03192,0x03195,2}, {0x031A0,0x031BF,1}, {0x031F0,0x031FF,1},
    {0x03220,0x03229,2}, {0x03248,0x0324F,2}, {0x03251,0x0325F,2}, {0x03280,0x03289,2},
    {0x032B1,0x032BF,2}, {0x03400,0x04DBF,1}, {0x04E00,0x0A48C,1}, {0x0A4D0,0x0A4FD,1},
    {0x0A500,0x0A60C,1}, {0x0A610,0x0A61F,1}, {0x0A620,0x0A629,2}, {0x0A62A,0x0A62B,1},
    {0x0A640,0x0A66E,1}, {0x0A67F,0x0A69D,1}, {0x0A6A0,0x0A6E5,1}, {0x0A6E6,0x0A6EF,2},
    {0x0A717,0x0A71F,1}, {0x0A722,0x0A788,1}, {0x0A78B,0x0A7CD,1}, {0x0A7D0,0x0A7D1,1},
    {0x0A7D3,0x0A7D3,1}, {0x0A7D5,0x0A7DC,1}, {0x0A7F2,0x0A801,1}, {0x0A803,0x0A805,1},
    {0x0A807,0x0A80A,1}, {0x0A80C,0x0A822,1}, {0x0A830,0x0A835,2}, {0x0A840,0x0A873,1},
    {0x0A882,0x0A8B3,1}, {0x0A8D0,0x0A8D9,2}, {0x0A8F2,0x0A8F7,1}, {0x0A8FB,0x0A8FB,1},
    {0x0A8FD,0x0A8FE,1}, {0x0A900,0x0A909,2}, {0x0A90A,0x0A925,1}, {0x0A930,0x0A946,1},
    {0x0A960,0x0A97C,1}, {0x0A984,0x0A9B2,1}, {0x0A9CF,0x0A9CF,1}, {0x0A9D0,0x0A9D9,2},
    {0x0A9E0,0x0A9E4,1}, {0x0A9E6,0x0A9EF,1}, {0x0A9F0,0x0A9F9,2}, {0x0A9FA,0x0A9FE,1},
    {0x0AA00,0x0AA28,1}, {0x0AA40,0x0AA42,1}, {0x0AA44,0x0AA4B,1}, {0x0AA50,0x0AA59,2},
    {0x0AA60,0x0AA76,1}, {0x0AA7A,0x0AA7A,1}, {0x0AA7E,0x0AAAF,1}, {0x0AAB1,0x0AAB1,1},
    {0x0AAB5,0x0AAB6,1}, {0x0AAB9,0x0AABD,1}, {0x0AAC0,0x0AAC0,1}, {0x0AAC2,0x0AAC2,1},
    {0x0AADB,0x0AADD,1}, {0x0AAE0,0x0AAEA,1}, {0x0AAF2,0x0AAF4,1}, {0x0AB01,0x0AB06,1},
    {0x0AB09,0x0AB0E,1}, {0x0AB11,0x0AB16,1}, {0x0AB20,0x0AB26,1}, {0x0AB28,0x0AB2E,1},
    {0x0AB30,0x0AB5A,1}, {0x0AB5C,0x0AB69,1}, {0x0AB70,0x0ABE2,1}, {0x0ABF0,0x0ABF9,2},
    {0x0AC00,0x0D7A3,1}, {0x0D7B0,0x0D7C6,1}, {0x0D7CB,0x0D7FB,1}, {0x0F900,0x0FA6D,1},
    {0x0FA70,0x0FAD9,1}, {0x0FB00,0x0FB06,1}, {0x0FB13,0x0FB17,1}, {0x0FB1D,0x0FB1D,1},
    {0x0FB1F,0x0FB28,1}, {0x0FB2A,0x0FB36,1}, {0x0FB38,0x0FB3C,1}, {0x0FB3E,0x0FB3E,1},
    {0x0FB40,0x0FB41,1}, {0x0FB43,0x0FB44,1}, {0x0FB46,0x0FBB1,1}, {0x0FBD3,0x0FD3D,1},
    {0x0FD50,0x0FD8F,1}, {0x0FD92,0x0FDC7,1}, {0x0FDF0,0x0FDFB,1}, {0x0FE70,0x0FE74,1},
    {0x0FE76,0x0FEFC,1}, {0x0FF10,0x0FF19,2}, {0x0FF21,0x0FF3A,1}, {0x0FF41,0x0FF5A,1},
    {0x0FF66,0x0FFBE,1}, {0x0FFC2,0x0FFC7,1}, {0x0FFCA,0x0FFCF,1}, {0x0FFD2,0x0FFD7,1},
    {0x0FFDA,0x0FFDC,1}, {0x10000,0x1000B,1}, {0x1000D,0x10026,1}, {0x10028,0x1003A,1},
    {0x1003C,0x1003D,1}, {0x1003F,0x1004D,1}, {0x10050,0x1005D,1}, {0x10080,0x100FA,1},
    {0x10107,0x10133,2}, {0x10140,0x10178,2}, {0x1018A,0x1018B,2}, {0x10280,0x1029C,1},
    {0x102A0,0x102D0,1}, {0x102E1,0x102FB,2}, {0x10300,0x1031F,1}, {0x10320,0x10323,2},
    {0x1032D,0x10340,1}, {0x10341,0x10341,2}, {0x10342,0x10349,1}, {0x1034A,0x1034A,2},
    {0x10350,0x10375,1}, {0x10380,0x1039D,1}, {0x103A0,0x103C3,1}, {0x103C8,0x103CF,1},
    {0x103D1,0x103D5,2}, {0x10400,0x1049D,1}, {0x104A0,0x104A9,2}, {0x104B0,0x104D3,1},
    {0x104D8,0x104FB,1}, {0x10500,0x10527,1}, {0x10530,0x10563,1}, {0x10570,0x1057A,1},
    {0x1057C,0x1058A,1}, {0x1058C,0x10592,1}, {0x10594,0x10595,1}, {0x10597,0x105A1,1},
    {0x105A3,0x105B1,1}, {0x105B3,0x105B9,1}, {0x105BB,0x105BC,1}, {0x105C0,0x105F3,1},
    {0x10600,0x10736,1}, {0x10740,0x10755,1}, {0x10760,0x10767,1}, {0x10780,0x10785,1},
    {0x10787,0x107B0,1}, {0x107B2,0x107BA,1}, {0x10800,0x10805,1}, {0x10808,0x10808,1},
    {0x1080A,0x10835,1}, {0x10837,0x10838,1}, {0x1083C,0x1083C,1}, {0x1083F,0x10855,1},
    {0x10858,0x1085F,2}, {0x10860,0x10876,1}, {0x10879,0x1087F,2}, {0x10880,0x1089E,1},
    {0x108A7,0x108AF,2}, {0x108E0,0x108F2,1}, {0x108F4,0x108F5,1}, {0x108FB,0x108FF,2},
    {0x10900,0x10915,1}, {0x10916,0x1091B,2}, {0x10920,0x10939,1}, {0x10980,0x109B7,1},
    {0x109BC,0x109BD,2}, {0x109BE,0x109BF,1}, {0x109C0,0x109CF,2}, {0x109D2,0x109FF,2},
    {0x10A00,0x10A00,1}, {0x10A10,0x10A13,1}, {0x10A15,0x10A17,1}, {0x10A19,0x10A35,1},
    {0x10A40,0x10A48,2}, {0x10A60,0x10A7C,1}, {0x10A7D,0x10A7E,2}, {0x10A80,0x10A9C,1},
    {0x10A9D,0x10A9F,2}, {0x10AC0,0x10AC7,1}, {0x10AC9,0x10AE4,1}, {0x10AEB,0x10AEF,2},
    {0x10B00,0x10B35,1}, {0x10B40,0x10B55,1}, {0x10B58,0x10B5F,2}, {0x10B60,0x10B72,1},
    {0x10B78,0x10B7F,2}, {0x10B80,0x10B91,1}, {0x10BA9,0x10BAF,2}, {0x10C00,0x10C48,1},
    {0x10C80,0x10CB2,1}, {0x10CC0,0x10CF2,1}, {0x10CFA,0x10CFF,2}, {0x10D00,0x10D23,1},
    {0x10D30,0x10D39,2}, {0x10D40,0x10D49,2}, {0x10D4A,0x10D65,1}, {0x10D6F,0x10D85,1},
    {0x10E60,0x10E7E,2}, {0x10E80,0x10EA9,1}, {0x10EB0,0x10EB1,1}, {0x10EC2,0x10EC4,1},
    {0x10F00,0x10F1C,1}, {0x10F1D,0x10F26,2}, {0x10F27,0x10F27,1}, {0x10F30,0x10F45,1},
    {0x10F51,0x10F54,2}, {0x10F70,0x10F81,1}, {0x10FB0,0x10FC4,1}, {0x10FC5,0x10FCB,2},
    {0x10FE0,0x10FF6,1}, {0x11003,0x11037,1}, {0x11052,0x1106F,2}, {0x11071,0x11072,1},
    {0x11075,0x11075,1}, {0x11083,0x110AF,1}, {0x110D0,0x110E8,1}, {0x110F0,0x110F9,2},
    {0x11103,0x11126,1}, {0x11136,0x1113F,2}, {0x11144,0x11144,1}, {0x11147,0x11147,1},
    {0x11150,0x11172,1}, {0x11176,0x11176,1}, {0x11183,0x111B2,1}, {0x111C1,0x111C4,1},
    {0x111D0,0x111D9,2}, {0x111DA,0x111DA,1}, {0x111DC,0x111DC,1}, {0x111E1,0x111F4,2},
    {0x11200,0x11211,1}, {0x11213,0x1122B,1}, {0x1123F,0x11240,1}, {0x11280,0x11286,1},
    {0x11288,0x11288,1}, {0x1128A,0x1128D,1}, {0x1128F,0x1129D,1}, {0x1129F,0x112A8,1},
    {0x112B0,0x112DE,1}, {0x112F0,0x112F9,2}, {0x11305,0x1130C,1}, {0x1130F,0x11310,1},
    {0x11313,0x11328,1}, {0x1132A,0x11330,1}, {0x11332,0x11333,1}, {0x11335,0x11339,1},
    {0x1133D,0x1133D,1}, {0x11350,0x11350,1}, {0x1135D,0x11361,1}, {0x11380,0x11389,1},
    {0x1138B,0x1138B,1}, {0x1138E,0x1138E,1}, {0x11390,0x113B5,1}, {0x113B7,0x113B7,1},
    {0x113D1,0x113D1,1}, {0x113D3,0x113D3,1}, {0x11400,0x11434,1}, {0x11447,0x1144A,1},
    {0x11450,0x11459,2}, {0x1145F,0x11461,1}, {0x11480,0x114AF,1}, {0x114C4,0x114C5,1},
    {0x114C7,0x114C7,1}, {0x114D0,0x114D9,2}, {0x11580,0x115AE,1}, {0x115D8,0x115DB,1},
    {0x11600,0x1162F,1}, {0x11644,0x11644,1}, {0x11650,0x11659,2}, {0x11680,0x116AA,1},
    {0x116B8,0x116B8,1}, {0x116C0,0x116C9,2}, {0x116D0,0x116E3,2}, {0x11700,0x1171A,1},
    {0x11730,0x1173B,2}, {0x11740,0x11746,1}, {0x11800,0x1182B,1}, {0x118A0,0x118DF,1},
    {0x118E0,0x118F2,2}, {0x118FF,0x11906,1}, {0x11909,0x11909,1}, {0x1190C,0x11913,1},
    {0x11915,0x11916,1}, {0x11918,0x1192F,1}, {0x1193F,0x1193F,1}, {0x11941,0x11941,1},
    {0x11950,0x11959,2}, {0x119A0,0x119A7,1}, {0x119AA,0x119D0,1}, {0x119E1,0x119E1,1},
    {0x119E3,0x119E3,1}, {0x11A00,0x11A00,1}, {0x11A0B,0x11A32,1}, {0x11A3A,0x11A3A,1},
    {0x11A50,0x11A50,1}, {0x11A5C,0x11A89,1}, {0x11A9D,0x11A9D,1}, {0x11AB0,0x11AF8,1},
    {0x11BC0,0x11BE0,1}, {0x11BF0,0x11BF9,2}, {0x11C00,0x11C08,1}, {0x11C0A,0x11C2E,1},
    {0x11C40,0x11C40,1}, {0x11C50,0x11C6C,2}, {0x11C72,0x11C8F,1}, {0x11D00,0x11D06,1},
    {0x11D08,0x11D09,1}, {0x11D0B,0x11D30,1}, {0x11D46,0x11D46,1}, {0x11D50,0x11D59,2},
    {0x11D60,0x11D65,1}, {0x11D67,0x11D68,1}, {0x11D6A,0x11D89,1}, {0x11D98,0x11D98,1},
    {0x11DA0,0x11DA9,2}, {0x11EE0,0x11EF2,1}, {0x11F02,0x11F02,1}, {0x11F04,0x11F10,1},
    {0x11F12,0x11F33,1}, {0x11F50,0x11F59,2}, {0x11FB0,0x11FB0,1}, {0x11FC0,0x11FD4,2},
    {0x12000,0x12399,1}, {0x12400,0x1246E,2}, {0x12480,0x12543,1}, {0x12F90,0x12FF0,1},
    {0x13000,0x1342F,1}, {0x13441,0x13446,1}, {0x13460,0x143FA,1}, {0x14400,0x14646,1},
    {0x16100,0x1611D,1}, {0x16130,0x16139,2}, {0x16800,0x16A38,1}, {0x16A40,0x16A5E,1},
    {0x16A60,0x16A69,2}, {0x16A70,0x16ABE,1}, {0x16AC0,0x16AC9,2}, {0x16AD0,0x16AED,1},
    {0x16B00,0x16B2F,1}, {0x16B40,0x16B43,1}, {0x16B50,0x16B59,2}, {0x16B5B,0x16B61,2},
    {0x16B63,0x16B77,1}, {0x16B7D,0x16B8F,1}, {0x16D40,0x16D6C,1}, {0x16D70,0x16D79,2},
    {0x16E40,0x16E7F,1}, {0x16E80,0x16E96,2}, {0x16F00,0x16F4A,1}, {0x16F50,0x16F50,1},
    {0x16F93,0x16F9F,1}, {0x16FE0,0x16FE1,1}, {0x16FE3,0x16FE3,1}, {0x17000,0x187F7,1},
    {0x18800,0x18CD5,1}, {0x18CFF,0x18D08,1}, {0x1AFF0,0x1AFF3,1}, {0x1AFF5,0x1AFFB,1},
    {0x1AFFD,0x1AFFE,1}, {0x1B000,0x1B122,1}, {0x1B132,0x1B132,1}, {0x1B150,0x1B152,1},
    {0x1B155,0x1B155,1}, {0x1B164,0x1B167,1}, {0x1B170,0x1B2FB,1}, {0x1BC00,0x1BC6A,1},
    {0x1BC70,0x1BC7C,1}, {0x1BC80,0x1BC88,1}, {0x1BC90,0x1BC99,1}, {0x1CCF0,0x1CCF9,2},
    {0x1D2C0,0x1D2D3,2}, {0x1D2E0,0x1D2F3,2}, {0x1D360,0x1D378,2}, {0x1D400,0x1D454,1},
    {0x1D456,0x1D49C,1}, {0x1D49E,0x1D49F,1}, {0x1D4A2,0x1D4A2,1}, {0x1D4A5,0x1D4A6,1},
    {0x1D4A9,0x1D4AC,1}, {0x1D4AE,0x1D4B9,1}, {0x1D4BB,0x1D4BB,1}, {0x1D4BD,0x1D4C3,1},
    {0x1D4C5,0x1D505,1}, {0x1D507,0x1D50A,1}, {0x1D50D,0x1D514,1}, {0x1D516,0x1D51C,1},
    {0x1D51E,0x1D539,1}, {0x1D53B,0x1D53E,1}, {0x1D540,0x1D544,1}, {0x1D546,0x1D546,1},
    {0x1D54A,0x1D550,1}, {0x1D552,0x1D6A5,1}, {0x1D6A8,0x1D6C0,1}, {0x1D6C2,0x1D6DA,1},
    {0x1D6DC,0x1D6FA,1}, {0x1D6FC,0x1D714,1}, {0x1D716,0x1D734,1}, {0x1D736,0x1D74E,1},
    {0x1D750,0x1D76E,1}, {0x1D770,0x1D788,1}, {0x1D78A,0x1D7A8,1}, {0x1D7AA,0x1D7C2,1},
    {0x1D7C4,0x1D7CB,1}, {0x1D7CE,0x1D7FF,2}, {0x1DF00,0x1DF1E,1}, {0x1DF25,0x1DF2A,1},
    {0x1E030,0x1E06D,1}, {0x1E100,0x1E12C,1}, {0x1E137,0x1E13D,1}, {0x1E140,0x1E149,2},
    {0x1E14E,0x1E14E,1}, {0x1E290,0x1E2AD,1}, {0x1E2C0,0x1E2EB,1}, {0x1E2F0,0x1E2F9,2},
    {0x1E4D0,0x1E4EB,1}, {0x1E4F0,0x1E4F9,2}, {0x1E5D0,0x1E5ED,1}, {0x1E5F0,0x1E5F0,1},
    {0x1E5F1,0x1E5FA,2}, {0x1E7E0,0x1E7E6,1}, {0x1E7E8,0x1E7EB,1}, {0x1E7ED,0x1E7EE,1},
    {0x1E7F0,0x1E7FE,1}, {0x1E800,0x1E8C4,1}, {0x1E8C7,0x1E8CF,2}, {0x1E900,0x1E943,1},
    {0x1E94B,0x1E94B,1}, {0x1E950,0x1E959,2}, {0x1EC71,0x1ECAB,2}, {0x1ECAD,0x1ECAF,2},
    {0x1ECB1,0x1ECB4,2}, {0x1ED01,0x1ED2D,2}, {0x1ED2F,0x1ED3D,2}, {0x1EE00,0x1EE03,1},
    {0x1EE05,0x1EE1F,1}, {0x1EE21,0x1EE22,1}, {0x1EE24,0x1EE24,1}, {0x1EE27,0x1EE27,1},
    {0x1EE29,0x1EE32,1}, {0x1EE34,0x1EE37,1}, {0x1EE39,0x1EE39,1}, {0x1EE3B,0x1EE3B,1},
    {0x1EE42,0x1EE42,1}, {0x1EE47,0x1EE47,1}, {0x1EE49,0x1EE49,1}, {0x1EE4B,0x1EE4B,1},
    {0x1EE4D,0x1EE4F,1}, {0x1EE51,0x1EE52,1}, {0x1EE54,0x1EE54,1}, {0x1EE57,0x1EE57,1},
    {0x1EE59,0x1EE59,1}, {0x1EE5B,0x1EE5B,1}, {0x1EE5D,0x1EE5D,1}, {0x1EE5F,0x1EE5F,1},
    {0x1EE61,0x1EE62,1}, {0x1EE64,0x1EE64,1}, {0x1EE67,0x1EE6A,1}, {0x1EE6C,0x1EE72,1},
    {0x1EE74,0x1EE77,1}, {0x1EE79,0x1EE7C,1}, {0x1EE7E,0x1EE7E,1}, {0x1EE80,0x1EE89,1},
    {0x1EE8B,0x1EE9B,1}, {0x1EEA1,0x1EEA3,1}, {0x1EEA5,0x1EEA9,1}, {0x1EEAB,0x1EEBB,1},
    {0x1F100,0x1F10C,2}, {0x1FBF0,0x1FBF9,2}, {0x20000,0x2A6DF,1}, {0x2A700,0x2B739,1},
    {0x2B740,0x2B81D,1}, {0x2B820,0x2CEA1,1}, {0x2CEB0,0x2EBE0,1}, {0x2EBF0,0x2EE5D,1},
    {0x2F800,0x2FA1D,1}, {0x30000,0x3134A,1}, {0x31350,0x323AF,1},
};
static const size_t TOK_N_CLASS_RANGES =
    sizeof TOK_CLASS_RANGES / sizeof TOK_CLASS_RANGES[0];

static int cp_class(uint32_t cp)
{
    size_t lo = 0, hi = TOK_N_CLASS_RANGES;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < TOK_CLASS_RANGES[mid].lo)      hi = mid;
        else if (cp > TOK_CLASS_RANGES[mid].hi) lo = mid + 1;
        else return (int)TOK_CLASS_RANGES[mid].cls;
    }
    return TC_OTHER;
}
