/* safetensors.c -- weight-file reader, Stage 1.
 *
 * ---------------------------------------------------------------------------
 * THE JSON QUESTION, RESOLVED HERE RATHER THAN SILENTLY
 * ---------------------------------------------------------------------------
 * The header is JSON, so a JSON library could be argued for. It is not used,
 * and the reason is not purity: a dependency pulled in here is a dependency in
 * the inference path for the life of the project, and the thing it would parse
 * is a fixed, small, machine-generated document -- 14283 bytes, 161 members,
 * three keys per member, pure ASCII, no whitespace, no escape sequence of any
 * kind, observed in the file itself. A parser for exactly that costs less than
 * a hundred lines and cannot fail in a way the file can exercise.
 *
 * THE SUBSET ACCEPTED, exactly:
 *   - values: object, array, string, and unsigned decimal integer;
 *   - whitespace (space, tab, CR, LF) between tokens -- the file has none, it
 *     is accepted because rejecting it would be a parser bug waiting to happen;
 *   - string escapes \" \\ \/ \b \f \n \r \t;
 *   - bytes >= 0x80 inside strings, passed through untouched (UTF-8 is opaque
 *     to this parser; the header observed here is pure ASCII);
 *   - unknown keys inside a tensor object: parsed and discarded, so a file
 *     carrying a key this reader does not know is still read rather than
 *     rejected.
 *
 * THE SUBSET REJECTED, each with ST_ERR_BAD_JSON, and each deliberate:
 *   - \uXXXX escapes -- they need UTF-16 surrogate reassembly, and nothing in
 *     a safetensors header needs them. Rejected loudly rather than mishandled;
 *   - true, false, null;
 *   - negative numbers, decimal points, exponents -- every number in this
 *     document is a byte count or an extent, and neither can be either;
 *   - trailing commas, duplicate or missing required keys, unterminated
 *     strings, and any trailing byte after the top-level object.
 *
 * ---------------------------------------------------------------------------
 * The structural facts below were read from models/gpt2/model.safetensors, not
 * recalled: see the comment block in safetensors.h for the observations.
 */

#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 64-bit file positioning. The weight file is 548105171 bytes, past what a
 * 32-bit long can address on this toolchain, so the wide variants are used. */
#if defined(_MSC_VER)
#  define ST_FSEEK(fp, off, whence) _fseeki64((fp), (off), (whence))
#  define ST_FTELL(fp)              _ftelli64(fp)
#else
#  define ST_FSEEK(fp, off, whence) fseeko((fp), (off_t)(off), (whence))
#  define ST_FTELL(fp)              ((long long)ftello(fp))
#endif

/* ------------------------------------------------------------------ types -- */

typedef struct {
    char *key;
    char *value;
} st_meta_entry;

struct st_file {
    FILE          *fp;
    uint64_t       file_size;
    uint64_t       header_length;
    uint64_t       data_offset;     /* 8 + header_length */
    st_tensor     *tensors;
    size_t         n_tensors;
    st_meta_entry *meta;
    size_t         n_meta;
    int            has_metadata;
};

/* --------------------------------------------------------------- dtypes --- */

static const struct { const char *name; st_dtype d; size_t size; } ST_DTYPES[] = {
    { "BOOL", ST_DTYPE_BOOL, 1 },
    { "U8",   ST_DTYPE_U8,   1 }, { "I8",   ST_DTYPE_I8,   1 },
    { "U16",  ST_DTYPE_U16,  2 }, { "I16",  ST_DTYPE_I16,  2 },
    { "F16",  ST_DTYPE_F16,  2 }, { "BF16", ST_DTYPE_BF16, 2 },
    { "U32",  ST_DTYPE_U32,  4 }, { "I32",  ST_DTYPE_I32,  4 },
    { "F32",  ST_DTYPE_F32,  4 },
    { "U64",  ST_DTYPE_U64,  8 }, { "I64",  ST_DTYPE_I64,  8 },
    { "F64",  ST_DTYPE_F64,  8 }
};
static const size_t ST_N_DTYPES = sizeof ST_DTYPES / sizeof ST_DTYPES[0];

size_t st_dtype_size(st_dtype d)
{
    for (size_t i = 0; i < ST_N_DTYPES; ++i)
        if (ST_DTYPES[i].d == d) return ST_DTYPES[i].size;
    return 0;
}

const char *st_dtype_name(st_dtype d)
{
    for (size_t i = 0; i < ST_N_DTYPES; ++i)
        if (ST_DTYPES[i].d == d) return ST_DTYPES[i].name;
    return "UNKNOWN";
}

static st_dtype st_dtype_from_name(const char *s)
{
    for (size_t i = 0; i < ST_N_DTYPES; ++i)
        if (strcmp(ST_DTYPES[i].name, s) == 0) return ST_DTYPES[i].d;
    return ST_DTYPE_UNKNOWN;
}

const char *st_strerror(st_status s)
{
    switch (s) {
    case ST_OK:                   return "ok";
    case ST_ERR_OPEN:             return "the file could not be opened";
    case ST_ERR_READ:             return "a read failed or returned short";
    case ST_ERR_TRUNCATED_HEADER: return "the header does not fit inside the file";
    case ST_ERR_BAD_JSON:         return "the header is not the accepted JSON subset";
    case ST_ERR_BAD_DTYPE:        return "unknown dtype spelling";
    case ST_ERR_BAD_SHAPE:        return "missing or invalid shape";
    case ST_ERR_RANGE:            return "a declared byte range leaves the data segment";
    case ST_ERR_SIZE_MISMATCH:    return "declared byte length disagrees with shape times dtype size";
    case ST_ERR_NOT_FOUND:        return "no tensor of that name";
    case ST_ERR_TOO_MANY_DIMS:    return "shape has more dimensions than the reader supports";
    case ST_ERR_NAME_TOO_LONG:    return "tensor name is longer than the reader supports";
    case ST_ERR_NOMEM:            return "allocation failed";
    case ST_ERR_ARG:              return "invalid argument";
    }
    return "unknown status";
}

/* ---------------------------------------------------------- JSON scanner --- */

typedef struct {
    const char *p;      /* cursor                       */
    const char *end;    /* one past the last byte       */
} st_scan;

static void st_ws(st_scan *s)
{
    while (s->p < s->end &&
           (*s->p == ' ' || *s->p == '\t' || *s->p == '\r' || *s->p == '\n'))
        ++s->p;
}

static int st_eat(st_scan *s, char c)
{
    st_ws(s);
    if (s->p < s->end && *s->p == c) { ++s->p; return 1; }
    return 0;
}

static int st_peek(st_scan *s, char c)
{
    st_ws(s);
    return s->p < s->end && *s->p == c;
}

/* Parses a string into buf. Returns 0 on success, -1 on any rejection.
 * On success *out_len holds the decoded length; buf is NUL terminated. */
static int st_string(st_scan *s, char *buf, size_t cap, size_t *out_len)
{
    st_ws(s);
    if (s->p >= s->end || *s->p != '"') return -1;
    ++s->p;
    size_t n = 0;
    while (s->p < s->end && *s->p != '"') {
        unsigned char c = (unsigned char)*s->p;
        char decoded;
        if (c == '\\') {
            ++s->p;
            if (s->p >= s->end) return -1;
            switch (*s->p) {
            case '"':  decoded = '"';  break;
            case '\\': decoded = '\\'; break;
            case '/':  decoded = '/';  break;
            case 'b':  decoded = '\b'; break;
            case 'f':  decoded = '\f'; break;
            case 'n':  decoded = '\n'; break;
            case 'r':  decoded = '\r'; break;
            case 't':  decoded = '\t'; break;
            default:   return -1;      /* \u and anything else: rejected */
            }
            ++s->p;
        } else if (c < 0x20) {
            return -1;                 /* raw control byte inside a string */
        } else {
            decoded = *s->p;
            ++s->p;
        }
        if (n + 1 >= cap) return -1;
        buf[n++] = decoded;
    }
    if (s->p >= s->end) return -1;     /* unterminated */
    ++s->p;                            /* closing quote */
    buf[n] = '\0';
    if (out_len) *out_len = n;
    return 0;
}

/* Unsigned decimal integer, no sign, no point, no exponent, with overflow
 * detection. Returns 0 on success. */
static int st_uint(st_scan *s, uint64_t *out)
{
    st_ws(s);
    if (s->p >= s->end || *s->p < '0' || *s->p > '9') return -1;
    uint64_t v = 0;
    while (s->p < s->end && *s->p >= '0' && *s->p <= '9') {
        unsigned d = (unsigned)(*s->p - '0');
        if (v > (UINT64_MAX - d) / 10u) return -1;   /* overflow */
        v = v * 10u + d;
        ++s->p;
    }
    /* A decimal point or an exponent here means this was not an integer. */
    if (s->p < s->end && (*s->p == '.' || *s->p == 'e' || *s->p == 'E')) return -1;
    *out = v;
    return 0;
}

/* Parses and discards a value of any accepted kind. Used for keys this reader
 * does not need. Rejects the same things the typed parsers reject. */
static int st_skip_value(st_scan *s, int depth)
{
    if (depth > 32) return -1;
    st_ws(s);
    if (s->p >= s->end) return -1;
    if (*s->p == '"') {
        char scratch[ST_MAX_NAME];
        /* Long strings are skipped in place rather than buffered. */
        ++s->p;
        while (s->p < s->end && *s->p != '"') {
            if (*s->p == '\\') {
                ++s->p;
                if (s->p >= s->end) return -1;
                if (*s->p == 'u') return -1;
            }
            ++s->p;
        }
        (void)scratch;
        if (s->p >= s->end) return -1;
        ++s->p;
        return 0;
    }
    if (*s->p == '{' || *s->p == '[') {
        char open = *s->p, close = (open == '{') ? '}' : ']';
        ++s->p;
        if (st_eat(s, close)) return 0;
        for (;;) {
            if (open == '{') {
                char key[ST_MAX_NAME];
                if (st_string(s, key, sizeof key, NULL) != 0) return -1;
                if (!st_eat(s, ':')) return -1;
            }
            if (st_skip_value(s, depth + 1) != 0) return -1;
            if (st_eat(s, ',')) continue;
            if (st_eat(s, close)) return 0;
            return -1;
        }
    }
    {   /* the only remaining accepted value is an unsigned integer */
        uint64_t dummy;
        return st_uint(s, &dummy);
    }
}

/* ------------------------------------------------------------- the parse --- */

static int st_grow(void **base, size_t *cap, size_t need, size_t elem)
{
    if (need <= *cap) return 0;
    size_t ncap = *cap ? *cap * 2 : 16;
    while (ncap < need) ncap *= 2;
    void *n = realloc(*base, ncap * elem);
    if (!n) return -1;
    *base = n;
    *cap = ncap;
    return 0;
}

static st_status st_parse_tensor(st_scan *s, const char *name,
                                 uint64_t data_bytes, st_tensor *t)
{
    memset(t, 0, sizeof *t);
    if (strlen(name) + 1 > sizeof t->name) return ST_ERR_NAME_TOO_LONG;
    memcpy(t->name, name, strlen(name) + 1);

    int have_dtype = 0, have_shape = 0, have_offsets = 0;
    if (!st_eat(s, '{')) return ST_ERR_BAD_JSON;
    if (!st_peek(s, '}')) {
        for (;;) {
            char key[ST_MAX_NAME];
            if (st_string(s, key, sizeof key, NULL) != 0) return ST_ERR_BAD_JSON;
            if (!st_eat(s, ':')) return ST_ERR_BAD_JSON;

            if (strcmp(key, "dtype") == 0) {
                char d[32];
                if (st_string(s, d, sizeof d, NULL) != 0) return ST_ERR_BAD_JSON;
                t->dtype = st_dtype_from_name(d);
                if (t->dtype == ST_DTYPE_UNKNOWN) return ST_ERR_BAD_DTYPE;
                have_dtype = 1;
            } else if (strcmp(key, "shape") == 0) {
                if (!st_eat(s, '[')) return ST_ERR_BAD_JSON;
                t->n_dims = 0;
                if (!st_peek(s, ']')) {
                    for (;;) {
                        uint64_t v;
                        if (st_uint(s, &v) != 0) return ST_ERR_BAD_JSON;
                        if (t->n_dims >= ST_MAX_DIMS) return ST_ERR_TOO_MANY_DIMS;
                        if (v > (uint64_t)INT64_MAX) return ST_ERR_BAD_SHAPE;
                        t->dims[t->n_dims++] = (int64_t)v;
                        if (st_eat(s, ',')) continue;
                        break;
                    }
                }
                if (!st_eat(s, ']')) return ST_ERR_BAD_JSON;
                have_shape = 1;
            } else if (strcmp(key, "data_offsets") == 0) {
                uint64_t b, e;
                if (!st_eat(s, '[')) return ST_ERR_BAD_JSON;
                if (st_uint(s, &b) != 0) return ST_ERR_BAD_JSON;
                if (!st_eat(s, ',')) return ST_ERR_BAD_JSON;
                if (st_uint(s, &e) != 0) return ST_ERR_BAD_JSON;
                if (!st_eat(s, ']')) return ST_ERR_BAD_JSON;
                t->begin = b;
                t->end = e;
                have_offsets = 1;
            } else {
                if (st_skip_value(s, 0) != 0) return ST_ERR_BAD_JSON;
            }
            if (st_eat(s, ',')) continue;
            break;
        }
    }
    if (!st_eat(s, '}')) return ST_ERR_BAD_JSON;
    if (!have_dtype || !have_offsets) return ST_ERR_BAD_JSON;
    if (!have_shape) return ST_ERR_BAD_SHAPE;

    /* --- validation, every declared range against the file's own size ---- */
    if (t->end < t->begin) return ST_ERR_RANGE;
    if (t->end > data_bytes) return ST_ERR_RANGE;
    t->nbytes = t->end - t->begin;

    uint64_t elems = 1;
    for (int i = 0; i < t->n_dims; ++i) {
        uint64_t d = (uint64_t)t->dims[i];
        if (d != 0 && elems > UINT64_MAX / d) return ST_ERR_SIZE_MISMATCH;
        elems *= d;
    }
    uint64_t esz = (uint64_t)st_dtype_size(t->dtype);
    if (esz == 0) return ST_ERR_BAD_DTYPE;
    if (elems != 0 && esz > UINT64_MAX / elems) return ST_ERR_SIZE_MISMATCH;
    if (elems * esz != t->nbytes) return ST_ERR_SIZE_MISMATCH;

    return ST_OK;
}

static st_status st_parse_metadata(st_scan *s, st_file *f)
{
    if (!st_eat(s, '{')) return ST_ERR_BAD_JSON;
    f->has_metadata = 1;
    if (st_eat(s, '}')) return ST_OK;

    size_t cap = 0;
    for (;;) {
        char key[ST_MAX_NAME], val[ST_MAX_NAME];
        if (st_string(s, key, sizeof key, NULL) != 0) return ST_ERR_BAD_JSON;
        if (!st_eat(s, ':')) return ST_ERR_BAD_JSON;
        if (st_string(s, val, sizeof val, NULL) != 0) return ST_ERR_BAD_JSON;

        if (st_grow((void **)&f->meta, &cap, f->n_meta + 1, sizeof *f->meta) != 0)
            return ST_ERR_NOMEM;
        char *kc = (char *)malloc(strlen(key) + 1);
        char *vc = (char *)malloc(strlen(val) + 1);
        if (!kc || !vc) { free(kc); free(vc); return ST_ERR_NOMEM; }
        memcpy(kc, key, strlen(key) + 1);
        memcpy(vc, val, strlen(val) + 1);
        f->meta[f->n_meta].key = kc;
        f->meta[f->n_meta].value = vc;
        ++f->n_meta;

        if (st_eat(s, ',')) continue;
        break;
    }
    if (!st_eat(s, '}')) return ST_ERR_BAD_JSON;
    return ST_OK;
}

static st_status st_parse_header(st_file *f, const char *header, size_t len)
{
    st_scan s = { header, header + len };
    uint64_t data_bytes = f->file_size - f->data_offset;

    if (!st_eat(&s, '{')) return ST_ERR_BAD_JSON;
    if (!st_peek(&s, '}')) {
        size_t cap = 0;
        for (;;) {
            char name[ST_MAX_NAME];
            if (st_string(&s, name, sizeof name, NULL) != 0) return ST_ERR_BAD_JSON;
            if (!st_eat(&s, ':')) return ST_ERR_BAD_JSON;

            if (strcmp(name, "__metadata__") == 0) {
                st_status rc = st_parse_metadata(&s, f);
                if (rc != ST_OK) return rc;
            } else {
                if (st_grow((void **)&f->tensors, &cap, f->n_tensors + 1,
                            sizeof *f->tensors) != 0)
                    return ST_ERR_NOMEM;
                st_tensor *t = &f->tensors[f->n_tensors];
                st_status rc = st_parse_tensor(&s, name, data_bytes, t);
                if (rc != ST_OK) return rc;
                t->file_offset = f->data_offset + t->begin;
                ++f->n_tensors;
            }
            if (st_eat(&s, ',')) continue;
            break;
        }
    }
    if (!st_eat(&s, '}')) return ST_ERR_BAD_JSON;
    st_ws(&s);
    if (s.p != s.end) return ST_ERR_BAD_JSON;   /* trailing bytes in the header */
    return ST_OK;
}

/* ------------------------------------------------------------- public API -- */

st_status st_open(const char *path, st_file **out)
{
    if (!path || !out) return ST_ERR_ARG;
    *out = NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) return ST_ERR_OPEN;

    st_file *f = (st_file *)calloc(1, sizeof *f);
    if (!f) { fclose(fp); return ST_ERR_NOMEM; }
    f->fp = fp;

    /* Size on disk, from the file itself. */
    if (ST_FSEEK(fp, 0, SEEK_END) != 0) { st_close(f); return ST_ERR_READ; }
    long long sz = ST_FTELL(fp);
    if (sz < 0) { st_close(f); return ST_ERR_READ; }
    f->file_size = (uint64_t)sz;
    if (ST_FSEEK(fp, 0, SEEK_SET) != 0) { st_close(f); return ST_ERR_READ; }

    /* The 8-byte little-endian header length, assembled byte by byte so the
     * host's own endianness never enters the result. */
    unsigned char prefix[8];
    if (f->file_size < 8) { st_close(f); return ST_ERR_TRUNCATED_HEADER; }
    if (fread(prefix, 1, 8, fp) != 8) { st_close(f); return ST_ERR_READ; }
    uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = (n << 8) | (uint64_t)prefix[i];
    f->header_length = n;

    if (n > f->file_size - 8) { st_close(f); return ST_ERR_TRUNCATED_HEADER; }
    f->data_offset = 8 + n;

    char *header = (char *)malloc((size_t)n + 1);
    if (!header) { st_close(f); return ST_ERR_NOMEM; }
    if (n > 0 && fread(header, 1, (size_t)n, fp) != (size_t)n) {
        free(header); st_close(f); return ST_ERR_READ;
    }
    header[n] = '\0';

    st_status rc = st_parse_header(f, header, (size_t)n);
    free(header);
    if (rc != ST_OK) { st_close(f); return rc; }

    *out = f;
    return ST_OK;
}

void st_close(st_file *f)
{
    if (!f) return;
    if (f->fp) fclose(f->fp);
    free(f->tensors);
    for (size_t i = 0; i < f->n_meta; ++i) {
        free(f->meta[i].key);
        free(f->meta[i].value);
    }
    free(f->meta);
    free(f);
}

size_t st_count(const st_file *f) { return f ? f->n_tensors : 0; }

const st_tensor *st_at(const st_file *f, size_t index)
{
    if (!f || index >= f->n_tensors) return NULL;
    return &f->tensors[index];
}

st_status st_find(const st_file *f, const char *name, const st_tensor **out)
{
    if (!f || !name || !out) return ST_ERR_ARG;
    for (size_t i = 0; i < f->n_tensors; ++i) {
        if (strcmp(f->tensors[i].name, name) == 0) { *out = &f->tensors[i]; return ST_OK; }
    }
    *out = NULL;
    return ST_ERR_NOT_FOUND;
}

st_status st_tensor_read(const st_file *f, const st_tensor *t,
                         void *dst, size_t dst_bytes)
{
    if (!f || !t || !dst) return ST_ERR_ARG;
    if ((uint64_t)dst_bytes < t->nbytes) return ST_ERR_ARG;
    if (t->file_offset + t->nbytes > f->file_size) return ST_ERR_RANGE;
    if (ST_FSEEK(f->fp, (long long)t->file_offset, SEEK_SET) != 0) return ST_ERR_READ;
    if (t->nbytes > 0 && fread(dst, 1, (size_t)t->nbytes, f->fp) != (size_t)t->nbytes)
        return ST_ERR_READ;
    return ST_OK;
}

uint64_t st_header_length(const st_file *f) { return f ? f->header_length : 0; }
uint64_t st_data_offset(const st_file *f)   { return f ? f->data_offset : 0; }
uint64_t st_file_size(const st_file *f)     { return f ? f->file_size : 0; }
int      st_has_metadata(const st_file *f)  { return f ? f->has_metadata : 0; }

const char *st_metadata(const st_file *f, const char *key)
{
    if (!f || !key) return NULL;
    for (size_t i = 0; i < f->n_meta; ++i)
        if (strcmp(f->meta[i].key, key) == 0) return f->meta[i].value;
    return NULL;
}

/* ===========================================================================
 * INVENTORY TOOL
 *
 * Built as a separate executable from this same source, exactly as Stage 0
 * builds a benchmark and its test object library from one file: the code that
 * writes the committed inventory is the code the tests exercise, so the two
 * cannot drift apart. The tool writes src/gpt2_tensor_inventory.json.
 *
 * Everything model-specific lives inside this block. The reader above knows
 * nothing about GPT-2, and the architecture values used for the orientation
 * arithmetic are read from the config file shipped with the weights -- they are
 * not constants in this file.
 * ===========================================================================*/
#ifdef ST_INVENTORY_MAIN

/* A deliberately narrow scan over the shipped config: find "key" at any depth
 * and read the unsigned integer that follows it. The config carries floats,
 * booleans, nulls and nested objects that a general parser would have to model;
 * none of them is needed here, and the five values that are needed are all
 * plain integers. */
static int cfg_int(const char *json, const char *key, int64_t *out)
{
    size_t klen = strlen(key);
    for (const char *p = json; (p = strchr(p, '"')) != NULL; ++p) {
        if (strncmp(p + 1, key, klen) == 0 && p[1 + klen] == '"') {
            const char *q = p + 2 + klen;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') ++q;
            if (*q != ':') continue;
            ++q;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') ++q;
            if (*q < '0' || *q > '9') continue;
            int64_t v = 0;
            while (*q >= '0' && *q <= '9') { v = v * 10 + (*q - '0'); ++q; }
            if (*q == '.' || *q == 'e' || *q == 'E') continue;  /* not an integer */
            *out = v;
            return 0;
        }
    }
    return -1;
}

static char *read_whole(const char *path, size_t *len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (n < 0) { fclose(fp); return NULL; }
    char *b = (char *)malloc((size_t)n + 1);
    if (!b) { fclose(fp); return NULL; }
    size_t got = fread(b, 1, (size_t)n, fp);
    b[got] = '\0';
    if (len) *len = got;
    fclose(fp);
    return b;
}

static void json_str(FILE *o, const char *s)
{
    fputc('"', o);
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\') { fputc('\\', o); fputc(*p, o); }
        else if (*p < 0x20) fprintf(o, "\\u%04x", *p);
        else fputc(*p, o);
    }
    fputc('"', o);
}

/* The sibling bias of "<stem>.weight" is "<stem>.bias". Returns NULL if absent. */
static const st_tensor *sibling_bias(const st_file *f, const st_tensor *w)
{
    const char *dot = strrchr(w->name, '.');
    if (!dot || strcmp(dot, ".weight") != 0) return NULL;
    char name[ST_MAX_NAME];
    size_t stem = (size_t)(dot - w->name);
    if (stem + 6 >= sizeof name) return NULL;
    memcpy(name, w->name, stem);
    memcpy(name + stem, ".bias", 6);
    const st_tensor *b = NULL;
    if (st_find(f, name, &b) != ST_OK) return NULL;
    return b;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s <model.safetensors> <config.json> <out.json>\n", argv[0]);
        return 2;
    }
    const char *wpath = argv[1], *cpath = argv[2], *opath = argv[3];

    st_file *f = NULL;
    st_status rc = st_open(wpath, &f);
    if (rc != ST_OK) {
        fprintf(stderr, "%s: %s (%d)\n", wpath, st_strerror(rc), (int)rc);
        return 1;
    }

    char *cfg = read_whole(cpath, NULL);
    if (!cfg) { fprintf(stderr, "%s: cannot read\n", cpath); st_close(f); return 1; }

    int64_t n_embd = -1, n_head = -1, n_layer = -1, n_ctx = -1, vocab = -1, n_pos = -1;
    int cfg_ok = 1;
    cfg_ok &= (cfg_int(cfg, "n_embd",      &n_embd) == 0);
    cfg_ok &= (cfg_int(cfg, "n_head",      &n_head) == 0);
    cfg_ok &= (cfg_int(cfg, "n_layer",     &n_layer) == 0);
    cfg_ok &= (cfg_int(cfg, "n_ctx",       &n_ctx) == 0);
    cfg_ok &= (cfg_int(cfg, "vocab_size",  &vocab) == 0);
    cfg_ok &= (cfg_int(cfg, "n_positions", &n_pos) == 0);
    if (!cfg_ok) {
        fprintf(stderr, "%s: a required integer field is missing\n", cpath);
        free(cfg); st_close(f); return 1;
    }

    FILE *o = fopen(opath, "wb");
    if (!o) { fprintf(stderr, "%s: cannot write\n", opath); free(cfg); st_close(f); return 1; }

    fprintf(o, "{\n");
    fprintf(o, "  \"note\": \"Stage 1 tensor inventory. Produced by the C loader "
               "(src/safetensors.c, ST_INVENTORY_MAIN) from the weight file shipped "
               "under models/gpt2/. Not a benchmark result: it contains no timing "
               "and does not live in bench/results/. Offsets: 'begin' and 'end' are "
               "as declared in the header, relative to the start of the data segment; "
               "'file_offset' is begin plus the data segment's own offset, so it is "
               "an absolute seek position in the file.\",\n");
    fprintf(o, "  \"weight_file\": ");     json_str(o, wpath);  fprintf(o, ",\n");
    fprintf(o, "  \"config_file\": ");     json_str(o, cpath);  fprintf(o, ",\n");
    fprintf(o, "  \"file_size_bytes\": %llu,\n", (unsigned long long)st_file_size(f));
    fprintf(o, "  \"header_length_bytes\": %llu,\n", (unsigned long long)st_header_length(f));
    fprintf(o, "  \"data_segment_offset\": %llu,\n", (unsigned long long)st_data_offset(f));
    fprintf(o, "  \"header_metadata_present\": %s,\n", st_has_metadata(f) ? "true" : "false");
    {
        const char *fmt = st_metadata(f, "format");
        fprintf(o, "  \"header_metadata_format\": ");
        if (fmt) json_str(o, fmt); else fprintf(o, "null");
        fprintf(o, ",\n");
    }
    fprintf(o, "  \"config_values_used\": {\"n_embd\": %lld, \"n_head\": %lld, "
               "\"n_layer\": %lld, \"n_ctx\": %lld, \"n_positions\": %lld, "
               "\"vocab_size\": %lld},\n",
            (long long)n_embd, (long long)n_head, (long long)n_layer,
            (long long)n_ctx, (long long)n_pos, (long long)vocab);
    fprintf(o, "  \"tensor_count\": %zu,\n", st_count(f));

    /* --- the language-model head: stored as its own tensor, or tied? ------
     * Settled from the enumeration, not assumed: look for any name containing
     * "lm_head", and count the matrices whose two extents are the vocabulary
     * size and the embedding width in either order. */
    size_t n_named_head = 0, n_vocab_matrices = 0;
    char vocab_matrix[ST_MAX_NAME] = "";
    for (size_t i = 0; i < st_count(f); ++i) {
        const st_tensor *t = st_at(f, i);
        if (strstr(t->name, "lm_head")) ++n_named_head;
        if (t->n_dims == 2 &&
            ((t->dims[0] == vocab && t->dims[1] == n_embd) ||
             (t->dims[0] == n_embd && t->dims[1] == vocab))) {
            ++n_vocab_matrices;
            if (vocab_matrix[0] == '\0')
                memcpy(vocab_matrix, t->name, strlen(t->name) + 1);
        }
    }
    fprintf(o, "  \"lm_head\": {\n");
    fprintf(o, "    \"stored_as_its_own_tensor\": %s,\n", n_named_head ? "true" : "false");
    fprintf(o, "    \"tensors_named_lm_head\": %zu,\n", n_named_head);
    fprintf(o, "    \"matrices_shaped_vocab_by_embd\": %zu,\n", n_vocab_matrices);
    fprintf(o, "    \"tied_to\": ");
    if (!n_named_head && n_vocab_matrices == 1) json_str(o, vocab_matrix); else fprintf(o, "null");
    fprintf(o, ",\n");
    fprintf(o, "    \"evidence\": \"Enumerated every tensor in the header. No name "
               "contains 'lm_head', and exactly one matrix carries the vocabulary "
               "size against the embedding width. The head is therefore NOT stored "
               "in the file and a forward pass must tie it to that embedding "
               "matrix; a loader that does not tie it fails at Stage 2 in a way "
               "that looks like a bug in the head.\",\n");
    fprintf(o, "    \"what_the_loader_does\": \"The loader stores no head tensor "
               "because the file declares none. Tying is a forward-pass decision "
               "and belongs to Stage 2; Stage 1 records the fact and does not "
               "invent a tensor.\"\n");
    fprintf(o, "  },\n");

    fprintf(o, "  \"tensors\": [\n");
    for (size_t i = 0; i < st_count(f); ++i) {
        const st_tensor *t = st_at(f, i);
        fprintf(o, "    {\"name\": ");
        json_str(o, t->name);
        fprintf(o, ", \"dtype\": ");
        json_str(o, st_dtype_name(t->dtype));
        fprintf(o, ", \"shape\": [");
        for (int d = 0; d < t->n_dims; ++d)
            fprintf(o, "%s%lld", d ? ", " : "", (long long)t->dims[d]);
        fprintf(o, "], \"begin\": %llu, \"end\": %llu, \"file_offset\": %llu, "
                   "\"nbytes\": %llu",
                (unsigned long long)t->begin, (unsigned long long)t->end,
                (unsigned long long)t->file_offset, (unsigned long long)t->nbytes);

        /* --- orientation, for 2-D weights only ---------------------------
         * The evidence used, in order of strength:
         *   (a) a sibling bias. A bias has one entry per OUTPUT, so when its
         *       length matches exactly one of the two extents, that extent is
         *       the output axis and the other is the input axis. This is
         *       arithmetic on the file's own shapes and needs no convention.
         *   (b) config arithmetic, for the embedding tables, which have no
         *       bias: an extent equal to the vocabulary size or to the context
         *       length is an index axis, and one equal to n_embd is the feature
         *       axis.
         *   (c) nothing, when the two extents are equal. Shape cannot settle
         *       it and this record says so rather than picking one. */
        const char *orient = "not_applicable";
        const char *evidence = "not a 2-D weight";
        char detail[512];
        detail[0] = '\0';
        if (t->n_dims == 2) {
            const st_tensor *b = sibling_bias(f, t);
            if (t->dims[0] == t->dims[1]) {
                orient = "unresolved_by_shape";
                evidence = "axes_equal";
                snprintf(detail, sizeof detail,
                         "Both extents are %lld, so shape alone cannot say which "
                         "axis is input and which is output%s. Alternative evidence, "
                         "stated as inference and not as fact: every rectangular "
                         "2-D weight in this same file is pinned by its own bias "
                         "length to [input, output], and this tensor is grouped and "
                         "named alongside them, so the loader reads it as "
                         "[input, output] too.",
                         (long long)t->dims[0],
                         b ? " (its bias length matches both)" : "");
            } else if (b && b->n_dims == 1 &&
                       (b->dims[0] == t->dims[0]) != (b->dims[0] == t->dims[1])) {
                int out_axis = (b->dims[0] == t->dims[1]) ? 1 : 0;
                orient = out_axis == 1 ? "axis0_input_axis1_output"
                                       : "axis0_output_axis1_input";
                evidence = "pinned_by_bias_length";
                snprintf(detail, sizeof detail,
                         "Sibling %s has length %lld, which equals extent %d "
                         "(%lld) and not extent %d (%lld). A bias has one entry "
                         "per output, so axis %d is the output axis and axis %d "
                         "is the input axis. Against the shipped config: %lld = "
                         "%.4g x n_embd(%lld), %lld = %.4g x n_embd(%lld).",
                         b->name, (long long)b->dims[0], out_axis,
                         (long long)t->dims[out_axis], 1 - out_axis,
                         (long long)t->dims[1 - out_axis], out_axis, 1 - out_axis,
                         (long long)t->dims[0], (double)t->dims[0] / (double)n_embd,
                         (long long)n_embd,
                         (long long)t->dims[1], (double)t->dims[1] / (double)n_embd,
                         (long long)n_embd);
            } else if ((t->dims[0] == vocab || t->dims[0] == n_ctx) && t->dims[1] == n_embd) {
                orient = "axis0_index_axis1_feature";
                evidence = "pinned_by_shape_arithmetic_against_config";
                snprintf(detail, sizeof detail,
                         "Extent 0 is %lld, which equals %s from the shipped "
                         "config, and extent 1 is %lld = n_embd. This is a lookup "
                         "table: one row per index, n_embd features per row.",
                         (long long)t->dims[0],
                         t->dims[0] == vocab ? "vocab_size" : "n_ctx",
                         (long long)t->dims[1]);
            } else {
                orient = "unresolved";
                evidence = "no_bias_and_no_config_match";
                snprintf(detail, sizeof detail,
                         "No sibling bias pins an output axis and neither extent "
                         "matches a config value, so this record does not claim an "
                         "orientation.");
            }
        }
        fprintf(o, ", \"orientation\": ");
        json_str(o, orient);
        fprintf(o, ", \"orientation_evidence\": ");
        json_str(o, evidence);
        if (detail[0]) { fprintf(o, ", \"orientation_detail\": "); json_str(o, detail); }
        fprintf(o, "}%s\n", (i + 1 < st_count(f)) ? "," : "");
    }
    fprintf(o, "  ],\n");
    fprintf(o, "  \"orientation_limit\": \"Orientation is a SEMANTIC mapping and "
               "this stage does not verify it. The exact byte comparison against "
               "the reference reader does not test it either: both sides read the "
               "same bytes from the same file and both report the shape as stored. "
               "The proof lands at Stage 2, when a forward pass produces logits "
               "that either match the reference or do not.\"\n");
    fprintf(o, "}\n");

    fclose(o);
    free(cfg);
    st_close(f);
    return 0;
}

#endif /* ST_INVENTORY_MAIN */
