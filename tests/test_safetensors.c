/* Unit test for src/safetensors.c.
 *
 * Structural, like every Stage 0 test: it asserts that the loader reads the
 * file the file actually is, and it times nothing. The malformed cases are
 * built here as temporary files in the working directory (the build tree under
 * ctest) rather than shipped as fixtures, so a truncated or corrupt weight file
 * never sits in the repository waiting to be mistaken for a real one.
 */
#include "test_util.h"
#include "safetensors.h"

#include <stdint.h>

#ifndef TIE_MODEL_DIR
#define TIE_MODEL_DIR "models/gpt2"
#endif

static const char *WEIGHTS = TIE_MODEL_DIR "/model.safetensors";

/* Writes buf to path. Returns 1 on success. */
static int write_file(const char *path, const void *buf, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    size_t w = fwrite(buf, 1, n, f);
    fclose(f);
    return w == n;
}

/* A minimal well-formed file: one F32 tensor of two elements. The header text
 * is built here so the malformed variants below differ from it in exactly one
 * way each. */
static size_t build_file(char *out, size_t cap, const char *header,
                         uint64_t declared_len, size_t data_bytes)
{
    size_t hlen = strlen(header);
    size_t total = 8 + hlen + data_bytes;
    if (total > cap) return 0;
    for (int i = 0; i < 8; ++i) out[i] = (char)((declared_len >> (8 * i)) & 0xFF);
    memcpy(out + 8, header, hlen);
    memset(out + 8 + hlen, 0, data_bytes);
    return total;
}

int main(void)
{
    printf("test_safetensors\n");
    tie_redirect_results();

    st_file *f = NULL;
    st_status rc = st_open(WEIGHTS, &f);
    CHECK(rc == ST_OK, "the shipped weight file opens and parses: %s [%s]",
          st_strerror(rc), WEIGHTS);
    if (rc != ST_OK) {
        printf("       the weight file is required for this test and is NOT "
               "optional: it is gitignored, so a fresh clone must have the "
               "artifacts placed under models/gpt2/ before the suite can run.\n");
        TIE_SUMMARY("test_safetensors");
    }

    uint64_t fsize = st_file_size(f);
    uint64_t hlen  = st_header_length(f);
    uint64_t dstart = st_data_offset(f);
    printf("       file %llu B, header %llu B, data segment starts at %llu\n",
           (unsigned long long)fsize, (unsigned long long)hlen,
           (unsigned long long)dstart);

    /* (1) the length prefix plus the header lies within the file ----------- */
    CHECK(8 + hlen <= fsize,
          "the 8-byte length prefix plus the %llu-byte header fits inside the "
          "%llu-byte file", (unsigned long long)hlen, (unsigned long long)fsize);
    CHECK(dstart == 8 + hlen,
          "the data segment begins at 8 + header length (%llu)",
          (unsigned long long)dstart);

    size_t n = st_count(f);
    CHECK(n > 0, "the header declares %zu tensors", n);

    /* (2) every byte range lies wholly within the data segment ------------- */
    uint64_t data_bytes = fsize - dstart;
    int in_range = 1, sized = 1;
    const st_tensor *worst = NULL;
    for (size_t i = 0; i < n; ++i) {
        const st_tensor *t = st_at(f, i);
        if (!(t->begin <= t->end && t->end <= data_bytes &&
              t->file_offset + t->nbytes <= fsize)) { in_range = 0; worst = t; }

        /* (4) declared length == product of extents times the dtype size --- */
        uint64_t elems = 1;
        for (int d = 0; d < t->n_dims; ++d) elems *= (uint64_t)t->dims[d];
        if (elems * (uint64_t)st_dtype_size(t->dtype) != t->nbytes) { sized = 0; worst = t; }
    }
    CHECK(in_range, "every one of the %zu declared byte ranges lies wholly "
          "inside the %llu-byte data segment%s%s", n, (unsigned long long)data_bytes,
          in_range ? "" : ", first offender ", in_range ? "" : (worst ? worst->name : "?"));
    CHECK(sized, "every tensor's declared byte length equals the product of its "
          "extents times its dtype size%s%s", sized ? "" : ", first offender ",
          sized ? "" : (worst ? worst->name : "?"));

    /* (3) no two byte ranges overlap --------------------------------------- */
    {
        size_t *order = (size_t *)malloc(n * sizeof *order);
        for (size_t i = 0; i < n; ++i) order[i] = i;
        for (size_t i = 1; i < n; ++i) {            /* insertion sort by begin */
            size_t k = order[i], j = i;
            while (j > 0 && st_at(f, order[j - 1])->begin > st_at(f, k)->begin) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = k;
        }
        int disjoint = 1;
        const char *a = NULL, *b = NULL;
        for (size_t i = 1; i < n; ++i) {
            const st_tensor *p = st_at(f, order[i - 1]), *q = st_at(f, order[i]);
            if (q->begin < p->end) { disjoint = 0; a = p->name; b = q->name; break; }
        }
        CHECK(disjoint, "no two tensors' byte ranges overlap%s%s%s",
              disjoint ? "" : " (", disjoint ? "" : (a ? a : "?"),
              disjoint ? "" : (b ? b : "?"));
        free(order);
    }

    /* (8) lookup by an enumerated name returns the same descriptor ---------- */
    {
        const st_tensor *first = st_at(f, 0);
        const st_tensor *found = NULL;
        st_status lrc = st_find(f, first->name, &found);
        CHECK(lrc == ST_OK && found != NULL &&
              memcmp(first, found, sizeof *first) == 0,
              "lookup of \"%s\", a name taken from the parser's own enumeration, "
              "returns a descriptor identical to the enumerated one", first->name);

        const st_tensor *last = st_at(f, n - 1);
        found = NULL;
        lrc = st_find(f, last->name, &found);
        CHECK(lrc == ST_OK && found != NULL &&
              memcmp(last, found, sizeof *last) == 0,
              "the same holds for the last enumerated tensor, \"%s\"", last->name);
    }

    /* (9) lookup of an absent name is an error, not a crash ---------------- */
    {
        const st_tensor *found = (const st_tensor *)0x1;
        st_status lrc = st_find(f, "this.tensor.does.not.exist", &found);
        CHECK(lrc == ST_ERR_NOT_FOUND && found == NULL,
              "lookup of an absent name returns ST_ERR_NOT_FOUND (%s) and clears "
              "the out pointer", st_strerror(lrc));
    }

    /* (10) the tensor count equals the header's tensor entries, metadata
     *      excluded. Counted independently of the parser, by reading the raw
     *      header bytes and counting the "dtype" keys -- one per tensor entry,
     *      none in the metadata object. ------------------------------------- */
    {
        FILE *raw = fopen(WEIGHTS, "rb");
        CHECK(raw != NULL, "the raw header can be re-read for an independent count");
        if (raw) {
            char *hbuf = (char *)malloc((size_t)hlen + 1);
            fseek(raw, 8, SEEK_SET);
            size_t got = fread(hbuf, 1, (size_t)hlen, raw);
            hbuf[got] = '\0';
            int dtypes = tie_count(hbuf, "\"dtype\":");
            int metas  = tie_count(hbuf, "\"__metadata__\":");
            CHECK((size_t)dtypes == n,
                  "the parser's tensor count (%zu) equals the number of tensor "
                  "entries in the header (%d \"dtype\" keys)", n, dtypes);
            CHECK(metas == 1 && st_has_metadata(f),
                  "the header carries one __metadata__ entry and it is excluded "
                  "from the tensor count (format=%s)",
                  st_metadata(f, "format") ? st_metadata(f, "format") : "(none)");
            free(hbuf);
            fclose(raw);
        }
    }

    st_close(f);

    /* ---- the rejection cases, each built as a temporary file ------------- */
    {
        char buf[512];
        const char *good =
            "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        size_t total = build_file(buf, sizeof buf, good, strlen(good), 8);
        CHECK(total > 0 && write_file("tie_st_good.tmp", buf, total),
              "a minimal well-formed file can be written for the rejection cases");

        st_file *g = NULL;
        st_status grc = st_open("tie_st_good.tmp", &g);
        CHECK(grc == ST_OK && st_count(g) == 1,
              "the minimal file parses, so the rejections below are about the "
              "corruption and not about the shape of the test file: %s",
              st_strerror(grc));
        st_close(g);

        /* (5) a truncated header: the prefix declares more than the file holds */
        total = build_file(buf, sizeof buf, good, strlen(good) + 4096, 8);
        CHECK(write_file("tie_st_trunc.tmp", buf, total), "truncated-header case written");
        st_file *t1 = (st_file *)0x1;
        st_status trc = st_open("tie_st_trunc.tmp", &t1);
        CHECK(trc == ST_ERR_TRUNCATED_HEADER && t1 == NULL,
              "a header length that runs past end of file is rejected with "
              "ST_ERR_TRUNCATED_HEADER (%s) and nothing is read out of bounds",
              st_strerror(trc));

        /* (6) a declared offset beyond end of file */
        {
            const char *far =
                "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,999999]}}";
            total = build_file(buf, sizeof buf, far, strlen(far), 8);
            CHECK(write_file("tie_st_far.tmp", buf, total), "out-of-range case written");
            st_file *t2 = (st_file *)0x1;
            st_status frc = st_open("tie_st_far.tmp", &t2);
            CHECK(frc == ST_ERR_RANGE && t2 == NULL,
                  "a declared end offset beyond the data segment is rejected with "
                  "ST_ERR_RANGE (%s)", st_strerror(frc));
        }

        /* (7) a malformed header is rejected rather than parsed into garbage */
        {
            const char *bad[] = {
                "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}",  /* unbalanced */
                "{\"t\":{\"dtype\":\"F32\",\"shape\":[2,],\"data_offsets\":[0,8]}}", /* trailing comma */
                "{\"t\":{\"dtype\":\"F32\",\"shape\":[-2],\"data_offsets\":[0,8]}}", /* negative extent */
                "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}xx",/* trailing bytes */
                "{\"t\":{\"dtype\":\"F33\",\"shape\":[2],\"data_offsets\":[0,8]}}",  /* unknown dtype */
                "{\"t\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,8]}}",  /* size mismatch */
                "not json at all"
            };
            const st_status want[] = {
                ST_ERR_BAD_JSON, ST_ERR_BAD_JSON, ST_ERR_BAD_JSON, ST_ERR_BAD_JSON,
                ST_ERR_BAD_DTYPE, ST_ERR_SIZE_MISMATCH, ST_ERR_BAD_JSON
            };
            for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
                total = build_file(buf, sizeof buf, bad[i], strlen(bad[i]), 8);
                write_file("tie_st_bad.tmp", buf, total);
                st_file *t3 = (st_file *)0x1;
                st_status brc = st_open("tie_st_bad.tmp", &t3);
                CHECK(brc == want[i] && t3 == NULL,
                      "malformed header %zu is rejected with the documented error "
                      "(%s), not parsed into garbage", i, st_strerror(brc));
            }
        }

        remove("tie_st_good.tmp");
        remove("tie_st_trunc.tmp");
        remove("tie_st_far.tmp");
        remove("tie_st_bad.tmp");
    }

    TIE_SUMMARY("test_safetensors");
}
