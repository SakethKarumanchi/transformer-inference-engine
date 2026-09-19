/* Unit test for src/tokenizer.c.
 *
 * The round trip is checked BOTH ways, and the second is the one that matters:
 * decode(encode(record)) byte-exact, and the id sequence element for element
 * against the committed reference sequence. A decode round trip alone passes
 * with a completely wrong tokenization -- any bijection would satisfy it -- so
 * only the id comparison can catch a merge-order bug.
 *
 * Structural, like every Stage 0 test: nothing here is timed.
 */
#include "test_util.h"
#include "tokenizer.h"

#include <stdint.h>

#ifndef TIE_MODEL_DIR
#define TIE_MODEL_DIR "models/gpt2"
#endif
#ifndef TIE_REPO_DIR
#define TIE_REPO_DIR "."
#endif

#define MAX_RECORDS 256
#define MAX_BYTES   2048
#define MAX_IDS     2048

static const char *TOKENIZER_JSON = TIE_MODEL_DIR "/tokenizer.json";
static const char *MERGES_TXT     = TIE_MODEL_DIR "/merges.txt";
static const char *CORPUS         = TIE_REPO_DIR "/tests/fixtures/tokenizer_roundtrip_corpus.tsv";
static const char *EXPECTED       = TIE_REPO_DIR "/tests/fixtures/tokenizer_expected_ids.tsv";

static const char *REQUIRED_CLASSES[] = {
    "utf8_multibyte", "whitespace_runs", "newlines_tabs", "empty",
    "mid_merge_prefix", "special_literal", "invalid_utf8"
};
static const size_t N_REQUIRED = sizeof REQUIRED_CLASSES / sizeof REQUIRED_CLASSES[0];

typedef struct {
    char           cls[32];
    unsigned char  bytes[MAX_BYTES];
    size_t         n_bytes;
    char           label[160];
    int            has_reference;
    int32_t        ids[MAX_IDS];
    size_t         n_ids;
} record;

static record  RECORDS[MAX_RECORDS];
static size_t  N_RECORDS = 0;

static int hexval(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Splits a line on tabs in place, returning the field count. */
static int split_tabs(char *line, char *fields[], int max_fields)
{
    int n = 0;
    fields[n++] = line;
    for (char *p = line; *p && n < max_fields; ++p) {
        if (*p == '\t') { *p = '\0'; fields[n++] = p + 1; }
    }
    return n;
}

static int load_corpus(void)
{
    FILE *f = fopen(CORPUS, "rb");
    if (!f) return 0;
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (line[0] == '#' || line[0] == '\0') continue;

        char *fields[4];
        int nf = split_tabs(line, fields, 4);
        if (nf < 2) { fclose(f); return 0; }
        if (N_RECORDS >= MAX_RECORDS) { fclose(f); return 0; }

        record *r = &RECORDS[N_RECORDS];
        memset(r, 0, sizeof *r);
        snprintf(r->cls, sizeof r->cls, "%s", fields[0]);
        if (nf >= 3) snprintf(r->label, sizeof r->label, "%s", fields[2]);

        const char *hex = fields[1];
        size_t hl = strlen(hex);
        if (hl % 2 || hl / 2 > MAX_BYTES) { fclose(f); return 0; }
        for (size_t i = 0; i < hl; i += 2) {
            int hi = hexval(hex[i]), lo = hexval(hex[i + 1]);
            if (hi < 0 || lo < 0) { fclose(f); return 0; }
            r->bytes[i / 2] = (unsigned char)(hi * 16 + lo);
        }
        r->n_bytes = hl / 2;
        ++N_RECORDS;
    }
    fclose(f);
    return 1;
}

static int load_expected(void)
{
    FILE *f = fopen(EXPECTED, "rb");
    if (!f) return 0;
    char line[65536];
    size_t seen = 0;
    while (fgets(line, sizeof line, f)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (line[0] == '#' || line[0] == '\0') continue;

        char *fields[5];
        int nf = split_tabs(line, fields, 5);
        if (nf < 3) { fclose(f); return 0; }
        size_t index = (size_t)atoi(fields[0]);
        if (index >= N_RECORDS || index != seen) { fclose(f); return 0; }
        record *r = &RECORDS[index];
        if (strcmp(r->cls, fields[1]) != 0) { fclose(f); return 0; }

        if (strcmp(fields[2], "NO_REFERENCE") == 0) {
            r->has_reference = 0;
        } else {
            r->has_reference = 1;
            r->n_ids = 0;
            char *p = fields[2];
            while (*p) {
                if (*p == ',') { ++p; continue; }
                if (r->n_ids >= MAX_IDS) { fclose(f); return 0; }
                r->ids[r->n_ids++] = (int32_t)strtol(p, &p, 10);
            }
        }
        ++seen;
    }
    fclose(f);
    return seen == N_RECORDS;
}

int main(void)
{
    printf("test_tokenizer\n");
    tie_redirect_results();

    CHECK(load_corpus() && N_RECORDS > 0,
          "the round-trip corpus fixture loads: %zu records [%s]", N_RECORDS, CORPUS);
    if (N_RECORDS == 0) TIE_SUMMARY("test_tokenizer");
    CHECK(load_expected(),
          "the committed reference id fixture loads and lines up record for "
          "record with the corpus [%s]", EXPECTED);

    tokenizer *tk = NULL;
    tok_status rc = tok_load(TOKENIZER_JSON, &tk);
    CHECK(rc == TOK_OK, "the tokenizer loads from the shipped artifact: %s [%s]",
          tok_strerror(rc), TOKENIZER_JSON);
    if (rc != TOK_OK) {
        printf("       the tokenizer artifacts are required for this test and are "
               "NOT optional: they are gitignored, so a fresh clone must have "
               "them placed under models/gpt2/ before the suite can run.\n");
        TIE_SUMMARY("test_tokenizer");
    }

    size_t vocab = tok_vocab_size(tk);
    printf("       vocabulary %zu entries, %zu merges, %zu added tokens\n",
           vocab, tok_merge_count(tk), tok_added_token_count(tk));

    /* (6) the corpus carries every required class ------------------------- */
    for (size_t c = 0; c < N_REQUIRED; ++c) {
        size_t count = 0;
        for (size_t i = 0; i < N_RECORDS; ++i)
            if (strcmp(RECORDS[i].cls, REQUIRED_CLASSES[c]) == 0) ++count;
        CHECK(count >= 1, "the corpus carries %zu record(s) of the required class "
              "\"%s\" -- this assert is what stops a later edit weakening the corpus",
              count, REQUIRED_CLASSES[c]);
    }

    /* (1) decode(encode(record)) is byte-exact, every record ---------------- */
    /* (2) the id sequence matches the reference, element for element ------- */
    /* (4) every id produced is below the vocabulary size ------------------- */
    /* (5) the invalid-UTF-8 class round-trips byte-exactly ----------------- */
    {
        int rt_ok = 1, ids_ok = 1, bound_ok = 1, invalid_ok = 1;
        size_t rt_checked = 0, id_checked = 0, invalid_checked = 0, ids_total = 0;
        char first_rt[256] = "", first_id[256] = "";

        for (size_t i = 0; i < N_RECORDS; ++i) {
            record *r = &RECORDS[i];
            int32_t ids[MAX_IDS];
            size_t n_ids = 0;
            tok_status erc = tok_encode(tk, r->bytes, r->n_bytes, ids, MAX_IDS, &n_ids);
            if (erc != TOK_OK) {
                rt_ok = 0;
                if (!first_rt[0])
                    snprintf(first_rt, sizeof first_rt, "record %zu (%s) failed to "
                             "encode: %s", i, r->cls, tok_strerror(erc));
                continue;
            }
            ids_total += n_ids;

            for (size_t k = 0; k < n_ids; ++k)
                if (ids[k] < 0 || (size_t)ids[k] >= vocab) bound_ok = 0;

            unsigned char back[MAX_BYTES * 2];
            size_t n_back = 0;
            tok_status drc = tok_decode(tk, ids, n_ids, back, sizeof back, &n_back);
            int round_trips = (drc == TOK_OK && n_back == r->n_bytes &&
                               memcmp(back, r->bytes, n_back) == 0);
            if (!round_trips) {
                rt_ok = 0;
                if (!first_rt[0])
                    snprintf(first_rt, sizeof first_rt, "record %zu (%s, \"%s\") does "
                             "not round-trip: %zu bytes in, %zu out", i, r->cls,
                             r->label, r->n_bytes, n_back);
            }
            ++rt_checked;
            if (strcmp(r->cls, "invalid_utf8") == 0) {
                ++invalid_checked;
                if (!round_trips) invalid_ok = 0;
            }

            if (r->has_reference) {
                ++id_checked;
                int same = (n_ids == r->n_ids);
                for (size_t k = 0; same && k < n_ids; ++k)
                    if (ids[k] != r->ids[k]) same = 0;
                if (!same) {
                    ids_ok = 0;
                    if (!first_id[0])
                        snprintf(first_id, sizeof first_id, "record %zu (%s, \"%s\"): "
                                 "%zu ids, reference has %zu", i, r->cls, r->label,
                                 n_ids, r->n_ids);
                }
            }
        }

        CHECK(rt_ok, "decode(encode(record)) is byte-exact for all %zu records%s%s",
              rt_checked, first_rt[0] ? " -- " : "", first_rt);
        CHECK(ids_ok, "the id sequence equals the committed reference sequence, "
              "element for element and in length, for all %zu records the reference "
              "can ingest (%zu ids in total)%s%s", id_checked, ids_total,
              first_id[0] ? " -- " : "", first_id);
        CHECK(bound_ok, "every id produced across the whole corpus is below the "
              "vocabulary size %zu read from the artifact", vocab);
        CHECK(invalid_ok && invalid_checked > 0,
              "all %zu records of the invalid-UTF-8 class round-trip byte-exactly",
              invalid_checked);

        /* The records with no committed reference sequence are exactly the
         * invalid-UTF-8 ones, and that is a property of the reference API
         * rather than a gap in the fixture. */
        int aligned = 1;
        for (size_t i = 0; i < N_RECORDS; ++i) {
            int is_invalid = (strcmp(RECORDS[i].cls, "invalid_utf8") == 0);
            if (is_invalid == RECORDS[i].has_reference) aligned = 0;
        }
        CHECK(aligned, "the only records without a reference id sequence are the "
              "invalid-UTF-8 ones -- the reference tokenizer takes str and rejects "
              "those inputs outright, so no reference sequence exists for them");
    }

    /* (3) the empty string ------------------------------------------------- */
    {
        size_t empty_index = N_RECORDS;
        for (size_t i = 0; i < N_RECORDS; ++i)
            if (RECORDS[i].n_bytes == 0) { empty_index = i; break; }
        CHECK(empty_index < N_RECORDS, "the corpus carries the empty record");
        if (empty_index < N_RECORDS) {
            record *r = &RECORDS[empty_index];
            int32_t ids[8];
            size_t n_ids = 1;
            tok_status erc = tok_encode(tk, r->bytes, 0, ids, 8, &n_ids);
            int same = (erc == TOK_OK && n_ids == r->n_ids);
            for (size_t k = 0; same && k < n_ids; ++k) if (ids[k] != r->ids[k]) same = 0;
            CHECK(same, "the empty string encodes to exactly the reference's id "
                  "sequence for it (%zu ids, reference %zu)", n_ids, r->n_ids);

            unsigned char back[8];
            size_t n_back = 1;
            tok_status drc = tok_decode(tk, ids, n_ids, back, sizeof back, &n_back);
            CHECK(drc == TOK_OK && n_back == 0,
                  "and decodes back to zero bytes (%zu)", n_back);
        }
    }

    /* (7) merge ranks agree with the merges artifact ------------------------ */
    {
        FILE *m = fopen(MERGES_TXT, "rb");
        CHECK(m != NULL, "the merges artifact opens for an independent check [%s]",
              MERGES_TXT);
        if (m) {
            char line[512];
            int rank = -1;
            int sampled = 0, agree = 1, first_line = 1;
            char first_bad[256] = "";
            while (fgets(line, sizeof line, m)) {
                size_t len = strlen(line);
                while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                    line[--len] = '\0';
                /* ONLY the first line is the "#version" header. Eight genuine
                 * merge lines also begin with '#' -- pairs built out of the
                 * hash character itself, "# #" at rank 1979 among them -- and
                 * skipping every '#' line would silently shift the ranks of
                 * all the merges after it. */
                if (first_line) {
                    first_line = 0;
                    if (strncmp(line, "#version", 8) == 0) continue;
                }
                if (line[0] == '\0') continue;
                ++rank;
                if (rank % 997 != 0) continue;     /* sample the artifact */
                char *sp = strchr(line, ' ');
                if (!sp) continue;
                *sp = '\0';
                int got = tok_merge_rank(tk, line, sp + 1);
                ++sampled;
                if (got != rank) {
                    agree = 0;
                    if (!first_bad[0])
                        snprintf(first_bad, sizeof first_bad,
                                 "pair at line rank %d looked up as %d", rank, got);
                }
            }
            fclose(m);
            CHECK(sampled > 0 && agree,
                  "every one of the %d sampled merge pairs has the rank the merges "
                  "artifact gives it%s%s", sampled, first_bad[0] ? " -- " : "",
                  first_bad);
            CHECK((size_t)(rank + 1) == tok_merge_count(tk),
                  "the merges artifact has %d merges and the loaded table has %zu -- "
                  "the two artifacts agree on the merge list", rank + 1,
                  tok_merge_count(tk));
        }
    }

    /* (8) decode of an id at or above the vocabulary size ------------------- */
    {
        int32_t bad[] = { (int32_t)vocab };
        unsigned char out[16];
        size_t n_out = 1;
        tok_status drc = tok_decode(tk, bad, 1, out, sizeof out, &n_out);
        CHECK(drc == TOK_ERR_BAD_ID && n_out == 0,
              "decode of id %zu, the first id at or above the vocabulary size, "
              "returns TOK_ERR_BAD_ID (%s) rather than reading out of bounds",
              vocab, tok_strerror(drc));

        int32_t bad2[] = { (int32_t)vocab + 1000 };
        n_out = 1;
        drc = tok_decode(tk, bad2, 1, out, sizeof out, &n_out);
        CHECK(drc == TOK_ERR_BAD_ID, "and likewise for an id well past the end (%s)",
              tok_strerror(drc));
    }

    tok_free(tk);
    TIE_SUMMARY("test_tokenizer");
}
