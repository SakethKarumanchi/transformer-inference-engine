/* Unit test for src/kv_cache.c -- the Stage 4 key/value cache.
 *
 * REDUCED SHAPES, STATED. Most checks below run at small shapes (2 to 3 layers,
 * 4 to 8 embedding width) because what is being asserted is addressing and
 * refusal behaviour, which does not depend on the model's size. The footprint
 * check is the exception: it runs at this model's real shapes, including
 * capacity 1024, because the number 75,497,472 is a claim the Stage 4 entry
 * makes and it should be the code that proves it. The reduction applies to
 * NOTHING in the benchmark.
 *
 * NO TOLERANCE APPEARS HERE. Every assertion is exact: a byte is stored or it
 * is not, a refusal leaves memory unchanged or it does not, a row pointer lands
 * on offset p * n_embd or it does not.
 */
#include "test_util.h"
#include "kv_cache.h"

#include <stdint.h>

/* The footprint, recomputed from the arguments rather than copied from the
 * header's arithmetic, so the two derivations have to agree. */
static size_t expect_bytes(int n_layer, int n_embd, int capacity)
{
    return (size_t)n_layer * 2u * (size_t)capacity * (size_t)n_embd * sizeof(float);
}

static void fill(float *v, int n, float base)
{
    for (int i = 0; i < n; ++i) v[i] = base + (float)i;
}

int main(void)
{
    printf("test_kv_cache\n");
    printf("  reduced shapes for the addressing and refusal checks; the footprint check "
           "runs at this model's real shapes including capacity 1024\n");

    /* ---- footprint = n_layer * 2 * capacity * n_embd * 4, at three capacities,
     *      the last of them this model's n_ctx ---- */
    {
        const int caps[3] = { 1, 128, 1024 };
        for (int i = 0; i < 3; ++i) {
            size_t want = expect_bytes(12, 768, caps[i]);
            CHECK(kv_cache_footprint_for(12, 768, caps[i]) == want,
                  "footprint_for(12, 768, %d) = %zu bytes, computed from the arguments",
                  caps[i], want);
        }
        /* The one literal in this file, and it is the number the Stage 4 entry
         * reports: 12 layers x 2 x 1024 positions x 768 floats x 4 bytes. */
        CHECK(kv_cache_footprint_for(12, 768, 1024) == 75497472u,
              "at capacity n_ctx = 1024 the footprint is exactly 75,497,472 bytes");
        CHECK(kv_cache_footprint_for(12, 768, 1) == 73728u,
              "one position across twelve layers is 73,728 bytes");

        /* A non-positive shape yields 0 rather than a size derived from nonsense. */
        CHECK(kv_cache_footprint_for(0, 768, 1024) == 0u &&
              kv_cache_footprint_for(12, 0, 1024) == 0u &&
              kv_cache_footprint_for(12, 768, 0) == 0u,
              "footprint_for returns 0 for a non-positive n_layer, n_embd or capacity");
    }

    /* ---- creation records its shapes, and the allocated footprint matches ---- */
    {
        kv_cache *c = NULL;
        CHECK(kv_cache_create(3, 8, 5, &c) == KV_OK && c != NULL,
              "create(3 layers, n_embd 8, capacity 5) succeeds");
        CHECK(kv_cache_n_layer(c) == 3 && kv_cache_n_embd(c) == 8 &&
              kv_cache_capacity(c) == 5,
              "the cache reports back the shapes it was created with");
        CHECK(kv_cache_footprint_bytes(c) == expect_bytes(3, 8, 5),
              "the ALLOCATED footprint equals n_layer x 2 x capacity x n_embd x 4 = %zu bytes",
              expect_bytes(3, 8, 5));
        CHECK(kv_cache_length(c) == 0, "a fresh cache has length 0");
        kv_cache_destroy(c);
    }

    /* ---- create with zero or negative arguments fails cleanly ---- */
    {
        kv_cache *c = (kv_cache *)(intptr_t)0xdeadbeef;
        CHECK(kv_cache_create(0, 8, 4, &c) == KV_ERR_ARG && c == NULL,
              "create with n_layer 0 returns KV_ERR_ARG and leaves the out pointer NULL");
        c = (kv_cache *)(intptr_t)0xdeadbeef;
        CHECK(kv_cache_create(2, -1, 4, &c) == KV_ERR_ARG && c == NULL,
              "create with a negative n_embd returns KV_ERR_ARG");
        c = (kv_cache *)(intptr_t)0xdeadbeef;
        CHECK(kv_cache_create(2, 8, 0, &c) == KV_ERR_ARG && c == NULL,
              "create with capacity 0 returns KV_ERR_ARG");
        CHECK(kv_cache_create(2, 8, 4, NULL) == KV_ERR_ARG,
              "create with a NULL out pointer returns KV_ERR_ARG rather than writing through it");
    }

    /* ---- destroy of NULL is safe ---- */
    kv_cache_destroy(NULL);
    CHECK(1, "destroy(NULL) returns without faulting");

    /* ---- row pointers address offset (pos x n_embd) within the layer ---- */
    {
        enum { L = 3, E = 8, CAP = 5 };
        kv_cache *c = NULL;
        CHECK(kv_cache_create(L, E, CAP, &c) == KV_OK, "cache for the addressing checks");
        int all = 1;
        for (int l = 0; l < L; ++l)
            for (int p = 0; p < CAP; ++p) {
                if (kv_cache_key_row(c, l, p) != kv_cache_keys(c, l) + (size_t)p * E) all = 0;
                if (kv_cache_value_row(c, l, p) != kv_cache_values(c, l) + (size_t)p * E) all = 0;
            }
        CHECK(all, "the key and value row for layer l, position p sit at offset p x n_embd "
                   "within that layer's buffer, for all %d layers x %d positions", L, CAP);
        CHECK(kv_cache_keys(c, L) == NULL && kv_cache_keys(c, -1) == NULL &&
              kv_cache_values(c, L) == NULL,
              "a layer index outside [0, n_layer) yields NULL, not an out-of-range pointer");
        CHECK(kv_cache_key_row(c, 0, CAP) == NULL && kv_cache_key_row(c, 0, -1) == NULL,
              "a position outside [0, capacity) yields NULL");

        /* Keys and values are SEPARATE storage: writing one must not move the
         * other. The layouts are identical, so a fused buffer would pass every
         * offset check above and fail this one. */
        float k[E], v[E];
        fill(k, E, 100.0f);
        fill(v, E, 900.0f);
        CHECK(kv_cache_write(c, 1, 2, k, v) == KV_OK, "write(layer 1, pos 2) succeeds");
        int ok = 1;
        for (int i = 0; i < E; ++i) {
            if (kv_cache_key_row(c, 1, 2)[i]   != 100.0f + (float)i) ok = 0;
            if (kv_cache_value_row(c, 1, 2)[i] != 900.0f + (float)i) ok = 0;
        }
        CHECK(ok, "the keys and the values of one position read back exactly as stored, "
                  "from separate buffers");

        /* Every other row is still the zero the constructor wrote. */
        int others_zero = 1;
        for (int l = 0; l < L; ++l)
            for (int p = 0; p < CAP; ++p) {
                if (l == 1 && p == 2) continue;
                for (int i = 0; i < E; ++i)
                    if (kv_cache_key_row(c, l, p)[i] != 0.0f ||
                        kv_cache_value_row(c, l, p)[i] != 0.0f) others_zero = 0;
            }
        CHECK(others_zero, "one store touched exactly one row of one layer and no other byte");
        kv_cache_destroy(c);
    }

    /* ---- append at the current length, advance, and the capacity refusal
     *      leaving every byte unchanged (compared against a copy) ---- */
    {
        enum { L = 2, E = 4, CAP = 3 };
        kv_cache *c = NULL;
        CHECK(kv_cache_create(L, E, CAP, &c) == KV_OK, "cache for the append checks");

        float k[E], v[E];
        for (int p = 0; p < CAP; ++p) {
            for (int l = 0; l < L; ++l) {
                fill(k, E, (float)(10 * (p + 1) + l));
                fill(v, E, (float)(50 * (p + 1) + l));
                CHECK(kv_cache_append(c, l, k, v) == KV_OK,
                      "append position %d of layer %d at the current length", p, l);
            }
            CHECK(kv_cache_advance(c) == KV_OK && kv_cache_length(c) == p + 1,
                  "advance moves the length to %d once every layer has stored", p + 1);
        }
        CHECK(kv_cache_length(c) == CAP, "the cache is full at length %d", CAP);

        /* The byte-for-byte snapshot the refusal is checked against. */
        const size_t per_layer = (size_t)CAP * E;
        float snap_k[L][CAP * E], snap_v[L][CAP * E];
        for (int l = 0; l < L; ++l) {
            memcpy(snap_k[l], kv_cache_keys(c, l),   per_layer * sizeof(float));
            memcpy(snap_v[l], kv_cache_values(c, l), per_layer * sizeof(float));
        }

        fill(k, E, 7777.0f);
        fill(v, E, 8888.0f);
        CHECK(kv_cache_append(c, 0, k, v) == KV_ERR_CAPACITY,
              "an append AT capacity returns KV_ERR_CAPACITY");
        CHECK(kv_cache_write(c, 0, CAP, k, v) == KV_ERR_CAPACITY,
              "a write at position == capacity returns KV_ERR_CAPACITY");
        CHECK(kv_cache_advance(c) == KV_ERR_CAPACITY,
              "advance at capacity returns KV_ERR_CAPACITY and does not move the length");
        CHECK(kv_cache_length(c) == CAP, "the length is still %d after the refusals", CAP);

        int unchanged = 1;
        for (int l = 0; l < L; ++l)
            for (size_t i = 0; i < per_layer; ++i) {
                if (kv_cache_keys(c, l)[i]   != snap_k[l][i]) unchanged = 0;
                if (kv_cache_values(c, l)[i] != snap_v[l][i]) unchanged = 0;
            }
        CHECK(unchanged, "the refused append and write left EVERY byte of the cache unchanged, "
                         "compared against a copy taken before them");

        /* set-length then re-append overwrites exactly the expected row ---- */
        CHECK(kv_cache_set_length(c, CAP - 1) == KV_OK && kv_cache_length(c) == CAP - 1,
              "set_length restores the counter to %d", CAP - 1);
        fill(k, E, 4242.0f);
        fill(v, E, 2424.0f);
        CHECK(kv_cache_append(c, 1, k, v) == KV_OK,
              "an append after the restoration is accepted again");
        int row_ok = 1, rest_ok = 1;
        for (int i = 0; i < E; ++i) {
            if (kv_cache_key_row(c, 1, CAP - 1)[i]   != 4242.0f + (float)i) row_ok = 0;
            if (kv_cache_value_row(c, 1, CAP - 1)[i] != 2424.0f + (float)i) row_ok = 0;
        }
        for (int l = 0; l < L; ++l)
            for (size_t i = 0; i < per_layer; ++i) {
                if (l == 1 && i >= (size_t)(CAP - 1) * E) continue;
                if (kv_cache_keys(c, l)[i]   != snap_k[l][i]) rest_ok = 0;
                if (kv_cache_values(c, l)[i] != snap_v[l][i]) rest_ok = 0;
            }
        CHECK(row_ok, "the re-append overwrote exactly row %d of layer 1", CAP - 1);
        CHECK(rest_ok, "and no other byte of any layer moved");

        CHECK(kv_cache_set_length(c, CAP + 1) == KV_ERR_CAPACITY,
              "set_length beyond capacity is refused");
        CHECK(kv_cache_set_length(c, -1) == KV_ERR_ARG,
              "a negative set_length is refused");
        CHECK(kv_cache_set_length(c, 0) == KV_OK && kv_cache_length(c) == 0,
              "set_length(0) is accepted: a cache can be emptied for a new prefill");
        kv_cache_destroy(c);
    }

    /* ---- bad arguments on the write path ---- */
    {
        kv_cache *c = NULL;
        kv_cache_create(2, 4, 2, &c);
        float k[4] = {1, 2, 3, 4}, v[4] = {5, 6, 7, 8};
        CHECK(kv_cache_write(c, 2, 0, k, v) == KV_ERR_ARG,
              "a write to a layer index == n_layer returns KV_ERR_ARG");
        CHECK(kv_cache_write(c, -1, 0, k, v) == KV_ERR_ARG,
              "a write to a negative layer returns KV_ERR_ARG");
        CHECK(kv_cache_write(c, 0, -1, k, v) == KV_ERR_ARG,
              "a write to a negative position returns KV_ERR_ARG");
        CHECK(kv_cache_write(c, 0, 0, NULL, v) == KV_ERR_ARG &&
              kv_cache_write(c, 0, 0, k, NULL) == KV_ERR_ARG,
              "a NULL key or value pointer returns KV_ERR_ARG");
        CHECK(kv_cache_write(NULL, 0, 0, k, v) == KV_ERR_ARG &&
              kv_cache_append(NULL, 0, k, v) == KV_ERR_ARG &&
              kv_cache_advance(NULL) == KV_ERR_ARG &&
              kv_cache_set_length(NULL, 0) == KV_ERR_ARG,
              "every entry point refuses a NULL cache rather than faulting");
        CHECK(kv_cache_length(NULL) == 0 && kv_cache_capacity(NULL) == 0 &&
              kv_cache_footprint_bytes(NULL) == 0u && kv_cache_keys(NULL, 0) == NULL,
              "the accessors answer for a NULL cache without faulting");
        kv_cache_destroy(c);
    }

    /* ---- the error strings are distinct, so a reported failure names itself ---- */
    CHECK(strcmp(kv_strerror(KV_OK), kv_strerror(KV_ERR_ARG)) != 0 &&
          strcmp(kv_strerror(KV_ERR_ARG), kv_strerror(KV_ERR_CAPACITY)) != 0,
          "kv_strerror distinguishes ok, the argument error and the capacity error");

    TIE_SUMMARY("test_kv_cache");
}
