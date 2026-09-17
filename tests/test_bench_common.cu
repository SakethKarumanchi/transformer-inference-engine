/* Unit test for the shared measurement harness. */
#include "test_util.h"
#include "bench_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ---- a minimal JSON validator -----------------------------------------
 * Enough of RFC 8259 to prove the emitted document is well formed. Written
 * here rather than pulled in as a dependency so the offline gate has no
 * third-party surface. Returns a pointer past the parsed value, or NULL. */
static const char *jv_value(const char *p);

static const char *jv_ws(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    return p;
}

static const char *jv_string(const char *p)
{
    if (*p != '"') return NULL;
    ++p;
    while (*p && *p != '"') {
        if (*p == '\\') {
            ++p;
            if (*p == 'u') { for (int i = 0; i < 4; ++i) if (!isxdigit((unsigned char)*++p)) return NULL; }
            else if (!strchr("\"\\/bfnrt", *p)) return NULL;
        }
        ++p;
    }
    return (*p == '"') ? p + 1 : NULL;
}

static const char *jv_number(const char *p)
{
    const char *s = p;
    if (*p == '-') ++p;
    if (!isdigit((unsigned char)*p)) return NULL;
    while (isdigit((unsigned char)*p)) ++p;
    if (*p == '.') { ++p; if (!isdigit((unsigned char)*p)) return NULL;
                     while (isdigit((unsigned char)*p)) ++p; }
    if (*p == 'e' || *p == 'E') { ++p; if (*p == '+' || *p == '-') ++p;
                                  if (!isdigit((unsigned char)*p)) return NULL;
                                  while (isdigit((unsigned char)*p)) ++p; }
    return (p > s) ? p : NULL;
}

static const char *jv_value(const char *p)
{
    p = jv_ws(p);
    switch (*p) {
        case '"': return jv_string(p);
        case '{': {
            p = jv_ws(p + 1);
            if (*p == '}') return p + 1;
            for (;;) {
                p = jv_ws(p);
                p = jv_string(p);      if (!p) return NULL;
                p = jv_ws(p);
                if (*p != ':') return NULL;
                p = jv_value(p + 1);   if (!p) return NULL;
                p = jv_ws(p);
                if (*p == ',') { ++p; continue; }
                return (*p == '}') ? p + 1 : NULL;
            }
        }
        case '[': {
            p = jv_ws(p + 1);
            if (*p == ']') return p + 1;
            for (;;) {
                p = jv_value(p);       if (!p) return NULL;
                p = jv_ws(p);
                if (*p == ',') { ++p; continue; }
                return (*p == ']') ? p + 1 : NULL;
            }
        }
        case 't': return strncmp(p, "true", 4)  == 0 ? p + 4 : NULL;
        case 'f': return strncmp(p, "false", 5) == 0 ? p + 5 : NULL;
        case 'n': return strncmp(p, "null", 4)  == 0 ? p + 4 : NULL;
        default:  return jv_number(p);
    }
}

static int json_is_valid(const char *text)
{
    const char *end = jv_value(text);
    if (!end) return 0;
    end = jv_ws(end);
    return *end == '\0';
}

/* ---- a body whose "timing" is just the iteration index ----------------- */
static int g_calls = 0;
static double counting_body(void *ctx, int iteration)
{
    (void)ctx;
    ++g_calls;
    return (double)iteration;
}

int main(void)
{
    printf("test_bench_common\n");

    /* 1. statistics on a fixed sample array with a hand-computed answer.
     *    {10,12,23,23,16,23,21,16}: mean 18, median 18.5, min 10, max 23,
     *    sum of squared deviations 192, sample variance 192/7,
     *    sample stddev sqrt(192/7) = 5.237229365663817 */
    {
        const double fixed[8] = { 10, 12, 23, 23, 16, 23, 21, 16 };
        bench_stats s = bench_compute_stats(fixed, 8);
        CHECK_NEAR(s.mean,   18.0, 1e-12, "mean of the fixed array");
        CHECK_NEAR(s.median, 18.5, 1e-12, "median of the fixed array (even n)");
        CHECK_NEAR(s.min,    10.0, 1e-12, "min of the fixed array");
        CHECK_NEAR(s.max,    23.0, 1e-12, "max of the fixed array");
        CHECK_NEAR(s.stddev, 5.237229365663817, 1e-12,
                   "sample stddev (divisor n-1) of the fixed array");
        CHECK(s.n == 8, "n is reported as 8");
        CHECK(s.valid == 0 && strstr(s.invalid_reason, "below the protocol minimum") != NULL,
              "8 samples is INVALID: below the %d-sample protocol minimum", BENCH_MIN_SAMPLES);
    }

    /* 2. the 5% rule, both sides of the threshold. 20 samples split evenly
     *    between 100-d and 100+d give median 100 and stddev d*sqrt(20/19). */
    {
        const double k = 1.0259783520851541;   /* sqrt(20/19) */
        double under[20], over[20];
        for (int i = 0; i < 20; ++i) {
            under[i] = (i < 10) ? 100.0 - 4.7 : 100.0 + 4.7;
            over[i]  = (i < 10) ? 100.0 - 5.0 : 100.0 + 5.0;
        }
        bench_stats su = bench_compute_stats(under, 20);
        bench_stats so = bench_compute_stats(over,  20);

        CHECK_NEAR(su.stddev, 4.7 * k, 1e-12, "stddev of the just-under set");
        CHECK_NEAR(su.stddev_pct_of_median, 4.7 * k, 1e-9,
                   "stddev as a percentage of the median, just under");
        CHECK(su.stddev_pct_of_median < BENCH_MAX_STDDEV_PCT,
              "just-under set is below the %.1f%% threshold", (double)BENCH_MAX_STDDEV_PCT);
        CHECK(su.valid == 1, "a sample set just under the threshold is returned VALID");

        CHECK(so.stddev_pct_of_median > BENCH_MAX_STDDEV_PCT,
              "over set is above the %.1f%% threshold", (double)BENCH_MAX_STDDEV_PCT);
        CHECK(so.valid == 0, "a sample set over the threshold is returned INVALID");
        CHECK(strstr(so.invalid_reason, "protocol limit") != NULL,
              "the INVALID verdict states the rule that was broken");
        CHECK(so.median > 0.0 && so.stddev > 0.0,
              "an INVALID run still carries its computed statistics, not zeros");
    }

    /* 3. warmup iterations are excluded from the reported statistics. */
    {
        double samples[20];
        int eff_w = 0, eff_s = 0;
        g_calls = 0;
        bench_stats s = bench_run(counting_body, NULL, 5, 20, samples, &eff_w, &eff_s);
        CHECK(eff_w == 5 && eff_s == 20, "effective warmup 5 and samples 20 reported back");
        CHECK(g_calls == 25, "the body ran warmup+samples = 25 times");
        CHECK(s.n == 20, "only the 20 timed samples reach the statistics");
        CHECK_NEAR(s.min, 5.0, 1e-12,
                   "the smallest reported sample is iteration 5, so 0..4 were discarded");
        CHECK_NEAR(s.max, 24.0, 1e-12, "the largest reported sample is iteration 24");
        CHECK(samples[0] == 5.0, "the first retained sample is the first post-warmup one");
    }

    /* 4. protocol minimums are enforced upward, never downward. */
    {
        double samples[BENCH_MIN_SAMPLES];
        int eff_w = 0, eff_s = 0;
        g_calls = 0;
        bench_run(counting_body, NULL, 1, 3, samples, &eff_w, &eff_s);
        CHECK(eff_w >= BENCH_MIN_WARMUP && eff_s >= BENCH_MIN_SAMPLES,
              "a caller asking for fewer than the minimums is raised to them");
    }

    /* 5. the emitted JSON parses, and carries the raw per-sample array. */
    {
        double raw[20];
        for (int i = 0; i < 20; ++i) raw[i] = 1.0 + 0.001 * i;
        bench_stats s = bench_compute_stats(raw, 20);

        bench_kv_num nums[] = { { "some_counter", 42.5 } };
        bench_kv_str strs[] = { { "note", "quote \" backslash \\ newline \n" } };
        bench_record rec;
        memset(&rec, 0, sizeof rec);
        rec.benchmark = "unit_test";
        rec.configuration = "synthetic";
        rec.units = "ms";
        rec.value = s.median;
        rec.warmup = 5;
        rec.samples_requested = 20;
        rec.samples_ms = raw;
        rec.n_samples = 20;
        rec.stats = s;
        rec.meta_num = nums; rec.n_meta_num = 1;
        rec.meta_str = strs; rec.n_meta_str = 1;

        const char *path = "test_bench_common_output.json";
        FILE *f = fopen(path, "wb");
        CHECK(f != NULL, "the test can open a scratch file for the JSON");
        if (f) {
            CHECK(bench_write_results_fp(f, &rec, 1) == 0, "bench_write_results_fp succeeds");
            fclose(f);

            f = fopen(path, "rb");
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = (char *)malloc((size_t)n + 1);
            size_t got = fread(buf, 1, (size_t)n, f);
            buf[got] = '\0';
            fclose(f);

            CHECK(json_is_valid(buf), "the emitted document parses as valid JSON");
            CHECK(strstr(buf, "\"raw_samples_ms\":[") != NULL,
                  "the emitted document contains the raw per-sample array");
            CHECK(strstr(buf, "1.0009999999999999") != NULL ||
                  strstr(buf, "1.001") != NULL,
                  "a specific raw sample value appears in the array");
            CHECK(strstr(buf, "\"git_commit\"") != NULL,
                  "the document records the git commit hash");
            CHECK(strstr(buf, "\"cxx_flags\"") != NULL &&
                  strstr(buf, "\"cuda_flags\"") != NULL,
                  "the document records the compiler flags");
            CHECK(strstr(buf, "\"run_timestamp_utc\"") != NULL,
                  "the document records a timestamp");
            CHECK(strstr(buf, "\"device\"") != NULL, "the document records the device");
            CHECK(strstr(buf, "\"stage\":\"stage-0\"") != NULL,
                  "the document records the stage identifier");

            /* Count the elements of the raw array to prove none were dropped. */
            const char *a = strstr(buf, "\"raw_samples_ms\":[");
            int commas = 0;
            if (a) { a += strlen("\"raw_samples_ms\":[");
                     for (; *a && *a != ']'; ++a) if (*a == ',') ++commas; }
            CHECK(commas == 19, "the raw array holds all 20 samples");

            free(buf);
            remove(path);
        }
    }

    /* 6. the protocol constants are the ones the contract fixes. */
    CHECK(BENCH_MIN_WARMUP == 5,        "BENCH_MIN_WARMUP is 5");
    CHECK(BENCH_MIN_SAMPLES == 20,      "BENCH_MIN_SAMPLES is 20");
    CHECK(BENCH_MAX_STDDEV_PCT == 5.0,  "BENCH_MAX_STDDEV_PCT is 5.0");

    TIE_SUMMARY("test_bench_common");
}
