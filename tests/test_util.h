/* test_util.h -- the smallest assertion helper that does the job.
 *
 * These are STRUCTURAL tests. They assert that a benchmark is correctly
 * constructed, not that the machine is fast, so they run in the offline gate
 * before any timed measurement and do not require an idle machine.
 */
#ifndef TIE_TEST_UTIL_H
#define TIE_TEST_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tie_checks   = 0;
static int tie_failures = 0;

/* Tests that exercise a benchmark would otherwise drop files into
 * bench/results/, which holds deliverables. Point them at the working
 * directory (the build tree under ctest) instead. */
static void tie_redirect_results(void)
{
#if defined(_WIN32)
    _putenv_s("BENCH_RESULTS_DIR", ".");
#else
    setenv("BENCH_RESULTS_DIR", ".", 1);
#endif
}

/* Reads a whole text file; caller frees. Returns NULL if it cannot be read. */
static char *tie_read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);
    return buf;
}

/* Counts non-overlapping occurrences of needle in haystack. */
static int tie_count(const char *haystack, const char *needle)
{
    int c = 0;
    size_t n = strlen(needle);
    for (const char *p = haystack; (p = strstr(p, needle)) != NULL; p += n) ++c;
    return c;
}

#define CHECK(cond, ...) do {                                                 \
        ++tie_checks;                                                         \
        if (cond) { printf("  ok   "); printf(__VA_ARGS__); printf("\n"); }   \
        else { ++tie_failures;                                                \
               printf("  FAIL "); printf(__VA_ARGS__);                        \
               printf("   [%s:%d]\n", __FILE__, __LINE__); }                  \
    } while (0)

#define CHECK_NEAR(actual, expected, tol, ...) do {                           \
        double a_ = (double)(actual), e_ = (double)(expected);                 \
        ++tie_checks;                                                         \
        if (fabs(a_ - e_) <= (tol)) {                                         \
            printf("  ok   "); printf(__VA_ARGS__);                           \
            printf(" (%.12g)\n", a_);                                         \
        } else { ++tie_failures;                                              \
            printf("  FAIL "); printf(__VA_ARGS__);                           \
            printf(" expected %.12g, got %.12g   [%s:%d]\n",                  \
                   e_, a_, __FILE__, __LINE__); }                             \
    } while (0)

#define TIE_SUMMARY(name) do {                                                \
        printf("%s: %d checks, %d failures\n", (name), tie_checks, tie_failures); \
        return tie_failures == 0 ? 0 : 1;                                     \
    } while (0)

#endif /* TIE_TEST_UTIL_H */
