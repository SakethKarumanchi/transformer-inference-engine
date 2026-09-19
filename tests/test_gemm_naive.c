/* Unit test for src/gemm/gemm_naive.c.
 *
 * Structural and numerical, and it times nothing -- it runs in the offline gate
 * before any measurement and does not need an idle machine.
 *
 * The last check is the one that matters most to the project: it asserts from
 * the GENERATED CODE, not from a comment, that the matmul inner loops were not
 * auto-vectorized. The frozen host flags include /O2 and /arch:AVX2, and if
 * MSVC vectorized this loop then Stage 2 is not measuring scalar code, the
 * baseline is wrong, and Stage 6 has nothing left to demonstrate. The build
 * emits an assembly listing for the translation unit with exactly the frozen
 * flags and this test reads it, in the same spirit as the two Stage 0 tests
 * that read emitted PTX rather than trusting a comment.
 */
#include "test_util.h"
#include "gemm/gemm.h"

#include <stdint.h>

#ifndef TIE_ASM_FILE
#define TIE_ASM_FILE "gemm_naive.asm"
#endif

/* An independent reference, written in the transposed loop order so a shared
 * mistake in loop nesting cannot cancel out: k outermost, accumulating into C.
 * Its arithmetic is deliberately not gemm_naive's. */
static void ref_mul(int M, int N, int K, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) C[(size_t)i * ldc + j] = 0.0f;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[(size_t)i * ldc + j] += A[(size_t)i * lda + k] * B[(size_t)k * ldb + j];
}

static void ref_mul_bt(int M, int N, int K, const float *A, int lda,
                       const float *Bt, int ldbt, float *C, int ldc)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) C[(size_t)i * ldc + j] = 0.0f;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[(size_t)i * ldc + j] += A[(size_t)i * lda + k] * Bt[(size_t)j * ldbt + k];
}

static float lcg_next(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (float)((double)(*s >> 8) / 8388608.0 - 1.0);
}

static int close_enough(float a, float b)
{
    float d = a - b;
    if (d < 0) d = -d;
    float m = (a < 0 ? -a : a) + (b < 0 ? -b : b);
    return d <= 1e-4f * (m + 1.0f);
}

int main(void)
{
    printf("test_gemm_naive\n");

    /* 1. A hand-computed product, element for element.
     *    [1 2 3]   [ 7  8]     [ 58  64]
     *    [4 5 6] x [ 9 10]  =  [139 154]
     *              [11 12]                                              */
    {
        float A[6] = {1, 2, 3, 4, 5, 6};
        float B[6] = {7, 8, 9, 10, 11, 12};
        float C[4] = {-1, -1, -1, -1};
        gemm_naive(2, 2, 3, A, 3, B, 2, C, 2);
        CHECK(C[0] == 58.0f,  "hand-computed C[0,0] = 58 (got %g)", (double)C[0]);
        CHECK(C[1] == 64.0f,  "hand-computed C[0,1] = 64 (got %g)", (double)C[1]);
        CHECK(C[2] == 139.0f, "hand-computed C[1,0] = 139 (got %g)", (double)C[2]);
        CHECK(C[3] == 154.0f, "hand-computed C[1,1] = 154 (got %g)", (double)C[3]);

        /* The same product through the transposed-B form, with B stored [N, K]. */
        float Bt[6] = {7, 9, 11, 8, 10, 12};
        float D[4] = {-1, -1, -1, -1};
        gemm_naive_bt(2, 2, 3, A, 3, Bt, 3, D, 2);
        CHECK(D[0] == 58.0f && D[1] == 64.0f && D[2] == 139.0f && D[3] == 154.0f,
              "gemm_naive_bt reproduces the same hand-computed product");

        /* C is written, never accumulated: prior contents cannot survive. */
        C[0] = 1e9f; C[3] = -1e9f;
        gemm_naive(2, 2, 3, A, 3, B, 2, C, 2);
        CHECK(C[0] == 58.0f && C[3] == 154.0f,
              "C is written rather than accumulated (prior contents discarded)");
    }

    /* 2. The identity leaves an operand unchanged, on both sides. */
    {
        enum { N = 7 };
        float I[N * N], A[N * N], C[N * N];
        uint32_t s = 12345u;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                I[i * N + j] = (i == j) ? 1.0f : 0.0f;
                A[i * N + j] = lcg_next(&s);
            }
        gemm_naive(N, N, N, A, N, I, N, C, N);
        int ok = 1;
        for (int i = 0; i < N * N; ++i) if (C[i] != A[i]) ok = 0;
        CHECK(ok, "A x I == A, exactly, for a %dx%d operand", N, N);

        gemm_naive(N, N, N, I, N, A, N, C, N);
        ok = 1;
        for (int i = 0; i < N * N; ++i) if (C[i] != A[i]) ok = 0;
        CHECK(ok, "I x A == A, exactly");

        gemm_naive_bt(N, N, N, A, N, I, N, C, N);
        ok = 1;
        for (int i = 0; i < N * N; ++i) if (C[i] != A[i]) ok = 0;
        CHECK(ok, "A x I^T == A, exactly (the identity is its own transpose)");
    }

    /* 3. Non-square shapes, both orientations, against the independent
     *    reference. 5x3 times 3x8, and then 8x3 times 3x5, so a transposed
     *    dimension mix-up cannot pass both. */
    {
        enum { M = 5, K = 3, N = 8 };
        float A[M * K], B[K * N], C[M * N], R[M * N];
        uint32_t s = 999u;
        for (int i = 0; i < M * K; ++i) A[i] = lcg_next(&s);
        for (int i = 0; i < K * N; ++i) B[i] = lcg_next(&s);
        gemm_naive(M, N, K, A, K, B, N, C, N);
        ref_mul(M, N, K, A, K, B, N, R, N);
        int ok = 1;
        for (int i = 0; i < M * N; ++i) if (!close_enough(C[i], R[i])) ok = 0;
        CHECK(ok, "%dx%d times %dx%d matches the independent reference", M, K, K, N);

        float A2[N * K], B2[K * M], C2[N * M], R2[N * M];
        for (int i = 0; i < N * K; ++i) A2[i] = lcg_next(&s);
        for (int i = 0; i < K * M; ++i) B2[i] = lcg_next(&s);
        gemm_naive(N, M, K, A2, K, B2, M, C2, M);
        ref_mul(N, M, K, A2, K, B2, M, R2, M);
        ok = 1;
        for (int i = 0; i < N * M; ++i) if (!close_enough(C2[i], R2[i])) ok = 0;
        CHECK(ok, "the reversed shape %dx%d times %dx%d also matches", N, K, K, M);
    }

    /* 4. Leading dimensions larger than the logical width: a sub-block of a
     *    wider buffer, which is exactly how attention passes one head out of a
     *    [tokens, n_embd] activation matrix. */
    {
        enum { M = 4, K = 3, N = 2, LDA = 11, LDB = 9, LDC = 7 };
        float A[M * LDA], B[K * LDB], C[M * LDC], R[M * LDC];
        uint32_t s = 4242u;
        for (int i = 0; i < M * LDA; ++i) A[i] = lcg_next(&s);
        for (int i = 0; i < K * LDB; ++i) B[i] = lcg_next(&s);
        for (int i = 0; i < M * LDC; ++i) { C[i] = 12345.0f; R[i] = 12345.0f; }
        gemm_naive(M, N, K, A, LDA, B, LDB, C, LDC);
        ref_mul(M, N, K, A, LDA, B, LDB, R, LDC);
        int ok = 1, untouched = 1;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < LDC; ++j) {
                float got = C[i * LDC + j];
                if (j < N) { if (!close_enough(got, R[i * LDC + j])) ok = 0; }
                else if (got != 12345.0f) untouched = 0;
            }
        CHECK(ok, "a sub-block with lda/ldb/ldc larger than M/N/K is correct");
        CHECK(untouched, "nothing outside the N columns of C is written");
    }

    /* 5. A rectangular case at one of the model's real shapes, against the
     *    independent reference: 8 tokens through the attention output
     *    projection, 768 x 768. */
    {
        const int M = 8, K = 768, N = 768;
        float *A = (float *)malloc((size_t)M * K * sizeof(float));
        float *B = (float *)malloc((size_t)K * N * sizeof(float));
        float *C = (float *)malloc((size_t)M * N * sizeof(float));
        float *R = (float *)malloc((size_t)M * N * sizeof(float));
        uint32_t s = 7u;
        for (size_t i = 0; i < (size_t)M * K; ++i) A[i] = lcg_next(&s);
        for (size_t i = 0; i < (size_t)K * N; ++i) B[i] = lcg_next(&s);
        gemm_naive(M, N, K, A, K, B, N, C, N);
        ref_mul(M, N, K, A, K, B, N, R, N);
        int ok = 1;
        for (size_t i = 0; i < (size_t)M * N; ++i) if (!close_enough(C[i], R[i])) ok = 0;
        CHECK(ok, "a real model shape, %d x %d times %d x %d, matches the reference",
              M, K, K, N);
        free(A); free(B); free(C); free(R);
    }

    /* 6. Zero-sized and single-element edge cases. */
    {
        float A[1] = {3.0f}, B[1] = {4.0f}, C[1] = {-1.0f};
        gemm_naive(1, 1, 1, A, 1, B, 1, C, 1);
        CHECK(C[0] == 12.0f, "the 1x1x1 case is 3 x 4 = 12 (got %g)", (double)C[0]);

        C[0] = 5.0f;
        gemm_naive(0, 1, 1, A, 1, B, 1, C, 1);
        CHECK(C[0] == 5.0f, "M == 0 writes nothing");
        gemm_naive(1, 0, 1, A, 1, B, 1, C, 1);
        CHECK(C[0] == 5.0f, "N == 0 writes nothing");
        gemm_naive(1, 1, 0, A, 1, B, 1, C, 1);
        CHECK(C[0] == 0.0f, "K == 0 writes the empty sum, zero (got %g)", (double)C[0]);
        C[0] = 5.0f;
        gemm_naive_bt(1, 1, 0, A, 1, B, 1, C, 1);
        CHECK(C[0] == 0.0f, "K == 0 writes zero in the transposed form too");
    }

    /* 7. Output elements are independent: computing the whole block at once and
     *    computing each element on its own as a 1x1 must agree exactly. A
     *    blocked or vectorized implementation must preserve this, so the check
     *    is written once here and outlives this stage. */
    {
        enum { M = 6, K = 5, N = 4 };
        float A[M * K], B[K * N], C[M * N], one;
        uint32_t s = 20260918u;
        for (int i = 0; i < M * K; ++i) A[i] = lcg_next(&s);
        for (int i = 0; i < K * N; ++i) B[i] = lcg_next(&s);
        gemm_naive(M, N, K, A, K, B, N, C, N);
        int ok = 1;
        for (int i = M - 1; i >= 0; --i)          /* reverse order deliberately */
            for (int j = N - 1; j >= 0; --j) {
                gemm_naive(1, 1, K, A + (size_t)i * K, K, B + j, N, &one, 1);
                if (one != C[i * N + j]) ok = 0;
            }
        CHECK(ok, "each output element computed alone equals the block result, bit for bit");
    }

    /* 8. THE SCALAR-CODEGEN CHECK, read from the generated assembly.
     *    Packed AVX arithmetic on ymm registers is what auto-vectorization of
     *    the inner loop would produce. Scalar FMA and multiply-add (vfmadd*ss,
     *    vmulss, vaddss) are what this stage must be measuring. */
    {
        char *asmtext = tie_read_file(TIE_ASM_FILE);
        CHECK(asmtext != NULL, "the generated assembly listing was emitted and is readable "
              "[%s]", TIE_ASM_FILE);
        if (asmtext) {
            int ymm = tie_count(asmtext, "ymm");
            int packed = 0;
            static const char *packed_ops[] = {
                "vfmadd132ps", "vfmadd213ps", "vfmadd231ps",
                "vmulps", "vaddps", "vsubps", "mulps", "addps", "vdpps", "vhaddps"
            };
            for (size_t i = 0; i < sizeof packed_ops / sizeof *packed_ops; ++i)
                packed += tie_count(asmtext, packed_ops[i]);
            int scalar_fma = tie_count(asmtext, "vfmadd231ss") + tie_count(asmtext, "vfmadd213ss")
                           + tie_count(asmtext, "vfmadd132ss") + tie_count(asmtext, "vmulss")
                           + tie_count(asmtext, "vaddss") + tie_count(asmtext, "mulss")
                           + tie_count(asmtext, "addss");
            CHECK(ymm == 0, "no ymm register appears anywhere in gemm_naive's generated code "
                  "(count %d)", ymm);
            CHECK(packed == 0, "no packed floating-point arithmetic appears (count %d)", packed);
            CHECK(scalar_fma > 0, "scalar floating-point arithmetic IS present (count %d), so "
                  "the loop was compiled rather than eliminated", scalar_fma);
            free(asmtext);
        }
    }

    CHECK(gemm_impl_naive.mul == gemm_naive && gemm_impl_naive.mul_bt == gemm_naive_bt,
          "the substitutable gemm_impl points at this implementation, named \"%s\"",
          gemm_impl_naive.name);

    TIE_SUMMARY("test_gemm_naive");
}
