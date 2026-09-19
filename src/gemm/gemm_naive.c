/* gemm_naive.c -- the Stage 2 baseline matmul. Deliberately unoptimized.
 *
 * This file is the denominator of every later GEMM figure in this project, so
 * what it does NOT do matters as much as what it does. Each omission below is
 * owned by a named later stage and is left undone here on purpose; taking any
 * of them now would silently consume part of that stage's measured gain.
 *
 *   loop order       ijk -- i outer, j middle, k inner, dot product inner.
 *                    Interchanging to ikj is a cache optimization and belongs
 *                    to Stage 5, not here.
 *   blocking/tiling  none. Stage 5.
 *   vectorization    none, and the generated code is checked rather than
 *                    trusted: the build emits an assembly listing for this
 *                    translation unit and tests/test_gemm_naive.c asserts the
 *                    inner loops contain no packed AVX arithmetic. Stage 6.
 *   threading        none. Single threaded for the life of the CPU stages.
 *   accumulator      ONE. Several independent accumulators would hide the FMA
 *                    dependency chain; that is an optimization with a
 *                    measurable effect and is not this stage's to take.
 *
 * ACCESS PATTERN, stated because the baseline's cost is mostly here. In
 * gemm_naive the inner loop walks B down a column, touching a fresh 64-byte
 * line for every 4 bytes used. In gemm_naive_bt both operands are walked
 * contiguously. The two are reported separately by the isolated GEMM
 * configuration in the Stage 2 benchmark for exactly that reason -- they are
 * different memory behaviours and averaging them would hide the difference.
 */
#include "gemm.h"

void gemm_naive(int M, int N, int K,
                const float *A, int lda,
                const float *B, int ldb,
                float *C, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[(size_t)i * lda + k] * B[(size_t)k * ldb + j];
            }
            C[(size_t)i * ldc + j] = sum;
        }
    }
}

void gemm_naive_bt(int M, int N, int K,
                   const float *A,  int lda,
                   const float *Bt, int ldbt,
                   float *C, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[(size_t)i * lda + k] * Bt[(size_t)j * ldbt + k];
            }
            C[(size_t)i * ldc + j] = sum;
        }
    }
}

const gemm_impl gemm_impl_naive = { "gemm_naive", gemm_naive, gemm_naive_bt };
