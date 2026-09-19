/* gemm.h -- the single matrix-multiply interface the forward pass calls.
 *
 * WHY THIS IS A SEPARATE HEADER AND A SEPARATE TRANSLATION UNIT.
 * Stage 5 (cache blocking) and Stage 6 (SIMD) replace the matmul and nothing
 * else. A matmul written inline inside the forward pass makes that swap a
 * rewrite and makes the later measurements incomparable with the Stage 2
 * baseline. Everything below is therefore fixed at Stage 2 and is what those
 * stages implement against: the operand layouts, the leading dimensions, the
 * dimension order, and the two operations. gemm_naive.c (Stage 2),
 * gemm_blocked.c (Stage 5) and gemm_simd.c (Stage 6) each provide their own
 * symbols; a caller selects one through gemm_impl below, so substituting an
 * implementation touches no arithmetic in the forward pass.
 *
 * LAYOUT CONVENTION -- ROW MAJOR THROUGHOUT, with explicit leading dimensions
 * so a caller may pass a sub-block of a larger buffer (attention does exactly
 * that, one head at a time, out of a [tokens, n_embd] activation matrix).
 *
 *   A is M x K, element (i, k) at A[(size_t)i * lda + k], lda >= K.
 *   C is M x N, element (i, j) at C[(size_t)i * ldc + j], ldc >= N.
 *
 * TWO OPERATIONS, because the weight file forces both. Every 2-D weight in
 * models/gpt2/model.safetensors is stored [input, output] (Stage 1 pinned that
 * by bias length for every rectangular tensor). That is exactly the B layout of
 * gemm_f32. The language-model head is the exception: it is TIED to
 * wte.weight, which is stored [vocab, n_embd] = [output, input], and tying
 * means using that same storage rather than a transposed copy of it. The
 * transposed-B form exists for that case and for attention's Q x K^T, not as a
 * convenience.
 *
 *   gemm_f32:    C = A * B     B is K x N, element (k, j) at B[(size_t)k * ldb + j]
 *   gemm_f32_bt: C = A * Bt^T  Bt is N x K, element (j, k) at Bt[(size_t)j * ldbt + k]
 *
 * C is WRITTEN, not accumulated: an implementation must not require C to be
 * zeroed first, and must produce the same result for any prior contents.
 *
 * No bias, no activation, no transpose of A. Bias addition and the elementwise
 * work stay in the forward pass, so what these stages measure is a matmul and
 * not a fused kernel that changed shape between stages.
 *
 * ZERO-SIZED CASES: any of M, N, K equal to zero is legal and writes nothing
 * (K == 0 writes C = 0 over the M x N block, which is the empty sum).
 * Negative dimensions are a programming error and are not checked.
 */
#ifndef TIE_GEMM_H
#define TIE_GEMM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The two operation signatures. Stages 5 and 6 declare their own symbols with
 * exactly these types and are substituted through gemm_impl. */
typedef void (*gemm_fn)(int M, int N, int K,
                        const float *A,  int lda,
                        const float *B,  int ldb,
                        float *C, int ldc);

typedef void (*gemm_bt_fn)(int M, int N, int K,
                           const float *A,  int lda,
                           const float *Bt, int ldbt,
                           float *C, int ldc);

/* An implementation is one pair plus a name recorded in the results file, so a
 * measurement always says which matmul produced it. */
typedef struct {
    const char *name;
    gemm_fn     mul;
    gemm_bt_fn  mul_bt;
} gemm_impl;

/* ---- Stage 2: the naive implementation ---------------------------------
 * Triple nested loop, ijk order (i outer, j middle, k inner), dot-product
 * inner loop, one accumulator, scalar, no blocking, no tiling, no intrinsics,
 * no vectorizing pragma, single threaded. See gemm_naive.c for why each of
 * those is deliberate and which later stage owns changing it. */
void gemm_naive(int M, int N, int K,
                const float *A, int lda,
                const float *B, int ldb,
                float *C, int ldc);

void gemm_naive_bt(int M, int N, int K,
                   const float *A,  int lda,
                   const float *Bt, int ldbt,
                   float *C, int ldc);

/* The naive pair, for a caller that wants to name it explicitly. */
extern const gemm_impl gemm_impl_naive;

#ifdef __cplusplus
}
#endif

#endif /* TIE_GEMM_H */
