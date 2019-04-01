/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <magma_v2.h>

namespace MLCommon {
namespace LinAlg {
template <typename T>
void
magmablas_gemm_batched(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        T alpha,
        T const * const * dA_array, magma_int_t ldda,
        T const * const * dB_array, magma_int_t lddb,
        T beta,
        T **dC_array, magma_int_t lddc,
        magma_int_t batchCount, magma_queue_t queue );

template <>
inline void
magmablas_gemm_batched(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        float alpha,
        float const * const * dA_array, magma_int_t ldda,
        float const * const * dB_array, magma_int_t lddb,
        float beta,
        float **dC_array, magma_int_t lddc,
        magma_int_t batchCount, magma_queue_t queue )

{
        return magmablas_sgemm_batched( transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue);
}

template <>
inline void
magmablas_gemm_batched(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        double alpha,
        double const * const * dA_array, magma_int_t ldda,
        double const * const * dB_array, magma_int_t lddb,
        double beta,
        double **dC_array, magma_int_t lddc,
        magma_int_t batchCount, magma_queue_t queue )

{
        return magmablas_dgemm_batched( transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue);
}

template <typename T>
magma_int_t
magma_potrf_batched(
        magma_uplo_t uplo, magma_int_t n,
        T **dA_array, magma_int_t lda,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

template <>
inline magma_int_t
magma_potrf_batched(
        magma_uplo_t uplo, magma_int_t n,
        float **dA_array, magma_int_t lda,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magma_spotrf_batched( uplo, n, dA_array, lda, info_array, batchCount, queue);
}

template <>
inline magma_int_t
magma_potrf_batched(
        magma_uplo_t uplo, magma_int_t n,
        double **dA_array, magma_int_t lda,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magma_dpotrf_batched( uplo, n, dA_array, lda, info_array, batchCount, queue);
}

template <typename T>
magma_int_t
magma_getrf_batched(
        magma_int_t m, magma_int_t n,
        T **dA_array,
        magma_int_t lda,
        magma_int_t **ipiv_array,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

template <>
inline magma_int_t
magma_getrf_batched(
        magma_int_t m, magma_int_t n,
        float **dA_array,
        magma_int_t lda,
        magma_int_t **ipiv_array,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magma_sgetrf_batched( m, n, dA_array, lda, ipiv_array, info_array, batchCount, queue);
}

template <>
inline magma_int_t
magma_getrf_batched(
        magma_int_t m, magma_int_t n,
        double **dA_array,
        magma_int_t lda,
        magma_int_t **ipiv_array,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magma_dgetrf_batched( m, n, dA_array, lda, ipiv_array, info_array, batchCount, queue);
}

template <typename T>
magma_int_t
magma_getri_outofplace_batched(
        magma_int_t n,
        T **dA_array, magma_int_t ldda,
        magma_int_t **dipiv_array,
        T **dinvA_array, magma_int_t lddia,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

template <>
inline magma_int_t
magma_getri_outofplace_batched(
        magma_int_t n,
        float **dA_array, magma_int_t ldda,
        magma_int_t **dipiv_array,
        float **dinvA_array, magma_int_t lddia,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magma_sgetri_outofplace_batched( n, dA_array, ldda, dipiv_array, dinvA_array, lddia, info_array, batchCount, queue);
}

template <>
inline magma_int_t
magma_getri_outofplace_batched(
        magma_int_t n,
        double **dA_array, magma_int_t ldda,
        magma_int_t **dipiv_array,
        double **dinvA_array, magma_int_t lddia,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magma_dgetri_outofplace_batched( n, dA_array, ldda, dipiv_array, dinvA_array, lddia, info_array, batchCount, queue);
}

template <typename T>
void
magmablas_gemm(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        T alpha,
        T* dA, magma_int_t ldda,
        T* dB, magma_int_t lddb,
        T beta,
        T* dC, magma_int_t lddc,
        magma_queue_t queue );

template <>
inline void
magmablas_gemm(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        float alpha,
        float* dA, magma_int_t ldda,
        float* dB, magma_int_t lddb,
        float beta,
        float* dC, magma_int_t lddc,
        magma_queue_t queue )

{
        return magmablas_sgemm( transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
}

template <>
inline void
magmablas_gemm(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        double alpha,
        double* dA, magma_int_t ldda,
        double* dB, magma_int_t lddb,
        double beta,
        double* dC, magma_int_t lddc,
        magma_queue_t queue )

{
        return magmablas_dgemm( transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
}

template <typename T>
void
magmablas_gemv_batched(
        magma_trans_t trans, magma_int_t m, magma_int_t n,
        T alpha,
        T** dA_array, magma_int_t ldda,
        T** dx_array, magma_int_t incx,
        T beta,
        T** dy_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue);

template <>
inline void
magmablas_gemv_batched(
        magma_trans_t trans, magma_int_t m, magma_int_t n,
        float alpha,
        float** dA_array, magma_int_t ldda,
        float** dx_array, magma_int_t incx,
        float beta,
        float** dy_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magmablas_sgemv_batched( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue);
}

template <>
inline void
magmablas_gemv_batched(
        magma_trans_t trans, magma_int_t m, magma_int_t n,
        double alpha,
        double** dA_array, magma_int_t ldda,
        double** dx_array, magma_int_t incx,
        double beta,
        double** dy_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue)

{
        return magmablas_dgemv_batched( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue);
}





}
}