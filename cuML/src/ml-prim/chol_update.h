/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <iostream>
#include "cuda_utils.h"
#include "random/mvg.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"


namespace MLCommon {
namespace LinAlg {

using MLCommon::Random::fill_uplo;
using MLCommon::Random::Filler;

/**
 * @defgroup Cholesky rank 1 update
 * @tparam T: the data type of the matrices
 * @param sqrt: input sqroot matrix that is rank 1 updated, 
          at the same location updated sqroot matrix is kept after the function returns
 * @param vctr: vector that would be update sqrt by taking outer product with itself
 * @param dim: dimension of input vctr
 * @param sign: bool, if true, add.
 * @param uplo: if matrix is lower or upper triangular
 * @param workspace: if value is nullptr, work_size is updated to give the workspace 
          size requirement. Else pointer to the workspace should be provided.
 * @param work_size: size of the workspace required by the function
 */
template <typename T>
void chol_update(T *sqrt, T *vctr, int dim, bool sign,
                 cublasFillMode_t uplo, cusolverDnHandle_t cusolverH,
                 cublasHandle_t cublasH, void *workspace, int *work_size){
    // give the workspace requirements
    int granuality = 256;
    if (workspace == nullptr) {
        CUSOLVER_CHECK(cusolverDnpotrf_bufferSize(cusolverH, uplo, dim,
                                                  sqrt, dim, work_size));
        CUDA_CHECK(cudaDeviceSynchronize());
        size_t offset = 0;
        offset += alignTo(sizeof(T) * (*work_size), (size_t) granuality);
        offset += alignTo(sizeof(int) , (size_t) granuality);
        *work_size = offset;
        return;
    }
    // will recieve a lower or upper filled sqrt matrix, will probably have to take care.
    // set workspace
    int Lwork = *work_size - granuality;
    int *info = (int *)(&((T *)workspace)[Lwork]); // correcting types
    // find the full error cov.
    T alpha = (T)1.0, beta = (T)0.0;
    if (sign == false) {
        alpha = (T)-1.0;
    }
    CUBLAS_CHECK(cublasgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, dim, dim, dim,
                            &alpha, sqrt, dim, sqrt, dim, &beta, sqrt, dim));
    // find the outer product of the vector and add/sub to matrix
    CUBLAS_CHECK(cublasger(cublasH, dim, dim, &alpha, vctr, 1, vctr, 1, sqrt, dim));
    // find the sqrt of the modified matrix
    CUSOLVER_CHECK(cusolverDnpotrf(cusolverH, uplo, dim, sqrt, dim,
                                   (T *)workspace, Lwork, info));
    int info_h;
    updateHost(&info_h, info, 1);
    ASSERT(info_h == 0, "chol_update: error in potrf, info=%d | expected=0", info_h);
    // opposite part being filled with 0.0
    dim3 block(32, 32);
    dim3 grid(ceildiv(dim, (int)block.x), ceildiv(dim, (int)block.y));
    if (uplo == CUBLAS_FILL_MODE_LOWER) {
        fill_uplo<T> <<< grid, block >>>(dim, Filler::UPPER, (T)0.0, sqrt);
    }
    else { // (uplo == CUBLAS_FILL_MODE_UPPER)
        fill_uplo<T> <<< grid, block >>>(dim, Filler::LOWER, (T)0.0, sqrt);
    }
}

}; // end namespace LinAlg
}; // end namespace MLCommon

