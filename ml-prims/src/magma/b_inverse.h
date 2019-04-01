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

#include "magma/magma_test_utils.h"
#include "magma/magma_batched_wrappers.h"

using namespace MLCommon::LinAlg;

// TODO : ADD batched cublas
// Using cublas for large batch sizes and magma otherwise
// https://github.com/pytorch/pytorch/issues/13546

namespace MLCommon {


template <typename T>
__global__ void ID_kernel (int n, T *A, int ldda,
                           int nThreads_x, int nThreads_y) {
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        for (size_t j = j_start; j < n; j+=nThreads_y) {
                for (size_t i = i_start; i <  n; i+=nThreads_x) {
                        if (i == j)
                                A[IDX(i, j, ldda)] = 1.0;
                        else
                                A[IDX(i, j, ldda)] = 0.0;
                }
        }
}


template <typename T>
void make_ID_matrix(int n, T *A, int ldda) {
        dim3 block(32,32);
        dim3 grid(ceildiv(n, (int)block.x),
                  ceildiv(n, (int)block.y),
                  1);
        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;

        ID_kernel<T> <<< grid, block >>>(n, A, ldda,
                                         nThreads_x, nThreads_y);
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void inverse_batched_magma(magma_int_t n, T** dA_array, magma_int_t ldda,
                           T**& dinvA_array, magma_int_t batchCount,
                           magma_queue_t queue){

        int **dipiv_array, *info_array;
        T **dA_array_cpy;
        allocate_pointer_array(dipiv_array, n, batchCount);
        allocate_pointer_array(dA_array_cpy, ldda * n, batchCount);
        allocate(info_array, batchCount);
        copy_batched(batchCount, dA_array_cpy, dA_array, ldda * n);

        magma_getrf_batched(n, n, dA_array_cpy, ldda, dipiv_array, info_array,
                            batchCount, queue);
        // assert_batched(batchCount, info_array);

        magma_getri_outofplace_batched(n, dA_array_cpy, ldda, dipiv_array,
                                       dinvA_array, ldda, info_array,
                                       batchCount, queue);
        // assert_batched(batchCount, info_array);

        free_pointer_array(dipiv_array, batchCount);
        free_pointer_array(dA_array_cpy, batchCount);
        CUDA_CHECK(cudaFree(info_array));
}

template <typename T>
void inverse_batched(magma_int_t n, T** dA_array, magma_int_t ldda,
                     T** dinvA_array, magma_int_t batchCount, magma_queue_t queue){
        inverse_batched_magma(n, dA_array, ldda, dinvA_array, batchCount, queue);

}
}