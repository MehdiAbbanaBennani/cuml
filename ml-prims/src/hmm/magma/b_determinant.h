#pragma once

#include "hmm/magma/magma_test_utils.h"
#include "hmm/magma/magma_batched_wrappers.h"

using namespace MLCommon;
using namespace MLCommon::LinAlg;

template <typename T>
__global__
void diag_batched_kernel(magma_int_t n, T** dU_array, magma_int_t lddu,
                         T* dDet_array, magma_int_t batchCount, int numThreads){
        int idxThread = threadIdx.x + blockDim.x * blockIdx.x;
        for (size_t i = idxThread; i < batchCount; i+=numThreads) {
                dDet_array[i] = 1;
                for (size_t j = 0; j < n; j++) {
                        dDet_array[i] *= dU_array[i][IDX(j, j, lddu)];
                }
        }
}

template <typename T>
void diag_product_batched(magma_int_t n, T** dU_array, magma_int_t lddu,
                          T*& dDet_array, magma_int_t batchCount){
        dim3 block(32, 1, 1);
        dim3 grid(ceildiv(batchCount, (int)block.x), 1, 1);
        int numThreads = grid.x * block.x;

        diag_batched_kernel<T> <<< grid, block >>>(n, dU_array, lddu,
                                                   dDet_array, batchCount,
                                                   numThreads);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
void det_batched(magma_int_t n, T** dA_array, magma_int_t ldda,
                 T*& dDet_array, magma_int_t batchCount, magma_queue_t queue){

        int **dipiv_array, *info_array;
        T **dA_array_cpy; // U and L are stored here after getrf
        allocate_pointer_array(dipiv_array, n, batchCount);
        allocate_pointer_array(dA_array_cpy, ldda * n, batchCount);
        allocate(info_array, batchCount);

        copy_batched(batchCount, dA_array_cpy, dA_array, ldda * n);

        // Getting errors with getrf
        magma_getrf_batched(n, n, dA_array_cpy, ldda, dipiv_array, info_array, batchCount, queue);

        // magma_potrf_batched(MagmaLower,
        //                     n,
        //                     dA_array,
        //                     ldda,
        //                     info_array,
        //                     batchCount,
        //                     queue
        //                     );
        // assert_batched(batchCount, info_array);

        diag_product_batched(n, dA_array_cpy, ldda, dDet_array, batchCount);


        free_pointer_array(dipiv_array, batchCount);
        free_pointer_array(dA_array_cpy, batchCount);
        CUDA_CHECK(cudaFree(info_array));
}