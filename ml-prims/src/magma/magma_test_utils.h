#include <stdio.h>
#include <stdlib.h>

#include <magma_v2.h>

#include "cuda_utils.h"
#include "linalg/mean_squared_error.h"

#define IDX(i,j,lda) ((i)+(j)*(lda))
#define RUP_SIZE 32

namespace MLCommon {


template <typename T>
void allocate_pointer_array(T **&dA_array, magma_int_t n, magma_int_t batchCount){
        T **A_array;

        A_array = (T **)malloc(sizeof(T*) * batchCount);
        for(int i = 0; i < batchCount; i++) {
                allocate(A_array[i], n);
        }

        allocate(dA_array, batchCount);
        updateDevice(dA_array, A_array, batchCount);
        free(A_array);
}

template <typename T>
void free_pointer_array(T **&dA_array, magma_int_t batchCount){
        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);
        for(int i = 0; i < batchCount; i++) {
                CUDA_CHECK(cudaFree(A_array[i]));
        }
        CUDA_CHECK(cudaFree(dA_array));
}

template <typename T>
void print_matrix_host(magma_int_t m, magma_int_t n, T *A, magma_int_t lda,
                       const std::string& msg){

          #define A(i_, j_) A[ (i_) + (j_)*lda ]

        printf("%s\n", msg.c_str());

        for (magma_int_t i=0; i < m; ++i) {
                for (magma_int_t j=0; j < n; ++j) {
                        printf("%f | ", (float) A(i,j));
                }
                printf("\n");
        }
        printf("\n");

          #undef A
}

template <typename T>
void print_matrix_device(magma_int_t m, magma_int_t n, T *dA, magma_int_t lda, const std::string& msg){
        T *A;
        A = (T *)malloc(sizeof(T) * lda * n);
        updateHost(A, dA, lda * n);
        print_matrix_host(m, n, A, lda, msg);
        free(A);
}

template <typename T>
void print_matrix_batched(magma_int_t m, magma_int_t n, magma_int_t batchCount, T **&dA_array, magma_int_t lda, const std::string& msg){

        printf("%s\n", msg.c_str());

        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);

        for (magma_int_t l=0; l < batchCount; ++l) {
                printf("Matrix %d \n", l);
                print_matrix_device(m, n, A_array[l], lda, msg);
        }

        free(A_array);

}

template <typename T>
void fill_matrix(magma_int_t m, magma_int_t n, T *A, magma_int_t lda)
{
      #define A(i_, j_) A[ (i_) + (j_)*lda ]

        for (magma_int_t j=0; j < n; ++j) {
                for (magma_int_t i=0; i < m; ++i) {
                        A(i,j) = rand() / ((T) RAND_MAX);
                }
        }

      #undef A
}


template <typename T>
void fill_matrix_gpu(
        magma_int_t m, magma_int_t n, T *dA, magma_int_t ldda)
{
        T *A;
        int lda = ldda;

        A = (T *)malloc(n * lda * sizeof(T));

        if (A == NULL) {
                fprintf( stderr, "malloc failed at %d in file %s\n", __LINE__, __FILE__ );
                return;
        }

        fill_matrix(m, n, A, lda);

        updateDevice(dA, A, n * lda);

        free(A);
}



template <typename T>
void fill_matrix_gpu_batched(
        magma_int_t m, magma_int_t n, magma_int_t batchCount, T **&dA_array, magma_int_t ldda)
{
        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);
        for (size_t i = 0; i < batchCount; i++) {
                fill_matrix_gpu(m, n, A_array[i], ldda);
        }
        free(A_array);
}


template <typename T>
T array_mse_batched(
        magma_int_t m, magma_int_t n, magma_int_t batchCount, T **&dA_array, magma_int_t ldda, T **&dB_array, magma_int_t lddb)
{
        // ASSERT(ldda == lldb, "the two arrays much have the same leading dimensions");

        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);

        T **B_array;
        B_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(B_array, dB_array, batchCount);

        T error_h=0, *error_d, error=0;
        allocate(error_d, 1);

        for (size_t bId = 0; bId < batchCount; bId++) {
                // print_matrix_device(m, n, A_array[bId], ldda, 'a');
                MLCommon::LinAlg::meanSquaredError(error_d, A_array[bId], B_array[bId], m * ldda);
                updateHost(&error_h, error_d, 1);
                error += error_h;
        }
        error /= batchCount;

        free(A_array);
        free(B_array);

        return error;

}

template <typename T>
void fill_pointer_array(magma_int_t batchCount, T **&dA_array, T *dB){
        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);
        for (size_t bId = 0; bId < batchCount; bId++) {
                A_array[bId] = dB;
        }
        updateDevice(dA_array, A_array, batchCount);
        free(A_array);
}

template <typename T>
void copy_batched(magma_int_t batchCount, T **dA_dest_array, T **dA_src_array,
                  size_t len){
        T **A_dest_array, **A_src_array;

        A_src_array = (T **)malloc(sizeof(T*) * batchCount);
        A_dest_array = (T **)malloc(sizeof(T*) * batchCount);

        updateHost(A_src_array, dA_src_array, batchCount);
        updateHost(A_dest_array, dA_dest_array, batchCount);

        for (size_t bId = 0; bId < batchCount; bId++) {
                copy(A_dest_array[bId], A_src_array[bId], len);
        }

        free(A_src_array);
        free(A_dest_array);
}


void assert_batched(int batchCount, int *d_info_array){
        int *h_info_array;
        h_info_array = (int *)malloc(sizeof(int) * batchCount);
        updateHost(h_info_array, d_info_array, batchCount);
        for (size_t i = 0; i < batchCount; i++) {
                ASSERT(h_info_array[i] == 0, "info returned val=%d", h_info_array[i]);
        }
        free(h_info_array);
}

}
