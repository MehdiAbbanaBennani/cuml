#include "hmm/hmm.h"
#include "hmm/hmm_py.h"

void init_f32(GMM<float> &gmm,
              float *dmu, float *dsigma, float *dPis, float *dPis_inv, float *dLlhd, float *cur_llhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full, int lddPis, int lddLlhd,
              int nCl, int nDim, int nObs){
        init(gmm,
             dmu, dsigma, dPis, dPis_inv, dLlhd, cur_llhd,
             lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd,
             nCl, nDim, nObs);
}


void forward_f32(HMM<float>& hmm,
                 float* dX,
                 int* len_array,
                 int nObs){
        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        forward(dX, len_array, hmm, nObs, cublasHandle, queue);

        CUBLAS_CHECK(cublasDestroy(cublasHandle));
}

void backward_f32(HMM<float>& hmm,
                  float* dX,
                  int* len_array,
                  int nObs){
        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        forward(dX, len_array, hmm, nObs, cublasHandle, queue);

        CUBLAS_CHECK(cublasDestroy(cublasHandle));
}

void setup_f32(HMM<float> &hmm) {
        setup(hmm);
}

void free_f32(HMM<float> &hmm) {
        free(hmm);
}
