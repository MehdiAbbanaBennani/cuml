#include "hmm/gmm.h"

template <typename T>
void run(magma_int_t nCl, magma_int_t nDim, magma_int_t nObs, int n_iter)
{
        T *dX;
// declaration:
        T *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd;
        magma_int_t lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;
        lddx = magma_roundup(nDim, RUP_SIZE);
        lddmu = magma_roundup(nDim, RUP_SIZE);
        lddsigma = magma_roundup(nDim, RUP_SIZE);
        lddsigma_full = nDim * lddsigma;
        lddLlhd = magma_roundup(nCl, RUP_SIZE);
        lddPis = lddLlhd;

        // Random parameters
        T start=0;
        T end = 1;
        unsigned long long seed = 1234ULL;

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        allocate(dX, lddx * nObs);
        allocate(dmu, lddmu * nCl);
        allocate(dsigma, lddsigma_full * nCl);
        allocate(dLlhd, lddLlhd * nObs);
        allocate(dPis, lddPis);
        allocate(dPis_inv, lddPis);

        random_matrix_batched(nDim, 1, nCl, dmu, lddmu, false, seed, start, end);
        random_matrix_batched(nDim, nDim, nCl, dsigma, lddsigma, true, seed, start, end);
        generate_trans_matrix(nCl, nObs, dLlhd, lddLlhd, true);
        generate_trans_matrix(nCl, 1, dPis, lddPis, true);

// filling:
        fill_matrix_gpu(nDim, nObs, dX, lddx);
        print_matrix_device(nDim, nObs, dX, lddx, "dX");


// // computation:
        GMM<T> gmm;
        init(gmm,
             dmu, dsigma, dPis, dPis_inv, dLlhd,
             lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd,
             nCl, nDim, nObs);
        setup(gmm);
        fit(dX, n_iter, gmm, cublasHandle, queue);

// cleanup:
        CUDA_CHECK(cudaFree(dX));
}


int main( int argc, char** argv )
{
        magma_int_t nCl = 2;
        magma_int_t nDim = 2;
        magma_int_t nObs = 4;
        int n_iter = 5;

        run<double>(nCl, nDim, nObs, n_iter);

        magma_finalize();
        return 0;
}