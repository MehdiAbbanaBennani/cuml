#include "kalman_filter/KalmanFilter.cuh"
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"

namespace kf
{
namespace unscented
{

using namespace MLCommon;

template <typename T>
struct UnKFInputs
{
    T tolerance;
    int dim_x, dim_z, iterations;
    T alpha, beta, kappa;
    Inverse inv;
    unsigned long long int seed;
    bool sqrt;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const UnKFInputs<T> &dims)
{
    return os;
}

template <typename T>
class SUNKF
{
public:
    SUNKF(){};

    ~SUNKF()
    {
        // freeing gpu mallocs
        CUDA_CHECK(cudaFree(workspace));
        CUDA_CHECK(cudaFree(Phi_d));
        CUDA_CHECK(cudaFree(P_up_d));
        CUDA_CHECK(cudaFree(Q_d));
        CUDA_CHECK(cudaFree(R_d));
        CUDA_CHECK(cudaFree(H_d));
        CUDA_CHECK(cudaFree(x_est_d));
        CUDA_CHECK(cudaFree(x_up_d));
        CUDA_CHECK(cudaFree(z_d));

        // freeing cpu mallocs
        free(Phi);
        free(x_up);
        free(x_est);
        free(P_up);
        free(Q);
        free(H);
        free(R);
        free(z);
    }

    void SetUp(UnKFInputs<T> params)
    {
        // getting params
        params = params;
        inv = params.inv;
        dim_x = params.dim_x;
        dim_z = params.dim_z;
        iterations = params.iterations;
        tolerance = params.tolerance;
        T alpha_s = params.alpha;
        T beta_s = params.beta;
        T kappa_s = params.kappa;
        sqrt = params.sqrt;

        // cpu mallocs
        Phi = (T *)malloc(dim_x * dim_x * sizeof(T));
        x_up = (T *)malloc(dim_x * 1 * sizeof(T));
        x_est = (T *)malloc(dim_x * 1 * sizeof(T));
        P_up = (T *)malloc(dim_x * dim_x * sizeof(T));
        Q = (T *)malloc(dim_x * dim_x * sizeof(T));
        H = (T *)malloc(dim_z * dim_x * sizeof(T));
        R = (T *)malloc(dim_z * dim_z * sizeof(T));
        z = (T *)malloc(dim_z * 1 * sizeof(T));

        cublasHandle_t cublas_handle;
        cusolverDnHandle_t cusolver_handle = NULL;

        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

        // making sane model
        x_up[0] = 0.0;
        x_up[1] = 1.0;
        Phi[0] = 1.0;
        Phi[1] = 0.0;
        Phi[2] = 1.0;
        Phi[3] = 1.0;
        P_up[0] = 10.0;
        P_up[1] = 0.0;
        P_up[2] = 0.0;
        P_up[3] = 10.0;
        R[0] = 10.0;
        T var = 0.001;
        Q[0] = 0.25 * var;
        Q[1] = 0.5 * var;
        Q[2] = 0.5 * var;
        Q[3] = 1.1 * var;
        H[0] = 1.0;
        H[1] = 0.0;

        // gpu mallocs
        CUDA_CHECK(cudaMalloc((void **)&x_est_d, dim_x * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void **)&x_up_d, dim_x * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void **)&Phi_d, dim_x * dim_x * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void **)&P_up_d, dim_x * dim_x * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void **)&Q_d, dim_x * dim_x * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void **)&R_d, dim_z * dim_z * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void **)&H_d, dim_z * dim_x * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void **)&z_d, dim_z * sizeof(T)));

        // copy data to gpu (available in ml-common/cuda_utils.h)
        updateDevice(Phi_d, Phi, dim_x * dim_x);
        updateDevice(x_up_d, x_up, dim_x);
        updateDevice(P_up_d, P_up, dim_x * dim_x);
        updateDevice(Q_d, Q, dim_x * dim_x);
        updateDevice(R_d, R, dim_z * dim_z);
        updateDevice(H_d, H, dim_z * dim_x);

        // kf initialization
        Variables<T> vars;
        size_t workspaceSize;
        init(vars, dim_x, dim_z, alpha_s, beta_s, kappa_s, inv,
             x_est_d, x_up_d, Phi_d, P_up_d, Q_d, R_d, H_d, sqrt, nullptr,
             workspaceSize, cusolver_handle);
        CUDA_CHECK(cudaMalloc((void **)&workspace, workspaceSize));
        init(vars, dim_x, dim_z, alpha_s, beta_s, kappa_s, inv,
             x_est_d, x_up_d, Phi_d, P_up_d, Q_d, R_d, H_d, sqrt, workspace,
             workspaceSize, cusolver_handle);

        // for random noise
        std::default_random_engine generator(params.seed);
        std::normal_distribution<T> distribution(0.0, 1.0);
        rmse_x = 0.0;
        rmse_v = 0.0;

        for (int q = 0; q < iterations; q++)
        {
            printf(" \n  ----- Step %d -----", q);
            printf(" \n Predict ... \n ");
            predict(vars, cublas_handle);
            // generating measurement
            z[0] = q + distribution(generator);
            updateDevice(z_d, z, dim_z);
            printf(" \n Update ... \n");
            update(vars, z_d, cublas_handle, cusolver_handle);
            // getting update
            updateHost(x_up, x_up_d, dim_x);
            // summing squared ratios
            rmse_v += pow(x_up[1] - 1, 2); // true velo is alwsy 1
            rmse_x += pow(x_up[0] - q, 2);
        }
        rmse_x /= iterations;
        rmse_v /= iterations;
        rmse_x = pow(rmse_x, 0.5);
        rmse_v = pow(rmse_v, 0.5);
    }

  protected: // variables
    UnKFInputs<T> params;
    T alpha, beta, kappa;
    Inverse inv;
    bool sqrt;
    T *Phi, *x_up, *x_est, *P_up, *Q, *H, *R, *z; //cpu pointers
    T *x_est_d, *x_up_d, *Phi_d, *P_up_d,
        *Q_d, *R_d, *H_d, *z_d, *workspace; //gpu pointers
    T rmse_x, rmse_v, tolerance;            // root mean squared error
    int dim_x, dim_z, iterations;
}; // end of UnKFTest class

}; // end namespace unscented
}; // end namespace kf

int main(){
    // double
    const kf::unscented::UnKFInputs<double> inputd = 
    {0.6, 2, 1, 100, 1e-3, 2.0, 0.0, kf::unscented::Inverse::Explicit, 6ULL, true};

    kf::unscented::SUNKF<double> unkf_test;
    unkf_test.SetUp(inputd);
    return 0;
}