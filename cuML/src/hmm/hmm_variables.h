#pragma once

template <typename T>
struct GMM {
        T *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd;
        T **dX_array=NULL, **dmu_array=NULL, **dsigma_array=NULL;

        int lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;

        int nCl, nDim, nObs;
};

template <typename T>
struct HMM {
        GMM<T>& gmm;

        // Transition and emission matrixes
        T *dT, dB;
        T **dAlpha_array, **dBeta_array, **dGamma_array;
        int lddt, lddv, lddb, lddalpha, lddbeta, lddgamma;
};
