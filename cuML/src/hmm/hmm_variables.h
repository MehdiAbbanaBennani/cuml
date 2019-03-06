#pragma once

template <typename T>
struct GMM {
        T *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd;
        T **dX_array=NULL, **dmu_array=NULL, **dsigma_array=NULL;

        int lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;

        int nCl, nDim, nObs;
};