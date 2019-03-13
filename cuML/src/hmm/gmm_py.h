#pragma once
#include "hmm/hmm_variables.h"

namespace gmm {

void init_f32(GMM<float> &gmm,
              float *dmu,  float *dsigma,
              float *dPis,  float *dPis_inv,  float *dLlhd,
              int lddx, int lddmu, int lddsigma, int lddsigma_full,
              int lddPis, int lddLlhd,
              float *cur_llhd, float reg_covar,
              int nCl, int nDim, int nObs);

void compute_lbow_f32(GMM<float> &gmm);

void update_llhd_f32(float* dX, GMM<float>& gmm);

void update_rhos_f32(GMM<float>& gmm, float* dX);

void update_mus_f32(float* dX, GMM<float>& gmm);

void update_sigmas_f32(float* dX, GMM<float>& gmm);

void update_pis_f32(GMM<float>& gmm);

void setup_f32(GMM<float> &gmm);

void free_f32(GMM<float> &gmm);



}