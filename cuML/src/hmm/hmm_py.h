#pragma once

void init_f32(HMM<float> &hmm);

void forward_f32(HMM<float>& hmm,
                 float* dX,
                 int* len_array,
                 int nObs);

void backward_f32(HMM<float>& hmm,
                  float* dX,
                  int* len_array,
                  int nObs);

void setup_f32(GMM<float> &gmm);

void free_f32(GMM<float> &gmm);
