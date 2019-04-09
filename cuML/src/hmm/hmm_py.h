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

#include "hmm/hmm_variables.h"
#include "gmm/gmm_variables.h"
#include "hmm/dists/dists_variables.h"

namespace multinomial {

void init_multinomial_f64(multinomial::Multinomial<double> &multinomial,
                          double* dPis, int nCl);
}

namespace hmm {
void init_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                     std::vector<gmm::GMM<double> > &gmms,
                     int nStates,
                     double* dStartProb, int lddsp,
                     double* dT, int lddt,
                     double* dB, int lddb,
                     double* dGamma, int lddgamma,
                     double* logllhd,
                     int nObs, int nSeq,
                     double* dLlhd
                     );

void setup_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm, double* dLlhd);

void forward_backward_gmmhmm_f64(HMM<double, gmm::GMM<double> > &hmm,
                                 double* dX, unsigned short int* dlenghts, int nSeq,
                                 bool doForward, bool doBackward, bool doGamma);

void init_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                   std::vector<multinomial::Multinomial<double> > &gmms,
                   int nStates,
                   double* dStartProb, int lddsp,
                   double* dT, int lddt,
                   double* dB, int lddb,
                   double* dGamma, int lddgamma,
                   double* logllhd,
                   int nObs, int nSeq,
                   double* dLlhd
                   );

size_t get_workspace_size_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm);

void create_handle_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                            void* workspace);

void setup_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                    int nObs, int nSeq, double* dLlhd);

void forward_backward_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                               unsigned short int* dX, unsigned short int* dlenghts, int nSeq,
                               bool doForward, bool doBackward, bool doGamma);

void viterbi_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                      unsigned short int* dVstates, unsigned short int* dX, unsigned short int* dlenghts, int nSeq);


void m_step_mhmm_f64(HMM<double, multinomial::Multinomial<double> > &hmm,
                     unsigned short int* dX, unsigned short int* dlenghts, int nSeq);
}
