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

#include <stdlib.h>
#include <vector>

#include <gmm/gmm_variables.h>

namespace hmm {

/** Train options for HMM */
enum TrainOption {
        Vitebri,
        ForwardBackward
};

enum DistOption {
        MultinomialDist,
        GaussianMixture
};


// D is the emission distribution
template <typename T, typename D>
struct HMM {
        int nStates;
        std::vector<D> dists;
        // All dLlhd point to dGamma
        T *dStartProb, *dT, *dB, *dAlpha, *dBeta, *dGamma;
        int lddsp, lddt, lddb, lddalpha, lddbeta, lddgamma;

        int nObs, nSeq, max_len;
        T* dLlhd;
        T* logllhd;

        T **dB_array, **dPi_array;

        T *dV;
        int lddv;

        int nFeatures;
        unsigned short int *dcumlenghts_inc, *dcumlenghts_exc;

        TrainOption train;
        DistOption distOption;
};

}