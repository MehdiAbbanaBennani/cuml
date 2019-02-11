/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "sigma.h"

namespace kf {
namespace linear {

    /** Solver options for LinearKF */
    enum Option {
        /** long form of KF equation */
         LongForm = 0,
         /** short (optimal) form of KF equation with explicit inverse finding */
         ShortFormExplicit,
         /** short (optimal) form of KF equation with implicit inverse evaluation */
         ShortFormImplicit
    };

    template <typename T>
    struct Variables{
         // these device pointers are NOT owned by this class
        T *x_est, *x_up, *Phi; // predict_x
        T *P_est, *P_up, *Q; // predict_P
        T *R, *H;  // kalman gain
        T *z; // update_x
        bool initialized=false;
        // cublasHandle_t handle;
        // cusolverDnHandle_t handleSol;
        Option solver;
        // state and measurement vector dimensions, respectively
        int dim_x, dim_z;
        // all the workspace related pointers
        int Lwork;
        T *workspace_lu = 0, *placeHolder0 = 0, *R_cpy = 0;
        T *placeHolder1 = 0, *placeHolder2 = 0, *K = 0;//kalman_gain
        int *piv, *info;
    };


}; // end namespace linear

namespace unscented {

    /** implicit/explicit kalman gain calculation */
    enum Inverse {
        Explicit,
        Implicit
    };

    template <typename T>
    struct Variables {
        T *x_est, *x_up, *Phi, *P_xx;
        T *Q, *R, *H;
        bool initialized=false;
        cublasHandle_t handle;
        cusolverDnHandle_t handleSol;
        Inverse inverse_method;
        // Sigma Points generator
        kf::sigmagen::VanDerMerwe<T> *sg;
        T *Wm, *Wc, *K;
        // state and measurement vector dimensions
        int dim_x, dim_z;
        int nPoints;
        T alpha_s, beta_s, kappa_s;
        bool sqrt;
        // all the workspace related pointers
        int Lwork;
        T *workspace_cholesky, *workspace_sg;
        int *info;
        // these come from splitting workpace ptr provided by user
        T *P_xz, *P_zz, *X_h, *X, *X_er, *X_h_er;
        T *eig_z, *placeHolder0, *z;
        T *U;

    };

}; // end namespace unscented
}; // end namespace kf
