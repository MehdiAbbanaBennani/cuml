# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import cudf
import numpy as np

import cuml
from cuml.hmm.sample_utils import *
from sklearn.mixture import GaussianMixture
from cuml.hmm.utils import timer, info


def np_to_dataframe(df):
    pdf = cudf.DataFrame()
    for c in range(df.shape[1]):
        pdf[c] = df[:, c]
    return pdf

def mse(x, y):
    return np.mean((x - y) ** 2)

def compute_error(params_pred, params_true):
    mse_dict = dict(
        (key, mse(params_pred[key], params_true[key]) )
         for key in params_pred.keys())
    error = sum([mse_dict[key] for key in mse_dict.keys()])
    return mse_dict, error

# @info
def sample(nDim, nCl, nObs, precision):
    if precision == 'single':
        dt = np.float32
    else:
        dt = np.float64

    params = sample_parameters(nDim=nDim, nCl=nCl)
    X = sample_data(nObs, params)
    params = cast_parameters(params, dtype=dt)
    return X, params

# @timer("sklearn")
# @info
def run_sklearn(X, n_iter, nCl):
    gmm = GaussianMixture(n_components=nCl,
                             covariance_type="full",
                             tol=10000,
                             reg_covar=0,
                             max_iter=n_iter,
                             n_init=1,
                             init_params="random",
                             weights_init=None,
                             means_init=None,
                             precisions_init=None,
                             random_state=10,
                             warm_start=False,
                             verbose=1,
                             verbose_interval=1)
    gmm.fit(X)
    # #
    # from cuml.hmm.sk_debug import fit_predict
    # fit_predict(gmm, X)

    params = {"mus" : gmm.means_,
              "sigmas" : gmm.covariances_,
              "pis" : gmm.weights_}

    return params

# @timer("cuml")
# @info
def run_cuml(X, n_iter, precision, nCl):
    gmm = cuml.GaussianMixture(n_components=nCl,
                               max_iter=n_iter,
                               precision=precision,
                               reg_covar=0,
                               random_state=10,
                               warm_start=False)

    gmm.fit(X)

    params = {"mus": gmm.means_,
              "sigmas": gmm.covariances_,
              "pis": gmm.weights_}
    return params


def print_info(true_params, sk_params, cuml_params):
    print("\n true params")
    print(true_params)

    print('\nsklearn')
    mse_dict_sk, error_sk = compute_error(sk_params, true_params)
    print('error')
    print(mse_dict_sk)
    print("params")
    print(sk_params)

    print('\ncuml')
    mse_dict_cuml, error_cuml = compute_error(cuml_params, true_params)
    print('error')
    print(mse_dict_cuml)
    print("params")
    print(cuml_params)

    print('\ncuml-sk')
    mse_dict_cuml, error_cuml = compute_error(cuml_params, sk_params)
    print('errors')
    print(mse_dict_cuml)


@pytest.mark.parametrize('n_iter', [2, 10])
@pytest.mark.parametrize('nCl', [1, 2, 5])
@pytest.mark.parametrize('nDim', [1, 5, 10])
@pytest.mark.parametrize('nObs', [100, 1000])
@pytest.mark.parametrize('precision', ["single"])
def test_gmm(n_iter, nCl, nDim, nObs, precision):

    X, true_params = sample(nDim=nDim, nCl=nCl, nObs=nObs, precision=precision)

    sk_params = run_sklearn(X, n_iter, nCl)
    cuml_params = run_cuml(X, n_iter, precision, nCl)

    print_info(true_params, sk_params, cuml_params)
    error_dict, error = compute_error(cuml_params, sk_params)
    assert error < 1e-02
