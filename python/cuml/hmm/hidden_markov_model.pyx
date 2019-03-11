#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cudf
import numpy as np

from numba import cuda
from cuml import numba_utils

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.hmm.sample_utils import *

cdef extern from "hmm/hmm_variables.h" :
    cdef cppclass HMM[T]:
        pass

cdef extern from "hmm/hmm_py.h" nogil:

    cdef void init_f32(HMM[float]&)
    cdef void setup_f32(HMM[float]&)

    cdef void forward_f32(HMM[float]&,
                          float*,
                          int*,
                          int)
    cdef void backward_f32(HMM[float]&,
                          float*,
                          int*,
                          int)

RUP_SIZE = 32

class GMMHMM:

    def _get_ctype_ptr(self, obj):
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _get_dtype(self, precision):
        return {
            'single': np.float32,
            'double': np.float64,
        }[precision]

    # TODO : Fix the default values
    def __init__(self, n_components=1, n_mix=1, min_covar=0.001, startprob_prior=1.0, transmat_prior=1.0, weights_prior=1.0, means_prior=0.0, means_weight=0.0, covars_prior=None, covars_weight=None, algorithm='viterbi', covariance_type='diag', random_state=None, n_iter=10, tol=0.01, verbose=False, params="stmcw", init_params="stmcw"):
        self.precision = precision
        self.dtype = self._get_dtype(precision)

        self.n_components = n_components
        self.tol = tol


      def _initialize_parameters(self, X, lengths):

        if lengths is None :
          self.nObs = X.shape[0]
        else :
          self.nObs =

        self.nDim = X.shape[1]
        self.nCl = self.n_components

        self.ldd = {"x" : roundup(self.nDim, RUP_SIZE),
                    "mus" : roundup(self.nDim, RUP_SIZE),
                    "sigmas" : roundup(self.nDim, RUP_SIZE),
                     "llhd" : roundup(self.nCl, RUP_SIZE),
                    "pis" : roundup(self.nCl, RUP_SIZE),
                    "inv_pis" : roundup(self.nCl, RUP_SIZE)}

        params = sample_parameters(self.nDim, self.nCl)
        params["llhd"] = sample_matrix(self.nCl, self.nObs, isColNorm=True)
        params['inv_pis'] = sample_matrix(1, self.nCl, isRowNorm=True)
        params["x"] = X.T

        params = align_parameters(params, self.ldd)
        params = flatten_parameters(params)
        params = cast_parameters(params ,self.dtype)

        self.dParams = dict(
            (key, cuda.to_device(params[key])) for key in self.ldd.keys())

        self.cur_llhd = cuda.to_device(np.zeros(1, dtype=self.dtype))

   def score(X, lengths=None):
