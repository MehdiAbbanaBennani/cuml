template <typename T>
__global__
void createForwardBatchesKernel(int nObs, int nCl,
                                T **dX_batches, T **dmu_batches, T **dInvSigma_batches,
                                T **dX_array, T **dmu_array, T **dInvSigma_array,
                                int nThreads_x, int nThreads_y){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        for (size_t stateId = j_start; stateId < nObs; stateId+=nThreads_y) {
                for (size_t ObsId = i_start; ObsId < nCl; ObsId+=nThreads_x) {
                        size_t idx = IDX(clId, obsId, nCl);
                        dX_batches[stateId][bId] = dX_array[obsId];
                }
        }
}

template <typename T>
__device__
T sum(T array, int len){
        T sum = 0;
        for (size_t i = 0; i < len; i++) {
                sum += array[i];
        }
        return sum
}

template <typename T>
__device__
void elementwise(T out, T in_a, T in_b, int len){
        for (size_t i = 0; i < len; i++) {
                out[i] = in_a[i] * in_b[i];
        }
}


template <typename T>
__global__
void _forwardLikelihoodKernel(){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;
        llhd_array[bId] = 1;
        llhd_array[bId] *= std::exp(scaleCoefs[bId]);

}

template <typename T>
__global__
void _forwardKernel(){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        if (i_start == 0 && j_start == 0) {
                for (size_t bId = 0; bId < batchCount; bId++) {
                        // Multiply with Bs elementwise
                        elementwise(dAlpha_array[bId], dB_array[bId], dAlpha_array[bId], nStates);

                        // Update the scale factors
                        T sum = compute_sum(alphas + bId * lddalpha, lddalpha);
                        scaleCoefs[bId] += std::log(sum);

                        // Normalize alphas
                        for (size_t i = 0; i < nStates; i++) {
                                scaleCoefs[bId][i] /= sum;
                        }

                }
        }
}

template <typename T>
void forward(T* dX, int* len_array, HMM<T>& hmm, int nObs,
             cublasHandle_t cublasHandle, magma_queue_t queue){

        int max_len = std::max(len_array);
        allocate(scaleCoefs, batchCount);

        // Compute the emissions likelihoods B (batchCount, max_len)

        // Set up the batches

        for (size_t t = 0; t < max_len; t++) {
                // Update state distribution
                magmablas_dgemv_batched ( MagmaTrans,
                                          n,
                                          n,
                                          1,
                                          dT_batches,
                                          hmm.lddt,
                                          dAlpha_batches[t],
                                          1,
                                          0,
                                          dAlpha_batches[t],
                                          1,
                                          batchCount_array[t],
                                          queue
                                          );

                dim3 block(32,32);
                dim3 grid(ceildiv(n, (int)block.x),
                          ceildiv(n, (int)block.y),
                          1);
                int nThreads_x = grid.x * block.x;
                int nThreads_y = grid.y * block.y;
                int nThreads_z = grid.z * block.z;

                _forwardKernel<T> <<< grid, block >>>();
                cudaDeviceSynchronize();
                CUDA_CHECK(cudaPeekAtLastError());
        }

        // Rescale alphas and compute llhd
        _forwardLikelihoodKernel<T> <<< grid, block >>>();
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
__device__
void max_state(){
        // Compute value and max pointer and update the value of alphas
        // Compute the value
        T val, maxVal=0.;
        int maxValIdx;
        int idx = IDX(prevStateId, curStateId, lddT);
        for (size_t stateId = 0; stateId < nStates; stateId++) {
                val = std::log(T[idx]) + std::log(alphas[prevStateId]) + std::log(B[curStateId]);
                if (val > maxVal) {
                        maxVal = val;
                        maxValIdx = prevStateId;
                }
        }
        alphas[stateId] = maxVal;
        dV_ptr[idx] = maxValIdx;
}

template <typename T>
__device__
int arg_max(T array, int len){
        T maxVal = array[0];
        int max_idx = 0;
        for (size_t i = 0; i < len; i++) {
                if (array[i] > maxVal) {
                        maxVal = array[i];
                        max_idx = i;
                }
        }
        return max_idx;
}


template <typename T>
__device__
void get_max_path(){
        max_path[len - 1] = arg_max(dAlpha);
        for (i = len - 2; i >= 0; --i) {
                max_path[i] = dVPtrs[IDX(max_path[i+1], i, lddvptrs)];
        }
}

template <typename T>
__global__
void virtebriKernel(){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        if (i_start == 0 && j_start == 0) {

                for (size_t bId = 0; bId < batchCount; bId++) {
                        for (size_t tau = 0; tau < len_array[bId]; tau++) {
                                for (size_t stateId = 0; stateId < nStates; stateId++) {
                                        max_state();
                                }
                        }
                }
                // Get max_path
                if (i_start == 0 && j_start == 0) {
                        get_max_path();
                }
        }
}

template <typename T>
void virtebri(T* dX, int* len_array, HMM<T>& hmm){
        int **dV_ptr_array;
        allocate(dV_ptr_array, lddv * max_len, nObs);

        dim3 block(32,32);
        dim3 grid(ceildiv(n, (int)block.x),
                  ceildiv(n, (int)block.y),
                  1);
        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        int nThreads_z = grid.z * block.z;

        virtebriKernel<T> <<< grid, block >>>();
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());

        free_pointer_array(dV_ptr_array);
}


template <typename T>
void _train_virterbi(T* dX, int* len_array, HMM<T>& hmm){
        // Compute the hidden states

        // Train using gmm toolkit


}

template <typename T>
void _train_em(T* dX, int* len_array, HMM<T>& hmm){
// Compute gammas

// Train using gmm toolkit
}
