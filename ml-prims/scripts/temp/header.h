cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const double          *alpha,
                                  const double          *Aarray[], int lda,
                                  const double          *Barray[], int ldb,
                                  const double          *beta,
                                  double          *Carray[], int ldc,
                                  int batchCount)
