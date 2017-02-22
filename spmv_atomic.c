#include "genresult.cuh"
#include <sys/time.h>

__global__ void getMulAtomic_kernel(int nnz, float *A, float *x, float *y,
                                    int *coord_row, int *coord_col) {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int thread_num = blockDim.x * gridDim.x;
  int iter = nnz % thread_num ? nnz / thread_num + 1 : nnz / thread_num;

  int i;
  for (i = 0; i < iter; i++) {
    int dataid = thread_id + i * thread_num;
    if (dataid < nnz) {
      float data = A[dataid];
      int row = coord_row[dataid];
      int col = coord_col[dataid];
      float temp = data * x[col];
      atomicAdd(&y[row], temp);
    }
  }
}

void getMulAtomic(MatrixInfo *mat, MatrixInfo *vec, MatrixInfo *res,
                  int blockSize, int blockNum) {
  /*Allocate here...*/
  float *cuda_A_val, *cuda_X_val, *cuda_Y_val;
  int *cuda_A_coord_row, *cuda_A_coord_col;

  cudaMalloc((void **)&cuda_A_val, mat->nz * sizeof(float));
  cudaMalloc((void **)&cuda_X_val, vec->M * sizeof(float));
  cudaMalloc((void **)&cuda_Y_val, res->M * sizeof(float));

  cudaMalloc((void **)&cuda_A_coord_row, mat->nz * sizeof(int));
  cudaMalloc((void **)&cuda_A_coord_col, mat->nz * sizeof(int));

  cudaMemcpy(cuda_A_val, mat->val, mat->nz * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_X_val, vec->val, vec->M * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_Y_val, res->val, res->M * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(cuda_A_coord_row, mat->rIndex, mat->nz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_A_coord_col, mat->cIndex, mat->nz * sizeof(int),
             cudaMemcpyHostToDevice);

  /* Sample timing code */
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  /*Invoke kernels...*/

  getMulAtomic_kernel<<<blockNum, blockSize>>>(mat->nz, cuda_A_val, cuda_X_val,
                                               cuda_Y_val, cuda_A_coord_row,
                                               cuda_A_coord_col);

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  printf("Atomic Kernel Time: %lu milli-seconds\n",
         1000000 * (end.tv_sec - start.tv_sec) +
             (end.tv_nsec - start.tv_nsec) / 1000000);
  /*Deallocate.*/

  cudaMemcpy(res->val, cuda_Y_val, res->M * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(cuda_A_val);
  cudaFree(cuda_X_val);
  cudaFree(cuda_Y_val);

  cudaFree(cuda_A_coord_row);
  cudaFree(cuda_A_coord_col);

  writeVect(res, "output.txt");
}
