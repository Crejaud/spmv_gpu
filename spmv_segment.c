#include "genresult.cuh"
#include <sys/time.h>

void mergeSort(MatrixInfo *mat, int l, int r);
void merge(MatrixInfo *mat, int l, int m, int r);

__global__ void putProduct_kernel(int nnz, float *A, float *x, float *y,
                                  int *coord_col, int *coord_row) {
  extern __shared__ float s_data[ ];

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_num = blockDim.x * gridDim.x;
  int iter = nnz % thread_num ? nnz / thread_num + 1 : nnz / thread_num;

  int offset;
  float t;
  int i;

  for (i = 0; i < iter; i++) {
    uint dataid = thread_id + i * thread_num;

    if (dataid >= nnz)
      return;

    //printf("accessing s_data before: %d %d\n", threadIdx.x, blockDim.x);

    s_data[threadIdx.x] = A[dataid] * x[coord_col[dataid]];
    //printf("done accessing s_data\n");
    __syncthreads();

    //printf("sync\n");

    for (offset = 1; offset < blockDim.x; offset <<= 1) {
      __syncthreads();
      t = s_data[threadIdx.x];
      /*threadIdx.x <= thread_id*/
      /*ensure threadIdx.x - offset is within bounds*/
      if (threadIdx.x >= offset) {
        /*if the offsets belong to the same row, then add them together */
        if (coord_row[dataid] == coord_row[dataid - offset])
          t += s_data[threadIdx.x - offset];
      }
      __syncthreads();
      s_data[threadIdx.x] = t;
    }

    /*store in y*/

    /*data_id is in bounds*/
    if (dataid + 1 < nnz) {
      /*thread_id corresponds to the end of the row*/
      if (coord_row[dataid] != coord_row[dataid + 1]) {
        /*If there's an existing value in y already, then atomic add to it*/
        if (y[coord_row[dataid]] != 0) {
          atomicAdd(&y[coord_row[dataid]], s_data[threadIdx.x]);
	  //printf("ATOMIC\n");
        }
        /*Otherwise, just store it*/
        else {
          y[coord_row[dataid]] = s_data[threadIdx.x];
          //printf("y %d set to %d\n", coord_row[dataid], s_data[threadIdx.x]);
        }
      }
      /*thread_id does not correspond tot he end of the row*/
      else {
        /*thread_id is the last index in the thread block*/
        if (dataid % blockDim.x == 0) {
          //printf("ITERATING\n");
          /*iterate to the last row then atomic add thread_id to it*/
          int endOfRowIndex = dataid;
          while (endOfRowIndex + 1 < nnz &&
                 coord_row[dataid] != coord_row[endOfRowIndex]) {
            endOfRowIndex++;
          }
          atomicAdd(&y[coord_row[endOfRowIndex]], s_data[threadIdx.x]);
	  //printf("ATOMIC\n");
        }
      }
    }
    /*this is the last element, so just store in y*/
    else {
      //printf("LAST\n");
      /*If there's an existing value in y already, then atomic add to it*/
      if (y[coord_row[dataid]] != 0)
        atomicAdd(&y[coord_row[dataid]], s_data[threadIdx.x]);
      /*Otherwise, just store it */
      else
        y[coord_row[dataid]] = s_data[threadIdx.x];
    }
  }
}

void getMulScan(MatrixInfo *mat, MatrixInfo *vec, MatrixInfo *res,
                int blockSize, int blockNum) {
  /*sort mat values based on row order*/
  /*int i;
  for (i = 0; i < mat->nz; i++) {
    printf("%d %d %d\n", mat->rIndex[i], mat->cIndex[i], mat->val[i]);
  }*/
  //printf("Starting sort\n");
  mergeSort(mat, 0, mat->nz - 1);
  //printf("Sort done\n");

  /*for (i = 0; i < mat->nz; i++) {
    printf("%d %d %d\n", mat->rIndex[i], mat->cIndex[i], mat->val[i]);
  }*/
  /*int i, j;
  for (i = 0; i < mat->nz; i++) {
    for (j = i + 1; j < mat->nz; j++) {
      if (mat->rIndex[i] > mat->rIndex[j]) {
        int t = mat->rIndex[i];
        mat->rIndex[i] = mat->rIndex[j];
        mat->rIndex[j] = t;

	t = mat->cIndex[i];
        mat->cIndex[i] = mat->cIndex[j];
        mat->cIndex[j] = t;

        float tv = mat->val[i];
        mat->val[i] = mat->val[j];
        mat->val[j] = tv;
      }
    }
  }
  */
  /*Allocate things...*/
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

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  /*Invoke kernel(s)*/

  putProduct_kernel<<<blockNum, blockSize, blockSize * sizeof(float)>>>(
      mat->nz, cuda_A_val, cuda_X_val, cuda_Y_val, cuda_A_coord_col,
      cuda_A_coord_row);

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  printf("Segmented Kernel Time: %lu milli-seconds\n",
         1000000 * (end.tv_sec - start.tv_sec) +
             (end.tv_nsec - start.tv_nsec) / 1000000);

  /*Deallocate, please*/
  cudaMemcpy(res->val, cuda_Y_val, res->M * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(cuda_A_val);
  cudaFree(cuda_X_val);
  cudaFree(cuda_Y_val);

  cudaFree(cuda_A_coord_row);
  cudaFree(cuda_A_coord_col);

  writeVect(res, "output.txt");
}

void mergeSort(MatrixInfo *mat, int l, int r) {
  if (l < r) {
    int m = l + (r - l)/2;

    mergeSort(mat, l, m);
    mergeSort(mat, m+1, r);
    //printf("%d %d %d\n", l, r, m);


    merge(mat, l, m, r);
  }
}

void merge(MatrixInfo *mat, int l, int m, int r) {
  //printf("prepre\n");
  int i, j, k;
  int n1 = m - l + 1;
  int n2 = r - m;

  //printf("pre: %d %d\n", n1, n2);

  int *L1 = (int *) malloc(n1 * sizeof(int));
  int *L2 = (int *) malloc(n1 * sizeof(int));
  int *R1 = (int *) malloc(n2 * sizeof(int));
  int *R2 = (int *) malloc(n2 * sizeof(int));

  float *L3 = (float *) malloc(n1 * sizeof(float));
  float *R3 = (float *) malloc(n2 * sizeof(float));

  //printf("inside\n");

  for (i = 0; i < n1; i++) {
    //printf("%d %d\n", l, l + i);
    L1[i] = mat->rIndex[l + i];
    L2[i] = mat->cIndex[l + i];
    L3[i] = mat->val[l + i];
  }
  for (j = 0; j < n2; j++) {
    //printf("%d %d\n", j, m + j + 1);
    R1[j] = mat->rIndex[m + j + 1];
    R2[j] = mat->cIndex[m + j + 1];
    R3[j] = mat->val[m + j + 1];
  }

  i = 0;
  j = 0;
  k = l;
  while (i < n1 && j < n2) {
    if (L1[i] <= R1[j]) {
      mat->rIndex[k] = L1[i];
      mat->cIndex[k] = L2[i];
      mat->val[k] = L3[i];
      i++;
    }
    else {
      mat->rIndex[k] = R1[j];
      mat->cIndex[k] = R2[j];
      mat->val[k] = R3[j];
      j++;
    }
    k++;
  }
  
  while (i < n1) {
    mat->rIndex[k] = L1[i];
    mat->cIndex[k] = L2[i];
    mat->val[k] = L3[i];
    i++;
    k++;
  }
  while (j < n2) {
    mat->rIndex[k] = R1[j];
    mat->cIndex[k] = R2[j];
    mat->val[k] = R3[j];
    j++;
    k++;
  }

  free(L1);
  free(L2);
  free(L3);
  free(R1);
  free(R2);
  free(R3);
}
