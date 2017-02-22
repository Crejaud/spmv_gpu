#include "genresult.cuh"
#include <sys/time.h>

/* Put your own kernel(s) here*/
__global__ void design_kernel(int nnz, float *A, float *x, float *y, int *coord_row, int *coord_col, int *start_of_rows, int *num_nz_in_row, int num_nz_rows) {

	/*One row will be held in shared memory*/
	extern __shared__ float row_s_data[ ];

	/*The iter is to iterate this process for different rows in the case that there are more rows than there are number of blocks*/
	int iter = num_nz_rows % gridDim.x ? num_nz_rows / gridDim.x + 1 : num_nz_rows / gridDim.x;
	//printf("Block #%d iter: %d\n", blockIdx.x, iter);
	//printf("Block #%d iter: %d num nz rows %d\n", blockIdx.x, iter, num_nz_rows);
	int iter_index, row_iter_index;
	int offset;
	/*This is a temporary float to avoid race conditions with segment scan*/
	float t;
	for (iter_index = 0; iter_index < iter; iter_index++) {
		__syncthreads();
		/*This is the ith nz row that this block will be computing*/
		int nz_row_index = blockIdx.x + iter_index * gridDim.x;

		/*This is the current row being processed*/
		int current_row = coord_row[start_of_rows[nz_row_index]];
		//printf("current row %d\n", current_row);
		/*This is the starting index of the thread*/
		int start_index = start_of_rows[nz_row_index] + threadIdx.x;
		
		/*The row_iter is the # of times to iterate this process in the case that the size of the block is less than the number of nz in the blockIdx.x'th row*/
		int row_iter = num_nz_in_row[nz_row_index] % blockDim.x ? num_nz_in_row[nz_row_index] / blockDim.x + 1 : num_nz_in_row[nz_row_index] / blockDim.x;
		for (row_iter_index = 0; row_iter_index < row_iter; row_iter_index++) {
			__syncthreads();
			int dataid = start_index + blockDim.x * row_iter_index;
			//if (dataid == nnz - 1) {printf("data_id %d\n", dataid);}
			/*Check if inbounds*/
			if (dataid >= nnz) {
				//printf("dataid too large. threadid %d current row %d num nz in row %d row iter %d\n", threadIdx.x, current_row, num_nz_in_row[nz_row_index], row_iter_index);
				continue;
			}
			/*Check if still on current row*/
			//printf("row compare %d %d\n", coord_row[dataid], current_row);
			if (coord_row[dataid] != current_row)
				continue;
		
			/*Set shared data to hold multiple*/
			row_s_data[threadIdx.x] = A[dataid] * x[coord_col[dataid]];
			__syncthreads();

			/*Segment scan*/
			for (offset = 1; offset < blockDim.x; offset <<= 1) {
				__syncthreads();
				t = row_s_data[threadIdx.x];
				if (threadIdx.x >= offset) {
					t += row_s_data[threadIdx.x - offset];
				}
				__syncthreads();
				row_s_data[threadIdx.x] = t;
			}

			__syncthreads();

			//printf("");
			//printf("");
			//if (threadIdx.x > 50000) { printf("test %d %d\n", dataid, threadIdx.x); }
			//printf("Segment scan complete: %d\n", row_s_data[threadIdx.x]);
			/*If this thread is not the last element*/
			if (dataid + 1 < nnz) {
				/*If the current row is not the same as the proceeding one, then it is the last in the row, since coord_row is sorted*/
				if (coord_row[dataid] != coord_row[dataid + 1]) {
					//printf("y row %d set to %d\n", coord_row[dataid], row_s_data[threadIdx.x]);
					//atomicAdd(&y[coord_row[dataid]], row_s_data[threadIdx.x]);
					y[coord_row[dataid]] += row_s_data[threadIdx.x];
				}
				/*This thread is not the end of the row*/
				else {
					//printf("thread not end of the row, dataid: %d row: %d\n", dataid, coord_row[dataid]);
					/*If this thread is at the end of the block, then add to it anyways*/
					if (threadIdx.x == blockDim.x - 1) {
						//atomicAdd(&y[coord_row[dataid]], row_s_data[threadIdx.x]);
						y[coord_row[dataid]] += row_s_data[threadIdx.x];
					}
				}
			}
			/*If this thread is the last element*/
			else {
				printf("y row %d set to %d\n", coord_row[dataid], row_s_data[threadIdx.x]);
				//atomicAdd(&y[coord_row[dataid]], row_s_data[threadIdx.x]);
				y[coord_row[dataid]] += row_s_data[threadIdx.x];
			}
		}
	}

}


void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
	/*Sort in coord_row order*/
	mergeSort(mat, 0, mat->nz - 1);
	//printf("nz %d\n", mat->nz);
	//printf("%d %d %d\n", mat->rIndex[0], mat->rIndex[mat->nz/2], mat->rIndex[mat->nz-1]);
	/*Get start_of_rows, num_nz_in_row, and num_nz_rows*/
	int *start_of_rows, *num_nz_in_row;
	int num_nz_rows = 1;
	int i;
	/*assume matrix has at least 1 nz*/
	int cur_row = mat->rIndex[0];
	/*Find number of nz rows*/
	for (i = 0; i < mat->nz; i++) {
		if (cur_row != mat->rIndex[i]) {
			cur_row = mat->rIndex[i];
			num_nz_rows++;
		}
	}
	//printf("num: %d\n", num_nz_rows);

	start_of_rows = (int *) malloc(num_nz_rows * sizeof(int));
	num_nz_in_row = (int *) malloc(num_nz_rows * sizeof(int));
	
	/*Find the start indices of all rows and how many nz elements are in those rows*/
	cur_row = mat->rIndex[0];
	start_of_rows[0] = 0;
	int nz_row_index = 1;
	int num_nz_in_one_row = 0;
	for (i = 0; i < mat->nz; i++) {
		if (cur_row != mat->rIndex[i]) {
			start_of_rows[nz_row_index] = i;
			cur_row = mat->rIndex[i];
			num_nz_in_row[nz_row_index - 1] = num_nz_in_one_row;
	
			//printf("row index %d num nz %d\n", start_of_rows[nz_row_index-1], num_nz_in_row[nz_row_index-1]);

			num_nz_in_one_row = 0;
			nz_row_index++;
		}
		num_nz_in_one_row++;
	}
	num_nz_in_row[num_nz_rows - 1] = num_nz_in_one_row;
	//printf("row index %d num nz %d\n", start_of_rows[nz_row_index-1], num_nz_in_row[nz_row_index-1]);

	/*Allocate*/
	float *cuda_A_val, *cuda_X_val, *cuda_Y_val;
	int *cuda_A_coord_row, *cuda_A_coord_col, *cuda_start_of_rows, *cuda_num_nz_in_row;

	cudaMalloc((void **)&cuda_A_val, mat->nz * sizeof(float));
	cudaMalloc((void **)&cuda_X_val, vec->M * sizeof(float));
	cudaMalloc((void **)&cuda_Y_val, res->M * sizeof(float));

	cudaMalloc((void **)&cuda_A_coord_row, mat->nz * sizeof(int));
	cudaMalloc((void **)&cuda_A_coord_col, mat->nz * sizeof(int));
	cudaMalloc((void **)&cuda_start_of_rows, num_nz_rows * sizeof(int));
	cudaMalloc((void **)&cuda_num_nz_in_row, num_nz_rows * sizeof(int));

	cudaMemcpy(cuda_A_val, mat->val, mat->nz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_X_val, vec->val, vec->M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_Y_val, res->val, res->M * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(cuda_A_coord_row, mat->rIndex, mat->nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_A_coord_col, mat->cIndex, mat->nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_start_of_rows, start_of_rows, num_nz_rows * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_num_nz_in_row, num_nz_in_row, num_nz_rows * sizeof(int), cudaMemcpyHostToDevice);

    struct timespec start, end;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Your own magic here!*/
	design_kernel<<<blockNum, blockSize, blockSize * sizeof(float)>>>(mat->nz, cuda_A_val, cuda_X_val, cuda_Y_val, cuda_A_coord_row, cuda_A_coord_col, cuda_start_of_rows, cuda_num_nz_in_row, num_nz_rows);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Your Own Kernel Time: %lu milli-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);

    /*Deallocate*/
	//printf("test\n");
	cudaMemcpy(res->val, cuda_Y_val, res->M * sizeof(float), cudaMemcpyDeviceToHost);
	//printf("test\n");
	
	//printf("test\n");
	cudaFree(cuda_A_val);
	cudaFree(cuda_X_val);
	cudaFree(cuda_Y_val);
	//printf("test\n");
	cudaFree(cuda_A_coord_row);
	cudaFree(cuda_A_coord_col);
	cudaFree(cuda_start_of_rows);
	cudaFree(cuda_num_nz_in_row);
	//printf("test\n");
	free(start_of_rows);
	free(num_nz_in_row);

	writeVect(res, "output.txt");
}
