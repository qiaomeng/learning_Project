#include <stdio.h>

#define N 32

__global__ void matrix_add(const int a[][N], const int b[][N], int c[][N])
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	c[idx][idy] = a[idx][idy] + b[idx][idy];
}

int main()
{
	int *h_a, *h_b, *h_c;
	int *dev_a, *dev_b, *dev_c;

	dim3 threads_in_block (N,N); 
	cudaError_t err = cudaSuccess;

	h_a = (int *)malloc(sizeof(int) * N * N);
	h_b = (int *)malloc(sizeof(int) * N * N);
	h_c = (int *)malloc(sizeof(int) * N * N);
	
	if (h_a == NULL || h_b == NULL | h_c == NULL){
		fprintf(stderr, "Malloc() failed.\n");
		return -1;
	}

	err = cudaMalloc((void **)&dev_a, sizeof(int) * N * N);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc() failed.\n");
		return -1;
	}
	err = cudaMalloc((void **)&dev_b, sizeof(int) * N * N);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc() failed.\n");
		return -1;
	}
	err = cudaMalloc((void **)&dev_c, sizeof(int) * N * N);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc() failed.\n");
		return -1;
	}

	for(int i = 0; i < N*N; i++){
		h_a[i] = 2 * i + 1;
		h_b[i] = -1 * i + 5;
	}

	err = cudaMemcpy(dev_a, h_a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc() failed.\n");
		return -1;
	}
	err = cudaMemcpy(dev_b, h_b, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc() failed.\n");
		return -1;
	}

	matrix_add<<<1, threads_in_block>>>((int (*)[N])dev_a, (int (*)[N])dev_b, (int (*)[N])dev_c);
	err = cudaMemcpy(h_c, dev_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		fprintf(stderr, "cudaMemcpy() failed.\n");
		return -1;
	}
	for(int i = 0; i < N * N ; i++){
		if(h_a[i] + h_b[i] != h_c[i]){
			fprintf(stderr, "a[%d]%d + b[%d]%d != c[%d]%d.\n", i, h_a[i], i, h_b[i], i, h_c[i]);
			return -1;
		}
	}

	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%d ", h_c[i]);
		}
		printf("\n");
	}
	
	printf("done.\n");
	return 0;
}
