#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>

#define NUMTHREADS 512 

#define MATRIX_SIZE 1000

void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalFlbalMem : %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %d.\n", prop.regsPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %d.\n", prop.totalConstMem);
	printf("major.minor : %d.%d.\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %d.\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

bool InitCUDA()
{
	int count;
	cudaGetDeviceCount(&count);

	if(count == 0){
		fprintf(stderr, "There is no device.\n");

		return false;
	}

	int i;

	for(i = 0; i < count; i++)
	{
		cudaDeviceProp prop;
		int status;
		status = cudaGetDeviceProperties(&prop, i);
		//printDeviceProp(prop);

		if(status == cudaSuccess){
			if(prop.major >= 1){
				break;
			}
		}
	}
	printf("Total %d GPU\n", count);
	if(i == count){
		fprintf(stderr, "There is no device supporting cuda 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

void matgen(float* a, int n)
{
	int i, j;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);

		}
	}
}

__global__ static void matMul

int main()
{
	if(!InitCUDA())
		return 0;

	float *a, *b, *c, *d, *e;

	int n = MATRIX_SIZE;

	a = (float*)malloc(sizeof(float) * n * n);
	b = (float*)malloc(sizeof(float) *n * n);
	c = (float*)malloc(sizeof(float) * n * n);
	d = (float*)malloc(sizeof(float) * n * n);
	srand(0);
	matgen(a, n);
	matgen(b, n);

	for(int i = 0; i < 10; i++)
		printf("%f ", a[i]);
	printf("\n");

	float *cuda_a, *cuda_b, *cuda_c, *cuda_e;
	clock_t* time;

	cudaMalloc((void**)&cuda_a, sizeof(float) * n * n);
	cudaMalloc((void**)&cuda_b, sizeof(float) * n * n);
	cudaMalloc((void**)&cuda_c, sizeof(float) * n * n);
	cudaMalloc((void**)&time, sizeof(clock_t) * n * 2);

	cudaMemcpy(cuda_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice);

	//matMultCUDA<<<blocks_num, THREAD_NUM, 0>>>(cuda_a, cuda_b, cuda_c, n, time);
	matMultCUDA<<<n, THREAD_NUM, sizeof(float) * n >>> (cuda_a, cuda_b, cuda_c, n, time);

	clock_t time_use[blocks_num * 2];

	cudaMemcpy(c, cuda_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_use, time, sizeof(clock_t) * 2 * blocks_num, cudaMemcpyDeviceToHost);

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
	cudaFree(time);
	printf("sizeof(c)= %d\n",sizeof(c));

	for(int i = 0; i < 10; i++)
		printf("%f ", c[i]);
	
	clock_t min_start, max_end;
	min_start = time_use[0];
	max_end = time_use[blocks_num];
	for(int i = 1; i < blocks_num; i++){
		if(min_start > time_use[i])
			min_start = time_use[i];
		if(max_end < time_use[i + blocks_num])
			max_end = time_use[i + blocks_num];
	}
	clock_t  final_time = max_end - min_start;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			double t = 0;
			for(int k = 0; k < n; k++){
				t += a[i * n + k] * b[k * n + j];
			}
			d[i * n + j] = t;
		}
	}

	float max_err = 0;
	float average_err = 0;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(d[i * n + j] != 0){
				float err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);
				if(max_err < err)
					max_err = err;
				average_err += err;
			}
		}
	}
	printf("Max error: %g Average error: %g\n", max_err, average_err / (n*n));

	printf("gputime: %d\n", final_time);
	return 0;
}
