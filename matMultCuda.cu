#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define THREAD_NUM 256

#define MATRIX_SIZE 1000

//int blocks_num = MATRIX_SIZE * MATRIX_SIZE / 1024;;
//const int blocks_num = MATRIX_SIZE*(MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;

//打印设备信息
void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalFlbalMem : %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
}

bool InitCUDA()
{
	int count;

	//取得CUDA装置的数目
	cudaGetDeviceCount(&count);

	if(count == 0){
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;

	for(i = 0; i < count; i ++){

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess){
			if(prop.major >= 1){
				break;
			}
		}
	}

	if(i == count){
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);
	return true;
}

void matgen(float* a, int n)
{
	int i, j;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++)
		{
			a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
		}
	}
}

__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n, clock_t* time, const int* blocks_num)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	
	const int idx = bid * THREAD_NUM + tid;
	const int row = idx / n;
	const int column = idx % n;
	
	int i;
	
	 clock_t start;
	 
	 if(tid == 0) 
		 time[bid] = clock();
		 
	if(row < n && column < n){
		float t = 0;
		float y = 0;
		for(i = 0; i < n; i++){
			float r;
			y -= a[row * n + i] * b[i * n + column];
			r = t - y;
			y = (r-t)+y;
			t = r;
		}
		c[row * n + column] = t;
	}
	
	if(tid == 0){
		time[bid + *blocks_num] = clock();
	}
}

int func(int blocks_num)
{
	printf("%d\n", blocks_num);
	if(!InitCUDA()){
		return 0;
	}

	float *a, *b, *c, *d;
	int *e;
	
	int n = MATRIX_SIZE;

	a = (float*)malloc(sizeof(float)* n * n);
	b = (float*)malloc(sizeof(float)* n * n);
	c = (float*)malloc(sizeof(float)* n * n);
	d = (float*)malloc(sizeof(float)* n * n);
	e = (int*)malloc(sizeof(int));
	*e = blocks_num;
	srand(0);

	matgen(a, n);
	matgen(b, n);

	float *cuda_a, *cuda_b, *cuda_c;
	int *cuda_e;
	clock_t* time;

	cudaMalloc((void**)&cuda_a, sizeof(float) * n * n);
	cudaMalloc((void**)&cuda_b, sizeof(float) * n * n);
	cudaMalloc((void**)&cuda_c, sizeof(float) * n * n);
	cudaMalloc((void**)&cuda_e, sizeof(int));
	cudaMalloc((void**)&time, sizeof(clock_t) * blocks_num * 2);

	cudaMemcpy(cuda_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_e, e, sizeof(int), cudaMemcpyHostToDevice);
	matMultCUDA<<<blocks_num, THREAD_NUM, 0>>>(cuda_a, cuda_b, cuda_c, n, time, cuda_e);

	clock_t time_use[blocks_num * 2];

	cudaMemcpy(c, cuda_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_use, time, sizeof(clock_t) * blocks_num * 2, cudaMemcpyHostToDevice);

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
	cudaFree(time);

	clock_t min_start, max_end;
	min_start = time_use[0];
	max_end = time_use[blocks_num];
	for(int i = 1; i < blocks_num; i++){
		if(min_start > time_use[i]){
			min_start = time_use[i];
		}
		if(max_end < time_use[i + blocks_num]){
			max_end = time_use[i + blocks_num];
		}
	}

	clock_t final_time = max_end - min_start;

	//CPU矩阵乘法，存入d中
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

	printf("Max error: %g Average error: %g\n", max_err, average_err / (n * n));

	printf("gputime: %d\n", final_time);

	return 0;
}


int main()
{
	int blocks_num;
	//blocks_num = MATRIX_SIZE*(MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;
	//func(blocks_num);
	blocks_num = MATRIX_SIZE * MATRIX_SIZE / 1024;;
	func(blocks_num);
	//blocks_num = MATRIX_SIZE*(MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;
	//func(blocks_num);
}
