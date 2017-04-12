#include <string.h>
#include <stdlib.h>
#include <stdio.h>
//CUDA RunTime API
#include <cuda_runtime.h>
#include <time.h>

#define THREAD_NUM 1024
#define BLOCK_NUM 16
#define DATA_SIZE 1048576

// __global__ 函数（GPU上执行） 计算立方和
__global__ static void sumOfSquares(int *num, int* result, clock_t* time)
{
	extern __shared__ int shared[];

	const int tid = threadIdx.x;

	const int bid = blockIdx.x;

	shared[tid] = 0;

	int i;

	if(tid == 0)
		time[bid] = clock();

	for(i = bid * THREAD_NUM + tid; i < DATA_SIZE; i+= BLOCK_NUM * THREAD_NUM){
		
		shared[tid] += num[i] * num[i] * num[i];

	}
	//第一个同步的意思是确保所有的线程完成了立方和的操作
	__syncthreads();

	int offset = 1, mask = 1;

	while(offset < THREAD_NUM){
		//由于为二进制，逢2进一，因此进行当两个相差2的数进行&操作时势必为0
		if((tid & mask) == 0){
			shared[tid] = shared[tid + offset];
		}

		offset += offset;
		mask = offset + mask;
		//第二个同步的意思是确保每次的累加完成，这样才能进入下一次的累加操作
		__syncthreads();
	}

	if(tid == 0){
		result[bid] = shared[0];
		time[bid + BLOCK_NUM] = clock();
	}
}

void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
	printf("shareMemPredBLock : %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %d.\n", prop.totalConstMem);
	printf("major.minor : %d.%d\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %d\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

int data[DATA_SIZE];

//产生大量0-9之间的随机数
void GenerateNumbers(int *number, int size)
{
	for(int i = 0; i < size; i++){
		number[i] = rand() % 10;
	}
}

//CUDA初始化
bool InitCUDA()
{
	int count;

	//取得cuda的装置的数目
	cudaGetDeviceCount(&count);

	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	
	int i;
	
	for(i = 0; i < count; i++){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//打印设备信息
		printDeviceProp(prop);
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



int main()
{
	if (!InitCUDA()){
		return 0;
	}

	printf("CUDA initialized.\n");
	//生成随机数
	GenerateNumbers(data, DATA_SIZE);

	/*吧数据复制到显卡内部*/

	int* gpudata, *result;
	clock_t* time;

	//cudaMalloc取得一块显存内存（其中result存储结果)
	cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int) * BLOCK_NUM);
	cudaMalloc((void**)&time, sizeof(clock_t) * BLOCK_NUM * 2);

	//cudaMemcpy 将产生的随机数复制到显存内存中
	//cudaMemcpyHostToDevice - 从内存复制到显存
	//cudaMemcpyDeviceToHost - 从显存复制到内存
	cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares<<<BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int)>>>(gpudata, result, time);

	int sum[BLOCK_NUM];
	clock_t time_used[BLOCK_NUM * 2];

	//cudaMemcpy 将结果从显存中复制回内存
	cudaMemcpy(&sum, result, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2 ,cudaMemcpyDeviceToHost);
	int final_sum = 0;
	for(int i = 0; i < BLOCK_NUM; i++){
		final_sum += sum[i];
	}
	//Free
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);
	clock_t min_start, max_end;
	min_start = time_used[0];
	max_end = time_used[BLOCK_NUM];

	for(int i = 1; i < BLOCK_NUM; i++){
		if(min_start > time_used[i])
			min_start = time_used[i];
		if(max_end < time_used[i + BLOCK_NUM])
			max_end = time_used[i + BLOCK_NUM];
	}
	printf("GPUsum: %d gputime: %d\n", final_sum, max_end - min_start) ;
	final_sum = 0;
	for(int i = 0; i < DATA_SIZE; i++){
		final_sum += data[i] * data[i] * data[i];
	}

	printf("CPUsum: %d \n", final_sum);
	return 0;
}
