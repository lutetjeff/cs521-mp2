#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define MATMUL_TILE_WIDTH 16

__global__ void matmul_conv_fused(const float * __restrict__ mask, const float * __restrict__ input, float * __restrict__ output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
	// moved up to ensure use of register 0
    const int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    const int row = by * MATMUL_TILE_WIDTH + ty, col = bx * MATMUL_TILE_WIDTH + tx;
    
    __shared__ float tileA[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];
    __shared__ float tileB[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];

    // hopefully ordering doesnt matter and all these are chucked into registers
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int hw = (Height_out * Width_out);
    const size_t mhw = hw * Map_out;

   	const int kk = K * K;

    const int b = col / hw;
    const int clmhw = col - b * hw; // again
   	const int h = clmhw / Width_out;
   	const int w = clmhw - h * Width_out; // again	
    const int hwin = Height * Width;
    const size_t chwin = Channel * hwin;

    const size_t numARows = Map_out;
    const size_t numAColumns = Channel * K * K;
    const size_t numBRows = numAColumns;
    const size_t numBColumns = Batch * hw;
    const size_t numCRows = numARows;
    const size_t numCColumns = numBColumns;
    
    float val = 0;

	#pragma unroll 2
    for (int tileId = 0; tileId < (numAColumns - 1) / MATMUL_TILE_WIDTH + 1; tileId++) {
       	size_t tcol = tileId * MATMUL_TILE_WIDTH + tx;
        if (row < numARows && tcol < numAColumns) {
            tileA[ty][tx] = mask[row * numAColumns + tcol];
        } else {
            tileA[ty][tx] = 0;
        }
       	size_t trow = tileId * MATMUL_TILE_WIDTH + ty;
        if (col < numBColumns && trow < numBRows) {
        	// reverse engineer the original loop
        	int ch = trow / kk;
        	int trmkk = trow - ch * kk; // use ch result to avoid multiple divisions
        	
        	int p = trmkk / K;
        	int q = trmkk - p * K; // again

        	int h_original = h + p;
        	int w_original = w + q;
        	// (batch, channel, y, x)
            tileB[ty][tx] = input[
            	b * chwin +
            	ch * hwin +
            	h_original * Width +
            	w_original
            ];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
        	#pragma unroll
            for (int i = 0; i < MATMUL_TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
    	// (b, m, h, w)
	    size_t idx = b * mhw +
	    			 row * hw +
	    			 h * Width_out + w;
        output[idx] = val;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const size_t Height_out = Height - K + 1;
    const size_t Width_out = Width - K + 1;
	const size_t input_size = Batch * Channel * Height * Width * sizeof(float);
	const size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
	const size_t mask_size = Map_out * Channel * K * K * sizeof(float);

	cudaMalloc(device_input_ptr, input_size);
	cudaMalloc(device_output_ptr, output_size);
	cudaMalloc(device_mask_ptr, mask_size);

	cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Set the kernel dimensions and call the fused kernel
    // Set the kernel dimensions and call the kernel
    const size_t Height_out = Height - K + 1;
    const size_t Width_out = Width - K + 1;
    size_t out_rows = Map_out;
    size_t out_cols = Batch * Height_out * Width_out;

	// using parallel specs from ch. 16 pg. 356
	dim3 DimGrid(
		ceil(out_cols / (float) MATMUL_TILE_WIDTH), 
		ceil(out_rows / (float) MATMUL_TILE_WIDTH),
		1
	);
	dim3 DimBlock(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);

	/*printf("launching conv kernel with (%d, %d, %d)\n", Batch, Map_out, block_count);*/

	matmul_conv_fused<<<DimGrid, DimBlock>>>(device_mask, device_input, device_output,
																			 Batch, Map_out, Channel,
																			 Height, Width, K);
																			
	cudaDeviceSynchronize();
	// Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
         std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
         exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
	    // Copy the output back to host
    const size_t Height_out = Height - K + 1;
    const size_t Width_out = Width - K + 1;
	const size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    /*// print first output layer
    printf("layer 0:\n");
    for (int i = 0; i < Width_out; i++) {
    	for (int j = 0; j < Height_out; j++) {
    		printf("%.4f ", host_output[i + j * Width_out]);
    	}
    	printf("\n");
    }*/

    // Free device memory
	cudaFree(device_input);
	cudaFree(device_output);
	cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
