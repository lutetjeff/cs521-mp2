#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// example
#define MATMUL_TILE_WIDTH 16

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C_out x out_h x out_w
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w
) {
    __shared__ float tileA[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];  // shared memory for partial sums
    __shared__ float tileB[MATMUL_TILE_WIDTH][MATMUL_TILE_WIDTH];
    
    // TO DO : Tiled matrix multiplication by using shmem
    const int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    const int row = by * MATMUL_TILE_WIDTH + ty, col = bx * MATMUL_TILE_WIDTH + tx;

    const int hw = out_h * out_w;
    const size_t mhw = (size_t)hw * C_out;
    const int kk = KH * KW;

    const int b = col / hw;
    const int clmhw = col - b * hw;
    const int h = clmhw / out_w;
    const int w_out = clmhw - h * out_w;
    const int hwin = H * W;
    const size_t chwin = (size_t)C_in * hwin;

    const size_t numARows = C_out;
    const size_t numAColumns = (size_t)C_in * KH * KW;
    const size_t numBRows = numAColumns;
    const size_t numBColumns = (size_t)N * hw;
    const size_t numCRows = numARows;
    const size_t numCColumns = numBColumns;

    float val = 0;

    #pragma unroll 2
    for (int tileId = 0; tileId < (numAColumns - 1) / MATMUL_TILE_WIDTH + 1; tileId++) {
        size_t tcol = tileId * MATMUL_TILE_WIDTH + tx;
        if (row < numARows && tcol < numAColumns) {
            tileA[ty][tx] = w[row * numAColumns + tcol];
        } else {
            tileA[ty][tx] = 0;
        }
        size_t trow = tileId * MATMUL_TILE_WIDTH + ty;
        if (col < numBColumns && trow < numBRows) {
            int ch = trow / kk;
            int trmkk = trow - ch * kk;

            int p = trmkk / KW;
            int q = trmkk - p * KW;

            int h_original = h * stride - pad + p;
            int w_original = w_out * stride - pad + q;
            tileB[ty][tx] = (h_original >= 0 && h_original < H &&
                            w_original >= 0 && w_original < W) ? x[
                b * chwin +
                ch * hwin +
                h_original * W +
                w_original
            ] : 0;
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
        size_t idx = b * mhw +
                     row * hw +
                     h * out_w + w_out;
        out[idx] = val;
    }
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                          int stride, int pad) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
    TORCH_CHECK(x.device() == w.device(), "x and w must be on the same device");
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4, "x and w must be 4D tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && w.scalar_type() == torch::kFloat32,
                "x and w must be float32 tensors");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "x and w must be contiguous");
    TORCH_CHECK(x.size(1) == w.size(1), "input channels must match weight channels");
    TORCH_CHECK(stride > 0 && pad >= 0, "stride must be positive and padding nonnegative");
    const c10::cuda::CUDAGuard device_guard(x.device());

    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    TORCH_CHECK(C_in > 0 && C_out > 0 && H > 0 && W > 0 && KH > 0 && KW > 0,
                "channel, spatial, and kernel dimensions must be positive");
    TORCH_CHECK(H + 2 * pad >= KH && W + 2 * pad >= KW,
                "kernel must fit within the padded input");
    int out_h = (H + 2 * pad - KH) / stride + 1;
    int out_w = (W + 2 * pad - KW) / stride + 1;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());
    if (N == 0) return out;

    size_t out_rows = C_out;
    size_t out_cols = (size_t)N * out_h * out_w;
    dim3 block(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH);
    dim3 grid((out_cols + block.x - 1)/block.x,
              (out_rows + block.y - 1)/block.y,
              1);

    gemm_gpu_o4_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
