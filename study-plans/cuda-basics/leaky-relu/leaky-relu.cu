#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    if(tid>=N) return;
    output[tid] = (input[tid]>=0)? input[tid] : alpha*input[tid];
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}