#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    if(id>N) return;
    output[id] = (input[id]>0.0f) ? input[id] : 0.0f;
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}