#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id>=N) return;
    float sqr = sqrt(2.0);
    output[id] = 0.5 * input[id] * (1 + erff(input[id]/sqr));
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
