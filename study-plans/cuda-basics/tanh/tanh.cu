#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id>=N) return;
    float exp = expf(input[id]);
    float nexp = expf(-input[id]);
    output[id] = ((exp - nexp) / (exp + nexp));
    // output[id] = tanh(input[id]);
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}