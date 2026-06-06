#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    __shared__ float sdata[256];
    //find max
    float local_max = -INFINITY;
    for(int i =tid;i<N;i+=stride)
        local_max = fmaxf(local_max,input[i]);
    sdata[tid] = local_max;
    __syncthreads();
     for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }

    float max_val = sdata[0];

    // 2) Compute sum of exp(input[i] - max)
    float local_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        local_sum += expf(input[i] - max_val);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    float sum_val = sdata[0];

    // 3) Write normalized output
    for (int i = tid; i < N; i += stride) {
        output[i] = expf(input[i] - max_val) / sum_val;
    }
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}