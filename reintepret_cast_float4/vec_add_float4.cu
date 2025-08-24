#include <cassert>
#include <iostream>

constexpr int N = 1 << 20;
constexpr int BlockSize = 256;
constexpr int NumIterations = 100;
constexpr int FloatPerThread = 4;
constexpr int GridSize = ((N + FloatPerThread - 1) / FloatPerThread + BlockSize - 1) / BlockSize;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

__global__
void vec_add_float4(const float* a_ptr, const float* b_ptr, float* c_ptr, int n) {
    float a[FloatPerThread], b[FloatPerThread], c[FloatPerThread];

    const int idx = FloatPerThread * (blockDim.x * blockIdx.x + threadIdx.x);

    if (idx < n) {
        if (idx + FloatPerThread - 1 < n) {
            *reinterpret_cast<float4*>(a) = *reinterpret_cast<const float4*>(a_ptr + idx);
            *reinterpret_cast<float4*>(b) = *reinterpret_cast<const float4*>(b_ptr + idx);
        } else {
            for (int i = 0; i < FloatPerThread; ++i) {
                if (idx + i < n) {
                    a[i] = a_ptr[idx + i];
                    b[i] = b_ptr[idx + i];
                } else {
                    a[i] = 0.0f;
                    b[i] = 0.0f;
                }
            }
        }

        for (int i = 0; i < FloatPerThread; ++i) {
            c[i] = a[i] + b[i];
        }

        if (idx + FloatPerThread - 1 < n) {
            *reinterpret_cast<float4*>(c_ptr + idx) = *reinterpret_cast<const float4*>(c);
        } else {
            for (int i = 0; i < FloatPerThread; ++i) {
                if (idx + i < n) {
                    c_ptr[idx + i] = c[i];
                }
            }
        }
    }
}

int main() {
    size_t size = N * sizeof(float);

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    std::cout << "=== float4 version ===" << std::endl;
    std::cout << "GridSize: " << GridSize << ", BlockSize: " << BlockSize << ", N: " << N << std::endl;
    std::cout << "Running " << NumIterations << " iterations for benchmarking..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warmup run
    vec_add_float4<<<GridSize, BlockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < NumIterations; ++iter) {
        vec_add_float4<<<GridSize, BlockSize>>>(d_A, d_B, d_C, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float total_milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_milliseconds, start, stop));
    
    float avg_milliseconds = total_milliseconds / NumIterations;
    
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Total time for " << NumIterations << " iterations: " << total_milliseconds << " ms" << std::endl;
    std::cout << "Average kernel execution time: " << avg_milliseconds << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << (3 * N * sizeof(float)) / (avg_milliseconds * 1e6) << " GB/s" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    for (int i = 0; i < N; i++) {
        assert(h_C[i] == h_A[i] + h_B[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}