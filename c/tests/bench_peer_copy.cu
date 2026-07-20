#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_OK(call) do {                                                   \
    cudaError_t e_ = (call);                                                 \
    if (e_ != cudaSuccess) {                                                 \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                     cudaGetErrorString(e_));                                \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

static void enable_peer(int device, int peer) {
    CUDA_OK(cudaSetDevice(device));
    int accessible = 0;
    CUDA_OK(cudaDeviceCanAccessPeer(&accessible, device, peer));
    if (!accessible) {
        std::fprintf(stderr, "GPU %d cannot access GPU %d\n", device, peer);
        std::exit(1);
    }
    cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
        CUDA_OK(e);
    if (e == cudaErrorPeerAccessAlreadyEnabled)
        (void)cudaGetLastError();
}

static void measure(int src_device, int dst_device, size_t bytes, int iterations) {
    void *src = nullptr, *dst = nullptr;
    CUDA_OK(cudaSetDevice(src_device));
    CUDA_OK(cudaMalloc(&src, bytes));
    CUDA_OK(cudaMemset(src, 0x5a, bytes));
    CUDA_OK(cudaSetDevice(dst_device));
    CUDA_OK(cudaMalloc(&dst, bytes));

    cudaStream_t stream;
    cudaEvent_t begin, end;
    CUDA_OK(cudaStreamCreate(&stream));
    CUDA_OK(cudaEventCreate(&begin));
    CUDA_OK(cudaEventCreate(&end));
    CUDA_OK(cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream));
    CUDA_OK(cudaStreamSynchronize(stream));

    CUDA_OK(cudaEventRecord(begin, stream));
    for (int i = 0; i < iterations; ++i)
        CUDA_OK(cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream));
    CUDA_OK(cudaEventRecord(end, stream));
    CUDA_OK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&milliseconds, begin, end));

    const double seconds = milliseconds / 1000.0;
    const double gib = (double)bytes * iterations / (1024.0 * 1024.0 * 1024.0);
    std::printf("GPU%d -> GPU%d: %.3f GiB x %d, %.3f ms/copy, %.2f GiB/s\n",
                src_device, dst_device, (double)bytes / (1024.0 * 1024.0 * 1024.0),
                iterations, milliseconds / iterations, gib / seconds);

    CUDA_OK(cudaEventDestroy(end));
    CUDA_OK(cudaEventDestroy(begin));
    CUDA_OK(cudaStreamDestroy(stream));
    CUDA_OK(cudaFree(dst));
    CUDA_OK(cudaSetDevice(src_device));
    CUDA_OK(cudaFree(src));
}

int main(int argc, char **argv) {
    const long long tokens = argc > 1 ? std::atoll(argv[1]) : 23606;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 3;
    if (tokens <= 0 || iterations <= 0) {
        std::fprintf(stderr, "usage: %s [tokens [iterations]]\n", argv[0]);
        return 2;
    }
    int count = 0;
    CUDA_OK(cudaGetDeviceCount(&count));
    if (count < 2) {
        std::fprintf(stderr, "two CUDA devices are required\n");
        return 1;
    }
    enable_peer(0, 1);
    enable_peer(1, 0);
    const size_t bytes = (size_t)tokens * 6144u * sizeof(float);
    std::printf("FP32 residual boundary for %lld tokens (%zu bytes)\n", tokens, bytes);
    measure(0, 1, bytes, iterations);
    measure(1, 0, bytes, iterations);
    return 0;
}
