
#if defined(__NVCC__)
    #define __HOST_DEVICE__ __host__ __device__ __forceinline__
#else
    #define __HOST_DEVICE__ inline
#endif
