#include <iomanip>
#include <string>
//#include <cuda_runtime.h>

#include "Base/Host/fUtil.hpp"

#if defined(LITERAL)
constexpr int operator"" ULL_BIT ( unsigned long long value ) {
    return value;
}
constexpr size_t operator"" ULL_KB ( unsigned long long value ) {
    return static_cast<size_t>(value) * 1024;
}
constexpr size_t operator"" ULL_MB ( unsigned long long value ) {
    return static_cast<size_t>(value) * 1024 * 1024;
}
#endif

namespace xlib {

template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool equalCuda(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B) {
    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    R* ArrayCMP = new R[size];
    cudaMemcpy(ArrayCMP, &(*start_B), size * sizeof(R), cudaMemcpyDeviceToHost);
    __CUDA_ERROR("Copy To Host");

    bool flag = xlib::equal<FAULT>(start_A, end_A, ArrayCMP);
    delete[] ArrayCMP;
    return flag;
}

template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool equalCuda(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
        bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type)) {

    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    R* ArrayCMP = new R[size];
    cudaMemcpy(ArrayCMP, &(*start_B), size * sizeof(R), cudaMemcpyDeviceToHost);
    __CUDA_ERROR("Copy To Host");

    bool flag = xlib::equal<FAULT>(start_A, end_A, ArrayCMP, equalFunction);
    delete[] ArrayCMP;
    return flag;
}

template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool equalSortedCuda(iteratorA_t start_A, iteratorA_t end_A,
                     iteratorB_t start_B) {
    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    R* ArrayCMP = new R[size];
    cudaMemcpy(ArrayCMP, &(*start_B), size * sizeof(R), cudaMemcpyDeviceToHost);
    __CUDA_ERROR("Copy To Host");

    bool flag = xlib::equalSorted<FAULT>(start_A, end_A, ArrayCMP);
    delete[] ArrayCMP;
    return flag;
}

template<class T>
inline int gridConfig(T FUN,
                      const int block_dim,
                      const int dyn_shared_mem,
                      const int problem_size) {
    int num_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, FUN,
                                                  block_dim, dyn_shared_mem);
    return std::min(deviceProperty::getNum_of_SMs() * num_blocks, problem_size);
}

template<typename T>
__global__ void scatter(const int* __restrict__ toScatter,
                        const int scatter_size,
                        T*__restrict__ Dest,
                        const T value) {

    const int ID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = ID; i < scatter_size; i += blockDim.x * gridDim.x)
        Dest[ toScatter[i] ] = value;
}

template<typename T>
__global__ void fill(T* devArray, const int fill_size, const T value) {

    const int ID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = ID; i < fill_size; i += blockDim.x * gridDim.x)
        devArray[ i ] = value;
}


template <typename T>
__global__ void fill(T* devMatrix, const int n_of_rows, const int n_of_columns,
                     const T value, int integer_pitch) {

    const int X = blockDim.x * blockIdx.x + threadIdx.x;
    const int Y = blockDim.y * blockIdx.y + threadIdx.y;
    if (integer_pitch == 0)
        integer_pitch = n_of_columns;

    for (int i = Y; i < n_of_rows; i += blockDim.y * gridDim.y) {
        for (int j = X; j < n_of_columns; j += blockDim.x * gridDim.x)
            devMatrix[i * integer_pitch + j] = value;
    }
}

} //@xlib
