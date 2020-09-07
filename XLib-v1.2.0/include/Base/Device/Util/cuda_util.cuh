/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#pragma once

#include <string>
#include <cuda_runtime.h>
#include <limits>

#if !defined(NO_CHECK_CUDA_ERROR)
#define __CUDA_ERROR(msg)                                                      \
                    {                                                          \
                        cudaDeviceSynchronize();                               \
                        xlib::__getLastCudaError (msg, __FILE__, __LINE__);    \
                    }
#else
#define __CUDA_ERROR(msg)
#endif

#define __SAFE_CALL(function)                                                  \
    {                                                                          \
        __safe_call(function, __FILE__, __LINE__);                             \
    }



#define    def_SWITCH(x)    switch (x) {                                       \
                            case -1:                                           \
                            case 0:    fun(1)                                  \
                                break;                                         \
                            case 1: fun(2)                                     \
                                break;                                         \
                            case 2: fun(4)                                     \
                                break;                                         \
                            case 3: fun(8)                                     \
                                break;                                         \
                            case 4: fun(16)                                    \
                                break;                                         \
                            case 5: fun(32)                                    \
                                break;                                         \
                            case 6: fun(64)                                    \
                                break;                                         \
                            case 7: fun(128)                                   \
                                break;                                         \
                            default: fun(32)                                   \
                        }

#define    def_SWITCHB(x)    switch (x) {                                      \
                            case -1:                                           \
                            case 0:    funB(1)                                 \
                                break;                                         \
                            case 1: funB(2)                                    \
                                break;                                         \
                            case 2: funB(4)                                    \
                                break;                                         \
                            case 3: funB(8)                                    \
                                break;                                         \
                            case 4: funB(16)                                   \
                                break;                                         \
                            case 5: funB(32)                                   \
                                break;                                         \
                            case 6: funB(64)                                   \
                                break;                                         \
                            case 7: funB(128)                                  \
                                break;                                         \
                            default: fun(32)                                   \
                        }

namespace xlib {

//iteratorB_t = device

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equalCuda(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B);

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equalCuda(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
        bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type));

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equalSortedCuda(iteratorA_t start_A, iteratorA_t end_A,
                     iteratorB_t start_B);


void __getLastCudaError(const char *errorMessage, const char *file,
                               int line);

void __safe_call(cudaError_t err, const char *file, int line);

class deviceProperty {
    public:
        static int getNum_of_SMs();
    private:
        static int NUM_OF_STREAMING_MULTIPROCESSOR;
};

template<class T>
inline int gridConfig(T FUN, const int block_dim,
                      const int dyn_shared_mem = 0,
                      const int problem_size = std::numeric_limits<int>::max());

bool memInfoCUDA(std::size_t Req);
void memCheckCUDA(std::size_t Req);
void cudaStatics();

template<typename T>
__global__ void scatter(const int* __restrict__ toScatter,
                        const int scatter_size,
                        T*__restrict__ Dest,
                        const T value);

template<typename T>
__global__ void fill(T* devArray, const int fill_size, const T value);

template <typename T>
__global__ void fill(T* devMatrix, const int n_of_rows, const int n_of_columns,
                     const T value, int integer_pitch = 0);

} //@xlib

namespace NVTX {

const int GREEN = 0x0000ff00, BLUE = 0x000000ff, YELLOW = 0x00ffff00,
          PURPLE = 0x00ff00ff, CYAN = 0x0000ffff, RED = 0x00ff0000,
          WHITE = 0x00ffffff;

void PushRange(std::string s, const int color);
void PopRange();

} //@NNVTX

#include "impl/cuda_util.i.cuh"
