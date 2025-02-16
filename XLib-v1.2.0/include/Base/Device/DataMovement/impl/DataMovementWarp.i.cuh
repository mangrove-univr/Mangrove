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
#include "../../../Util/Util.cuh"
using namespace PTX;

namespace data_movement {
namespace warp {

/**
 * not documented
 * @tparam SIZE local thread size
 *//*
template<MEM_SPACE _MEM_SPACE, int SIZE, typename T>
void __device__ __forceinline__ computeOffset(T* __restrict__ &Pointer) {
    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 : 1;

    if (_MEM_SPACE == GLOBAL && SIZE % AGGR_SIZE_16 == 0)
        Pointer += LaneID() * AGGR_SIZE_16;
    else if (SIZE % AGGR_SIZE_8 == 0)
        Pointer += LaneID() * AGGR_SIZE_8;
    else if (SIZE % AGGR_SIZE_4 == 0)
        Pointer += LaneID() * AGGR_SIZE_4;
    else if (SIZE % AGGR_SIZE_2 == 0)
        Pointer += LaneID() * AGGR_SIZE_2;
    else
        Pointer += LaneID();
}*/


template<MEM_SPACE _MEM_SPACE, int SIZE, typename T>
void __device__ __forceinline__ computeOffset(T* __restrict__ &Pointer) {
    const int SIZE_BYTE = SIZE * sizeof(T);

    if (_MEM_SPACE == GLOBAL && SIZE_BYTE % 16 == 0)
        Pointer += LaneID() * (16 / sizeof(T));
    else if (SIZE_BYTE % 8 == 0)
        Pointer += LaneID() * (8 / sizeof(T));
    else if (SIZE_BYTE % 4 == 0)
        Pointer += LaneID() * (4 / sizeof(T));
    else if (SIZE_BYTE % 2 == 0)
        Pointer += LaneID() * (2 / sizeof(T));
    else
        Pointer += LaneID();
}

template<MEM_SPACE _MEM_SPACE, int SIZE, typename T>
void __device__ __forceinline__ computeStride(int& stride) {
    const int SIZE_BYTE = SIZE * sizeof(T);

    if (_MEM_SPACE == GLOBAL && SIZE_BYTE % 16 == 0)
        stride *= (16 / sizeof(T));
    else if (SIZE_BYTE % 8 == 0)
        stride *= (8 / sizeof(T));
    else if (SIZE_BYTE % 4 == 0)
        stride *= (4 / sizeof(T));
    else if (SIZE_BYTE % 2 == 0)
        stride *= (2 / sizeof(T));
}

/*
template<int SIZE, typename T>
void __device__ __forceinline__ computeGlobalOffset(T* __restrict__ &SMem_ptr,
                                                    T* __restrict__ &Glob_ptr) {

    const int SIZE_BYTE = SIZE * sizeof(T);
    if (SIZE_BYTE % 16 == 0) {
        SMem_ptr += LaneID() * (16 / sizeof(T));
        Glob_ptr += LaneID() * (16 / sizeof(T));
    }
    else if (SIZE_BYTE % 8 == 0) {
        SMem_ptr += LaneID() * (8 / sizeof(T));
        Glob_ptr += LaneID() * (8 / sizeof(T));
    }
    else if (SIZE_BYTE % 4 == 0) {
        SMem_ptr += LaneID() * (4 / sizeof(T));
        Glob_ptr += LaneID() * (4 / sizeof(T));
    }
    else if (SIZE_BYTE % 2 == 0) {
        SMem_ptr += LaneID() * (2 / sizeof(T));
        Glob_ptr += LaneID() * (2 / sizeof(T));
    }
    else {
        SMem_ptr += LaneID();
        Glob_ptr += LaneID();
    }
}*/

//==============================================================================

template<int SIZE, typename T>
struct bestGlobalAlignStr {
    using type = typename std::conditional<(SIZE * sizeof(T)) % 16 == 0, int4,
              typename std::conditional<(SIZE * sizeof(T)) % 8 == 0, int2,
              typename std::conditional<(SIZE * sizeof(T)) % 4 == 0, int,
              typename std::conditional<(SIZE * sizeof(T)) % 2 == 0, short,
              char>::type>::type>::type>::type;
};

template<int SIZE, typename T>
struct bestSharedAlignStr {
    using type = typename std::conditional<(SIZE * sizeof(T)) % 8 == 0, int2,
              typename std::conditional<(SIZE * sizeof(T)) % 4 == 0, int,
              typename std::conditional<(SIZE * sizeof(T)) % 2 == 0, short,
              char>::type>::type>::type;
};

template<int SIZE, typename T,
         typename = typename std::enable_if<std::is_signed<T>::value>::type >
auto __device__ __forceinline__ bestGlobalAlign(T* var) ->
                                    typename bestGlobalAlignStr<SIZE, T>::type* {

    using R = typename bestGlobalAlignStr<SIZE, T>::type;
    return reinterpret_cast<R*>(var);
}

template<int SIZE, typename T,
         typename = typename std::enable_if<std::is_signed<T>::value>::type >
auto __device__ __forceinline__ bestSharedAlign(T* var) ->
                                    typename bestSharedAlignStr<SIZE, T>::type* {

    using R = typename bestSharedAlignStr<SIZE, T>::type;
    return reinterpret_cast<R*>(var);
}


template<cub::CacheStoreModifier M, int SIZE, typename T>
void __device__ __forceinline__ SharedToGlobalSupport(T* __restrict__ SMem,
                                                      T* __restrict__ Pointer) {

    const int SIZE_CHAR = SIZE * sizeof(T);
    static_assert(SIZE_CHAR % WARP_SIZE == 0,
                "SharedToGlobalSupport : SIZE must be divisible by WARP_SIZE");

    const int WARP_CHUNK_16 = 16 * WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < SIZE_CHAR / WARP_CHUNK_16; i++)
        cub::ThreadStore<M>(reinterpret_cast<int4*>(Pointer) + i * WARP_SIZE,
                            reinterpret_cast<int4*>(SMem)[i * WARP_SIZE]);

    const int REMAIN_16 = SIZE_CHAR % WARP_CHUNK_16;
    const int START_8 = ((SIZE_CHAR / WARP_CHUNK_16) * WARP_CHUNK_16) / sizeof(int2);
    const int WARP_CHUNK_8 = 8 * WARP_SIZE;
    if (REMAIN_16 >= WARP_CHUNK_8)
        cub::ThreadStore<M>(reinterpret_cast<int2*>(Pointer) + START_8,
                            reinterpret_cast<int2*>(SMem)[START_8]);

    const int REMAIN_8 = SIZE_CHAR % WARP_CHUNK_8;
    const int START_4 = ((SIZE_CHAR / WARP_CHUNK_8) * WARP_CHUNK_8) / sizeof(int);
    const int WARP_CHUNK_4 = 4 * WARP_SIZE;
    if (REMAIN_8 >= WARP_CHUNK_4)
        cub::ThreadStore<M>(reinterpret_cast<int*>(Pointer) + START_4,
                            reinterpret_cast<int*>(SMem)[START_4]);

    const int REMAIN_4 = SIZE_CHAR % WARP_CHUNK_4;
    const int START_2 = ((SIZE_CHAR / WARP_CHUNK_4) * WARP_CHUNK_4) / sizeof(short);
    const int WARP_CHUNK_2 = 2 * WARP_SIZE;
    if (REMAIN_4 >= WARP_CHUNK_2)
        cub::ThreadStore<M>(reinterpret_cast<short*>(Pointer) + START_2,
                            reinterpret_cast<short*>(SMem)[START_2]);

    const int REMAIN_2 = SIZE_CHAR % WARP_CHUNK_2;
    const int START_1 = (SIZE_CHAR / WARP_CHUNK_2) * WARP_CHUNK_2;
    if (REMAIN_2 >= 1)
        cub::ThreadStore<M>(Pointer + START_1, SMem[START_1]);
}

template<cub::CacheLoadModifier M, int SIZE, typename T>
void __device__ __forceinline__ GlobalToSharedSupport(T* __restrict__ Pointer,
                                                      T* __restrict__ SMem) {

    const int SIZE_CHAR = SIZE * sizeof(T);
    static_assert(SIZE_CHAR % WARP_SIZE == 0,
            "SharedToGlobalSupport : SIZE must be divisible by WARP_SIZE");

    const int WARP_CHUNK_16 = 16 * WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < SIZE_CHAR / WARP_CHUNK_16; i++)
       reinterpret_cast<int4*>(SMem)[i * WARP_SIZE] =
           cub::ThreadLoad<M>(reinterpret_cast<int4*>(Pointer) + i * WARP_SIZE);

    const int REMAIN_16 = SIZE_CHAR % WARP_CHUNK_16;
    const int START_8 = ((SIZE_CHAR / WARP_CHUNK_16) * WARP_CHUNK_16) / sizeof(int2);
    const int WARP_CHUNK_8 = 8 * WARP_SIZE;
    if (REMAIN_16 >= WARP_CHUNK_8)
        reinterpret_cast<int2*>(SMem)[START_8] =
            cub::ThreadLoad<M>(reinterpret_cast<int2*>(Pointer) + START_8);

    const int REMAIN_8 = SIZE_CHAR % WARP_CHUNK_8;
    const int START_4 = ((SIZE_CHAR / WARP_CHUNK_8) * WARP_CHUNK_8) / sizeof(int);
    const int WARP_CHUNK_4 = 4 * WARP_SIZE;
    if (REMAIN_8 >= WARP_CHUNK_4)
        reinterpret_cast<int*>(SMem)[START_4] =
            cub::ThreadLoad<M>(reinterpret_cast<int*>(Pointer) + START_4);

    const int REMAIN_4 = SIZE_CHAR % WARP_CHUNK_4;
    const int START_2 = ((SIZE_CHAR / WARP_CHUNK_4) * WARP_CHUNK_4) / sizeof(short);
    const int WARP_CHUNK_2 = 2 * WARP_SIZE;
    if (REMAIN_4 >= WARP_CHUNK_2)
        reinterpret_cast<short*>(SMem)[START_2] =
            cub::ThreadLoad<M>(reinterpret_cast<short*>(Pointer) + START_2);

    const int REMAIN_2 = SIZE_CHAR % WARP_CHUNK_2;
    const int START_1 = (SIZE_CHAR / WARP_CHUNK_2) * WARP_CHUNK_2;
    if (REMAIN_2 >= 1)
        SMem[START_1] = cub::ThreadLoad<M>(Pointer + START_1);
}

//==============================================================================

template<int SIZE, cub::CacheStoreModifier M, typename T>
void __device__ __forceinline__ SharedToGlobal(T* __restrict__ SMem,
                                               T* __restrict__ Pointer) {

    computeGlobalOffset<SIZE / WARP_SIZE>(SMem, Pointer);
    SharedToGlobalSupport<M, SIZE>(SMem, Pointer);
}

template<int SIZE, cub::CacheLoadModifier M, typename T>
void __device__ __forceinline__ GlobalToShared(T* __restrict__ Pointer,
                                               T* __restrict__ SMem) {

    computeGlobalOffset<SIZE / WARP_SIZE>(SMem, Pointer);
    GlobalToSharedSupport<M, SIZE>(Pointer, SMem);
}

} //@warp
} //@data_movement
