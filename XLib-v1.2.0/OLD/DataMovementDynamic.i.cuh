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
#include "Base/Device/Util/definition.cuh"
#include "Base/Device/Util/basic.cuh"
#include "Base/Device/Util/PTX.cuh"
#include "Base/Device/Primitives/WarpReduce.cuh"
#include "Base/Device/Primitives/WarpScan.cuh"

namespace xlib {

template<cub::CacheStoreModifier CM>
struct thread_dyn<THREAD_MODE::SIMPLE, CM> {

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            const int size,
                            T* __restrict__ devPointer) {

        for (int i = 0; i < size; i++)
            cub::ThreadStore<CM>(devPointer + i, Queue[i]);
    }
};

template<cub::CacheStoreModifier CM>
struct thread_dyn<THREAD_MODE::UNROLL, CM> {

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            const int size,
                            T* __restrict__ devPointer) {

        #pragma unroll
        for (int i = 0; i < SIZE; i++) {
            if (i < size)
                cub::ThreadStore<CM>(devPointer + i, Queue[i]);
        }
    }
};

//==============================================================================

template<int BlockSize, cub::CacheStoreModifier CM, int Items_per_block>
struct block_dyn<BlockSize, BLOCK_MODE::SIMPLE, CM, Items_per_block> {

    template<typename T, int SIZE>
    __device__ __forceinline__ void regToGlobal(T (&Queue)[SIZE],
                                                const int size,
                                                int thread_offset,
                                                const int total,
                                                T* __restrict__ SMem,
                                                T* __restrict__ devPointer) {

        devPointer += thread_offset;
        for (int i = 0; i < size; i++)
            cub::ThreadStore<CM>(devPointer + i, Queue[i]);
    }
};

template<int BlockSize, cub::CacheStoreModifier CM, int Items_per_block>
struct block_dyn<BlockSize, BLOCK_MODE::UNROLL, CM, Items_per_block> {

    template<typename T, int SIZE>
    __device__ __forceinline__ void regToGlobal(T (&Queue)[SIZE],
                                                const int size,
                                                int thread_offset,
                                                const int total,
                                                T* __restrict__ SMem,
                                                T* __restrict__ devPointer) {

        devPointer += thread_offset;
        #pragma unroll
        for (int i = 0; i < SIZE; i++) {
            if (i < size)
                cub::ThreadStore<CM>(devPointer + i, Queue[i]);
        }
    }
};

template<int BlockSize, cub::CacheStoreModifier CM, int Items_per_block>
struct block_dyn<BlockSize, BLOCK_MODE::SHAREDMEM, CM, Items_per_block> {

    template<typename T, int SIZE>
    __device__ __forceinline__ void regToGlobal(T (&Queue)[SIZE],
                                                const int size,
                                                int thread_offset,
                                                const int total,
                                                T* __restrict__ SMem,
                                                T* __restrict__ devPointer) {
        int j = 0;
     	while (true) {
     		while (j < size && thread_offset < Items_per_block) {
     			SMem[thread_offset] = Queue[j];
     			j++;
                thread_offset++;
     		}
            __syncthreads();
     		#pragma unroll
     		for (int i = 0; i < Items_per_block; i += BlockSize) {
     			const int index = threadIdx.x + i;
     			if (index < total)
     				cub::ThreadStore<CM>(devPointer + index, SMem[index]);
     		}
            total -= Items_per_block;
            if (total <= 0)
                 break;
     		thread_offset -= Items_per_block;
     		devPointer += Items_per_block;
            __syncthreads();
     	}
    }
};

}
