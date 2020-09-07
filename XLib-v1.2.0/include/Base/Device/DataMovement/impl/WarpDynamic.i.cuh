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
#include "Base/Device/Util/basic.cuh"
#include "Base/Device/Util/PTX.cuh"
#include "Base/Device/Util/definition.cuh"
#include "Base/Device/Primitives/WarpScan.cuh"
#include "Base/Device/Primitives/WarpReduce.cuh"

namespace xlib {

template<cuQUEUE_MODE mode, cub::CacheStoreModifier CM, int Items_per_warp>
template<typename T, typename R, int SIZE>
__device__ __forceinline__
void QueueWarp<mode, CM, Items_per_warp>
::store(T (&Queue)[SIZE],
        const int size,
        T* __restrict__ queue_ptr,
        R* __restrict__ queue_size_ptr,
        T* __restrict__ SMem) {

    int thread_offset = size, total;
    int warp_offset = WarpExclusiveScan<>::AtomicAdd(thread_offset,
                                                     queue_size_ptr,
                                                     total);
    warp_dyn<mode, CM, Items_per_warp>
        ::regToGlobal(Queue, size, thread_offset, queue_ptr + warp_offset,
                      total, SMem + WarpID() * Items_per_warp);
}

//------------------------------------------------------------------------------

template<cub::CacheStoreModifier CM, int Items_per_warp>
struct warp_dyn<cuQUEUE_MODE::SIMPLE, CM, Items_per_warp> {

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            const int size,
                            int thread_offset,
                            T* __restrict__ devPointer,
                            const int total,          //optional
                            T* __restrict__ SMem) {   //optional

        devPointer += thread_offset;
        for (int i = 0; i < size; i++)
            cub::ThreadStore<CM>(devPointer + i, Queue[i]);
    }
};


template<cub::CacheStoreModifier CM, int Items_per_warp>
struct warp_dyn<cuQUEUE_MODE::UNROLL, CM, Items_per_warp> {

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                     const int size,
                     int thread_offset,
                     T* __restrict__ devPointer,
                     const int total,          //optional
                     T* __restrict__ SMem) {   //optional

        devPointer += thread_offset;
        #pragma unroll
        for (int i = 0; i < SIZE; i++) {
            if (i < size)
                cub::ThreadStore<CM>(devPointer + i, Queue[i]);
        }
    }
};

template<cub::CacheStoreModifier CM, int Items_per_warp>
struct warp_dyn<cuQUEUE_MODE::SHAREDMEM, CM, Items_per_warp> {
    //static_assert(Items_per_warp != 0, PRINT_ERR("Items_per_warp == 0"));

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            const int size,
                            int thread_offset,
                            T* __restrict__ devPointer,
                            int total,
                            T* __restrict__ SMem) {

        T* SMemTMP = SMem;
        devPointer += LaneID();
        SMem += LaneID();
        int j = 0;
         while (true) {
             while (j < size && thread_offset < Items_per_warp) {
                 SMemTMP[thread_offset] = Queue[j];
                 j++;
                 thread_offset++;
             }
            if (total < Items_per_warp) {
                #pragma unroll
                for (int i = 0; i < Items_per_warp; i += WARP_SIZE) {
                    if (LaneID() + i < total)
                        cub::ThreadStore<CM>(devPointer + i, SMem[i]);
                }
                break;
            }
            else {
                #pragma unroll
                for (int i = 0; i < Items_per_warp; i += WARP_SIZE)
                    cub::ThreadStore<CM>(devPointer + i, SMem[i]);
            }
            total -= Items_per_warp;
            thread_offset -= Items_per_warp;
            devPointer += Items_per_warp;
         }
    }
};

template<cub::CacheStoreModifier CM, int Items_per_warp>
struct warp_dyn<cuQUEUE_MODE::MIN, CM, Items_per_warp> {
    static_assert(Items_per_warp != 0, PRINT_ERR("Items_per_warp == 0"));

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            const int size,
                            int thread_offset,
                            T* __restrict__ devPointer,
                            const int total,          //optional
                            T* __restrict__ SMem) {   //optional

        int minValue = size;
        WarpReduce<>::MinBcast(minValue);

        T* devPointerTMP = devPointer + LaneID();
        for (int i = 0; i < minValue; i++)
            cub::ThreadStore<CM>(devPointerTMP + i * WARP_SIZE, Queue[i]);

        size -= minValue;
        thread_offset -= LaneID() * minValue;
        total -= minValue * WARP_SIZE;
        devPointer += minValue * WARP_SIZE;

        RegToGlobal(Queue + minValue, size, thread_offset, total,
                    SMem, devPointer);
    }
};


template<cub::CacheStoreModifier CM, int Items_per_warp>
struct QueueWarp<cuQUEUE_MODE::BALLOT, CM, Items_per_warp> {
//    static_assert(Items_per_warp != 0, PRINT_ERR("Items_per_warp == 0"));

    template<typename T, typename R>
    __device__ __forceinline__
    static void store(T value, T* __restrict__ queue_ptr,
                      R* __restrict__ queue_size_ptr) {

        int predicate = value != -1;
        unsigned ballot = __ballot(predicate);
        //if (ballot) {
            int warp_offset;
            if (LaneID() == 0)
                warp_offset = atomicAdd(queue_size_ptr, __popc(ballot));
            int th_offset = __popc(ballot & LaneMaskLT()) +
                            __shfl(warp_offset, 0);
            if (predicate)
                queue_ptr[th_offset] = value;
        //}
    }
};

} //@xlib
