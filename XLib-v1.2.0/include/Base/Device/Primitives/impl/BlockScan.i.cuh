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
#include "Base/Device/Primitives/WarpScan.cuh"
#include "Base/Device/Util/basic.cuh"
#include "Base/Device/Util/definition.cuh"

namespace xlib {

template<unsigned BLOCK_SIZE>
struct BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::ATOMIC> {
    template<typename T>
    __device__ __forceinline__
    static void BinaryAdd(int predicate, int& value, T* smem_total) {
        if (threadIdx.x < WARP_SIZE)
            *smem_total = 0;
        __syncthreads();

        const unsigned ballot = __ballot(predicate);
        int warp_offset;
        if (LaneID() == 0)
            warp_offset = atomicAdd(smem_total, __popc(ballot));
        value = __popc(ballot & LaneMaskLT()) + __shfl(warp_offset, 0);
        // !!!!! __syncthreads(); !!!!!
    }

    template<typename T>
    __device__ __forceinline__
    static void Add(T& value, T* shared_mem) {
        if (threadIdx.x < WARP_SIZE)
            *shared_mem = 0;
        __syncthreads();

        const T warp_offset = WarpExclusiveScan<>::AtomicAdd(value, shared_mem);
        value += warp_offset;
        __syncthreads();
    }
};

template<unsigned BLOCK_SIZE>
struct BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::REDUCE> {
    static_assert(BLOCK_SIZE != 0, PRINT_ERR("Missing template paramenter"));

    template<typename T>
    __device__ __forceinline__
    static void BinaryAdd(int predicate, int& value, T* shared_mem) {
        const unsigned N_OF_WARPS = BLOCK_SIZE / WARP_SIZE;
        const unsigned ballot = __ballot(predicate);
        const unsigned warp_id = WarpID();
        value = __popc(ballot & LaneMaskLT());
        shared_mem[warp_id] = value;
        __syncthreads();

        if (threadIdx.x < N_OF_WARPS) {
            T tmp = shared_mem[threadIdx.x];
            WarpExclusiveScan<N_OF_WARPS>::Add(tmp, shared_mem + N_OF_WARPS);
            shared_mem[threadIdx.x] = tmp;
        }
        __syncthreads();
        value += shared_mem[warp_id];
    }

    template<typename T>
    __device__ __forceinline__
    static void Add(T& value, T* shared_mem) {
        const unsigned N_OF_WARPS = BLOCK_SIZE / WARP_SIZE;
        const unsigned warp_id = WarpID();

        WarpExclusiveScan<>::Add(value, shared_mem + warp_id);
        __syncthreads();
        if (threadIdx.x < N_OF_WARPS) {
            T tmp = shared_mem[threadIdx.x];
            WarpExclusiveScan<N_OF_WARPS>::Add(tmp, shared_mem + N_OF_WARPS);
            shared_mem[threadIdx.x] = tmp;
        }
        __syncthreads();
        value += shared_mem[warp_id];
    }
};

} //@xlib
