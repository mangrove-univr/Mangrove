/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

Mangrove is provided under the terms of The MIT License (MIT):

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

#include <limits>
#include "DataTypes/TraceProp.cuh"
#include "XLib.hpp"
#include "config.cuh"

namespace mangrove {

template<typename T>
struct LIMITS {
    static const T MAX = std::numeric_limits<T>::max();
    static const T MIN = std::numeric_limits<T>::lowest();
};

template<typename PROP>
__global__ void GPUNumericRange(devTraceSTR<numeric_t> devProp) {
    using namespace xlib;
    using T = numeric_t;

    __shared__ T resultUp[32];
    __shared__ T resultDown[32];
    const int WarpID = threadIdx.x >> 5;
    const int N_WARPS = PROP::BlockSize / 32;
    const int QUEUE_SIZE = PROP::Steps * 8;

    const int vars = devProp.vars;
    const int sim_instants = devProp.template getRoundTraceLength<PROP>()
                             / sizeof(T);

    const int pitch = devProp.pitch / sizeof(T);
    T* blockTraceInit = devProp.dev_trace + threadIdx.x * 4;

    for (int BlockID = blockIdx.x; BlockID < vars; BlockID += gridDim.x) {

        T* blockTrace = blockTraceInit + BlockID * pitch;
        T rangeUp = LIMITS<T>::MIN, rangeDown = LIMITS<T>::MAX;

        for (int LOOP = 0; LOOP < sim_instants;
                 LOOP += PROP::BlockSize * QUEUE_SIZE) {
            T Queue[QUEUE_SIZE];

            #pragma unroll
            for (int K = 0; K < PROP::Steps; K++) {
                reinterpret_cast<float4*>(Queue)[K * 2] =
                                 reinterpret_cast<float4*>( blockTrace )
                                         [K * 2 * PROP::BlockSize];
                reinterpret_cast<float4*>(Queue)[K * 2 + 1] =
                                 __ldg( &reinterpret_cast<float4*>( blockTrace )
                                        [(K * 2 + 1) * PROP::BlockSize] );
            }

            #pragma unroll
            for (int i = 0; i < QUEUE_SIZE; i++) {
                rangeDown = min(rangeDown, Queue[i]);
                rangeUp = max(rangeUp, Queue[i]);
            }
            /*T TMP[QUEUE_SIZE];
            ADV::ThreadReductionOOP2<QUEUE_SIZE, T, ADV::MAX<T>,
                                        ADV::MIN<T>>::OP(Queue, TMP);
            rangeUp = max(rangeUp, TMP[0]);
            rangeDown = min(rangeDown, TMP[QUEUE_SIZE - 1]);*/

            blockTrace += PROP::BlockSize * QUEUE_SIZE;
        }
        WarpReduce<>::Max(rangeUp);
        WarpReduce<>::Min(rangeDown);
        if (LaneID() == 0) {
            resultDown[WarpID] = rangeDown;
            resultUp[WarpID] = rangeUp;
        }
        __syncthreads();
        if (threadIdx.x < N_WARPS) {
            T val = resultUp[LaneID()];
            WarpReduce<N_WARPS>::Max(val);
            if (LaneID() == 0)
                reinterpret_cast<T*>(devResult)[ BlockID * 2 + 1] = val;
        } else if (threadIdx.x >= 32 && threadIdx.x < 32 + N_WARPS) {
            T val = resultDown[LaneID()];
            WarpReduce<N_WARPS>::Min(val);
            if (LaneID() == 0)
                reinterpret_cast<T*>(devResult)[ BlockID * 2 ] = val;
        }
        __syncthreads();
    }
}

} //@mangrove
