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

#include "DataTypes/TraceProp.cuh"
#include "XLib.hpp"

namespace mangrove {

template<typename PROP>
__global__ void GPUUnaryBitCounting(devTraceSTR<bool> devProp) {
    using namespace xlib;
    const int QUEUE_SIZE = PROP::Steps * 8;

    const int vars = devProp.vars;
    const int length = devProp.getRoundTraceLength<PROP>() / sizeof(unsigned);

    const int pitch = devProp.pitch / sizeof(unsigned);
    unsigned* blockTraceTMP = devProp.dev_trace + threadIdx.x * 4;

    for (int BlockID = blockIdx.x; BlockID < vars; BlockID += gridDim.x) {
        unsigned* blockTrace = blockTraceTMP + BlockID * pitch;

        int counter = 0;
        for (int LOOP = 0; LOOP < length; LOOP += PROP::BlockSize * QUEUE_SIZE){
            int Queue[QUEUE_SIZE];

            #pragma unroll
            for (int K = 0; K < PROP::Steps; K++) {
                reinterpret_cast<int4*>(Queue)[K * 2] =
                  reinterpret_cast<int4*>( blockTrace )[K *2 * PROP::BlockSize];
                reinterpret_cast<int4*>(Queue)[K * 2 + 1] =
                  __ldg( &reinterpret_cast<int4*>( blockTrace )[(K * 2 + 1)
                                                            * PROP::BlockSize]);
            }
#if defined(ILP)
            #pragma unroll
            for (int i = 0; i < QUEUE_SIZE; i++)
                Queue[i] = __popc(Queue[i]);

            ThreadReduce::Add(Queue);
            counter += Queue[0];
#else
            #pragma unroll
            for (int i = 0; i < QUEUE_SIZE; i++)
                counter += __popc(Queue[i]);
#endif
            blockTrace += PROP::BlockSize * QUEUE_SIZE;
        }
        WarpReduce<>::AtomicAdd(counter,
                                reinterpret_cast<int*>(devResult + BlockID));
    }
}


template<typename PROP>
__global__ void GPUBinaryBitCounting(devTraceSTR<bool> devProp) {
    using namespace xlib;
    const unsigned QUEUE_SIZE = PROP::Steps * 8;

    const int length = devProp.getRoundTraceLength<PROP>() / sizeof(unsigned);
    const int pitch = devProp.pitch / sizeof(unsigned);

    unsigned* blockTrace_tmp = devProp.dev_trace + threadIdx.x * 4;

    for (int BlockID = blockIdx.x; BlockID < devProp.problem_size;
             BlockID += gridDim.x) {

        uchar2 dictionaryEntry = reinterpret_cast<uchar2*>(devDictionary)
                                                            [BlockID];
        unsigned* blockTraceA = blockTrace_tmp + dictionaryEntry.x * pitch;
        unsigned* blockTraceB = blockTrace_tmp + dictionaryEntry.y * pitch;

        int counter_AB = 0;
        for (int LOOP = 0; LOOP < length; LOOP += PROP::BlockSize * QUEUE_SIZE) {
            int QueueA[QUEUE_SIZE];
            int QueueB[QUEUE_SIZE];

            #pragma unroll
            for (int K = 0; K < PROP::Steps; K++) {
                reinterpret_cast<int4*>(QueueA)[K * 2] =
                                        reinterpret_cast<int4*>( blockTraceA )
                                        [K * 2 * PROP::BlockSize];
                reinterpret_cast<int4*>(QueueA)[K * 2 + 1] =
                                __ldg( &reinterpret_cast<int4*>( blockTraceA )
                                        [(K * 2 + 1) * PROP::BlockSize] );
                reinterpret_cast<int4*>(QueueB)[K * 2] =
                                        reinterpret_cast<int4*>( blockTraceB )
                                        [K * 2 * PROP::BlockSize];
                reinterpret_cast<int4*>(QueueB)[K * 2 + 1] =
                                __ldg( &reinterpret_cast<int4*>( blockTraceB )
                                        [(K * 2 + 1) * PROP::BlockSize] );
            }

            #pragma unroll
            for (int i = 0; i < QUEUE_SIZE; i++)
                QueueA[i] = __popc(QueueA[i] & QueueB[i]);

            ThreadReduce::Add(QueueA);
            counter_AB += QueueA[0];

            blockTraceA += PROP::BlockSize * QUEUE_SIZE;
            blockTraceB += PROP::BlockSize * QUEUE_SIZE;
        }
        WarpReduce<>::AtomicAdd(counter_AB,
                                reinterpret_cast<int*>(devResult + BlockID));
    }
}

} //@mangrove
