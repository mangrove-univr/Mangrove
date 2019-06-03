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

#include "Mining/TemplateEngine.cuh"
#include "TemplateConf.cuh"
#include "DataTypes/TraceProp.cuh"
#include "XLib.hpp"

namespace mangrove {

namespace {

// recursion over invariant funtions
template<typename Invariant, int K,
         int INV_SIZE = Invariant::arity - 1, int F_INDEX = 0>
struct recursiveFunctionGPU {

    template<typename T, int TEMPL_ARITY, int QUEUE_SIZE>
    __device__ __forceinline__  static
    typename std::conditional<F_INDEX == 0, bool, T>::type
    eval(T (&Queue)[TEMPL_ARITY][QUEUE_SIZE]) {
        return APPLY_FUN<F_INDEX, Invariant>::eval(
                Queue[F_INDEX][K],
                recursiveFunctionGPU<Invariant, K,
                                     INV_SIZE, F_INDEX + 1>::eval(Queue));
    }
};

template<typename Invariant, int K, int INV_SIZE>
struct recursiveFunctionGPU<Invariant, K, INV_SIZE, INV_SIZE> {

    template<typename T, int TEMPL_ARITY, int QUEUE_SIZE>
    __device__ __forceinline__  static
    typename std::conditional<INV_SIZE == 0, bool, T>::type
    eval(T (&Queue)[TEMPL_ARITY][QUEUE_SIZE]) {
        return Queue[INV_SIZE][K];
    }
};

// =============================================================================
// recursion over template invariants

template<typename Invariant, int QUEUE_SIZE, int K = 0>
struct recursiveInvariantGPU {

    template<int TEMPL_ARITY, typename T>
    __device__ __forceinline__ static
    void eval(bool (&TMP)[QUEUE_SIZE], T (&Queue)[TEMPL_ARITY][QUEUE_SIZE]) {
        TMP[K] = recursiveFunctionGPU<Invariant, K, TEMPL_ARITY - 1>
                        ::eval(Queue);

        recursiveInvariantGPU<Invariant, QUEUE_SIZE, K + 1>::eval(TMP, Queue);
    }
};

template<typename Invariant, int QUEUE_SIZE>
struct recursiveInvariantGPU<Invariant, QUEUE_SIZE, QUEUE_SIZE> {
    template<int TEMPL_ARITY, typename T>
    __device__ __forceinline__
    static void eval(bool (&)[QUEUE_SIZE], T (&)[TEMPL_ARITY][QUEUE_SIZE]) {}
};

//==============================================================================

template<typename Template, int N_OF_INVARIANT = Template::size,
                            int INV_INDEX = 0>
struct recursiveTemplateGPU {

template<int TEMPL_ARITY, int QUEUE_SIZE, typename T>
__device__ __forceinline__ static void eval(
                                            T (&Queue)[TEMPL_ARITY][QUEUE_SIZE],
                                            result_t& SH_INVARIANT_FLAGS,
                                            result_t& INVARIANT_FLAGS) {
    using namespace xlib;
    if (INVARIANT_FLAGS & (static_cast<result_t>(1) << INV_INDEX)) {
        bool TMP[QUEUE_SIZE];

        using Invariant = typename GET_INVARIANT<INV_INDEX, Template>::type;
        recursiveInvariantGPU<Invariant, QUEUE_SIZE>::eval(TMP, Queue);
        ThreadReduce::LogicAnd(TMP);

        if (!__all(TMP[0])) {
            INVARIANT_FLAGS &= ~(static_cast<result_t>(1) << INV_INDEX);
            if (LaneID() == 0) {
                atomicAnd(&SH_INVARIANT_FLAGS,
                          ~(static_cast<result_t>(1) << INV_INDEX));
            }
        }
    }
    recursiveTemplateGPU<Template, N_OF_INVARIANT, INV_INDEX + 1>
                        ::eval(Queue, SH_INVARIANT_FLAGS, INVARIANT_FLAGS);
}

};

template<typename Template, int N_OF_INVARIANT>
struct recursiveTemplateGPU<Template, N_OF_INVARIANT, N_OF_INVARIANT> {
    template<int TEMPL_ARITY, int QUEUE_SIZE, typename T>
    __device__ __forceinline__ static void eval(
                                            T (&)[TEMPL_ARITY][QUEUE_SIZE],
                                            result_t&, result_t&) {}
};

} //@anonymous
//==============================================================================


template<typename PROP, typename R>
__global__ void GPUMining(devTraceSTR<R> devProp) {
    using namespace xlib;
    using Template = typename IndexToTemplate<PROP::TEMPL_INDEX>::type;;
    using T = typename Template::type;
    using T4 = typename std::conditional
               <std::is_same<unsigned, T>::value, uint4, float4>::type;

    const int QUEUE_SIZE = PROP::Steps * (sizeof(uint4) / sizeof(T)) * 2;
    const int TEMPL_ARITY = Template::arity;
    const result_t N_OF_INVARIANT = Template::size;
    const int SMEM_PER_BLOCK = SMem_Per_Block<PROP::BlockSize, result_t>::value;
    __shared__ result_t SH_INVARIANT_FLAGS[ SMEM_PER_BLOCK ];

    T* devTrace = static_cast<T*>(devProp.dev_trace);
    devTrace += threadIdx.x * (sizeof(uint4) / sizeof(T));
    const int pitch = devProp.pitch / sizeof(T);

    const int traceLength = devProp.template getRoundTraceLength<PROP>() /
                            sizeof(T);
    const result_t INIT_FLAGS = (1u << N_OF_INVARIANT) - 1u;

    // =================== LOOP OVER DICTIONARY ENTRY  =========================

    int flag_index = SMEM_PER_BLOCK;

    for (int BlockID = blockIdx.x; BlockID < devProp.problem_size;
             BlockID += gridDim.x) {
        if (flag_index == SMEM_PER_BLOCK) {
            flag_index = 0;
            #pragma unroll
            for (int i = 0; i < SMem_Per_Thread<result_t>::value; i++)
                SH_INVARIANT_FLAGS[i * PROP::BlockSize + threadIdx.x] =
                                                                     INIT_FLAGS;
            if (PROP::BlockSize > 32)
                __syncthreads();
        }


        // ============== TRACE ROW SELECTION -> DICTIONARY ====================

        const int blockOffset = BlockID * TEMPL_ARITY;
        T* blockTrace[TEMPL_ARITY];
        #pragma unroll
        for (int ARITY_INDEX = 0; ARITY_INDEX < TEMPL_ARITY; ARITY_INDEX++) {
            blockTrace[ARITY_INDEX] = devTrace
                                      + devDictionary[blockOffset + ARITY_INDEX]
                                      * pitch;
        }

        // ===================== LOOP OVER TRACE ROWS ==========================

        result_t INVARIANT_FLAGS = INIT_FLAGS;
        for (int LOOP = 0; LOOP < traceLength;
                 LOOP += PROP::BlockSize * QUEUE_SIZE) {
            T Queue[TEMPL_ARITY][QUEUE_SIZE];

            // ================== DATA LOAD ====================================

            #pragma unroll
            for (int K = 0; K < PROP::Steps; K++) {
                #pragma unroll
                for (int ARITY_INDEX = 0; ARITY_INDEX < TEMPL_ARITY;
                         ARITY_INDEX++) {                // DATA-PARALLELISM
                    reinterpret_cast<T4*>(Queue[ARITY_INDEX])[K * 2] =
                        reinterpret_cast<T4*>( blockTrace[ARITY_INDEX] )
                        [K * 2 * PROP::BlockSize];

                    reinterpret_cast<T4*>(Queue[ARITY_INDEX])[K * 2 + 1] =
                        __ldg( &reinterpret_cast<T4*>( blockTrace[ARITY_INDEX] )
                        [(K * 2 + 1) * PROP::BlockSize] );
                }
            }

            // ================== COMPUTATION ==================================

            recursiveTemplateGPU<Template>
               ::eval(Queue, SH_INVARIANT_FLAGS[flag_index], INVARIANT_FLAGS);

            if (PROP::BlockSize > 32) {
                __syncthreads();
                INVARIANT_FLAGS = SH_INVARIANT_FLAGS[flag_index];
            }
            if (INVARIANT_FLAGS == 0) break;

            #pragma unroll
            for (int ARITY_INDEX = 0; ARITY_INDEX < TEMPL_ARITY; ARITY_INDEX++)
                blockTrace[ARITY_INDEX] += PROP::BlockSize * QUEUE_SIZE;
        }
        flag_index++;
        if (threadIdx.x == 0)
            devResult[BlockID] = static_cast<result_t>(INVARIANT_FLAGS);
    }
}

} //@mangrove
