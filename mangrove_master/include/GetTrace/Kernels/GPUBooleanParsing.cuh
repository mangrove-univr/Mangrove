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

#include "XLib.hpp"

namespace mangrove {

namespace {

template<unsigned SIZE, unsigned K = 0>
struct textToBooleanDev {
    static __device__ __forceinline__ void apply(char* Queue, char* value) {
        if (Queue[K] == '1')
            value[0] |= 1 << (K / 2);
        textToBooleanDev<SIZE, K + 2>::apply(Queue, value);
    }
};
template<unsigned SIZE>
struct textToBooleanDev<SIZE, SIZE> {
    static __device__ __forceinline__ void apply(char*, char*) {}
};

} //@anonymous

//==============================================================================

template<typename PROP>
__global__ void GPUBooleanParsing(char* Input,
                                  devTraceSTR<bool> devProp,
                                  size_t devBooleanTMP_pitch) {
    using namespace xlib;
    static_assert((SMem_Per_Thread<char>::value % (PROP::Steps * 2)) == 0,
                  PRINT_ERR("Steps configuration error"));
    #if __CUDA_ARCH__ >= 520
        using VecT = int4;
    #else
        using VecT = int2;
    #endif

    const int QUEUE_SIZE = sizeof(int4) * 2 * PROP::Steps;
    const int BLOCKSMEM_CHAR = SMem_Per_Block<PROP::BlockSize, char>::value;
    const int BLOCKSMEM_VecT = SMem_Per_Block<PROP::BlockSize, VecT>::value;
    const int BLOCK_CHUNK_IN = BLOCKSMEM_CHAR * sizeof(int4);
     // each shared memory char correspont to 16 char of input (one int4)
    const int TOTAL_STEPS = SMem_Per_Thread<char>::value / (PROP::Steps * 2);

    const int trace_length_char = devProp.sim_instants * 2;
    char* Output = reinterpret_cast<char*>(devProp.dev_trace);

    __shared__ char SH_MEM[BLOCKSMEM_CHAR];
    VecT* SH_MEM_Th = (VecT*) SH_MEM + threadIdx.x;

    Input += threadIdx.x * sizeof(int4);
    const size_t out_stride = gridDim.x * BLOCKSMEM_VecT;

    for (int V = 0; V < devProp.vars; V++) {
        VecT* Output_local = (VecT*) Output + blockIdx.x * BLOCKSMEM_VecT
                             + threadIdx.x;

        for (int block_offset = blockIdx.x * BLOCK_CHUNK_IN;
                 block_offset < trace_length_char;
                 block_offset += BLOCK_CHUNK_IN * gridDim.x) {

            char* Input_local = Input + block_offset;
            char* SH_MEM_local = SH_MEM + threadIdx.x;

            for (int SUB_LOOP = 0; SUB_LOOP < TOTAL_STEPS; SUB_LOOP++) {
                char Queue[QUEUE_SIZE];

                #pragma unroll
                for (int i = 0; i < PROP::Steps; i++) {
                    reinterpret_cast<int4*>(Queue)[i * 2] =
                       reinterpret_cast<int4*>(Input_local)
                            [i * 2 * PROP::BlockSize];
                    reinterpret_cast<int4*>(Queue)[i * 2 + 1] =
                       __ldg(&reinterpret_cast<int4*>(Input_local)
                             [(i * 2 + 1) * PROP::BlockSize]);
                }
                char values[PROP::Steps * 2] = {};
                #pragma unroll
                for (int i = 0; i < PROP::Steps * 2; i++)
                    textToBooleanDev<16>::apply(&Queue[i * 16], &values[i]);

                #pragma unroll
                for (int i = 0; i < PROP::Steps * 2; i++)
                    SH_MEM_local[i * PROP::BlockSize] = values[i];
                SH_MEM_local += PROP::BlockSize * PROP::Steps * 2;

                Input_local += PROP::BlockSize * PROP::Steps * sizeof(int4) * 2;
            }
            __syncthreads();

            //------------------------------------------------------------------

            if (block_offset + BLOCK_CHUNK_IN < trace_length_char) {
                #pragma unroll
                for (int i = 0; i < BLOCKSMEM_VecT; i += PROP::BlockSize)
                    Output_local[i] = SH_MEM_Th[i];

                Output_local += out_stride;
            }
            else {
                const int sim_instants = devProp.sim_instants;
                const int last_index = ((((sim_instants * 2u) %
                                        (BLOCKSMEM_CHAR * sizeof(int4))) - 1)
                                        / sizeof(int4)) / sizeof(unsigned);
                unsigned* SH_MEM_unsigned = reinterpret_cast<unsigned*>(SH_MEM);
                const unsigned last_char = SH_MEM_unsigned[last_index];
                const bool read_bit = last_char &
                                            (1u << ((sim_instants - 1u) % 32u));

                if (threadIdx.x == 0) {
                    //printf("V %d \t %u \t %d\n", V , last_char, last_index);
                    SH_MEM_unsigned[last_index] = read_bit ?
                        last_char | ~((1u << ((sim_instants - 1) % 32u)) - 1u) :
                        last_char & ((1u << ((sim_instants - 1) % 32u)) - 1u);
                }
                const unsigned mask = read_bit ? 0xFFFFFFFF : 0x00000000;
                for (int i = last_index + 1 + threadIdx.x;
                     i < BLOCKSMEM_CHAR / sizeof(unsigned);
                     i += PROP::BlockSize) {

                    SH_MEM_unsigned[i] = mask;
                }
                if (PROP::BlockSize > 32)
                    __syncthreads();
                #pragma unroll
                for (int i = 0; i < BLOCKSMEM_VecT; i += PROP::BlockSize)
                    Output_local[i] = SH_MEM_Th[i];

                Output_local += BLOCKSMEM_VecT;

                #if __CUDA_ARCH__ >= 520
                    const int4 maskV = read_bit ?
                                       make_int4(0xFFFFFFFF, 0xFFFFFFFF,
                                                 0xFFFFFFFF, 0xFFFFFFFF) :
                                       make_int4(0, 0, 0, 0);
                #else
                    const int2 maskV = read_bit ?
                                       make_int2(0xFFFFFFFF, 0xFFFFFFFF) :
                                       make_int2(0, 0);
                #endif

                const int remain = (VecT*) (Output + devProp.trace_length_bytes)
                                   - Output_local;
                for (int i = 0; i < remain; i += PROP::BlockSize)
                    Output_local[i] = maskV;
            }
            if (PROP::BlockSize > 32)
                __syncthreads();
        }
        Input += devBooleanTMP_pitch;
        Output += devProp.pitch;
    }
}

} //@mangrove
