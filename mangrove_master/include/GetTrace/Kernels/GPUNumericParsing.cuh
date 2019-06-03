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


__device__ __forceinline__ float textToFloatDev(const char* input, int& pos) {
    using namespace xlib;
    using namespace std;
    bool neg = false;
    if (input[pos] == '-') {
        neg = true;
        pos++;
    }

    int S = 0;
    while (input[pos + S] >= '0' && input[pos + S] <= '9')
        ++S;

    unsigned value = fastStringToInt(input + pos, S);
    float result = static_cast<float>(value);

    pos += S;
    if (input[pos] == '.') {
        pos++;
        int K = 0;
        while (input[pos + K] >= '0' && input[pos + K] <= '9')
            ++K;

        int lenM = min(K, 9);
        const float int_frac = static_cast<float>(fastStringToInt(input + pos, lenM));

        switch (lenM) {
            case 1: result += int_frac *  (1 / 10.0f); break;
            case 2: result += int_frac *  (1 / 100.0f); break;
            case 3: result += int_frac *  (1 / 1000.0f); break;
            case 4: result += int_frac *  (1 / 10000.0f); break;
            case 5: result += int_frac *  (1 / 100000.0f); break;
            case 6: result += int_frac *  (1 / 1000000.0f); break;
            case 7: result += int_frac *  (1 / 10000000.0f); break;
            case 8: result += int_frac *  (1 / 100000000.0f); break;
            case 9: result += int_frac *  (1 / 1000000000.0f); break;
            default:;
        }
        pos += K;
    }
    return neg ? -result : result;
}


//==============================================================================

const int DEV_SYNC_SIZE = 1 << 23;

__device__ int devSync[DEV_SYNC_SIZE];
const int UNLOCK_VALUE_1 = -0xFFFFFFFF;

template<unsigned BlockSize, typename T>
__global__ void GPUNumericParsing(char* Input, int trace_sizeVec,
                                  devTraceSTR<T> devProp) {

    using namespace xlib;
    using VecT = int2;
    const unsigned VecSIZE = sizeof(VecT);
    const unsigned LOCAL_SIZE = SMem_Per_Thread<>::value;
    const unsigned BLOCK_SMEM_VEC = SMem_Per_Block<BlockSize, VecT>::value;
    const unsigned N_OF_WARP = BlockSize / WARP_SIZE;
    const unsigned STRIDE =  BLOCK_SMEM_VEC - LOCAL_SIZE / VecSIZE;
    static_assert(LOCAL_SIZE % VecSIZE == 0,
                  PRINT_ERR("LOCAL_SIZE is not a multiple of VecSIZE"));

    const int block_offset = blockIdx.x * STRIDE;
    const unsigned warp_id = WarpID();
    int BlockID = blockIdx.x;

    __shared__ VecT SMemVec[BLOCK_SMEM_VEC];
    VecT* inputVec = reinterpret_cast<VecT*>(Input);
    int* SMemScanTMP = reinterpret_cast<int*>(SMemVec);
    numeric_t* devTrace = devProp.dev_trace;
    const int pitch = devProp.pitch / sizeof(T);

    for (int i = block_offset; i < trace_sizeVec; i += gridDim.x * STRIDE) {

        VecT* input_tmp = inputVec + i;
        #pragma unroll
        for (int j = 0; j < BLOCK_SMEM_VEC; j += BlockSize) {
            int index = j + threadIdx.x;
            if (index < BLOCK_SMEM_VEC)
                SMemVec[index] = input_tmp[index];
        }
        __syncthreads();

        int founds = 0;
        numeric_t Queue[8];
        if (threadIdx.x != BlockSize - 1) {
            char* SMemLocal = reinterpret_cast<char*>(SMemVec) +
                                threadIdx.x * LOCAL_SIZE;

            int pos = 0;
            while (true) {
                while (pos <= LOCAL_SIZE) {
                    if (SMemLocal[pos] == ' ' || SMemLocal[pos] == '\n') {
                        pos++;
                        if (SMemLocal[pos] == '\n')
                            pos++;
                        break;
                    }
                    pos++;
                }
                if (pos > LOCAL_SIZE)
                    break;
                Queue[founds++] = textToFloatDev(SMemLocal, pos);
            }
        }
        __syncthreads();
        int prefix_sum = founds;
        WarpExclusiveScan<>::Add(prefix_sum, SMemScanTMP + warp_id);

        //======================================================================
        // ------ WARP STREAM PROCEDURE ----------------------------------------

        __syncthreads();
        if ( threadIdx.x < N_OF_WARP ) {
            int value = SMemScanTMP[threadIdx.x];

            int block_total = 0;
            WarpExclusiveScan<N_OF_WARP>::Add(value, block_total);

            int previous_block_value = 0;
            if (threadIdx.x == N_OF_WARP - 1) {
                if (BlockID != 0) {
                    int* previous_value_ptr = devSync + BlockID;
                    previous_block_value = cub::ThreadLoad<cub::LOAD_CG>
                                                           (previous_value_ptr);
                    while ( previous_block_value == UNLOCK_VALUE_1 ) {
                        previous_block_value = cub::ThreadLoad<cub::LOAD_CG>
                                                           (previous_value_ptr);
                    }
                }
                cub::ThreadStore<cub::STORE_CG>(devSync + BlockID + 1,
                                        block_total + previous_block_value);
            }
            SMemScanTMP[threadIdx.x] = value + __shfl(previous_block_value, N_OF_WARP - 1);
        }
        __syncthreads();
        prefix_sum += SMemScanTMP[warp_id];

        int row = prefix_sum / devProp.sim_instants;
        int col = prefix_sum % devProp.sim_instants;
        auto devTraceTMP = devTrace + row * pitch + col;

        if (col + founds >= devProp.sim_instants) {
            for (int k = 0; k < founds; k++) {
                int t_row = (prefix_sum + k) / devProp.sim_instants;
                int t_col = (prefix_sum + k) % devProp.sim_instants;
                auto t_devTraceTMP = devTrace + t_row * pitch + t_col;
                *t_devTraceTMP = Queue[k];
            }
        }
        else {
            for (int k = 0; k < founds; k++)
                devTraceTMP[k] = Queue[k];
        }
        BlockID += gridDim.x;
        __syncthreads();
    }
}

template<unsigned BlockSize, typename T>
__global__ void GPUNumericExtend(devTraceSTR<T> devProp) {
    auto devTrace = devProp.dev_trace;
    const int last = devProp.sim_instants - 1;
    const int trace_length = devProp.trace_length;

    for (int i = blockIdx.x; i < devProp.vars; i += gridDim.x) {
        auto devTraceTMP = devTrace + i * trace_length;
        auto last_value = devTraceTMP[last];
        for (int j = last + 1 + threadIdx.x; j < trace_length; j += BlockSize)
            devTraceTMP[j] = last_value;
    }
}

} //@mangrove
