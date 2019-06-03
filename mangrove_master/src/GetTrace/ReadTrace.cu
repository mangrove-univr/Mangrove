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
#include <fstream>
#include <iostream>

#if defined(__linux__)
    #include <stdio.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/types.h>
    #include <sys/mman.h>
#endif

#include "XLib.hpp"
#include "GetTrace/ReadTrace.cuh"
#include "GetTrace/GetTraceSupport.hpp"

#if defined(__NVCC__)
    #include "GetTrace/Kernels/GPUBooleanParsing.cuh"
    #include "GetTrace/Kernels/GPUNumericParsing.cuh"
    #include <cub/cub.cuh>
#endif

//#include "Mining/Device/impl/AutoTuning.cuh"

using namespace xlib;

namespace mangrove {

//AutoTuning(GPUBooleanParsing)

namespace support {

template<>
void readStream<bool>(const char* trace_name, size_t offset,
                      void* host_trace_tmp, const TracePropSTR<bool>& TraceProp,
                       __attribute__((__unused__)) bool overlapping) {

    std::ifstream traceFile(trace_name);
    traceFile.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    unsigned* host_trace_var = static_cast<unsigned*>(host_trace_tmp);

    for (int V = 0; V < TraceProp.vars; V++) {
        for (int j = 0; j < TraceProp.sim_instants; j++) {
            char tmpC;
            traceFile >> tmpC;
            if (tmpC == '1')
                writeBit(host_trace_var, j);
        }
        booleanFilling(host_trace_var, TraceProp.sim_instants,
                       TraceProp.trace_length);
#if defined(__NVCC__)
        if (overlapping) {
            cudaMemcpyAsync(TraceProp.dev_trace +
                                (TraceProp.pitch / sizeof(unsigned))
                                * static_cast<size_t>(V),
                            host_trace_var,
                            TraceProp.trace_length_bytes,
                            cudaMemcpyHostToDevice);
        }
#endif
        host_trace_var += TraceProp.trace_length;
    }
    traceFile.close();
}

template<>
void memoryMapped<bool>(char* memory_mapped,
                        const TracePropSTR<bool>& TraceProp,
                         __attribute__((__unused__)) bool overlapping) {

    const unsigned file_trace_length =
                        static_cast<unsigned>(TraceProp.sim_instants) * 2u + 1u;
    const int file_trace_length_down =
                           static_cast<int>(lowerApprox<64>(file_trace_length));
    const unsigned remain = file_trace_length % 64u;
    const unsigned div = static_cast<unsigned>(file_trace_length_down) / 64u;

    for (int V = 0; V < TraceProp.vars; V++) {
        unsigned* host_trace_var = TraceProp.host_trace +
                                   V * TraceProp.trace_length;
        for (int i = 0; i < file_trace_length_down; i += 64) {
            textToBoolean<64>::apply(memory_mapped + i,
                               &host_trace_var[static_cast<unsigned>(i) / 64u]);
        }
        if (remain) {
            textToBooleanDyn(memory_mapped + file_trace_length_down,
                             &host_trace_var[div], remain);
        }
        booleanFilling(host_trace_var, TraceProp.sim_instants,
                       TraceProp.trace_length);
#if defined(__NVCC__)
    if (overlapping) {
            cudaMemcpyAsync(TraceProp.dev_trace +
                                (TraceProp.pitch / sizeof(unsigned)) *
                                static_cast<size_t>(V),
                            host_trace_var,
                            TraceProp.trace_length_bytes,
                            cudaMemcpyHostToDevice);
    }
#endif
        memory_mapped += file_trace_length;
    }
}

template<>
void memoryMapped<numeric_t>(char* memory_mapped,
                             const TracePropSTR<numeric_t>& TraceProp,
                              __attribute__((__unused__)) bool overlapping) {

    numeric_t* host_trace_var = TraceProp.host_trace;
    for (int V = 0; V < TraceProp.vars; V++) {
        for (int i = 0 ; i < TraceProp.sim_instants; i++)
            host_trace_var[i] = textToFloat(memory_mapped);

#if defined(__NVCC__)
        std::fill(host_trace_var + TraceProp.sim_instants,
                  host_trace_var + TraceProp.trace_length,
                  host_trace_var[TraceProp.sim_instants - 1]);
        if (overlapping) {
            cudaMemcpyAsync(TraceProp.dev_trace +
                                (TraceProp.pitch / sizeof(numeric_t)) *
                                static_cast<size_t>(V),
                            host_trace_var,
                            TraceProp.trace_length_bytes,
                            cudaMemcpyHostToDevice);
        }
#endif
        memory_mapped++;
        host_trace_var += TraceProp.trace_length;
    }
}

//==============================================================================
//==============================================================================

#if defined(__NVCC__)

template<>
void GPUAcceleration<bool>(char* memory_mapped,
                           const TracePropSTR<bool>& TraceProp,
                           size_t,
                           bool check_read) {
    using namespace timer;
    using PROP = PROPERTY<512, 1>;
    Timer_cuda TM(1, 35, Color::FG_RED);

    const size_t BLOCK_CHAR_CHUNK = sizeof(int4)
                                * SMem_Per_Block<PROP::BlockSize, char>::value;
    //for each 1-byte in shared memory are needed 16-bytes in global memory

    char* devBooleanTMP;
    size_t devBooleanTMP_pitch;
    cudaMallocPitch(&devBooleanTMP, &devBooleanTMP_pitch,
                    upperApprox(static_cast<size_t>(TraceProp.sim_instants * 2),
                                BLOCK_CHAR_CHUNK),
                    static_cast<size_t>(TraceProp.vars));

    if (check_read) {
        cudaMemset(devBooleanTMP, 0x0,
                   devBooleanTMP_pitch * static_cast<size_t>(TraceProp.vars));
        cudaMemset(TraceProp.dev_trace, 0x0,
                   TraceProp.pitch * static_cast<size_t>(TraceProp.vars));
    }

    TM.start();

    cudaMemcpy2DAsync(devBooleanTMP, devBooleanTMP_pitch,
                      memory_mapped,
                      static_cast<size_t>(TraceProp.sim_instants * 2 + 1),
                      static_cast<size_t>(TraceProp.sim_instants * 2 + 1),
                      static_cast<size_t>(TraceProp.vars),
                      cudaMemcpyHostToDevice);


    TM.getTime("File Transfer : Host -> GPU");
    devTraceSTR<bool> devTrace(TraceProp);

    //AutoTuningClass::Init(TraceProp.vars);
    //AutoTuningGPUBooleanParsing<PROP>::Apply(devBooleanTMP, devTrace, devBooleanTMP_pitch);

    const unsigned grid_size = gridConfig(GPUBooleanParsing<PROP>,
                                          PROP::BlockSize, 0u, TraceProp.vars);

    TM.start();

    GPUBooleanParsing<PROP><<<grid_size, PROP::BlockSize>>>
                                (devBooleanTMP, devTrace, devBooleanTMP_pitch);

    TM.getTimeA("GPU Parsing");

    cudaFree(devBooleanTMP);
    if (check_read) {
        cudaMemcpy2D(TraceProp.host_trace, TraceProp.trace_length_bytes,
                    TraceProp.dev_trace, TraceProp.pitch,
                    TraceProp.trace_length_bytes,
                    static_cast<size_t>(TraceProp.vars),
                    cudaMemcpyDeviceToHost);
    }
    __CUDA_ERROR("GPU Accelerated Parsing");
}


template<>
void GPUAcceleration<numeric_t>(char* memory_mapped,
                                const TracePropSTR<numeric_t>& TraceProp,
                                size_t file_size,
                                bool check_read) {
    using namespace timer;
    const int BlockSize = 256;
    const size_t BLOCK_INT4_CHUNK = SMem_Per_Block<BlockSize, int4>::value;
    const size_t allocated_size = upperApprox(file_size + 1, BLOCK_INT4_CHUNK);

    if (Div(allocated_size, BlockSize) >= DEV_SYNC_SIZE) {
        __ERROR("DEV_SYNC_SIZE too small : "
                << Div(allocated_size, BlockSize) <<
                "increase the value in GPUNumericParsing.cuh");
    }

    char* devNumericTMP;
    int* devSyncPTR;
    cudaMalloc(&devNumericTMP, allocated_size);
    cudaMemset(devNumericTMP, 0x0, allocated_size);
    cudaGetSymbolAddress((void**) &devSyncPTR, devSync);
    cudaMemset(devSyncPTR, 0xFF, DEV_SYNC_SIZE);

    if (check_read) {
        cudaMemset(devNumericTMP, 0x0, allocated_size);
        cudaMemset(TraceProp.dev_trace, 0x0,
                   TraceProp.pitch * static_cast<size_t>(TraceProp.vars));
    }
    //==========================================================================
    Timer_cuda TM(1, 35, Color::FG_RED);
    TM.start();

    cudaMemcpy(devNumericTMP, memory_mapped - 1, file_size, cudaMemcpyHostToDevice);

    TM.getTime("File Transfer : Host -> GPU");
    //==========================================================================
    // Computation

    const unsigned grid_size = Div(Div(allocated_size, sizeof(int4)), BlockSize);
    //gridConfig(GPUNumericParsing<PROP>, PROP::BlockSize, 0u, TraceProp.vars)
    devTraceSTR<numeric_t> devProp(TraceProp);

    TM.start();

    GPUNumericParsing<BlockSize><<<grid_size, BlockSize>>>
                    (devNumericTMP, Div(allocated_size, sizeof(int4)), devProp);

    GPUNumericExtend<BlockSize><<<grid_size, BlockSize>>>(devProp);

    TM.getTime("GPU Parsing");
    //==========================================================================

    cudaFree(devNumericTMP);
    if (check_read) {
        cudaMemcpy2D(TraceProp.host_trace, TraceProp.trace_length_bytes,
                    TraceProp.dev_trace, TraceProp.pitch,
                    TraceProp.trace_length_bytes,
                    static_cast<size_t>(TraceProp.vars),
                    cudaMemcpyDeviceToHost);
    }
    __CUDA_ERROR("GPU Accelerated Parsing");
}

#endif
} //@support
} //@mangrove
