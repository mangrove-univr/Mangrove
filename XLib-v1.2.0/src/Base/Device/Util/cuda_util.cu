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
#include <iostream>
#include <string>
#include <nvToolsExt.h>
#include "Base/Device/Util/cuda_util.cuh"
#include "Base/Device/Util/definition.cuh"

namespace xlib {

void __getLastCudaError(const char *errorMessage, const char *file,
                               int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        std::cerr << std::endl << " CUDA error   "
                  << Emph::SET_UNDERLINE << file
                  << "(" << line << ")"
                  << Emph::SET_RESET << " : " << errorMessage
                  << " -> " << cudaGetErrorString(err) << "(" << (int) err
                  << ") "<< std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
}

void __safe_call(cudaError_t err, const char *file, int line) {
    if (cudaSuccess != err) {
        std::cerr << std::endl << " CUDA error   "
                  << file  << "(" << line << ")"
                  << " -> " << cudaGetErrorString(err) << "("
                  << static_cast<int>(err) << ") "<< std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
}

void memCheckCUDA(size_t Req) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (Req > free)
        __ERROR("MEMORY TOO LOW. Req: " << std::setprecision(1) << std::fixed
                 << (float) Req / (1 << 20) << " MB, available: "
                 << (float) free / (1 << 20) << " MB");
}

bool memInfoCUDA(size_t Req) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "    Total Device Memory:\t" << (total >> 20) << " MB"
              << std::endl
              << "     Free Device Memory:\t" << (free >> 20) << " MB"
              << std::endl
              << "Requested Device memory:\t" << (Req >> 20) << " MB"
              << "\t(" << ((Req >> 20) * 100) / (total >> 20) << "%)"
              << std::endl  << std::endl;
    return free > Req;
}

int deviceProperty::NUM_OF_STREAMING_MULTIPROCESSOR = 0;

int deviceProperty::getNum_of_SMs() {
    if(NUM_OF_STREAMING_MULTIPROCESSOR == 0) {
        cudaDeviceProp devProperty;
        cudaGetDeviceProperties(&devProperty, 0);
        NUM_OF_STREAMING_MULTIPROCESSOR = devProperty.multiProcessorCount;
    }
    return NUM_OF_STREAMING_MULTIPROCESSOR;
}

namespace NVTX {
    /*void PushRange(std::string s, const int color) {
        nvtxEventAttributes_t eventAttrib = {};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = color; //colors[color_id];
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = s.c_str();
        nvtxRangePushEx(&eventAttrib);
    }

    void PopRange() {
        nvtxRangePop();
    }*/
}

void cudaStatics() {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    __CUDA_ERROR("statics");

    std::cout << std::endl << std::left
        << "     Graphic Card: "
            << devProp.name << " (cc: "
            << devProp.major << "."
            << devProp.minor << ")" << std::endl
        << "             # SM: " << std::setw(15)
            << devProp.multiProcessorCount
        << "Threads per SM: " << std::setw(10)
            << devProp.maxThreadsPerMultiProcessor
        << "Max Resident Thread: "
            << devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor
            << std::endl
        /*<< "       Global Mem: " << std::setw(15)
            << (std::to_string(devProp.totalGlobalMem / (1 << 20)) + " MB")
        << "    Shared Mem: " << std::setw(10)
            << (std::to_string(devProp.sharedMemPerBlock / 1024) + " KB")
        << "           L2 Cache: "
            << (std::to_string(devProp.l2CacheSize / 1024) + " KB") << std::endl
        << "   smemPerThreads: " << std::setw(15) <<
            (std::to_string(devProp.sharedMemPerBlock
                            / devProp.maxThreadsPerMultiProcessor)) + " Byte"
        << "regsPerThreads: " << std::setw(10)
            << devProp.regsPerBlock / devProp.maxThreadsPerMultiProcessor
        << "          regsPerSM: " << devProp.regsPerBlock << std::endl*/
        /*void cudaStatics() {
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, 0);
            __CUDA_ERROR("statics");

            std::cout << std::endl << std::left
                << "     Graphic Card: "
                    << devProp.name << " (cc: "
                    << devProp.major << "."
                    << devProp.minor << ")" << std::endl
                << "             # SM: " << std::setw(15)
                    << devProp.multiProcessorCount
                << "Threads per SM: " << std::setw(10)
                    << devProp.maxThreadsPerMultiProcessor
                << "Max Resident Thread: "
                    << devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor
                    << std::endl
                << "       Global Mem: " << std::setw(15)
                    << (std::to_string(devProp.totalGlobalMem / (1 << 20)) + " MB")
                << "    Shared Mem: " << std::setw(10)
                    << (std::to_string(devProp.sharedMemPerBlock / 1024) + " KB")
                << "           L2 Cache: "
                    << (std::to_string(devProp.l2CacheSize / 1024) + " KB") << std::endl
                << "Shared mem per threads: " << std::setw(15) <<
                    (std::to_string(static_cast<int>(devProp.sharedMemPerBlock)
                                    / devProp.maxThreadsPerMultiProcessor)) + " Bytes"
                << "Regs per threads: " << std::setw(10)
                    << devProp.regsPerBlock / devProp.maxThreadsPerMultiProcessor
                << "          Regs per SM: " << devProp.regsPerBlock << std::endl
                << std::endl;

            if (devProp.multiProcessorCount != NUM_SM)
                __ERROR("Config Error: Wrong number of Streaming Multiprocessor (SM)");
        }*/
        << std::endl;

    if (devProp.multiProcessorCount != NUM_SM)
        __ERROR("Config Error: Wrong number of Streaming Multiprocessor (SM)");
}

} // @xlib
