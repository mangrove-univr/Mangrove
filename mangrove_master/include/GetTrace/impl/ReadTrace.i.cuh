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
#include "XLib.hpp"

#ifdef __linux__
    #include <stdio.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/types.h>
    #include <sys/mman.h>
#endif

namespace mangrove {

template<typename T>
void readTrace(ReadParamStr& ReadParam, TracePropSTR<T>& TraceProp) {
    using namespace timer;
    using namespace xlib;
    using namespace mangrove::support;

    if (ReadParam.read_mode != READ_MODE::STREAM && !__linux__)
        __ERROR("Only Stream Mode is allowed in non-linux system");
    checkRegularFile(ReadParam.trace_file);

    std::ifstream trace_file(ReadParam.trace_file);
    int vars, sim_instants;
    trace_file >> vars >> sim_instants;
    if (vars < MAX_ARITY)
        __ERROR("vars < " << MAX_ARITY);
    if (vars > MAX_VARS)
        __ERROR("vars > MAX_VARS : " << vars << " < " << MAX_VARS);
    if (ReadParam.overlapping && ReadParam.read_mode == READ_MODE::GPU)
        __ERROR("Using \"overlapping\" with \"GPU parsing\" is not allowed");
    if (ReadParam.read_mode == READ_MODE::STREAM)
        ReadParam.check_read = false;

    TraceProp.init(vars, sim_instants);

    size_t offset = static_cast<size_t>(trace_file.tellg()) +
                                        static_cast<size_t>(1);
    trace_file.seekg(0, std::ios_base::end);
    size_t file_size = static_cast<size_t>(trace_file.tellg());
    trace_file.close();

#if defined(__NVCC__)
    memCheckCUDA(TraceProp.pitch * TraceProp.vars
                 + (ReadParam.read_mode == READ_MODE::GPU ? file_size : 0));
#endif
    std::cout << std::endl << "Trace reading...("
              << TraceProp.vars << " vars; "
              << TraceProp.sim_instants << " simulation instants)\t\t"
              << "file_size: " << file_size / (1 << 20)
              << " MB" << std::endl << std::endl;

    using T1 = typename TracePropSTR<T>::type;
    T1* host_trace_tmp = nullptr;
    if (ReadParam.check_read)
        host_trace_tmp = new T1[TraceProp.vars * TraceProp.trace_length]();

    Timer<HOST> TM(1, 25, Color::FG_L_CYAN);
    TM.start();

    if (ReadParam.read_mode == READ_MODE::STREAM) {
        readStream(ReadParam.trace_file, offset, TraceProp.host_trace,
                   TraceProp, ReadParam.overlapping);
    }
    if (ReadParam.check_read) {
        readStream(ReadParam.trace_file, offset, host_trace_tmp, TraceProp,
                   ReadParam.overlapping);
    }

#if defined(__linux__)
    if (ReadParam.read_mode != READ_MODE::STREAM) {
        FILE *fp = ::fopen(ReadParam.trace_file, "r");
        struct stat statbuf;
        ::stat(ReadParam.trace_file, &statbuf);
        size_t file_size = static_cast<size_t>(statbuf.st_size);
        char* memory_mapped = static_cast<char*>(
                                ::mmap(NULL, file_size, PROT_READ, MAP_PRIVATE,
                                       fp->_fileno, 0));
        ::madvise(memory_mapped, file_size, MADV_SEQUENTIAL);

        if (ReadParam.read_mode == READ_MODE::MMAP) {
            memoryMapped<T>(memory_mapped + offset, TraceProp,
                            ReadParam.overlapping);
        }
    #if defined(__NVCC__)
        else {
            GPUAcceleration(memory_mapped + offset, TraceProp,
                            file_size - offset, ReadParam.check_read);
        }
    #endif
        ::munmap(memory_mapped, file_size);
        ::fclose(fp);
    }
#endif
#if defined(__NVCC__)
    Timer_cuda TM_dev;
    TM_dev.start();
    if (!ReadParam.overlapping && ReadParam.read_mode != READ_MODE::GPU) {
        cudaMemcpy2DAsync(TraceProp.dev_trace, TraceProp.pitch,
                          TraceProp.host_trace,
                          TraceProp.trace_length_bytes,
                          TraceProp.trace_length_bytes,
                          static_cast<size_t>(TraceProp.vars),
                          cudaMemcpyHostToDevice);
    }
    TM_dev.getTime("Host trasfer");
#endif

    TM.getTime((ReadParam.read_mode == READ_MODE::STREAM ||
                ReadParam.check_read) ? "IOstream read" :
                        (ReadParam.read_mode == READ_MODE::MMAP ?
                         "MMAP read" : "+++ GPU Accelerated +++"));

    if (!ReadParam.check_read)
        return;

    for (int V = 0; V < TraceProp.vars; V++) {
        T1* host_trace_var = TraceProp.host_trace + V * TraceProp.trace_length;
        T1* host_trace_var_tmp = TraceProp.host_trace + V * TraceProp.trace_length;

        std::cout << "===>>> Var " << V << std::endl;
        /*if (std::is_same<T, bool>::value) {
            printBits(reinterpret_cast<char*>(host_trace_var), 96);
            printBits(reinterpret_cast<char*>(host_trace_var_tmp), 96);
        } else {
            printArray(host_trace_var, 32);
            printArray(host_trace_var_tmp, 32);
        }*/

        xlib::equal(host_trace_var,
                    host_trace_var + TraceProp.trace_length,
                    host_trace_var_tmp);
    }
    std::cout << " Checking Traces : ok " << std::endl;
    delete[] host_trace_tmp;
}


//------------------------------------------------------------------------------

namespace support {

template<typename T>
void readStream(const char* _trace_file, size_t offset, void* host_trace_tmp,
                const TracePropSTR<T>& TraceProp,
                 __attribute__((__unused__)) bool overlapping) {

    std::ifstream trace_file(_trace_file);
    trace_file.seekg(std::streampos(offset), std::ios::beg);
    using T1 = typename TracePropSTR<T>::type;

    for (int V = 0; V < TraceProp.vars; V++) {
        T1* host_trace_var = static_cast<T1*>(host_trace_tmp) +
                             (V * TraceProp.trace_length);
        for (int j = 0; j < TraceProp.sim_instants; j++)
            trace_file >> host_trace_var[j];

        std::fill(host_trace_var + TraceProp.sim_instants,
                  host_trace_var + TraceProp.trace_length,
                  host_trace_var[TraceProp.sim_instants - 1]);

#if defined(__NVCC__)
        if (overlapping) {
            cudaMemcpyAsync(TraceProp.dev_trace +
                                (TraceProp.pitch / sizeof(T1)) *
                                static_cast<size_t>(V),
                            host_trace_var, TraceProp.trace_length_bytes,
                            cudaMemcpyHostToDevice);
        }
        __CUDA_ERROR("Copy To DEVICE");
#endif
    }
    trace_file.close();
}


#if defined(__NVCC__)

template<typename T>
void GPUAcceleration(char*, const TracePropSTR<T>&, bool) {
    __ERROR("GPUAcceleration not implemented for type: "
            << xlib::type_name<T>())
}

#endif

template<typename T>
void MemoryMapped(char*, const TracePropSTR<T>&) {
    __ERROR("MemoryMapped not implemented for type: " << xlib::type_name<T>())
}

} //@support
} //@mangrove
