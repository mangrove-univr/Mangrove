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
 #include <type_traits>
#include "XLib.hpp"

namespace mangrove {

template<typename T>
void generateMain(GenerateParamStr& GenerateParam, TracePropSTR<T>& TraceProp) {
    TraceProp.init(GenerateParam.vars, GenerateParam.sim_instants);

    mangrove::support::generate(TraceProp, GenerateParam.overlapping,
                                GenerateParam.random,
                                GenerateParam.range_vector);

 #if defined(__NVCC__)
     if (!GenerateParam.overlapping) {
         cudaMemcpy2DAsync(TraceProp.dev_trace, TraceProp.pitch,
                           TraceProp.host_trace,
                           TraceProp.trace_length_bytes,
                           TraceProp.trace_length_bytes,
                           static_cast<size_t>(TraceProp.vars),
                           cudaMemcpyHostToDevice);
    }
    __CUDA_ERROR("Copy To DEVICE");
 #endif
}

namespace support {

template<typename T>
typename std::enable_if<std::is_integral<T>::value>::type
generate(TracePropSTR<T>& TraceProp,
         __attribute__((__unused__)) bool overlapping,
         bool random,
         const std::vector<std::pair<double, double>>& vector_range) {

    std::cout << std::endl << "Numeric Type Trace generation..." << std::endl;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    T* host_trace_var = TraceProp.host_trace;
    for (int V = 0; V < TraceProp.vars; V++) {
        std::uniform_int_distribution<T> distribution(vector_range[V].first,
                                                      vector_range[V].second);
        if (random)
            std::generate(host_trace_var,
                          host_trace_var + TraceProp.trace_length,
                          [&](){ return distribution(generator); });
        else
            std::fill(host_trace_var,
                      host_trace_var + TraceProp.trace_length, 1);
#if defined(__NVCC__)
        if (overlapping) {
            cudaMemcpyAsync(TraceProp.dev_trace +
                                (TraceProp.pitch / sizeof(T)) *
                                static_cast<size_t>(V),
                            host_trace_var,
                            TraceProp.trace_length_bytes,
                            cudaMemcpyHostToDevice);
        }
#endif
        host_trace_var += TraceProp.trace_length;
    }
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value>::type
generate(TracePropSTR<T>& TraceProp,
         __attribute__((__unused__)) bool overlapping,
         bool random,
         const std::vector<std::pair<double, double>>& vector_range) {

    std::cout << std::endl << "Numeric Type Trace generation..." << std::endl;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    T* host_trace_var = TraceProp.host_trace;
    for (int V = 0; V < TraceProp.vars; V++) {
        std::uniform_int_distribution<> distribution(vector_range[V].first,
                                                     vector_range[V].second);
        if (random) {
            std::generate_n(host_trace_var, TraceProp.trace_length,
              [&](){ return static_cast<numeric_t>(distribution(generator)); });
        } else {
            std::fill(host_trace_var,
                      host_trace_var + TraceProp.trace_length, 1.0f);
        }
#if defined(__NVCC__)
        if (overlapping) {
            cudaMemcpyAsync(TraceProp.dev_trace +
                                (TraceProp.pitch / sizeof(T)) *
                                static_cast<size_t>(V),
                            host_trace_var, TraceProp.trace_length_bytes,
                            cudaMemcpyHostToDevice);
        }
#endif
        host_trace_var += TraceProp.trace_length;
    }
}

} //@support
} //@mangrove
