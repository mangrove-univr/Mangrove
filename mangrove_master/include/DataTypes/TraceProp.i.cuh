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
#include "Mining/Mining.hpp"

namespace mangrove {

template<typename T>
TracePropSTR<T>::TracePropSTR() {}

template<typename T>
TracePropSTR<T>::TracePropSTR(int _vars, int _sim_instants) :
                              vars(_vars), sim_instants(_sim_instants) {
    init();
}

template<typename T>
TracePropSTR<T>::~TracePropSTR() {
    delete[] host_trace;
#if defined(__NVCC__)
    cudaFree(dev_trace);
#endif
}

template<typename T>
void TracePropSTR<T>::init(int _vars, int _sim_instants) {
    this->vars = _vars;
    this->sim_instants = _sim_instants;
    init();
}

template<typename T>
__HOST_DEVICE__ int TracePropSTR<T>::getTraceLength() const {
    return this->sim_instants;
}

template<typename T>
void TracePropSTR<T>::init() {
#if defined(__NVCC__)
    using PROP = PROPERTY<xlib::MAX_BLOCKSIZE, MAX_STEP>;
    this->trace_length_bytes = getRoundTraceLength<PROP>();
    this->trace_length = trace_length_bytes / sizeof(type);
    cudaMallocPitch(&dev_trace, &pitch, trace_length_bytes,
                    static_cast<size_t>(this->vars));
    __CUDA_ERROR("Trace Allocation");
#else
    this->trace_length = this->getTraceLength();
#endif
    this->host_trace = new type[vars * this->trace_length]();
}

#if defined(__NVCC__)

template<>
template<typename PROP>
__HOST_DEVICE__ size_t TracePropSTR<bool>::getRoundTraceLength() const {
    using namespace xlib;
    const unsigned BLOCK_CHUNK = PROP::BlockSize * PROP::Steps *
                                 static_cast<unsigned>(sizeof(int4)) * 2u;
    return upperApprox(static_cast<unsigned>(Div(this->sim_instants, 8)),
                       BLOCK_CHUNK);
}

template<typename T>
template<typename PROP>
__HOST_DEVICE__ size_t TracePropSTR<T>::getRoundTraceLength() const {
    const unsigned BLOCK_CHUNK = PROP::BlockSize * PROP::Steps *
                                 static_cast<unsigned>(sizeof(int4)) * 2u;
    return xlib::upperApprox(static_cast<unsigned>(this->sim_instants)
                             * sizeof(T), BLOCK_CHUNK);
}

//------------------------------------------------------------------------------

template<typename T>
devTraceSTR<T>::devTraceSTR(const TracePropSTR<T>& obj) :
                                vars(obj.vars),
                                sim_instants(obj.sim_instants),
                                trace_length(obj.trace_length),
                                dev_trace(obj.dev_trace),
                                pitch(obj.pitch),
                                trace_length_bytes(obj.trace_length_bytes),
                                problem_size(0) {}

template<typename T>
devTraceSTR<T>::devTraceSTR(const TracePropSTR<T>& obj, int _problem_size) :
                                vars(obj.vars),
                                sim_instants(obj.sim_instants),
                                trace_length(obj.trace_length),
                                dev_trace(obj.dev_trace),
                                pitch(obj.pitch),
                                trace_length_bytes(obj.trace_length_bytes),
                                problem_size(_problem_size) {}

template<typename T>
template<typename PROP>
__device__ __forceinline__
size_t devTraceSTR<T>::getRoundTraceLength() const {
    const unsigned BLOCK_CHUNK = PROP::BlockSize * PROP::Steps *
                                 static_cast<unsigned>(sizeof(int4)) * 2u;
    return xlib::upperApprox(static_cast<unsigned>(this->sim_instants) *
                             sizeof(T), BLOCK_CHUNK);
}

template<>
template<typename PROP>
__device__ __forceinline__
size_t devTraceSTR<bool>::getRoundTraceLength() const {
    using namespace xlib;
    const unsigned BLOCK_CHUNK = PROP::BlockSize * PROP::Steps *
                                 static_cast<unsigned>(sizeof(int4)) * 2u;
    return upperApprox(static_cast<unsigned>(Div(this->sim_instants, 8)),
                       BLOCK_CHUNK);
}
#endif

}  //@mangrove
