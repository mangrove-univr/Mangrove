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

#include <type_traits>
#include "XLib.hpp"
#include "config.cuh"

namespace mangrove {

template<typename T>
struct TracePropSTR {
private:
    void init();
    TracePropSTR(const TracePropSTR<T>& obj) = delete;
    TracePropSTR& operator=(const TracePropSTR& obj) = delete;

public:
    using type = typename std::conditional<std::is_same<bool, T>::value,
                                           unsigned, T>::type;
    type* host_trace;
    int vars;               // # of vars
    int sim_instants;       // # of instants
    int sim_instants32;     // ceil( # of instants / 32 )
    int trace_length;       // parallel : trace_length = sim_instants round to
                            //                           MAX BLOCK_CHUNK
                            // sequential : trace_length = sim_instants ||
                            //                             sim_instants32
#if defined(__NVCC__)
    type* dev_trace;
    size_t pitch;
    size_t trace_length_bytes; // trace_length in bytes

    template<typename PROP>
    __HOST_DEVICE__ size_t getRoundTraceLength() const;  //in bytes
#endif
    TracePropSTR();
    TracePropSTR(int _vars, int _sim_instants);

    ~TracePropSTR();
    void init(int _vars, int _sim_instants);

    /** return host trace lenght : # of elements of type <type> */
    __HOST_DEVICE__ int getTraceLength() const;
};

template<>
TracePropSTR<bool>::TracePropSTR();
template<>
TracePropSTR<bool>::TracePropSTR(int _vars, int _sim_instants);
template<>
void TracePropSTR<bool>::init(int _vars, int _sim_instants);
template<>
int TracePropSTR<bool>::getTraceLength() const;

//------------------------------------------------------------------------------

#if defined(__NVCC__)

template<typename T>
struct devTraceSTR {
private:
    devTraceSTR& operator=(const devTraceSTR& obj) = delete;
public:
    using type = typename std::conditional<std::is_same<bool, T>::value,
                                           unsigned, T>::type;
   type* dev_trace;
   size_t pitch;
   size_t trace_length_bytes; // trace_length in bytes

    int vars;               // # of vars
    int sim_instants;       // # of instants
    int trace_length;
    int problem_size;

    devTraceSTR(const TracePropSTR<T>& obj);
    devTraceSTR(const TracePropSTR<T>& obj, int _problem_size);

    template<typename PROP>
    __device__ size_t getRoundTraceLength() const;
};
#endif

//------------------------------------------------------------------------------

template<typename T, int VARS, int ARITY>
struct DictionarySize {
    static const int value = std::is_same<T, bool>::value?
       ARITY == 2 ? xlib::BINOMIAL_COEFF<VARS, 2>::value :
       (VARS - (ARITY - 1)) * xlib::BINOMIAL_COEFF<VARS, ARITY - 1>::value :
       xlib::BINOMIAL_COEFF<VARS, ARITY>::value * xlib::FACTORIAL<ARITY>::value;
};
//const int MAX_DICTIONARY_ENTRIES = DictionarySize<MAX_VARS, MAX_ARITY>::value;
template<int ARITY>
using     dictionary_ptr_t = entry_t (*)[ARITY];

template<int ARITY, typename T>
using trace_ptr_t = T* (*)[ARITY];

}  //@mangrove

#include "TraceProp.i.cuh"
