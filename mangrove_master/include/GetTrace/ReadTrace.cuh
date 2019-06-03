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
#include "DataTypes/ModuleSTR.hpp"

namespace mangrove {

template<typename T>
void readTrace(ReadParamStr& ReadParam, TracePropSTR<T>& TraceProp);

//------------------------------------------------------------------------------

namespace support {

template<typename T>
void readStream(const char* trace_file, size_t offset, void* host_trace_tmp,
                const TracePropSTR<T>& TraceProp, bool overlapping);

template<>
void readStream<bool>(const char* trace_file, size_t offset,
                      void* host_trace_tmp, const TracePropSTR<bool>& TraceProp,
                      bool overlapping);

//------------------------------------------------------------------------------

template<typename T>
void memoryMapped(char* memory_mapped, const TracePropSTR<T>& TraceProp,
                  bool overlapping);

template<>
void memoryMapped<bool>(char* memory_mapped,
                        const TracePropSTR<bool>& TraceProp,
                        bool overlapping);

template<>
void memoryMapped<numeric_t>(char* memory_mapped,
                             const TracePropSTR<numeric_t>& TraceProp,
                             bool overlapping);

//------------------------------------------------------------------------------

#if defined(__NVCC__)

template<typename T>
void GPUAcceleration(char* memory_mapped, const TracePropSTR<T>& TraceProp,
                     size_t file_size, bool check_read);
template<>
void GPUAcceleration<bool>(char* memory_mapped,
                           const TracePropSTR<bool>& TraceProp,
                           size_t,
                           bool check_read);

template<>
void GPUAcceleration<numeric_t>(char* memory_mapped,
                                const TracePropSTR<numeric_t>& TraceProp,
                                size_t file_size,
                                bool check_read);

#endif

} //@support
} //@mangrove

#include "impl/ReadTrace.i.cuh"
