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
#pragma once

#include <iostream>
#include <string>
#include <limits>

#include "Base/host_device.cuh"

namespace xlib {

template<typename T, int SIZE>
void printArray(T (&Array)[SIZE], std::string text = "",
                char sep = ' ', T inf = std::numeric_limits<T>::max());

template<typename T>
void printArray(T* Array, int size, std::string text = "",
                char sep = ' ', T inf = std::numeric_limits<T>::max());
template<>
void printArray<char>(char* Array, int size, std::string text,
                      char sep , char inf);
template<>
void printArray<unsigned char>(unsigned char* Array, int size, std::string text,
                               char sep , unsigned char inf);

template<typename T>
void printMatrix(T** Matrix, int ROW, int COL,
                 std::string text = "", T inf = std::numeric_limits<T>::max());

//------------------------------------------------------------------------------

template<typename T>
__HOST_DEVICE__
typename std::enable_if<std::is_floating_point<T>::value>::type
_printArray(T* Array, int size);

template<typename T>
__HOST_DEVICE__
typename std::enable_if<std::is_integral<T>::value>::type
_printArray(T* Array, int size);

template<>
__HOST_DEVICE__
void _printArray(char* Array, int size);
template<>
__HOST_DEVICE__
void _printArray(unsigned char* Array, int size);

template<typename T>
__HOST_DEVICE__
typename std::enable_if<std::is_integral<T>::value>::type
printBits(T* Array, int size);

//------------------------------------------------------------------------------

#if  defined(__NVCC__)

#include <cuda_runtime.h>

template<typename T>
void printCudaArray(T* devArray, int size, std::string text = "",
                    char sep = ' ', T inf = std::numeric_limits<T>::max());

template<>
void printCudaArray<int2>(int2* Array, int size, std::string text,
                          char sep, int2 inf);

template<>
void printCudaArray<int3>(int3* Array, int size, std::string text,
                          char sep, int3 inf);

template<>
void printCudaArray<int4>(int4* Array, int size, std::string text,
                          char sep, int4 inf);

#endif
} //@xlib

#include "impl/print_ext.i.hpp"
