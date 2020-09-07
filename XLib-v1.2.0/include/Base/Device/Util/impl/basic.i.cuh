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
#include "../PTX.cuh"

namespace xlib {

template<unsigned WARP_SZ>
__device__ __forceinline__ unsigned WarpBase() {
    return threadIdx.x & ~(WARP_SZ - 1u);
}

__device__ __forceinline__ unsigned WarpID() {
    return threadIdx.x >> 5u;
}

template<typename T>
__device__ __forceinline__ T WarpBroadcast(T value, int predicate) {
    const unsigned electedLane = xlib::__msb(__ballot(predicate));
    return __shfl(value, electedLane);
}

template<typename T>
__device__ __forceinline__ void swap(T*& A, T*& B) {
    T* tmp = A;
    A = B;
    B = tmp;
}

template<typename T>
__device__ __forceinline__  T _log2(T value) {
    return xlib::__msb(value);
}

#if defined(ENABLE_CUB)

template<typename P, typename T, typename R>
__device__ __forceinline__
T cuReadBit(T* devPointer, R pos)  {
    const unsigned SIZE = sizeof(P::ToCast) * 8;
    typename P::ToCast value = cub::ThreadLoad<P::LOAD>(
                reinterpret_cast<typename P::ToCast*>(devPointer) + pos / SIZE);
    //return value & (1 << (pos % SIZE));
    return xlib::__be(value, pos % SIZE);
}

template<typename P, typename T, typename R>
__device__ __forceinline__
void cuWriteBit(T* devPointer, R pos) {
    const unsigned SIZE = sizeof(T) * 8;
    typename P::ToCast* tmp_ptr =
                 reinterpret_cast<typename P::ToCast*>(devPointer) + pos / SIZE;
    //T value = cub::ThreadLoad<M1>(tmp_ptr) | (1 << (pos % SIZE));
    typename P::ToCast value = xlib::__bi(
                                 cub::ThreadLoad<P::LOAD>(tmp_ptr), pos % SIZE);
    cub::ThreadStore<P::STORE>(tmp_ptr, value);
}

#endif

} //@xlib
