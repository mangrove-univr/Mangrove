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

namespace xlib {

// the searched value must be in the intervall

template<typename T>
__device__ __forceinline__
void binarySearch(T* Mem, const T searched,
                  int& pos, int size) {
	int start = 0, end = size - 1;
	pos = end / 2u;

	while (start < end) {
		if (searched >= Mem[pos + 1])
			start = pos + 1;
		else if (searched < Mem[pos])
			end = pos - 1;
		else
			break;
		pos = (start + end) / 2u;
	}
}

template<unsigned GROUP_SIZE, typename T>
__device__ __forceinline__
void binarySearchFAST(T* Mem, const T searched, int& pos) {
    static_assert(IS_POWER2<GROUP_SIZE>::value,
                  PRINT_ERR("GROUP_SIZE must be a power of 2"));
	int low = 0;
	#pragma unroll
	for (int i = 1; i <= LOG2<GROUP_SIZE>::value; i++) {
		pos = low + ((GROUP_SIZE) >> i);
		if (searched >= Mem[pos])
			low = pos;
	}
	pos = low;
}

template<typename T>
__device__ __forceinline__
void binarySearchFAST_shfl(const T value, const T searched,
                           int& pos) {
	int low = 0;
	#pragma unroll
	for (int i = 1; i <= LOG2<WARP_SIZE>::value; i++) {
		pos = low + ((WARP_SIZE) >> i);
		if (searched >= __shfl(value, pos))
			low = pos;
	}
	pos = low;
}

} //xlib
