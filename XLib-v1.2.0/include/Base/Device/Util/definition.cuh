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

#include "Base/Host/numeric.hpp"

namespace xlib {

const unsigned        THREAD_PER_SM  =  2048;
const unsigned        MAX_BLOCKSIZE  =  1024;
const unsigned MAX_SHARED_PER_BLOCK  =  49152;
const unsigned         CONSTANT_MEM  =  49152;
const unsigned          MEMORY_BANK  =  32;
const unsigned            WARP_SIZE  =  32;

template<typename T>
struct Max_SMem_Per_Block {
    static const unsigned value = MAX_SHARED_PER_BLOCK / sizeof(T);
};

#if defined(__CUDACC__)
#if __CUDA_ARCH__  >= 300 || ARCH >= 300
    #if __CUDA_ARCH__ == 300 || ARCH == 300
        //#pragma message("\n\nCompute Capability: 3\n")
        const unsigned SMEM_PER_SM =    49152;
        const unsigned RESIDENT_BLOCKS_PER_SM = 16;

    #elif __CUDA_ARCH__ == 320 || ARCH == 320
        //#pragma message("\n\nCompute Capability: 3.2\n")
        const unsigned SMEM_PER_SM =    49152;
        const unsigned RESIDENT_BLOCKS_PER_SM = 16;

    #elif __CUDA_ARCH__ == 350 || ARCH == 350
        //#pragma message("\n\nCompute Capability: 3.5\n")
        const unsigned SMEM_PER_SM =    49152;
        const unsigned RESIDENT_BLOCKS_PER_SM = 16;

    #elif __CUDA_ARCH__ == 370 || ARCH == 370
        //#pragma message("\n\nCompute Capability: 3.7\n")
        const unsigned SMEM_PER_SM =    114688;
        const unsigned RESIDENT_BLOCKS_PER_SM = 16;

    #elif __CUDA_ARCH__ == 500 || ARCH == 500
        //#pragma message("\n\nCompute Capability: 5.0\n")
        const unsigned SMEM_PER_SM =    65536;
        const unsigned RESIDENT_BLOCKS_PER_SM = 32;

    #elif __CUDA_ARCH__ == 520 || ARCH == 520
        //#pragma message("\n\nCompute Capability: 5.2\n")
        const unsigned SMEM_PER_SM = 98304;
        const unsigned RESIDENT_BLOCKS_PER_SM = 32;

    #elif __CUDA_ARCH__ == 530 || ARCH == 530
        //#pragma message("\n\nCompute Capability: 5.3\n")
        const unsigned SMEM_PER_SM = 65536;
        const unsigned RESIDENT_BLOCKS_PER_SM = 32;
    #endif

    const unsigned SMEM_PER_THREAD = SMEM_PER_SM / THREAD_PER_SM;

    template<typename T = char>
    struct SMem_Per_Thread {
        static const unsigned value = SMEM_PER_THREAD / sizeof(T);
    };
    template<typename T = char>
    struct SMem_Per_Warp {
        static const unsigned value = (SMEM_PER_THREAD * WARP_SIZE) / sizeof(T);
    };
    template<int BlockDim, typename T = char>
    struct SMem_Per_Block {
        static const unsigned value =
                MIN< (SMEM_PER_THREAD * BlockDim) / sizeof(T),
                     (unsigned) (MAX_SHARED_PER_BLOCK / sizeof(T)) >::value;
    };
    #define ARCH_DEF
#else
    #pragma error(ERR_START "Unsupported Compute Cabalitity" ERR_END)
#endif
#endif

#if defined(SM)
    const int           NUM_SM  =  SM;
    const int RESIDENT_THREADS  =  SM * THREAD_PER_SM;
    const int   RESIDENT_WARPS  =  RESIDENT_THREADS / 32;

    template<int BlockDim>
    struct RESIDENT_BLOCKS {
        static const unsigned value = RESIDENT_THREADS / BlockDim;
    };
#endif
#undef SM

} //@xlib
