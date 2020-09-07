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

/** \namespace PTX
 *  provide simple interfaces for low-level PTX instructions
 */
namespace xlib {

#if _WIN32 || __i386__
    #define ASM_PTR "r"
#elif __x86_64__ || __ia64__ || _WIN64 || __ppc64__
    #define ASM_PTR "l"
#endif

// ---------------------------- THREAD PTX -------------------------------------

/** \fn unsigned int LaneID()
 *  \brief return the lane ID within the current warp
 *
 *  Provide the thread ID within the current warp (called lane).
 *  \return identification ID in the range 0 &le; ID &le; 31
 */
__device__ __forceinline__ unsigned int LaneID();

/** \fn void ThreadExit()
 *  \brief terminate the current thread
 */
__device__ __forceinline__ void ThreadExit();

// --------------------------------- MATH --------------------------------------

/** \fn unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
 *  \brief sum three operands with one instruction
 *
 *  Sum three operand with one instruction. Only in Maxwell architecture
 *  IADD3 is implemented in hardware, otherwise involves multiple instructions.
 *  \return x + y + z
 */
__device__ __forceinline__
unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z);

/** \fn unsigned int __msb(unsigned int word)
 *  \brief find most significant bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position in the range: 0 &le; bitPosition &le; 31.
 *  0xFFFFFFFF if no bit is found.
 */

/** \fn unsigned int __msb(unsigned long long int dword)
 *  \brief find most significant bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position in the range: 0 &le; bitPosition &le; 63.
 *          0xFFFFFFFF if no bit is found.
 */
//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8, unsigned>::type
__msb(T word);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, unsigned>::type
__msb(T dword);

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8, unsigned>::type
__be(T word, unsigned pos);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__be(T dword, unsigned pos);

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8, unsigned>::type
__bi(T word, unsigned pos);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__bi(T dword, unsigned pos);

//==============================================================================

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8, unsigned>::type
__bfe(T word, unsigned pos, unsigned length);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8, long long unsigned>::type
__bfe(T dword, unsigned pos, unsigned length);

//------------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) != 8>::type
__bfi(T& word, unsigned bitmask, unsigned pos, unsigned length);

template<typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8>::type
__bfi(T& dword, long long unsigned bitmask, unsigned pos, unsigned length);

//------------------------------------------------------------------------------



/** \fn unsigned int LaneMaskEQ()
 *  \brief 32-bit mask with bit set in position equal to the thread's
 *         lane number in the warp
 *  \return 1 << laneid
 */
__device__ __forceinline__ unsigned LaneMaskEQ();

/** \fn unsigned int LaneMaskLT()
 *  \brief 32-bit mask with bits set in positions less than the thread's lane
 *         number in the warp
 *  \return (1 << laneid) - 1
 */
__device__ __forceinline__ unsigned LaneMaskLT();

/** \fn unsigned int LaneMaskLE()
 *  \brief 32-bit mask with bits set in positions less than or equal to the
 *         thread's lane number in the warp
 *  \return (1 << (laneid + 1)) - 1
 */
__device__ __forceinline__ unsigned LaneMaskLE();

/** \fn unsigned int LaneMaskGT()
 *  \brief 32-bit mask with bit set in position equal to the thread's
 *         lane number in the warp
 *  \return ~((1 << (laneid + 1)) - 1)
 */
__device__ __forceinline__ unsigned LaneMaskGT();

/** \fn unsigned int LaneMaskGE()
 *  \brief 32-bit mask with bits set in positions greater than or equal to the
 *         thread's lane number in the warp
 *  \return ~((1 << laneid) - 1)
 */
__device__ __forceinline__ unsigned LaneMaskGE();

} //@xlib

#include "impl/PTX.i.cuh"
