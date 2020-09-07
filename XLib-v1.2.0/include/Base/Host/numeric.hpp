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

#include "Base/host_device.cuh"
#define NO_CONSTEXPR 1

namespace xlib {

// ======================= COMPILE TIME numeric methods ========================

template<int A, int B>                          struct MIN;
template<int A, int B>                          struct MAX;
template<int N>                                 struct LOG2;
template<int N>                                 struct MOD2;
template<long long int N>                       struct IS_POWER2;

template<int N>                                 struct FACTORIAL;
template<int N, int K>                          struct BINOMIAL_COEFF;
template<int LOW, int HIGH>                     struct PRODUCTS_SEQUENCES;

template<int N, int MUL>                        struct LOWER_APPROX;
template<long long int N, long long int MUL>    struct LOWER_APPROX_L;
template<int N, int MUL>                        struct UPPER_APPROX;
template<long long int N, long long int MUL>    struct UPPER_APPROX_L;

template<long long int N>                       struct NEAREST_POW2_UP;
template<long long int N>                       struct NEAREST_POW2_DOWN;
template<int N, int DIV>                        struct NEAREST_APPROX;

template<typename T>                            struct NUMERIC_MIN;
template<typename T>                            struct NUMERIC_MAX;

#if NO_CONSTEXPR
    #define CONST_EXPR
#else
    #define CONST_EXPR constexpr
#endif

// ==================== CONST EXPR TIME numeric methods ========================

template<typename T, typename R>
__HOST_DEVICE__ CONST_EXPR
typename std::common_type<T, R>::type Div   (T value, R div);

template<unsigned DIV, typename T>
__HOST_DEVICE__ CONST_EXPR T    Div         (T value);

template<typename T, typename R>
__HOST_DEVICE__ CONST_EXPR T    upperApprox (T n, R MUL);
template<unsigned MUL, typename T>
__HOST_DEVICE__ CONST_EXPR T    upperApprox (T value);

template<typename T, typename R>
__HOST_DEVICE__ CONST_EXPR T    lowerApprox (T n, R MUL);
template<int MUL, typename T>
__HOST_DEVICE__ CONST_EXPR T    lowerApprox (T value);

template<typename T>
__HOST_DEVICE__ CONST_EXPR bool isPower2    (T x);
template<typename T>
__HOST_DEVICE__ CONST_EXPR T    factorial   (T x);
template<typename T>
__HOST_DEVICE__ CONST_EXPR T    binomialCoeff    (T x, T y);

template<typename T, typename R>
__HOST_DEVICE__  T              readBit     (T* Array, R pos);
template<typename T, typename R>
__HOST_DEVICE__ void            writeBit    (T* Array, R pos);

// ========================== RUN TIME numeric methods =========================

inline unsigned NearestPower2_UP(unsigned v);
inline int log2(unsigned v);

template<typename T>
float perCent(T part, T max);

template<typename R>
struct compareFloatABS_Str {
    template<typename T>
    inline bool operator() (T a, T b);
};

template<typename R>
struct compareFloatRel_Str {
    template<typename T>
    inline bool operator() (T a, T b);
};

} //@xlib

#include "impl/numeric.i.hpp"
