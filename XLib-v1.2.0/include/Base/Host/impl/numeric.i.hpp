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
#include <algorithm>
#include <type_traits>
#include <ratio>
#include "Base/Host/fUtil.hpp"

namespace xlib {

// ==================== CONST EXPR TIME numeric methods ========================

template<typename T, typename R>
__HOST_DEVICE__ CONST_EXPR
typename std::common_type<T, R>::type Div(T value, R div) {
    return (value + div - 1) / div;
}

template<unsigned DIV, typename T>
__HOST_DEVICE__ CONST_EXPR
T Div(T value) {
    return (value + DIV - 1u) / DIV;
}

//------------------------------------------------------------------------------

template<typename T, typename R>
__HOST_DEVICE__ CONST_EXPR T upperApprox(T value, R MUL) {
    return Div(value, MUL) * MUL;
}

template<unsigned MUL, typename T>
__HOST_DEVICE__ CONST_EXPR T upperApprox(T value) {
    return !IS_POWER2<MUL>::value ? Div<MUL>(value) * MUL
                                  : (value + MUL - 1) & ~(MUL - 1);
}

//------------------------------------------------------------------------------

template<typename T, typename R>
__HOST_DEVICE__ CONST_EXPR T lowerApprox(T value, R MUL) {
    return (value / MUL) * MUL;
}

template<int MUL, typename T>
__HOST_DEVICE__ CONST_EXPR T lowerApprox(T value) {
    return !IS_POWER2<MUL>::value ? (value / MUL) * MUL
                                    : value & ~(MUL - 1);
}

//------------------------------------------------------------------------------

template<typename T>
__HOST_DEVICE__ CONST_EXPR bool isPower2(T x) {
    return (x != 0) && !(x & (x - 1));
}

template<typename T>
__HOST_DEVICE__ CONST_EXPR T factorial( T x) {
    return x <= 1 ? 1 : x * factorial(x - 1);
}

template<typename T, typename R>
__HOST_DEVICE__  T readBit(T* Array, R pos)  {
    using unsigned_t = typename std::make_unsigned<R>::type;
    const unsigned_t SIZE = sizeof(T) * 8u;
    return Array[static_cast<unsigned_t>(pos) / SIZE] &
           static_cast<T>(static_cast<unsigned_t>(1)
                          << (static_cast<unsigned_t>(pos) % SIZE));
}

template<typename T, typename R>
__HOST_DEVICE__ void writeBit(T* Array, R pos) {
    using unsigned_t = typename std::make_unsigned<R>::type;
    const unsigned_t SIZE = sizeof(T) * 8u;
    Array[static_cast<unsigned_t>(pos) / SIZE] |=
                     static_cast<T>(static_cast<unsigned_t>(1)
                                    << (static_cast<unsigned_t>(pos) % SIZE));
}

// ======================= COMPILE TIME numeric methods ========================

template<int A, int B> struct MAX { static const int value = A > B ? A : B; };
template<int A, int B> struct MIN { static const int value = A < B ? A : B; };

//lower bound
template<int N>
struct LOG2 {
    static_assert(N > 0, PRINT_ERR("LOG2 : N <= 0"));
    static const unsigned value = 1 + LOG2<N / 2>::value;
};
template<>    struct LOG2<1> { static const unsigned value = 0; };

template<int N>
struct MOD2 {
    static_assert(N > 0, PRINT_ERR("MOD2 : N <= 0"));
    static_assert(IS_POWER2<N>::value,
                  PRINT_ERR("MOD2 : N is not power of two"));
    static const int value = N - 1;
};

template<long long int N>
struct IS_POWER2 {
    static const bool value = (N != 0) && !(N & (N - 1));
};

template<int N>
struct FACTORIAL {
    static_assert(N >= 0, PRINT_ERR("FACTORIAL"));
    static const int value = N * FACTORIAL<N - 1>::value;
};
template<> struct FACTORIAL<0> { static const int value = 1; };

template<int LOW, int HIGH>
struct PRODUCTS_SEQUENCES {
    static const int value = LOW * PRODUCTS_SEQUENCES<LOW + 1, HIGH>::value;
};
template<int LOW>
struct PRODUCTS_SEQUENCES<LOW, LOW> {
    static const int value = LOW;
};

template<int N, int K>
struct BINOMIAL_COEFF {
static_assert(N >= 0 && K >= 0 && K <= N, PRINT_ERR("BINOMIAL_COEFF"));
private:
    static const int min = MIN<K, N - K>::value;
    static const int max = MAX<K, N - K>::value;
public:
    static const int value = PRODUCTS_SEQUENCES<max + 1, N>::value /
                             FACTORIAL<min>::value;
};
template<int N>
struct BINOMIAL_COEFF<N ,N> {
    static const int value = 1;
};

template<int N, int DIV>
struct NEAREST_APPROX {
    static const int value = (N % DIV <= DIV / 2) ? N / DIV : N / DIV + 1;
};

template<int N, int MUL>
struct LOWER_APPROX {
    static const int value = (N / MUL) * MUL;
};

template<long long int N, long long int MUL>
struct LOWER_APPROX_L {
    static const long long int value = (N / MUL) * MUL;
};

template<int N, int MUL>
struct UPPER_APPROX {
    static const int value = ((N + MUL - 1) / MUL) * MUL;
};

template<long long int N, long long int MUL>
struct UPPER_APPROX_L {
    static const long long int value = ((N + MUL - 1) / MUL) * MUL;
};

template<long long int N>
struct NEAREST_POW2_UP {
private:
    static const int V = N - 1;
public:
    static const int value = (V | (V >> 1) | (V >> 2) |
                             (V >> 4) | (V >> 8) | (V >> 16)) + 1;
};

template<long long int N>
struct NEAREST_POW2_DOWN {
private:
    static const int V = NEAREST_POW2_UP<N>::value;
public:
    static const int value = V == N ? N : V >> 1;
};

template<typename T>
struct NUMERIC_MIN {
    static const T value = std::numeric_limits<T>::lowest();
};

template<typename T>
struct NUMERIC_MAX {
    static const T value = std::numeric_limits<T>::max();
};

// ========================== RUN TIME numeric methods =========================

inline int nearestPower2_UP(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

inline int log2(unsigned v) {
    return 31 - __builtin_clz(v);
}

template<typename T>
float perCent(T part, T max) {
    return (static_cast<float>(part) / static_cast<float>(max)) * 100.0f;
}

template<std::intmax_t Num, std::intmax_t Den>
struct compareFloatABS_Str<std::ratio<Num, Den>> {
    template<typename T>
    inline bool operator() (T a, T b) {
        const T epsilon = static_cast<T>(Num) / static_cast<T>(Den);
        return std::abs(a - b) < epsilon;
    }
};

template<std::intmax_t Num, std::intmax_t Den>
struct compareFloatRel_Str<std::ratio<Num, Den>> {
    template<typename T>
    inline bool operator() (T a, T b) {
        const T epsilon = static_cast<T>(Num) / static_cast<T>(Den);
        const T diff = std::abs(a - b);
        //return (diff < epsilon) || (diff / std::max(std::abs(a), std::abs(b)) < epsilon);
        return (diff < epsilon) ||
        (diff /
         std::min(std::abs(a) + std::abs(b),
                  std::numeric_limits<float>::max())
            < epsilon);
    }
};

} //@numeric
