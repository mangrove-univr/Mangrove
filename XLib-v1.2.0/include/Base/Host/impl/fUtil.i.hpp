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
#include <stdexcept>
#include <type_traits>
#include <cxxabi.h>
#include <iostream>
#include <algorithm>
#include <memory>
#include <thread>

namespace xlib {

template <typename R, typename... Args>
struct get_arity<R(*)(Args...)> {
    static const int value = sizeof...(Args);
};

template <typename R, typename C, typename... Args>
struct get_arity<R(C::*)(Args...)> {
    static const int value = sizeof...(Args);
};

template<typename T>
struct Type64 {
static_assert(std::is_arithmetic<T>::value,
              PRINT_ERR("Type64 : Type not supported"));
using R    = typename std::remove_cv<T>::type;
using type = typename std::conditional<std::is_same<R, char>::value ||
                                       std::is_same<R, short>::value ||
                                       std::is_same<R, int>::value,
                                       long long int,
             typename std::conditional<std::is_same<R, unsigned char>::value ||
                                       std::is_same<R, unsigned short>::value ||
                                       std::is_same<R, unsigned>::value,
                                       long long unsigned,
             typename std::conditional<std::is_same<R, float>::value, double, R
             >::type>::type>::type;
};

template<class iterator_t>
struct Type64_it {
    using type = typename Type64<
                  typename std::iterator_traits<iterator_t>::value_type>::type;
};

template<class FUN_T, typename... T>
inline void Funtion_TO_multiThreads(bool MultiCore, FUN_T FUN, T... Args) {
    if (MultiCore) {
        const int concurrency = std::thread::hardware_concurrency();
        std::thread threadArray[32];

        for (int i = 0; i < concurrency; i++)
            threadArray[i] = std::thread(FUN, Args..., i, concurrency);
        for (int i = 0; i < concurrency; i++)
            threadArray[i].join();
    } else
        FUN(Args..., 0, 1);
}

constexpr int operator"" _BIT ( unsigned long long value ) {
    return static_cast<int>(value);
}
constexpr std::size_t operator"" _KB ( unsigned long long value ) {
    return static_cast<size_t>(value) * 1024;
}
constexpr std::size_t operator"" _MB ( unsigned long long value ) {
    return static_cast<size_t>(value) * 1024 * 1024;
}

//------------------------------------------------------------------------------

template <class T>
std::string type_name(T) {
    return type_name<T>();
}

template <class T>
std::string type_name() {
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
            std::free);
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B) {
    iteratorB_t it_B = start_B;
    for (iteratorA_t it_A = start_A; it_A != end_A; it_A++, it_B++) {
        if (*it_A != *it_B) {
            if (FAULT) {
                __ERROR("Array Difference at: " << std::distance(start_A, it_A)
                            << " -> Left Array: " << *it_A
                            << "     Right Array: " << *it_B);
            }
            return false;
        }
    }
    return true;
}

template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
        bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type)) {

    iteratorB_t it_B = start_B;
    for (iteratorA_t it_A = start_A; it_A != end_A; it_A++, it_B++) {
        if (!equalFunction(*it_A, *it_B)) {
            if (FAULT) {
                __ERROR("Array Difference at: " << std::distance(start_A, it_A)
                            << " -> Left Array: " << *it_A
                            << "     Right Array: " << *it_B);
            }
            return false;
        }
    }
    return true;
}

template<bool FAULT, class iteratorA_t, class iteratorB_t>
bool equalSorted(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B) {
    using T = typename std::iterator_traits<iteratorA_t>::value_type;
    using R = typename std::iterator_traits<iteratorB_t>::value_type;
    const int size = std::distance(start_A, end_A);
    T* tmpArray_A = new T[size];
    R* tmpArray_B = new R[size];

    std::copy(start_A, end_A, tmpArray_A);
    std::copy(start_B, start_B + size, tmpArray_B);
    std::sort(tmpArray_A, tmpArray_A + size);
    std::sort(tmpArray_B, tmpArray_B + size);

    bool flag = equal<FAULT>(tmpArray_A, tmpArray_A + size, tmpArray_B);

    delete[] tmpArray_A;
    delete[] tmpArray_B;
    return flag;
}

//------------------------------------------------------------------------------

template<typename T, typename R>
UniqueMap<T, R>::~UniqueMap() {}

template<typename T, typename R>
R UniqueMap<T, R>::insertValue(T id) {
    static_assert(std::is_integral<R>::value,
                  PRINT_ERR("UniqueMap accept only Integral types"));

    typename UniqueMap<T, R>::iterator IT = this->find(id);
    if (IT == this->end()) {
        R nodeID = static_cast<R>(this->size());
        this->insert(std::pair<T, R>(id, nodeID));
        return nodeID;
    }
    return IT->second;
}

} //@xlib
