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
#include <typeinfo>
#include "UserFunctions.cuh"
#include "XLib.hpp"

namespace mangrove {

template <template<typename> class T, typename R>
struct is_same_return_str {
    static const bool value = std::is_same<decltype(T<R>()(0, 0)), R>::value;
};

template <typename T>
struct get_arity_str {
    static const int value = xlib::get_arity< decltype(&T::operator()) >::value;
};

//==============================================================================

template <typename, template<typename> class...>
struct InvariantTail;

template <typename R, template<typename> class T,
                      template<typename> class... Ts>
struct InvariantTail<R, T, Ts...> : InvariantTail<R, Ts...> {
    static_assert(is_same_return_str<T, R>::value &&
                  get_arity_str<T<R>>::value == 2,
                  PRINT_ERR("Wrong type"));

    static const int size = 1 + InvariantTail<R, Ts...>::size;
};

template <typename R, template<typename> class T>
struct InvariantTail<R, T> {
    static_assert(is_same_return_str<T, R>::value &&
                  get_arity_str<T<R>>::value == 2,
                  PRINT_ERR("Wrong type"));

    static const int size = 1;
};

template <typename R,
          template<typename> class T1,
          template<typename> class T2 = identity,
          template<typename> class... Ts>
struct Invariant {
    static_assert(is_same_return_str<T1, bool>::value &&
                  get_arity_str< T1<R> >::value == 2,
                  PRINT_ERR("Wrong type"));

    static const bool is_unary = get_arity_str<T2<R>>::value == 1;
    static const int arity = 2 + !is_unary + InvariantTail<R, Ts...>::size;
    using unary = typename std::conditional< is_unary, T2<R>,
                                             identity<R> >::type;
    using type = R;
    using first = T1<R>;
};

template <typename R,
          template<typename> class T1,
          template<typename> class T2>
struct Invariant<R, T1, T2> {
    static_assert(is_same_return_str<T1, bool>::value &&
                  get_arity_str< T1<R> >::value == 2,
                  PRINT_ERR("Wrong type"));

    static const bool is_unary = get_arity_str<T2<R>>::value == 1;
    static const int arity = 2 + !is_unary;
    using unary = typename std::conditional< is_unary, T2<R>,
                                             identity<R> >::type;
    using type = R;
    using first = T1<R>;
};

//==============================================================================

template <int, typename, class Enable = void>
struct APPLY_FUN;

template <int F_INDEX, typename R, template<typename> class T,
                                   template<typename> class... Ts>
struct APPLY_FUN<F_INDEX, InvariantTail<R, T, Ts...>, void> {
    static_assert(F_INDEX >= 0 &&
                  F_INDEX < InvariantTail<R, T, Ts...>::size,
                  PRINT_ERR("WRONG INDEXING"));

    __HOST_DEVICE__
    static R eval(R a, R b) {
        return APPLY_FUN<F_INDEX - 1, InvariantTail<R, Ts...>>::eval(a, b);
    }
};

template <typename R, template<typename> class T,
                      template<typename> class... Ts>
struct APPLY_FUN<0, InvariantTail<R, T, Ts...>, void> {
    __HOST_DEVICE__
    static R eval(R a, R b) {
        return T<R>()(a, b);
    }
};

template <int F_INDEX, typename R,
                       template<typename> class T1,
                       template<typename> class T2,
                       template<typename> class... Ts>
struct APPLY_FUN<F_INDEX, Invariant<R, T1, T2, Ts...>,
                typename std::enable_if< get_arity_str<T2<R>>::value == 1
                                         && F_INDEX != 0>::type> {
    static_assert(F_INDEX >= 0 &&
                  F_INDEX < Invariant<R, T1, T2, Ts...>::arity - 1,
                  PRINT_ERR("WRONG INDEXING"));

    __HOST_DEVICE__
    static R eval(R a, R b) {
        return APPLY_FUN<F_INDEX - 1, InvariantTail<R, Ts...>>::eval(a, b);
    }
};

template <int F_INDEX, typename R,
                       template<typename> class T1,
                       template<typename> class T2,
                       template<typename> class... Ts>
struct APPLY_FUN<F_INDEX, Invariant<R, T1, T2, Ts...>,
                typename std::enable_if< get_arity_str<T2<R>>::value != 1
                                         && F_INDEX != 0>::type> {
    static_assert(F_INDEX >= 0 &&
                  F_INDEX < Invariant<R, T1, T2, Ts...>::arity - 1,
                  PRINT_ERR("WRONG INDEXING"));

    __HOST_DEVICE__
    static R eval(R a, R b) {
        return APPLY_FUN<F_INDEX - 1, InvariantTail<R, T2, Ts...>>::eval(a, b);
    }
};

template <int F_INDEX, typename R,
                       template<typename> class T1,
                       template<typename> class T2,
                       template<typename> class... Ts>
struct APPLY_FUN<F_INDEX, Invariant<R, T1, T2, Ts...>,
                typename std::enable_if< get_arity_str<T2<R>>::value != 1
                                         && F_INDEX == 0>::type> {
    __HOST_DEVICE__
    static bool eval(R a, R b) {
        return T1<R>()(a, b);
    }
};

template <int F_INDEX, typename R,
                       template<typename> class T1,
                       template<typename> class T2,
                       template<typename> class... Ts>
struct APPLY_FUN<F_INDEX, Invariant<R, T1, T2, Ts...>,
                typename std::enable_if< get_arity_str<T2<R>>::value == 1
                                         && F_INDEX == 0>::type> {
    __HOST_DEVICE__
    static bool eval(R a, R b) {
        return T1<R>()(a, T2<R>()(b));
    }
};

//==============================================================================

template <typename, typename...>
struct is_invariant : std::false_type {};

template <typename R, template<typename> class T1,
                      template<typename> class T2,
                      template<typename> class... Ts>
struct is_invariant<Invariant<R, T1, T2, Ts...>> : std::true_type {};

template<typename...>
struct are_all_same_invariant;

template<typename T1, typename T2, typename... Ts>
struct are_all_same_invariant<T1, T2, Ts...> {
    static const bool value = is_invariant<T1>::value &&
                              is_invariant<T2>::value &&
                              T1::arity == T2::arity &&
                              are_all_same_invariant<T2, Ts...>::value;
};

template<typename T1, typename T2>
struct are_all_same_invariant<T1, T2> {
    static const bool value = is_invariant<T1>::value &&
                              is_invariant<T2>::value &&
                              T1::arity == T2::arity;
};

template<typename T1>
struct are_all_same_invariant<T1> {
    static const bool value = is_invariant<T1>::value;
};

//------------------------------------------------------------------------------

template <typename...>
struct Template;

template <typename T, typename... Ts>
struct Template<T, Ts...> {
    static const int size = 1 + sizeof...(Ts);
    static const int arity = T::arity;
    using type = typename T::type;

    static_assert(are_all_same_invariant<T, Ts...>::value,
                  PRINT_ERR("Error: Template of different invariants\
                             types are not allowed"));
};

//==============================================================================

template <int, typename>
struct GET_INVARIANT;

template <int INDEX, typename T, typename... Ts>
struct GET_INVARIANT<INDEX, Template<T, Ts...>> {
    static_assert(INDEX >= 0 && INDEX < Template<T, Ts...>::size,
                  PRINT_ERR("WRONG INDEXING"));
    using type = typename GET_INVARIANT<INDEX - 1, Template<Ts...>>::type;
};

template <typename T, typename... Ts>
struct GET_INVARIANT<0, Template<T, Ts...>> {
    using type = T;
};

//==============================================================================

template <typename Template, int INV_INDEX, int F_INDEX>
struct GET {
    template<typename T>
    __HOST_DEVICE__
    static typename std::conditional<F_INDEX == 0, bool, T>::type
    eval(T a, T b) {
        using invariant = typename GET_INVARIANT<INV_INDEX, Template>::type;
        return APPLY_FUN<F_INDEX, invariant>::eval(a, b);
    }
};

//==============================================================================

namespace {

template<template<typename> class... Ts>
struct typeTList;

template<typename A, typename B>
struct is_same_Tlist : std::false_type {};

template<template<typename> class T1, template<typename> class... Ts>
struct is_same_Tlist<typeTList<T1, Ts...>,
                     typeTList<T1, Ts...>> : std::true_type {};

template<template<typename> class T1, template<typename> class T2,
         template<typename> class... Ts>
struct is_same_Tlist<typeTList<T1, T2, Ts...>,
                     typeTList<T1, T2, Ts...>> : std::true_type {};

//------------------------------------------------------------------------------

template<typename>
struct InvariantToTypeList;

template<typename T, template<typename> class T1, template<typename> class T2,
                     template<typename> class... Ts>
struct InvariantToTypeList<Invariant<T, T1, T2, Ts...>> {
    using type = typename
                 std::conditional<std::is_same<identity<T>, T2<T>>::value,
                 typeTList<T1, Ts...>, typeTList<T1, T2, Ts...>>::type;
};


template <int INDEX, int LIMIT, typename Template,
          template<typename> class... Ts>
struct GET_POSITION_AUX {
    using ActualInvariant = typename GET_INVARIANT<INDEX, Template>::type;
    using ActualList = typename InvariantToTypeList<ActualInvariant>::type;

    static const result_t value =
                    is_same_Tlist<ActualList, typeTList<Ts...>>::value ?
                    (1 << INDEX) :
                    GET_POSITION_AUX<INDEX + 1, LIMIT, Template, Ts...>::value;
};

template <int LIMIT, typename Template, template<typename> class... Ts>
struct GET_POSITION_AUX<LIMIT, LIMIT, Template, Ts...> {
    static const result_t value = 0;
};

} //@anonymous


template <typename Template, template<typename> class... Ts>
struct GET_POSITION {
    static const result_t value = GET_POSITION_AUX<0, Template::size,
                                                   Template, Ts...>::value;
    //static_assert(value != 0, PRINT_ERR("Invariant not found"));
};

} //@mangrove

#include "TemplateEngineSupport.cuh"
