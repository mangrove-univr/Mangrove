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

namespace mangrove {

template<typename T> struct TemplStrT;
template<>struct TemplStrT<unsigned> {
    static constexpr const char* NAME = "(Boolean)";
};
template<>struct TemplStrT<float> {
    static constexpr const char* NAME = "(Numeric)";
};

template<int ARITY> struct TemplStrA;
template<>struct TemplStrA<2> {
    static constexpr const char* NAME = "Binary";
};
template<>struct TemplStrA<3> {
    static constexpr const char* NAME = "Ternary";
};
template<>struct TemplStrA<4> {
    static constexpr const char* NAME = "Quaternary";
};

template<int ARITY> struct TemplStdNum;
template<>struct TemplStdNum<2> {
    static constexpr const int num = 5;
};
template<>struct TemplStdNum<3> {
    static constexpr const int num = 2;
};
template<>struct TemplStdNum<4> {
    static constexpr const int num = 0;
};

template <typename R, typename T>
using binary_op = R(*)(T, T);

//==============================================================================

template <typename OP_TYPE, binary_op<OP_TYPE, OP_TYPE>... Ts>
struct InvariantTail {};

template <typename OP_TYPE, binary_op<OP_TYPE, OP_TYPE> T,
          binary_op<OP_TYPE, OP_TYPE>... Ts>
struct InvariantTail<OP_TYPE, T, Ts...> : InvariantTail<OP_TYPE, Ts...> {

    __HOST_DEVICE__
    OP_TYPE operator()(OP_TYPE A, OP_TYPE B) {
        return T(A, B);
    }
};

template <int _ARITY, typename OP_TYPE, binary_op<bool, OP_TYPE> T,
          binary_op<OP_TYPE, OP_TYPE>... Ts>
struct Invariant : InvariantTail<OP_TYPE, Ts...> {

    __HOST_DEVICE__ bool operator()(OP_TYPE A, OP_TYPE B) {
        return T(A, B);
    }

    using type = OP_TYPE;
    static const int size = 1 + sizeof...(Ts);
    static const int ARITY = size + 1;
    static_assert(ARITY == _ARITY,
                  PRINT_ERR(Wrong Arity in Invariant Definition));
};

//==============================================================================

template <int, typename, typename> struct GET_F2;

template <int F_INDEX, typename OP_TYPE, binary_op<OP_TYPE, OP_TYPE> T,
          binary_op<OP_TYPE, OP_TYPE>... Ts>
struct GET_F2<F_INDEX, OP_TYPE, InvariantTail<OP_TYPE, T, Ts...>> {

    __HOST_DEVICE__
    OP_TYPE operator()(OP_TYPE A, OP_TYPE B) {
        return GET_F2<F_INDEX - 1, OP_TYPE, InvariantTail<OP_TYPE, Ts...>>()
                                                                         (A, B);
    }
};

template < typename OP_TYPE, binary_op<OP_TYPE, OP_TYPE> T,
           binary_op<OP_TYPE, OP_TYPE>... Ts>
struct GET_F2<0, OP_TYPE, InvariantTail<OP_TYPE, T, Ts...>> {

    __HOST_DEVICE__
    OP_TYPE operator()(OP_TYPE A, OP_TYPE B) {
        return InvariantTail<OP_TYPE, T>()(A, B);
    }
};

template < typename OP_TYPE, binary_op<OP_TYPE, OP_TYPE> T,
           binary_op<OP_TYPE, OP_TYPE>... Ts>
struct GET_F2<-1, OP_TYPE, InvariantTail<OP_TYPE, T, Ts...>> {

    __HOST_DEVICE__
    OP_TYPE operator()(OP_TYPE A, OP_TYPE B) {
        return A;
    }
};

//------------------------------------------------------------------------------

template <int, int, typename, typename> struct GET_F;

template <int F_INDEX, int TEMPL_ARITY, typename OP_TYPE,
          binary_op<bool, OP_TYPE> T, binary_op<OP_TYPE, OP_TYPE>... Ts>
struct GET_F<F_INDEX, TEMPL_ARITY, OP_TYPE,
             Invariant<TEMPL_ARITY, OP_TYPE, T, Ts...>> {

    __HOST_DEVICE__
    typename std::conditional<F_INDEX == 0, bool, OP_TYPE>::type
    operator()(OP_TYPE A, OP_TYPE B) {
        static_assert(F_INDEX >= 0, PRINT_ERR(WRONG INDEXING));
        return F_INDEX == 0 ? T(A, B) :
            GET_F2<F_INDEX - 1, OP_TYPE, InvariantTail<OP_TYPE, Ts...>>()(A, B);
    }
};

template <int TEMPL_ARITY, typename OP_TYPE,
          binary_op<bool, OP_TYPE> T, binary_op<OP_TYPE, OP_TYPE>... Ts>
struct GET_F<0, TEMPL_ARITY, OP_TYPE,
             Invariant<TEMPL_ARITY, OP_TYPE, T, Ts...>> {

    __HOST_DEVICE__ bool operator()(OP_TYPE A, OP_TYPE B) {
        return Invariant<TEMPL_ARITY, OP_TYPE, T, Ts...>()(A, B);
    }
};

//==============================================================================

template<typename... T> struct all_same {};

template<typename T>
struct all_same<T> {
    static const int size = T::size;
    using type = typename T::type;
    static const bool value = true;
};

template<typename T, typename... Ts>
struct all_same<T, Ts...> : all_same<Ts...> {
    static const int size = T::size == all_same<Ts...>::size ? T::size : -1;

    using type = typename std::conditional<
        std::is_same<typename T::type, typename all_same<Ts...>::type>::value,
        typename T::type,
        void  >::type;

    static const bool value = size != -1 && !std::is_same<type, void>::value;
};

//------------------------------------------------------------------------------

template <typename... T> struct Template {};

template <typename T, typename... Ts>
struct Template<T, Ts...> : Template<Ts...> {
    static const int size = 1 + sizeof...(Ts);
    using inv_t = T;
    using type = typename T::type;
    static const int  ARITY = T::ARITY;
    static_assert(all_same<T, Ts...>::value,
      PRINT_ERR(Error: Template of different invariants types are not allowed));
};

//==============================================================================

template <int, typename> struct GET_INV;

template <int INDEX, typename T, typename... Ts>
struct GET_INV<INDEX, Template<T, Ts...>> {
    using INV = typename GET_INV<INDEX - 1, Template<Ts...>>::INV;
};

template <typename T, typename... Ts>
struct GET_INV<0, Template<T, Ts...>> {
    using INV = T;
};

//==============================================================================

template <typename... T> struct TemplateSETstr {};

template <typename T, typename... Ts>
struct TemplateSETstr<T, Ts...> : TemplateSETstr<Ts...> {
    static const int size = 1 + sizeof...(Ts);
};

//==============================================================================

template <int, typename> struct GET_TEMPL;

template <int INDEX, typename T, typename... Ts>
struct GET_TEMPL<INDEX, TemplateSETstr<T, Ts...>> {
    using templ_t = typename GET_TEMPL<INDEX - 1,
                                       TemplateSETstr<Ts...>>::templ_t;
};

template <typename T, typename... Ts>
struct GET_TEMPL<0, TemplateSETstr<T, Ts...>> {
    using templ_t = T;
};

//==============================================================================

template <binary_op<bool, unsigned> T>
using BinaryBoolInvariants = Invariant<2, unsigned, T>;

template <binary_op<bool, unsigned> T, binary_op<unsigned, unsigned>... Ts>
using TernaryBoolInvariants = Invariant<3, unsigned, T, Ts...>;

template <binary_op<bool, float> T>
using BinaryNumericInvariants = Invariant<2, float, T>;

template <binary_op<bool, float> T, binary_op<float, float>... Ts>
using TernaryNumericInvariants = Invariant<3, float, T, Ts...>;

//==============================================================================

#if defined(USER_TEMPLATE)
    #include "../../TemplateConf/TemplateConfUser.cuh"
#else
    #include "../../TemplateConf/TemplateConf.cuh"
#endif

template <int TEMPL_INDEX, int INV_INDEX, int F_INDEX>
struct GET {
    template<typename S>
    __HOST_DEVICE__
    typename std::conditional<F_INDEX == 0, bool, S>::type operator()(S A, S B){
        static_assert(TEMPL_INDEX >= 0,
                PRINT_ERR("Template Index must be greater or equal than zero"));
        static_assert(TEMPL_INDEX < TemplateSET::size,
                PRINT_ERR("Template Index must be less than Template Size"));
        using templ = typename GET_TEMPL<TEMPL_INDEX, TemplateSET>::templ_t;
        using Inv = typename GET_INV<INV_INDEX, templ>::INV;
        return GET_F<F_INDEX, Inv::ARITY, S, Inv>()(A, B);
    }
};

//==============================================================================


} //@mangrove
