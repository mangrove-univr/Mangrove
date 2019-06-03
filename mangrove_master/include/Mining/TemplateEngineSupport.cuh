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

namespace mangrove {

template<template<typename> class T1>
using BoolBinaryInvariants = Invariant<unsigned, T1>;

template<template<typename> class T1, template<typename> class T2>
using BoolTernaryInvariants = Invariant<unsigned, T1, identity, T2>;

template<template<typename> class T1, template<typename> class T2 = identity>
using NumericBinaryInvariants = Invariant<numeric_t, T1, T2>;

template<template<typename> class T1, template<typename> class T2>
using NumericTernaryInvariants = Invariant<numeric_t, T1, identity, T2>;

//------------------------------------------------------------------------------


template<typename T>
struct TemplateToString;

template<>
struct TemplateToString<unsigned> {
    static constexpr const char* value = "(Boolean)";
};
template<>
struct TemplateToString<float> {
    static constexpr const char* value = "(Numeric)";
};

template<int ARITY>
struct TemplateToArity;

template<>
struct TemplateToArity<2> {
    static constexpr const char* value = "Binary";
};
template<>
struct TemplateToArity<3> {
    static constexpr const char* value = "Ternary";
};
template<>
struct TemplateToArity<4> {
    static constexpr const char* value = "Quaternary";
};


/*
inline void prova() {
    std::cout << GET<_BoolBinaryTemplate, 0, 0>::eval(4, 4) << std::endl;
    std::cout << GET<_BoolBinaryTemplate, 1, 0>::eval(4, 4) << std::endl
              << std::endl;

    std::cout << GET<_BoolTernaryTemplate, 0, 1>::eval(2, 4) << std::endl;
    std::cout << GET<_BoolTernaryTemplate, 1, 1>::eval(2, 4) << std::endl
              << std::endl;

    std::cout << GET<_NumericBinaryTemplate, 0, 0>::eval(2, 4) << std::endl;
    std::cout << GET<_NumericBinaryTemplate, 1, 0>::eval(3, 2) << std::endl;
    std::cout << GET<_NumericBinaryTemplate, 2, 0>::eval(2, 4) << std::endl
              << std::endl;

    std::cout << GET<_NumericTernaryTemplate, 0, 1>::eval(3, 2) << std::endl;
    std::cout << GET<_NumericTernaryTemplate, 1, 1>::eval(2, 4) << std::endl
              << std::endl;
}*/

} //@mangrove
