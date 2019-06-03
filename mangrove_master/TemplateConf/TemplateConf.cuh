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

#include "Mining/TemplateEngine.cuh"

namespace mangrove {

using BBinv1 = BoolBinaryInvariants<equal>;
using BBinv2 = BoolBinaryInvariants<complement>;

using BTinv1 = BoolTernaryInvariants<equal, AND>;
//using BTinv2 = BoolTernaryInvariants<equal, IMPLY>;
//using BTinv3 = BoolTernaryInvariants<equal, R_IMPLY>;
using BTinv4 = BoolTernaryInvariants<equal, XOR>;
using BTinv5 = BoolTernaryInvariants<equal, OR>;
/*
using BTinv6 = BoolTernaryInvariants<equal, NOR>;
using BTinv7 = BoolTernaryInvariants<equal, XNOR>;
using BTinv8 = BoolTernaryInvariants<equal, NOT_R_IMPLY>;
using BTinv9 = BoolTernaryInvariants<equal, NOT_IMPLY>;
using BTinv10 = BoolTernaryInvariants<equal, NAND>;*/

using BBtemplate = Template<BBinv1, BBinv2>;
using BTtemplate = Template<BTinv1, BTinv4, BTinv5/*, BTinv5,
                            BTinv6, BTinv7, BTinv8, BTinv9, BTinv10*/>;

//------------------------------------------------------------------------------

using NBinv1 = NumericBinaryInvariants<equal>;
using NBinv2 = NumericBinaryInvariants<notEqual>;
using NBinv3 = NumericBinaryInvariants<less>;
using NBinv4 = NumericBinaryInvariants<lessEq>;
/*using NBinv5 = NumericBinaryInvariants<less, Sqrt>;
using NBinv6 = NumericBinaryInvariants<equal, Log>;
using NBinv7 = NumericBinaryInvariants<less, succ>;
using NBinv8 = NumericBinaryInvariants<equal, succ>;*/

using NTinv1 = NumericTernaryInvariants<equal, Min>;
using NTinv2 = NumericTernaryInvariants<equal, Max>;
using NTinv3 = NumericTernaryInvariants<equal, mul>;
using NTinv4 = NumericTernaryInvariants<equal, add>;

using NBtemplate = Template<NBinv1, NBinv2, NBinv3, NBinv4/*,
                            NBinv5, NBinv6, NBinv7, NBinv8*/>;

using NTtemplate = Template<NTinv1, NTinv2, NTinv3, NTinv4>;

} //@mangrove




namespace mangrove {

namespace {

template<int INDEX = 0>
struct EXCLUDE_NT_NOTEQUAL : EXCLUDE_NT_NOTEQUAL<INDEX + 1> {
    using first = typename GET_INVARIANT<INDEX, NBtemplate>::type::first;
    using unary = typename GET_INVARIANT<INDEX, NBtemplate>::type::unary;

    static_assert(std::is_same<unary, identity<numeric_t>>::value ||
                  !std::is_same<first, notEqual<numeric_t>>::value,
                  PRINT_ERR("Numeric Binary Template does not accept notEqual\
                             as function in composed invariants"));
};

template<>
struct EXCLUDE_NT_NOTEQUAL<NBtemplate::size> {};

} //@anonymous

__attribute__((unused)) static EXCLUDE_NT_NOTEQUAL<> CTRL;

} //@mangrove
