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

#include "XLib.hpp"
#include <limits>
#include <type_traits>

namespace mangrove {

// ======================== Types ==============================================

using      entry_t = unsigned char;
using     result_t = unsigned;
using    numeric_t = float;
using       num2_t = numeric_t[2];
using bitCounter_t = int;

// ======================== ??? ==============================================

enum MonotonyProperty
{
    UNKNOWN    = 0,
    NOMONOTONY = 1,
    INCREASING = 2,
    DECREASING = 3
};

enum CommutativeProperty
{
    UNKNOWNC   = 0,
    NO         = 1,
    YES        = 2
};

// ======================== Assertions =========================================

static_assert(std::is_unsigned<entry_t>::value,
                   PRINT_ERR("<entry_t> must be integral and unsigned type"));
static_assert(std::is_unsigned<result_t>::value &&
              std::is_integral<result_t>::value,
                   PRINT_ERR("<entry_t> must be integral and unsigned type"));
static_assert(std::is_arithmetic<numeric_t>::value,
                   PRINT_ERR("<numeric_t> must be arithmetic"));

// ======================== INTERNAL VAR =======================================

const int        MAX_VARS = 255;
const int       MAX_ARITY = 3;
const int MAX_CONCURRENCY = 32;

static_assert(MAX_VARS <= std::numeric_limits<entry_t>::max(),
              PRINT_ERR(" entry_t type < MAX_VARS"));

// ======================== AutoTuing ==========================================

#if defined(__NVCC__)

static const int MAX_STEP = 4;

const int MAX_BLOCK_CHUNK  = xlib::MAX_BLOCKSIZE * MAX_STEP * sizeof(int4) * 2;

#endif

// -----------------------------------------------------------------------------

} //@mangrove
