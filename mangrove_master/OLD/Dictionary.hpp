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

#include "config.cuh"

namespace mangrove {

template<int VARS, int ARITY>
struct DictionarySize {
    static const int value = ARITY == 2 ?
                             numeric::BINOMIAL_COEFF<VARS, 2>::value * ARITY :
                    VARS * numeric::BINOMIAL_COEFF<VARS - 1, ARITY - 1>::value;
};
const int MAX_DICTIONARY_ENTRIES = DictionarySize<MAX_VARS, MAX_ARITY>::value;

template<int ARITY>
using     dictionary_t = entry_t (&)[MAX_DICTIONARY_ENTRIES][ARITY];

template<int ARITY>
using dictionary_ptr_t = entry_t (*)[ARITY];

template<int ARITY, typename T>
using trace_ptr_t = T (*)[ARITY];

unsigned* DictionaryPTR[MAX_DICTIONARY_ENTRIES][3];

//------------------------------------------------------------------------------
/*
template<int ARITY>
inline int generateDictionary(dictionary_ptr_t<ARITY> Dictionary, const int VARS,
                              const bool print = false);

template<>
int generateDictionary<2>(dictionary_ptr_t<2> Dictionary, const int VARS,
                          const bool print);

template<>
int generateDictionary<3>(dictionary_ptr_t<3> Dictionary, const int VARS,
                          const bool print);

template<>
int generateDictionary<4>(dictionary_ptr_t<4> Dictionary, const int VARS,
                          const bool print);*/

} //@mangrove

/*
            (V - 1) * V
BINARY:     ------------            -> BinCoeff(V, 2)
                  2

                (V - 1) * (V - 2)
TERNARY     V * -----------------  [* 2 if not Commutative] -> V * BinCoeff(V-1, 2)
                        2

//limit 101 variables

QUATERNARY  V * BinCoeff(V-1, 3)

*/
