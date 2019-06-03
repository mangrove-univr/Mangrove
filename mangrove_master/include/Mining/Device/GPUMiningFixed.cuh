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

#include "DataTypes/TraceProp.cuh"
#include "DataTypes/ModuleSTR.hpp"

namespace mangrove {

template<typename T>
void GPUMiningFixed(const GPUParamStr& GPUParam,
                    const TracePropSTR<T>& TraceProp);

template<>
void GPUMiningFixed<bool>(const GPUParamStr& GPUParam,
                          const TracePropSTR<bool>& TraceProp);

template<>
void GPUMiningFixed<numeric_t>(const GPUParamStr& GPUParam,
                               const TracePropSTR<numeric_t>& TraceProp);

//------------------------------------------------------------------------------
namespace support {

void bitCountingCheckUnary(result_t* devResult,
                           const TracePropSTR<bool>& TraceProp,
                           int length, bool check_result);

void bitCountingCheckBinary(result_t* devResult, dictionary_ptr_t<2> Dictionary,
                            int dictionarySize,
                            const TracePropSTR<bool>& TraceProp,
                            int length, bool check_result);

void numericRangeCheck(result_t* devResult,
                       const TracePropSTR<numeric_t>& TraceProp,
                       bool check_result);
} //@support
} //@mangrove
