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
#include "Inference/ResultCollector.hpp"


namespace mangrove {

template<typename T>
void HostMiningFixed(const HostParamStr& HostParam,
                     const TracePropSTR<T>& TraceProp);

template<>
void HostMiningFixed<bool>(const HostParamStr& HostParam,
                           const TracePropSTR<bool>& TraceProp);

template<>
void HostMiningFixed<numeric_t>(const HostParamStr& HostParam,
                                const TracePropSTR<numeric_t>& TraceProp);

namespace support {

inline void HostUnaryBitCounting(bitCounter_t* Result,
                                 const TracePropSTR<bool>& TraceProp,
                                 int thread_index = 0, int concurrency = 1);

inline void HostBinaryBitCounting(bitCounter_t* Result,
                                  dictionary_ptr_t<2> Dictionary2,
                                  int dictionarySize,
                                  const TracePropSTR<bool>& TraceProp,
                                  int thread_index = 0, int concurrency = 1);

void HostUnaryNumericRange(num2_t* Result,
                           const TracePropSTR<numeric_t>& TraceProp,
                           const int ThIndex, const int concurrency);

void MiningEquivalenceSets(ResultCollector<numeric_t> & results,
                           dictionary_ptr_t<2> dictionary2F,
                           int forward_dictionary_size);

} //@support
} //@mangrove
