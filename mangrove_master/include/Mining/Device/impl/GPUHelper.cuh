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

#include <sstream>
#include "Inference/Generator.hpp"

#include "Mining/TemplateEngine.cuh"
#include "TemplateConf.cuh"
#include "Mining/Device/impl/AutoTuning.cuh"
#include "Mining/Device/Kernels/GPUGenericMining.cuh"
#include "DataTypes/TraceProp.cuh"

namespace mangrove {

AutoTuningGen(GPUMining)

template<typename PROP, typename T>
void GPUHelper(result_t* results, entry_t* dictionary, int dictionary_size,
               const TracePropSTR<T>& TraceProp) {

    using namespace timer;
    using namespace xlib;
    using Template = typename IndexToTemplate<PROP::TEMPL_INDEX>::type;

    std::ostringstream oss;
    oss << TemplateToString<typename Template::type>::value << " "
        << TemplateToArity<Template::arity>::value;

    Timer_cuda TM(1, 30, Color::FG_L_RED);

    if (dictionary_size == 0) return; // nothing to do

    cudaMemcpyToSymbolAsync(devDictionary, dictionary,
                            dictionary_size * Template::arity* sizeof(entry_t));

    const unsigned gridDim = gridConfig(GPUMining<PROP, T>,
                                        PROP::BlockSize, 0, dictionary_size);

    devTraceSTR<T> devProp(TraceProp, dictionary_size);
#if defined(AUTO_TUNING)
    AutoTuningClass::Init(dictionary_size);
    AutoTuningGenGPUMining<PROP, T>::Apply(devProp);
#endif
    TM.start();

    GPUMining<PROP><<< gridDim, PROP::BlockSize >>>(devProp);

    TM.getTimeA(oss.str());

    cudaMemcpyFromSymbol(results, devResult, dictionary_size* sizeof(result_t));
    __CUDA_ERROR("Mining Kernel");
}

} //@mangrove
