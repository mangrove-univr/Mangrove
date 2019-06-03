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

extern __constant__ mangrove::TracePropSTR<bool> devProp;

#include "Mining/Device/impl/GPUHelper.cuh"
#include "Mining/Dictionary.hpp"
#include "Mining/TemplateEngine.cuh"

namespace mangrove {

namespace {

template<typename T, typename PROP> void GPUHelperSimple(const int Vars);

template<typename T, int TEMPL_SIZE, int INDEX = 0>
struct testAllGPU {
    static void Apply(const int Vars) {
        using PROP = PROPERTY<INDEX, 256, 1>;
        GPUHelperSimple<PROP>(Vars);

        testAllGPU<T, TEMPL_SIZE, INDEX + 1>::Apply(Vars);
    }
};
template<typename T, int TEMPL_SIZE>
struct testAllGPU<T, TEMPL_SIZE, TEMPL_SIZE> {
    static void Apply(__attribute__((unused)) int Vars) {}
};

} //@anonymous

// =============================================================================

template<typename T>
void GPUMiningUser(TracePropSTR<T>& h_devProp) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // early initialization !!

    cudaMemcpyToSymbolAsync(devProp, &h_devProp, sizeof(devPropSTR<T>));
    __CUDA_ERROR("INIT");

    testAllGPU<T, TemplateSET::size>::Apply(h_devProp.Vars);
}

// =============================================================================

namespace {

template<typename T, typename PROP>
void GPUHelperSimple(const int Vars) {
    using templ = typename GET_TEMPL<T, PROP::TEMPL_INDEX,TemplateSET>::templ_t;
    std::ostringstream oss;
    oss << "GPU Kernel - " << TemplStrT<typename templ::type>::NAME << " "
        << TemplStrA<templ::ARITY>::NAME;
    Timer<DEVICE> TM(23, StreamModifier::FG_RED, 2);

    entry_t Dictionary[MAX_DICTIONARY_ENTRIES][templ::ARITY];

    const int dictionarySize = generateDictionary(Dictionary, Vars);
    std::cout << TemplStrT<typename templ::type>::NAME << std::left
              << std::setw(30)
              << " " << TemplStrA<templ::ARITY>::NAME << " Dictionary size : "
              << dictionarySize << std::endl << std::endl;

    cudaMemcpyToSymbolAsync(devDictionary, Dictionary,
                            dictionarySize * templ::ARITY * sizeof(entry_t));
    const int gridDim = gridConfig(GPUMining<PROP>, PROP::BLOCKDIM,
                                   dictionarySize);

    #ifdef AUTOTUNING
        AutoTuningClass::Init(dictionarySize);
        AutoTuningGPUMining<PROP::TEMPL_INDEX>::Apply(dictionarySize);
    #endif

    TM.start();

    GPUMining<PROP><<< gridDim, PROP::BLOCKDIM >>> (dictionarySize);

    TM.getTime(oss.str());

    result_t Result[MAX_DICTIONARY_ENTRIES];
    cudaMemcpyFromSymbol(Result, devResult, dictionarySize * sizeof(result_t));
    InvariantsCounting<templ>(Result, dictionarySize);

    __CUDA_ERROR("Mining Kernel");
}

} //@anonymous

} //@mangrove
