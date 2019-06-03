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
#include "Inference/Generator.hpp"
#include "Printer/InvariantPrinter.hpp"
#include "Inference/ResultCollector.hpp"
#include "Inference/CheckBinaryMonotony.hpp"
#include "Inference/CheckTernaryMonotony.hpp"


#include "Mining/Mining.hpp"
#include "Mining/Device/impl/GPUHelper.cuh"
#include "Mining/Device/GPUMiningFixed.cuh"
#include "Mining/Device/Kernels/NumericRange.cuh"
#include "Mining/Host/HostMiningFixed.hpp"

#include "XLib.hpp"
#include "config.cuh"


using namespace xlib;
using namespace timer;

namespace mangrove {

AutoTuning(GPUNumericRange)

template<>
void GPUMiningFixed<numeric_t>(const GPUParamStr& GPUParam,
                               const TracePropSTR<numeric_t>& TraceProp) {

    result_t* devResultPTR;
    cudaGetSymbolAddress((void**) &devResultPTR, devResult);

    using PROP_BASE = PROPERTY<128>;
    using PROP_NB = PROPERTY<128, 1, NB>;
    using PROP_NT = PROPERTY<256, 1, NT>;
    Timer_cuda TM(1, 30, Color::FG_L_RED);

    //--------------------------------------------------------------------------
    ResultCollector<numeric_t> results_numeric(TraceProp.vars);
    Generator<numeric_t> generator(TraceProp.vars);

    // Unary Range
    //--------------------------------------------------------------------------
    devTraceSTR<numeric_t> devProp(TraceProp);
/*
#if defined(AUTO_TUNING)
     AutoTuningClass::Init(TraceProp.vars);
     AutoTuningGPUNumericRange<PROP_BASE>::Apply(devProp);
#endif*/

    unsigned gridDim = gridConfig(GPUNumericRange<PROP_BASE>,
                                  PROP_BASE::BlockSize, 0, TraceProp.vars);
    TM.start();

    GPUNumericRange<PROP_BASE><<<gridDim, PROP_BASE::BlockSize>>>(devProp);

    TM.getTimeA("Unary Range");
    __CUDA_ERROR("GPURangeChecking");
    support::numericRangeCheck(devResultPTR, TraceProp, GPUParam.check_result);

    cudaMemcpyFromSymbol(results_numeric.getVectorResult<1>(), devResult,
                         static_cast<size_t>(TraceProp.vars) * sizeof(float2));

    // Monotony checking
    //==========================================================================
#if defined(NUMERIC_INFERENCE)
    CheckBinaryMonotony<NBtemplate::size> (results_numeric, TraceProp.vars);
    CheckTernaryMonotony<NTtemplate::size> (results_numeric, TraceProp.vars);
#endif

    // Binary Mining
    // =========================================================================
    // Generate FORWARD binary dictionary
    generator.setForwardDirection(true);
    results_numeric.setForwardDirection(true);
    // -------------------------------------------------------------------------

    int forward_dictionary_size = generator.generator<2>(results_numeric);
    entry_t* dictionary2F = generator.dictionary;

    std::cout << "(Numeric)" << std::setw(30)
              << "Forward Dictionary size : "
              << forward_dictionary_size << std::endl << std::endl;

    // launch FORWARD binary mining
    result_t* binaryResult = static_cast<result_t*>(
                                          results_numeric.getVectorResult<2>());

    GPUHelper<PROP_NB>(binaryResult, dictionary2F, forward_dictionary_size,
                       TraceProp);

    // Generate BACKWARD binary dictionary
    generator.setForwardDirection(false);
    results_numeric.setForwardDirection(true);
    // -------------------------------------------------------------------------

    int backward_dictionary_size = generator.generator<2>(results_numeric);
    entry_t* dictionary2B = generator.dictionary + halfNumericDictionary;

    std::cout << "(Numeric)" << std::setw(30)
              << "Backward Dictionary size : "
              << backward_dictionary_size << std::endl << std::endl;

    // launch BACKWARD binary mining
    results_numeric.setForwardDirection(false);
    binaryResult = static_cast<result_t*>(results_numeric.getVectorResult<2>());

    GPUHelper<PROP_NB>(binaryResult, dictionary2B, backward_dictionary_size,
                        TraceProp);

    generator.setForwardDirection(true);
    results_numeric.setForwardDirection(true);

#if defined(NUMERIC_INFERENCE)
    support::MiningEquivalenceSets(results_numeric,
                                   dictionary_ptr_t<2>(dictionary2F),
                                   forward_dictionary_size);
#endif
    // Ternary Mining
    // -------------------------------------------------------------------------
    int dictionary_size = generator.generator<3>(results_numeric);
    entry_t  *dictionaryT = generator.dictionary;

    std::cout << "(Numeric)" << std::setw(30)
              << "Ternary Dictionary size : " << dictionary_size
              << std::endl << std::endl;

    result_t *ternaryResult = static_cast<result_t*>(
                                results_numeric.getVectorResult<3>());

    GPUHelper<PROP_NT>(ternaryResult, dictionaryT, dictionary_size, TraceProp);

    // Print Invariants
    //--------------------------------------------------------------------------
    std::ofstream stream;
    if (GPUParam.output_file)
        stream.open(GPUParam.output_file);

    InvariantPrinter<numeric_t> printer_numeric(GPUParam, TraceProp.vars);
    std::cout << " Unary Result: "
              << printer_numeric.unaryInvariants(results_numeric, stream)
              << std::endl;
    std::cout << " Binary Result: "
              << printer_numeric.binaryInvariants(results_numeric, stream)
              << std::endl;
    std::cout << "Ternary Result: "
              << printer_numeric.ternaryInvariants(results_numeric, stream)
              << std::endl;
}

namespace support {

void numericRangeCheck(result_t* devResult,
                       const TracePropSTR<numeric_t>& TraceProp,
                       bool check_result) {
    if (check_result) {
        float2* NumericRange = new float2[TraceProp.vars];
        cudaMemcpy(NumericRange, devResult,
                   static_cast<size_t>(TraceProp.vars) * sizeof(float2),
                   cudaMemcpyDeviceToHost);

        numeric_t* host_trace_var = TraceProp.host_trace;
        for (int V = 0; V < TraceProp.vars; V++) {
            auto minmax = std::minmax_element(host_trace_var,
                                       host_trace_var + TraceProp.trace_length);

            if (NumericRange[V].x != *minmax.first) {
                __ERROR("Var: " << V << " -> NumericRange Min: "
                        << std::setprecision(10) << std::fixed
                        << NumericRange[V].x << " " << *minmax.first);
            }
            if (NumericRange[V].y != *minmax.second) {
                __ERROR("Var: " << V << " -> NumericRange Max: "
                       << std::setprecision(10) << std::fixed
                       << NumericRange[V].y << " " << *minmax.second);
            }
            host_trace_var += TraceProp.trace_length;
        }
        std::cout << __func__ << ": OK" << std::endl << std::endl;
        delete[] NumericRange;
    }
}

} //@support
} //@mangrove
